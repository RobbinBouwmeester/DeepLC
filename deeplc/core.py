"""DeepLC core functions."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from psm_utils import PSM, Peptidoform, PSMList
from torch.utils.data import DataLoader

from deeplc import _model_ops
from deeplc._reference_selection import deduplicate_psms, select_reference_psms
from deeplc.calibration import (
    Calibration,
    MultiHeadCalibration,
    MultiHeadRidgeCalibration,
    upgrade_calibration,
)
from deeplc.data import DeepLCDataset, split_datasets

LOGGER = logging.getLogger(__name__)

DEEPLC_DIR = Path(__file__).resolve().parent
#: The model every core function uses when none is given: the fused-trunk multitask
#: model trained across 6,543 LC setups (see :data:`FLEXCNN_MULTITASK_MODEL`).
DEFAULT_MODEL = DEEPLC_DIR / "package_data" / "models" / "multitask_flexcnn_model.pt"

#: The 4.0 default, a shared-trunk model with one head per LC setup, kept so that
#: existing workflows can pin it: ``predict(psms, model=LEGACY_MULTITASK_MODEL)``.
LEGACY_MULTITASK_MODEL = DEEPLC_DIR / "package_data" / "models" / "multitask_model.pt"

#: The LC setup an uncalibrated ``predict()`` reports for a multitask model. The
#: default model has 6,543 setups and no reference to choose between them, so the
#: setup of the DeepLC 1.x to 3.x training data (PXD005573, 200-minute gradient) is
#: used, which keeps uncalibrated output on the gradient earlier versions reported.
#: Calibration and fine-tuning pick or fit the setup from the reference instead.
DEFAULT_TASK_NAME = "PXD005573_mcp"

#: Below this many reference PSMs, fine-tuning measured worse than calibration on
#: every held-out setup tried, so it is warned about rather than silently attempted.
#: On six unseen LC setups the crossover sat between 300 and 735 reference
#: peptidoforms: at 230 the error went from 1.47 to 91.7 min, at 300 from 0.37 to
#: 0.48, and at 735 and above fine-tuning helped every version.
MIN_FINETUNE_REFERENCE = 500

#: A fine-tuned model whose validation error exceeds this fraction of the reference
#: retention-time span has not converged onto the gradient, whatever the loss curve
#: said. Measured failures collapse the output range and predict near the mean, so
#: the error lands at a large fraction of the span: on one 133-minute gradient the
#: adapter path reached 92 minutes, or 69 % of the span, while its correlation stayed
#: above 0.9 because the peptide ordering was never what broke. Ordinary fits sit
#: near 1 %.
MAX_FINETUNE_ERROR_FRACTION = 0.15

#: Fused-trunk multitask model, trained across 6,543 LC setups. The default since
#: 4.1.1; the name is kept for callers that pass it explicitly.
FLEXCNN_MULTITASK_MODEL = DEFAULT_MODEL


def predict(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    model: torch.nn.Module | PathLike | str | None = None,
    predict_kwargs: dict | None = None,
    return_matrix: bool = False,
) -> np.ndarray:
    """
    Predict retention times for a list of PSMs using a trained model.

    Parameters
    ----------
    psm_list
        List of PSMs to predict retention times for.
    model
        Trained model or path to model file. If None, the default DeepLC model is used.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.
    return_matrix
        If True, return the full prediction matrix of shape ``(n, n_heads)`` when using a
        multitask model. If False (default), return a 1D array of shape ``(n,)`` for the
        setup named by :data:`DEFAULT_TASK_NAME` when the model knows its setups, and head
        0 otherwise. Uncalibrated output is on that setup's gradient; use
        :func:`predict_and_calibrate` to map it onto your own.

    Returns
    -------
    np.ndarray
        Retention time predictions. Shape ``(n,)`` unless ``return_matrix=True`` and model
        produces multitask output, in which case shape is ``(n, n_heads)``.

    """
    # The model is loaded before the dataset is built because the features it
    # needs depend on the model. A checkpoint that describes itself carries a
    # feature specification, and a model trained on the 67-dimensional global
    # vector cannot be fed the 55-dimensional default.
    #
    # The device is taken from predict_kwargs rather than left to default, so a
    # caller asking for CPU does not first have the checkpoint placed on a GPU
    # it may not fit on.
    kwargs = dict(predict_kwargs or {})
    loaded_model = _model_ops.load_model(model or DEFAULT_MODEL, device=kwargs.get("device"))
    feature_spec = getattr(loaded_model, "feature_spec", None) or {}

    # Only one column is wanted unless the caller asked for the matrix. A model
    # trained on thousands of LC setups would otherwise materialise a column per
    # setup, which at a million peptides is tens of gigabytes of output for a
    # result the caller then throws away.
    if (
        not return_matrix
        and "task_idx" not in kwargs
        and _model_ops.supports_task_subset(loaded_model)
    ):
        kwargs["task_idx"] = [_default_task_idx(loaded_model)]

    result = _model_ops.predict(
        model=loaded_model,
        data=DeepLCDataset.from_psm_list(
            _parse_psms(psm_list), **_feature_kwargs_from_spec(feature_spec)
        ),
        **kwargs,
    ).numpy()
    if not return_matrix:
        return result[:, 0 if "task_idx" in kwargs else _default_task_idx(loaded_model)]
    return result


def _default_task_idx(model: torch.nn.Module) -> int:
    """
    Index of the setup an uncalibrated prediction reports.

    :data:`DEFAULT_TASK_NAME` when the model carries setup names and lists it, else 0.
    A model without names, or a model of a single setup, has nothing to choose from.
    """
    names = getattr(model, "task_names", None) or []
    try:
        return list(names).index(DEFAULT_TASK_NAME)
    except ValueError:
        return 0


def calibrate(
    psm_list_reference: PSMList,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: Calibration | MultiHeadCalibration | None = None,
    predict_kwargs: dict | None = None,
) -> MultiHeadCalibration:
    """
    Return a `MultiHeadCalibration` instance fitted to the reference dataset.

    Parameters
    ----------
    psm_list_reference
        List of PSMs to use as reference for calibration.
    model
        Trained model or path to model file.
    calibration
        Calibration instance to use. If None, MultiHeadRidgeCalibration is used. A model with a
        single LC setup still predicts a one-column matrix, so this default also covers
        single-task models. See ``deeplc.calibration.multihead`` for lighter alternatives (e.g.
        ``MultiHeadSplineCalibration``) when the ridge over many heads isn't the right fit, for
        example for a custom, non-multitask model. An unfitted naive
        :class:`~deeplc.calibration.simple.Calibration` is accepted too and upgraded to its
        ``MultiHead*Calibration`` counterpart; see :func:`deeplc.calibration.upgrade_calibration`.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.

    Returns
    -------
    MultiHeadCalibration
        Fitted calibration instance.

    """
    # One point per peptidoform: a reference taken from a search result repeats a
    # peptidoform once per spectrum it was identified in, each time with a different observed
    # retention time, which gives the fit conflicting targets and weighs peptidoforms by how
    # often they happened to be identified. A caller who wants the repeats to count fits a
    # MultiHeadCalibration itself and passes it in already fitted.
    psm_list_reference = deduplicate_psms(psm_list_reference)

    if calibration is None:
        LOGGER.debug("No calibration provided, using MultiHeadRidgeCalibration by default.")
        calibration = MultiHeadRidgeCalibration()
    else:
        calibration = upgrade_calibration(calibration)
    if calibration.is_fitted:
        LOGGER.warning(
            "Provided calibration is already fitted. Refitting will overwrite existing fit."
        )

    if any(psm_list_reference["is_decoy"]):
        LOGGER.warning(
            "Reference PSM list contains decoy PSMs. "
            "These will be included in the calibration fitting."
        )

    # Predict initial retention times for the reference dataset
    LOGGER.debug("Predicting retention times for reference...")
    source_rt_cal = predict(
        psm_list=psm_list_reference,
        model=model,
        predict_kwargs=predict_kwargs,
        return_matrix=True,
    )

    # Fit calibration; every MultiHeadCalibration takes the whole matrix and selects its own
    # head(s), setting selected_model_head itself for callers that want to know which setup came
    # out on top.
    LOGGER.debug("Fitting calibration...")
    target_rt_cal = np.array(psm_list_reference["retention_time"], dtype=np.float32)
    calibration.fit(target=target_rt_cal, source=source_rt_cal)

    return calibration


class HeadColumnSource:
    """
    A model's predictions for whichever heads are asked for, evaluated on demand.

    Stands in for the ``(n, n_heads)`` matrix wherever a calibration is given its source. A
    multitask model has one head per LC setup, so that matrix is 26 kB per peptide at 6,543
    setups, and a fitted calibration reads a few dozen columns of it; asking the model for
    those columns instead costs 320 bytes per peptide and skips the rest of the head layer.

    Passing this or a real matrix makes no difference to the calibration, and none to the
    caller, which hands over one source either way. ``np.asarray`` on it still yields the
    whole matrix, so code that genuinely needs every head, such as ranking them during
    ``fit``, keeps working.

    Parameters
    ----------
    psm_list
        The peptides to predict.
    model
        Model or path, as :func:`predict` takes it.
    predict_kwargs
        Extra arguments for the prediction, such as the device and batch size.
    n_heads
        How many heads the model has, so the shape is known without predicting anything.

    """

    def __init__(
        self, psm_list, model=None, predict_kwargs: dict | None = None, n_heads: int | None = None
    ):
        """Initialize the source; nothing is predicted until a column is asked for."""
        self._psm_list = _parse_psms(psm_list)
        self._model = model
        self._predict_kwargs = dict(predict_kwargs or {})
        loaded = _model_ops.load_model(
            model or DEFAULT_MODEL, device=self._predict_kwargs.get("device")
        )
        self._n_heads = int(n_heads if n_heads is not None else getattr(loaded, "n_tasks", 1))
        self._cache: tuple[tuple[int, ...], np.ndarray] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and head count, without evaluating anything."""
        return (len(self._psm_list), self._n_heads)

    @property
    def ndim(self) -> int:
        """Always two: this stands in for a matrix."""
        return 2

    def columns(self, indices) -> np.ndarray:
        """Predictions for the given heads, shape ``(n, len(indices))``, in that order."""
        wanted = tuple(int(i) for i in indices)
        # Callers ask for the same heads more than once - prediction_report transforms the
        # queries and then asks the same calibration for its head disagreement - and each ask
        # would otherwise repeat the forward pass.
        if self._cache is None or self._cache[0] != wanted:
            self._cache = (
                wanted,
                predict(
                    self._psm_list,
                    model=self._model,
                    predict_kwargs={**self._predict_kwargs, "task_idx": list(wanted)},
                    return_matrix=True,
                ),
            )
        return self._cache[1]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Every head, for the callers that really need the whole matrix."""
        matrix = predict(
            self._psm_list,
            model=self._model,
            predict_kwargs=self._predict_kwargs,
            return_matrix=True,
        )
        return matrix if dtype is None else matrix.astype(dtype)

    def __len__(self) -> int:
        """Return the number of peptides."""
        return len(self._psm_list)


def predict_and_calibrate(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    psm_list_reference: PSMList | list[PSM | Peptidoform | str] | None = None,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: Calibration | MultiHeadCalibration | None = None,
    predict_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Predict retention times and calibrate to a reference.

    Parameters
    ----------
    psm_list
        List of PSMs to predict retention times for.
    psm_list_reference
        List of PSMs to use as reference for calibration. If None, the best PSMs are
        automatically selected from ``psm_list`` (auto-calibration). This requires that the input
        PSM list contains observed retention times, score and decoy status to select the best PSMs
        for auto-calibration.
    model
        Trained model or path to model file.
    calibration
        Calibration instance to use. If None, MultiHeadRidgeCalibration is used. See
        ``deeplc.calibration.multihead`` for lighter alternatives. An unfitted naive
        :class:`~deeplc.calibration.simple.Calibration` is accepted too and upgraded to its
        ``MultiHead*Calibration`` counterpart; see :func:`deeplc.calibration.upgrade_calibration`.
        A fitted one is not, since it carries no record of which head it was fit on: fit a
        ``MultiHead*Calibration`` instead to pass in an already fitted calibration.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.


    Returns
    -------
    np.ndarray
        Calibrated retention time predictions.

    """
    parsed_psm_list = _parse_psms(psm_list)

    if psm_list_reference is None:
        parsed_psm_list_ref = select_reference_psms(parsed_psm_list)
    else:
        parsed_psm_list_ref = _parse_psms(psm_list_reference)

    # Predict initial retention times
    LOGGER.info("Predicting retention times...")
    # A source rather than a matrix: the calibration pulls the heads it reads, which for a
    # multitask model is a few dozen of thousands.
    predicted_rt = HeadColumnSource(parsed_psm_list, model=model, predict_kwargs=predict_kwargs)

    if calibration is not None:
        calibration = upgrade_calibration(calibration)

    # Fit calibration if not already fitted
    if calibration is None or not calibration.is_fitted:
        calibration = calibrate(
            psm_list_reference=parsed_psm_list_ref,
            model=model,
            calibration=calibration,
            predict_kwargs=predict_kwargs,
        )
    else:
        LOGGER.info("Calibration is already fitted, skipping fitting step.")

    # Every MultiHeadCalibration selects its own head(s), so it takes the matrix as it is.
    return calibration.transform(predicted_rt)


def finetune_and_predict(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    psm_list_reference: PSMList | list[PSM | Peptidoform | str] | None = None,
    model: torch.nn.Module | PathLike | str | None = None,
    train_kwargs: dict | None = None,
    predict_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Fine-tune the model to a reference and predict new retention times.

    Parameters
    ----------
    psm_list
        List of PSMs to predict retention times for.
    psm_list_reference
        List of PSMs to use as reference for calibration. If None, the best PSMs are automatically
        selected from ``psm_list`` (auto-calibration). This requires that the input PSM list
        contains observed retention times, score and decoy status to select the best PSMs for
        auto-calibration.
    model
        Trained model or path to model file.
    train_kwargs
        Additional keyword arguments to pass to the training function.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.


    Returns
    -------
    np.ndarray
        Calibrated retention time predictions after fine-tuning.

    """
    parsed_psm_list = _parse_psms(psm_list)

    if psm_list_reference is None:
        parsed_psm_list_ref = select_reference_psms(parsed_psm_list)
    else:
        parsed_psm_list_ref = _parse_psms(psm_list_reference)

    # Fine-tune the model
    finetuned_model = finetune(
        psm_list_reference=parsed_psm_list_ref,
        model=model,
        train_kwargs=train_kwargs,
    )

    # Predict retention times with fine-tuned model
    LOGGER.info("Predicting retention times with fine-tuned model...")
    predicted_rt = predict(
        psm_list=parsed_psm_list,
        model=finetuned_model,
        predict_kwargs=predict_kwargs,
        return_matrix=True,
    )

    # Fit calibration to the fine-tuned model predictions
    LOGGER.info("Fitting calibration with fine-tuned model predictions...")
    calibration = calibrate(
        psm_list_reference=parsed_psm_list_ref,
        model=finetuned_model,
        predict_kwargs=predict_kwargs,
    )

    # Apply calibration to predictions
    calibrated_rt = calibration.transform(predicted_rt)

    return calibrated_rt


def _feature_kwargs_from_spec(spec: dict | None) -> dict:
    """
    Feature settings a model expects, taken from the specification it carries.

    A checkpoint that records no specification was written before 4.1.0, which
    means it also predates the 4.0.1 correction to positional modification
    deltas, so it is fed the encoding it was trained on. Every model DeepLC has
    released so far is in that position: all five bundled checkpoints are bare
    state dicts. A checkpoint that does record a specification is read literally,
    and one written by this version always records the encoding it used.
    """
    spec = spec or {}
    return {
        "add_ccs_features": bool(spec.get("add_ccs_features", False)),
        "add_terminal_composition": bool(spec.get("add_terminal_composition", False)),
        "padding_length": int(spec.get("padding_length", 60)),
        "legacy_positional_deltas": bool(spec.get("legacy_positional_deltas", not spec)),
    }


def _solve_reference_affine(
    model: torch.nn.Module,
    dataset,
    device: str | None = None,
    adapter: bool = False,
) -> bool:
    """
    Put a freshly attached output on the right axis before training starts.

    Both adaptation paths end in a layer that is linear in its input, so the values
    that best fit the reference data follow in closed form. Solving them first is what
    keeps a small reference set from producing a fit that predicts every peptide near
    the mean retention time.

    Returns True when the solve was applied. A failure is not fatal: training then
    proceeds from the initialised values, which is the previous behaviour.
    """
    try:
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        features, targets = next(iter(loader))
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(selected)
        moved = tuple(f.to(selected) for f in features)
        targets = targets.to(selected).float()
        if adapter:
            return bool(model.solve_adapter_output(*moved, targets))
        model.solve_new_task_affine(moved, targets)
        return True
    except Exception:  # noqa: BLE001 - never let the anchor break the fit
        LOGGER.warning("Could not anchor the output layer; training from the initialised values.")
        return False


def finetune(
    psm_list_reference: PSMList,
    psm_list_validation: PSMList | None = None,
    validation_split: float = 0.1,
    model: torch.nn.Module | PathLike | str | None = None,
    train_kwargs: dict | None = None,
) -> torch.nn.Module:
    """
    Fine-tune an existing model.

    Parameters
    ----------
    psm_list_reference
        List of PSMs to use as reference for fine-tuning.
    psm_list_validation
        List of PSMs to use for validation during fine-tuning. If None, a split from psm_list is
        used.
    validation_split
        Fraction of ``psm_list_reference`` to use for validation when ``psm_list_validation``
        is None. Raised to at least 0.25 when the reference set is smaller than
        :data:`MIN_FINETUNE_REFERENCE`, because early stopping cannot work on a
        handful of PSMs and an unchecked fit can end up far worse than calibration.
    model
        Trained model or path to model file.
    train_kwargs
        Additional keyword arguments to pass to the training function.

    Returns
    -------
    torch.nn.Module
        Fine-tuned model.

    """
    LOGGER.info("Fine-tuning model...")

    # One point per peptidoform, as in calibrate(): training on the same peptidoform several
    # times with contradictory retention times teaches the model the average of a
    # disagreement. Use deeplc.train() for full control over the training set.
    psm_list_reference = deduplicate_psms(psm_list_reference)

    # Fine-tuning needs enough reference data to both fit and validate on. The
    # default validation split leaves too few PSMs to early-stop against on a small
    # reference set, which is how a fit ends up worse than the model it started
    # from; the split is widened here so the stopping signal is usable.
    n_reference = len(psm_list_reference)
    if n_reference < MIN_FINETUNE_REFERENCE:
        LOGGER.warning(
            "Only %d reference PSMs. Fine-tuning measured worse than calibration "
            "below about %d on held-out setups, in the worst case by sixty-fold. "
            "Consider predict() with calibrate() instead.",
            n_reference,
            MIN_FINETUNE_REFERENCE,
        )
        validation_split = max(validation_split, 0.25)
        LOGGER.info(
            "Using a %.0f %% validation split so early stopping has signal.",
            validation_split * 100,
        )

    if any(psm_list_reference["is_decoy"]):
        # TODO: Move to reusable validation step?
        LOGGER.warning("PSM list contains decoy PSMs. These will be used for fine tuning.")
    # The model is loaded further down, but the datasets have to match its feature
    # specification, so peek at it first.
    _peek = _model_ops.load_model(
        model or DEFAULT_MODEL, device=(train_kwargs or {}).get("device")
    )
    _spec = getattr(_peek, "feature_spec", None) or {}
    _feature_kwargs = _feature_kwargs_from_spec(_spec)
    training_data = DeepLCDataset.from_psm_list(psm_list_reference, **_feature_kwargs)
    validation_data = (
        DeepLCDataset.from_psm_list(psm_list_validation, **_feature_kwargs)
        if psm_list_validation
        else None
    )
    training_dataset, validation_dataset = split_datasets(
        training_data, validation_data=validation_data, validation_split=validation_split
    )
    train_kwargs_local = dict(train_kwargs or {})
    adapter_hidden_size = int(train_kwargs_local.pop("adapter_hidden_size", 256))
    freeze_epochs = int(train_kwargs_local.pop("freeze_epochs", 5))
    train_kwargs_local.setdefault("epochs", 50)

    loaded_model = _peek
    if hasattr(loaded_model, "add_task_head"):
        # A low-rank multitask head is adapted by fitting the new setup's own
        # rank + 2 parameters with everything else frozen, rather than by training
        # an adapter over the full head vector. There is nothing to unfreeze part
        # way through, so freeze_epochs does not apply.
        targets = psm_list_reference["retention_time"]
        targets = torch.as_tensor(
            np.asarray([t for t in targets if t is not None], dtype=np.float32)
        )
        n_trainable = loaded_model.add_task_head(targets=targets)

        # Solve the affine part on the reference data before training. Left to the
        # optimiser on a small reference set it collapses: on one 133-minute gradient
        # with 230 reference peptides the output range shrank to 17 minutes and the
        # error reached 91 minutes, with the correlation still above 0.9 because the
        # ordering was never what broke.
        if _solve_reference_affine(loaded_model, training_data, train_kwargs_local.get("device")):
            LOGGER.info("Anchored the new setup's scale and shift by least squares.")
        # Sixty-six parameters tolerate, and need, a far larger step than the
        # whole-network default: at 1e-3 the fit is still short of its optimum after
        # twenty-five epochs (1.32 min against 0.86 on a held-out setup).
        train_kwargs_local.setdefault("learning_rate", 0.05)
        LOGGER.info(
            "Fitting %d parameters for the new setup at lr %.3g; encoder and "
            "pretrained setups are frozen.",
            n_trainable,
            train_kwargs_local["learning_rate"],
        )
    elif hasattr(loaded_model, "add_adapter"):
        loaded_model.add_adapter(hidden_size=adapter_hidden_size)
        train_kwargs_local["freeze_epochs"] = freeze_epochs

        # Put the adapter's output on the right axis before training. From a default
        # initialisation on a small reference set the fit can collapse onto the mean
        # retention time: on a 133-minute gradient with 230 reference peptides the
        # output range shrank to 26 minutes and the error reached 92, with the
        # correlation still above 0.9 because only the scale was lost.
        if _solve_reference_affine(
            loaded_model,
            training_data,
            train_kwargs_local.get("device"),
            adapter=True,
        ):
            LOGGER.info("Anchored the adapter's output layer by least squares.")
    else:
        raise NotImplementedError(
            f"{type(loaded_model).__name__} supports neither adapter-based "
            "fine-tuning nor a low-rank task head."
        )

    finetuned_model = _model_ops.train(
        model=loaded_model,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        **train_kwargs_local,
    )

    _warn_if_fit_collapsed(
        finetuned_model,
        validation_dataset,
        psm_list_reference,
        device=train_kwargs_local.get("device"),
    )
    return finetuned_model


def _warn_if_fit_collapsed(
    model: torch.nn.Module,
    validation_dataset,
    psm_list_reference: PSMList,
    device: str | None = None,
) -> None:
    """
    Say so, loudly, when a fine-tuned model is far worse than its own reference data.

    Fine-tuning can converge on a degenerate solution that predicts every peptide
    near the mean retention time. The loss curve looks unremarkable and the
    correlation stays high, because the ordering is preserved and only the scale is
    lost, so nothing in training flags it. Comparing the validation error against the
    span of the reference retention times does: a collapsed fit lands at a large
    fraction of the span where a working one sits near a hundredth of it.

    This is a report, not a repair. Whether to fall back to calibration is the
    caller's decision, and for the adapter path there is no untrained state worth
    reverting to.
    """
    observed = [t for t in psm_list_reference["retention_time"] if t is not None]
    if len(observed) < 3:
        return
    span = float(np.max(observed) - np.min(observed))
    if span <= 0:
        return

    try:
        error = _model_ops.evaluate(model, validation_dataset, device=device)
    except Exception:  # noqa: BLE001 - the check must never break the fit
        LOGGER.debug("Could not evaluate the fine-tuned model for the sanity check.")
        return

    fraction = error / span
    if fraction > MAX_FINETUNE_ERROR_FRACTION:
        LOGGER.error(
            "Fine-tuned validation error is %.2f, which is %.0f %% of the reference "
            "retention-time span of %.1f. A fit this far off has collapsed rather "
            "than converged: predictions are probably clustered near the mean "
            "retention time. Prefer predict() with calibrate() for this dataset.",
            error,
            fraction * 100,
            span,
        )
    else:
        LOGGER.info(
            "Fine-tuned validation error %.3f, %.1f %% of the reference span.",
            error,
            fraction * 100,
        )


def train(
    psm_list_reference: PSMList,
    psm_list_validation: PSMList | None = None,
    validation_split: float = 0.1,
    train_kwargs: dict | None = None,
) -> torch.nn.Module:
    """
    Train a new model from scratch.

    Parameters
    ----------
    psm_list_reference
        List of PSMs to use as reference for fine-tuning.
    psm_list_validation
        List of PSMs to use for validation. If None, a split from psm_list is used.
    validation_split
        If psm_list_validation is None, this fraction of psm_list will be used for validation.
    train_kwargs
        Additional keyword arguments to pass to the training function.

    Returns
    -------
    torch.nn.Module
        Trained model.

    """
    # A model trained here is new, so it gets the corrected encoding rather than
    # the compatibility default the dataset applies for existing checkpoints.
    training_data = DeepLCDataset.from_psm_list(psm_list_reference, legacy_positional_deltas=False)
    validation_data = (
        DeepLCDataset.from_psm_list(psm_list_validation, legacy_positional_deltas=False)
        if psm_list_validation
        else None
    )
    training_dataset, validation_dataset = split_datasets(
        training_data, validation_data=validation_data, validation_split=validation_split
    )
    LOGGER.info("Training new model...")
    trained_model = _model_ops.train(
        model=None,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        **(train_kwargs or {}),
    )
    return trained_model


def save_model(model: torch.nn.Module, path: PathLike | str) -> None:
    """
    Save a model's state dict to a file.

    Use :func:`load_model` (via :func:`predict`) to reload the saved checkpoint.

    Parameters
    ----------
    model
        Trained model instance to save.
    path
        Destination file path.

    Models that can describe themselves are saved with their architecture,
    constructor arguments and feature specification alongside the weights, so
    that :func:`load_model` can rebuild them. Saving a bare state dict for such a
    model produced a file that could not be reloaded, because the loader would
    fall back to inferring the architecture from tensor names.

    """
    if hasattr(model, "describe"):
        torch.save(model.describe(), path)
    else:
        torch.save(model.state_dict(), path)


def _parse_psms(psm_list: PSMList | list[PSM | Peptidoform | str]) -> PSMList:
    """
    Parse a list of PSMs, Peptidoforms, or strings into a PSMList.

    Note that this function can only be used for inputs that do not require additional data,
    such as retention times or decoy status. It cannot be used for reference or validation
    data sets that require observed retention times for calibration or training.

    """
    if isinstance(psm_list, PSMList):
        return psm_list
    elif isinstance(psm_list, list):
        if all(isinstance(psm, PSM) for psm in psm_list):
            return PSMList(psm_list=psm_list)
        elif all(isinstance(psm, Peptidoform) for psm in psm_list) or all(
            isinstance(psm, str) for psm in psm_list
        ):
            return PSMList(
                psm_list=[PSM(spectrum_id=i, peptidoform=pf) for i, pf in enumerate(psm_list)]
            )
        else:
            raise ValueError("List must contain either PSMs, Peptidoforms, or strings.")
    else:
        raise ValueError("Input must be a PSMList or a list of PSMs, Peptidoforms, or strings.")
