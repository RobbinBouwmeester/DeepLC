"""
Multitask-aware calibration utilities.

Every class here maps the full ``(n, n_heads)`` prediction matrix of a multitask model onto a
single series of observed values, and is responsible for picking which head(s) to use itself:
``fit(target: (n,), source: (n, n_heads))`` / ``transform(source: (n, n_heads)) -> (n,)``. A
single-setup model still produces a matrix, just with ``n_heads=1``, so these classes are the
default way to calibrate any DeepLC model. See :mod:`deeplc.calibration.simple` for the naive,
single-series calibrations these delegate to.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

import numpy as np
from sklearn.linear_model import RidgeCV  # type: ignore[import]

from deeplc.calibration.simple import (
    Calibration,
    PiecewiseLinearCalibration,
    SplineTransformerCalibration,
)
from deeplc.exceptions import CalibrationError

LOGGER = logging.getLogger(__name__)


def take_columns(source, indices: Sequence[int]) -> np.ndarray:
    """
    Take the named head columns from a source, as float64 of shape ``(n, len(indices))``.

    The source is normally the ``(n, n_heads)`` matrix a model returned. It may instead be an
    object offering ``head_columns(indices)``, such as :class:`deeplc.core.HeadColumnSource`,
    which evaluates only the heads asked for: a calibration reads a few dozen of the thousands
    a multitask model has, and at 6,543 setups the unread columns are 26 kB per peptide. Which
    of the two it is makes no difference to a calibration, and the caller hands over the same
    thing either way.

    The method is called ``head_columns`` rather than ``columns`` because a pandas DataFrame
    has a ``columns`` attribute, and a caller passing one deserves to have it read as a matrix
    rather than mistaken for a lazy provider.
    """
    if callable(getattr(source, "head_columns", None)):
        taken = source.head_columns(indices)
    else:
        matrix = np.asarray(source)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        taken = matrix[:, list(indices)]
    return np.asarray(taken, dtype=np.float64)


def source_shape(source) -> tuple[int, int]:
    """
    Give the rows and head count of a source, without materialising a lazy one.

    Anything array-like is accepted, a list of predictions included: ``transform`` used to
    coerce its argument with ``np.asarray`` before reading a shape off it, and that let
    callers pass whatever numpy would take. A source that reports its own shape, such as a
    lazy column provider, is asked rather than converted. A one-dimensional source is one
    head, which is what a single-task model returns.
    """
    shape = getattr(source, "shape", None)
    if shape is None:
        shape = np.asarray(source).shape
    shape = tuple(shape)
    if not shape:
        raise CalibrationError("source has no rows to calibrate")
    return (shape[0], shape[1] if len(shape) > 1 else 1)


class MultiHeadCalibration(ABC):
    """Abstract base class for a calibration that selects its own head(s) from a matrix."""

    #: The best-correlating head, set by ``fit``. Meaningful for every implementer, including
    #: :class:`MultiHeadRidgeCalibration`, which reports the single best head even though it
    #: fits on several.
    selected_model_head: int

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the calibration model."""
        super().__init__()

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Indicates whether the calibration model has been fitted."""
        ...

    @abstractmethod
    def fit(self, target: np.ndarray, source: np.ndarray) -> None:
        """Fit the calibration from the source matrix to target."""
        ...

    @abstractmethod
    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform the source matrix into the calibrated target space."""
        ...

    def disagreement(self, source: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        """
        Per-input uncertainty score, or None when the calibration has none.

        A calibration that combines several estimates of the same retention time can report
        how far they lie apart for each input, which :func:`deeplc.report.prediction_report`
        uses to scale its prediction intervals per peptide. One fitted on a single head has
        no spread to report and keeps this default.
        """
        return None


class _SingleHeadCalibration(MultiHeadCalibration):
    """
    Shared logic for wrapping one naive, single-series :class:`Calibration`.

    Ranks the heads of the matrix by Pearson correlation to the target, fits ``self._inner`` on
    the winner, and transforms by picking that same column. Subclasses only need to construct
    ``self._inner``.
    """

    _inner: Calibration

    @property
    def is_fitted(self) -> bool:
        """True if the wrapped calibration has been fitted."""
        return self._inner.is_fitted

    def fit(self, target: np.ndarray, source: np.ndarray) -> None:
        """
        Select the best-correlating head and fit the wrapped calibration on it.

        Parameters
        ----------
        target
            Observed retention times of the reference, shape ``(n,)``.
        source
            Reference predictions for every head, shape ``(n, n_heads_total)``. A 1-D array is
            accepted and treated as a single head, so a single-task model still works.

        """
        source = np.asarray(source, dtype=np.float64)
        if source.ndim == 1:
            source = source[:, None]
        target = np.asarray(target, dtype=np.float64).ravel()
        if source.shape[0] != target.shape[0]:
            raise CalibrationError(
                f"source has {source.shape[0]} rows and target {target.shape[0]}"
            )
        finite = np.isfinite(target) & np.isfinite(source).all(axis=1)
        if int(finite.sum()) < 3:
            raise CalibrationError("Fewer than three reference points with finite values.")
        source, target = source[finite], target[finite]

        order = _rank_heads_by_correlation(source, target)
        self.selected_model_head = int(order[0])
        self._inner.fit(
            target=target.astype(np.float32),
            source=source[:, self.selected_model_head].astype(np.float32),
        )

    def transform(self, source: np.ndarray) -> np.ndarray:
        """
        Select the fitted head from the matrix and transform it.

        Parameters
        ----------
        source
            Predictions for every head, shape ``(n, n_heads_total)``, as returned by
            ``predict(..., return_matrix=True)``.

        """
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")
        head = self.selected_model_head
        rows, n_heads = source_shape(source)
        if n_heads <= head:
            raise CalibrationError(
                f"source has {n_heads} heads, but the calibration was fitted on a model "
                f"with at least {head + 1}."
            )
        if rows == 0:
            return np.array([])
        column = take_columns(source, [head])[:, 0]
        return np.asarray(self._inner.transform(column.astype(np.float32)), dtype=np.float64)


class MultiHeadPiecewiseLinearCalibration(_SingleHeadCalibration):
    """Piece-wise linear calibration on whichever head correlates best with the reference."""

    def __init__(
        self,
        number_of_splits: int = 10,
        extrapolate: bool = True,
        use_median: bool = False,
        min_samples_per_segment: int = 20,
    ) -> None:
        """
        Piece-wise linear calibration on whichever head correlates best with the reference.

        Parameters
        ----------
        number_of_splits : int
            Number of segments to split the source value range into.
            More segments allow more flexibility but may lead to overfitting.
        extrapolate : bool
            If True, allows extrapolation outside the fitted source value range.
            If False, clips input values to the fitted range.
        use_median : bool
            If True, uses the median of each segment to define anchors. If False, uses the mean.
        min_samples_per_segment : int
            Minimum number of samples required for a segment to contribute an anchor.
            Segments with fewer samples are skipped, which helps avoid unstable anchors in
            sparse regions when using many splits.

        """
        super().__init__()
        self._inner = PiecewiseLinearCalibration(
            number_of_splits=number_of_splits,
            extrapolate=extrapolate,
            use_median=use_median,
            min_samples_per_segment=min_samples_per_segment,
        )


class MultiHeadSplineCalibration(_SingleHeadCalibration):
    """Spline-based calibration on whichever head correlates best with the reference."""

    def __init__(self) -> None:
        """Initialize MultiHeadSplineCalibration."""
        super().__init__()
        self._inner = SplineTransformerCalibration()


class MultiHeadRidgeCalibration(MultiHeadCalibration):
    """
    Calibrate a multitask model against several of its LC-setup heads at once.

    Heads are ranked by Pearson correlation to the reference, the ``n_heads`` best are each
    calibrated with :class:`~deeplc.calibration.simple.SplineTransformerCalibration`, and a ridge
    regression maps the calibrated estimates onto the observed retention times. Never fits more
    head weights than half the reference size. For a single-task model (one head) this reduces to
    a spline followed by a linear rescaling.

    Parameters
    ----------
    n_heads
        How many of the best-correlating heads to combine.
    alphas
        Ridge strengths offered to the internal cross-validation.

    """

    def __init__(self, n_heads: int = 80, alphas: np.ndarray | None = None) -> None:
        """Initialize MultiHeadRidgeCalibration."""
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be at least 1, got {n_heads}")
        self.n_heads = n_heads
        self.alphas = np.logspace(-3, 6, 19) if alphas is None else np.asarray(alphas)
        self._head_idx: np.ndarray | None = None
        self._head_calibrations: list[SplineTransformerCalibration] = []
        self._ridge = None

    @property
    def is_fitted(self) -> bool:
        """True once the heads are selected, calibrated and weighted."""
        return self._head_idx is not None and self._ridge is not None

    def fit(self, target: np.ndarray, source: np.ndarray) -> None:
        """
        Select, calibrate and weight the heads.

        Parameters
        ----------
        target
            Observed retention times of the reference, shape ``(n,)``.
        source
            Reference predictions for every head, shape ``(n, n_heads_total)``. A 1-D array is
            accepted and treated as a single head, so a single-task model still works.

        """
        source = np.asarray(source, dtype=np.float64)
        if source.ndim == 1:
            source = source[:, None]
        target = np.asarray(target, dtype=np.float64).ravel()
        if source.shape[0] != target.shape[0]:
            raise CalibrationError(
                f"source has {source.shape[0]} rows and target {target.shape[0]}"
            )
        finite = np.isfinite(target) & np.isfinite(source).all(axis=1)
        if int(finite.sum()) < 3:
            raise CalibrationError("Fewer than three reference points with finite values.")
        source, target = source[finite], target[finite]

        order = _rank_heads_by_correlation(source, target)
        # never fit more weights than half the reference: a 230-peptide reference cannot support
        # eighty of them, and the ridge would be extrapolating its own regularisation
        n_heads = int(min(self.n_heads, source.shape[1], max(1, len(target) // 2)))
        self._head_idx = order[:n_heads]
        self.selected_model_head = int(order[0])

        calibrated = np.empty((len(target), n_heads), dtype=np.float64)
        self._head_calibrations = []
        for position, head in enumerate(self._head_idx):
            head_calibration = SplineTransformerCalibration()
            column = source[:, head].astype(np.float32)
            head_calibration.fit(target=target.astype(np.float32), source=column)
            calibrated[:, position] = np.asarray(
                head_calibration.transform(column), dtype=np.float64
            )
            self._head_calibrations.append(head_calibration)

        n_splits = int(min(5, max(2, len(target) // 20)))
        self._ridge = RidgeCV(alphas=self.alphas, cv=n_splits).fit(calibrated, target)
        LOGGER.info(
            "Calibrated on %d of %d heads with ridge strength %.4g; head %d correlates best.",
            n_heads,
            source.shape[1],
            float(getattr(self._ridge, "alpha_", float("nan"))),
            self.selected_model_head,
        )

    def transform(self, source: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions of the model this calibration was fitted with.

        Parameters
        ----------
        source
            Predictions for every head, shape ``(n, n_heads_total)``, as returned by
            ``predict(..., return_matrix=True)``.

        """
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")
        head_idx = cast(np.ndarray, self._head_idx)
        rows, n_heads = source_shape(source)
        if n_heads <= int(head_idx.max()):
            raise CalibrationError(
                f"source has {n_heads} heads, but the calibration was fitted on a model "
                f"with at least {int(head_idx.max()) + 1}."
            )
        if rows == 0:
            return np.array([])
        return np.asarray(self._ridge.predict(self._calibrated_columns(source)), dtype=np.float64)

    def _calibrated_columns(self, source) -> np.ndarray:
        """Give each selected head's own estimate of the retention time, in reference units."""
        head_idx = cast(np.ndarray, self._head_idx)
        # One request for every selected head, so a lazy source evaluates them in a single
        # pass rather than once per head.
        columns = take_columns(source, head_idx)
        return np.column_stack(
            [
                np.asarray(
                    cal.transform(columns[:, position].astype(np.float32)), dtype=np.float64
                )
                for position, cal in enumerate(self._head_calibrations)
            ]
        )

    def disagreement(self, source: np.ndarray) -> np.ndarray | None:
        """
        Report how far the combined setup heads lie apart for each input, in reference units.

        Every selected head estimates the retention time of the same peptide, so the spread of
        those estimates, weighted by the ridge weight each head received, is an uncertainty
        that varies from peptide to peptide rather than only along the gradient. Returns None
        while the calibration is unfitted or combines a single head, which carries no spread.
        """
        if not self.is_fitted:
            return None
        rows, _ = source_shape(source)
        if rows == 0 or len(cast(np.ndarray, self._head_idx)) < 2:
            return None
        weights = np.abs(np.asarray(self._ridge.coef_, dtype=np.float64).ravel())
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
        estimates = self._calibrated_columns(source)
        mean = estimates @ weights
        return np.sqrt(((estimates - mean[:, None]) ** 2) @ weights)


def upgrade_calibration(calibration: Calibration | MultiHeadCalibration) -> MultiHeadCalibration:
    """
    Wrap a naive, single-series calibration in its matching ``MultiHeadCalibration``.

    A ``MultiHeadCalibration`` instance is returned unchanged. A naive
    :class:`~deeplc.calibration.simple.Calibration` must not be fitted yet: a fitted instance
    carries no record of which head it was fit on, so there is nothing to wrap it around.

    Parameters
    ----------
    calibration
        The calibration to wrap, naive or already multi-head.

    Returns
    -------
    MultiHeadCalibration
        ``calibration`` itself if it already is one, otherwise a new wrapper around it.

    """
    if isinstance(calibration, MultiHeadCalibration):
        return calibration
    if not isinstance(calibration, Calibration):
        raise ValueError(
            f"Expected calibration to be of type `Calibration` or `MultiHeadCalibration`, got "
            f"{type(calibration)}"
        )
    if not isinstance(calibration, (PiecewiseLinearCalibration, SplineTransformerCalibration)):
        raise ValueError(
            f"No MultiHeadCalibration counterpart is known for {type(calibration).__name__}."
        )
    if calibration.is_fitted:
        raise CalibrationError(
            "A fitted, naive Calibration cannot be upgraded to a MultiHeadCalibration: it "
            "carries no record of which head it was fit on. Fit a MultiHead*Calibration instead."
        )
    if isinstance(calibration, PiecewiseLinearCalibration):
        return MultiHeadPiecewiseLinearCalibration(
            number_of_splits=calibration.number_of_splits,
            extrapolate=calibration.extrapolate,
            use_median=calibration.use_median,
            min_samples_per_segment=calibration.min_samples_per_segment,
        )
    return MultiHeadSplineCalibration()


def _rank_heads_by_correlation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Head indices by decreasing Pearson correlation with the target, in one pass.

    Vectorised because a fused-trunk multitask model can have thousands of heads to rank at once.
    """
    centred = source - source.mean(axis=0)
    target_centred = target - target.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = np.sqrt((centred**2).sum(axis=0) * (target_centred**2).sum())
        correlation = (centred * target_centred[:, None]).sum(axis=0) / denominator
    correlation = np.where(np.isfinite(correlation), correlation, -np.inf)
    return np.argsort(-correlation)
