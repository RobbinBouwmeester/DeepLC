"""Training, predicting, and evaluating with PyTorch."""

import copy
import inspect
import logging
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    track,
)
from torch.utils.data import DataLoader, Dataset, Subset

from deeplc._architecture import DeepLCModel, FlexCNNMultitaskModel
from deeplc.data import DeepLCDataset

logger = logging.getLogger(__name__)


def load_model(
    model: torch.nn.Module | PathLike | str | None = None,
    device: str | None = None,
) -> DeepLCModel:
    """Load a model from a file or return a randomly initialized model if none is provided."""
    # If device is not specified, use the default device (GPU if available, else CPU)
    selected_device: torch.device | str = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if isinstance(model, (str, PathLike, Path)):
        raw = torch.load(model, weights_only=True, map_location=selected_device)

        # Newer checkpoints are a dict describing the model rather than a bare
        # state dict, so the architecture and its hyperparameters do not have to
        # be guessed from tensor shapes. Older files keep working unchanged.
        if isinstance(raw, dict) and "architecture" in raw:
            return _load_described_model(raw, selected_device)

        # Infer architecture hyperparameters from the saved state dict
        # Only checks n_heads and final_num_layers; other hyperparameters are set to defaults
        # May break for models saved with different architectures.
        n_heads = raw["heads.b2"].shape[0]
        final_num_layers = sum(
            1 for k in raw if k.startswith("shared_trunk.") and k.endswith(".weight")
        )
        loaded_model = DeepLCModel(n_heads=n_heads, final_num_layers=final_num_layers)
        if "adapter.0.weight" in raw:
            loaded_model.add_adapter(hidden_size=raw["adapter.0.weight"].shape[0])
        loaded_model.load_state_dict(raw)
    elif isinstance(model, (DeepLCModel, FlexCNNMultitaskModel)):
        loaded_model = model
        logger.debug("Using provided PyTorch model instance")
    elif model is None:
        loaded_model = DeepLCModel(n_heads=1)
        logger.debug("Initialized new DeepLCModel with default architecture")
    else:
        raise TypeError(f"Expected a DeepLCModel or a file path, got {type(model)} instead.")

    loaded_model.to(selected_device)

    return loaded_model


#: Architectures a described checkpoint may name, and the class to build.
_DESCRIBED_ARCHITECTURES = {
    "FlexCNNMultitaskModel": FlexCNNMultitaskModel,
}


def _load_described_model(blob: dict, device: torch.device | str) -> torch.nn.Module:
    """
    Build a model from a checkpoint that describes itself.

    The checkpoint carries the architecture name, its constructor arguments and
    the feature specification it was trained against. That last part matters:
    the model's first dense layer fixes the width of the global feature vector,
    so a model expecting terminal composition cannot be fed the shorter default
    vector. Attaching the specification to the returned module lets the caller
    build a matching dataset instead of inferring it.
    """
    name = blob["architecture"]
    try:
        cls = _DESCRIBED_ARCHITECTURES[name]
    except KeyError:
        raise ValueError(
            f"Checkpoint names architecture {name!r}, which this version of DeepLC "
            f"does not know. Known architectures: "
            f"{sorted(_DESCRIBED_ARCHITECTURES)}."
        ) from None

    kwargs = dict(blob.get("encoder_kwargs") or {})
    kwargs.update(blob.get("head_kwargs") or {})
    built = cls(n_tasks=blob["n_tasks"], **kwargs)
    built.load_state_dict(blob["state_dict"])

    # Carried on the instance so predict() can build a matching dataset.
    built.feature_spec = blob.get("feature_spec")
    built.target_units = blob.get("target_units")
    built.task_names = blob.get("task_names")

    logger.debug(
        "Loaded %s with %d tasks, feature spec %s, targets in %s",
        name,
        blob["n_tasks"],
        (built.feature_spec or {}).get("name", "unspecified"),
        built.target_units or "unspecified units",
    )
    built.to(device)
    built.eval()
    return built


def train(
    model: DeepLCModel | PathLike | str | None,
    train_dataset: DeepLCDataset | Subset[DeepLCDataset],
    validation_dataset: DeepLCDataset | Subset[DeepLCDataset],
    device: str | None = None,
    num_workers: int = 0,
    num_threads: int | None = None,
    learning_rate: float = 0.001,
    epochs: int = 25,
    batch_size: int = 512,
    patience: int = 10,
    freeze_epochs: int = 0,
    unfreeze_lr_scale: float = 0.1,
    show_progress: bool = True,
) -> torch.nn.Module:
    """
    Train or fine-tune the model.

    Parameters
    ----------
    model
        Model to train or path to model file.
    train_dataset
        Training dataset.
    validation_dataset
        Validation dataset.
    device
        Device to train on ('cpu' or 'cuda').
    num_workers
        Number of worker processes for data loading.
    num_threads
        Number of threads for model operations on CPU (ignored if using GPU).
    learning_rate
        Learning rate for optimizer.
    epochs
        Maximum number of training epochs.
    batch_size
        Batch size for training and validation.
    patience
        Number of epochs with no improvement before early stopping.
    freeze_epochs
        Number of initial epochs to train with backbone frozen (adapter only).
    unfreeze_lr_scale
        Learning rate multiplier applied after unfreezing the backbone.
    show_progress
        If True, display a Rich progress bar during training. If False, run silently.

    Returns
    -------
    torch.nn.Module
        Trained model.

    """
    torch.set_num_threads(num_threads or torch.get_num_threads())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, device)

    # Parse datasets; setup loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if len(train_loader) == 0:
        raise ValueError("Training data loader is empty. Provide at least one training sample.")
    if len(val_loader) == 0:
        raise ValueError(
            "Validation data loader is empty. Adjust validation data or validation_split."
        )

    has_freeze = hasattr(model, "freeze_backbone") and hasattr(model, "unfreeze_backbone")
    if has_freeze and freeze_epochs > 0:
        model.freeze_backbone()
    optimizer = _get_optimizer(model, learning_rate)
    loss_fn = torch.nn.L1Loss()

    best_model_wts = copy.deepcopy(model.state_dict())

    # Score the starting point, so training can never return a model worse than the
    # one it began with. With this left at infinity the first epoch always became the
    # best, even when it was worse: fine-tuning a small reference set could hand back
    # a fit whose predictions had collapsed onto the mean retention time, at ninety
    # times the error of the model it started from.
    best_val_loss = _validate_epoch(model, val_loader, loss_fn, device)
    logger.debug("Validation loss before training: %.4f", best_val_loss)
    epochs_no_improve = 0

    with _create_progress(disable=not show_progress) as progress:
        epoch_task = progress.add_task("Epochs", total=epochs, status="")

        for epoch in range(epochs):
            if has_freeze and freeze_epochs > 0 and epoch == freeze_epochs:
                model.unfreeze_backbone()
                optimizer = _get_optimizer(model, learning_rate * unfreeze_lr_scale)

            avg_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
            avg_val_loss = _validate_epoch(model, val_loader, loss_fn, device)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Update epoch bar with loss info
            status = f"loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  best={best_val_loss:.4f}"
            if epochs_no_improve >= patience:
                status += "  [yellow]early stop[/yellow]"
            progress.update(epoch_task, advance=1, status=status)

            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_model_wts)
    return model


def predict(
    model: torch.nn.Module | PathLike | str | None,
    data: Dataset,
    device: str | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
    num_threads: int | None = None,
    show_progress: bool = True,
    task_idx: Sequence[int] | None = None,
    length_buckets: bool = True,
) -> torch.Tensor:
    """
    Predict using the model for the given dataset.

    ``length_buckets`` runs length-sorted chunks in a window that fits them rather than
    padding every peptide to the model's full window; see :func:`_length_buckets`. It is
    exact for models that report a ``padding_reach`` and ignored for those that do not.
    Set it to False to force one pass over the data in the dataset's own window.
    """
    # ``task_idx`` selects which LC setups a multitask model evaluates. Without
    # it a model trained on thousands of setups returns a column per setup: at
    # 6,543 setups and a million peptides that output alone is tens of gigabytes,
    # so a caller wanting one column should ask for one column. Models whose
    # forward does not accept it ignore the argument.
    torch.set_num_threads(num_threads or torch.get_num_threads())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, device)

    buckets = _length_buckets(model, data, batch_size) if length_buckets else None
    if buckets is None:
        data_loader = DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        predictions = _predict_epoch(
            model, data_loader, device, show_progress=show_progress, task_idx=task_idx
        )
        return predictions.cpu().detach()

    out: torch.Tensor | None = None
    for indices, subset in buckets:
        part = _predict_epoch(
            model,
            DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            device,
            show_progress=show_progress,
            task_idx=task_idx,
        ).cpu()
        if out is None:
            out = torch.empty((len(data), part.shape[1]), dtype=part.dtype)
        out[indices] = part
    if out is None:
        raise ValueError("Dataset is empty — nothing to predict.")
    return out.detach()


#: Widest spread of peptide lengths allowed inside one prediction chunk. Small enough that
#: no peptide carries much padding, large enough that the dense middle of a length
#: distribution still fills a batch.
#: A tight window is worth more than a full batch: enforcing a minimum chunk size, so that the
#: sparse long tail rides along in a wider window, was measured slower (1,546 against 1,503
#: peptidoforms/s at a floor of 512 and 1,449 at 2,048).
_LENGTH_BAND = 4


def _residue_count(peptidoform: object) -> int:
    """
    Residues in a peptidoform, whether it arrives parsed or as a ProForma string.

    A dataset may hold either. The string form cannot be counted by its length, since
    modifications and the charge state are part of it, so it is parsed once here rather
    than per encoded item.
    """
    sequence = getattr(peptidoform, "sequence", None)
    if sequence is None:
        from psm_utils import Peptidoform

        sequence = Peptidoform(str(peptidoform)).sequence
    return len(sequence)


def _length_buckets(
    model: torch.nn.Module, data: Dataset, batch_size: int
) -> list[tuple[torch.Tensor, Dataset]] | None:
    """
    Split the dataset into length-sorted chunks, each encoded in a window that fits it.

    Padding every peptide to the model's full window makes the convolutions work on
    padding: at a 60-position window and a median peptide of 16 residues most of the
    trunk's arithmetic is spent on positions that are masked out again before pooling.
    Sorting by length and giving each chunk a window of its own longest peptide plus the
    trunk's reach is exact - it was measured identical to the full window over 50,000
    peptides - and about three times faster on CPU.

    Returns None when the model does not report a reach, when the data is not a
    DeepLCDataset, or when there is nothing to gain, so the caller falls back to one pass.
    """
    reach = getattr(model, "padding_reach", None)
    if reach is None or not isinstance(data, DeepLCDataset) or len(data) == 0:
        return None

    lengths = np.fromiter(
        (_residue_count(p) for p in data.peptidoforms), dtype=np.int64, count=len(data)
    )
    window = data.padding_length
    order = np.argsort(lengths, kind="stable")
    sorted_lengths = lengths[order]

    # A chunk is cut either at the batch size or as soon as its longest peptide would exceed
    # the shortest by more than _LENGTH_BAND, so no peptide is padded much beyond its own
    # length. Fixed-size chunks are not enough: the longest chunk of a length-sorted set holds
    # thousands of ordinary peptides alongside the few long ones and inherits their window,
    # which on a 20,000-peptide set cost 43 % of the throughput.
    buckets: list[tuple[torch.Tensor, Dataset]] = []
    start = 0
    while start < len(order):
        stop = min(start + batch_size, len(order))
        band = sorted_lengths[start] + _LENGTH_BAND
        within = int(np.searchsorted(sorted_lengths[start:stop], band, side="right"))
        stop = start + max(within, 1)
        padding = int(min(window, sorted_lengths[stop - 1] + reach))
        chunk = order[start:stop]
        buckets.append((torch.as_tensor(chunk), data.variant(chunk.tolist(), padding)))
        start = stop

    if len(buckets) == 1 and buckets[0][1].padding_length >= window:
        return None  # one chunk at the full window is what the plain path already does
    return buckets


def supports_task_subset(model: torch.nn.Module) -> bool:
    """Whether ``model.forward`` accepts a ``task_idx`` argument."""
    try:
        return "task_idx" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def evaluate(
    model: torch.nn.Module | PathLike | str | None,
    data: Dataset,
    device: str | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
    num_threads: int | None = None,
) -> float:
    """Evaluate the model on the given dataset."""
    torch.set_num_threads(num_threads or torch.get_num_threads())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, device)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss_fn = torch.nn.L1Loss()
    avg_loss = _validate_epoch(model, data_loader, loss_fn, device)
    return avg_loss


def _get_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )


def _train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str,
) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    for features, targets in data_loader:
        features = [feature_tensor.to(device) for feature_tensor in features]
        targets = targets.to(device).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(*features)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return float(running_loss / len(data_loader))


def _validate_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable,
    device: str,
) -> float:
    """Validate the model for one epoch."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, targets in data_loader:
            features = [feature_tensor.to(device) for feature_tensor in features]
            targets = targets.to(device).view(-1, 1)
            outputs = model(*features)
            val_loss += loss_fn(outputs, targets).item()
    return float(val_loss / len(data_loader))


def _predict_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    show_progress: bool = False,
    task_idx: Sequence[int] | None = None,
) -> torch.Tensor:
    """Predict using the model for one epoch."""
    model.eval()
    selected = None
    if task_idx is not None and supports_task_subset(model):
        selected = torch.as_tensor(list(task_idx), dtype=torch.long, device=device)
    predictions = []
    with torch.no_grad():
        for features, _ in track(
            data_loader, description="Predicting...", transient=True, disable=not show_progress
        ):
            features = [feature_tensor.to(device) for feature_tensor in features]
            outputs = model(*features) if selected is None else model(*features, task_idx=selected)
            predictions.append(outputs.cpu())
    if not predictions:
        raise ValueError("Dataset is empty — nothing to predict.")
    return torch.cat(predictions, dim=0)


def _create_progress(disable: bool = False) -> Progress:
    """Create a Rich progress bar for training."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        TextColumn("|"),
        TextColumn("{task.fields[status]}"),
        disable=disable,
        transient=True,
    )
