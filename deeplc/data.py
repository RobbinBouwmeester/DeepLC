"""Dataset classes and utilities for DeepLC."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar, overload

import numpy as np
import torch
from psm_utils import Peptidoform, PSMList
from torch.utils.data import Dataset, Subset

from deeplc._batch_features import encode_batch_features, supports
from deeplc._features import encode_peptidoform

_DatasetT = TypeVar("_DatasetT", bound=Dataset)

LOGGER = logging.getLogger(__name__)


class DeepLCDataset(Dataset):
    """Custom Dataset class for DeepLC used for loading features from peptide sequences."""

    def __init__(
        self,
        peptidoforms: list[Peptidoform | str],
        target_retention_times: np.ndarray | None = None,
        add_ccs_features: bool = False,
        add_terminal_composition: bool = False,
        padding_length: int = 60,
        legacy_positional_deltas: bool = True,
        include_rolling_sum: bool = True,
        vectorised_encoding: bool = True,
    ):
        """
        Initialize the DeepLCDataset.

        Parameters
        ----------
        peptidoforms
            A list of peptidoforms, which can be either Peptidoform objects or their string
            representations.
        target_retention_times
            An array of target retention times corresponding to the peptidoforms. If None, targets
            will be set to NaN.
        add_ccs_features
            Whether to include CCS features in the encoded representation. Default is False.
        add_terminal_composition
            Whether to append the N- and C-terminal group composition to the global
            feature vector, lengthening it from 55 to 67. Required by models trained on
            that layout; see the ``feature_spec`` recorded in such a model. Default is
            False.
        padding_length
            Length the per-position matrices are padded or truncated to. Must match the
            value the model was trained with: the fused-trunk architecture pools rather
            than flattens, so a mismatch changes the representation without changing any
            shape and would not raise. Default is 60.
        legacy_positional_deltas
            Whether to place modification deltas in the positional block the way
            versions before 4.0.1 did. That placement was wrong and 4.0.1 corrected
            it, but every model released against this dataset class was trained on
            it, so **the default is True**: a dataset exists to feed a model, and
            feeding a model an encoding it was not trained on changes its
            predictions on modified peptides without any error.

            Set it to False for a model trained after the correction.
            :func:`deeplc.core.predict` and :func:`deeplc.core.finetune` do this
            automatically from the ``feature_spec`` a self-describing checkpoint
            carries, and :func:`deeplc.core.train` does it for newly trained
            models. Note that :func:`deeplc._features.encode_peptidoform`, whose
            job is correct featurisation rather than model compatibility, defaults
            the other way. Affects modified peptidoforms only.
        include_rolling_sum
            Whether to build the rolling-sum matrix. A convolutional trunk reads the
            per-position matrix directly and ignores this one, so building it is pure cost
            for such a model; False puts an empty array in its place. Default is True.
        vectorised_encoding
            Whether :meth:`encode_batch` may gather the residue-only part of a batch in one
            pass instead of encoding peptide by peptide. It produces the same values, dtype
            included, and falls back per peptide for anything it does not cover; False forces
            the per-peptide encoder for everything. Default is True.

        Raises
        ------
        ValueError
            If ``target_retention_times`` is provided and its length does not match the number of
            peptidoforms.

        """
        self.peptidoforms = peptidoforms
        self.target_retention_times = target_retention_times
        self.add_ccs_features = add_ccs_features
        self.add_terminal_composition = add_terminal_composition
        self.padding_length = padding_length
        self.legacy_positional_deltas = legacy_positional_deltas
        self.include_rolling_sum = include_rolling_sum
        self.vectorised_encoding = vectorised_encoding
        if self.target_retention_times is not None and len(self.target_retention_times) != len(
            self.peptidoforms
        ):
            raise ValueError(
                f"Length of target_retention_times ({len(self.target_retention_times)}) does not "
                f"match length of peptidoforms ({len(self.peptidoforms)})"
            )

    def __len__(self) -> int:
        """Return number of peptidoforms in the dataset."""
        return len(self.peptidoforms)

    def variant(self, indices: Sequence[int], padding_length: int) -> DeepLCDataset:
        """
        Return a subset of this dataset's peptidoforms, encoded in a shorter window.

        Used by the prediction path to run short peptides in a window that fits them
        instead of padding every one to the model's full length. The peptidoform objects
        themselves are shared rather than copied, so the parsing psm_utils caches on them
        is not paid twice.

        Parameters
        ----------
        indices
            Positions in this dataset to include, in the order wanted.
        padding_length
            Window the subset is encoded in.

        """
        targets = self.target_retention_times
        return type(self)(
            peptidoforms=[self.peptidoforms[i] for i in indices],
            target_retention_times=None if targets is None else targets[list(indices)],
            add_ccs_features=self.add_ccs_features,
            add_terminal_composition=self.add_terminal_composition,
            padding_length=padding_length,
            legacy_positional_deltas=self.legacy_positional_deltas,
            include_rolling_sum=self.include_rolling_sum,
            vectorised_encoding=self.vectorised_encoding,
        )

    #: Arrays the encoder returns, in the order the models take them.
    FEATURE_KEYS = ("matrix", "matrix_sum", "matrix_global", "matrix_hc")

    def _encode_kwargs(self) -> dict:
        """Return the encoder settings this dataset was built with."""
        return {
            "add_ccs_features": self.add_ccs_features,
            "add_terminal_composition": self.add_terminal_composition,
            "padding_length": self.padding_length,
            "legacy_positional_deltas": self.legacy_positional_deltas,
            "include_rolling_sum": self.include_rolling_sum,
        }

    def encode_batch(self, indices: Sequence[int]) -> tuple[torch.Tensor, ...]:
        """
        Encode several peptidoforms straight into one tensor per feature.

        :meth:`__getitem__` builds four arrays and four tensors for a single peptide, which a
        DataLoader then stacks into a batch: several copies of a few hundred bytes each with a
        good deal of Python around them. Writing the encoder output into batch buffers
        instead measured 1.7x faster over the whole feature path, 0.153 against 0.089 ms per
        peptide, and returns the same values.

        Parameters
        ----------
        indices
            Positions in this dataset to encode, in the order wanted.

        """
        indices = list(indices)
        if not indices:
            raise ValueError("No indices to encode.")

        if self.vectorised_encoding and supports(self.add_ccs_features, self.include_rolling_sum):
            # The residue-only part of the batch is a table gather; modifications and
            # anything the gather does not cover go through the per-peptide encoder, so the
            # values are the same either way. See deeplc._batch_features.
            features = encode_batch_features(
                [self.peptidoforms[index] for index in indices],
                padding_length=self.padding_length,
                add_terminal_composition=self.add_terminal_composition,
                legacy_positional_deltas=self.legacy_positional_deltas,
            )
            empty = np.zeros((len(indices), 0, features["matrix"].shape[-1]), dtype=np.float32)
            return (
                torch.from_numpy(features["matrix"].astype(np.float32)),
                torch.from_numpy(empty),
                torch.from_numpy(features["matrix_global"].astype(np.float32)),
                torch.from_numpy(features["matrix_hc"].astype(np.float32)),
            )

        buffers: list[np.ndarray] | None = None
        for row, index in enumerate(indices):
            features = encode_peptidoform(self.peptidoforms[index], **self._encode_kwargs())
            if buffers is None:
                buffers = [
                    np.empty((len(indices), *features[key].shape), dtype=np.float32)
                    for key in self.FEATURE_KEYS
                ]
            for buffer, key in zip(buffers, self.FEATURE_KEYS, strict=True):
                buffer[row] = features[key]
        return tuple(torch.from_numpy(buffer) for buffer in buffers or [])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Return encoded features and target RT for peptidoform at index."""
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)} instead.")
        features = encode_peptidoform(self.peptidoforms[idx], **self._encode_kwargs())
        feature_tuples = (
            torch.from_numpy(features["matrix"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_sum"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_global"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_hc"]).to(dtype=torch.float32),
        )
        targets = (
            self.target_retention_times[idx]
            if self.target_retention_times is not None
            else torch.tensor(float("nan"), dtype=torch.float32)
        )
        return feature_tuples, targets

    @classmethod
    def from_psm_list(
        cls,
        psm_list: PSMList,
        add_ccs_features: bool = False,
        add_terminal_composition: bool = False,
        padding_length: int = 60,
        legacy_positional_deltas: bool = True,
        include_rolling_sum: bool = True,
        vectorised_encoding: bool = True,
    ) -> DeepLCDataset:
        """
        Create a DeepLCDataset from a PSMList.

        Parameters
        ----------
        psm_list
            A PSMList containing the peptidoforms and their corresponding retention times.
        add_ccs_features
            Whether to include CCS features in the encoded representation. Default is False.
        add_terminal_composition
            Whether to append the N- and C-terminal group composition to the global
            feature vector, lengthening it from 55 to 67. Default is False.
        padding_length
            Length the per-position matrices are padded or truncated to. Must match the
            value the model was trained with: the fused-trunk architecture pools rather
            than flattens, so a mismatch changes the representation without changing any
            shape and would not raise. Default is 60.
        legacy_positional_deltas
            Whether to place modification deltas in the positional block the way
            versions before 4.0.1 did. That placement was wrong and 4.0.1 corrected
            it, but every model released against this dataset class was trained on
            it, so **the default is True**: a dataset exists to feed a model, and
            feeding a model an encoding it was not trained on changes its
            predictions on modified peptides without any error.

            Set it to False for a model trained after the correction.
            :func:`deeplc.core.predict` and :func:`deeplc.core.finetune` do this
            automatically from the ``feature_spec`` a self-describing checkpoint
            carries, and :func:`deeplc.core.train` does it for newly trained
            models. Note that :func:`deeplc._features.encode_peptidoform`, whose
            job is correct featurisation rather than model compatibility, defaults
            the other way. Affects modified peptidoforms only.

        include_rolling_sum
            Whether to build the rolling-sum matrix. Models with a convolutional trunk
            ignore it, and :func:`deeplc.core.predict` sets this from the model.
        vectorised_encoding
            Whether a batch may be encoded in one pass; see the class docstring.

        Returns
        -------
        DeepLCDataset
            A DeepLCDataset instance created from the provided PSMList.

        """
        peptidoforms = list(psm_list["peptidoform"])
        retention_times = psm_list["retention_time"]
        if None not in retention_times:
            target_retention_times = np.array(retention_times, dtype=np.float32)
        else:
            target_retention_times = None
        return cls(
            peptidoforms=peptidoforms,
            target_retention_times=target_retention_times,
            add_ccs_features=add_ccs_features,
            add_terminal_composition=add_terminal_composition,
            padding_length=padding_length,
            legacy_positional_deltas=legacy_positional_deltas,
            include_rolling_sum=include_rolling_sum,
            vectorised_encoding=vectorised_encoding,
        )


@overload
def split_datasets(
    train_data: _DatasetT,
    validation_data: _DatasetT,
    validation_split: float,
) -> tuple[_DatasetT, _DatasetT]: ...


@overload
def split_datasets(
    train_data: _DatasetT,
    validation_data: None,
    validation_split: float,
) -> tuple[Subset[_DatasetT], Subset[_DatasetT]]: ...


def split_datasets(
    train_data: Dataset,
    validation_data: Dataset | None,
    validation_split: float,
) -> tuple[Dataset, Dataset] | tuple[Subset, Subset]:
    """
    Split the dataset into training and validation sets.

    Parameters
    ----------
    train_data
        The dataset to be split.
    validation_data
        If provided, this dataset will be used as the validation set. If None, the train_data will
        be split.
    validation_split
        The fraction of the dataset to be used as the validation set if validation_data is None.

    Returns
    -------
    tuple[Dataset, Dataset] | tuple[Subset, Subset]
        A tuple containing the training and validation datasets.

    Raises
    ------
    ValueError
        If validation_data is None and train_data does not implement ``__len__`` method.

    """
    # TODO: Implement stratified splitting based on stripped sequence
    if validation_data is None:
        if not 0 < validation_split < 1:
            raise ValueError(
                f"validation_split must be between 0 and 1 (exclusive), got {validation_split}."
            )
        if not hasattr(train_data, "__len__"):
            raise ValueError("Dataset must implement __len__ method for automatic splitting")
        dataset_len = len(train_data)  # type: ignore[arg-type]
        if dataset_len < 2:
            raise ValueError(
                "Need at least 2 samples in train_data when validation_data is not provided."
            )
        val_size = max(1, int(dataset_len * validation_split))
        val_size = min(val_size, dataset_len - 1)
        train_size = dataset_len - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_data, [train_size, val_size]
        )
        LOGGER.info(
            "No validation data provided. Split training dataset into validation set of size "
            f"{len(val_dataset)} and training set of size {len(train_dataset)}"
        )
        return train_dataset, val_dataset
    else:
        LOGGER.info("Using provided validation dataset")
        return train_data, validation_data
