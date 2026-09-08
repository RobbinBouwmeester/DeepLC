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


def as_head_matrix(source):
    """
    Coerce a source to a two-dimensional float64 matrix, unless it is a head source.

    Anything array-like is converted exactly as before, so a list, a tuple or the
    one-dimensional output of a single-task model all keep working. An object that marks
    itself with ``is_head_source``, such as :class:`deeplc.core.HeadColumnSource`, is passed
    through: it answers ``.shape`` and ``source[:, heads]`` like an array but evaluates only
    the heads that are asked for, which for a multitask model is a few dozen of thousands.
    """
    if getattr(source, "is_head_source", False):
        return source
    source = np.asarray(source, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, None]
    return source


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
        # The matrix arrives from the model as float32 and is (n, 6,543) wide; promoting it
        # here doubled a 276 MB reference to 552 MB for no gain, since every head's column is
        # cast to float32 again for its spline and the ranking accumulates in float64 itself.
        source = np.asarray(source)
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
        source = as_head_matrix(source)
        head = self.selected_model_head
        if source.shape[1] <= head:
            raise CalibrationError(
                f"source has {source.shape[1]} heads, but the calibration was fitted on a model "
                f"with at least {head + 1}."
            )
        if source.shape[0] == 0:
            return np.array([])
        return np.asarray(
            self._inner.transform(
                np.asarray(source[:, head], dtype=np.float64).astype(np.float32)
            ),
            dtype=np.float64,
        )


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
        # The matrix arrives from the model as float32 and is (n, 6,543) wide; promoting it
        # here doubled a 276 MB reference to 552 MB for no gain, since every head's column is
        # cast to float32 again for its spline and the ranking accumulates in float64 itself.
        source = np.asarray(source)
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

        # RidgeCV without an explicit cv uses the closed-form leave-one-out route, which
        # decomposes the design once instead of refitting it for every fold and alpha: 0.08 s
        # against 0.84 s on a 10,541-peptide reference, and accuracy-neutral over ten held-out
        # setups (median MAE ratio 1.0000, better on five and worse on five).
        #
        # Small references keep the fold-based search. The eighty calibrated columns are
        # nearly collinear, which makes a leave-one-out estimate jumpy when there are few
        # rows: on a 725-peptide reference it chose alpha 1 against 316 and cost 3.7 % of
        # accuracy. Below the threshold the fold-based search costs about 0.15 s, so there is
        # nothing to win there anyway.
        n_splits = int(min(5, max(2, len(target) // 20)))
        self._ridge = RidgeCV(
            alphas=self.alphas, cv=None if len(target) >= 2000 else n_splits
        ).fit(calibrated, target)
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
        source = as_head_matrix(source)
        head_idx = cast(np.ndarray, self._head_idx)
        if source.shape[1] <= int(head_idx.max()):
            raise CalibrationError(
                f"source has {source.shape[1]} heads, but the calibration was fitted on a model "
                f"with at least {int(head_idx.max()) + 1}."
            )
        if source.shape[0] == 0:
            return np.array([])
        return np.asarray(self._ridge.predict(self._calibrated_columns(source)), dtype=np.float64)

    def _calibrated_columns(self, source: np.ndarray) -> np.ndarray:
        """Give each selected head's own estimate of the retention time, in reference units."""
        head_idx = cast(np.ndarray, self._head_idx)
        # Asked for in one go rather than head by head: a lazy source then evaluates them in
        # a single pass, and for an array this is the same slice.
        columns = np.asarray(source[:, head_idx], dtype=np.float64)
        return np.column_stack(
            [
                np.asarray(cal.transform(columns[:, i].astype(np.float32)), dtype=np.float64)
                for i, cal in enumerate(self._head_calibrations)
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
        columns = as_head_matrix(source)
        if columns.shape[0] == 0 or len(cast(np.ndarray, self._head_idx)) < 2:
            return None
        weights = np.abs(np.asarray(self._ridge.coef_, dtype=np.float64).ravel())
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
        estimates = self._calibrated_columns(columns)
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
    target_centred = np.asarray(target, dtype=np.float64).ravel()
    target_centred = target_centred - target_centred.mean()
    target_norm = float(np.sqrt((target_centred**2).sum()))
    n_rows, n_heads = source.shape

    # Centring the source would allocate a copy of the whole matrix, and squaring it another:
    # at 6,543 heads and a 20,000-peptide reference that is well over a gigabyte of
    # temporaries for a vector of 6,543 numbers. Because the centred target sums to zero,
    # sum_i (x_i - xbar) * yc_i is just sum_i x_i * yc_i, so the covariance is one dot
    # product per block and the variance follows from the block's own sums. Blocks keep the
    # accumulation in float64 without ever holding more than a slice of the matrix.
    correlation = np.empty(n_heads, dtype=np.float64)
    block = 512
    with np.errstate(invalid="ignore", divide="ignore"):
        for start in range(0, n_heads, block):
            chunk = np.asarray(source[:, start : start + block], dtype=np.float64)
            covariance = chunk.T @ target_centred
            variance = np.einsum("ij,ij->j", chunk, chunk) - n_rows * chunk.mean(axis=0) ** 2
            correlation[start : start + block] = covariance / np.sqrt(variance * target_norm**2)
    correlation = np.where(np.isfinite(correlation), correlation, -np.inf)
    return np.argsort(-correlation)
