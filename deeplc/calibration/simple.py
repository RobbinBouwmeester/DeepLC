"""
Naive calibration utilities.

Every class here maps a single series of raw predictions onto a single series of observed
values: ``fit(target: (n,), source: (n,))`` / ``transform(source: (n,)) -> (n,)``. None of them
know about multitask models or LC-setup heads; see :mod:`deeplc.calibration.multihead` for the
classes that select a head from a ``(n, n_heads)`` prediction matrix and delegate to one of these.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore[import]
from sklearn.pipeline import Pipeline, make_pipeline  # type: ignore[import]
from sklearn.preprocessing import SplineTransformer  # type: ignore[import]

from deeplc.exceptions import CalibrationError

LOGGER = logging.getLogger(__name__)


class Calibration(ABC):
    """Abstract base class for a single-series calibration."""

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
        """Fit the calibration from source to target."""
        ...

    @abstractmethod
    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source values into the calibrated target space."""
        ...


class IdentityCalibration(Calibration):
    """No calibration; returns inputs unchanged."""

    def __init__(self) -> None:
        """Initialize IdentityCalibration."""
        super().__init__()

    @property
    def is_fitted(self) -> bool:
        """Always fitted; identity calibration requires no fitting."""
        return True

    def fit(self, target: np.ndarray, source: np.ndarray) -> None:  # noqa: ARG002
        """No-op; identity calibration does not fit."""
        return None

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Return source unchanged."""
        return source


class PiecewiseLinearCalibration(Calibration):
    """Piece-wise linear calibration based on per-split anchors."""

    def __init__(
        self,
        number_of_splits: int = 10,
        extrapolate: bool = True,
        use_median: bool = False,
        min_samples_per_segment: int = 20,
    ) -> None:
        """
        Piece-wise linear calibration based on per-split anchors.

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
        self.number_of_splits = int(number_of_splits)
        self.extrapolate = bool(extrapolate)
        self.use_median = bool(use_median)
        self.min_samples_per_segment = int(min_samples_per_segment)

        if self.min_samples_per_segment < 1:
            raise ValueError("`min_samples_per_segment` must be >= 1.")

        self._calibrate_min: float | None = None
        self._calibrate_max: float | None = None
        self._source_breakpoints: np.ndarray | None = None
        self._slopes: np.ndarray | None = None
        self._intercepts: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """True if the calibration model has been fitted."""
        return (
            self._calibrate_min is not None
            and self._calibrate_max is not None
            and self._source_breakpoints is not None
            and self._slopes is not None
            and self._intercepts is not None
        )

    @property
    def calibrate_min(self) -> float | None:
        """Minimum source value seen during fitting."""
        return self._calibrate_min

    @property
    def calibrate_max(self) -> float | None:
        """Maximum source value seen during fitting."""
        return self._calibrate_max

    def fit(self, target: np.ndarray, source: np.ndarray) -> None:
        """Fit a piece-wise linear model mapping source to target values."""
        target, source = _prepare_series(target, source)

        cal_min = float(source[0])
        cal_max = float(source[-1])
        if (not np.isfinite(cal_min)) or (not np.isfinite(cal_max)) or (cal_max <= cal_min):
            raise CalibrationError("Source values have zero or invalid range; cannot calibrate.")

        boundaries = np.linspace(cal_min, cal_max, self.number_of_splits + 1, dtype=np.float32)
        starts_raw: np.ndarray = np.searchsorted(source, boundaries[:-1], side="left")  # type: ignore[var-annotated]
        ends_raw: np.ndarray = np.searchsorted(source, boundaries[1:], side="left")  # type: ignore[var-annotated]

        # Merge adjacent sparse segments by assigning each segment to a group based on
        # how many min_samples-sized chunks the cumulative count has crossed so far.
        # Segments whose cumulative count falls within the same chunk share a group id
        # and are merged into a single anchor.
        counts = ends_raw - starts_raw
        group_ids = (np.cumsum(counts) - 1) // self.min_samples_per_segment
        group_start_indices = np.concatenate(([0], np.flatnonzero(np.diff(group_ids)) + 1))
        group_end_indices = np.concatenate((group_start_indices[1:] - 1, [len(starts_raw) - 1]))

        starts = starts_raw[group_start_indices]
        ends = ends_raw[group_end_indices]

        # Compute anchors for all segments
        aggregate_func = np.median if self.use_median else np.mean
        tgt_anchors = np.array(
            [aggregate_func(target[s:e]) for s, e in zip(starts, ends, strict=True)],
            dtype=np.float32,
        )
        src_anchors = np.array(
            [aggregate_func(source[s:e]) for s, e in zip(starts, ends, strict=True)],
            dtype=np.float32,
        )

        if len(src_anchors) < 2:
            raise CalibrationError(
                "Not enough anchor points to build a piecewise calibration (need >= 2)."
            )

        src_arr = np.asarray(src_anchors, dtype=np.float32)
        tgt_arr = np.asarray(tgt_anchors, dtype=np.float32)
        keep = np.concatenate(([True], src_arr[1:] > src_arr[:-1]))
        src_arr = src_arr[keep]
        tgt_arr = tgt_arr[keep]
        if src_arr.size < 2:
            raise CalibrationError(
                "After removing degenerate anchors, not enough points remain to define segments."
            )

        delta_src = src_arr[1:] - src_arr[:-1]
        delta_tgt = tgt_arr[1:] - tgt_arr[:-1]
        slopes = delta_tgt / delta_src
        intercepts = (-src_arr[:-1] * slopes) + tgt_arr[:-1]

        self._source_breakpoints = src_arr.astype(np.float32)
        self._slopes = slopes.astype(np.float32)
        self._intercepts = intercepts.astype(np.float32)
        self._calibrate_min = cal_min
        self._calibrate_max = cal_max

        LOGGER.debug(
            "Piecewise fit: anchors=%d, segments=%d, range=[%.3f, %.3f]",
            len(self._source_breakpoints),
            len(self._slopes),
            self._calibrate_min,
            self._calibrate_max,
        )

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source values using the fitted piece-wise linear model."""
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")

        # Ensure type checking knows these are not None
        self._source_breakpoints = cast(np.ndarray, self._source_breakpoints)
        self._slopes = cast(np.ndarray, self._slopes)
        self._intercepts = cast(np.ndarray, self._intercepts)

        if source.shape[0] == 0:
            return np.array([])

        x = source.astype(np.float32, copy=False)
        x_eval = (
            np.clip(x, self._calibrate_min, self._calibrate_max) if not self.extrapolate else x
        )

        idx = np.searchsorted(self._source_breakpoints, x_eval, side="right") - 1
        idx = np.clip(idx, 0, len(self._source_breakpoints) - 2)
        y = self._slopes[idx] * x_eval + self._intercepts[idx]
        return y

    def get_calibration_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the calibration anchors as two arrays (x, y)."""
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")

        # Ensure type checking knows these are not None
        self._source_breakpoints = cast(np.ndarray, self._source_breakpoints)
        self._slopes = cast(np.ndarray, self._slopes)
        self._intercepts = cast(np.ndarray, self._intercepts)

        x = self._source_breakpoints.astype(np.float64)
        y = np.empty_like(x, dtype=np.float64)
        y[0] = float(self._slopes[0] * x[0] + self._intercepts[0])
        if len(x) > 1:
            prev_idx = np.arange(0, len(x) - 1)
            y[1:] = (self._slopes[prev_idx] * x[1:] + self._intercepts[prev_idx]).astype(
                np.float64
            )
        return x, y


class SplineTransformerCalibration(Calibration):
    """Spline-based calibration using sklearn's SplineTransformer."""

    def __init__(self) -> None:
        """Initialize SplineTransformerCalibration."""
        super().__init__()
        self._calibrate_min: float | None = None
        self._calibrate_max: float | None = None
        self._model_left: LinearRegression | None = None
        self._model_main: Pipeline | LinearRegression | None = None
        self._model_right: LinearRegression | None = None

    @property
    def is_fitted(self) -> bool:
        """True if the calibration model has been fitted."""
        return (
            self._calibrate_min is not None
            and self._calibrate_max is not None
            and self._model_left is not None
            and self._model_main is not None
            and self._model_right is not None
        )

    def fit(
        self,
        target: np.ndarray,
        source: np.ndarray,
    ) -> None:
        """Fit a spline-based model mapping source to target values."""
        target, source = _prepare_series(target, source)

        # Spline model
        spline = SplineTransformer(degree=4, n_knots=int(len(source) / 500) + 5)
        spline_model = make_pipeline(spline, LinearRegression())
        spline_model.fit(source.reshape(-1, 1), target)

        # Linear fit for left trail
        n_top = int(len(source) * 0.1)
        X_left = source[:n_top]
        y_left = target[:n_top]
        linear_model_left = LinearRegression()
        linear_model_left.fit(X_left.reshape(-1, 1), y_left)

        # Linear fit for right trail
        X_right = source[-n_top:]
        y_right = target[-n_top:]
        linear_model_right = LinearRegression()
        linear_model_right.fit(X_right.reshape(-1, 1), y_right)

        self._calibrate_min = float(np.min(source))
        self._calibrate_max = float(np.max(source))
        self._model_left = linear_model_left
        self._model_main = spline_model
        self._model_right = linear_model_right

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source values using the fitted spline model."""
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")

        # Ensure type checking knows the models are not None
        model_main = cast(Pipeline | LinearRegression, self._model_main)
        model_left = cast(LinearRegression, self._model_left)
        model_right = cast(LinearRegression, self._model_right)
        calibrate_min = cast(float, self._calibrate_min)
        calibrate_max = cast(float, self._calibrate_max)

        if source.shape[0] == 0:
            return np.array([])

        flat = source.ravel()
        cal_preds = np.asarray(model_main.predict(source.reshape(-1, 1)), dtype=float)

        # The trails only ever supply the points outside the fitted range, which on a
        # reference that covers its own gradient is usually none of them. Predicting them
        # for every point tripled the work of this method.
        below = flat < calibrate_min
        above = flat > calibrate_max
        if below.any():
            cal_preds[below] = model_left.predict(flat[below].reshape(-1, 1))
        if above.any():
            cal_preds[above] = model_right.predict(flat[above].reshape(-1, 1))
        return cal_preds


def _prepare_series(
    target: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare target/source arrays: shape, sort by source, cast to float32."""
    if len(target) != len(source):
        raise ValueError(
            "Target and source values must have the same length. Got "
            f"{len(target)} and {len(source)}."
        )
    if len(target.shape) > 1:
        target = target.flatten()
    if len(source.shape) > 1:
        source = source.flatten()

    idx = np.argsort(source)
    target = np.array(target, dtype=np.float32)[idx]
    source = np.array(source, dtype=np.float32)[idx]

    return target, source
