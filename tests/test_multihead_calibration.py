"""Calibrating against several LC setups at once instead of the single best one."""

from __future__ import annotations

import numpy as np
import pytest
from psm_utils import PSM, PSMList

from deeplc import core
from deeplc.calibration import (
    Calibration,
    IdentityCalibration,
    MultiHeadCalibration,
    MultiHeadPiecewiseLinearCalibration,
    MultiHeadRidgeCalibration,
    MultiHeadSplineCalibration,
    PiecewiseLinearCalibration,
    SplineTransformerCalibration,
    upgrade_calibration,
)
from deeplc.exceptions import CalibrationError

_PEPTIDES = [
    "AAGPSLSHTSGGTQSK",
    "AGFAGDDAPR",
    "AIQEYNQDK",
    "AAYFGILEK",
    "ADTQLDESSEQIDEEELTSK",
    "AHQVVEDGYEFFAK",
    "ALDQFVNFSEQK",
    "AAPFSPAEK",
    "VGAHAGEYGAEALER",
    "LNLSPLGEEMR",
]

#: (multi-head class, plain single-series class it wraps), for the shared behavioral tests below.
_SELECTOR_CLASSES = [
    (MultiHeadPiecewiseLinearCalibration, PiecewiseLinearCalibration),
    (MultiHeadSplineCalibration, SplineTransformerCalibration),
]


def _synthetic(n: int = 200, n_heads: int = 12, seed: int = 0):
    """
    Build a target that mixes two heads, which no single head can reproduce.

    Head 0 tracks the first half of the gradient and head 1 the second, both with noise; the
    remaining heads are unrelated. A single-head calibration has to pick one and lose the other,
    while a combination can use both.
    """
    rng = np.random.default_rng(seed)
    latent = rng.uniform(0, 100, n)
    target = latent
    source = rng.normal(size=(n, n_heads)) * 20 + 50
    source[:, 0] = latent + rng.normal(0, 8, n) + np.where(latent > 50, 30, 0)
    if n_heads > 1:
        source[:, 1] = latent + rng.normal(0, 8, n) - np.where(latent <= 50, 30, 0)
    return target, source


def test_multihead_calibrations_share_a_common_type():
    """`core` dispatches on this type; every head-selecting calibration must be one."""
    assert isinstance(MultiHeadRidgeCalibration(), MultiHeadCalibration)
    assert isinstance(MultiHeadPiecewiseLinearCalibration(), MultiHeadCalibration)
    assert isinstance(MultiHeadSplineCalibration(), MultiHeadCalibration)
    assert not isinstance(SplineTransformerCalibration(), MultiHeadCalibration)
    assert not isinstance(PiecewiseLinearCalibration(), MultiHeadCalibration)


def test_beats_a_single_head_when_the_target_mixes_two():
    """A combination wins when the gradient sits between two setups."""
    target, source = _synthetic()
    train, test = slice(0, 150), slice(150, None)

    multi = MultiHeadRidgeCalibration(n_heads=5)
    multi.fit(target=target[train], source=source[train])
    multi_mae = float(np.mean(np.abs(multi.transform(source[test]) - target[test])))

    single = SplineTransformerCalibration()
    best = multi.selected_model_head
    single.fit(target=target[train], source=source[train][:, best])
    single_mae = float(np.mean(np.abs(single.transform(source[test][:, best]) - target[test])))

    assert multi_mae < single_mae


def test_records_the_best_head():
    """selected_model_head still reports the top-correlating setup, for callers that ask."""
    target, source = _synthetic()
    calibration = MultiHeadRidgeCalibration(n_heads=3)
    calibration.fit(target=target, source=source)
    assert calibration.selected_model_head in (0, 1)


def test_is_fitted_and_transform_guard():
    """Transforming before fitting is an error, not silent nonsense."""
    calibration = MultiHeadRidgeCalibration()
    assert not calibration.is_fitted
    with pytest.raises(CalibrationError, match="not been fitted"):
        calibration.transform(np.zeros((3, 5)))


def test_rejects_a_model_with_fewer_heads_than_it_was_fitted_on():
    """A calibration is tied to the model it was fitted on."""
    target, source = _synthetic(n_heads=12)
    calibration = MultiHeadRidgeCalibration(n_heads=6)
    calibration.fit(target=target, source=source)
    with pytest.raises(CalibrationError, match="heads"):
        calibration.transform(source[:, :2])


def test_single_head_input_is_accepted():
    """A single-task model gives a 1-D series; the calibration degrades to one spline."""
    target, source = _synthetic(n_heads=1)
    calibration = MultiHeadRidgeCalibration()
    calibration.fit(target=target, source=source[:, 0])
    out = calibration.transform(source[:, 0])
    assert out.shape == target.shape
    assert np.isfinite(out).all()


def test_never_fits_more_weights_than_half_the_reference():
    """A ten-point reference must not be asked to support eighty weights."""
    target, source = _synthetic(n=10, n_heads=40)
    calibration = MultiHeadRidgeCalibration(n_heads=80)
    calibration.fit(target=target, source=source)
    assert len(calibration._head_calibrations) <= 5


def test_disagreement_is_per_input_and_zero_only_when_heads_agree():
    """The spread of the calibrated heads varies from input to input."""
    target, source = _synthetic(n_heads=12)
    calibration = MultiHeadRidgeCalibration(n_heads=6)
    assert calibration.disagreement(source) is None  # unfitted
    calibration.fit(target=target, source=source)

    spread = calibration.disagreement(source)
    assert spread.shape == target.shape
    assert (spread >= 0).all()
    assert np.unique(spread.round(9)).size > len(target) // 2

    # heads that are affine views of one latent retention time calibrate onto each other, so
    # after calibration they agree and the spread collapses
    rng = np.random.default_rng(0)
    latent = rng.uniform(0, 100, len(target))
    scales, shifts = rng.uniform(0.5, 2, 12), rng.uniform(-20, 20, 12)
    agreeing = latent[:, None] * scales + shifts
    agreed = MultiHeadRidgeCalibration(n_heads=6)
    agreed.fit(target=latent, source=agreeing)
    assert agreed.disagreement(agreeing).mean() < 0.05 * spread.mean()


def test_single_head_combination_reports_no_disagreement():
    """One head carries no spread, so there is nothing to scale an interval by."""
    target, source = _synthetic(n_heads=1)
    calibration = MultiHeadRidgeCalibration()
    calibration.fit(target=target, source=source[:, 0])
    assert calibration.disagreement(source[:, 0]) is None


def test_rejects_a_nonsensical_head_count():
    """Zero heads cannot calibrate anything."""
    with pytest.raises(ValueError, match="at least 1"):
        MultiHeadRidgeCalibration(n_heads=0)


def test_too_few_finite_points():
    """Two points cannot support a spline and a ridge."""
    calibration = MultiHeadRidgeCalibration()
    with pytest.raises(CalibrationError, match="three reference points"):
        calibration.fit(target=np.array([1.0, np.nan]), source=np.zeros((2, 4)))


def test_empty_source_returns_empty():
    """No PSMs in, no predictions out."""
    target, source = _synthetic()
    calibration = MultiHeadRidgeCalibration(n_heads=4)
    calibration.fit(target=target, source=source)
    assert calibration.transform(np.zeros((0, source.shape[1]))).shape == (0,)


@pytest.mark.parametrize(("selector_cls", "inner_cls"), _SELECTOR_CLASSES)
def test_selector_matches_a_manual_fit_of_its_inner_calibration(
    selector_cls: type[MultiHeadCalibration], inner_cls: type[Calibration]
):
    """The wrapper picks a head and defers to its inner calibration, nothing more."""
    target, source = _synthetic()
    selector = selector_cls()
    selector.fit(target=target, source=source)
    assert selector.selected_model_head in (0, 1)

    inner = inner_cls()
    head_column = source[:, selector.selected_model_head]
    inner.fit(target=target, source=head_column)

    # Exclude the exact extreme points: SplineTransformerCalibration switches between the spline
    # and its linear trail model right at the fitted min/max, so a sub-ULP float32 rounding
    # difference there can flip the branch and jump the output. That instability is inherent to
    # the wrapped calibration, not something the selector introduces.
    not_extreme = (head_column != head_column.min()) & (head_column != head_column.max())
    np.testing.assert_allclose(
        selector.transform(source)[not_extreme],
        inner.transform(head_column)[not_extreme],
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(("selector_cls", "_inner_cls"), _SELECTOR_CLASSES)
def test_selector_is_fitted_and_transform_guard(selector_cls, _inner_cls):
    """Transforming before fitting is an error, not silent nonsense."""
    calibration = selector_cls()
    assert not calibration.is_fitted
    with pytest.raises(CalibrationError, match="not been fitted"):
        calibration.transform(np.zeros((3, 5)))


@pytest.mark.parametrize(("selector_cls", "_inner_cls"), _SELECTOR_CLASSES)
def test_selector_rejects_a_model_with_fewer_heads_than_it_was_fitted_on(selector_cls, _inner_cls):
    """A calibration is tied to the model it was fitted on."""
    target, source = _synthetic(n_heads=12)
    calibration = selector_cls()
    calibration.fit(target=target, source=source)
    with pytest.raises(CalibrationError, match="heads"):
        # 0 columns is fewer than the fitted head's index whichever head that turned out to be.
        calibration.transform(source[:, :0])


@pytest.mark.parametrize(("selector_cls", "_inner_cls"), _SELECTOR_CLASSES)
def test_selector_single_head_input_is_accepted(selector_cls, _inner_cls):
    """A single-task model gives a 1-D series; the calibration still works."""
    target, source = _synthetic(n_heads=1)
    calibration = selector_cls()
    calibration.fit(target=target, source=source[:, 0])
    out = calibration.transform(source[:, 0])
    assert out.shape == target.shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize(("selector_cls", "_inner_cls"), _SELECTOR_CLASSES)
def test_selector_too_few_finite_points(selector_cls, _inner_cls):
    """Two points cannot support a fit."""
    calibration = selector_cls()
    with pytest.raises(CalibrationError, match="three reference points"):
        calibration.fit(target=np.array([1.0, np.nan]), source=np.zeros((2, 4)))


@pytest.mark.parametrize(("selector_cls", "_inner_cls"), _SELECTOR_CLASSES)
def test_selector_empty_source_returns_empty(selector_cls, _inner_cls):
    """No PSMs in, no predictions out."""
    target, source = _synthetic()
    calibration = selector_cls()
    calibration.fit(target=target, source=source)
    assert calibration.transform(np.zeros((0, source.shape[1]))).shape == (0,)


def _psm_list(rts: list[float] | None = None) -> PSMList:
    return PSMList(
        psm_list=[
            PSM(
                spectrum_id=str(i),
                peptidoform=f"{seq}/2",
                retention_time=None if rts is None else rts[i],
            )
            for i, seq in enumerate(_PEPTIDES)
        ]
    )


def test_core_hands_the_matrix_over_and_predicts_end_to_end():
    """
    ``calibrate`` and ``predict_and_calibrate`` accept it with the bundled multitask model.

    This is the integration path a user takes: pass the calibration in, get one prediction per
    PSM back.
    """
    reference = _psm_list([5.0 + 3.0 * i for i in range(len(_PEPTIDES))])
    calibration = core.calibrate(
        reference,
        calibration=MultiHeadRidgeCalibration(n_heads=4),
        predict_kwargs={"device": "cpu"},
    )
    assert calibration.is_fitted
    assert calibration.selected_model_head is not None

    predicted = core.predict_and_calibrate(
        _psm_list(),
        psm_list_reference=reference,
        calibration=calibration,
        predict_kwargs={"device": "cpu"},
    )
    assert predicted.shape == (len(_PEPTIDES),)
    assert np.isfinite(predicted).all()


def test_default_calibration_combines_heads_for_the_multitask_model():
    """With no calibration given, the bundled multitask model is calibrated on several heads."""
    reference = _psm_list([5.0 + 3.0 * i for i in range(len(_PEPTIDES))])
    calibration = core.calibrate(reference, predict_kwargs={"device": "cpu"})
    assert isinstance(calibration, MultiHeadRidgeCalibration)
    assert calibration.is_fitted
    assert calibration.selected_model_head is not None


def test_lighter_calibration_can_be_passed_explicitly():
    """Passing MultiHeadSplineCalibration opts out of the ridge combination."""
    reference = _psm_list([5.0 + 3.0 * i for i in range(len(_PEPTIDES))])
    calibration = core.calibrate(
        reference,
        calibration=MultiHeadSplineCalibration(),
        predict_kwargs={"device": "cpu"},
    )
    assert isinstance(calibration, MultiHeadSplineCalibration)
    assert calibration.selected_model_head is not None


def test_upgrade_calibration_returns_a_multihead_instance_unchanged():
    """A MultiHeadCalibration is not touched: nothing to upgrade."""
    calibration = MultiHeadRidgeCalibration(n_heads=3)
    assert upgrade_calibration(calibration) is calibration


def test_upgrade_calibration_wraps_a_naive_spline_calibration():
    """A naive SplineTransformerCalibration becomes a MultiHeadSplineCalibration."""
    upgraded = upgrade_calibration(SplineTransformerCalibration())
    assert isinstance(upgraded, MultiHeadSplineCalibration)
    assert not upgraded.is_fitted


def test_upgrade_calibration_carries_over_piecewise_linear_parameters():
    """The wrapped PiecewiseLinearCalibration keeps the constructor arguments it was given."""
    naive = PiecewiseLinearCalibration(number_of_splits=25, use_median=True)
    upgraded = upgrade_calibration(naive)
    assert isinstance(upgraded, MultiHeadPiecewiseLinearCalibration)
    assert upgraded._inner.number_of_splits == 25
    assert upgraded._inner.use_median is True


def test_upgrade_calibration_rejects_an_already_fitted_naive_calibration():
    """A fitted naive calibration carries no record of which head it was fit on."""
    naive = SplineTransformerCalibration()
    naive.fit(target=np.linspace(0, 10, 50), source=np.linspace(0, 10, 50))
    with pytest.raises(CalibrationError, match="fitted, naive Calibration"):
        upgrade_calibration(naive)


def test_upgrade_calibration_rejects_a_calibration_with_no_multihead_counterpart():
    """IdentityCalibration has no MultiHead* counterpart; upgrading it is a clear error."""
    with pytest.raises(ValueError, match="No MultiHeadCalibration counterpart"):
        upgrade_calibration(IdentityCalibration())


def test_upgrade_calibration_rejects_a_nonsensical_type():
    """Neither a Calibration nor a MultiHeadCalibration cannot be upgraded."""
    with pytest.raises(ValueError, match="Expected calibration to be of type"):
        upgrade_calibration(object())


def test_core_accepts_an_unfitted_naive_calibration_for_backward_compatibility():
    """`calibrate()` upgrades a plain SplineTransformerCalibration instead of rejecting it."""
    reference = _psm_list([5.0 + 3.0 * i for i in range(len(_PEPTIDES))])
    calibration = core.calibrate(
        reference,
        calibration=SplineTransformerCalibration(),
        predict_kwargs={"device": "cpu"},
    )
    assert isinstance(calibration, MultiHeadSplineCalibration)
    assert calibration.is_fitted


def test_core_rejects_a_fitted_naive_calibration():
    """predict_and_calibrate() cannot recover the head a naive calibration was fit on."""
    reference = _psm_list([5.0 + 3.0 * i for i in range(len(_PEPTIDES))])
    naive = SplineTransformerCalibration()
    naive.fit(target=np.array([5.0 + 3.0 * i for i in range(len(_PEPTIDES))]), source=np.zeros(10))
    with pytest.raises(CalibrationError, match="fitted, naive Calibration"):
        core.predict_and_calibrate(
            _psm_list(),
            psm_list_reference=reference,
            calibration=naive,
            predict_kwargs={"device": "cpu"},
        )


class _CountingSource:
    """A column source that records which heads were asked for."""

    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix
        self.requests: list[tuple[int, ...]] = []

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def ndim(self):
        return 2

    def columns(self, indices) -> np.ndarray:
        wanted = tuple(int(i) for i in indices)
        self.requests.append(wanted)
        return self._matrix[:, list(wanted)]

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("the whole matrix should not be materialised")


def test_column_source_matches_a_matrix():
    """
    A calibration must not care whether its source is a matrix or a column provider.

    That equivalence is what lets the caller hand over one source and never branch on the
    calibration, while a multitask model evaluates only the heads that get read.
    """
    rng = np.random.RandomState(0)
    source = rng.randn(200, 300) * 5 + 40
    target = source[:, 11] * 1.1 + 2 + rng.randn(200) * 0.1
    query = rng.randn(40, 300) * 5 + 40

    for calibration in (MultiHeadRidgeCalibration(n_heads=12), SplineTransformerCalibration()):
        fitted = upgrade_calibration(calibration)
        fitted.fit(target, source)
        lazy = _CountingSource(query)
        np.testing.assert_allclose(fitted.transform(lazy), fitted.transform(query), atol=1e-8)
        # every read is one request for all the heads that calibration uses
        assert len(lazy.requests) == 1
        assert len(lazy.requests[0]) == len(getattr(fitted, "_head_idx", [0]))


def test_column_source_serves_the_disagreement_too():
    """The per-peptide spread reads the same columns, so it works off a lazy source as well."""
    rng = np.random.RandomState(1)
    source = rng.randn(200, 120) * 5 + 40
    target = source[:, 3] * 0.9 + 1 + rng.randn(200) * 0.2
    query = rng.randn(30, 120) * 5 + 40

    calibration = MultiHeadRidgeCalibration(n_heads=10)
    calibration.fit(target, source)
    np.testing.assert_allclose(
        calibration.disagreement(_CountingSource(query)),
        calibration.disagreement(query),
        atol=1e-8,
    )
