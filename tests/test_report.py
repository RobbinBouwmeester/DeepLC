"""Prediction reports: membership, novelty and conformal intervals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from psm_utils import PSM, PSMList

from deeplc.report import (
    TrainingIndex,
    _ConformalInterval,
    canonical_peptidoform_key,
    prediction_report,
)

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
    "AAGPSLSHTSGGTQSR",
    "AGFAGDDAPK",
    "AIQEYNQDR",
    "AAYFGILER",
    "ADTQLDESSEQIDEEELTSR",
    "AHQVVEDGYEFFAR",
    "ALDQFVNFSEQR",
    "AAPFSPAER",
    "VGAHAGEYGAEALEK",
    "LNLSPLGEEMK",
]


# --------------------------------------------------------------------------- #
# canonical keys


def test_key_of_an_unmodified_peptidoform_ends_with_a_bare_pipe():
    """No modifications means an empty modification part, not a missing pipe."""
    assert canonical_peptidoform_key("PEPTIDEK/2") == "PEPTIDEK|"


def test_key_uses_unimod_accessions_and_peprec_positions():
    """1-based positions, 0 for N-terminal; names resolve to U:<id>."""
    assert canonical_peptidoform_key("PEPTM[Oxidation]IDEK/2") == "PEPTMIDEK|5|U:35"
    assert canonical_peptidoform_key("[Acetyl]-PEPTIDEK/2") == "PEPTIDEK|0|U:1"


def test_key_ignores_charge_and_sorts_modifications():
    """The corpus keys carry no charge, and modifications are position-sorted."""
    two = canonical_peptidoform_key("PEPS[Phospho]TM[Oxidation]IDEK/3")
    assert two == "PEPSTMIDEK|4|U:21|6|U:35"
    assert canonical_peptidoform_key("PEPS[Phospho]TM[Oxidation]IDEK") == two


def test_key_keeps_an_unknown_modification_as_its_lowercased_name():
    """An unmapped modification matches itself across sources instead of merging."""
    key = canonical_peptidoform_key("PEPT[Formula:C1H2O]IDEK/2")
    assert key.startswith("PEPTIDEK|4|")
    assert key == key.lower().replace("peptidek", "PEPTIDEK")


# --------------------------------------------------------------------------- #
# conformal interval


def test_interval_covers_at_nominal_rate_on_synthetic_residuals():
    """Fresh residuals from the same distribution land inside at about the nominal rate."""
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0, 100, 4000)
    residuals = rng.normal(0, 1 + predicted / 50, 4000)  # width grows along the gradient
    interval = _ConformalInterval.fit(predicted, residuals, coverage=0.90)

    new_predicted = rng.uniform(0, 100, 4000)
    new_residuals = rng.normal(0, 1 + new_predicted / 50, 4000)
    covered = np.abs(new_residuals) <= interval.widths(new_predicted)
    assert 0.87 <= covered.mean() <= 0.94


def test_interval_is_wider_where_residuals_are_wider():
    """The per-bin quantiles track a width that changes along the gradient."""
    rng = np.random.default_rng(1)
    predicted = rng.uniform(0, 100, 2000)
    residuals = rng.normal(0, np.where(predicted > 50, 5.0, 1.0), 2000)
    interval = _ConformalInterval.fit(predicted, residuals, coverage=0.90)
    assert interval.widths(np.array([90.0]))[0] > 2 * interval.widths(np.array([10.0]))[0]


def test_difficulty_score_gives_each_input_its_own_width():
    """With a per-input score, two inputs at the same predicted RT get different widths."""
    rng = np.random.default_rng(3)
    predicted = rng.uniform(0, 100, 3000)
    difficulty = rng.uniform(0.5, 4.0, 3000)
    residuals = rng.normal(0, difficulty, 3000)
    interval = _ConformalInterval.fit(predicted, residuals, 0.90, difficulty)

    widths = interval.widths(np.full(2, 50.0), np.array([0.6, 3.5]))
    assert widths[1] > 2 * widths[0]

    new_predicted = rng.uniform(0, 100, 3000)
    new_difficulty = rng.uniform(0.5, 4.0, 3000)
    covered = np.abs(rng.normal(0, new_difficulty, 3000)) <= interval.widths(
        new_predicted, new_difficulty
    )
    assert 0.87 <= covered.mean() <= 0.94


def test_difficulty_scaled_interval_needs_a_score_to_predict_with():
    """An interval fitted on a difficulty score cannot silently drop it."""
    rng = np.random.default_rng(4)
    predicted = rng.uniform(0, 100, 500)
    difficulty = rng.uniform(1, 2, 500)
    interval = _ConformalInterval.fit(predicted, rng.normal(0, 1, 500), 0.90, difficulty)
    with pytest.raises(ValueError, match="difficulty"):
        interval.widths(predicted)


def test_thin_bins_fall_back_to_the_global_quantile():
    """Too few residuals per bin means one global width, not five noisy ones."""
    rng = np.random.default_rng(2)
    predicted = rng.uniform(0, 100, 60)  # 12 per bin, below the per-bin minimum
    residuals = rng.normal(0, 2, 60)
    interval = _ConformalInterval.fit(predicted, residuals, coverage=0.90)
    assert len(set(np.round(interval.half_width, 9))) == 1


# --------------------------------------------------------------------------- #
# training index, built small and on the fly


@pytest.fixture(params=["directory", "packed"])
def tiny_index(request, tmp_path: Path) -> TrainingIndex:
    """Three peptidoforms over three setups, in both on-disk formats."""
    keys = ["AAGPSLSHTSGGTQSK|", "AGFAGDDAPR|7|U:35", "LNLSPLGEEMR|"]
    tasks = [[0], [0, 2], [1]]
    hashes = TrainingIndex._hash(keys)
    order = np.argsort(hashes)
    indptr = np.zeros(len(keys) + 1, dtype=np.int64)
    cols: list[int] = []
    for new_row, old in enumerate(order):
        cols.extend(tasks[old])
        indptr[new_row + 1] = len(cols)
    np.save(tmp_path / "key_hashes.npy", hashes[order])
    np.save(tmp_path / "task_indptr.npy", indptr)
    np.save(tmp_path / "task_cols.npy", np.array(cols, dtype=np.int16))
    sequences = sorted({k.split("|", 1)[0] for k in keys})
    (tmp_path / "sequences.txt").write_bytes("\n".join(sequences).encode("ascii"))
    np.save(tmp_path / "seq_lengths.npy", np.array([len(s) for s in sequences], dtype=np.int16))
    (tmp_path / "task_names.json").write_text(json.dumps(["setup_a", "setup_b", "setup_c"]))
    (tmp_path / "meta.json").write_text(
        json.dumps({"format_version": 1, "n_peptidoforms": 3, "n_tasks": 3, "n_observations": 4})
    )
    if request.param == "directory":
        return TrainingIndex(tmp_path)

    import zipfile

    h40 = (hashes[order] >> np.uint64(24)).astype(np.uint64)
    counts = np.bincount((h40 >> np.uint64(16)).astype(np.int64), minlength=1 << 24)
    packed = tmp_path / "tiny.dlcidx"
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_LZMA) as archive:
        archive.writestr(
            "meta.json",
            json.dumps(
                {
                    "format_version": 2,
                    "hash_bits": 40,
                    "n_tasks": 3,
                    "n_peptidoforms": 3,
                    "n_observations": 4,
                }
            ),
        )
        archive.writestr("hash_bucket_counts.u8", counts.astype(np.uint8).tobytes())
        archive.writestr(
            "hash_remainders.u16", (h40 & np.uint64(0xFFFF)).astype(np.uint16).tobytes()
        )
        archive.writestr("row_lengths.u16", np.diff(indptr).astype(np.uint16).tobytes())
        archive.writestr("task_cols.i16", np.array(cols, dtype=np.int16).tobytes())
        archive.writestr("sequences.txt", chr(10).join(sequences).encode("ascii"))
        archive.writestr("task_names.json", json.dumps(["setup_a", "setup_b", "setup_c"]))
    return TrainingIndex(packed)


def test_index_membership_and_per_task_membership(tiny_index: TrainingIndex):
    """Exact keys are found globally and within the right setups only."""
    keys = ["AAGPSLSHTSGGTQSK|", "AGFAGDDAPR|7|U:35", "AGFAGDDAPR|", "PEPTIDEK|"]
    assert tiny_index.contains(keys).tolist() == [True, True, False, False]
    in_a = tiny_index.contains_in_tasks(keys, np.array([0]))
    assert in_a.tolist() == [True, True, False, False]
    in_b = tiny_index.contains_in_tasks(keys, np.array([1]))
    assert in_b.tolist() == [False, False, False, False]


def test_index_distances_are_capped_and_exact_below_the_cap(tiny_index: TrainingIndex):
    """Distances are exact up to the cap and reported as cap + 1 beyond it."""
    distances = tiny_index.distance_to_training(
        ["AAGPSLSHTSGGTQSK", "AAGPSLSHTSGGTQSR", "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"],
        max_distance=5,
    )
    assert distances[0] == 0
    assert distances[1] == 1
    assert distances[2] == 6  # cap + 1


def test_index_refuses_a_directory_that_is_not_an_index(tmp_path: Path):
    """A random directory raises instead of pretending to be an index."""
    with pytest.raises(FileNotFoundError, match="training index"):
        TrainingIndex(tmp_path)


def test_packed_index_with_an_unknown_format_version_is_refused(tmp_path: Path):
    """A future format fails loudly instead of being misread."""
    import zipfile

    packed = tmp_path / "future.dlcidx"
    with zipfile.ZipFile(packed, "w") as archive:
        archive.writestr("meta.json", json.dumps({"format_version": 99}))
    with pytest.raises(ValueError, match="format_version"):
        TrainingIndex(packed)


# --------------------------------------------------------------------------- #
# the full report


def _reference() -> PSMList:
    return PSMList(
        psm_list=[
            PSM(spectrum_id=str(i), peptidoform=f"{seq}/2", retention_time=5.0 + 2.5 * i)
            for i, seq in enumerate(_PEPTIDES)
        ]
    )


def test_report_end_to_end_with_index(tiny_index: TrainingIndex):
    """One row per PSM with prediction, interval, membership and distances."""
    queries = PSMList(
        psm_list=[
            PSM(spectrum_id="q0", peptidoform="AAGPSLSHTSGGTQSK/2"),  # in reference and corpus
            PSM(spectrum_id="q1", peptidoform="AGFAGDDAPM[Oxidation]R/2"),
            PSM(spectrum_id="q2", peptidoform="WWWWWWWWWWWWWWWW/2"),
        ]
    )
    report = prediction_report(
        queries,
        psm_list_reference=_reference(),
        training_index=tiny_index,
        predict_kwargs={"device": "cpu"},
    )
    assert list(report.peptidoform) == [str(p.peptidoform) for p in queries.psm_list]
    assert np.isfinite(report.predicted_rt).all()
    assert (report.ci_lower <= report.predicted_rt).all()
    assert (report.ci_upper >= report.predicted_rt).all()
    assert report.attrs["coverage"] == 0.90

    assert report.in_reference.tolist() == [True, False, False]
    assert report.dist_to_reference.tolist()[0] == 0
    assert report.dist_to_reference.tolist()[2] > 5

    assert report.in_training.tolist() == [True, False, False]
    assert bool(report.in_selected_heads_training[0]) in (True, False)  # depends on the head


def test_report_without_index_has_only_reference_columns():
    """The report works with nothing but the reference; corpus columns are absent."""
    report = prediction_report(
        PSMList(psm_list=[PSM(spectrum_id="q", peptidoform="AGFAGDDAPR/2")]),
        psm_list_reference=_reference(),
        predict_kwargs={"device": "cpu"},
    )
    assert "in_training" not in report.columns
    assert report.in_reference.tolist() == [True]
    assert report.dist_to_reference.tolist() == [0]


def test_report_rejects_a_prefitted_calibration():
    """The report needs to fit per fold, so a fitted calibration cannot be reused."""
    from deeplc.calibration import SplineTransformerCalibration

    calibration = SplineTransformerCalibration()
    calibration.fit(target=np.arange(20, dtype=np.float32), source=np.arange(20, dtype=np.float32))
    with pytest.raises(ValueError, match="unfitted"):
        prediction_report(
            PSMList(psm_list=[PSM(spectrum_id="q", peptidoform="PEPTIDEK/2")]),
            psm_list_reference=_reference(),
            calibration=calibration,
            predict_kwargs={"device": "cpu"},
        )


def test_report_widths_vary_per_peptide_with_a_multihead_calibration():
    """Peptides get their own interval; per_peptide_width=False restores the RT-only widths."""
    from deeplc.calibration import MultiHeadRidgeCalibration

    queries = PSMList(
        psm_list=[
            PSM(spectrum_id=str(i), peptidoform=f"{seq}/2") for i, seq in enumerate(_PEPTIDES)
        ]
    )
    per_peptide, per_bin = (
        prediction_report(
            queries,
            psm_list_reference=_reference(),
            calibration=MultiHeadRidgeCalibration(n_heads=8),
            per_peptide_width=flag,
            predict_kwargs={"device": "cpu"},
        )
        for flag in (True, False)
    )
    widths = (per_peptide["ci_upper"] - per_peptide["ci_lower"]).round(9)
    binned_widths = (per_bin["ci_upper"] - per_bin["ci_lower"]).round(9)
    assert widths.nunique() > binned_widths.nunique()
    assert (widths > 0).all()


def test_report_falls_back_to_rt_only_widths_without_disagreement():
    """A single-head calibration has no per-peptide signal, so the flag changes nothing."""
    queries = PSMList(
        psm_list=[
            PSM(spectrum_id=str(i), peptidoform=f"{seq}/2") for i, seq in enumerate(_PEPTIDES)
        ]
    )
    report = prediction_report(
        queries,
        psm_list_reference=_reference(),
        per_peptide_width=True,
        predict_kwargs={"device": "cpu"},
    )
    widths = (report["ci_upper"] - report["ci_lower"]).round(9)
    assert widths.nunique() <= 5


def test_report_with_multihead_calibration_lists_every_selected_head(tiny_index: TrainingIndex):
    """With a multi-head calibration the membership covers every selected head."""
    from deeplc.calibration import MultiHeadRidgeCalibration

    report = prediction_report(
        PSMList(psm_list=[PSM(spectrum_id="q", peptidoform="AGFAGDDAPR/2")]),
        psm_list_reference=_reference(),
        calibration=MultiHeadRidgeCalibration(n_heads=4),
        training_index=tiny_index,
        predict_kwargs={"device": "cpu"},
    )
    assert len(report.attrs["selected_heads"]) == 4
    assert "in_selected_heads_training" in report.columns
