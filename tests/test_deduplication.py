"""Reference deduplication: one PSM per peptidoform, the first observation."""

from __future__ import annotations

import logging

import numpy as np
from psm_utils import PSM, PSMList

from deeplc import core
from deeplc._reference_selection import deduplicate_psms
from deeplc.calibration import MultiHeadSplineCalibration

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


def _psms(pairs: list[tuple[str, float | None]], charge: int = 2) -> PSMList:
    """Build a PSMList from (peptide, retention time) pairs."""
    return PSMList(
        psm_list=[
            PSM(spectrum_id=str(i), peptidoform=f"{seq}/{charge}", retention_time=rt)
            for i, (seq, rt) in enumerate(pairs)
        ]
    )


def test_keeps_the_first_observation_of_each_peptidoform():
    """Repeats are dropped and the retention time of the first one survives."""
    psm_list = _psms([("PEPTIDEK", 10.0), ("PEPTIDEK", 40.0), ("ACDEFGHIK", 20.0)])

    deduplicated = deduplicate_psms(psm_list)

    assert [str(p.peptidoform) for p in deduplicated] == ["PEPTIDEK/2", "ACDEFGHIK/2"]
    assert [p.retention_time for p in deduplicated] == [10.0, 20.0]


def test_order_is_preserved():
    """The kept PSMs stay in the order they were given in."""
    psm_list = _psms([(s, float(i)) for i, s in enumerate(_PEPTIDES)])
    assert [str(p.peptidoform) for p in deduplicate_psms(psm_list)] == [
        str(p.peptidoform) for p in psm_list
    ]


def test_idempotent():
    """Deduplicating an already deduplicated list changes nothing."""
    psm_list = _psms([("PEPTIDEK", 10.0), ("PEPTIDEK", 40.0), ("ACDEFGHIK", 20.0)])
    once = deduplicate_psms(psm_list)
    assert len(deduplicate_psms(once)) == len(once)


def test_charge_states_are_duplicates_by_default():
    """Retention time does not depend on charge, so charge states are repeats."""
    psm_list = PSMList(
        psm_list=[
            PSM(spectrum_id="1", peptidoform="PEPTIDEK/2", retention_time=10.0),
            PSM(spectrum_id="2", peptidoform="PEPTIDEK/3", retention_time=11.0),
        ]
    )

    assert len(deduplicate_psms(psm_list)) == 1
    assert len(deduplicate_psms(psm_list, ignore_charge=False)) == 2


def test_modified_peptidoforms_are_not_duplicates():
    """A modification makes a different peptidoform, which elutes at a different time."""
    psm_list = PSMList(
        psm_list=[
            PSM(spectrum_id="1", peptidoform="PEPTM[Oxidation]IDEK/2", retention_time=10.0),
            PSM(spectrum_id="2", peptidoform="PEPTMIDEK/2", retention_time=12.0),
        ]
    )
    assert len(deduplicate_psms(psm_list)) == 2


def test_exotic_peptidoforms_keep_their_identity():
    """
    The key is ``Peptidoform.modified_sequence``, so nothing but charge is ignored.

    Terminal, global and labile modifications all distinguish two peptidoforms, and a
    modification label may itself contain a slash, which is why the charge is not cut off the
    ProForma string by hand.
    """
    distinct = [
        "PEPTIDEK/2",
        "[Acetyl]-PEPTIDEK/2",
        "PEPTIDEK-[Amidated]/2",
        "PEPTM[Oxidation]IDEK/2",
        "PEPT[Phospho]IDEK/2",
        "<[Carbamidomethyl]@C>PEPCTIDEK/2",
        "PROT[Phospho|+79.966]EIN/2",
    ]
    psm_list = PSMList(
        psm_list=[
            PSM(spectrum_id=str(i), peptidoform=pf, retention_time=10.0 + i)
            for i, pf in enumerate(distinct)
        ]
    )

    assert len(deduplicate_psms(psm_list)) == len(distinct)


def test_charge_adducts_do_not_create_a_second_peptidoform():
    """``/2`` and ``/2[+2H]`` are the same peptidoform measured twice."""
    psm_list = PSMList(
        psm_list=[
            PSM(spectrum_id="1", peptidoform="PEPTIDEK/2", retention_time=10.0),
            PSM(spectrum_id="2", peptidoform="PEPTIDEK/2[+2H]", retention_time=11.0),
        ]
    )
    assert len(deduplicate_psms(psm_list)) == 1


def test_warns_when_repeats_disagree_on_the_retention_time(caplog):
    """
    A disagreement of the order of the gradient is a data problem, not jitter.

    Deduplication silently fixes the fit, so the case a user needs to hear about is when the
    repeated peptidoform was not the same elution event at all.
    """
    pairs = [(s, float(i)) for i, s in enumerate(_PEPTIDES)]
    pairs.append((_PEPTIDES[0], 500.0))  # same peptidoform, 500 minutes later

    with caplog.at_level(logging.WARNING, logger="deeplc._reference_selection"):
        deduplicate_psms(_psms(pairs))

    assert any("disagreed on the observed retention time" in r.message for r in caplog.records)


def test_missing_retention_times_do_not_break_the_report(caplog):
    """PSMs without an observed retention time are deduplicated like any other."""
    psm_list = _psms([("PEPTIDEK", None), ("PEPTIDEK", None), ("ACDEFGHIK", 20.0)])
    with caplog.at_level(logging.INFO, logger="deeplc._reference_selection"):
        assert len(deduplicate_psms(psm_list)) == 2


def _reference_with_duplicates() -> PSMList:
    """Ten peptidoforms on a clean gradient, each repeated once at a wrong retention time."""
    pairs = [(s, 5.0 + 3.0 * i) for i, s in enumerate(_PEPTIDES)]
    pairs += [(s, 100.0 - 2.0 * i) for i, s in enumerate(_PEPTIDES)]
    return _psms(pairs)


def test_calibrate_always_uses_the_first_observations():
    """
    ``calibrate`` fits one point per peptidoform; there is no switch to turn that off.

    The reference holds each peptidoform twice: once on a clean 5 to 32 minute gradient and
    once at a contradictory 82 to 100 minutes. The fit must reproduce the clean gradient.
    """
    reference = _reference_with_duplicates()
    targets = _psms([(s, None) for s in _PEPTIDES])

    calibration = core.calibrate(reference, predict_kwargs={"device": "cpu"})
    predicted = core.predict(targets, return_matrix=True)
    calibrated = calibration.transform(predicted)

    assert np.isfinite(calibrated).all()
    clean_low, clean_high = 5.0, 5.0 + 3.0 * (len(_PEPTIDES) - 1)
    mean = float(calibrated.mean())
    assert clean_low <= mean <= clean_high, f"fit at {mean:.1f} left the clean gradient"


def test_a_prefitted_calibration_is_the_way_to_keep_the_repeats():
    """
    The escape hatch for the rare caller who wants every reference PSM to count.

    ``calibrate`` deduplicates unconditionally, so a caller who wants the repeats weighed fits
    a ``MultiHeadCalibration`` on its own targets and passes it in; ``predict_and_calibrate`` then
    uses it as given instead of fitting one.
    """
    reference = _reference_with_duplicates()
    psm_list = _psms([(s, None) for s in _PEPTIDES])

    source = core.predict(reference, predict_kwargs={"device": "cpu"}, return_matrix=True)
    own = MultiHeadSplineCalibration()
    own.fit(target=np.array(reference["retention_time"], dtype=np.float32), source=source)

    kept = core.predict_and_calibrate(
        psm_list, psm_list_reference=reference, calibration=own, predict_kwargs={"device": "cpu"}
    )
    deduplicated = core.predict_and_calibrate(
        psm_list, psm_list_reference=reference, predict_kwargs={"device": "cpu"}
    )

    assert kept.shape == deduplicated.shape == (len(_PEPTIDES),)
    assert not np.allclose(kept, deduplicated)


def test_predict_and_calibrate_runs_on_a_duplicated_reference():
    """One prediction per input PSM, in the input order, whatever the reference looks like."""
    psm_list = _psms([(s, None) for s in _PEPTIDES])
    result = core.predict_and_calibrate(
        psm_list,
        psm_list_reference=_reference_with_duplicates(),
        predict_kwargs={"device": "cpu"},
    )
    assert result.shape == (len(_PEPTIDES),)
    assert np.isfinite(result).all()
