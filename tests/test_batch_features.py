"""
The batched encoder must agree with the per-peptide one, value for value and dtype for dtype.

:func:`deeplc._features.encode_peptidoform` is the definition of a feature; everything here
compares the batched route against it. Two of these cases were divergences found that way
during development rather than hypotheticals: selenocysteine, whose atoms the reference reads
from pyteomics even though the one-hot block has no slot for it, and peptides shorter than
four residues, where the reference's negative positional indices wrap around the sequence.
Both now take the per-peptide fallback, which is why they agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from psm_utils import Peptidoform

from deeplc._batch_features import MIN_LENGTH, encode_batch_features, supports
from deeplc._features import encode_peptidoform
from deeplc.data import DeepLCDataset

FEATURES = ("matrix", "matrix_global", "matrix_hc")
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
MODIFICATIONS = ("", "[UNIMOD:4]", "[UNIMOD:35]", "[UNIMOD:1]", "[UNIMOD:21]")


def assert_agrees(peptidoforms, padding_length=20, terminal=True, legacy=True):
    """Compare the batched route against stacking the per-peptide encoder."""
    parsed = [Peptidoform(p) if isinstance(p, str) else p for p in peptidoforms]
    one_by_one = [
        encode_peptidoform(
            p,
            add_terminal_composition=terminal,
            padding_length=padding_length,
            legacy_positional_deltas=legacy,
            include_rolling_sum=False,
        )
        for p in parsed
    ]
    batched = encode_batch_features(
        parsed,
        padding_length=padding_length,
        add_terminal_composition=terminal,
        legacy_positional_deltas=legacy,
    )
    for key in FEATURES:
        reference = np.stack([features[key] for features in one_by_one])
        assert batched[key].shape == reference.shape, key
        assert batched[key].dtype == reference.dtype, key
        assert np.array_equal(batched[key], reference), key


def random_peptidoforms(count, seed, min_length=4, max_length=18):
    """A spread of peptides with modifications at a rate like a real peptide list."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(count):
        length = int(rng.randint(min_length, max_length + 1))
        residues = list(rng.choice(list(RESIDUES), size=length))
        if rng.rand() < 0.3:
            site = int(rng.randint(0, length))
            residues[site] = residues[site] + str(rng.choice(MODIFICATIONS[1:]))
        sequence = "".join(residues)
        if rng.rand() < 0.1:
            sequence = "[UNIMOD:1]-" + sequence
        if rng.rand() < 0.05:
            sequence = sequence + "-[UNIMOD:2]"
        out.append(Peptidoform(f"{sequence}/2"))
    return out


def test_agrees_on_a_random_batch():
    """The everyday case: a few hundred peptides, a third of them modified."""
    assert_agrees(random_peptidoforms(400, seed=0))


@pytest.mark.parametrize("padding_length", [8, 20, 30, 60])
def test_agrees_at_every_window(padding_length):
    """Windows vary per batch because prediction is length-bucketed."""
    assert_agrees(random_peptidoforms(120, seed=1), padding_length=padding_length)


def test_agrees_when_peptides_are_truncated():
    """A peptide longer than the window is truncated, modifications included."""
    assert_agrees(
        ["A" * 30 + "/2", "A" * 24 + "C[UNIMOD:4]" + "AAAAA/2", "PEPTIDEK/2"],
        padding_length=20,
    )


@pytest.mark.parametrize(
    "sequence",
    [
        "PEPTIDEK/2",
        "C[UNIMOD:4]EPTIDEC[UNIMOD:4]/2",
        "PEPTC[UNIMOD:4]IDEM[UNIMOD:35]K/2",
        "[UNIMOD:1]-PEPTIDEK/2",
        "PEPTIDEK-[UNIMOD:2]/2",
        "[UNIMOD:1]-PEPTIDEK-[UNIMOD:2]/2",
        "PEPUIDEK/2",
        "PEPOIDEK/2",
        "AAAA/2",
    ],
    ids=[
        "plain",
        "modified first and last",
        "two modifications",
        "n-terminal",
        "c-terminal",
        "both termini",
        "selenocysteine",
        "pyrrolysine",
        "shortest on the fast path",
    ],
)
def test_agrees_on_one_peptide(sequence):
    """Each of these is a case the batched route has to hand over or handle exactly."""
    assert_agrees([sequence])


@pytest.mark.parametrize("length", [1, 2, 3])
def test_agrees_below_the_fast_path_minimum(length):
    """
    Short peptides go to the per-peptide encoder, so its wrap-around behaviour is preserved.

    With one residue the reference reads ``seq[seq_len - 2]``, which Python resolves to the
    last residue rather than to nothing; the batched route would have skipped it.
    """
    assert length < MIN_LENGTH
    assert_agrees(["A" * length + "/2", "PEPTIDEK/2"])


def test_agrees_without_terminal_composition_and_without_legacy_deltas():
    """Both feature-layout switches the route supports."""
    batch = random_peptidoforms(60, seed=2)
    assert_agrees(batch, terminal=False)
    assert_agrees(batch, legacy=False)


def test_unknown_residues_raise_the_same_way():
    """An ambiguous residue has no composition, and both routes say so identically."""
    for sequence in ("PEPBIDEK/2", "PEPZIDEK/2", "PEPXIDEK/2"):
        with pytest.raises(KeyError):
            encode_peptidoform(
                Peptidoform(sequence), add_terminal_composition=True, padding_length=20
            )
        with pytest.raises(KeyError):
            encode_batch_features([Peptidoform(sequence)], padding_length=20)


def test_layouts_the_route_does_not_cover_are_declined():
    """The rolling sum and the collision cross section extras keep the per-peptide path."""
    assert supports(add_ccs_features=False, include_rolling_sum=False)
    assert not supports(add_ccs_features=True, include_rolling_sum=False)
    assert not supports(add_ccs_features=False, include_rolling_sum=True)


def test_dataset_batch_matches_item_by_item_either_way():
    """
    Through the dataset, the switch must not change what comes out.

    ``encode_batch`` picks the route; ``__getitem__`` always uses the per-peptide encoder, so
    comparing the two covers the wiring as well as the encoder.
    """
    peptides = [str(p) for p in random_peptidoforms(50, seed=3)]
    for vectorised in (True, False):
        dataset = DeepLCDataset(
            peptides,
            add_terminal_composition=True,
            padding_length=20,
            include_rolling_sum=False,
            vectorised_encoding=vectorised,
        )
        batch = dataset.encode_batch(range(len(peptides)))
        for position in range(4):
            stacked = torch.stack([dataset[i][0][position] for i in range(len(peptides))])
            assert torch.equal(batch[position], stacked), (vectorised, position)


def test_dataset_keeps_the_per_peptide_path_for_the_rolling_sum():
    """A model that reads the rolling sum still gets it, batched the old way."""
    peptides = [str(p) for p in random_peptidoforms(20, seed=4)]
    dataset = DeepLCDataset(
        peptides, add_terminal_composition=True, padding_length=20, include_rolling_sum=True
    )
    batch = dataset.encode_batch(range(len(peptides)))
    assert batch[1].shape[1] > 0
    stacked = torch.stack([dataset[i][0][1] for i in range(len(peptides))])
    assert torch.equal(batch[1], stacked)


def test_a_label_beyond_the_window_warns_instead_of_raising():
    """
    An isotope-labelled modification past the padding window must not raise.

    The unlabelled path already warned and carried on; the branch that strips isotope
    brackets, so ``C[13]`` and ``N[15]`` as TMT and SILAC labels carry, did not, and raised
    IndexError from inside the encoder instead. Both routes go through the same helper, so
    one test covers both.
    """
    sequence = "A" * 24 + "K[UNIMOD:737]" + "AAAAA/2"  # TMT6plex, which carries 13C
    with pytest.warns(UserWarning):
        reference = encode_peptidoform(
            Peptidoform(sequence), add_terminal_composition=True, padding_length=20,
            legacy_positional_deltas=True, include_rolling_sum=False,
        )
    assert np.isfinite(reference["matrix"]).all()
    assert_agrees([sequence], padding_length=20)


def test_a_ccs_dataset_keeps_the_per_peptide_path():
    """
    The collision cross section layout must never reach the batched route.

    IM2Deep holds a CCS model trained against the pre-4.0.1 encoding and reaches DeepLC
    only through ``DeepLCDataset.from_psm_list(psm_list, add_ccs_features=True)``, so this
    is the call that has to keep the per-peptide encoder. Two things decline it - the CCS
    extras and the rolling sum, which ``from_psm_list`` leaves on - and this asserts the
    outcome rather than either reason, so removing one of them still fails here.
    """
    from psm_utils import PSM, PSMList

    import deeplc.data as data_module

    peptidoforms = ["AC[UNIMOD:4]DEK/2", "[UNIMOD:737]-PEPTIDEK/2", "PEPTM[UNIMOD:35]IDEKR/3"]
    psm_list = PSMList(
        psm_list=[
            PSM(peptidoform=Peptidoform(p), spectrum_id=str(i), retention_time=float(i))
            for i, p in enumerate(peptidoforms)
        ]
    )
    dataset = DeepLCDataset.from_psm_list(psm_list, add_ccs_features=True)
    assert dataset.vectorised_encoding, "the switch is on; the layout is what declines it"

    calls = []
    original = data_module.encode_batch_features
    data_module.encode_batch_features = lambda *args, **kwargs: calls.append(1)
    try:
        batch = dataset.encode_batch(range(len(peptidoforms)))
    finally:
        data_module.encode_batch_features = original

    assert not calls, "a CCS dataset must not enter the batched encoder"
    for position in range(4):
        stacked = torch.stack([dataset[i][0][position] for i in range(len(peptidoforms))])
        assert torch.equal(batch[position], stacked), position
