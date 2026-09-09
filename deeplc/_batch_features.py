"""
Batched feature encoding: the residue-only base for a whole batch in one pass.

:func:`deeplc._features.encode_peptidoform` stays the definition of what a feature is. What
this module adds is a faster route to the same numbers for a batch of peptidoforms, and it
only accelerates the part that depends on residue identity alone:

* the per-position atom composition matrix and the one-hot residue matrix are a table gather
  and a scatter over the batch;
* the positional atom block reads the first few and last few residues of each peptide, which
  is the same gather with different rows;
* the global vector is a sum along the batch's position axis.

Modifications are not reimplemented here. They occur on a quarter of a typical peptide list,
and their placement has a legacy quirk that a shipped model was trained against, so they are
applied by calling :func:`deeplc._features._apply_modifications` and its terminal counterpart
on **views into the batch arrays**. The exactness therefore comes from reusing that code
rather than from matching it.

Anything the batched route is not verified for falls back to
:func:`~deeplc._features.encode_peptidoform` per peptide, which keeps its warnings and its
exceptions rather than approximating them: peptides shorter than four residues, whose
negative positional indices wrap around the sequence in the reference encoder, and peptides
carrying a residue the one-hot block has no slot for, such as selenocysteine. Feature layouts
the route does not cover at all are rejected by :func:`supports`, and the caller keeps the
per-peptide path for those.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from psm_utils import Peptidoform
from pyteomics import mass

from deeplc._features import (
    DEFAULT_DICT_AA,
    DEFAULT_DICT_INDEX,
    DEFAULT_DICT_INDEX_POS,
    DEFAULT_POSITIONS,
    DEFAULT_POSITIONS_NEG,
    DEFAULT_POSITIONS_POS,
    _apply_modifications,
    _apply_terminal_modifications,
    _terminal_composition,
    encode_peptidoform,
)

#: Residues the one-hot block has a slot for, in its own order.
RESIDUES: tuple[str, ...] = tuple(sorted(DEFAULT_DICT_AA, key=lambda key: DEFAULT_DICT_AA[key]))
CODE: dict[str, int] = {residue: index for index, residue in enumerate(RESIDUES)}
#: Row reserved for padding in the gather tables, holding zeros.
PAD = len(RESIDUES)
N_ATOMS = len(DEFAULT_DICT_INDEX)
_POSITIONS = DEFAULT_POSITIONS_POS | DEFAULT_POSITIONS_NEG
POS_ROWS = max(_POSITIONS) - min(_POSITIONS) + 1
POS_OFFSET = min(_POSITIONS)

#: Shortest peptide the batched route takes. Below four residues the reference encoder's
#: negative positional indices wrap around the sequence - with a single residue,
#: ``seq[seq_len - 2]`` is ``seq[-1]``, that same residue - and such peptides are rare enough
#: that reproducing the wrap in vectorised form is not worth the risk of getting it wrong.
MIN_LENGTH = 4


def _gather_table(atom_index: dict[str, int]) -> np.ndarray:
    """Atom counts per residue under one atom ordering, with a zero row for padding."""
    table = np.zeros((PAD + 1, N_ATOMS), dtype=np.float16)
    for residue, row in CODE.items():
        for atom, count in mass.std_aa_comp[residue].items():
            column = atom_index.get(atom)
            if column is not None:
                table[row, column] = count
    return table


COMPOSITION = _gather_table(DEFAULT_DICT_INDEX)
POSITIONAL = _gather_table(DEFAULT_DICT_INDEX_POS)
#: An unmodified peptidoform contributes nothing to the terminal block: the reference reads
#: the ``n_term`` and ``c_term`` properties, which are empty, and returns zeros. The backbone
#: termini are already inside the residue compositions.
TERMINI = np.zeros((2, N_ATOMS), dtype=np.float16)


def supports(add_ccs_features: bool, include_rolling_sum: bool) -> bool:
    """
    Whether the batched route covers a feature layout.

    It does not build the rolling-sum matrix, which the four-branch model reads, and it does
    not build the collision cross section extras, which need the precursor charge and a
    handful of residue fractions. A caller asking for either keeps the per-peptide path.
    """
    return not add_ccs_features and not include_rolling_sum


def _as_peptidoform(peptidoform: Peptidoform | str) -> Peptidoform:
    """Parse a ProForma string if that is what a dataset holds, else pass it through."""
    return peptidoform if hasattr(peptidoform, "sequence") else Peptidoform(str(peptidoform))


def encode_batch_features(
    peptidoforms: Sequence[Peptidoform | str],
    padding_length: int = 60,
    add_terminal_composition: bool = True,
    legacy_positional_deltas: bool = True,
) -> dict[str, np.ndarray]:
    """
    Encode a batch of peptidoforms, gathering the base and patching the modified ones.

    Parameters
    ----------
    peptidoforms
        The peptidoforms to encode, parsed or as ProForma strings.
    padding_length
        Window the per-position matrices are padded or truncated to.
    add_terminal_composition
        Whether to append the N- and C-terminal group compositions to the global vector.
    legacy_positional_deltas
        Whether modification deltas go into the positional block the way versions before
        4.0.1 placed them, which every released model was trained against.

    Returns
    -------
    dict of str to numpy.ndarray
        ``matrix``, ``matrix_global`` and ``matrix_hc``, batched along the first axis and
        equal value for value, dtype included, to stacking what
        :func:`~deeplc._features.encode_peptidoform` returns for each peptidoform.

    """
    parsed = [_as_peptidoform(p) for p in peptidoforms]
    n = len(parsed)
    sequences = [p.sequence for p in parsed]
    lengths = np.fromiter(
        (min(len(sequence), padding_length) for sequence in sequences), dtype=np.int64, count=n
    )
    reference_rows = {
        row
        for row, sequence in enumerate(sequences)
        if len(sequence) < MIN_LENGTH or any(residue not in CODE for residue in sequence)
    }

    residues = np.full((n, padding_length), PAD, dtype=np.int64)
    for row, (sequence, length) in enumerate(zip(sequences, lengths, strict=True)):
        if row not in reference_rows:
            residues[row, :length] = [CODE[residue] for residue in sequence[:length]]

    matrix = COMPOSITION[residues]
    one_hot = np.zeros((n, padding_length, len(DEFAULT_DICT_AA)), dtype=np.float16)
    rows, columns = np.nonzero(residues != PAD)
    one_hot[rows, columns, residues[rows, columns]] = 1.0

    positional = np.zeros((n, POS_ROWS, N_ATOMS), dtype=np.float16)
    for position in sorted(DEFAULT_POSITIONS_POS):
        taken = lengths > position
        positional[taken, position - POS_OFFSET] = POSITIONAL[residues[taken, position]]
    for position in sorted(DEFAULT_POSITIONS_NEG):
        taken = lengths + position >= 0
        positional[taken, position - POS_OFFSET] = POSITIONAL[
            residues[np.nonzero(taken)[0], (lengths + position)[taken]]
        ]

    terminal = np.repeat(TERMINI[None, :, :], n, axis=0) if add_terminal_composition else None

    for row, peptidoform in enumerate(parsed):
        if row in reference_rows:
            continue
        tokens = peptidoform.parsed_sequence
        if any(token[1] is not None for token in tokens):
            _apply_modifications(
                matrix[row],
                positional[row],
                tokens,
                int(lengths[row]),
                DEFAULT_DICT_INDEX,
                DEFAULT_DICT_INDEX_POS,
                DEFAULT_POSITIONS,
                legacy_positional_deltas,
            )
        properties = peptidoform.properties
        if properties.get("n_term") or properties.get("c_term"):
            _apply_terminal_modifications(
                matrix[row],
                positional[row],
                peptidoform,
                int(lengths[row]),
                DEFAULT_DICT_INDEX,
                DEFAULT_DICT_INDEX_POS,
                DEFAULT_POSITIONS,
                legacy_positional_deltas,
            )
            if terminal is not None:
                terminal[row] = _terminal_composition(peptidoform, DEFAULT_DICT_INDEX)

    # float64, because the reference promotes when it appends the integer length to a float16
    # sum. The dataset casts to float32 afterwards either way, but the dtype is observable.
    blocks = [
        matrix.sum(axis=1).astype(np.float64),
        lengths[:, None].astype(np.float64),
        positional.reshape(n, -1).astype(np.float64),
    ]
    if terminal is not None:
        blocks.append(terminal.reshape(n, -1).astype(np.float64))
    matrix_global = np.concatenate(blocks, axis=1)

    for row in sorted(reference_rows):
        features = encode_peptidoform(
            parsed[row],
            add_terminal_composition=add_terminal_composition,
            padding_length=padding_length,
            legacy_positional_deltas=legacy_positional_deltas,
            include_rolling_sum=False,
        )
        matrix[row] = features["matrix"]
        one_hot[row] = features["matrix_hc"]
        matrix_global[row] = features["matrix_global"]

    return {"matrix": matrix, "matrix_global": matrix_global, "matrix_hc": one_hot}
