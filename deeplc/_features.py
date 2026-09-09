"""Feature extraction for DeepLC."""

# TODO: Consider ProForma fixed modifications (that are not applied yet) for feature extraction.

from __future__ import annotations

import logging
import warnings
from re import sub

import numpy as np
from psm_utils import Peptidoform
from pyteomics import mass

logger = logging.getLogger(__name__)


# fmt: off
DEFAULT_POSITIONS: set[int] = {0, 1, 2, 3, -1, -2, -3, -4}
DEFAULT_POSITIONS_POS: set[int] = {0, 1, 2, 3}
DEFAULT_POSITIONS_NEG: set[int] = {-1, -2, -3, -4}
DEFAULT_DICT_AA: dict[str, int] = {
    "K": 0, "R": 1, "P": 2, "T": 3, "N": 4, "A": 5, "Q": 6, "V": 7, "S": 8, "G": 9, "I": 10,
    "L": 11, "C": 12, "M": 13, "H": 14, "F": 15, "Y": 16, "W": 17, "E": 18, "D": 19,
}
DEFAULT_DICT_INDEX_POS: dict[str, int] = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5}
DEFAULT_DICT_INDEX: dict[str, int] = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5}
# fmt: on


def encode_peptidoform(
    peptidoform: Peptidoform | str,
    add_ccs_features: bool = False,
    add_terminal_composition: bool = False,
    padding_length: int = 60,
    positions: set[int] | None = None,
    positions_pos: set[int] | None = None,
    positions_neg: set[int] | None = None,
    dict_aa: dict[str, int] | None = None,
    dict_index_pos: dict[str, int] | None = None,
    dict_index: dict[str, int] | None = None,
    legacy_positional_deltas: bool = False,
    include_rolling_sum: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extract features from a single peptidoform.

    Parameters
    ----------
    peptidoform
        The peptidoform to encode, either as a Peptidoform object or a string.
    add_ccs_features
        Whether to include CCS features. Default is False.
    add_terminal_composition
        Whether to append the N- and C-terminal group compositions to
        ``matrix_global``, adding 12 values. Default is False, which keeps
        ``matrix_global`` at its existing width so models trained against it
        remain valid. Without this, a terminal modification and a side-chain
        modification on the same residue are indistinguishable.
    padding_length
        The maximum length of the sequence after padding. Default is 60.
    include_rolling_sum
        Whether to build ``matrix_sum``, the rolling sum over pairs of positions. Models
        with a convolutional trunk ignore it; False returns an empty array in its place.
        Default is True.
    legacy_positional_deltas
        Whether to place modification deltas in the positional block the way versions
        before 4.0.1 did, which was to index ``pos_mat`` without the sorted-layout
        offset and to reach only one row. That placement is wrong, but a model trained
        against it expects it, so this reproduces it exactly for such models. Affects
        modified peptidoforms only; unmodified ones encode identically either way.
        Default is False, meaning the corrected placement.
    positions
        The positions to consider for feature extraction. Default is DEFAULT_POSITIONS.
    positions_pos
        The positive positions to consider for feature extraction. Default is
        DEFAULT_POSITIONS_POS.
    positions_neg
        The negative positions to consider for feature extraction. Default is
        DEFAULT_POSITIONS_NEG.
    dict_aa
        A dictionary mapping amino acids to indices. Default is DEFAULT_DICT_AA.
    dict_index_pos
        A dictionary mapping atoms to indices for the positional matrix. Default is
        DEFAULT_DICT_INDEX_POS.
    dict_index
        A dictionary mapping atoms to indices. Default is DEFAULT_DICT_INDEX.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary of Numpy arrays containing the extracted features.

    """
    positions = positions or DEFAULT_POSITIONS
    positions_pos = positions_pos or DEFAULT_POSITIONS_POS
    positions_neg = positions_neg or DEFAULT_POSITIONS_NEG
    dict_aa = dict_aa or DEFAULT_DICT_AA
    dict_index_pos = dict_index_pos or DEFAULT_DICT_INDEX_POS
    dict_index = dict_index or DEFAULT_DICT_INDEX

    if isinstance(peptidoform, str):
        peptidoform = Peptidoform(peptidoform)
    seq = peptidoform.sequence
    charge = peptidoform.precursor_charge
    seq, seq_len = _truncate_sequence(seq, padding_length)

    std_matrix = _fill_standard_matrix(seq, padding_length, dict_index)
    onehot_matrix = _fill_onehot_matrix(peptidoform.parsed_sequence, padding_length, dict_aa)
    pos_matrix = _fill_pos_matrix(
        seq, seq_len, positions_pos, positions_neg, dict_index, dict_index_pos
    )
    _apply_modifications(
        std_matrix,
        pos_matrix,
        peptidoform.parsed_sequence,
        seq_len,
        dict_index,
        dict_index_pos,
        positions,
        legacy_positional_deltas,
    )
    _apply_terminal_modifications(
        std_matrix,
        pos_matrix,
        peptidoform,
        seq_len,
        dict_index,
        dict_index_pos,
        positions,
        legacy_positional_deltas,
    )

    matrix_all = np.sum(std_matrix, axis=0)
    matrix_all = np.append(matrix_all, seq_len)
    if add_ccs_features:
        if not charge:
            raise ValueError(f"Peptidoform has no charge: {peptidoform}")
        matrix_all = np.append(matrix_all, (seq.count("H")) / seq_len)
        matrix_all = np.append(
            matrix_all, (seq.count("F") + seq.count("W") + seq.count("Y")) / seq_len
        )
        matrix_all = np.append(matrix_all, (seq.count("D") + seq.count("E")) / seq_len)
        matrix_all = np.append(matrix_all, (seq.count("K") + seq.count("R")) / seq_len)
        matrix_all = np.append(matrix_all, charge)

    # The fused trunk reads the per-position matrix directly and deletes this one on the
    # first line of its forward, so building it is pure cost for those models: the cumulative
    # sum and its slicing are about a tenth of the encoding work.
    matrix_sum = (
        _compute_rolling_sum(std_matrix.T, n=2)[:, ::2].T
        if include_rolling_sum
        else np.zeros((0, len(dict_index)), dtype=np.float16)
    )

    matrix_global = np.concatenate([matrix_all, pos_matrix.flatten()])
    if add_terminal_composition:
        matrix_global = np.concatenate(
            [matrix_global, _terminal_composition(peptidoform, dict_index).flatten()]
        )

    return {
        "matrix": std_matrix,
        "matrix_sum": matrix_sum,
        "matrix_global": matrix_global,
        "matrix_hc": onehot_matrix,
    }


def _truncate_sequence(seq: str, max_length: int) -> tuple[str, int]:
    """Truncate the sequence if it exceeds the max_length."""
    if len(seq) > max_length:
        warnings.warn(f"Truncating peptide (too long): {seq}", stacklevel=2)
        seq = seq[:max_length]
    return seq, len(seq)


def _fill_standard_matrix(seq: str, padding_length: int, dict_index: dict[str, int]) -> np.ndarray:
    """Fill the standard composition matrix using mass.std_aa_comp."""
    mat = np.zeros((padding_length, len(dict_index)), dtype=np.float16)
    for i, aa in enumerate(seq):
        for atom, value in mass.std_aa_comp[aa].items():
            try:
                mat[i, dict_index[atom]] = value
            except (KeyError, IndexError):
                warnings.warn(f"Skipping atom {atom} at pos {i}", stacklevel=2)
    return mat


def _fill_onehot_matrix(
    parsed_seq: list, padding_length: int, dict_aa: dict[str, int]
) -> np.ndarray:
    """Fill a one-hot matrix based on the parsed sequence tokens."""
    onehot = np.zeros((padding_length, len(dict_aa)), dtype=np.float16)
    for i, token in enumerate(parsed_seq):
        try:
            onehot[i, dict_aa[token[0]]] = 1.0
        except (KeyError, IndexError):
            warnings.warn(f"One-hot skip: {i} {token}", stacklevel=2)
    return onehot


def _fill_pos_matrix(
    seq: str,
    seq_len: int,
    positions_pos: set[int],
    positions_neg: set[int],
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
) -> np.ndarray:
    """Fill positional matrix for atoms at specific positions."""
    pos_total = positions_pos.union(positions_neg)
    pos_mat = np.zeros((max(pos_total) - min(pos_total) + 1, len(dict_index)), dtype=np.float16)
    # For positive positions
    for pos in positions_pos:
        try:
            aa = seq[pos]
        except Exception:
            warnings.warn(f"Unable to get pos {pos}", stacklevel=2)
            continue
        for atom, value in mass.std_aa_comp[aa].items():
            try:
                # shift index for matrix row since positions may be negative.
                pos_mat[pos - min(pos_total), dict_index_pos[atom]] = value
            except (KeyError, IndexError):
                warnings.warn(f"Pos matrix skip: {atom} at pos {pos}", stacklevel=2)
    # For negative positions
    for pos in positions_neg:
        try:
            aa = seq[seq_len + pos]
        except Exception:
            warnings.warn(f"Unable to get pos {pos}", stacklevel=2)
            continue
        for atom, value in mass.std_aa_comp[aa].items():
            try:
                pos_mat[pos - min(pos_total), dict_index_pos[atom]] = value
            except (KeyError, IndexError):
                warnings.warn(f"Pos matrix skip: {atom} at neg pos {pos}", stacklevel=2)
    return pos_mat


def _positional_rows(i: int, seq_len: int, positions: set[int]) -> list[int]:
    """
    Rows of the positional matrix that sequence position ``i`` occupies.

    Two corrections over indexing ``pos_mat`` by ``i`` directly:

    * ``_fill_pos_matrix`` lays rows out as ``sorted(positions)``, so row 0 is
      ``min(positions)``. Raw indexing puts an N-terminal delta in the row that
      means "fourth residue from the C-terminus".
    * In a short peptide one residue can occupy both a positive and a negative
      row, for example index 3 of a 7-mer is also -4. ``_fill_pos_matrix``
      writes the base residue to both, so a delta must reach both as well.
    """
    offset = min(positions)
    rows = []
    if i in positions:
        rows.append(i - offset)
    if (i - seq_len) in positions:
        rows.append((i - seq_len) - offset)
    return rows


def _legacy_positional_rows(i: int, seq_len: int, positions: set[int]) -> list[int]:
    """
    Positional rows as written before version 4.0.1.

    Kept because models trained against that encoding expect it. Two differences
    from :func:`_positional_rows`, both wrong and both reproduced here:

    * ``pos_mat`` is indexed by ``i`` directly, without subtracting
      ``min(positions)``. With the default sets the offset is 4, so a delta at
      sequence position 1 landed in the row meaning position -3, and a negative
      index wrapped in from the end of the block.
    * The two cases were ``if``/``elif``, so a residue occupying both a positive
      and a negative row received the delta in only one of them, while
      :func:`_fill_pos_matrix` wrote its base composition to both.
    """
    if i in positions:
        return [i]
    if (i - seq_len) in positions:
        return [i - seq_len]
    return []


def _terminal_composition(
    peptidoform: Peptidoform,
    dict_index: dict[str, int],
) -> np.ndarray:
    """
    Composition of the N- and C-terminal groups, as two stacked atom vectors.

    Terminal groups are already folded into ``matrix`` and the positional block,
    but there they are indistinguishable from a modification on the side chain
    of the first or last residue. ``[Acetyl]-PEPTIDEK`` and ``P[Acetyl]EPTIDEK``
    otherwise produce identical features, and they do not elute alike.
    """
    out = np.zeros((2, len(dict_index)), dtype=np.float16)
    for row, key in ((0, "n_term"), (1, "c_term")):
        for tag in peptidoform.properties.get(key) or []:
            try:
                composition = tag.composition
            except Exception:
                warnings.warn(f"No composition for terminal modification {tag}", stacklevel=2)
                continue
            for atom, change in composition.items():
                index = dict_index.get(atom, dict_index.get(sub(r"\[.*?\]", "", atom)))
                if index is not None:
                    out[row, index] += change
    return out


def _apply_composition_to_matrices(
    mat: np.ndarray,
    pos_mat: np.ndarray,
    composition: mass.Composition,
    i: int,
    seq_len: int,
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
    positions: set[int],
    legacy_positional_deltas: bool = False,
) -> None:
    """
    Apply a composition delta to the standard and positional matrices.

    Positional rows come from :func:`_positional_rows`, which applies the same
    offset and the same both-ends handling that :func:`_fill_pos_matrix` uses
    for base residue compositions, or from :func:`_legacy_positional_rows` when
    reproducing the pre-4.0.1 placement for a model trained against it.
    """
    rows = (
        _legacy_positional_rows(i, seq_len, positions)
        if legacy_positional_deltas
        else _positional_rows(i, seq_len, positions)
    )
    for atom_comp, change in composition.items():
        try:
            mat[i, dict_index[atom_comp]] += change
            for row in rows:
                pos_mat[row, dict_index_pos[atom_comp]] += change
        except KeyError:
            try:
                warnings.warn(f"Replacing pattern for atom: {atom_comp}", stacklevel=2)
                atom_comp_clean = sub(r"\[.*?\]", "", atom_comp)
                mat[i, dict_index[atom_comp_clean]] += change
                for row in rows:
                    pos_mat[row, dict_index_pos[atom_comp_clean]] += change
            except KeyError:
                warnings.warn(f"Ignoring atom {atom_comp} at pos {i}", stacklevel=2)
                continue
        except IndexError:
            warnings.warn(f"Index error for atom {atom_comp} at pos {i}", stacklevel=2)


def _apply_modifications(
    mat: np.ndarray,
    pos_mat: np.ndarray,
    parsed_seq: list,
    seq_len: int,
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
    positions: set[int],
    legacy_positional_deltas: bool = False,
) -> None:
    """Apply modification changes to the matrices."""
    for i, token in enumerate(parsed_seq):
        if token[1] is None:
            continue
        try:
            mod_comp = token[1][0].composition
        except Exception:
            warnings.warn(
                f"Skipping modification without known composition: {token[1]}", stacklevel=2
            )
            continue
        _apply_composition_to_matrices(
            mat,
            pos_mat,
            mod_comp,
            i,
            seq_len,
            dict_index,
            dict_index_pos,
            positions,
            legacy_positional_deltas,
        )


def _apply_terminal_modifications(
    mat: np.ndarray,
    pos_mat: np.ndarray,
    peptidoform: Peptidoform,
    seq_len: int,
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
    positions: set[int],
    legacy_positional_deltas: bool = False,
) -> None:
    """Apply N- and C-terminal modification changes to the matrices."""
    terminal_mods = [
        (0, peptidoform.properties.get("n_term")),  # N-terminus at position 0
        (seq_len - 1, peptidoform.properties.get("c_term")),  # C-terminus at last position
    ]
    for i, mods in terminal_mods:
        if not mods:
            continue
        for tag in mods:
            try:
                mod_comp = tag.composition
            except Exception:
                warnings.warn(
                    f"Skipping terminal modification without known composition: {tag}",
                    stacklevel=2,
                )
                continue
            _apply_composition_to_matrices(
                mat,
                pos_mat,
                mod_comp,
                i,
                seq_len,
                dict_index,
                dict_index_pos,
                positions,
                legacy_positional_deltas,
            )


def _compute_rolling_sum(matrix: np.ndarray, n: int = 2) -> np.ndarray:
    """Compute a rolling sum over the matrix."""
    ret = np.cumsum(matrix, axis=1, dtype=np.float32)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1 :]
