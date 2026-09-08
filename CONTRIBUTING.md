# Contributing

This document briefly describes how to contribute to
[DeepLC](https://github.com/compomics/DeepLC).

## Before you begin

If you have an idea for a feature, use case to add or an approach for a bugfix,
it is best to communicate with the community by creating an issue in
[GitHub issues](https://github.com/compomics/DeepLC/issues).

## Codebase overview

DeepLC predicts peptide retention times using a 1D-CNN model trained on atomic composition
features.

- **Feature extraction** (`deeplc/_features.py`) — `encode_peptidoform()` converts a ProForma
  peptidoform string into per-position atomic composition arrays (C, H, N, O, S, P) plus a 20-AA
  one-hot encoding. Padding to 60 residues. Uses `psm_utils.Peptidoform` as the canonical peptide
  representation.
- **Dataset** (`deeplc/data.py`) — `DeepLCDataset` wraps lists of `Peptidoform` objects into a
  PyTorch `Dataset`. Features are encoded lazily in `__getitem__`. `split_datasets()` handles
  train/validation splits.
- **Model architecture** (`deeplc/_architecture.py`) — `DeepLCModel`: Conv1D branches (atomic,
  summed-atomic, global, one-hot) feed a shared dense trunk into `BatchedHeads` returning
  `[batch, n_heads]`. An optional fine-tuning adapter (`self.adapter`, attached via
  `add_adapter()`) maps head output to `[batch, 1]`.
- **Training/inference** (`deeplc/_model_ops.py`) — `load_model()`, `train()`, `predict()`,
  `evaluate()`. Checkpoints are plain state dicts loaded with `weights_only=True`.
- **Calibration** (`deeplc/calibration/` package) — `predict(..., return_matrix=True)` always
  returns a `(n, n_heads)` matrix, `n_heads=1` for a single-setup model.
  `deeplc/calibration/simple.py` holds naive, single-series calibrations (`Calibration` ABC,
  `IdentityCalibration`, `PiecewiseLinearCalibration`, `SplineTransformerCalibration`) that map
  one column onto observed RT space and know nothing about heads.
  `deeplc/calibration/multihead.py` holds `MultiHeadCalibration` ABC and the classes that pick
  their own head(s) from the matrix: `MultiHeadPiecewiseLinearCalibration` and
  `MultiHeadSplineCalibration` (rank heads by correlation, delegate to the matching `simple.py`
  class) and `MultiHeadRidgeCalibration` (combines the best-correlating heads with a ridge fit).
  `core.calibrate()`/`predict_and_calibrate()` only accept `MultiHeadCalibration` instances and
  always hand over the full matrix; the default is `MultiHeadRidgeCalibration`.
- **Reference selection** (`deeplc/_reference_selection.py`) — selects high-confidence PSMs from
  input for auto-calibration.
- **Core API** (`deeplc/core.py`) — top-level functions: `predict()`, `calibrate()`,
  `predict_and_calibrate()`, `finetune_and_predict()`.
- **CLI** (`deeplc/__main__.py`) — two subcommands, `predict` and `gui`. Reads PSM files via
  `psm_utils.io.read_file()`.
- **GUI** (`deeplc/gui.py`) — NiceGUI-based web UI, launchable as browser app or native desktop
  window via `pywebview`.

Conventions: peptide sequences use ProForma notation throughout (via `psm_utils.Peptidoform`);
PSM collections are `psm_utils.PSMList`; line length is 99 characters (ruff); Python >= 3.11 with
`from __future__ import annotations`.

## How to contribute

- Fork [DeepLC](https://github.com/compomics/DeepLC) on GitHub to
make your changes.
- Commit and push your changes to your
[fork](https://help.github.com/articles/pushing-to-a-remote/).
- Open a
[pull request](https://help.github.com/articles/creating-a-pull-request/)
with these changes. You pull request message ideally should include:
   - A description of why the changes should be made.
   - A description of the implementation of the changes.
   - A description of how to test the changes.
- The pull request should pass all the continuous integration tests which are
  automatically run by
  [GitHub Actions](https://github.com/compomics/DeepLC/actions).


## Development workflow

- When a new version is ready to be published:

    1. Change the version number in `setup.py` using
    [semantic versioning](https://semver.org/).
    2. Update the changelog (if not already done) in `CHANGELOG.md` according to
    [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
    3. Set a new tag with the version number, e.g. `git tag v0.1.5`.
    4. Push to GitHub, with the tag: `git push; git push --tags`.

- When a new tag is pushed to (or made on) GitHub that matches `v*`, the
following GitHub Actions are triggered:

    1. The Python package is build and published to PyPI.
    2. A zip archive is made of the `./deeplc_gui/` directory, excluding
    `./deeplc_gui/src` with
    [Zip Release](https://github.com/marketplace/actions/zip-release).
    3. A GitHub release is made with the zipped GUI files as assets and the new
    changes listed in `CHANGELOG.md` with
    [Git Release](https://github.com/marketplace/actions/git-release).
    4. After some time, the bioconda package should get updated automatically.
