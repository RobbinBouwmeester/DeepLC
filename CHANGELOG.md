# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.3.0] - 2026-09-02

### Added

- `deeplc.calibration` is now a package. `deeplc.calibration.simple` holds the naive,
  single-series calibrations unchanged (`Calibration` ABC, `IdentityCalibration`,
  `PiecewiseLinearCalibration`, `SplineTransformerCalibration`). New
  `deeplc.calibration.multihead` holds `MultiHeadCalibration`, the ABC for a calibration that
  takes the whole `(n, n_heads)` prediction matrix `predict(..., return_matrix=True)` returns and
  selects its own head(s): `MultiHeadPiecewiseLinearCalibration` and `MultiHeadSplineCalibration`
  rank heads by Pearson correlation and delegate to the matching naive class on the winner, and
  `MultiHeadRidgeCalibration` combines several best-correlating heads with a ridge fit. All three
  are re-exported from `deeplc.calibration`, unchanged import path for existing names.

  On the eight PRIDE setups that no DeepLC model was trained on, `MultiHeadRidgeCalibration`
  lowered the held-out error on all eight, by a median of 13 % relative to the observed gradient.

- `deeplc.calibration.upgrade_calibration`, wrapping an unfitted naive `Calibration` in its
  `MultiHead*Calibration` counterpart. `calibrate()` and `predict_and_calibrate()` call it on
  whatever `calibration` they are given, so passing a naive instance (e.g.
  `SplineTransformerCalibration()`) still works, calibrated on the best-correlating head instead
  of head 0. A fitted naive instance is rejected: it carries no record of which head it was fit
  on, so fit a `MultiHead*Calibration` instead in that case.

### Changed

- `calibrate()` and `predict_and_calibrate()` default to `MultiHeadRidgeCalibration()` for every
  model, single-task included (it reduces to a spline plus a linear rescaling on one column). Pass
  `MultiHeadSplineCalibration()` or `MultiHeadPiecewiseLinearCalibration()` for a lighter
  calibration. `Calibration.selected_model_head`/`uses_all_heads` are gone: head selection now
  lives entirely in the `MultiHeadCalibration` implementations.

  **Breaking**: an already-fitted naive `Calibration` passed directly to
  `predict_and_calibrate()` is now rejected; fit a `MultiHead*Calibration` instead.

### Fixed

- `IdentityCalibration()` could not be instantiated: it never overrode the abstract
  `Calibration.__init__`.

## [4.2.0] - 2026-08-28

### Changed

- Calibration and fine-tuning now use **one PSM per peptidoform**, the first observation in
  the reference. A reference built from a search result repeats a peptidoform once per
  spectrum it was identified in, each time with a different observed retention time, so the
  fit was given conflicting targets and weighed peptidoforms by how often they happened to be
  identified. On a reported MS2Rescore case (201,593 PSMs, one run) the reference selected by
  auto-calibration held 6,331 PSMs but only 2,623 peptidoforms; the repeats of one peptidoform
  disagreed on the observed retention time by 236 s, two thirds of the 354 s range covered by
  the identifications. Fitting on the first observations improved the calibration on those
  2,623 peptidoforms from 9.69 to 4.51 s mean absolute error (median 5.98 to 2.09 s; within
  5 s 44.9 % to 81.4 %). Retention times are in whatever unit the input uses; DeepLC does not
  convert them.

  Charge states of one peptidoform count as repeats, since retention time does not depend on
  precursor charge. When repeats disagree by a large fraction of the observed range, DeepLC
  now says so: that means the reference mixes runs or contains low-confidence PSMs, which
  deduplication hides rather than fixes.

  There is no option for it: the public functions keep the signature they had. A caller who
  really wants every reference PSM to count fits a `Calibration` on its own targets and passes
  it to `predict_and_calibrate`, which uses an already fitted calibration as given; `train`
  remains available for full control over a training set.

### Added

- `deeplc._reference_selection.deduplicate_psms`, which returns the first PSM of every
  peptidoform in a `PSMList`.

## [4.1.1] - 2026-08-26

### Changed

- The default model is now the fused-trunk multitask model trained across 6,543 LC
  setups (`multitask_flexcnn_model.pt`, bundled since 4.1.0 as an opt-in). Every core
  function and the command line use it when no `model` is given. The 4.0 default,
  `multitask_model.pt`, stays bundled as `deeplc.core.LEGACY_MULTITASK_MODEL`; pass it
  as `model=` to reproduce 4.0 and 4.1.0 predictions exactly.
- Uncalibrated `predict()` on a multitask model that carries setup names reports the
  setup named by `deeplc.core.DEFAULT_TASK_NAME` (`PXD005573_mcp`, the 200-minute
  gradient the DeepLC 1.x to 3.x models were trained on) instead of head 0, which for
  the new default was an arbitrary setup. `return_matrix=True` is unchanged, and so are
  calibration and fine-tuning, which select or fit the setup from the reference.

## [4.1.0] - 2026-08-20

### Added

- `legacy_positional_deltas`, on `encode_peptidoform` and `DeepLCDataset`, reproducing
  the placement of modification deltas in the positional block exactly as versions
  before 4.0.1 did. Verified bit-identical to a v4.0.0 checkout across 4,760 feature
  arrays from 1,190 peptidoforms covering lengths 2 to 70, every modified position, and
  terminal modifications. Unmodified peptidoforms are unaffected either way.
- The feature layout a model expects is now resolved from the specification it carries,
  in one place. A checkpoint that records no specification was written before 4.1.0, so
  it also predates the 4.0.1 correction and is fed the encoding it was trained on. One
  that records a specification is read literally, and every checkpoint this version
  writes records the encoding it used. `DeepLCDataset` therefore defaults to the
  pre-4.0.1 placement, since a dataset exists to feed a model; `encode_peptidoform`,
  whose job is correct featurisation, still defaults to the corrected placement.
  Newly trained models get the corrected placement.

### Fixed

- Predictions from models trained before the 4.0.1 feature correction no longer change.
  4.0.1 corrected where modification deltas land in the positional block, which altered
  the encoding of every modified peptidoform, while every model released up to that
  point had been trained on the old encoding. Modified peptides were therefore predicted
  from input those models had never seen. All five bundled checkpoints are bare state
  dicts and are now recognised as predating the correction, so their predictions match
  v4.0.0 exactly again, with no retraining and no change to any calling code.

  This also covers models held by downstream packages. IM2Deep 2.0.2, unmodified,
  reproduces its v4.0.0 CCS predictions exactly on this release; under 4.0.1 its modified
  peptides had shifted by up to 7.6 A^2. Note that this means predictions for such models
  differ from 4.0.1, which is the point: 4.0.1's change to them was not intended.

- Fine-tuning onto a new LC setup for models with a low-rank multitask head, fitting the
  setup's own `rank + 2` parameters with the encoder and pretrained setups frozen: 66
  values at rank 64, against roughly 1.7 million for an adapter over a 6,543-wide head
  vector. `finetune()` previously refused this architecture.
- `MIN_FINETUNE_REFERENCE`, with a warning when fine-tuning is attempted on fewer
  reference PSMs. Measured on six unseen LC setups, fine-tuning was worse than
  calibration below roughly 500 reference peptidoforms and better above 700; the
  validation split is widened automatically below the threshold so early stopping has
  signal to work with.
- The new setup's `scale` and `shift` are solved by least squares on the reference data
  before training, rather than learned. Left to the optimiser on a small reference set
  they collapse: on a 133-minute gradient with 230 reference peptides the output range
  shrank to 17 minutes and the error reached 91 minutes, with the correlation still above
  0.9 because the ordering was never what broke. Anchoring them brought that setup to
  2.2 minutes.

- Fused-trunk multitask architecture (`FlexCNNMultitaskModel`), which merges atomic
  composition with a learned residue embedding in a single convolutional trunk and pools
  over the valid length instead of flattening four separate branches. Available as
  `deeplc.core.FLEXCNN_MULTITASK_MODEL`; the bundled model was trained across 6,543 LC
  setups and reaches 0.82 min test MAE and 0.24 min median against 1.26 min and 0.51 min
  for the four-branch backbone on the same data and the same head.
- `FactorHead`, a low-rank multitask head where a setup owns only `rank + 2` parameters,
  so adding an LC setup means fitting 66 values with the encoder frozen rather than
  training a head.
- Self-describing checkpoints. A model file may now record its architecture, constructor
  arguments, feature specification and target units, so loading no longer infers the
  architecture from tensor shapes. Bare state dicts continue to load unchanged.
- `add_terminal_composition` on `DeepLCDataset` and `DeepLCDataset.from_psm_list`,
  passed through to `encode_peptidoform`.

### Fixed

- Fine-tuning could return a model far worse than the one it started from. Three
  defects compounded: `train()` began with an infinite best validation loss, so the
  first epoch always became the best however bad it was; the output layer's scale was
  learned rather than solved, though it is linear in its input; and the adapter's ReLU
  stack is largely dead at its default initialisation, which left the activations rank
  deficient and made a CUDA least-squares solve return non-finite values and silently
  decline. On a 133-minute gradient with 230 reference peptides the adapter path
  returned predictions spanning 1.3 to 27.2 minutes at an error of 92 minutes, with
  the correlation still above 0.9 because only the scale was lost. Training now scores
  its starting point, both adaptation paths solve their output layer on the reference
  data first, and that solve is rank tolerant. The same setup now gives 1.63 minutes
  for the adapter path and 2.05 for the low-rank head, against 1.47 and 1.28 for
  calibration.
- A fine-tuned model whose validation error exceeds a large fraction of the reference
  retention-time span is now reported at error level, since a collapsed fit leaves the
  loss curve and the correlation looking unremarkable.

### Changed

- `predict()` loads the model before encoding features, so a model that records a feature
  specification gets the features it was trained on. Previously the dataset was always
  built with defaults.

## [4.0.0] - 2026-07-24

### Changed

- Default epochs for finetuning to 50

## [4.0.0b1] - 2026-07-10

### Added

- Bundled multitask pretrained model as the new default, trained across multiple LC setups. Automatic head selection in `calibrate()` based on Pearson correlation ensures the best fitting setup is used for predictions. Fine-tuning uses adapter-based transfer learning: a small MLP is attached on top of the multi-setup prediction heads.
- NiceGUI-based web interface, launchable as a browser app (`deeplc gui`) or native desktop window (`deeplc gui --native`)
- `[gui]` optional dependency group (nicegui, plotly, pywebview) for desktop use
- `[web]` optional dependency group (nicegui, plotly) for server/Docker use
- Docker image for containerized web server deployment
- Updated Windows installer (PyInstaller and Inno Setup) to new GUI
- `predict_and_calibrate()` core function combining prediction and calibration in one call, with optional automatic reference PSM selection
- `finetune_and_predict()` core function for transfer learning followed by calibrated prediction
- Automatic calibration reference selection from input PSMs using q-value filtering or top-scoring fraction
- `Calibration.selected_model_head` field to record which model output head a calibration was fitted to
- Publish workflow with Windows installer build, Docker image build, and dry-run mode for CI testing without publishing

### Changed

- CLI restructured into `predict` and `gui` subcommands
- Example datasets updated from legacy CSV format to psm_utils TSV and peprec formats

## [4.0.0-alpha.1]

### Changed

- Simplified the public package API by splitting up the single class-based API into core functions (`predict`, `finetune`, `train`, etc.)
- Switched deep learning framework from Tensorflow to PyTorch
- Speed up predictions by removing ensemble method where output from three models with differing kernel sizes was averaged to one prediction
- Separated calibration logic to dedicated reusable module with sklearn-like API.
- Improved computational efficiency of piece-wise linear calibration and set sensible default parameters
- Built-in transfer learning functionality, instead of using external `deeplcretrainer` package.
- Cleaned up package, removing legacy and unused code and files, and improving modularity
- Modernized CI workflows to use `uv`
- Added sphinx-based documentation for readthedocs

### Removed

- Removed library-feature for storing past predictions
- Removed legacy CALLC functionality

## [3.1.13] - 2025-09-01

### Changed

- Bump version

## [3.1.12] - 2025-09-01

### Changed

- Bump version

## [3.1.11] - 2025-09-01

### Changed

- Bump version

## [3.1.10] - 2025-08-27

### Changed

- Fix no calibration peptides when looking for best model

## [3.1.9] - 2025-07-14

### Changed

- When no calibration peptides are present, just fit a model that returns the original predicted value

## [3.1.8] - 2025-02-21

### Changed

- Allow for much smaller peptides in feature calculation

## [3.1.7] - 2025-02-03

### Changed

- Log warnings in obtaining atoms only once

## [3.1.6] - 2025-02-03

### Changed

- Remove unimod, replace with psm_utils

## [3.1.5] - 2025-02-03

### Changed

- Bump minimal requirements python version

## [3.1.4] - 2025-02-03

### Changed

- Bump minimal requirements python version

## [3.1.3] - 2024-11-22

### Changed

- Bioconda fix import

## [3.1.2] - 2024-11-21

### Changed

- Remove dependencies

## [3.1.1] - 2024-10-10

### Changed

- Revert to linear calibration after transfer learning only

## [3.1.0] - 2024-08-31

### Changed

- Use scikit-learn instead of pygam for calibration

## [3.0.8] - 2024-08-25

### Changed

- Fix custom activation by using default implementation
- Loosen requirements

## [3.0.7] - 2024-08-19

### Changed

- Remove batch number parameter from GUI

## [3.0.6] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.5] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.4] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.3] - 2024-08-08

### Changed

- Remove old python build

## [3.0.2] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.1] - 2024-08-08

### Changed

- Windows release fix

## [3.0.0] - 2024-08-08

### Changed

- New TensorFlow versions that break support with earlier models

## [2.2.38] - 2024-07-01

### Changed

- Relax tensorflow version for mac

## [2.2.37] - 2024-07-01

### Changed

- Pin scipy version

## [2.2.36] - 2024-04-14

### Changed

- Set max threads

## [2.2.35] - 2024-04-14

### Changed

- Remove limit threads feature calc

## [2.2.34] - 2024-04-13

### Fixed

- Fix issue transfer learning single model mode

## [2.2.33] - 2024-04-13

### Changed

- Make single model mode available and the default

## [2.2.32] - 2024-02-16

### Changed

- Bump version

## [2.2.31] - 2024-02-16

### Changed

- Bump version

## [2.2.30] - 2024-02-16

### Changed

- Bump version with new workflows

## [2.2.29] - 2024-02-16

### Changed

- Bump version with .toml

## [2.2.28] - 2024-02-16

### Changed

- Removed dependencies (most importantly sklearn; which is optional now)

## [2.2.27] - 2024-01-22

### Changed

- Support for custom labels in plot

## [2.2.26] - 2023-11-15

### Fixed

- Fix memory usage, limit threads

## [2.2.25] - 2023-11-15

### Fixed

- Fixed multiprocessing

## [2.2.24] - 2023-11-14

### Changed

- Pass flag CCS feature extract

## [2.2.23] - 2023-11-13

### Changed

- Reintroduce ability to predict CCS

## [2.2.22] - 2023-09-22

### Fixed

- Fix pypi token GA

## [2.2.21] - 2023-09-22

### Fixed

- Fix non-initialized modifications

### Fixed

- Allow for setting number of epochs

## [2.2.20] - 2023-09-19

### Fixed

- Fix selenium atom logging error

## [2.2.19] - 2023-09-19

### Fixed

- Fix selenium atom on AA

## [2.2.18] - 2023-09-18

### Fixed

- Fix wrong error except

## [2.2.17] - 2023-09-18

### Fixed

- Fix peptides that are too long and their modifcations

## [2.2.16] - 2023-09-18

### Changed

- bump version

## [2.2.15] - 2023-09-18

### Changed

- reduce logging (change to debug)

## [2.2.14] - 2023-09-03

### Fixed

- Fix missing debug time in feature extractor

## [2.2.13] - 2023-09-03

### Fixed

- Fix potential issues in feat extractor (uncommon AA)

## [2.2.12] - 2023-08-19

### Changed

- Set plotly param to false

## [2.2.11] - 2023-08-19

### Changed

- Remove assumption psm_utils_obj

## [2.2.10] - 2023-08-19

### Changed

- Add plotly diagnostic plots

## [2.2.9] - 2023-08-08

### Fixed

- Fix logger issue

## [2.2.8] - 2023-08-08

### Changed

- Activate garbage collection to clear GPU memory

## [2.2.7] - 2023-08-03

### Changed

- Reintroduce support for batched calculation
- Introduce batch_num_tf parameter for DeepLC that determines TF batch size

## [2.2.6] - 2023-08-02

### Changed

- Allow for linear piecewise calibration again
- More strict version of TF, as 2.13.0 crashes for transfer learning

## [2.2.5] - 2023-08-02

### Fixed

- Fixed a bug where a reinit led to issues with setting parallelism

## [2.2.4] - 2023-07-06

### Fixed

- Fixed a bug where it checked the calibration file while this was not initialized

## [2.2.3] - 2023-06-28

### Fixed

- Fixed a bug where missing atoms in encoding cause a crash

## [2.2.2] - 2023-06-23

### Fixed

- Fixed a bug where isotopes are incorrectly parsed

## [2.2.1] - 2023-06-18

### Fixed

- Fixed a bug where numpy no longer accepts dict_values, explicit list conversion

## [2.2.0] - 2023-06-14

### Fixed

- Fixed a bug where atom counts were wrong, fixed by retraining models with new features

## [2.1.9] - 2023-05-09

### Fixed

- Fixed a bug where detection of legacy csv is wrong

## [2.1.8] - 2023-05-09

### Changed

- remove pygam from GUI, set to true as default

## [2.1.7] - 2023-05-09

### Fixed

- fix setting cmd line calibration default

## [2.1.6] - 2023-05-09

### Fixed

- fix setting GUI calibration default

## [2.1.5] - 2023-05-08

### Fixed

- fix library feature

## [2.1.4] - 2023-05-08

### Changed

- slight refractoring

## [2.1.3] - 2023-05-08

### Changed

- slight refractoring

## [2.1.2] - 2023-05-08

### Changed

- add Arthur Declercq as contributor

## [2.1.2] - 2023-05-08

### Fixed

- dependency fix

## [2.1.1] - 2023-05-08

### Fixed

- Calibration bug fix

## [2.1.0] - 2023-05-08

### Changed

- Support for GUI and PSM utils
- Support for GUI and transfer learning

## [2.0.4] - 2023-05-06

### Changed

- Log peptide sequences that are too long

## [2.0.3] - 2023-05-02

### Changed

- bump version
- second try

## [2.0.2] - 2023-05-02

### Changed

- bump version

## [2.0.1] - 2023-05-02

### Changed

- bug fix for sequences > 60 AA
- bug fix non-resolved modifications

## [2.0.0] - 2023-04-30

### Added

- psm_utils integration
- pygam default calibrator

### Changed

- additional support transfer learning
- code refractoring of feature extraction

### Removed

- prediction library support
- piecewise linear calibration support

## [1.2.1] - 2023-02-13

### Changed

- Bump version

## [1.2.0] - 2023-02-13

### Added

- Support for DeepLCRetrainer

### Changed

- Optimized library usage (~85% reduce in DeepLC prediction time) @markmipt

## [1.1.2] - 2022-04-06

### Changed

- Made Gooey an optional dependency (facilitates install on Linux). Install the optional
  dependencies for the graphical user interface with `pip install deeplc[gui]`

## [1.1.1] - 2022-04-03

### Added

- New native Python GUI, based on the Gooey package
- New standalone installer for Windows using PyInstaller and Inno Setup
- Added `deeplc-gui` entrypoint to start GUI from the command line

### Changed

- CLI: Restructured help message
- Made DeepLC class API docstring consistent with CLI help message
- Docs: Moved `dict_divider` and `split_cal` explanation to README Q&A section.
- CI: Only run tests on commits to `main` or from a PR
- Refactoring: Cleaned up `__main__.py`
- Logging: Changed some loggings from DEBUG to INFO level, some from WARNING to INFO or
  DEBUG level

### Removed

- Removed Java-based GUI in favor of new Python-based GUI

### Fixed

- If run through CLI/GUI, all Tensorflow warnings are now fully suppressed
- Added `--legacy_calibration` CLI option to allow for old piecewise linear calibration
  while new pyGAM calibration method is default. `--legacy_calibration` is mutually
  exclusive with `--pygam_calibration`.

## [1.0.1] - 2022-03-17

- Make version compatible with pip release

## [1.0.0] - 2022-02-23

- Make version compatible with pip release

## [1.0] - 2022-02-23

- Make pygam the default calibration method

## [0.2.2] - 2022-02-17

- Bug fix where split_cal was not correctly passed
- CMD support for pygam calibration

## [0.2.1] - 2022-02-03

- Scikit-learn and pygam dependency in setup

## [0.2.0] - 2022-01-21

- Bug fix duplicate peptide+mod for DeepCALLC
- New feature DeepCALLC

## [0.1.39] - 2022-01-12

- Version bump

## [0.1.38] - 2022-01-10

- Deep(CAL)LC functionality

## [0.1.37] - 2021-09-09

- Pygam as calibration function

## [0.1.36] - 2021-09-09

- Update to Streamlit webserver: Use `st.form` and new official download button

## [0.1.35] - 2021-08-09

- More elegant solution for library call as global

## [0.1.34] - 2021-07-29

- Fix var call dict object

## [0.1.33] - 2021-07-29

- Fix library delete call

## [0.1.32] - 2021-07-29

- Temporary fix suggested by markmipt for the library

## [0.1.31] - 2021-07-01

- Change logging to specified logger object instead of standard logger

## [0.1.30] - 2021-06-09

- GUI: Fix small font through starting jar with cmd to increase font size
- Added testing for Python 3.9
- Relax h5py requirement to allow v3
- Fixed GitHub Action workflow for Streamlit docker image build

## [0.1.29] - 2021-03-24

- Bug in writing library where a list was assumed so library only partially filled

## [0.1.28] - 2021-03-17

- Make it optional to reload library

## [0.1.27] - 2021-03-16

- Ignore library messages

## [0.1.26] - 2021-03-15

- Force older library h5py for compatability

## [0.1.25] - 2021-03-12

- Library gets appended for non-calibration peptides too

## [0.1.24] - 2021-03-12

- Change the default windows install to pip instead of conda
- Change reporting of identifiers used from library

## [0.1.23] - 2021-03-04

- Log the amount of identifiers in library used

## [0.1.22] - 2021-03-04

- Publish PyPI and GitHub release

## [0.1.21] - 2021-03-04

- Add library functionality that allows for storing and retrieving predictions (without running the model)

## [0.1.20] - 2021-02-19

- Describe hyperparameters and limit CPU threads
- Additional modfications, including those exclusive to pFind
- Change calibration error to warning (since it is a warning if it is out of range...)

## [0.1.18] - 2021-01-11

- Limit CPU usage by tensorflow by connecting to n_jobs

## [0.1.17] - 2020-08-07

- Support for Python 3.8

## [0.1.16] - 2020-05-18

- Bug fix in the calibration function
- Had to order the predicted instead of observed retention times of the calibration analytes
- Thanks to @courcelm for both finding and fixing the issue

## [0.1.15] - 2020-05-15

- Different calibration function, should not contain gaps anymore
- Changed to more accurate rounding
- Changed to splitting in groups of retention time instead of groups of peptides

## [0.1.14] - 2020-02-21

- Changed default model to a different data set

## [0.1.13] - 2020-02-21

- Duplicate peptides charge fix

## [0.1.12] - 2020-02-15

- Support for charges and spaces in peprec

## [0.1.11] - 2020-02-13

- Fixes in GUI

## [0.1.10] - 2020-02-10

- Include less models in package to meet PyPI 60MB size limitation

## [0.1.9] - 2020-02-09

- Bugfix: Pass custom activation function

## [0.1.8] - 2020-02-07

- Fixed support for averaging predictions of groups of models (ensemble) when no models were passed
- New models for ensemble

## [0.1.7] - 2020-02-07

- Support for averaging predictions of groups of models (ensemble)

## [0.1.6] - 2020-01-21

- Fix the latest release

## [0.1.5] - 2020-01-21

- Spaces in paths to files and installation allowed
- References to other CompOmics tools removed in GUI

## [0.1.5] - 2020-02-13

- Fixes in GUI

## [0.1.4] - 2020-01-17

- Fix the latest release

## [0.1.3] - 2020-01-17

- Fixed the .bat installer (now uses bioconda)

## [0.1.2] - 2019-12-19

- Example files in GUI folder
- Unnecesary bat and sh for running GUI removed

## [0.1.1] - 2019-12-18

- Switch to setuptools
- Reorder publish workflow; build wheels

## [0.1.1.dev8] - 2019-12-16

- Remove xgboost dependancy

## [0.1.1.dev7] - 2019-12-16

- Use dot instead of dash in versioning for bioconda

## [0.1.1-dev6] - 2019-12-15

- Fix publish action branch specification (2)

## [0.1.1-dev5] - 2019-12-15

- Fix publish action branch specification

## [0.1.1-dev4] - 2019-12-15

- Test other trigger for publish action

## [0.1.1-dev3] - 2019-12-15

- Update documentation, specify branch in publish action

## [0.1.1-dev2] - 2019-12-15

- Add long description to setup.py

## [0.1.1-dev1] - 2019-12-15

- Initial pre-release
