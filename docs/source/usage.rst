*****
Usage
*****

Web application
===============

A hosted web application is available at
`iomics.ugent.be/deeplc <https://iomics.ugent.be/deeplc/>`_ — no installation required.


Graphical interface
===================

**Windows:** download the one-click installer from the
`releases page <https://github.com/compomics/DeepLC/releases/latest>`_.

**Other platforms:** install with GUI dependencies:

.. code-block:: sh

   pip install deeplc[gui]

Then launch as a browser app or native desktop window:

.. code-block:: sh

   deeplc gui            # opens in browser
   deeplc gui --native   # opens as desktop window


Command line interface
======================

Install DeepLC:

.. code-block:: sh

   pip install deeplc
   # or
   conda install -c bioconda -c conda-forge deeplc

Predict retention times for a PSM file:

.. code-block:: sh

   deeplc predict <psm_file>

The input file format is inferred from the extension. All formats supported by
`psm_utils <https://psm-utils.readthedocs.io/en/stable/api/psm_utils.io.html>`_
are accepted, including Sage, MaxQuant msms.txt, mzTab, and others. A
tab-separated file with at least ``peptidoform`` and ``spectrum_id`` columns is
also accepted directly.

For calibration, pass a reference file with observed retention times using the
``--psm-file-reference`` option. If no reference file is provided, DeepLC
attempts automatic calibration from high-confidence PSMs in the input file.

The reference is deduplicated before it is used: only the first PSM of each
peptidoform is kept, charge states included. A search result usually contains
the same peptidoform from many spectra with a different observed retention time
each time, and those repeats give the calibration conflicting targets while
weighing peptidoforms by how often they were identified. This is unconditional;
a caller that really wants every reference PSM to count fits a
:class:`~deeplc.calibration.Calibration` on its own targets and passes it to
:func:`~deeplc.core.predict_and_calibrate`, which uses an already fitted
calibration as given. When the repeats of a peptidoform disagree by a large
fraction of the observed retention-time range, DeepLC reports it: the reference
then mixes runs or contains low-confidence PSMs, and deduplication only hides
that.

For a full list of options:

.. code-block:: sh

   deeplc predict --help


Prediction reports
==================

:func:`deeplc.prediction_report` returns predictions together with what a bare number cannot
say: whether the model has seen the peptidoform, how far the nearest known sequence is, and how
far off the prediction may plausibly be.

.. code-block:: python

   from deeplc import prediction_report

   report = prediction_report(psm_list, psm_list_reference=reference, coverage=0.90)
   report[["peptidoform", "predicted_rt", "ci_lower", "ci_upper",
           "in_reference", "dist_to_reference"]]

The interval is a cross-fitted conformal interval calibrated on the reference, so its coverage
holds on peptides exchangeable with the reference, without retraining and regardless of the
model. Pass ``calibration=MultiHeadRidgeCalibration()`` to combine setups; the membership
column then covers every selected head.

The width of that interval depends on the predicted retention time and, with a multi-head
calibration, on the peptide itself: the combined setup heads each estimate the same retention
time, and how far those estimates lie apart is an uncertainty that varies per peptide. Two
peptides predicted at the same retention time therefore get different intervals. Pass
``per_peptide_width=False`` for widths that depend on the predicted retention time alone.

With a training index (built from the multitask training corpus and distributed separately),
three more columns appear: ``in_training`` (exact peptidoform match anywhere in the corpus),
``in_selected_heads_training`` (match within the setups the calibration selected) and
``dist_to_training`` (Levenshtein distance to the closest training sequence, exact up to ten
edits and capped beyond):

.. code-block:: python

   report = prediction_report(psm_list, psm_list_reference=reference,
                              training_index="deeplc_training_index_v6f.dlcidx")

Python API
==========

The public API consists of a small set of functions in :mod:`deeplc.core`.

Direct prediction
-----------------

Predict retention times without calibration:

.. code-block:: python

   from psm_utils.io import read_file
   from deeplc import predict

   psm_list = read_file("results.sage.tsv")
   rt_predictions = predict(psm_list)  # numpy array, shape (n,)

Prediction with calibration
----------------------------

Calibrate to observed retention times in a reference set, then predict:

.. code-block:: python

   from psm_utils.io import read_file
   from deeplc import predict_and_calibrate

   psm_list = read_file("results.sage.tsv")

   # Auto-calibration: selects reference PSMs from psm_list automatically
   calibrated_rt = predict_and_calibrate(psm_list)

   # Or provide an explicit reference set
   psm_list_reference = read_file("reference.tsv")
   calibrated_rt = predict_and_calibrate(psm_list, psm_list_reference=psm_list_reference)

Fine-tuning
-----------

Fine-tune the model to a specific dataset, then predict:

.. code-block:: python

   from psm_utils.io import read_file
   from deeplc import finetune_and_predict

   psm_list = read_file("results.sage.tsv")
   calibrated_rt = finetune_and_predict(psm_list)

For lower-level control, :func:`deeplc.calibrate` and :func:`deeplc.finetune`
return a fitted :class:`~deeplc.calibration.Calibration` instance and a
fine-tuned model respectively, which can be reused across multiple prediction
calls.

See the :doc:`API reference <api/deeplc>` for all parameters and return types.


Input file format
=================

DeepLC accepts any PSM file format supported by
`psm_utils <https://psm-utils.readthedocs.io/en/stable/api/psm_utils.io.html>`_.
The format is inferred from the file extension, or can be set explicitly with
``--psm-filetype``.

A tab-separated file with the following columns is also accepted:

.. code-block:: text

   spectrum_id	peptidoform	retention_time
   0	AAGPSLSHTSGGTQSK/2	12.16
   1	AAINQK[Acetyl]LIETGER/2	34.10
   2	AANDAGYFNDEM[Oxidation]APIEVK/2	37.38

``peptidoform``
    Peptide sequence in
    `ProForma 2.0 <https://pubs.acs.org/doi/10.1021/acs.jproteome.1c00771>`_ notation
    (see `Modifications`_ below).

``spectrum_id``
    Unique identifier for each PSM.

``retention_time``
    Observed retention time, required for calibration and fine-tuning.

See `example datasets <https://github.com/compomics/DeepLC/tree/main/examples/datasets>`_
for additional input file examples.


Modifications
-------------

Modifications are specified as bracketed labels in ProForma 2.0 notation. Labels must be
resolvable to a known chemical formula:

- A name or accession from a controlled vocabulary: Unimod or PSI-MOD
  (e.g. ``Oxidation``, ``U:21``, ``MOD:00046``)
- An elemental formula (e.g. ``Formula:C2H2O``)

All modifications — including fixed modifications such as carbamidomethylation — must be
present in the peptidoform string. Labels that cannot be resolved to a chemical formula
(e.g. mass shifts) are ignored; predictions fall back to the unmodified peptide.

DeepLC can predict retention times for any modification, but accuracy depends on whether
similar modifications were seen during training. For modifications involving elements or
structural changes not well represented in the training data, prediction accuracy may be
lower. In such cases, `fine-tuning <Fine-tuning_>`_ the model on a dataset containing
the modification of interest is recommended.

Custom modifications not in Unimod or PSI-MOD can be encoded with an elemental formula
directly:

.. code-block:: text

   PEPTI[Formula:C12H20O2]DE
