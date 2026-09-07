*****************
Prediction models
*****************

Default model
=============

DeepLC ships a pretrained multitask model as the default. Since 4.1.1 this is
``multitask_flexcnn_model.pt``: a fused-trunk convolutional model with a low-rank
multitask head, trained jointly across 6,543 LC setups from public repositories.
It outputs one retention time prediction (in minutes) per setup. The best-fitting
setup is selected automatically during calibration based on Pearson correlation to
the observed retention times in the reference set, and fine-tuning fits a new setup
head (66 parameters) on the reference with the trunk frozen.

Without calibration, :func:`deeplc.predict` reports the setup named by
:data:`deeplc.core.DEFAULT_TASK_NAME` (``PXD005573_mcp``, the 200-minute gradient
that DeepLC 1.x to 3.x models were trained on), or the full matrix with
``return_matrix=True``. The setup names are available as ``model.task_names``.

The 4.0 default, ``multitask_model.pt`` (shared trunk, one head per setup), stays
bundled as :data:`deeplc.core.LEGACY_MULTITASK_MODEL` and can be passed as
``model=`` to any core function to reproduce 4.0 and 4.1.0 predictions.

Calibrating against several setups at once
==========================================

:func:`deeplc.calibrate` and :func:`deeplc.predict_and_calibrate` calibrate against the full
``(n, n_heads)`` prediction matrix of a model (``n_heads=1`` for a single-setup model), and the
calibration instance is responsible for selecting which head(s) to use.
:class:`~deeplc.calibration.MultiHeadCalibration` is the base for such calibrations; the default
is :class:`~deeplc.calibration.MultiHeadRidgeCalibration`: every head is ranked by Pearson
correlation to the reference, the 80 best are calibrated individually, and a ridge regression maps
those calibrated estimates onto the observed retention times, so several setups contribute. The
number of heads is the one parameter worth changing: 80 sits on a flat optimum between roughly 40
and 320, and the class never fits more weights than half the reference allows. Prediction costs
nothing extra, because the full head matrix is computed either way.

A lighter alternative, a single naive calibration on the best-correlating head, is available via
:class:`~deeplc.calibration.MultiHeadSplineCalibration` or
:class:`~deeplc.calibration.MultiHeadPiecewiseLinearCalibration`:

.. code-block:: python

   from deeplc import predict_and_calibrate
   from deeplc.calibration import MultiHeadSplineCalibration

   calibrated_rt = predict_and_calibrate(
       psm_list,
       psm_list_reference=reference,
       calibration=MultiHeadSplineCalibration(),
   )

The naive, single-series calibrations in :mod:`deeplc.calibration.simple`
(:class:`~deeplc.calibration.SplineTransformerCalibration`,
:class:`~deeplc.calibration.PiecewiseLinearCalibration`,
:class:`~deeplc.calibration.IdentityCalibration`) know nothing about heads; the ``MultiHead*``
classes above delegate to them once a head is picked. An unfitted
:class:`~deeplc.calibration.SplineTransformerCalibration` or
:class:`~deeplc.calibration.PiecewiseLinearCalibration` passed to ``calibrate`` or
``predict_and_calibrate`` is upgraded to its ``MultiHead*`` counterpart automatically via
:func:`deeplc.calibration.upgrade_calibration`. An already fitted naive calibration is not
accepted, since it carries no record of which head it was fit on; fit a ``MultiHead*Calibration``
instead in that case.

Training a model from scratch
==============================

:func:`deeplc.train` trains a new model on a PSM list with observed retention
times:

.. code-block:: python

   from psm_utils.io import read_file
   from deeplc import train, save_model

   psm_list = read_file("training_psms.tsv")
   model = train(psm_list)
   save_model(model, "my_model.pt")

Using a custom model
====================

While the default model should work well for nearly all LC setups, a custom
model checkpoint can be passed to any core function via the ``model`` argument:

.. code-block:: python

   from deeplc import predict_and_calibrate

   calibrated_rt = predict_and_calibrate(psm_list, model="path/to/model.pt")

Checkpoints must be plain PyTorch state dicts saved with
``torch.save(model.state_dict(), path)``. See :func:`deeplc.save_model`.
