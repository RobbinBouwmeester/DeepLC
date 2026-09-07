"""Tests for the fused-trunk multitask architecture and self-describing checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deeplc import _model_ops, core
from deeplc._architecture import (
    DeepLCModel,
    FactorHead,
    FlexCNNMultitaskModel,
    InputNorm,
)
from deeplc._features import encode_peptidoform
from deeplc.data import DeepLCDataset

MAXLEN, N_ATOMS, N_RESIDUES, PAD = 60, 6, 20, 20
GLOBAL_DIM = 67

# A model small enough to build in a test but structurally identical to the shipped
# one: same modules, same forward, only narrower.
SMALL = dict(
    global_dim=GLOBAL_DIM,
    embed_dim=4,
    channels=(8, 8),
    kernel_size=5,
    stem_channels=6,
    stem_layers=2,
    width=12,
    depth=2,
    rank=3,
)


def make_batch(lengths, seed=0):
    """Structurally valid features for peptides of the given lengths."""
    rng = np.random.RandomState(seed)
    batch = len(lengths)
    x_atom = np.zeros((batch, MAXLEN, N_ATOMS), dtype=np.float32)
    one_hot = np.zeros((batch, MAXLEN, N_RESIDUES), dtype=np.float32)
    for i, length in enumerate(lengths):
        x_atom[i, :length] = rng.randint(0, 12, size=(length, N_ATOMS))
        residues = rng.randint(0, N_RESIDUES, size=length)
        one_hot[i, np.arange(length), residues] = 1.0
    x_global = rng.randn(batch, GLOBAL_DIM).astype(np.float32)
    return (
        torch.from_numpy(x_atom),
        torch.empty(0),
        torch.from_numpy(x_global),
        torch.from_numpy(one_hot),
    )


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #


def test_forward_returns_one_prediction_per_task():
    """Every LC setup gets a prediction for every peptide in the batch."""
    model = FlexCNNMultitaskModel(n_tasks=7, **SMALL).eval()
    with torch.no_grad():
        out = model(*make_batch([9, 14, 30]))
    assert out.shape == (3, 7)
    assert torch.isfinite(out).all()


def test_task_subset_matches_full_matrix():
    """Selecting tasks must equal slicing the full output, as calibration relies on it."""
    model = FlexCNNMultitaskModel(n_tasks=9, **SMALL).eval()
    batch = make_batch([12, 21])
    idx = torch.tensor([0, 4, 8])
    with torch.no_grad():
        full = model(*batch)
        subset = model(*batch, task_idx=idx)
    torch.testing.assert_close(full[:, idx], subset, rtol=1e-5, atol=1e-5)


def test_padding_does_not_change_prediction():
    """
    A peptide's prediction must not depend on how much padding follows it.

    Masking is central to this architecture, so the two inputs have to differ in
    padded length for the test to mean anything: the same ten residues are placed
    once in a length-20 array and once in a length-60 one.
    """
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    rng = np.random.RandomState(1)
    length = 10
    atoms = rng.randint(0, 12, size=(length, N_ATOMS)).astype(np.float32)
    residues = rng.randint(0, N_RESIDUES, size=length)
    x_global = torch.from_numpy(rng.randn(1, GLOBAL_DIM).astype(np.float32))

    def build(padded_to):
        x_atom = np.zeros((1, padded_to, N_ATOMS), dtype=np.float32)
        one_hot = np.zeros((1, padded_to, N_RESIDUES), dtype=np.float32)
        x_atom[0, :length] = atoms
        one_hot[0, np.arange(length), residues] = 1.0
        return torch.from_numpy(x_atom), torch.from_numpy(one_hot)

    short_atom, short_hot = build(20)
    long_atom, long_hot = build(60)
    with torch.no_grad():
        a = model(short_atom, torch.empty(0), x_global, short_hot)
        b = model(long_atom, torch.empty(0), x_global, long_hot)
    torch.testing.assert_close(a, b)


def test_length_one_peptide_is_handled():
    """A single residue leaves the max-pool with one valid position, not none."""
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    with torch.no_grad():
        out = model(*make_batch([1]))
    assert torch.isfinite(out).all()


def test_all_padding_row_does_not_produce_nan():
    """
    An empty peptide masks every position.

    The max over an entirely masked row is -inf before the guard, so this checks
    the guard rather than a realistic input.
    """
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    x_atom = torch.zeros(1, MAXLEN, N_ATOMS)
    one_hot = torch.zeros(1, MAXLEN, N_RESIDUES)
    x_global = torch.zeros(1, GLOBAL_DIM)
    with torch.no_grad():
        out = model(x_atom, torch.empty(0), x_global, one_hot)
    assert torch.isfinite(out).all()


def test_x_atom_sum_is_ignored():
    """The fused trunk reads x_atom directly, so the rolling-sum array is unused."""
    model = FlexCNNMultitaskModel(n_tasks=4, **SMALL).eval()
    x_atom, _, x_global, one_hot = make_batch([15, 25])
    with torch.no_grad():
        a = model(x_atom, torch.empty(0), x_global, one_hot)
        b = model(x_atom, torch.randn(2, 30, N_ATOMS), x_global, one_hot)
    torch.testing.assert_close(a, b)


def test_wrong_global_width_fails_loudly():
    """
    Feeding the 55-dimensional default vector must raise, not silently rescale.

    This is the failure mode worth protecting: a shape error is recoverable, a
    quietly wrong retention time is not.
    """
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    x_atom, _, _, one_hot = make_batch([12])
    with pytest.raises(RuntimeError):
        model(x_atom, torch.empty(0), torch.zeros(1, 55), one_hot)


# --------------------------------------------------------------------------- #
# encoder details
# --------------------------------------------------------------------------- #


def test_residue_indices_marks_padding():
    """All-zero one-hot rows are padding; argmax alone would call them residue 0."""
    encoder = FlexCNNMultitaskModel(n_tasks=2, **SMALL).encoder
    one_hot = torch.zeros(1, 4, N_RESIDUES)
    one_hot[0, 0, 0] = 1.0  # residue 0, genuinely
    one_hot[0, 1, 7] = 1.0
    # rows 2 and 3 left empty
    idx = encoder.residue_indices(one_hot)
    assert idx.tolist() == [[0, 7, PAD, PAD]]


def test_residue_counts_ignores_padding():
    """Counts cover the twenty residues and exclude padding positions."""
    encoder = FlexCNNMultitaskModel(n_tasks=2, **SMALL).encoder
    idx = torch.tensor([[3, 3, 5, PAD, PAD]])
    counts = encoder.residue_counts(idx)
    assert counts.shape == (1, N_RESIDUES)
    assert counts[0, 3].item() == 2
    assert counts[0, 5].item() == 1
    assert counts.sum().item() == 3  # padding contributes nothing


def test_input_norm_leaves_constant_features_alone():
    """
    A feature with no variance keeps raw units.

    Clamping its standard deviation to a floor would multiply any non-zero test
    value by one over that floor, which is how a single phosphate once destroyed
    the forward pass for a model trained without phosphorus.
    """
    norm = InputNorm(3)
    values = torch.tensor([[1.0, 5.0, 0.0], [1.0, 7.0, 0.0], [1.0, 9.0, 0.0]])
    norm.fit(values)
    assert norm.std[0].item() == pytest.approx(1.0)
    assert norm.std[2].item() == pytest.approx(1.0)
    assert norm.std[1].item() > 1.0
    out = norm(torch.tensor([[1.0, 7.0, 1000.0]]))
    assert out[0, 2].item() == pytest.approx(1000.0)


def test_factor_head_parameter_count():
    """Adding a setup costs rank + 2 parameters, which is the point of the head."""
    head = FactorHead(trunk_dim=16, n_tasks=100, rank=8)
    per_task = head.embedding.shape[1] + 2
    assert per_task == 10
    shared = head.proj.weight.numel() + head.proj.bias.numel()
    assert shared == 16 * 8 + 8


# --------------------------------------------------------------------------- #
# self-describing checkpoints
# --------------------------------------------------------------------------- #


def _write_described(tmp_path, **overrides):
    model = FlexCNNMultitaskModel(n_tasks=5, **SMALL)
    encoder_kwargs = {k: v for k, v in SMALL.items() if k != "rank"}
    blob = {
        "state_dict": model.state_dict(),
        "architecture": "FlexCNNMultitaskModel",
        "encoder_kwargs": encoder_kwargs,
        "head_kwargs": {"rank": SMALL["rank"]},
        "n_tasks": 5,
        "feature_spec": {
            "name": "global67_terminal",
            "global_dim": GLOBAL_DIM,
            "add_terminal_composition": True,
            "add_ccs_features": False,
            "padding_length": MAXLEN,
        },
        "target_units": "minutes",
        "task_names": [f"setup_{i}" for i in range(5)],
    }
    blob.update(overrides)
    path = tmp_path / "described.pt"
    torch.save(blob, path)
    return model, path


def test_described_checkpoint_round_trips(tmp_path):
    """A described checkpoint rebuilds the same model and carries its metadata."""
    original, path = _write_described(tmp_path)
    loaded = _model_ops.load_model(path, device="cpu")
    assert isinstance(loaded, FlexCNNMultitaskModel)
    assert loaded.feature_spec["add_terminal_composition"] is True
    assert loaded.target_units == "minutes"
    assert loaded.task_names[0] == "setup_0"

    batch = make_batch([11, 19], seed=3)
    original.eval()
    with torch.no_grad():
        torch.testing.assert_close(original(*batch), loaded(*batch))


def test_unknown_architecture_is_rejected(tmp_path):
    """An unrecognised architecture name must fail with a clear message."""
    _, path = _write_described(tmp_path, architecture="SomeFutureModel")
    with pytest.raises(ValueError, match="does not know"):
        _model_ops.load_model(path, device="cpu")


def test_bare_state_dict_still_loads(tmp_path):
    """The old checkpoint format must keep working."""
    legacy = DeepLCModel(n_heads=3)
    path = tmp_path / "legacy.pt"
    torch.save(legacy.state_dict(), path)
    loaded = _model_ops.load_model(path, device="cpu")
    assert isinstance(loaded, DeepLCModel)
    assert loaded.heads.b2.shape[0] == 3


# --------------------------------------------------------------------------- #
# features and the prediction path
# --------------------------------------------------------------------------- #


def test_terminal_composition_gives_the_expected_width():
    """The terminal block lengthens the global vector from 55 to 67."""
    without = encode_peptidoform("PEPTIDEK")["matrix_global"]
    with_terminal = encode_peptidoform("PEPTIDEK", add_terminal_composition=True)["matrix_global"]
    assert len(without) == 55
    assert len(with_terminal) == GLOBAL_DIM
    # The shorter vector is a prefix of the longer one.
    np.testing.assert_allclose(with_terminal[:55], without)


def test_dataset_passes_terminal_composition_through():
    """The dataset honours the flag, and still defaults to the short vector."""
    dataset = DeepLCDataset(["PEPTIDEK", "ACDEFGHIK"], add_terminal_composition=True)
    features, _ = dataset[0]
    assert features[2].shape == (GLOBAL_DIM,)

    default = DeepLCDataset(["PEPTIDEK"])
    features, _ = default[0]
    assert features[2].shape == (55,)


def test_predict_builds_features_the_model_needs(tmp_path):
    """
    ``predict`` must consult the model before encoding.

    The dataset default produces a 55-wide global vector, so a model needing 67
    would fail unless its feature specification is honoured.
    """
    _, path = _write_described(tmp_path)
    out = core.predict(["PEPTIDEK", "LGEYGFQNALIVR"], model=path, return_matrix=True)
    assert out.shape == (2, 5)
    assert np.isfinite(out).all()

    single = core.predict(["PEPTIDEK"], model=path)
    assert single.shape == (1,)


def test_predictions_are_deterministic(tmp_path):
    """Repeated calls on the same input return identical values."""
    _, path = _write_described(tmp_path)
    first = core.predict(["PEPTIDEK", "ACDEFGHIK"], model=path, return_matrix=True)
    second = core.predict(["PEPTIDEK", "ACDEFGHIK"], model=path, return_matrix=True)
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# integration: saving, device handling, and the bundled model
# --------------------------------------------------------------------------- #


def test_public_save_model_round_trips(tmp_path):
    """
    ``save_model`` must produce a file ``predict`` can read back.

    Saving a bare state dict for this architecture produced a checkpoint the
    loader could not rebuild: with no recorded architecture it fell back to
    inferring one from tensor names and failed on ``heads.b2``.
    """
    _, path = _write_described(tmp_path)
    model = _model_ops.load_model(path, device="cpu")

    copy_path = tmp_path / "copy.pt"
    core.save_model(model, copy_path)

    reloaded = _model_ops.load_model(copy_path, device="cpu")
    assert isinstance(reloaded, FlexCNNMultitaskModel)
    assert reloaded.feature_spec["add_terminal_composition"] is True
    assert reloaded.task_names == model.task_names

    batch = make_batch([13, 22], seed=5)
    with torch.no_grad():
        torch.testing.assert_close(model(*batch), reloaded(*batch))

    # And through the public API, which is how the failure was first seen.
    out = core.predict(["PEPTIDEK"], model=copy_path)
    assert out.shape == (1,)


def test_single_column_request_does_not_evaluate_every_task(tmp_path):
    """
    ``return_matrix=False`` must ask the model for one column, not all of them.

    A model trained on thousands of setups would otherwise build a column per
    setup and discard all but one; at a million peptides that intermediate is
    tens of gigabytes.
    """
    _, path = _write_described(tmp_path)
    model = _model_ops.load_model(path, device="cpu")

    seen = {}
    original_forward = type(model).forward

    def spy(self, *args, task_idx=None, **kwargs):
        seen["task_idx"] = task_idx
        return original_forward(self, *args, task_idx=task_idx, **kwargs)

    monkey = type(model)
    monkey.forward = spy
    try:
        single = core.predict(["PEPTIDEK", "ACDEFGHIK"], model=model)
    finally:
        monkey.forward = original_forward

    assert single.shape == (2,)
    assert seen["task_idx"] is not None, "predict should have requested a task subset"
    assert len(seen["task_idx"]) == 1


def test_matrix_request_still_returns_every_task(tmp_path):
    """Asking for the matrix must not be narrowed by the single-column shortcut."""
    _, path = _write_described(tmp_path)
    out = core.predict(["PEPTIDEK"], model=path, return_matrix=True)
    assert out.shape == (1, 5)


def test_requested_device_is_used_for_loading(tmp_path, monkeypatch):
    """
    The device from ``predict_kwargs`` must reach ``load_model``.

    Loading first onto the default device and moving afterwards wastes a copy and
    can fail with a GPU out-of-memory error for a caller who explicitly asked for
    CPU.
    """
    _, path = _write_described(tmp_path)
    seen = {}
    real_load = _model_ops.load_model

    def spy(model, device=None):
        seen["device"] = device
        return real_load(model, device=device)

    monkeypatch.setattr(_model_ops, "load_model", spy)
    core.predict(["PEPTIDEK"], model=path, predict_kwargs={"device": "cpu"})
    assert seen["device"] == "cpu"


def test_add_task_head_trains_only_the_new_setup(tmp_path):
    """
    Adapting to a setup must cost rank + 2 parameters and freeze everything else.

    This is the architecture's reason for existing, so the count is asserted rather
    than assumed.
    """
    _, path = _write_described(tmp_path)
    model = _model_ops.load_model(path, device="cpu")
    total = sum(p.numel() for p in model.parameters())

    trainable = model.add_task_head(targets=torch.tensor([5.0, 10.0, 15.0, 20.0]))
    assert trainable == SMALL["rank"] + 2
    assert trainable < total
    assert all(not p.requires_grad for p in model.encoder.parameters())

    # Output collapses to one column for the new setup, so the training loop and
    # predict() need no special case.
    with torch.no_grad():
        out = model(*make_batch([12, 20]))
    assert out.shape == (2, 1)


def test_finetune_fits_the_low_rank_head(tmp_path):
    """
    ``finetune`` adapts a fused-trunk model instead of refusing.

    It previously raised NotImplementedError for this architecture; the low-rank
    head is now the adaptation path.
    """
    from psm_utils import PSM, PSMList

    _, path = _write_described(tmp_path)
    peptides = [
        "PEPTIDEK",
        "ACDEFGHIK",
        "LGEYGFQNALIVR",
        "TVMENFVAFVDK",
        "DAFLGSFLYEYSR",
        "YICDNQDTISSK",
        "SDKPDMAEIEK",
        "MNDPKTLLQK",
    ]
    psms = PSMList(
        psm_list=[
            PSM(peptidoform=f"{p}/2", spectrum_id=str(i), retention_time=float(10 + 3 * i))
            for i, p in enumerate(peptides)
        ]
    )
    tuned = core.finetune(
        psms,
        model=path,
        validation_split=0.25,
        train_kwargs={"epochs": 2, "device": "cpu", "show_progress": False, "batch_size": 4},
    )
    assert isinstance(tuned, FlexCNNMultitaskModel)
    assert tuned.head.has_new_task

    out = core.predict(peptides[:3], model=tuned)
    assert out.shape == (3,)
    assert np.isfinite(out).all()


def test_new_task_scale_starts_at_a_trained_magnitude(tmp_path):
    """
    The affine part must not be seeded from target minutes.

    ``scale`` multiplies a dot product in the normalised space the model was trained
    in, roughly 0 to 100, not in minutes. Seeding it with a spread measured in
    minutes overshoots by about thirtyfold; on a real setup that put the first
    prediction some 600 minutes out.
    """
    model = FlexCNNMultitaskModel(n_tasks=5, **SMALL)
    with torch.no_grad():
        model.head.scale.fill_(0.8)
        model.head.shift.fill_(3.0)
    targets = torch.tensor([10.0, 40.0, 70.0, 100.0])  # std about 39 minutes
    model.head.add_task(targets=targets)
    assert model.head.new_scale.item() == pytest.approx(0.8, abs=1e-6)
    assert model.head.new_scale.item() < targets.std().item() / 10


def test_padding_length_is_taken_from_the_feature_spec(tmp_path):
    """
    A recorded ``padding_length`` must reach the encoder.

    This architecture pools rather than flattens, so a mismatch changes the
    representation without changing any shape and would not raise.
    """
    _, path = _write_described(
        tmp_path,
        feature_spec={
            "name": "global67_terminal",
            "global_dim": GLOBAL_DIM,
            "add_terminal_composition": True,
            "add_ccs_features": False,
            "padding_length": 40,
        },
    )
    model = _model_ops.load_model(path, device="cpu")
    assert model.feature_spec["padding_length"] == 40

    dataset = DeepLCDataset(["PEPTIDEK"], add_terminal_composition=True, padding_length=40)
    features, _ = dataset[0]
    assert features[0].shape[0] == 40


def test_bundled_model_predicts_in_minutes():
    """
    The packaged checkpoint must load and predict plausible retention times.

    The synthetic round-trip cannot catch packaging drift: wrong metadata, a
    truncated file, or target scaling left in normalised units would all pass the
    other tests and fail here.
    """
    path = core.FLEXCNN_MULTITASK_MODEL
    if not path.exists():  # pragma: no cover - packaged with the distribution
        pytest.skip("bundled model not present")

    model = _model_ops.load_model(path, device="cpu")
    assert model.n_tasks == 6543
    assert model.task_names is not None
    assert len(model.task_names) == 6543
    assert model.target_units == "minutes"
    assert model.feature_spec["global_dim"] == GLOBAL_DIM

    peptides = ["LGEYGFQNALIVR", "TVMENFVAFVDK", "PEPTIDEK", "M[UNIMOD:35]NDPKTLLQK"]
    out = core.predict(peptides, model=path, return_matrix=True)
    assert out.shape == (len(peptides), 6543)
    assert np.isfinite(out).all()

    # Retention times in minutes on real gradients: negative values occur on
    # indexed scales, but nothing should be near a normalised 0-1 range or
    # implausibly late.
    assert -100.0 < out.min() < 60.0
    assert 20.0 < out.max() < 1000.0

    # Uncalibrated output reports the DeepLC 1.x to 3.x setup, not whichever sorted first.
    default_idx = list(model.task_names).index(core.DEFAULT_TASK_NAME)
    single = core.predict(peptides, model=path)
    np.testing.assert_allclose(single, out[:, default_idx], rtol=1e-5)


def test_small_reference_set_warns_and_widens_validation(tmp_path, caplog):
    """
    A reference set too small to fine-tune on must say so.

    On held-out setups, fine-tuning below roughly five hundred reference peptides was
    worse than calibration every time, once by sixty-fold, because the default
    validation split left too few PSMs to early-stop against.
    """
    import logging

    from psm_utils import PSM, PSMList

    _, path = _write_described(tmp_path)
    peptides = [
        "PEPTIDEK",
        "ACDEFGHIK",
        "LGEYGFQNALIVR",
        "TVMENFVAFVDK",
        "DAFLGSFLYEYSR",
        "YICDNQDTISSK",
        "SDKPDMAEIEK",
        "MNDPKTLLQK",
    ]
    psms = PSMList(
        psm_list=[
            PSM(peptidoform=f"{p}/2", spectrum_id=str(i), retention_time=float(10 + 3 * i))
            for i, p in enumerate(peptides)
        ]
    )

    with caplog.at_level(logging.WARNING, logger="deeplc.core"):
        core.finetune(
            psms,
            model=path,
            train_kwargs={"epochs": 2, "device": "cpu", "show_progress": False, "batch_size": 4},
        )
    assert any("reference PSMs" in r.getMessage() for r in caplog.records)


def test_adapter_output_layer_is_anchored():
    """
    Solving the adapter's output layer must put it on the retention-time axis.

    The adapter's ReLU stack is largely dead at its default initialisation, so before
    this solve its output was near zero for every peptide, and the activations
    reaching the output layer are rank deficient. A CUDA least-squares driver
    requires full rank and returns non-finite values on such a system, which is why
    the solve has to be done with a rank-tolerant method.
    """
    model = DeepLCModel(n_heads=4)
    model.add_adapter(hidden_size=32)
    model.eval()

    lengths = [8, 12, 20, 31, 44, 7]
    x_atom, x_atom_sum, x_global, one_hot = _deeplc_batch(lengths)
    targets = torch.tensor([20.0, 35.0, 55.0, 80.0, 110.0, 15.0])

    with torch.no_grad():
        before = model(x_atom, x_atom_sum, x_global, one_hot).reshape(-1)
    assert before.max() - before.min() < 5.0, "expected a near-constant start"

    applied = model.solve_adapter_output(x_atom, x_atom_sum, x_global, one_hot, targets)
    assert applied

    with torch.no_grad():
        after = model(x_atom, x_atom_sum, x_global, one_hot).reshape(-1)
    # The solve cannot fit dead activations perfectly, but it must at least land on
    # the right axis rather than near zero.
    assert after.mean().item() > 5.0
    assert abs(after.mean().item() - targets.mean().item()) < 25.0


def _deeplc_batch(lengths):
    """Four-branch features for the released architecture."""
    rng = np.random.RandomState(2)
    batch = len(lengths)
    x_atom = np.zeros((batch, MAXLEN, N_ATOMS), dtype=np.float32)
    x_atom_sum = np.zeros((batch, 30, N_ATOMS), dtype=np.float32)
    one_hot = np.zeros((batch, MAXLEN, N_RESIDUES), dtype=np.float32)
    for i, length in enumerate(lengths):
        x_atom[i, :length] = rng.randint(0, 12, size=(length, N_ATOMS))
        x_atom_sum[i, : max(1, length // 2)] = rng.randint(
            0, 20, size=(max(1, length // 2), N_ATOMS)
        )
        residues = rng.randint(0, N_RESIDUES, size=length)
        one_hot[i, np.arange(length), residues] = 1.0
    # The four-branch model reads the 55-dimensional global vector, not the 67-
    # dimensional one the fused trunk needs.
    x_global = rng.randn(batch, 55).astype(np.float32)
    return (
        torch.from_numpy(x_atom),
        torch.from_numpy(x_atom_sum),
        torch.from_numpy(x_global),
        torch.from_numpy(one_hot),
    )


def test_training_scores_its_starting_point(tmp_path):
    """
    Training must not return a model worse than the one it started from.

    With the best validation loss left at infinity the first epoch always became the
    best, however bad, so a fine-tune on a small reference set could hand back a
    collapsed fit at ninety times the error of its own starting point.
    """
    import inspect

    from deeplc import _model_ops

    source = inspect.getsource(_model_ops.train)
    assert 'best_val_loss = float("inf")' not in source
    assert "_validate_epoch(model, val_loader, loss_fn, device)" in source


# --------------------------------------------------------------------------- #
# length-bucketed prediction
# --------------------------------------------------------------------------- #


def test_padding_reach_counts_the_convolutions():
    """The reach is the sum over convolutions of ``(kernel - 1) // 2``, dilation aside."""
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL)
    # Two pointwise stem layers reach nothing; two kernel-5 convolutions reach two each.
    assert model.padding_reach == 4
    assert FlexCNNMultitaskModel(n_tasks=3, **{**SMALL, "kernel_size": 3}).padding_reach == 2


def test_length_buckets_predict_what_the_full_window_predicts():
    """
    Running short peptides in a short window must not change their predictions.

    The trunk masks by the true residue count and the features do not depend on the
    window, so a window of the longest peptide plus the trunk's reach holds everything
    that can influence a valid position. That exactness is what allows the prediction
    path to stop padding every peptide to sixty positions.
    """
    model = FlexCNNMultitaskModel(n_tasks=5, **SMALL).eval()
    peptides = ["PEPTIDEK", "ACDEFGHIK", "PEPTIDEPEPTIDEPEPTIDEK", "ACDK", "SEQUENCEWITHK"]
    dataset = DeepLCDataset(peptides, add_terminal_composition=True)

    full = _model_ops.predict(
        model=model,
        data=dataset,
        device="cpu",
        batch_size=2,
        show_progress=False,
        length_buckets=False,
    )
    bucketed = _model_ops.predict(
        model=model,
        data=dataset,
        device="cpu",
        batch_size=2,
        show_progress=False,
    )
    assert bucketed.shape == full.shape
    assert torch.allclose(bucketed, full, atol=1e-5)


def test_length_buckets_are_skipped_when_they_cannot_help():
    """A dataset whose longest peptide fills the window is left as one pass."""
    model = FlexCNNMultitaskModel(n_tasks=2, **SMALL).eval()
    short_window = DeepLCDataset(["PEPTIDEK"], add_terminal_composition=True, padding_length=10)
    assert _model_ops._length_buckets(model, short_window, batch_size=8) is None
    # A four-branch model pools across positions and reports no reach, so it never buckets.
    plain = DeepLCModel(n_heads=3)
    assert getattr(plain, "padding_reach", None) is None


def test_batch_encoding_matches_item_by_item():
    """A batch encoded in one pass must equal the per-peptide encoding a DataLoader stacks."""
    peptides = ["PEPTIDEK", "ACDEFGHIK", "SEQUENCEWITHK", "ACDK"]
    dataset = DeepLCDataset(peptides, add_terminal_composition=True)
    batch = dataset.encode_batch(range(len(peptides)))
    for position in range(4):
        stacked = torch.stack([dataset[i][0][position] for i in range(len(peptides))])
        assert torch.allclose(batch[position], stacked, atol=0)


def test_rolling_sum_is_skipped_for_the_fused_trunk():
    """
    The fused trunk deletes the rolling sum, so encoding does not build it.

    It stays for the four-branch model, which reads it, and the flag travels with the
    dataset rather than being decided inside the encoder.
    """
    assert FlexCNNMultitaskModel.uses_rolling_sum is False
    assert getattr(DeepLCModel(n_heads=2), "uses_rolling_sum", True) is True

    with_sum = DeepLCDataset(["PEPTIDEK"], add_terminal_composition=True)
    without = DeepLCDataset(["PEPTIDEK"], add_terminal_composition=True, include_rolling_sum=False)
    assert with_sum[0][0][1].shape[0] > 0
    assert without[0][0][1].shape[0] == 0
    # every other feature is untouched by the flag
    for position in (0, 2, 3):
        assert torch.allclose(with_sum[0][0][position], without[0][0][position], atol=0)
