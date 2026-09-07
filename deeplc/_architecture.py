"""
PyTorch architecture definitions for DeepLC.

This module contains the neural network architectures used by DeepLC for predicting peptide
retention times based on atomic composition and other features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LeakyReLUSaturation(nn.Module):
    """
    Leaky ReLU activation with saturation (max value clipping).

    This custom activation function applies leaky ReLU followed by clamping to a maximum value,
    matching the original TensorFlow implementation's behavior.

    Parameters
    ----------
    negative_slope
        Negative slope coefficient for leaky ReLU (default: 0.1)
    max_value
        Maximum output value for saturation (default: 20.0)

    """

    def __init__(self, negative_slope: float = 0.1, max_value: float = 20.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.max_value = max_value
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(x)
        return torch.clamp(x, max=self.max_value)


class ConvBlock(nn.Module):
    """
    Convolutional block with two Conv1D layers and optional max pooling.

    Parameters
    ----------
    in_channels
        Number of input channels
    out_channels
        Number of output channels
    kernel_size
        Size of the convolutional kernel
    use_pooling
        Whether to apply max pooling after convolutions
    pool_size
        Size of the max pooling window (default: 2)
    regularizer_val
        L1 regularization coefficient (default: 0.000005)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_pooling: bool = True,
        pool_size: int = 2,
        regularizer_val: float = 0.000005,
    ):
        super().__init__()
        self.use_pooling = use_pooling
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation1 = LeakyReLUSaturation()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation2 = LeakyReLUSaturation()

        self.pool: nn.MaxPool1d | None = None
        if use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # Store regularizer value for potential use in training
        self.regularizer_val = regularizer_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class GlobalFeatureBranch(nn.Module):
    """
    Dense network branch for processing global peptide features.

    Parameters
    ----------
    input_size
        Size of the input feature vector
    num_layers
        Number of dense layers
    hidden_size
        Layer size for each hidden layer
    regularizer_val
        L1 regularization coefficient (default: 0.000005)

    """

    def __init__(
        self,
        input_size: int,
        num_layers: int = 4,
        hidden_size: int = 64,
        regularizer_val: float = 0.000005,
    ):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(input_size if len(layers) == 0 else hidden_size, hidden_size),
                    LeakyReLUSaturation(),
                ]
            )

        self.network = nn.Sequential(*layers)
        self.regularizer_val = regularizer_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OneHotBranch(nn.Module):
    """
    Convolutional branch for processing one-hot encoded amino acid sequences.

    This branch uses tanh activation instead of leaky ReLU and processes one-hot encoded amino
    acid features.

    Parameters
    ----------
    input_channels
        Number of input channels (20 for standard amino acids)
    sequence_length
        Length of the input sequence
    kernel_size
        Size of the convolutional kernel (default: 2)

    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        kernel_size: int = 2,
    ):
        super().__init__()

        # Use 'same' padding to maintain sequence length through convolutions
        self.conv1 = nn.Conv1d(
            input_channels,
            2,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation1 = nn.Tanh()

        self.conv2 = nn.Conv1d(
            2,
            2,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation2 = nn.Tanh()

        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one-hot encoded amino acid features."""
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class BatchedHeads(nn.Module):
    """
    Parallel output heads sharing a hidden projection.

    Each head maps the shared trunk output to a scalar via a two-step computation: a batched
    linear projection followed by a per-head dot product with a learned weight vector.

    Parameters
    ----------
    input_size
        Size of the input feature vector (output of shared trunk).
    n_heads
        Number of parallel output heads.
    hidden
        Hidden dimension per head (default: 32).

    """

    def __init__(self, input_size: int, n_heads: int, hidden: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(input_size, n_heads * hidden)
        self.w2 = nn.Parameter(torch.zeros(n_heads, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_heads))
        nn.init.normal_(self.w2, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        n_heads = self.b2.shape[0]
        h = torch.relu(h.view(h.shape[0], n_heads, h.shape[1] // n_heads))
        return (h * self.w2.unsqueeze(0)).sum(dim=-1) + self.b2  # (batch, n_heads)


class DeepLCModel(nn.Module):
    """
    DeepLC model for peptide retention time prediction.

    Four parallel input branches — per-position atomic composition CNN, summed atomic composition
    CNN, global feature dense network, and one-hot amino acid CNN — are concatenated and passed
    through a shared dense trunk. Outputs are projected by :class:`BatchedHeads` to ``[batch,
    n_heads]``.

    When ``n_heads > 1`` the model is a multitask backbone trained across multiple LC setups.
    Call :meth:`add_adapter` to attach a fine-tuning MLP that maps the head vector to a single
    RT value ``[batch, 1]``.


    Parameters
    ----------
    n_heads
        Number of parallel output heads (default: 1).
    atom_sequence_length
        Length of the atomic composition sequence (default: 60)
    atom_channels
        Number of atomic feature channels (default: 6 for C,H,N,O,S,P)
    atom_sum_sequence_length
        Length of the summed atomic composition sequence (default: 30)
    global_feature_size
        Size of the global feature vector (default: 55)
    one_hot_sequence_length
        Length of the one-hot encoded sequence (default: 60)
    one_hot_channels
        Number of amino acid types for one-hot encoding (default: 20)
    one_hot_kernel_size
        Kernel size for one-hot branch convolutions (default: 2)
    atom_cnn_blocks
        Number of convolutional blocks in the atomic branch (default: 3)
    atom_cnn_kernel_size
        Kernel size for atomic branch convolutions (default: 5)
    atom_cnn_filters_start
        Starting number of filters in atomic branch (default: 256)
    atom_cnn_pool_size
        Max pooling size for atomic branch (default: 2)
    sum_cnn_blocks
        Number of convolutional blocks in the summed atomic branch (default: 3)
    sum_cnn_kernel_size
        Kernel size for summed atomic branch convolutions (default: 5)
    sum_cnn_filters_start
        Starting number of filters in summed atomic branch (default: 256)
    global_layer_size
        Layer size for global feature layers (default: 64)
    global_num_layers
        Number of dense layers in global branch (default: 4)
    final_layer_size
        Layer size for final dense layers (default: 128)
    final_num_layers
        Number of final dense layers (default: 5)
    regularizer_val
        L1 regularization coefficient (default: 0.000005)

    """

    def __init__(
        self,
        n_heads: int = 1,
        atom_sequence_length: int = 60,
        atom_channels: int = 6,
        atom_sum_sequence_length: int = 30,
        global_feature_size: int = 55,
        one_hot_sequence_length: int = 60,
        one_hot_channels: int = 20,
        one_hot_kernel_size: int = 2,
        atom_cnn_blocks: int = 3,
        atom_cnn_kernel_size: int = 5,
        atom_cnn_filters_start: int = 256,
        atom_cnn_pool_size: int = 2,
        sum_cnn_blocks: int = 3,
        sum_cnn_kernel_size: int = 5,
        sum_cnn_filters_start: int = 256,
        global_layer_size: int = 64,
        global_num_layers: int = 4,
        final_layer_size: int = 128,
        final_num_layers: int = 4,
        regularizer_val: float = 0.000005,
    ):
        super().__init__()

        # Branch A: Atomic composition CNN
        a_layers: list[nn.Module] = []
        in_channels = atom_channels
        for block_idx in range(atom_cnn_blocks):
            out_channels = int(atom_cnn_filters_start / (2**block_idx))
            use_pooling = block_idx < (atom_cnn_blocks - 1)
            a_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    atom_cnn_kernel_size,
                    use_pooling=use_pooling,
                    pool_size=atom_cnn_pool_size,
                    regularizer_val=regularizer_val,
                )
            )
            in_channels = out_channels
        self.branch_a = nn.Sequential(*a_layers, nn.Flatten())

        # Branch B: Summed atomic composition CNN
        b_layers: list[nn.Module] = []
        in_channels = atom_channels
        for block_idx in range(sum_cnn_blocks):
            out_channels = int(sum_cnn_filters_start / (2**block_idx))
            use_pooling = block_idx < (sum_cnn_blocks - 1)
            b_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    sum_cnn_kernel_size,
                    use_pooling=use_pooling,
                    pool_size=2,
                    regularizer_val=regularizer_val,
                )
            )
            in_channels = out_channels
        self.branch_b = nn.Sequential(*b_layers, nn.Flatten())

        # Branch C: Global features
        self.branch_c = GlobalFeatureBranch(
            global_feature_size,
            num_layers=global_num_layers,
            hidden_size=global_layer_size,
            regularizer_val=regularizer_val,
        )

        # Branch D: One-hot encoding
        self.branch_d = OneHotBranch(
            one_hot_channels, one_hot_sequence_length, kernel_size=one_hot_kernel_size
        )

        # Compute concatenated feature size via a dummy forward pass
        with torch.no_grad():
            concat_size = (
                self.branch_a(torch.zeros(1, atom_channels, atom_sequence_length)).shape[1]
                + self.branch_b(torch.zeros(1, atom_channels, atom_sum_sequence_length)).shape[1]
                + self.branch_c(torch.zeros(1, global_feature_size)).shape[1]
                + self.branch_d(torch.zeros(1, one_hot_channels, one_hot_sequence_length)).shape[1]
            )

        # Shared trunk: dense layers without output linear
        trunk_layers: list[nn.Module] = []
        for i in range(final_num_layers):
            in_features = concat_size if i == 0 else final_layer_size
            trunk_layers.extend([nn.Linear(in_features, final_layer_size), LeakyReLUSaturation()])
        self.shared_trunk = nn.Sequential(*trunk_layers)

        # Output heads
        self.heads = BatchedHeads(final_layer_size, n_heads)

        # Optional fine-tuning adapter (None until add_adapter() is called)
        self.adapter: nn.Module | None = None

        self._initialize_weights()

    def add_adapter(self, hidden_size: int = 256) -> None:
        """
        Attach a fine-tuning adapter mapping the head vector to one RT output.

        Adapter parameters are left at PyTorch default initialization and trained from scratch
        during fine-tuning.
        """
        n_heads = self.heads.b2.shape[0]
        self.adapter = nn.Sequential(
            nn.Linear(n_heads, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_size // 2), 1),
        )
        self.adapter.to(self.heads.b2.device)

    @torch.no_grad()
    def solve_adapter_output(
        self,
        x_atom: torch.Tensor,
        x_atom_sum: torch.Tensor,
        x_global: torch.Tensor,
        x_one_hot: torch.Tensor,
        targets: torch.Tensor,
    ) -> bool:
        """
        Set the adapter's output layer by least squares on the reference data.

        The adapter is otherwise trained from a default initialisation, so its output
        begins unrelated to minutes and, on a small reference set, can settle on a
        degenerate fit that predicts every peptide near the mean retention time. Its
        final layer is linear in its input, so the best output weights and bias for
        the current earlier layers follow in closed form.

        Returns True when the solve was applied. Called after :meth:`add_adapter` and
        before training.

        Parameters
        ----------
        x_atom, x_atom_sum, x_global, x_one_hot
            Encoded reference peptidoforms.
        targets
            Their observed retention times.

        """
        adapter = getattr(self, "adapter", None)
        if adapter is None or not isinstance(adapter[-1], nn.Linear):
            return False

        was_training = self.training
        self.eval()
        try:
            x_atom_t = x_atom.transpose(1, 2)
            x_atom_sum_t = x_atom_sum.transpose(1, 2)
            x_one_hot_t = x_one_hot.transpose(1, 2)
            concatenated = torch.cat(
                [
                    self.branch_a(x_atom_t),
                    self.branch_b(x_atom_sum_t),
                    self.branch_c(x_global),
                    self.branch_d(x_one_hot_t),
                ],
                dim=1,
            )
            head_vector = self.heads(self.shared_trunk(concatenated))
            # Everything up to, but not including, the output layer.
            penultimate = head_vector
            for layer in list(adapter)[:-1]:
                penultimate = layer(penultimate)
        finally:
            if was_training:
                self.train()

        features = penultimate.detach().double()
        y = targets.detach().reshape(-1).double()
        if features.shape[0] != y.shape[0] or features.shape[0] < 3:
            return False

        design = torch.cat([features, torch.ones_like(features[:, :1])], dim=1)
        # Solved on the CPU with a rank-revealing driver, and ridge-regularised. The
        # adapter's ReLU stack is largely dead at its default initialisation, so the
        # activations arriving here are often rank deficient; the default CUDA driver
        # requires full rank and returns non-finite values on such a system, which is
        # how this solve silently declined to apply.
        design = design.cpu()
        target = y.unsqueeze(1).cpu()
        ridge = 1e-6 * torch.eye(design.shape[1], dtype=design.dtype)
        try:
            gram = design.T @ design + ridge
            solution = torch.linalg.solve(gram, design.T @ target).reshape(-1)
        except Exception:  # noqa: BLE001 - a singular system leaves the init alone
            return False
        if not torch.isfinite(solution).all():
            return False
        solution = solution.to(features.device)

        output = adapter[-1]
        output.weight.copy_(solution[:-1].reshape(1, -1).to(output.weight.dtype))
        if output.bias is not None:
            output.bias.copy_(solution[-1].reshape(1).to(output.bias.dtype))
        return True

    def freeze_backbone(self) -> None:
        """Freeze all parameters except the adapter."""
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("adapter.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_atom: torch.Tensor,
        x_atom_sum: torch.Tensor,
        x_global: torch.Tensor,
        x_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, n_heads]``, or ``[batch, 1]`` when an adapter is attached.

        """
        x_atom = x_atom.transpose(1, 2)
        x_atom_sum = x_atom_sum.transpose(1, 2)
        x_one_hot = x_one_hot.transpose(1, 2)
        concatenated = torch.cat(
            [
                self.branch_a(x_atom),
                self.branch_b(x_atom_sum),
                self.branch_c(x_global),
                self.branch_d(x_one_hot),
            ],
            dim=1,
        )
        out = self.heads(self.shared_trunk(concatenated))  # [batch, n_heads]
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            return adapter(out)  # [batch, 1]
        return out


# ---------------------------------------------------------------------------
# Fused-trunk multitask architecture
# ---------------------------------------------------------------------------
#
# ``MultitaskDeepLCModel`` above runs four independent branches over four feature
# arrays and concatenates the flattened results, which is what DeepLC has always
# done. The architecture below instead fuses atomic composition with a learned
# residue embedding into a single convolutional trunk and pools it to a fixed
# vector. On the same training data and the same head it reaches roughly a third
# of the error of the four-branch backbone, and it is length-agnostic because it
# pools rather than flattens.


class _ConvSiLU(nn.Module):
    """A single ``Conv1d`` with ``same`` padding followed by SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve and activate ``(batch, channels, length)``."""
        return self.act(self.conv(x))


def _pointwise_stem(channels: int, layers: int, in_channels: int = 6) -> nn.Sequential:
    """
    Stack of width-1 convolutions decoding each position's atom counts alone.

    Atomic composition determines the residue uniquely except for the Leu/Ile
    pair, so this is the layer that can recover identity from the atom matrix
    before any neighbour mixing happens. Recovering a residue from six counts is
    a lookup rather than a linear map, so one layer is generally not enough.

    Returned as a bare ``Sequential`` rather than wrapped in a module, so the
    parameter names stay ``stem.0.conv.weight`` and match the trained checkpoint.
    """
    return nn.Sequential(
        *[
            _ConvSiLU(in_channels if i == 0 else channels, channels, kernel_size=1)
            for i in range(layers)
        ]
    )


class InputNorm(nn.Module):
    """
    Per-feature standardisation with buffers fitted on the training rows.

    The dense feature vector mixes raw atom counts, sequence length and
    positional compositions, whose scales differ by an order of magnitude.

    A feature that never varies during training is left in raw units rather than
    having its standard deviation clamped to a floor. Clamping would multiply any
    non-zero test value by one over that floor: with phosphorus absent from
    training, a single phosphate standardised to a value near a thousand and
    destroyed the forward pass. Constant features must not become high-gain
    inputs.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    @torch.no_grad()
    def fit(self, values: torch.Tensor) -> None:
        """Set the buffers from ``(n_rows, n_features)`` of training features."""
        self.mean.copy_(values.mean(dim=0))
        std = values.std(dim=0)
        self.std.copy_(torch.where(std < 1e-3, torch.ones_like(std), std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise ``(batch, n_features)``."""
        return (x - self.mean) / self.std


class FactorHead(nn.Module):
    """
    Low-rank multitask head: a per-setup embedding dotted with a shared trunk.

    ``pred[:, j] = (proj(trunk) . embedding[j]) * scale[j] + shift[j]``

    Unlike :class:`BatchedHeads`, which gives every LC setup its own hidden
    projection, the projection here is shared and only ``rank + 2`` parameters
    belong to a setup. Adding a setup therefore means fitting 66 values at
    rank 64, with the encoder frozen, rather than retraining a head.

    ``scale`` and ``shift`` are an affine map on that setup's output. Training
    normalises each setup's retention times to a fixed range, so a raw training
    checkpoint predicts in that normalised space; because both transforms are
    affine they compose, and packaging folds the normalisation into ``scale`` and
    ``shift`` so that a shipped model returns minutes directly. Whether that has
    happened is recorded as ``target_units`` in the checkpoint rather than
    assumed.

    Parameters
    ----------
    trunk_dim
        Width of the shared trunk output.
    n_tasks
        Number of LC setups the model was trained on.
    rank
        Size of the per-setup embedding.

    """

    def __init__(self, trunk_dim: int, n_tasks: int, rank: int = 64):
        super().__init__()
        self.n_tasks = n_tasks
        self.rank = rank
        self.proj = nn.Linear(trunk_dim, rank)
        self.embedding = nn.Parameter(torch.zeros(n_tasks, rank))
        self.scale = nn.Parameter(torch.ones(n_tasks))
        self.shift = nn.Parameter(torch.zeros(n_tasks))

    def add_task(self, targets: torch.Tensor | None = None, init_from: int | None = None) -> None:
        """
        Attach parameters for one new LC setup and switch to single-task output.

        The new setup gets its own ``rank + 2`` parameters as separate tensors rather
        than extra rows in the existing ones, so freezing the pretrained setups is a
        matter of ``requires_grad`` and does not need per-row gradient masking.

        The embedding starts at the mean of the trained setups, which is the least
        committed starting point available.

        ``scale`` and ``shift`` start from the mean of the trained setups rather than
        from the target retention times. Setting them directly from the targets looks
        natural and is wrong by more than an order of magnitude: the dot product they
        multiply is in the normalised space the model was trained in, roughly 0 to 100,
        not in minutes, so seeding ``scale`` with a spread measured in minutes
        overshoots by about thirty-fold and the fit starts hundreds of minutes away.
        The trained setups' own values are already the right magnitude for mapping that
        dot product onto a real gradient.

        Parameters
        ----------
        targets
            Observed retention times for the new setup. Used only to nudge the
            initial offset toward the right part of the gradient.
        init_from
            Index of an existing setup to copy from, instead of the mean. Useful when
            a similar gradient is known.

        """
        with torch.no_grad():
            if init_from is None:
                start = self.embedding.mean(dim=0, keepdim=True).clone()
                scale = self.scale.mean().reshape(1).clone()
                shift = self.shift.mean().reshape(1).clone()
            else:
                start = self.embedding[init_from : init_from + 1].clone()
                scale = self.scale[init_from].reshape(1).clone()
                shift = self.shift[init_from].reshape(1).clone()

            if targets is not None and targets.numel() > 1:
                # Centre the offset on the observed gradient while leaving the slope
                # at a trained magnitude, so the fit starts on the right window.
                shift = shift + (targets.mean() - shift)

            self.new_embedding = nn.Parameter(start)
            self.new_scale = nn.Parameter(scale)
            self.new_shift = nn.Parameter(shift)

    def project_new_task(self, trunk: torch.Tensor) -> torch.Tensor:
        """Return the new setup's prediction before scale and shift are applied."""
        return (self.proj(trunk) @ self.new_embedding.t()).squeeze(-1)

    @torch.no_grad()
    def solve_new_task_affine(self, projected: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Set ``scale`` and ``shift`` by least squares instead of learning them.

        The two are linear in the prediction, so the best values for a given
        embedding follow in closed form and do not need gradient descent. Leaving
        them to the optimiser is how fine-tuning fails on a small reference set: with
        few points the scale decays toward zero and every prediction collapses onto
        the mean retention time. On one 133-minute gradient with 230 reference
        peptides that produced a 17-minute output range and a 91-minute error, while
        the correlation stayed above 0.9 because the ordering was never the problem.

        Solving them first also gives the embedding a sane starting error to descend
        from, rather than one dominated by a mis-scaled output.
        """
        x = projected.detach().reshape(-1).double()
        y = targets.detach().reshape(-1).double()
        if x.numel() < 3 or torch.std(x) < 1e-9:
            return
        design = torch.stack([x, torch.ones_like(x)], dim=1)
        solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.reshape(-1)
        slope, intercept = solution[0], solution[1]
        if not torch.isfinite(slope) or not torch.isfinite(intercept):
            return
        self.new_scale.copy_(slope.reshape(1).to(self.new_scale.dtype))
        self.new_shift.copy_(intercept.reshape(1).to(self.new_shift.dtype))

    def freeze_pretrained(self) -> None:
        """Freeze everything except the newly added setup's parameters."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for name in ("new_embedding", "new_scale", "new_shift"):
            parameter = getattr(self, name, None)
            if parameter is not None:
                parameter.requires_grad = True

    @property
    def has_new_task(self) -> bool:
        """Whether :meth:`add_task` has been called."""
        return getattr(self, "new_embedding", None) is not None

    def forward(self, trunk: torch.Tensor, task_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Map trunk output to one prediction per task.

        Parameters
        ----------
        trunk
            Shape ``(batch, trunk_dim)``.
        task_idx
            Optional task indices to evaluate. Without it every task is
            returned, which for a model trained on thousands of setups is a wide
            matrix; pass the subset when only a few setups are of interest.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_tasks)``, or ``(batch, len(task_idx))``.

        """
        projected = self.proj(trunk)
        if self.has_new_task:
            # Fine-tuned onto one setup: return that column alone, so the shape
            # matches a single-output model and the training loop needs no change.
            return (projected @ self.new_embedding.t()) * self.new_scale + self.new_shift
        if task_idx is None:
            return projected @ self.embedding.t() * self.scale + self.shift
        return (projected @ self.embedding[task_idx].t()) * self.scale[task_idx] + self.shift[
            task_idx
        ]


class FlexCNNMultitaskModel(nn.Module):
    """
    Multitask RT model fusing atom composition and residue identity in one trunk.

    Atom counts pass through a pointwise stem, are concatenated with a learned
    residue embedding, and the result runs through a convolutional stack that is
    masked and pooled over the valid length. The pooled vector is concatenated
    with the global feature vector and the residue counts, and a small MLP feeds
    :class:`FactorHead`.

    The forward signature matches :class:`MultitaskDeepLCModel` so the two are
    interchangeable in the prediction path, but ``x_atom_sum`` is unused: the
    convolutional trunk sees the per-position matrix directly, which makes the
    rolling-sum array redundant.

    The global feature vector must be the 67-dimensional form produced with
    ``add_terminal_composition=True``; the 55-dimensional default will fail on
    shape at the first dense layer.

    Parameters
    ----------
    n_tasks
        Number of LC setups.
    global_dim
        Length of the global feature vector.
    embed_dim
        Width of the residue embedding.
    channels
        Output channels of each convolution stage.
    kernel_size
        Convolution width.
    stem_channels
        Width of the pointwise stem, or 0 to feed raw atom counts.
    stem_layers
        Number of pointwise stem layers.
    width
        Width of the dense trunk.
    depth
        Number of dense layers.
    rank
        Per-setup embedding size in the head.

    """

    #: Index reserved for padding positions in the residue encoding.
    PAD_INDEX = 20

    def __init__(
        self,
        n_tasks: int,
        global_dim: int = 67,
        embed_dim: int = 16,
        channels: tuple[int, ...] = (512, 512),
        kernel_size: int = 5,
        stem_channels: int = 128,
        stem_layers: int = 2,
        width: int = 256,
        depth: int = 3,
        rank: int = 64,
    ):
        super().__init__()
        # Kept so the model can describe itself when saved, rather than leaving
        # the serialiser to know how each architecture is constructed.
        self._encoder_kwargs = {
            "global_dim": global_dim,
            "embed_dim": embed_dim,
            "channels": tuple(channels),
            "kernel_size": kernel_size,
            "stem_channels": stem_channels,
            "stem_layers": stem_layers,
            "width": width,
            "depth": depth,
        }
        self._head_kwargs = {"rank": rank}
        self.n_tasks = n_tasks
        #: Feature layout this instance expects. Replaced by the value recorded in
        #: a checkpoint when one is loaded.
        self.feature_spec: dict | None = {
            "name": "global67_terminal" if global_dim == 67 else f"global{global_dim}",
            "global_dim": global_dim,
            "add_terminal_composition": global_dim == 67,
            "add_ccs_features": False,
            "padding_length": 60,
            # Trained after the 4.0.1 correction to positional modification deltas,
            # so it wants the corrected placement rather than the compatibility
            # default DeepLCDataset applies for checkpoints that declare nothing.
            "legacy_positional_deltas": False,
        }
        self.target_units: str | None = None
        self.task_names: list[str] | None = None

        self.encoder = _FlexCNNEncoder(
            global_dim=global_dim,
            embed_dim=embed_dim,
            channels=channels,
            kernel_size=kernel_size,
            stem_channels=stem_channels,
            stem_layers=stem_layers,
            width=width,
            depth=depth,
        )
        self.head = FactorHead(trunk_dim=width, n_tasks=n_tasks, rank=rank)

    def forward(
        self,
        x_atom: torch.Tensor,
        x_atom_sum: torch.Tensor,
        x_global: torch.Tensor,
        x_one_hot: torch.Tensor,
        task_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict retention time for every LC setup.

        Parameters
        ----------
        x_atom
            Shape ``(batch, length, 6)``, per-position atomic composition.
        x_atom_sum
            Unused; accepted so the signature matches the four-branch model.
        x_global
            Shape ``(batch, 67)``, global feature vector with terminal
            composition.
        x_one_hot
            Shape ``(batch, length, 20)``, one-hot residue encoding.
        task_idx
            Optional subset of task indices to evaluate.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_tasks)``, or ``(batch, len(task_idx))``.

        """
        del x_atom_sum  # the fused trunk reads x_atom directly
        return self.head(self.encoder(x_atom, x_global, x_one_hot), task_idx)

    @property
    def padding_reach(self) -> int | None:
        """
        How far a valid position can see across the right edge of the encoding window.

        The trunk masks its output by the true residue count and pools over that mask, and
        the features themselves do not depend on the window, so the only way padding can
        reach a valid position is through the convolutions. Each convolution of width ``k``
        and dilation ``d`` reaches ``(k - 1) // 2 * d`` positions, and the reaches add up.
        A batch encoded in a window of its longest peptide plus this many positions
        therefore predicts exactly what the full window predicts, which for a median
        peptide of sixteen residues is a fraction of the sixty positions used otherwise.

        Returns None when the trunk pools or strides across positions, because then the
        mask no longer lines up position by position and the argument does not hold.

        """
        reach = 0
        for module in self.encoder.modules():
            if isinstance(module, (nn.MaxPool1d, nn.AvgPool1d)):
                return None
            if isinstance(module, nn.Conv1d):
                if module.stride[0] != 1:
                    return None
                reach += ((module.kernel_size[0] - 1) // 2) * module.dilation[0]
        return reach

    def add_task_head(
        self, targets: torch.Tensor | None = None, init_from: int | None = None
    ) -> int:
        """
        Prepare the model to be fine-tuned onto one new LC setup.

        Adds ``rank + 2`` trainable parameters and freezes everything else, so
        adapting to a setup costs 66 values at rank 64 rather than retraining a head
        or fitting an adapter over the full head vector. Returns the number of
        trainable parameters, which callers log to make the cost visible.
        """
        self.head.add_task(targets=targets, init_from=init_from)
        self.head.freeze_pretrained()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def solve_new_task_affine(
        self, features: tuple[torch.Tensor, ...], targets: torch.Tensor
    ) -> None:
        """
        Anchor the new setup's affine parameters on the reference data.

        Call after :meth:`add_task_head` and before training, passing the reference
        features and their observed retention times.
        """
        trunk = self.encoder(features[0], features[2], features[3])
        self.head.solve_new_task_affine(self.head.project_new_task(trunk), targets)

    def describe(self) -> dict:
        """
        Return a serialisable description of this model, including its weights.

        Saving this rather than a bare state dict means a checkpoint can be
        reloaded without guessing the architecture from tensor names, and it
        keeps the knowledge of how to rebuild a model with the model rather than
        in the serialiser.
        """
        return {
            "architecture": type(self).__name__,
            "encoder_kwargs": dict(self._encoder_kwargs),
            "head_kwargs": dict(self._head_kwargs),
            "n_tasks": self.n_tasks,
            "feature_spec": self.feature_spec,
            "target_units": self.target_units,
            "task_names": self.task_names,
            "state_dict": self.state_dict(),
        }


class _FlexCNNEncoder(nn.Module):
    """Convolutional trunk shared by every LC setup in :class:`FlexCNNMultitaskModel`."""

    PAD_INDEX = 20

    def __init__(
        self,
        global_dim: int,
        embed_dim: int,
        channels: tuple[int, ...],
        kernel_size: int,
        stem_channels: int,
        stem_layers: int,
        width: int,
        depth: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(self.PAD_INDEX + 1, embed_dim, padding_idx=self.PAD_INDEX)
        self.stem = _pointwise_stem(stem_channels, stem_layers) if stem_channels > 0 else None

        in_channels = (stem_channels if stem_channels > 0 else 6) + embed_dim
        blocks = []
        for out_channels in channels:
            blocks.append(_ConvSiLU(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        # Sum and max pooling, masked to the valid length. Sum is extensive in
        # peptide length and max is not, so the pair carries both.
        pooled_dim = in_channels * 2
        self.pool_norm = nn.LayerNorm(pooled_dim)

        dense_dim = global_dim + self.PAD_INDEX
        self.norm = InputNorm(dense_dim)
        layers: list[nn.Module] = []
        sizes = [dense_dim + pooled_dim] + [width] * depth
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.trunk_dim = width

    def residue_indices(self, x_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Convert a one-hot residue matrix to integer indices.

        Padding positions are all-zero rows, for which ``argmax`` returns 0 and
        would collide with the first residue, so they are set to
        :attr:`PAD_INDEX` explicitly.
        """
        idx = x_one_hot.argmax(dim=2)
        return idx.masked_fill(x_one_hot.sum(dim=2) == 0, self.PAD_INDEX)

    def residue_counts(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Residue counts per peptidoform, shape ``(batch, 20)``.

        Built with ``scatter_add`` rather than by materialising a
        ``(batch, length, 21)`` one-hot tensor, which dominated inference time
        for every model that uses counts.
        """
        idx = idx.long().clamp(0, self.PAD_INDEX)
        counts = torch.zeros(
            idx.shape[0], self.PAD_INDEX + 1, device=idx.device, dtype=torch.float32
        )
        counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
        return counts[:, : self.PAD_INDEX]  # drop the padding column

    def forward(
        self, x_atom: torch.Tensor, x_global: torch.Tensor, x_one_hot: torch.Tensor
    ) -> torch.Tensor:
        """Encode a batch to ``(batch, trunk_dim)``."""
        idx = self.residue_indices(x_one_hot)
        valid = (idx != self.PAD_INDEX).unsqueeze(1)  # (batch, 1, length)

        atom = x_atom.float().transpose(1, 2)
        if self.stem is not None:
            atom = self.stem(atom)
        hidden = torch.cat([atom, self.embed(idx).transpose(1, 2)], dim=1)

        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden * valid

        summed = hidden.sum(dim=2)
        maxed = torch.nan_to_num(
            hidden.masked_fill(~valid, float("-inf")).max(dim=2).values, neginf=0.0
        )
        pooled = self.pool_norm(torch.cat([summed, maxed], dim=1))

        dense = torch.cat([x_global.float(), self.residue_counts(idx)], dim=1)
        return self.net(torch.cat([self.norm(dense), pooled], dim=1))
