"""A predictive-coding network with a supervised readout.

The network is generative top-down: layer ``l + 1`` predicts layer ``l`` through
``x_hat_l = f(W_l @ x_{l+1})``, and layer 0 is the clamped input. A linear
readout on the top latent produces the class scores. Learning minimises the
free energy

    F = sum_l 0.5 * ||x_l - f(W_l x_{l+1})||^2 + 0.5 * ||W_out x_L - y||^2

in two alternating phases: an inference relaxation that descends F in the
latent states, then a weight update that descends F in the parameters.

Gradients are derived analytically rather than through autograd, but they are
written to ``parameter.grad`` so any ``torch.optim`` optimizer drives the weight
phase. That keeps the module a plain ``nn.Module`` usable inside a larger
PyTorch model.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Leaky slope keeps the inference gradient alive everywhere. With hard ReLU a
# non-positive pre-activation zeroes out gain_modulated_error, which blocks the
# top-down error path entirely: starting the latents at zero leaves every layer
# above the first pinned at exactly 0.0 forever (measured). That trap is why the
# old code had to seed the latents with randn -- and that randn seed is what the
# relaxation never escaped. LeakyReLU removes the trap, so the latents can be
# initialised deterministically instead. See init_latents.
NEGATIVE_SLOPE = 0.1


def leaky_relu_deriv(a):
    """Derivative of ``LeakyReLU(NEGATIVE_SLOPE)`` at the given pre-activations.

    Args:
        a: Pre-activation tensor.

    Returns:
        torch.Tensor: Elementwise derivative, 1.0 where ``a > 0`` and
        ``NEGATIVE_SLOPE`` elsewhere.
    """
    return torch.where(a > 0, 1.0, NEGATIVE_SLOPE)


@dataclass
class PCNRunResult:
    """Result of a predictive or generative inference run.

    Attributes:
        mode: Either ``"predictive"`` or ``"generative"``.
        input_state: The visible layer x0. Clamped input in predictive mode,
            inferred in generative mode.
        states: All states ``[x0, x1, ..., xL]`` after the relaxation.
        states_by_layer: The same states keyed by ``"x0"``, ``"x1"``, ...
        output_target: The clamped target, or ``None`` if none was supplied.
        output_prediction: Readout applied to the top latent.
        history: Per-step snapshots of every state when ``return_history`` was
            requested, otherwise ``None``.
    """

    mode: str
    input_state: torch.Tensor
    states: list[torch.Tensor]
    states_by_layer: dict[str, torch.Tensor]
    output_target: torch.Tensor | None
    output_prediction: torch.Tensor
    history: list[list[torch.Tensor]] | None = None


class PCNLayer(torch.nn.Module):
    """One generative layer, predicting the layer below it.

    Holds a single weight matrix used in both directions: top-down as
    ``x_above @ W.T`` to predict the layer below, and bottom-up as ``error @ W``
    to propagate that layer's error upward during inference.

    Attributes:
        W: Weight matrix of shape ``(output_dim, input_dim)``.
        activation_fn: Nonlinearity applied to the top-down prediction.
        activation_deriv: Derivative of ``activation_fn``, evaluated on
            pre-activations to gain-modulate the prediction error.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 activation=torch.nn.LeakyReLU(NEGATIVE_SLOPE),
                 activation_deriv=leaky_relu_deriv):
        """Initialise the layer with Xavier-uniform weights.

        Args:
            input_dim: Width of the layer above, which this layer reads from.
            output_dim: Width of the layer below, which this layer predicts.
            activation: Nonlinearity for the top-down prediction.
            activation_deriv: Derivative of ``activation``, taking
                pre-activations and returning the elementwise gradient. It must
                match ``activation`` or the error signal will be wrong.
        """
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_dim, input_dim))
        # Xavier, not He: W is used in both directions here -- top-down to
        # predict (x @ W.T) and bottom-up to propagate error (err @ W) -- so the
        # symmetric scaling is the right compromise. He would also inflate ||W||,
        # and the stable inference step size falls off as ~1/||W||^2.
        torch.nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation
        self.activation_deriv = activation_deriv

    def forward(self, x_above):
        """Predict the layer below from the layer above.

        Args:
            x_above: States of the layer above, shape ``(batch, input_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The prediction
            ``f(x_above @ W.T)`` and the pre-activation ``x_above @ W.T``, both
            of shape ``(batch, output_dim)``. The pre-activation is returned
            because the error gain needs ``activation_deriv`` evaluated on it.
        """
        a = x_above @ self.W.T
        return self.activation_fn(a), a


class PredictiveCodingNetwork(torch.nn.Module):
    """Predictive-coding network with a linear supervised readout.

    Attributes:
        dims: Layer widths from the visible layer upward, e.g. ``[784, 256, 64]``.
        L: Number of generative layers, ``len(dims) - 1``.
        layers: The generative layers, ``layers[l]`` predicting ``x_l``.
        readout: Linear map from the top latent to the output scores.
    """

    def __init__(self,
                 dims,
                 output_dim,
                 activation=torch.nn.LeakyReLU(NEGATIVE_SLOPE),
                 activation_deriv=leaky_relu_deriv,
                 visible_activation=torch.nn.Identity(),
                 visible_activation_deriv=lambda a: torch.ones_like(a),
    ):
        """Build the layer stack and the readout.

        Args:
            dims: Layer widths from the visible layer upward. ``dims[0]`` must
                match the flattened input size.
            output_dim: Number of classes the readout produces.
            activation: Nonlinearity for the hidden generative layers.
            activation_deriv: Derivative of ``activation``.
            visible_activation: Nonlinearity for the prediction of the visible
                layer. Identity suits real-valued inputs.
            visible_activation_deriv: Derivative of ``visible_activation``.
        """
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.layers = torch.nn.ModuleList([
            PCNLayer(input_dim=dims[l + 1],
                     output_dim=dims[l],
                     activation=visible_activation if l == 0 else activation,
                     activation_deriv=(
                         visible_activation_deriv
                         if l == 0
                         else activation_deriv
                     ))
            for l in range(self.L)

        ])
        self.readout = torch.nn.Linear(dims[-1], output_dim, bias=False)

    def init_latents(self, x):
        """Seed the latent states with a bottom-up pass from the clamped input.

        The old version drew ``randn`` and relied on the relaxation to forget it.
        It never did: 50 steps at eta=0.01 left the top state ~94% its own
        initialisation, so two runs on the same input landed on uncorrelated
        latents and the readout was fitting noise. This init is deterministic
        given ``x`` and starts near the fixed point, so fewer steps are needed.

        Args:
            x: Clamped visible states, shape ``(batch, dims[0])``.

        Returns:
            list[torch.Tensor]: The ``L`` latent states ``[x1, ..., xL]``.
        """
        latents, h = [], x
        for layer in self.layers:
            h = layer.activation_fn(h @ layer.W)
            latents.append(h)
        return latents

    def init_free_states(self, batch_size, device, scale=0.1):
        """Draw small random states for every layer, including the visible one.

        Used by generative mode, where x0 is inferred rather than clamped and so
        has no input to be seeded from.

        Args:
            batch_size: Number of samples.
            device: Device to allocate on.
            scale: Standard deviation of the Gaussian draw.

        Returns:
            list[torch.Tensor]: One state per entry in ``dims``.
        """
        return [
            scale * torch.randn(batch_size, d, device=device, requires_grad=False)
            for d in self.dims
        ]

    def state_labels(self):
        """Return the layer names ``["x0", "x1", ...]``.

        Returns:
            list[str]: One label per layer, visible layer first.
        """
        return [f"x{i}" for i in range(len(self.dims))]

    def states_by_layer(self, inputs_latents):
        """Key a list of states by layer name.

        Args:
            inputs_latents: States ordered ``[x0, x1, ..., xL]``.

        Returns:
            dict[str, torch.Tensor]: States keyed by ``"x0"``, ``"x1"``, ...
        """
        return dict(zip(self.state_labels(), inputs_latents))

    def free_energy(self, inputs_latents, output_target=None):
        """Compute the free energy of a set of states.

        Args:
            inputs_latents: States ordered ``[x0, x1, ..., xL]``.
            output_target: Optional target for the readout. When given, its
                squared error is added to the total.

        Returns:
            dict[str, torch.Tensor]: Batch means under the keys
            ``"free_energy"``, ``"layer_free_energy"`` and
            ``"output_free_energy"``.
        """
        errors, _ = self.compute_errors(inputs_latents)
        layer_energy = sum(
            0.5 * err.float().flatten(start_dim=1).pow(2).sum(dim=1)
            for err in errors
        )
        output_energy = torch.zeros_like(layer_energy)
        if output_target is not None:
            y_hat = self.readout(inputs_latents[-1])
            output_energy = (
                0.5
                * (y_hat.float() - output_target.float())
                .flatten(start_dim=1)
                .pow(2)
                .sum(dim=1)
            )
        return {
            "free_energy": (layer_energy + output_energy).mean(),
            "layer_free_energy": layer_energy.mean(),
            "output_free_energy": output_energy.mean(),
        }

    def prepare_input_state(self, x, device=None):
        """Coerce an input into a clamped visible state.

        Args:
            x: Input, array-like or tensor. A 1-D input is treated as a single
                sample. Anything else is flattened to ``dims[0]`` features.
            device: Target device. Defaults to the model's device.

        Returns:
            torch.Tensor: Float tensor of shape ``(batch, dims[0])``, always a
            fresh copy so in-place state updates cannot touch the caller's data.
        """
        model_device = torch.device(device) if device is not None else next(self.parameters()).device
        x = torch.as_tensor(x, device=model_device).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # clone: generative mode updates index 0 in place, and as_tensor/view
        # would otherwise hand back a view of the caller's tensor.
        return x.reshape(x.shape[0], self.dims[0]).clone()

    def prepare_output_target(self, output, device=None):
        """Coerce a target into a batch of readout-shaped vectors.

        Accepts class indices (scalar or 1-D integer tensor), a single one-hot
        or real-valued vector, or an already-batched 2-D target.

        Args:
            output: The target in any of the accepted forms.
            device: Target device. Defaults to the model's device.

        Returns:
            torch.Tensor: Float tensor of shape ``(batch, output_dim)``.

        Raises:
            ValueError: If the target cannot be interpreted, or if its final
                dimension does not match the readout width.
        """
        model_device = torch.device(device) if device is not None else next(self.parameters()).device
        output = torch.as_tensor(output, device=model_device)

        if output.ndim == 0:
            if output.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                output = F.one_hot(
                    output.long().view(1),
                    num_classes=self.readout.out_features,
                ).float()
            else:
                output = output.float().view(1, 1)
        elif output.ndim == 1:
            if output.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                is_one_hot_vector = (
                    output.numel() == self.readout.out_features
                    and torch.all((output == 0) | (output == 1))
                    and torch.sum(output) == 1
                )
                if is_one_hot_vector:
                    output = output.float().unsqueeze(0)
                else:
                    output = F.one_hot(
                        output.long(),
                        num_classes=self.readout.out_features,
                    ).float()
            elif output.numel() == self.readout.out_features:
                output = output.float().unsqueeze(0)
            elif self.readout.out_features == 1:
                output = output.float().view(-1, 1)
            else:
                raise ValueError(
                    "1D floating output must be a single output vector with "
                    f"{self.readout.out_features} values."
                )
        else:
            output = output.float()

        if output.shape[-1] != self.readout.out_features:
            raise ValueError(
                "Output target has wrong final dimension: "
                f"expected {self.readout.out_features}, got {output.shape[-1]}."
            )

        return output

    def compute_errors(self, inputs_latents):
        """Compute the prediction error at every generative layer.

        Args:
            inputs_latents: States ordered ``[x0, x1, ..., xL]``.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: The raw errors
            ``x_l - f(W_l x_{l+1})`` and the same errors gain-modulated by
            ``activation_deriv`` at the pre-activation. The raw error drives the
            layer's own state; the gain-modulated one drives the layer below and
            the weight update.
        """
        errors, gain_modulated_errors = [], []
        for l, layer in enumerate(self.layers):
            x_hat, a = layer(inputs_latents[l + 1])
            err = inputs_latents[l] - x_hat
            gm_err = err * layer.activation_deriv(a)
            errors.append(err)
            gain_modulated_errors.append(gm_err)
        return errors, gain_modulated_errors

    def generation_step(self, inputs_latents, output_target, eta_infer):
        """Take one inference step with the output clamped and x0 free.

        Args:
            inputs_latents: States ordered ``[x0, x1, ..., xL]``, updated in
                place.
            output_target: Clamped target of shape ``(batch, output_dim)``.
            eta_infer: Step size for the state update.

        Returns:
            list[torch.Tensor]: The same list, updated in place.
        """
        errors, gain_modulated_errors = self.compute_errors(inputs_latents)
        y_hat = self.readout(inputs_latents[-1])
        eps_top = (y_hat - output_target) @ self.readout.weight
        errors_extended = errors + [eps_top]

        for layer_index in range(self.L + 1):
            if layer_index == 0:
                grad = errors_extended[0]
            else:
                grad = (
                    errors_extended[layer_index]
                    - gain_modulated_errors[layer_index - 1]
                    @ self.layers[layer_index - 1].W
                )
            inputs_latents[layer_index] -= eta_infer * grad

        return inputs_latents

    def predictive_step(self, inputs_latents, eta_infer, output_target=None):
        """Take one inference step with x0 clamped.

        Args:
            inputs_latents: States ordered ``[x0, x1, ..., xL]``, updated in
                place. Index 0 is left untouched because it is clamped.
            eta_infer: Step size for the state update.
            output_target: Optional target. When given, the readout error pulls
                the top latent as well; when ``None`` the relaxation is
                label-free, which is what evaluation must use.

        Returns:
            list[torch.Tensor]: The same list, updated in place.
        """
        errors, gain_modulated_errors = self.compute_errors(inputs_latents)
        if output_target is None:
            eps_top = torch.zeros_like(inputs_latents[-1])
        else:
            y_hat = self.readout(inputs_latents[-1])
            eps_top = (y_hat - output_target) @ self.readout.weight

        errors_extended = errors + [eps_top]
        for layer_index in range(1, self.L + 1):
            grad = (
                errors_extended[layer_index]
                - gain_modulated_errors[layer_index - 1]
                @ self.layers[layer_index - 1].W
            )
            inputs_latents[layer_index] -= eta_infer * grad

        return inputs_latents

    def run(
        self,
        mode,
        x=None,
        output=None,
        y=None,
        eta_infer=0.05,
        T_infer=50,
        device=None,
        state_scale=0.1,
        x0_init=None,
        latent_inits=None,
        return_history=False,
    ):
        """Run the inference relaxation to convergence and read out.

        Predictive mode clamps x0 from ``x`` and infers the latents. Generative
        mode clamps the target and infers x0 as a free state.

        ``eta_infer`` and ``T_infer`` should match the values used during
        training; a differently converged relaxation produces different latents
        and therefore different readout scores.

        Args:
            mode: ``"predictive"`` or ``"generative"``.
            x: Input for predictive mode.
            output: Target to clamp. Required for generative mode.
            y: Alias for ``output``.
            eta_infer: Step size for the relaxation.
            T_infer: Number of relaxation steps.
            device: Device to run on. Defaults to the model's device.
            state_scale: Spread of the random init used in generative mode.
            x0_init: Optional starting visible state for generative mode.
            latent_inits: Optional starting latents, one per generative layer.
            return_history: Whether to record every intermediate state.

        Returns:
            PCNRunResult: Final states, readout prediction, and optionally the
            full trajectory.

        Raises:
            ValueError: If ``mode`` is unknown, if a required argument for the
                chosen mode is missing, or if ``latent_inits`` has the wrong
                length.
        """
        if mode not in {"predictive", "generative"}:
            raise ValueError("mode must be 'predictive' or 'generative'")

        model_device = torch.device(device) if device is not None else next(self.parameters()).device
        self.to(model_device).eval()

        output_target = None
        if output is None and y is not None:
            output = y
        if output is not None:
            output_target = self.prepare_output_target(output, model_device)

        if mode == "predictive":
            if x is None:
                raise ValueError("predictive mode requires x.")
            input_state = self.prepare_input_state(x, model_device)
            batch_size = input_state.shape[0]
            inputs_latents = [input_state] + self.init_latents(input_state)
        else:
            if output_target is None:
                raise ValueError("generative mode requires output or y.")
            batch_size = output_target.shape[0]
            inputs_latents = self.init_free_states(batch_size, model_device, scale=state_scale)
            if x0_init is not None:
                inputs_latents[0] = self.prepare_input_state(x0_init, model_device)

        if latent_inits is not None:
            if len(latent_inits) != self.L:
                raise ValueError(f"latent_inits must contain {self.L} tensors.")
            for layer_index, latent in enumerate(latent_inits, start=1):
                inputs_latents[layer_index] = torch.as_tensor(
                    latent,
                    device=model_device,
                    dtype=inputs_latents[layer_index].dtype,
                ).view(batch_size, self.dims[layer_index])

        history = None
        if return_history:
            history = [[state.detach().clone() for state in inputs_latents]]

        with torch.no_grad():
            for _ in range(T_infer):
                if mode == "predictive":
                    self.predictive_step(inputs_latents, eta_infer, output_target)
                else:
                    self.generation_step(inputs_latents, output_target, eta_infer)

                if return_history:
                    history.append([state.detach().clone() for state in inputs_latents])

            output_prediction = self.readout(inputs_latents[-1]).detach().clone()

        detached_states = [state.detach().clone() for state in inputs_latents]
        return PCNRunResult(
            mode=mode,
            input_state=detached_states[0],
            states=detached_states,
            states_by_layer=self.states_by_layer(detached_states),
            output_target=(
                output_target.detach().clone()
                if output_target is not None
                else None
            ),
            output_prediction=output_prediction,
            history=history,
        )

    def set_weight_gradients(self, inputs_latents, output_target):
        """Write the analytic free-energy gradients into ``parameter.grad``.

        The gradients are ``dF/dW_l = -gain_modulated_error_l.T @ x_{l+1}`` for
        the generative layers and ``dF/dW_out = (y_hat - y).T @ x_L`` for the
        readout, both averaged over the batch. Writing them to ``.grad`` instead
        of applying them directly is what lets any ``torch.optim`` optimizer run
        the weight phase, momentum and weight decay included.

        Args:
            inputs_latents: Relaxed states ordered ``[x0, x1, ..., xL]``.
            output_target: Target of shape ``(batch, output_dim)``.
        """
        batch_size = inputs_latents[0].shape[0]
        _, gain_modulated_errors = self.compute_errors(inputs_latents)

        for l, layer in enumerate(self.layers):
            layer.W.grad = -(gain_modulated_errors[l].T @ inputs_latents[l + 1]) / batch_size

        output_error = self.readout(inputs_latents[-1]) - output_target
        self.readout.weight.grad = output_error.T @ inputs_latents[-1] / batch_size

    def fit(
        self,
        data,
        num_epochs,
        eta_infer,
        T_infer,
        batch_size=32,
        shuffle=True,
        optimizer=None,
        eta_learn=0.01,
        weight_decay=0.05,
        T_learn=1,
        validation_data=None,
        test_data=None,
        device=None,
        record_history=True,
        eval_every=1,
        eval_T_infer=None,
        eval_eta_infer=None,
        progress=True,
    ):
        """Train the network by alternating inference and weight updates.

        Each batch is relaxed for ``T_infer`` steps with the label clamped, then
        the weights take ``T_learn`` steps on the resulting states.

        Args:
            data: Training data. A ``DataLoader`` is used as given; any other
                ``Dataset`` is wrapped in one using ``batch_size`` and
                ``shuffle``.
            num_epochs: Number of passes over the data.
            eta_infer: Step size for the inference relaxation. Too small and the
                relaxation never reaches its fixed point; too large and it
                diverges. The stable ceiling falls roughly as ``1/||W||^2``.
            T_infer: Number of inference steps per batch.
            batch_size: Batch size, used only when ``data`` is not already a
                ``DataLoader``.
            shuffle: Whether to shuffle, same condition as ``batch_size``.
            optimizer: Any ``torch.optim.Optimizer`` over this model's
                parameters. Defaults to ``SGD(lr=eta_learn,
                weight_decay=weight_decay)``. When supplied, ``eta_learn`` and
                ``weight_decay`` are ignored because the optimizer owns them.
            eta_learn: Learning rate for the default optimizer.
            weight_decay: Weight decay for the default optimizer. This is the
                Gaussian prior on W in the same free energy, so it keeps the
                update MAP descent on F. Without it ``||W||`` grows without
                bound, which stiffens the relaxation until accuracy decays
                (measured: 0.74 -> 0.63 over 240 epochs at ``weight_decay=0``).
            T_learn: Weight updates per batch. Each one reuses the same, by then
                stale, latents, so values above 1 amount to a larger learning
                rate plus drift. Leave at 1.
            validation_data: Optional data to score each ``eval_every`` epochs.
            test_data: Optional data to score alongside it. Prefer leaving this
                unset and holding the test split back until training is done.
            device: Device to train on. Defaults to the model's device.
            record_history: Whether to record metrics at all.
            eval_every: Record metrics every this many epochs.
            eval_T_infer: Inference steps for evaluation. Defaults to
                ``T_infer``.
            eval_eta_infer: Inference step size for evaluation. Defaults to
                ``eta_infer``.
            progress: Whether to show the per-epoch progress bar.

        Returns:
            dict: Keys ``"epoch"``, ``"train"``, ``"validation"``, ``"test"``
            and ``"config"``. The first four line up index by index; each
            metrics entry is the dict returned by :meth:`evaluate`.
        """
        model_device = (
            torch.device(device) if device is not None
            else next(self.parameters()).device
        )
        self.to(model_device).train()

        loader = _as_loader(data, batch_size, shuffle)
        validation_loader = _as_loader(validation_data, batch_size, False)
        test_loader = _as_loader(test_data, batch_size, False)

        if optimizer is None:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=eta_learn, weight_decay=weight_decay
            )

        history = {
            "epoch": [],
            "train": [],
            "validation": [],
            "test": [],
            "config": {
                "eta_infer": eta_infer,
                "T_infer": T_infer,
                "T_learn": T_learn,
                "batch_size": loader.batch_size,
                "optimizer": type(optimizer).__name__,
                "optimizer_defaults": dict(optimizer.defaults),
                "eval_T_infer": eval_T_infer if eval_T_infer is not None else T_infer,
                "eval_eta_infer": eval_eta_infer if eval_eta_infer is not None else eta_infer,
            },
        }

        for epoch in tqdm(range(num_epochs), disable=not progress):
            for x_batch, y_batch in loader:
                x_batch = x_batch.reshape(x_batch.size(0), self.dims[0]).to(model_device)
                y_labels = y_batch.to(model_device).long()
                y_target = F.one_hot(
                    y_labels, num_classes=self.readout.out_features
                ).float()

                inputs_latents = [x_batch] + self.init_latents(x_batch)

                with torch.no_grad():
                    for _ in range(T_infer):
                        self.predictive_step(
                            inputs_latents,
                            eta_infer=eta_infer,
                            output_target=y_target,
                        )

                    for _ in range(T_learn):
                        optimizer.zero_grad(set_to_none=True)
                        self.set_weight_gradients(inputs_latents, y_target)
                        optimizer.step()

            if record_history and (epoch + 1) % eval_every == 0:
                eval_kwargs = dict(
                    eta_infer=eval_eta_infer if eval_eta_infer is not None else eta_infer,
                    T_infer=eval_T_infer if eval_T_infer is not None else T_infer,
                    device=model_device,
                )
                history["epoch"].append(epoch + 1)
                # Re-run label-free. Scoring the latents that were just inferred
                # with the target clamped would measure accuracy with the answer
                # already in hand.
                history["train"].append(self.evaluate(loader, **eval_kwargs))

                if validation_loader is not None:
                    history["validation"].append(
                        self.evaluate(validation_loader, **eval_kwargs)
                    )

                if test_loader is not None:
                    history["test"].append(self.evaluate(test_loader, **eval_kwargs))

                self.train()

        return history

    def evaluate(
        self,
        data,
        eta_infer=0.05,
        T_infer=50,
        batch_size=256,
        device=None,
    ):
        """Score the model on a dataset using label-free inference.

        The relaxation runs without the target clamped, so the latents depend on
        the input alone and the readout is genuinely predicting.

        Args:
            data: A ``DataLoader``, or any ``Dataset`` to wrap in one.
            eta_infer: Step size for the relaxation. Match training.
            T_infer: Number of relaxation steps. Match training.
            batch_size: Batch size, used only when ``data`` is not a
                ``DataLoader``.
            device: Device to run on. Defaults to the model's device.

        Returns:
            dict: ``n``, ``output_mse``, ``output_energy``, ``cross_entropy``,
            ``accuracy``, ``free_energy``, ``layer_free_energy``, ``f1_macro``,
            and for two-class problems ``f1`` (positive class).
        """
        model_device = (
            torch.device(device) if device is not None
            else next(self.parameters()).device
        )
        loader = _as_loader(data, batch_size, False)
        was_training = self.training
        self.to(model_device).eval()
        totals = _empty_metric_totals()

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.reshape(x_batch.size(0), self.dims[0]).to(model_device)
                y_labels = y_batch.to(model_device).long()
                y_target = F.one_hot(
                    y_labels,
                    num_classes=self.readout.out_features,
                ).float()

                result = self.run(
                    mode="predictive",
                    x=x_batch,
                    eta_infer=eta_infer,
                    T_infer=T_infer,
                    device=model_device,
                )
                _accumulate_metrics(totals, self, result.states, y_target, y_labels)

        if was_training:
            self.train()
        return _finalize_metric_totals(totals)


def _as_loader(data, batch_size, shuffle):
    """Wrap a dataset in a ``DataLoader``, passing an existing one through.

    Args:
        data: A ``DataLoader``, a ``Dataset``, or ``None``.
        batch_size: Batch size for the wrapped loader.
        shuffle: Whether the wrapped loader shuffles.

    Returns:
        DataLoader | None: ``None`` if ``data`` was ``None``, ``data`` itself if
        it was already a ``DataLoader``, otherwise a new loader over it.
    """
    if data is None or isinstance(data, DataLoader):
        return data
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def _empty_metric_totals():
    """Create a zeroed accumulator for the evaluation metrics.

    Returns:
        dict: Running sums, with ``confusion`` left as ``None`` until the first
        batch fixes the class count.
    """
    return {
        "n": 0,
        "output_elements": 0,
        "output_mse_sum": 0.0,
        "output_energy_sum": 0.0,
        "cross_entropy_sum": 0.0,
        "correct": 0,
        "free_energy_sum": 0.0,
        "layer_free_energy_sum": 0.0,
        "confusion": None,
    }


def _accumulate_metrics(totals, model, inputs_latents, y_target, y_labels):
    """Fold one batch into a metric accumulator.

    Args:
        totals: Accumulator from :func:`_empty_metric_totals`, updated in place.
        model: The network, used for its readout and free energy.
        inputs_latents: Relaxed states for the batch.
        y_target: One-hot targets of shape ``(batch, output_dim)``.
        y_labels: Integer class labels of shape ``(batch,)``.
    """
    y_hat = model.readout(inputs_latents[-1])
    batch_size = y_hat.shape[0]
    output_delta = y_hat.float() - y_target.float()
    output_energy = 0.5 * output_delta.flatten(start_dim=1).pow(2).sum(dim=1)
    free_energy = model.free_energy(inputs_latents, y_target)
    predicted_classes = torch.argmax(y_hat, dim=1)

    totals["n"] += batch_size
    totals["output_elements"] += y_hat.numel()
    totals["output_mse_sum"] += F.mse_loss(
        y_hat.float(),
        y_target.float(),
        reduction="sum",
    ).item()
    totals["output_energy_sum"] += output_energy.sum().item()
    totals["cross_entropy_sum"] += F.cross_entropy(
        y_hat.float(),
        y_labels,
        reduction="sum",
    ).item()
    totals["correct"] += (predicted_classes == y_labels).sum().item()
    totals["free_energy_sum"] += free_energy["free_energy"].item() * batch_size
    totals["layer_free_energy_sum"] += (
        free_energy["layer_free_energy"].item() * batch_size
    )

    # Confusion matrix as one bincount: rows = true label, cols = prediction.
    # Precision/recall/F1 all fall out of it at the end.
    num_classes = y_hat.shape[1]
    counts = torch.bincount(
        y_labels * num_classes + predicted_classes,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    totals["confusion"] = (
        counts if totals["confusion"] is None else totals["confusion"] + counts
    )


def _finalize_metric_totals(totals):
    """Turn accumulated sums into per-sample metrics.

    Args:
        totals: Accumulator filled by :func:`_accumulate_metrics`.

    Returns:
        dict: The averaged metrics, including the F1 scores.
    """
    n = max(1, totals["n"])
    output_elements = max(1, totals["output_elements"])
    metrics = {
        "n": totals["n"],
        "output_mse": totals["output_mse_sum"] / output_elements,
        "output_energy": totals["output_energy_sum"] / n,
        "cross_entropy": totals["cross_entropy_sum"] / n,
        "accuracy": totals["correct"] / n,
        "free_energy": totals["free_energy_sum"] / n,
        "layer_free_energy": totals["layer_free_energy_sum"] / n,
    }
    metrics.update(_f1_from_confusion(totals["confusion"]))
    return metrics


def _f1_from_confusion(confusion):
    """Derive F1 scores from a confusion matrix.

    A class that is predicted but never occurs (or vice versa) scores 0. A class
    absent from both is dropped from the macro average rather than counted as 0,
    which is what sklearn's ``average="macro"`` does. Verified against
    ``sklearn.metrics.f1_score`` for binary, 3-class and 10-class cases.

    Args:
        confusion: Square matrix with true labels on rows and predictions on
            columns, or ``None`` if no batch was seen.

    Returns:
        dict: ``f1_macro``, plus ``f1`` for the positive class when the problem
        is binary.
    """
    if confusion is None:
        return {"f1_macro": 0.0}

    confusion = confusion.float()
    true_positive = confusion.diag()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    present = denominator > 0
    f1 = torch.where(
        present,
        2.0 * true_positive / denominator.clamp(min=1.0),
        torch.zeros_like(denominator),
    )

    metrics = {"f1_macro": f1[present].mean().item() if present.any() else 0.0}
    if confusion.shape[0] == 2:
        # Positive-class F1, i.e. what sklearn's f1_score returns by default.
        metrics["f1"] = f1[1].item()
    return metrics
