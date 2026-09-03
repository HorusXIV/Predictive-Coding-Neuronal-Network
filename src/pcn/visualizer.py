from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch


InputDisplayMode = Literal["collapsed", "full"]
AxisLimits = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass
class PCNFrame:
    """One drawable snapshot of a predictive-coding inference state."""

    states: list[np.ndarray]
    predictions: list[np.ndarray]
    errors: list[np.ndarray]
    energy: float
    step: int | None = None
    output_prediction: np.ndarray | None = None
    output_target: np.ndarray | None = None


class PCN3DVisualizer:
    """3D state/error visualizer for the PCN defined in ``pcn.model``.

    Layers are arranged on the x-axis, neurons inside a layer are laid out on
    y, and the z-axis is the current activation or prediction value. Each
    prediction sits at the same x/y location as the neuron it predicts, so the
    vertical spring between them is the prediction error.
    """

    def __init__(
        self,
        input_mode: InputDisplayMode = "collapsed",
        max_nodes_per_layer: int = 80,
        layer_gap: float = 3.0,
        node_gap: float = 0.18,
        spring_coils: int = 5,
        spring_radius: float = 0.055,
        show_backbone: bool = False,
        fixed_axes: bool = True,
        axis_padding: float = 0.08,
        readout_gap: float = 0.55,
    ) -> None:
        if input_mode not in {"collapsed", "full"}:
            raise ValueError("input_mode must be 'collapsed' or 'full'")
        self.input_mode = input_mode
        self.max_nodes_per_layer = max_nodes_per_layer
        self.layer_gap = layer_gap
        self.node_gap = node_gap
        self.spring_coils = spring_coils
        self.spring_radius = spring_radius
        self.show_backbone = show_backbone
        self.fixed_axes = fixed_axes
        self.axis_padding = axis_padding
        self.readout_gap = readout_gap

    def snapshot(
        self,
        model,
        inputs_latents: list[torch.Tensor],
        output_target: torch.Tensor | None = None,
        sample_index: int = 0,
        step: int | None = None,
    ) -> PCNFrame:
        """Convert a model state into a lightweight, matplotlib-friendly frame."""

        model_device = next(model.parameters()).device
        inputs_latents = [x.detach().to(model_device) for x in inputs_latents]

        with torch.no_grad():
            errors, _ = model.compute_errors(inputs_latents)
            predictions = [
                layer(inputs_latents[layer_index + 1])[0]
                for layer_index, layer in enumerate(model.layers)
            ]
            output_prediction = None
            if hasattr(model, "readout"):
                output_prediction = model.readout(inputs_latents[-1])

        states_np = [self._sample_vector(x, sample_index) for x in inputs_latents]
        predictions_np = [self._sample_vector(x, sample_index) for x in predictions]
        errors_np = [self._sample_vector(x, sample_index) for x in errors]
        energy = float(0.5 * sum(np.square(err).sum() for err in errors_np))
        output_prediction_np = (
            self._sample_vector(output_prediction, sample_index)
            if output_prediction is not None
            else None
        )
        output_target_np = (
            self._sample_vector(output_target, sample_index)
            if output_target is not None
            else None
        )
        if output_prediction_np is not None and output_target_np is not None:
            energy += float(0.5 * np.square(output_target_np - output_prediction_np).sum())

        return PCNFrame(
            states=states_np,
            predictions=predictions_np,
            errors=errors_np,
            energy=energy,
            step=step,
            output_prediction=output_prediction_np,
            output_target=output_target_np,
        )

    def plot(
        self,
        frame: PCNFrame,
        ax=None,
        title: str | None = None,
        show: bool = True,
        elev: float = 24,
        azim: float = -58,
        axis_limits: AxisLimits | None = None,
    ):
        """Draw a static 3D visualization and return ``(fig, ax)``."""

        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        if ax is None:
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
            ax.clear()

        states, predictions, errors, layer_labels = self._drawable_layers(frame)
        max_error = max(
            [
                float(np.max(np.abs(err)))
                for err in errors
                if err is not None and err.size
            ]
            + [1e-6]
        )
        err_norm = Normalize(vmin=0.0, vmax=max_error)

        for layer_index, state_values in enumerate(states):
            state_indices, state_visible, state_collapsed = self._visible_values(
                state_values, layer_index
            )
            x_state, y_state = self._node_grid(
                layer_index,
                len(state_visible),
                states=states,
                layer_labels=layer_labels,
            )

            point_label = "state / target"
            ax.scatter(
                x_state,
                y_state,
                state_visible,
                s=34 if len(state_visible) < 120 else 14,
                c=state_visible,
                cmap="viridis",
                edgecolors="#17202a",
                linewidths=0.35,
                depthshade=True,
                label=point_label if layer_index == 0 else None,
            )

            if self._is_readout_source(layer_labels[layer_index]):
                ax.scatter(
                    x_state,
                    y_state,
                    state_visible,
                    marker="s",
                    s=60 if len(state_visible) < 120 else 24,
                    facecolors="none",
                    edgecolors="#4b5563",
                    linewidths=1.05,
                    depthshade=False,
                    label="readout source",
                )

            if predictions[layer_index] is not None:
                prediction_values = predictions[layer_index]
                error_values = errors[layer_index]
                pred_visible = self._matching_values(
                    prediction_values, state_indices, state_collapsed
                )
                err_visible = self._matching_values(
                    error_values, state_indices, state_collapsed
                )
                x_pred, y_pred = x_state, y_state

                ax.scatter(
                    x_pred,
                    y_pred,
                    pred_visible,
                    marker="^",
                    s=52 if len(pred_visible) < 120 else 22,
                    facecolors="none",
                    edgecolors="#3b2f00",
                    linewidths=0.95,
                    depthshade=False,
                    label=(
                        self._prediction_label(layer_index, layer_labels)
                        if layer_index == 0
                        else None
                    ),
                )

                for start, end, err in zip(
                    zip(x_state, y_state, state_visible),
                    zip(x_pred, y_pred, pred_visible),
                    err_visible,
                ):
                    curve = self._vertical_spring_curve(
                        x=float(start[0]),
                        y=float(start[1]),
                        z0=float(start[2]),
                        z1=float(end[2]),
                    )
                    color = "#c63f3f" if float(err) >= 0.0 else "#2e6fbb"
                    linewidth = 0.65 + 1.8 * err_norm(abs(float(err)))
                    ax.plot(
                        curve[:, 0],
                        curve[:, 1],
                        curve[:, 2],
                        color=color,
                        linewidth=linewidth,
                        alpha=0.78,
                    )

        if self.show_backbone:
            self._draw_layer_backbone(ax, states)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Neuron grid")
        ax.set_zlabel("Activation / prediction")
        tick_values, tick_labels = self._layer_ticks(layer_labels)
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)
        if self.fixed_axes or axis_limits is not None:
            self._apply_axis_limits(ax, axis_limits or self.axis_limits([frame]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title or self._frame_title(frame))
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        fig.subplots_adjust(right=0.82)

        if show:
            plt.show()
        return fig, ax

    def animate(
        self,
        frames: Iterable[PCNFrame],
        interval: int = 90,
        path: str | Path | None = None,
        elev: float = 24,
        azim: float = -58,
        fixed_axes: bool | None = None,
    ):
        """Create an animation from recorded frames.

        ``path`` may end in ``.gif`` or ``.mp4``. The returned animation can also
        be displayed in a notebook with ``HTML(anim.to_jshtml())``.
        """

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        frames = list(frames)
        if not frames:
            raise ValueError("animate() needs at least one frame")

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
        use_fixed_axes = self.fixed_axes if fixed_axes is None else fixed_axes
        axis_limits = self.axis_limits(frames) if use_fixed_axes else None

        def update(i):
            self.plot(
                frames[i],
                ax=ax,
                title=self._frame_title(frames[i]),
                show=False,
                elev=elev,
                azim=azim,
                axis_limits=axis_limits,
            )
            return list(ax.collections) + list(ax.lines)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=interval,
            blit=False,
        )

        if path is not None:
            path = Path(path)
            if path.suffix.lower() == ".gif":
                anim.save(path, writer=PillowWriter(fps=max(1, 1000 // interval)))
            else:
                anim.save(path)

        return anim

    def plot_interactive(
        self,
        frame: PCNFrame,
        title: str | None = None,
        axis_limits: AxisLimits | None = None,
    ):
        """Return a draggable Plotly 3D figure for a recorded PCN frame."""

        import plotly.graph_objects as go

        states, predictions, errors, layer_labels = self._drawable_layers(frame)
        axis_limits = axis_limits or self.axis_limits([frame])
        fig = go.Figure()

        max_error = max(
            [
                float(np.max(np.abs(err)))
                for err in errors
                if err is not None and err.size
            ]
            + [1e-6]
        )

        for layer_index, state_values in enumerate(states):
            state_indices, state_visible, state_collapsed = self._visible_values(
                state_values, layer_index
            )
            x_state, y_state = self._node_grid(
                layer_index,
                len(state_visible),
                states=states,
                layer_labels=layer_labels,
            )
            label = layer_labels[layer_index]
            if state_collapsed:
                label += " collapsed"

            fig.add_trace(
                go.Scatter3d(
                    x=x_state,
                    y=y_state,
                    z=state_visible,
                    mode="markers",
                    name=f"{label} state/target",
                    marker={
                        "size": 4.8,
                        "color": state_visible,
                        "colorscale": "Viridis",
                        "line": {"color": "#17202a", "width": 1},
                    },
                    text=[
                        f"{label}[{int(idx)}] value={value:.4g}"
                        for idx, value in zip(state_indices, state_visible)
                    ],
                    hoverinfo="text",
                    showlegend=layer_index == 0 or label == "Y",
                )
            )

            if self._is_readout_source(layer_labels[layer_index]):
                fig.add_trace(
                    go.Scatter3d(
                        x=x_state,
                        y=y_state,
                        z=state_visible,
                        mode="markers",
                        name=f"{label} free state",
                        marker={
                            "size": 7.5,
                            "symbol": "square-open",
                            "color": "#4b5563",
                            "line": {"color": "#4b5563", "width": 3},
                        },
                        text=[
                            f"{label}[{int(idx)}] is inferred and drives the readout"
                            for idx in state_indices
                        ],
                        hoverinfo="text",
                        showlegend=True,
                    )
                )

            if predictions[layer_index] is None:
                continue

            pred_visible = self._matching_values(
                predictions[layer_index], state_indices, state_collapsed
            )
            err_visible = self._matching_values(
                errors[layer_index], state_indices, state_collapsed
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x_state,
                    y=y_state,
                    z=pred_visible,
                    mode="markers",
                    name=self._prediction_label(layer_index, layer_labels),
                    marker={
                        "size": 7.0,
                        "symbol": "diamond-open",
                        "color": "#ffb000",
                        "line": {"color": "#3b2f00", "width": 3},
                    },
                    text=[
                        (
                            f"{label}[{int(idx)}] pred={pred:.4g}"
                            f"<br>{self._prediction_label(layer_index, layer_labels)}"
                            f"<br>err={err:.4g}"
                        )
                        for idx, pred, err in zip(
                            state_indices, pred_visible, err_visible
                        )
                    ],
                    hoverinfo="text",
                    showlegend=layer_index == 0 or label == "Y",
                )
            )

            for x, y, z0, z1, err in zip(
                x_state, y_state, state_visible, pred_visible, err_visible
            ):
                curve = self._vertical_spring_curve(
                    x=float(x),
                    y=float(y),
                    z0=float(z0),
                    z1=float(z1),
                )
                width = 1.5 + 4.0 * abs(float(err)) / max_error
                color = "#c63f3f" if float(err) >= 0.0 else "#2e6fbb"
                fig.add_trace(
                    go.Scatter3d(
                        x=curve[:, 0],
                        y=curve[:, 1],
                        z=curve[:, 2],
                        mode="lines",
                        name="error spring",
                        line={"color": color, "width": width},
                        opacity=0.75,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title=title or self._frame_title(frame),
            scene={
                "xaxis": {
                    "title": "Layer",
                    "tickmode": "array",
                    "tickvals": list(self._layer_ticks(layer_labels)[0]),
                    "ticktext": list(self._layer_ticks(layer_labels)[1]),
                    "range": list(axis_limits[0]),
                },
                "yaxis": {"title": "Neuron grid", "range": list(axis_limits[1])},
                "zaxis": {
                    "title": "Activation / prediction",
                    "range": list(axis_limits[2]),
                },
                "aspectmode": "manual",
                "aspectratio": self._plotly_axis_aspect(axis_limits),
            },
            margin={"l": 0, "r": 0, "t": 52, "b": 0},
            legend={"x": 0.0, "y": 1.0},
        )
        return fig

    def save_interactive_html(
        self,
        frame: PCNFrame,
        path: str | Path,
        title: str | None = None,
        axis_limits: AxisLimits | None = None,
        include_plotlyjs: bool | str = True,
    ) -> Path:
        """Write a standalone draggable Plotly visualization and return its path."""

        path = Path(path)
        html = self.interactive_html(
            frame,
            title=title,
            axis_limits=axis_limits,
            include_plotlyjs=include_plotlyjs,
        )
        path.write_text(html)
        return path

    def interactive_html(
        self,
        frame: PCNFrame,
        title: str | None = None,
        axis_limits: AxisLimits | None = None,
        include_plotlyjs: bool | str = True,
    ) -> str:
        """Return self-contained HTML for reliable notebook 3D interaction."""

        fig = self.plot_interactive(
            frame,
            title=title,
            axis_limits=axis_limits,
        )
        return fig.to_html(
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            config={"scrollZoom": True, "displaylogo": False},
        )

    def run_supervised_inference(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        eta_infer: float = 0.05,
        T_infer: int = 50,
        device: str | torch.device | None = None,
        record_every: int | None = None,
        sample_index: int = 0,
    ) -> tuple[list[torch.Tensor], list[PCNFrame]]:
        """Run PCN inference and optionally record frames for visualization.

        If ``y`` is provided, the top latent layer is pulled by the supervised
        readout error. If ``y`` is omitted, the top error is treated as zero.
        """

        model_device = torch.device(device) if device is not None else next(model.parameters()).device
        model.to(model_device).eval()

        x = x.detach().to(model_device)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, model.dims[0])
        inputs_latents = [x] + model.init_latents(x)

        y_target = None
        if y is not None:
            y = y.detach().to(model_device)
            if y.ndim == 1:
                y_target = torch.nn.functional.one_hot(
                    y.long(),
                    num_classes=model.readout.out_features,
                ).float()
            else:
                y_target = y.float()

        frames: list[PCNFrame] = []
        if record_every is not None:
            frames.append(
                self.snapshot(
                    model,
                    inputs_latents,
                    output_target=y_target,
                    sample_index=sample_index,
                    step=0,
                )
            )

        with torch.no_grad():
            for step in range(1, T_infer + 1):
                model.predictive_step(inputs_latents, eta_infer, y_target)

                if record_every is not None and step % record_every == 0:
                    frames.append(
                        self.snapshot(
                            model,
                            inputs_latents,
                            output_target=y_target,
                            sample_index=sample_index,
                            step=step,
                        )
                    )

        return inputs_latents, frames

    def run_generation(
        self,
        model,
        output: torch.Tensor,
        eta_infer: float = 0.05,
        T_infer: int = 50,
        device: str | torch.device | None = None,
        record_every: int | None = None,
        sample_index: int = 0,
        state_scale: float = 0.1,
        x0_init: torch.Tensor | None = None,
        latent_inits: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[PCNFrame]]:
        """Generate a free x0 state while clamping the readout/output target."""

        model_device = torch.device(device) if device is not None else next(model.parameters()).device
        model.to(model_device).eval()

        output_target = model.prepare_output_target(output, model_device)
        batch_size = output_target.shape[0]
        inputs_latents = model.init_free_states(batch_size, model_device, scale=state_scale)

        if x0_init is not None:
            inputs_latents[0] = torch.as_tensor(
                x0_init,
                device=model_device,
                dtype=output_target.dtype,
            ).view(batch_size, model.dims[0])

        if latent_inits is not None:
            if len(latent_inits) != model.L:
                raise ValueError(f"latent_inits must contain {model.L} tensors.")
            for layer_index, latent in enumerate(latent_inits, start=1):
                inputs_latents[layer_index] = torch.as_tensor(
                    latent,
                    device=model_device,
                    dtype=output_target.dtype,
                ).view(batch_size, model.dims[layer_index])

        frames: list[PCNFrame] = []
        if record_every is not None:
            frames.append(
                self.snapshot(
                    model,
                    inputs_latents,
                    output_target=output_target,
                    sample_index=sample_index,
                    step=0,
                )
            )

        with torch.no_grad():
            for step in range(1, T_infer + 1):
                model.generation_step(inputs_latents, output_target, eta_infer)

                if record_every is not None and step % record_every == 0:
                    frames.append(
                        self.snapshot(
                            model,
                            inputs_latents,
                            output_target=output_target,
                            sample_index=sample_index,
                            step=step,
                        )
                    )

        return inputs_latents[0].detach().clone(), inputs_latents, frames

    @staticmethod
    def _sample_vector(tensor: torch.Tensor, sample_index: int) -> np.ndarray:
        if tensor.ndim == 1:
            vector = tensor
        else:
            vector = tensor[min(sample_index, tensor.shape[0] - 1)]
        return vector.detach().float().cpu().numpy().reshape(-1)

    def _visible_values(
        self,
        values: np.ndarray,
        layer_index: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if layer_index == 0 and self.input_mode == "collapsed":
            return np.array([0]), np.array([float(np.mean(values))]), True

        if values.size <= self.max_nodes_per_layer:
            indices = np.arange(values.size)
        else:
            indices = np.linspace(
                0,
                values.size - 1,
                self.max_nodes_per_layer,
                dtype=int,
            )
        return indices, values[indices], False

    @staticmethod
    def _matching_values(
        values: np.ndarray,
        indices: np.ndarray,
        collapsed: bool,
    ) -> np.ndarray:
        if collapsed:
            return np.array([float(np.mean(values))])
        return values[indices]

    def _node_grid(
        self,
        layer_index: int,
        n_nodes: int,
        states: list[np.ndarray] | None = None,
        layer_labels: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if layer_labels is not None and layer_labels[layer_index] == "Y target":
            source_index = self._readout_source_index(layer_labels)
            x = np.full(n_nodes, source_index * self.layer_gap)
            y_center = self._readout_y_center(source_index, states)
            y = y_center + (np.arange(n_nodes) - (n_nodes - 1) / 2.0) * self.node_gap
            return x, y

        x = np.full(n_nodes, layer_index * self.layer_gap)
        y = (np.arange(n_nodes) - (n_nodes - 1) / 2.0) * self.node_gap
        return x, y

    def _readout_y_center(
        self,
        source_index: int,
        states: list[np.ndarray] | None,
    ) -> float:
        if states is None or source_index >= len(states):
            return self.readout_gap

        _, source_visible, _ = self._visible_values(states[source_index], source_index)
        source_half_span = max(0.0, (len(source_visible) - 1) * self.node_gap / 2.0)
        return source_half_span + self.readout_gap

    @staticmethod
    def _readout_source_index(layer_labels: list[str]) -> int:
        for index, label in enumerate(layer_labels):
            if PCN3DVisualizer._is_readout_source(label):
                return index
        return max(0, len(layer_labels) - 2)

    def _layer_ticks(self, layer_labels: list[str]) -> tuple[np.ndarray, list[str]]:
        labels = [label for label in layer_labels if label != "Y target"]
        values = np.arange(len(labels), dtype=float) * self.layer_gap
        return values, labels

    def axis_limits(self, frames: Iterable[PCNFrame]) -> AxisLimits:
        """Compute stable axis limits for one frame or a full animation."""

        frames = list(frames)
        if not frames:
            raise ValueError("axis_limits() needs at least one frame")

        x_values: list[np.ndarray] = []
        y_values: list[np.ndarray] = []
        z_values: list[np.ndarray] = []

        for frame in frames:
            states, predictions, _, layer_labels = self._drawable_layers(frame)
            for layer_index, state_values in enumerate(states):
                state_indices, state_visible, state_collapsed = self._visible_values(
                    state_values, layer_index
                )
                x_state, y_state = self._node_grid(
                    layer_index,
                    len(state_visible),
                    states=states,
                    layer_labels=layer_labels,
                )
                x_values.append(x_state)
                y_values.append(y_state)
                z_values.append(state_visible)

                if predictions[layer_index] is not None:
                    z_values.append(
                        self._matching_values(
                            predictions[layer_index],
                            state_indices,
                            state_collapsed,
                        )
                    )

        return (
            self._padded_limits(np.concatenate(x_values)),
            self._padded_limits(np.concatenate(y_values)),
            self._padded_limits(np.concatenate(z_values)),
        )

    def _apply_axis_limits(self, ax, axis_limits: AxisLimits) -> None:
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])
        try:
            ax.set_box_aspect(self._axis_aspect(axis_limits))
        except AttributeError:
            pass

    def _axis_aspect(self, axis_limits: AxisLimits) -> tuple[float, float, float]:
        spans = [max(high - low, 1e-6) for low, high in axis_limits]
        max_span = max(spans)
        return tuple(float(span / max_span) for span in spans)

    def _plotly_axis_aspect(self, axis_limits: AxisLimits) -> dict[str, float]:
        aspect = self._axis_aspect(axis_limits)
        return {"x": aspect[0], "y": aspect[1], "z": aspect[2]}

    def _padded_limits(self, values: np.ndarray) -> tuple[float, float]:
        low = float(np.min(values))
        high = float(np.max(values))
        if low == high:
            pad = max(1.0, abs(low) * self.axis_padding)
        else:
            pad = (high - low) * self.axis_padding
        return low - pad, high + pad

    def _drawable_layers(
        self,
        frame: PCNFrame,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray | None],
        list[np.ndarray | None],
        list[str],
    ]:
        states = list(frame.states)
        predictions: list[np.ndarray | None] = list(frame.predictions) + [None]
        errors: list[np.ndarray | None] = list(frame.errors) + [None]
        layer_labels = ["x0 input"] + [f"x{i}" for i in range(1, len(states))]
        if layer_labels:
            layer_labels[-1] = f"{layer_labels[-1]} readout src"

        if frame.output_prediction is None:
            return states, predictions, errors, layer_labels

        layer_labels.append("Y target")
        if frame.output_target is None:
            states.append(frame.output_prediction)
            predictions.append(None)
            errors.append(None)
            return states, predictions, errors, layer_labels

        states.append(frame.output_target)
        predictions.append(frame.output_prediction)
        errors.append(frame.output_target - frame.output_prediction)
        return states, predictions, errors, layer_labels

    def _vertical_spring_curve(
        self,
        x: float,
        y: float,
        z0: float,
        z1: float,
        n_points: int = 42,
    ) -> np.ndarray:
        z = np.linspace(z0, z1, n_points)
        length = abs(z1 - z0)
        if length == 0.0:
            return np.column_stack(
                [
                    np.full(n_points, x),
                    np.full(n_points, y),
                    z,
                ]
            )

        t = np.linspace(0.0, 1.0, n_points)
        radius = min(self.spring_radius, max(0.012, 0.12 * length))
        phase = 2.0 * np.pi * self.spring_coils * t
        taper = np.sin(np.pi * t)

        return np.column_stack(
            [
                x + radius * taper * np.cos(phase),
                y + radius * taper * np.sin(phase),
                z,
            ]
        )

    def _draw_layer_backbone(self, ax, states: list[np.ndarray]) -> None:
        centers = []
        for layer_index, values in enumerate(states):
            _, visible, _ = self._visible_values(values, layer_index)
            centers.append(
                (
                    layer_index * self.layer_gap,
                    0.0,
                    float(np.mean(visible)) if visible.size else 0.0,
                )
            )
        if len(centers) > 1:
            centers = np.asarray(centers)
            ax.plot(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                color="black",
                linewidth=1.0,
                alpha=0.25,
            )

    @staticmethod
    def _is_readout_source(label: str) -> bool:
        return label.endswith(" readout src")

    @staticmethod
    def _prediction_label(layer_index: int, layer_labels: list[str]) -> str:
        target = layer_labels[layer_index]
        if target == "Y target":
            source = next(
                (label for label in reversed(layer_labels) if label.endswith(" readout src")),
                "readout source",
            )
            return f"{target} prediction from {source}"
        if layer_index + 1 < len(layer_labels):
            return f"{target} prediction from {layer_labels[layer_index + 1]}"
        return f"{target} prediction"

    @staticmethod
    def _frame_title(frame: PCNFrame) -> str:
        step = "?" if frame.step is None else frame.step
        return f"PCN inference state | step {step} | energy {frame.energy:.4g}"
