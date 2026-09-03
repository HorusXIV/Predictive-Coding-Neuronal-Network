"""Space-filling hyperparameter sampling for the PCN grid search."""

import numpy as np
from scipy.stats import qmc

# (name, kind, spec), where kind picks how the unit interval is decoded:
#   "float" -> (low, high, log_scale)
#   "int"   -> (low, high, log_scale), rounded
#   "cat"   -> tuple of choices, split into equal slices
#
# Continuous ranges are log-scaled wherever they span orders of magnitude, so
# the design spreads points evenly across the exponent rather than piling them
# near the top of the range.
SEARCH_SPACE = [
    ("eta_infer", "float", (0.01, 0.5, True)),
    ("T_infer", "int", (10, 150, False)),
    ("lr", "float", (1e-4, 1e-2, True)),
    ("weight_decay", "float", (1e-4, 0.2, True)),
    # Leaky slope, not fixed at pcn.model.NEGATIVE_SLOPE: it sets how much
    # top-down error survives a non-positive pre-activation, which is what
    # keeps the inference relaxation from stalling.
    ("negative_slope", "float", (0.01, 0.3, True)),
    ("batch_size", "int", (32, 512, True)),
    ("hidden_width", "int", (128, 1024, True)),
    ("n_hidden_layers", "int", (1, 3, False)),
    ("optimizer", "cat", ("adam", "sgd")),
]


def sample_configs(n_samples: int, seed: int) -> list[dict]:
    """Draw ``n_samples`` hyperparameter combinations via Latin Hypercube
    sampling: a space-filling design that spreads points evenly across the
    whole space, unlike plain random search (which clumps and leaves gaps) or
    a fixed grid (which grows exponentially with the number of dimensions).
    """
    sampler = qmc.LatinHypercube(d=len(SEARCH_SPACE), seed=seed)
    return [_decode(point) for point in sampler.random(n=n_samples)]


def _decode(point) -> dict:
    config = {}
    for unit_value, (name, kind, spec) in zip(point, SEARCH_SPACE):
        if kind == "cat":
            config[name] = spec[min(int(unit_value * len(spec)), len(spec) - 1)]
            continue
        low, high, log_scale = spec
        if log_scale:
            value = np.exp(np.log(low) + unit_value * (np.log(high) - np.log(low)))
        else:
            value = low + unit_value * (high - low)
        config[name] = int(round(value)) if kind == "int" else float(value)
    return config


def dims_for_config(config: dict, input_dim: int) -> tuple[int, ...]:
    """Turn ``hidden_width``/``n_hidden_layers`` into a ``dims`` tuple.

    Hidden widths halve as they go up, so ``hidden_width=512`` with three
    layers gives ``(input_dim, 512, 256, 128)``.
    """
    widths = [max(16, config["hidden_width"] // 2**i) for i in range(config["n_hidden_layers"])]
    return (input_dim, *widths)
