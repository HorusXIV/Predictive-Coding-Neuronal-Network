"""Space-filling hyperparameter sampling for the PCN grid search."""

import numpy as np
from scipy.stats import qmc

# (name, kind, spec), where kind picks how the unit interval is decoded:
#   "float" -> (low, high, log_scale)
#   "int"   -> (low, high, log_scale), rounded
#   "cat"   -> tuple of choices. The optimizer is stratified separately; it
#              is not sampled as a numeric LHS dimension.
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
    ("optimizer", "cat", ("adam", "sgd", "muon")),
]


_OPTIMIZER_DIM = next(dim for dim in SEARCH_SPACE if dim[0] == "optimizer")
OPTIMIZERS = _OPTIMIZER_DIM[2]
_LHS_SPACE = [dim for dim in SEARCH_SPACE if dim[0] != "optimizer"]


def sample_configs(n_samples: int, seed: int) -> list[dict]:
    """Draw ``n_samples`` Latin-Hypercube configurations *per optimizer*.

    The continuous and integer hyperparameters form an independent
    space-filling design for each optimizer.  Thus ``--n-samples 250`` yields
    250 Adam, 250 SGD, and 250 Muon configurations (750 total), rather than
    assigning roughly a third of 250 configurations to each optimizer.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    seed_sequence = np.random.SeedSequence(seed)
    optimizer_seeds = seed_sequence.spawn(len(OPTIMIZERS))
    configs = []
    for optimizer, optimizer_seed in zip(OPTIMIZERS, optimizer_seeds):
        sampler = qmc.LatinHypercube(
            d=len(_LHS_SPACE), seed=np.random.default_rng(optimizer_seed)
        )
        configs.extend(
            {**_decode(point, _LHS_SPACE), "optimizer": optimizer}
            for point in sampler.random(n=n_samples)
        )

    # Avoid running all configurations of one optimizer consecutively, while
    # retaining a fully reproducible search design and config-id assignment.
    shuffle_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
    shuffle_rng.shuffle(configs)
    return configs


def _decode(point, space=SEARCH_SPACE) -> dict:
    config = {}
    for unit_value, (name, kind, spec) in zip(point, space):
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
