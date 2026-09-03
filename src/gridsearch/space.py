"""Space-filling hyperparameter sampling for the PCN grid search."""

import numpy as np
from scipy.stats import qmc

# name, low, high, log_scale. Ranges are centered on the values already used
# successfully in experiments/pcn_MNIST.ipynb (eta_infer=0.1, T_infer=100,
# lr=1e-3, weight_decay=0.05), widened enough either side to actually search.
SEARCH_SPACE = [
    ("eta_infer", 0.01, 0.5, True),
    ("T_infer", 10, 150, False),
    ("lr", 1e-4, 1e-2, True),
    ("weight_decay", 1e-4, 0.2, True),
]


def sample_configs(n_samples: int, seed: int) -> list[dict]:
    """Draw ``n_samples`` hyperparameter combinations via Latin Hypercube
    sampling: a space-filling design that spreads points evenly across the
    whole search space, unlike plain random search (which can clump/leave
    gaps) or a fixed grid (which scales exponentially with dimensions).
    """
    sampler = qmc.LatinHypercube(d=len(SEARCH_SPACE), seed=seed)
    unit_points = sampler.random(n=n_samples)

    configs = []
    for point in unit_points:
        config = {}
        for value, (name, low, high, log_scale) in zip(point, SEARCH_SPACE):
            if log_scale:
                scaled = np.exp(np.log(low) + value * (np.log(high) - np.log(low)))
            else:
                scaled = low + value * (high - low)
            config[name] = int(round(scaled)) if name == "T_infer" else float(scaled)
        configs.append(config)
    return configs
