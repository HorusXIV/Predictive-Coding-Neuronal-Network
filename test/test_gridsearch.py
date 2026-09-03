"""Checks for the grid-search design and optimizer selection."""

from collections import Counter

import torch

from comparison.models import build_pcn
from gridsearch.search import _build_optimizer
from gridsearch.space import OPTIMIZERS, sample_configs


def test_lhs_draws_n_samples_per_optimizer_and_is_reproducible():
    configs = sample_configs(n_samples=7, seed=123)

    assert len(configs) == 7 * len(OPTIMIZERS)
    assert Counter(config["optimizer"] for config in configs) == {optimizer: 7 for optimizer in OPTIMIZERS}
    assert configs == sample_configs(n_samples=7, seed=123)


def test_muon_is_not_silently_treated_as_sgd():
    model = build_pcn((8, 4), output_dim=3)
    optimizer = _build_optimizer({"optimizer": "muon", "lr": 1e-3, "weight_decay": 1e-2}, model)

    assert isinstance(optimizer, torch.optim.Muon)


def test_unknown_optimizer_is_rejected():
    model = build_pcn((8, 4), output_dim=3)

    try:
        _build_optimizer({"optimizer": "not-an-optimizer", "lr": 1e-3, "weight_decay": 1e-2}, model)
    except ValueError as error:
        assert "Unknown optimizer" in str(error)
    else:
        raise AssertionError("Unknown optimizers must not fall back to SGD")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
