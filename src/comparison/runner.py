"""Single-trial training + evaluation for the PCN vs. MLP comparison."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from pcn.model import _f1_from_confusion

from .data import build_mnist_loaders
from .models import build_mlp, build_pcn, count_params
from .seeding import seed_everything


def _mlp_metrics(model, loader, device):
    """Same metric set/definitions as ``PredictiveCodingNetwork.evaluate``, so
    PCN and MLP results are directly comparable (``pcn.model._f1_from_confusion``
    is reused rather than reimplemented)."""
    model.eval()
    n, correct, ce_sum, confusion = 0, 0, 0.0, None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += y.numel()
            num_classes = logits.shape[1]
            counts = torch.bincount(
                y * num_classes + preds, minlength=num_classes**2
            ).reshape(num_classes, num_classes)
            confusion = counts if confusion is None else confusion + counts

    metrics = {"n": n, "accuracy": correct / max(1, n), "cross_entropy": ce_sum / max(1, n)}
    metrics.update(_f1_from_confusion(confusion))
    return metrics


def run_pcn_trial(cfg, seed, device):
    """Train and test one PCN on MNIST for this seed. Returns a result dict."""
    generator = seed_everything(seed)
    train_loader, val_loader, test_loader = build_mnist_loaders(
        cfg.data_dir, cfg.batch_size, cfg.val_size, generator
    )

    model = build_pcn(cfg.dims, cfg.output_dim).to(device)
    n_params = count_params(model)

    t0 = time.perf_counter()
    history = model.fit(
        train_loader,
        num_epochs=cfg.num_epochs,
        eta_infer=cfg.eta_infer,
        T_infer=cfg.T_infer,
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay),
        validation_data=val_loader,
        device=device,
        eval_every=cfg.eval_every,
        progress=False,
    )
    train_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_metrics = model.evaluate(
        test_loader, eta_infer=cfg.eta_infer, T_infer=cfg.T_infer, device=device
    )
    test_time_s = time.perf_counter() - t0

    return {
        "model": "pcn",
        "seed": seed,
        "device": str(device),
        "n_params": n_params,
        "train_time_s": train_time_s,
        "test_time_s": test_time_s,
        "total_time_s": train_time_s + test_time_s,
        "test": test_metrics,
        "history": history,
    }


def run_mlp_trial(cfg, seed, device):
    """Train and test one equivalent MLP on MNIST for this seed. Returns a result dict."""
    generator = seed_everything(seed)
    train_loader, val_loader, test_loader = build_mnist_loaders(
        cfg.data_dir, cfg.batch_size, cfg.val_size, generator
    )

    model = build_mlp(cfg.dims, cfg.output_dim).to(device)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"epoch": [], "train": [], "validation": []}

    t0 = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()

        if (epoch + 1) % cfg.eval_every == 0:
            history["epoch"].append(epoch + 1)
            history["train"].append(_mlp_metrics(model, train_loader, device))
            history["validation"].append(_mlp_metrics(model, val_loader, device))
    train_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_metrics = _mlp_metrics(model, test_loader, device)
    test_time_s = time.perf_counter() - t0

    return {
        "model": "mlp",
        "seed": seed,
        "device": str(device),
        "n_params": n_params,
        "train_time_s": train_time_s,
        "test_time_s": test_time_s,
        "total_time_s": train_time_s + test_time_s,
        "test": test_metrics,
        "history": history,
    }
