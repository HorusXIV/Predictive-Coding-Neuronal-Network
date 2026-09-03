"""5-fold cross-validated scoring of one PCN hyperparameter configuration."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from comparison.models import build_pcn
from comparison.seeding import seed_everything

DIMS = (784, 512, 128)
OUTPUT_DIM = 10
N_SPLITS = 5


def cross_validate(config, dataset, num_epochs, batch_size, device, seed):
    """Train+validate a PCN built with ``config`` across 5 folds of ``dataset``.

    ``dataset`` must already be the train split -- this function never touches
    a test set. ``seed`` fixes both the fold split and every fold's model
    init/data order; passing the *same* seed for every sampled config (as
    :mod:`gridsearch.run_gridsearch` does) means all configs are compared on
    identical folds, so only the hyperparameters differ between rows.
    """
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    fold_accuracy, fold_cross_entropy = [], []
    t0 = time.perf_counter()
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        generator = seed_everything(seed * 100 + fold_idx)

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, generator=generator
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=256, shuffle=False)

        model = build_pcn(DIMS, OUTPUT_DIM).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        model.fit(
            train_loader,
            num_epochs=num_epochs,
            eta_infer=config["eta_infer"],
            T_infer=config["T_infer"],
            optimizer=optimizer,
            device=device,
            record_history=False,
            progress=False,
        )
        metrics = model.evaluate(
            val_loader, eta_infer=config["eta_infer"], T_infer=config["T_infer"], device=device
        )
        fold_accuracy.append(metrics["accuracy"])
        fold_cross_entropy.append(metrics["cross_entropy"])

    return {
        "mean_val_accuracy": float(np.mean(fold_accuracy)),
        "std_val_accuracy": float(np.std(fold_accuracy)),
        "mean_val_cross_entropy": float(np.mean(fold_cross_entropy)),
        "elapsed_s": time.perf_counter() - t0,
    }
