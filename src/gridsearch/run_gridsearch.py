"""Search PCN hyperparameters (eta_infer, T_infer, lr, weight_decay) with a
Latin Hypercube space-filling design, score each config by 5-fold
cross-validation on the MNIST *train* split, and write a results table.

Test data is never loaded by this command -- see :mod:`gridsearch.data`. The
same fold split and per-fold seeding is reused across every sampled config
(see :func:`gridsearch.search.cross_validate`), so the table's ranking
reflects the hyperparameters, not which config got lucky folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import load_mnist_train
from .search import cross_validate
from .space import sample_configs

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out" / "gridsearch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_mnist_train(args.data_dir)
    configs = sample_configs(args.n_samples, args.seed)

    print(
        f"device={args.device} n_samples={args.n_samples} num_epochs={args.num_epochs} "
        f"train_size={len(dataset)} (5-fold CV, train data only)"
    )

    rows = []
    for i, config in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] {config}")
        result = cross_validate(config, dataset, args.num_epochs, args.batch_size, args.device, args.seed)
        rows.append({**config, **result})
        print(
            f"  mean_val_accuracy={result['mean_val_accuracy']:.4f} "
            f"+/- {result['std_val_accuracy']:.4f}  ({result['elapsed_s']:.1f}s)"
        )

    table = pd.DataFrame(rows).sort_values("mean_val_accuracy", ascending=False).reset_index(drop=True)
    csv_path = args.out_dir / "results.csv"
    table.to_csv(csv_path, index=False)

    print(f"\nbest config:\n{table.iloc[0]}")
    print(f"\nfull results table written to {csv_path}")


if __name__ == "__main__":
    main()
