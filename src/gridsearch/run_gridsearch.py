"""Search PCN hyperparameters with a Latin Hypercube space-filling design,
score each candidate by 5-fold cross-validation on the MNIST *train* split,
and write both a per-fold table and a summary table.

``--n-samples`` is the number of configurations *per optimizer*. With the
three configured optimizers, the command evaluates three times that many
configurations in total.

Test data is never loaded by this command; see :mod:`gridsearch.data`. Every
candidate is scored on the same folds with the same per-fold seeding (see
:func:`gridsearch.search.cross_validate`), so the ranking reflects the
hyperparameters rather than which candidate drew easier folds.

Two files land in ``out/gridsearch/``:

``folds.csv``
    One row per (config, fold), with every validation metric and timing left
    raw. This is the long-format table to model from: real fold-level
    observations, rather than points resampled from an assumed normal.
``results.csv``
    One row per config, with mean, std, min and max for each metric, sorted
    by mean validation accuracy.

Both carry a ``config_id`` column, so a summary row joins back onto its folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .data import load_mnist_train
from .search import aggregate, cross_validate
from .space import dims_for_config, sample_configs

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out" / "gridsearch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_mnist_train(args.data_dir)
    configs = sample_configs(args.n_samples, args.seed)

    print(
        f"device={args.device} n_samples_per_optimizer={args.n_samples} "
        f"total_configs={len(configs)} num_epochs={args.num_epochs} "
        f"train_size={len(dataset)} (5-fold CV, train data only)"
    )

    fold_frames, summary_rows = [], []
    for config_id, config in enumerate(tqdm(configs, desc="configs", unit="cfg")):
        fold_rows = cross_validate(config, dataset, args.num_epochs, args.device, args.seed)

        labelled = {
            "config_id": config_id,
            "dims": str(dims_for_config(config, 784)),
            **config,
        }
        fold_frames.append(pd.DataFrame([{**labelled, **row} for row in fold_rows]))
        summary_rows.append({**labelled, **aggregate(fold_rows)})

        summary = summary_rows[-1]
        print(
            f"[{config_id + 1}/{len(configs)}] {config}\n"
            f"  val_accuracy={summary['mean_val_accuracy']:.4f} +/- {summary['std_val_accuracy']:.4f}"
            f"  val_f1_macro={summary['mean_val_f1_macro']:.4f} +/- {summary['std_val_f1_macro']:.4f}"
            f"  fit={summary['mean_fit_time_s']:.1f}s"
        )

        # Written every config, so an interrupted search keeps what it finished.
        folds = pd.concat(fold_frames, ignore_index=True)
        folds.to_csv(args.out_dir / "folds.csv", index=False)
        table = (
            pd.DataFrame(summary_rows)
            .sort_values("mean_val_accuracy", ascending=False)
            .reset_index(drop=True)
        )
        table.to_csv(args.out_dir / "results.csv", index=False)

    print(f"\nbest config:\n{table.iloc[0]}")
    print(f"\n{len(folds)} fold rows -> {args.out_dir / 'folds.csv'}")
    print(f"{len(table)} config rows -> {args.out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
