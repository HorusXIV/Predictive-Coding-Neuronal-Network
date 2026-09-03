"""Train and test the PCN and its equivalent MLP ``n_runs`` times each on
MNIST, logging every trial to ``out/`` (see :mod:`comparison.logging_utils`).

Each run index gets its own seed, and that seed drives both the PCN trial and
the MLP trial: same train/val split, same image order, same batch
composition. Only the model init and the training mechanism (predictive-coding
inference vs. backprop) differ, which is what makes the 10 paired runs usable
for a statistical comparison afterwards (e.g. a paired test on accuracy, or on
``total_time_s``, between the two model columns of ``out/results.csv``).

Nothing here forces CUDA/cuDNN into deterministic mode, so GPU training keeps
its own run-to-run kernel-selection noise on top of the seeded randomness --
by design, since that is itself one of the sources of variation these repeated
runs are meant to capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .logging_utils import log_trial
from .runner import run_mlp_trial, run_pcn_trial

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    print(
        f"device={args.device} dims={cfg.dims} output_dim={cfg.output_dim} "
        f"n_runs={args.n_runs} base_seed={args.base_seed} out_dir={args.out_dir}"
    )

    for run_idx in tqdm(range(args.n_runs), desc="running trials", unit="run"):
        seed = args.base_seed + run_idx

        pcn_result = run_pcn_trial(cfg, seed, args.device)
        log_trial(pcn_result, run_idx, args.out_dir)
        print(
            f"run {run_idx:02d} pcn seed={seed} "
            f"accuracy={pcn_result['test']['accuracy']:.4f} "
            f"time={pcn_result['total_time_s']:.1f}s"
        )

        mlp_result = run_mlp_trial(cfg, seed, args.device)
        log_trial(mlp_result, run_idx, args.out_dir)
        print(
            f"run {run_idx:02d} mlp seed={seed} "
            f"accuracy={mlp_result['test']['accuracy']:.4f} "
            f"time={mlp_result['total_time_s']:.1f}s"
        )

    print(f"done -- results in {args.out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
