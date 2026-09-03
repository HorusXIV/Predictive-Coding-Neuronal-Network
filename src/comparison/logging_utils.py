"""Result logging: a flat CSV of per-trial summaries plus one full JSON per trial."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

_CSV_COLUMNS = [
    "run_idx", "model", "seed", "timestamp", "device", "n_params",
    "train_time_s", "test_time_s", "total_time_s",
    "test_accuracy", "test_cross_entropy", "test_f1_macro", "test_n",
]


def log_trial(result: dict, run_idx: int, out_dir: Path) -> Path:
    """Append the trial's summary row to ``out_dir/results.csv`` and write its
    full detail (config-level fields, complete test metrics, epoch history) to
    ``out_dir/runs/<model>_run<run_idx>_seed<seed>.json``. Returns the JSON path.
    """
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    test = result["test"]
    row = {
        "run_idx": run_idx,
        "model": result["model"],
        "seed": result["seed"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": result["device"],
        "n_params": result["n_params"],
        "train_time_s": result["train_time_s"],
        "test_time_s": result["test_time_s"],
        "total_time_s": result["total_time_s"],
        "test_accuracy": test.get("accuracy"),
        "test_cross_entropy": test.get("cross_entropy"),
        "test_f1_macro": test.get("f1_macro"),
        "test_n": test.get("n"),
    }

    csv_path = out_dir / "results.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    json_path = runs_dir / f"{result['model']}_run{run_idx:02d}_seed{result['seed']}.json"
    with json_path.open("w") as f:
        json.dump({**row, "test_full": test, "history": result["history"]}, f, indent=2)

    return json_path
