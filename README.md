# Predictive Coding Networks

A PyTorch implementation of a predictive-coding network (PCN) with a supervised readout, plus tooling to compare it against an architecturally matched MLP and to search its hyperparameters.

## Setup

```bash
uv sync
```

## Layout

```
src/pcn/          the network and a 3D inference visualizer
src/comparison/   PCN vs. equivalent MLP, run repeatedly on MNIST
src/gridsearch/   PCN hyperparameter search
test/             regression tests for src/pcn
experiments/      notebooks: PCN on HerzCheck, MNIST, CIFAR-10
data/             datasets (gitignored)
ressources/       background papers (gitignored)
out/              results from comparison/ and gridsearch/ (gitignored)
```

## src/pcn

`PredictiveCodingNetwork` (in `model.py`) holds one weight matrix per generative layer, used both ways: top-down as `x_hat_l = f(W_l @ x_{l+1})` to predict the layer below, bottom-up through its error to drive inference above. Training alternates two phases per batch: relax every latent state with the input clamped, then take one weight step on the result. Gradients come from the analytic free-energy expressions rather than autograd, but land in `.grad`, so any `torch.optim` optimizer can run the weight phase.

`PCN3DVisualizer` (in `visualizer.py`) renders a recorded inference trajectory as a static 3D plot, a matplotlib animation, or an interactive Plotly figure.

Run the tests with:

```bash
uv run python test/test_pcn.py
```

## experiments/

Three notebooks, each with its own imports and its own data loading:

- `pcn_HerzCheck.ipynb`: heart-disease classification on `data/herzcheck_*.csv`.
- `pcn_MNIST.ipynb`: digit classification and class-conditional generation.
- `pcn_CIFAR.ipynb`: CIFAR-10 classification and class-conditional generation.

## src/comparison

The MLP in `models.py` uses the same layer widths as the PCN, with the same `bias=False` layers, so both networks carry the same parameter count (468,224 for the default `dims=(784, 512, 128)`). Only the training mechanism differs: predictive-coding inference against plain backprop.

```bash
uv run pcn-comparison              # 10 runs each, results go to out/
uv run pcn-comparison --n-runs 3 --device cpu
```

Each run index gets one seed, and that seed drives both the PCN trial and the MLP trial for that index: same train/val split, same image order, same batches. Init and the training mechanism are the only things left to vary, which supports a paired comparison across the ten runs rather than twenty independent draws.

Every trial's summary lands in `out/results.csv`. Its full per-epoch history goes to `out/runs/<model>_run<idx>_seed<seed>.json`.

## src/gridsearch

`eta_infer`, `T_infer`, `lr`, and `weight_decay` in the notebooks above were hand-picked. This searches them instead.

```bash
uv run pcn-gridsearch                          # 100 samples per optimizer, 5-fold CV
uv run pcn-gridsearch --n-samples 20 --num-epochs 5
```

`space.py` draws a separate Latin Hypercube design for each optimizer over the eight non-categorical dimensions: `eta_infer`, `T_infer`, `lr`, `weight_decay`, `negative_slope`, `batch_size`, `hidden_width`, and `n_hidden_layers`. The `--n-samples` value is therefore per optimizer: with Adam, SGD, and Muon, `--n-samples 250` evaluates 750 configurations total. `search.py` scores each candidate with 5-fold cross-validation, all folds shared across every candidate. `data.py` loads only the MNIST train split.

Two tables land in `out/gridsearch/`, joinable on `config_id`:

- `folds.csv`: one row per (config, fold), every metric raw. Model from this one.
- `results.csv`: one row per config, with mean, std, min and max for accuracy, macro F1, cross-entropy, the free energies and the timings.

## Cluster

Build the locked grid-search environment once, then submit the search from the repository root:

```bash
singularity build containers/pcn_uv.sif containers/pcn_uv.def
sbatch cluster/run_pcn_gridsearch.slurm --n-samples 250 --num-epochs 3
```

The Slurm job binds the checkout at `/workspace`, uses the image's pre-synced dependencies, writes CSVs to `out/gridsearch/`, and runs on CUDA.

## Background

- Stenlund, "Introduction to Predictive Coding Networks for Machine Learning" (arXiv:2506.06332)
- Anonymous, "Towards Stable Learning in Predictive Coding Networks" (under review, ICLR 2025)
