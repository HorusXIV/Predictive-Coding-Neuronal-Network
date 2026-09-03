from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared architecture/training config for the PCN vs. MLP comparison.

    Both models are built from the same ``dims``/``output_dim`` (see
    :mod:`comparison.models`) and trained with the same optimizer settings, so
    a trial's outcome differs only because of the training mechanism itself
    (predictive-coding inference vs. backprop), not because one model was
    given an easier setup.
    """

    dims: tuple[int, ...] = (784, 512, 128)
    output_dim: int = 10
    num_epochs: int = 10
    batch_size: int = 128
    val_size: int = 5000
    lr: float = 1e-3
    weight_decay: float = 0.05
    eta_infer: float = 0.1
    T_infer: int = 100
    eval_every: int = 1
    data_dir: str = "data"
