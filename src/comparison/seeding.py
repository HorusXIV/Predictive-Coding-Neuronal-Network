import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed every RNG that can affect a trial's outcome, for one seed.

    Covers Python's ``random``, NumPy, and Torch (CPU and all CUDA devices),
    which is what makes model init reproducible for a given seed. The
    returned generator should be passed to every ``DataLoader``/``random_split``
    call in the trial, so that image order and batch composition are driven
    by the seed itself rather than by incidental global RNG state left over
    from whatever ran before.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
