"""MNIST loading for the grid search -- train split only, on purpose.

The search must never see the held-out test set, so this module simply never
constructs it: ``torchvision.datasets.MNIST(..., train=False, ...)`` does not
appear anywhere in this package.
"""

import torchvision
from torchvision import transforms

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def load_mnist_train(data_dir: str):
    return torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=_TRANSFORM)
