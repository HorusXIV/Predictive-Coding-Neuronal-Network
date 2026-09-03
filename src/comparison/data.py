import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def build_mnist_loaders(data_dir, batch_size, val_size, generator):
    """Load MNIST and split it into train/val/test loaders.

    ``generator`` drives both the train/val split and the train loader's
    shuffling, so the exact data order for a trial is reproducible from its
    seed alone (see :func:`comparison.seeding.seed_everything`).
    """
    train_full = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=_TRANSFORM
    )
    test = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=_TRANSFORM
    )

    train, val = random_split(
        train_full, [len(train_full) - val_size, val_size], generator=generator
    )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader
