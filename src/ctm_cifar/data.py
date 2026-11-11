from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

# Per-dataset channel statistics (training set).
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic train/validation split of range(n)."""
    if not 0 <= val_size < n:
        raise ValueError(f"val_size must be in [0, {n})")
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    return sorted(perm[val_size:]), sorted(perm[:val_size])


def build_cifar_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_size: int = 5000,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns (train, val, test, num_classes).

    The validation set is ``val_size`` images held out from the training set (without augmentation), so
    model/epoch/tick selection never touches the test set. ``val_size=0`` reproduces the old behaviour of
    validating on the test set (then ``val`` and ``test`` are the same loader).
    """
    dataset = dataset.lower().strip()
    if dataset == "fake":  # tiny random dataset for smoke tests / CI, no download
        tf = T.ToTensor()
        train_full = torchvision.datasets.FakeData(256 + val_size, (3, 32, 32), 10, tf, random_offset=0)
        test_ds = torchvision.datasets.FakeData(128, (3, 32, 32), 10, tf, random_offset=10_000)
        train_idx, val_idx = split_indices(len(train_full), val_size, seed) if val_size > 0 else (None, None)
        if val_size > 0:
            train_ds, val_ds = Subset(train_full, train_idx), Subset(train_full, val_idx)
        else:
            train_ds, val_ds = train_full, test_ds
        return (*make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers, seed), 10)
    if dataset not in STATS:
        raise ValueError("dataset must be cifar10, cifar100 or fake")
    mean, std = STATS[dataset]

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    DS = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    num_classes = 10 if dataset == "cifar10" else 100

    train_full = DS(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = DS(root=data_dir, train=False, download=True, transform=test_tf)
    if val_size > 0:
        train_idx, val_idx = split_indices(len(train_full), val_size, seed)
        train_ds: Dataset = Subset(train_full, train_idx)
        val_ds: Dataset = Subset(DS(root=data_dir, train=True, download=False, transform=test_tf), val_idx)
    else:
        train_ds, val_ds = train_full, test_ds

    return (*make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers, seed), num_classes)


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int, seed: int):
    g = torch.Generator()
    g.manual_seed(int(seed))
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = val_loader if test_ds is val_ds else DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
