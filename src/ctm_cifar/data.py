from __future__ import annotations

from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def build_cifar_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset = dataset.lower().strip()
    if dataset not in ("cifar10", "cifar100"):
        raise ValueError("dataset must be cifar10 or cifar100")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    test_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    if dataset == "cifar10":
        DS = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        DS = torchvision.datasets.CIFAR100
        num_classes = 100

    train_ds = DS(root=data_dir, train=True, download=True, transform=train_tf)
    val_ds = DS(root=data_dir, train=False, download=True, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, num_classes

