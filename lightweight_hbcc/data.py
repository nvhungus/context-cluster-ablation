from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
PLANTVILLAGE_MEAN = (0.485, 0.456, 0.406)
PLANTVILLAGE_STD = (0.229, 0.224, 0.225)
FOOD101_MEAN = (0.485, 0.456, 0.406)
FOOD101_STD = (0.229, 0.224, 0.225)
TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD  = (0.229, 0.224, 0.225)


class _TinyImageNetVal(torch.utils.data.Dataset):
    """Tiny-ImageNet val split.

    The official val split uses a flat image directory plus an annotation file
    (val_annotations.txt) that maps each filename to its wnid.  This class
    reads that file and builds a list of (path, class_index) pairs using the
    same wnid→index ordering as the train ImageFolder (sorted alphabetically).
    """

    def __init__(self, root: Path, wnid_to_idx: dict[str, int], transform=None) -> None:
        ann_file = root / "val" / "val_annotations.txt"
        img_dir  = root / "val" / "images"
        self.samples: list[tuple[Path, int]] = []
        for line in ann_file.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            fname, wnid = parts[0].strip(), parts[1].strip()
            if wnid in wnid_to_idx:
                self.samples.append((img_dir / fname, wnid_to_idx[wnid]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def num_classes_for_dataset(name: str) -> int:
    name = name.lower()
    if name == "cifar10":
        return 10
    if name == "cifar100":
        return 100
    if name == "plantvillage":
        return 38
    if name == "food101":
        return 101
    if name == "tiny_imagenet":
        return 200
    if name == "fake":
        return 10
    raise ValueError(f"Unsupported dataset: {name}")


def _transforms(name: str, train: bool, augment: bool, cfg: dict[str, Any] | None = None) -> transforms.Compose:
    cfg = cfg or {}
    name = name.lower()
    if name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif name == "plantvillage":
        mean, std = PLANTVILLAGE_MEAN, PLANTVILLAGE_STD
    elif name == "food101":
        mean, std = FOOD101_MEAN, FOOD101_STD
    elif name == "tiny_imagenet":
        mean, std = TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    ops: list[Any] = []
    if name in ("plantvillage", "food101", "tiny_imagenet"):
        if train and augment:
            ops.extend([
                transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            ops.extend([transforms.Resize(36), transforms.CenterCrop(32)])
    elif train and augment:
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if train and augment:
        randaugment = cfg.get("randaugment", {})
        if isinstance(randaugment, dict) and randaugment.get("enabled", False):
            ops.append(
                transforms.RandAugment(
                    num_ops=int(randaugment.get("num_ops", 2)),
                    magnitude=int(randaugment.get("magnitude", 9)),
                )
            )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if train and augment:
        random_erasing = cfg.get("random_erasing", {})
        if isinstance(random_erasing, dict) and random_erasing.get("p", 0.0) > 0:
            ops.append(
                transforms.RandomErasing(
                    p=float(random_erasing.get("p", 0.25)),
                    scale=tuple(random_erasing.get("scale", [0.02, 0.2])),
                    ratio=tuple(random_erasing.get("ratio", [0.3, 3.3])),
                    value=float(random_erasing.get("value", 0.0)),
                )
            )
    return transforms.Compose(ops)


def _train_val_indices(length: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    if val_size <= 0 or val_size >= length:
        raise ValueError(f"val_size must be between 1 and {length - 1}, got {val_size}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_datasets(
    cfg: dict[str, Any],
    include_test: bool = True,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset | None]:
    name = str(cfg.get("name", "cifar10")).lower()
    root = Path(cfg.get("root", "data"))
    download = bool(cfg.get("download", True))
    augment = bool(cfg.get("augment", True))
    if name == "cifar10":
        train_full = datasets.CIFAR10(
            root=root,
            train=True,
            transform=_transforms(name, True, augment, cfg),
            download=download,
        )
        val_full = datasets.CIFAR10(
            root=root,
            train=True,
            transform=_transforms(name, False, False, cfg),
            download=download,
        )
        test = (
            datasets.CIFAR10(
                root=root,
                train=False,
                transform=_transforms(name, False, False, cfg),
                download=download,
            )
            if include_test
            else None
        )
        val_size = int(cfg.get("val_size", 5000))
        split_seed = int(cfg.get("split_seed", 42))
        train_indices, val_indices = _train_val_indices(len(train_full), val_size, split_seed)
        train = Subset(train_full, train_indices)
        val = Subset(val_full, val_indices)
    elif name == "cifar100":
        train_full = datasets.CIFAR100(
            root=root,
            train=True,
            transform=_transforms(name, True, augment, cfg),
            download=download,
        )
        val_full = datasets.CIFAR100(
            root=root,
            train=True,
            transform=_transforms(name, False, False, cfg),
            download=download,
        )
        test = (
            datasets.CIFAR100(
                root=root,
                train=False,
                transform=_transforms(name, False, False, cfg),
                download=download,
            )
            if include_test
            else None
        )
        val_size = int(cfg.get("val_size", 5000))
        split_seed = int(cfg.get("split_seed", 42))
        train_indices, val_indices = _train_val_indices(len(train_full), val_size, split_seed)
        train = Subset(train_full, train_indices)
        val = Subset(val_full, val_indices)
    elif name == "plantvillage":
        train_dir = cfg.get("train_dir")
        val_dir = cfg.get("val_dir")
        if not train_dir or not val_dir:
            raise ValueError("PlantVillage requires data.train_dir and data.val_dir in config or as overrides.")
        train = datasets.ImageFolder(str(train_dir), transform=_transforms(name, True, augment, cfg))
        val = datasets.ImageFolder(str(val_dir), transform=_transforms(name, False, False, cfg))
        test = (
            datasets.ImageFolder(str(val_dir), transform=_transforms(name, False, False, cfg))
            if include_test
            else None
        )
    elif name == "food101":
        # Official test split (250/class, manually reviewed) is used as test.
        # Validation is carved from the train split (750/class) with a fixed seed.
        train_full = datasets.Food101(
            root=root,
            split="train",
            transform=_transforms(name, True, augment, cfg),
            download=download,
        )
        val_full = datasets.Food101(
            root=root,
            split="train",
            transform=_transforms(name, False, False, cfg),
            download=download,
        )
        test = (
            datasets.Food101(
                root=root,
                split="test",
                transform=_transforms(name, False, False, cfg),
                download=download,
            )
            if include_test
            else None
        )
        val_size = int(cfg.get("val_size", 7575))  # 10 % of 75 750
        split_seed = int(cfg.get("split_seed", 42))
        train_indices, val_indices = _train_val_indices(len(train_full), val_size, split_seed)
        train = Subset(train_full, train_indices)
        val = Subset(val_full, val_indices)
    elif name == "tiny_imagenet":
        # Accepts root pointing to tiny-imagenet-200/ directly, or to its
        # parent (in which case the tiny-imagenet-200/ subdirectory is used).
        tiny_root = root
        if not (tiny_root / "train").is_dir():
            candidate = tiny_root / "tiny-imagenet-200"
            if (candidate / "train").is_dir():
                tiny_root = candidate
        if not (tiny_root / "train").is_dir():
            raise FileNotFoundError(
                f"Tiny-ImageNet 'train/' not found under {root}. "
                "Expected tiny-imagenet-200/train/ after extraction."
            )
        # ImageFolder discovers classes by sorted directory listing; we build
        # train_full first, then reuse its class_to_idx for _TinyImageNetVal
        # so the label mapping is guaranteed to be identical.
        train_full = datasets.ImageFolder(
            str(tiny_root / "train"),
            transform=_transforms(name, True, augment, cfg),
        )
        val_full = datasets.ImageFolder(
            str(tiny_root / "train"),
            transform=_transforms(name, False, False, cfg),
        )
        # Official val set (labeled, 10 k images) is used as the test split.
        test = (
            _TinyImageNetVal(tiny_root, train_full.class_to_idx, transform=_transforms(name, False, False, cfg))
            if include_test else None
        )
        val_size = int(cfg.get("val_size", 10000))  # 10 % of 100 k train
        split_seed = int(cfg.get("split_seed", 42))
        train_indices, val_indices = _train_val_indices(len(train_full), val_size, split_seed)
        train = Subset(train_full, train_indices)
        val = Subset(val_full, val_indices)
    elif name == "fake":
        transform = _transforms("cifar10", False, False, cfg)
        train = datasets.FakeData(size=int(cfg.get("fake_train_size", 512)), image_size=(3, 32, 32), num_classes=10, transform=transform)
        val = datasets.FakeData(size=int(cfg.get("fake_val_size", 128)), image_size=(3, 32, 32), num_classes=10, transform=transform)
        test = (
            datasets.FakeData(
                size=int(cfg.get("fake_test_size", cfg.get("fake_val_size", 128))),
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
            )
            if include_test
            else None
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    train_limit = cfg.get("train_limit")
    val_limit = cfg.get("val_limit")
    test_limit = cfg.get("test_limit")
    if train_limit:
        train = Subset(train, range(min(int(train_limit), len(train))))
    if val_limit:
        val = Subset(val, range(min(int(val_limit), len(val))))
    if test is not None and test_limit:
        test = Subset(test, range(min(int(test_limit), len(test))))
    return train, val, test


def _seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy RNGs in each worker from the torch seed the DataLoader assigned it."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(cfg: dict[str, Any], include_test: bool = True) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_set, val_set, test_set = build_datasets(cfg, include_test=include_test)
    batch_size = int(cfg.get("batch_size", 128))
    val_batch_size = int(cfg.get("val_batch_size", batch_size))
    test_batch_size = int(cfg.get("test_batch_size", val_batch_size))
    workers = int(cfg.get("workers", 2))
    pin_memory = bool(cfg.get("pin_memory", True))
    # Shuffle order and per-worker augmentation RNG both derive from loader_seed.
    loader_seed = int(cfg.get("loader_seed", 0))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=bool(cfg.get("drop_last", True)),
        generator=torch.Generator().manual_seed(loader_seed),
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(loader_seed + 1),
        worker_init_fn=_seed_worker,
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(loader_seed + 2),
            worker_init_fn=_seed_worker,
        )
    return train_loader, val_loader, test_loader
