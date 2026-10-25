from __future__ import annotations

import torch

from lightweight_hbcc import data


class DummyCIFAR(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, transform=None, download: bool = False) -> None:
        self.train = train
        self.transform = transform
        self.download = download

    def __len__(self) -> int:
        return 50_000 if self.train else 10_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), index % 10


class DummyCIFAR100(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, transform=None, download: bool = False) -> None:
        self.train = train
        self.transform = transform
        self.download = download

    def __len__(self) -> int:
        return 50_000 if self.train else 10_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), index % 100


def test_cifar10_uses_train_val_test_split(monkeypatch) -> None:
    monkeypatch.setattr(data.datasets, "CIFAR10", DummyCIFAR)

    train, val, test = data.build_datasets(
        {"name": "cifar10", "download": False, "val_size": 5000, "split_seed": 123}
    )

    assert len(train) == 45_000
    assert len(val) == 5_000
    assert len(test) == 10_000
    assert set(train.indices).isdisjoint(set(val.indices))


def test_cifar10_split_is_reproducible(monkeypatch) -> None:
    monkeypatch.setattr(data.datasets, "CIFAR10", DummyCIFAR)

    _, val_a, _ = data.build_datasets({"name": "cifar10", "download": False, "split_seed": 123})
    _, val_b, _ = data.build_datasets({"name": "cifar10", "download": False, "split_seed": 123})
    _, val_c, _ = data.build_datasets({"name": "cifar10", "download": False, "split_seed": 321})

    assert val_a.indices == val_b.indices
    assert val_a.indices != val_c.indices


def test_cifar100_uses_train_val_test_split(monkeypatch) -> None:
    monkeypatch.setattr(data.datasets, "CIFAR100", DummyCIFAR100)

    train, val, test = data.build_datasets(
        {"name": "cifar100", "download": False, "val_size": 5000, "split_seed": 123}
    )

    assert data.num_classes_for_dataset("cifar100") == 100
    assert len(train) == 45_000
    assert len(val) == 5_000
    assert len(test) == 10_000
    assert set(train.indices).isdisjoint(set(val.indices))


def test_train_loader_order_follows_loader_seed() -> None:
    cfg = {"name": "fake", "fake_train_size": 64, "fake_val_size": 8, "batch_size": 8, "workers": 0}

    def epoch_labels(seed: int) -> list[int]:
        train_loader, _, _ = data.build_loaders({**cfg, "loader_seed": seed}, include_test=False)
        return [label for _, labels in train_loader for label in labels.tolist()]

    assert epoch_labels(5) == epoch_labels(5)
    assert epoch_labels(5) != epoch_labels(6)


def test_dataset_builder_can_skip_test_split() -> None:
    train, val, test = data.build_datasets(
        {"name": "fake", "fake_train_size": 8, "fake_val_size": 4},
        include_test=False,
    )

    assert len(train) == 8
    assert len(val) == 4
    assert test is None
