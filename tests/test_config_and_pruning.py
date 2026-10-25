from __future__ import annotations

from pathlib import Path

import torch

from lightweight_hbcc.config import load_config
from lightweight_hbcc.models import build_model
from lightweight_hbcc.pruning import derive_pruned_config


def test_pruned_config_materializes_smaller_dims() -> None:
    cfg = load_config("configs/ablations/hbcc_tiny_pruning_mask.yaml")
    model = build_model(cfg)
    masks = []
    for dim in cfg["model"]["embed_dims"]:
        mask = torch.zeros(dim)
        mask[: max(8, dim // 2)] = 1.0
        masks.append(mask)
    pruned = derive_pruned_config(cfg, masks, threshold=0.5, min_channels=8, round_to=8)
    assert pruned["model"]["pruning_mask"] is False
    assert all(new <= old for new, old in zip(pruned["model"]["embed_dims"], cfg["model"]["embed_dims"]))
    smaller = build_model(pruned)
    y = smaller(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, 10)


def test_required_configs_exist() -> None:
    for path in [
        "configs/hbcc_latency_tiny.yaml",
        "configs/hbcc_latency_small.yaml",
        "configs/hbcc_current_reference.yaml",
        "configs/coc_cifar_baseline.yaml",
        "configs/hbcc_accuracy_small.yaml",
        "configs/hbcc_accuracy_medium.yaml",
        "configs/run_plan.yaml",
    ]:
        assert Path(path).exists()
