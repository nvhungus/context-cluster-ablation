from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from .config import save_config


def collect_stage_masks(model: torch.nn.Module) -> list[torch.Tensor]:
    masks = []
    for stage in getattr(model, "stages", []):
        gate = getattr(stage, "gate", None)
        if hasattr(gate, "mask"):
            masks.append(gate.mask().detach().cpu())
    return masks


def derive_pruned_config(
    base_config: dict[str, Any],
    masks: list[torch.Tensor],
    threshold: float = 0.5,
    min_channels: int = 8,
    round_to: int = 8,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    model_cfg = cfg.setdefault("model", {})
    old_dims = model_cfg.get("embed_dims")
    if not old_dims:
        raise ValueError("Pruning export requires explicit model.embed_dims in the config.")
    new_dims = []
    for old_dim, mask in zip(old_dims, masks, strict=False):
        keep = int((mask >= threshold).sum().item())
        keep = max(min_channels, keep)
        keep = min(int(old_dim), ((keep + round_to - 1) // round_to) * round_to)
        new_dims.append(keep)
    if len(new_dims) != len(old_dims):
        raise ValueError("Number of pruning masks does not match model.embed_dims.")
    model_cfg["embed_dims"] = new_dims
    model_cfg["pruning_mask"] = False
    cfg.setdefault("notes", "")
    cfg["notes"] = f"Materialized pruned config from masks with threshold={threshold}."
    return cfg


def export_pruned_config(
    base_config: dict[str, Any],
    model: torch.nn.Module,
    out_path: str | Path,
    threshold: float = 0.5,
    min_channels: int = 8,
    round_to: int = 8,
) -> dict[str, Any]:
    masks = collect_stage_masks(model)
    if not masks:
        raise ValueError("Model has no ChannelGate masks. Enable model.pruning_mask first.")
    cfg = derive_pruned_config(base_config, masks, threshold, min_channels, round_to)
    cfg.setdefault("experiment", {})["name"] = Path(out_path).stem
    save_config(cfg, out_path)
    return cfg


def summarize_masks(
    masks: list[torch.Tensor],
    thresholds: list[float],
    round_to: int = 8,
    min_channels: int = 8,
) -> list[dict[str, Any]]:
    rows = []
    for stage_idx, mask in enumerate(masks):
        stage = {
            "stage": stage_idx,
            "channels": int(mask.numel()),
            "min": float(mask.min().item()),
            "max": float(mask.max().item()),
            "mean": float(mask.mean().item()),
        }
        for threshold in thresholds:
            keep = int((mask >= threshold).sum().item())
            rounded_keep = max(min_channels, ((keep + round_to - 1) // round_to) * round_to)
            rounded_keep = min(int(mask.numel()), rounded_keep)
            stage[f"keep@{threshold:g}"] = keep
            stage[f"rounded_keep@{threshold:g}"] = rounded_keep
        rows.append(stage)
    return rows
