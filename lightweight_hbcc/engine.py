from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import DistillationLoss, smooth_one_hot
from .metrics import AverageMeter, accuracy


PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
PROGRESS_NCOLS = 96


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device, strict: bool = True) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=strict)
    return ckpt


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    cy = random.randint(0, height - 1)
    cx = random.randint(0, width - 1)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    return y1, y2, x1, x2


def apply_mixup_cutmix(
    images: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0 and random.random() < cutmix_prob
    if not use_mixup and not use_cutmix:
        return images, target
    index = torch.randperm(images.size(0), device=images.device)
    target_a = smooth_one_hot(target, num_classes, label_smoothing)
    target_b = target_a[index]
    if use_cutmix:
        lam = random.betavariate(cutmix_alpha, cutmix_alpha)
        y1, y2, x1, x2 = _rand_bbox(images.shape[-2], images.shape[-1], lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(images.shape[-2] * images.shape[-1]))
    else:
        lam = random.betavariate(mixup_alpha, mixup_alpha)
        images = images * lam + images[index] * (1.0 - lam)
    mixed_target = target_a * lam + target_b * (1.0 - lam)
    return images, mixed_target


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: DistillationLoss,
    epoch: int,
    teacher: nn.Module | None = None,
    amp: bool = True,
    limit_batches: int | None = None,
    pruning_regularization: tuple[float, float] | None = None,
    progress: bool = True,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    num_classes: int | None = None,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    # Only DKD uses this; standard KD ignores the epoch.
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)
    scaler = GradScaler(device.type, enabled=amp and device.type == "cuda")
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.perf_counter()
    pbar = tqdm(
        loader,
        desc=f"train {epoch}",
        leave=False,
        disable=not progress,
        dynamic_ncols=False,
        ncols=PROGRESS_NCOLS,
        mininterval=0.5,
        bar_format=PROGRESS_BAR_FORMAT,
    )
    for step, (images, target) in enumerate(pbar):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        hard_target = target
        if num_classes is not None:
            images, target = apply_mixup_cutmix(
                images,
                target,
                num_classes=num_classes,
                label_smoothing=criterion.label_smoothing,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                cutmix_prob=cutmix_prob,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images) if teacher is not None else None
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            # hard_target is the pre-MixUp/CutMix label; only DKD consumes it.
            loss = criterion(output, target, teacher_logits, hard_target=hard_target)
            if pruning_regularization and hasattr(model, "pruning_regularization"):
                loss = loss + model.pruning_regularization(*pruning_regularization)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(output.detach(), hard_target, (1,))[0].item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))
        if progress:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{acc_meter.avg:.2f}")
    return {"train_loss": loss_meter.avg, "train_acc1": acc_meter.avg, "train_time_s": time.perf_counter() - start}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
    limit_batches: int | None = None,
    progress: bool = True,
    prefix: str = "val",
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate top-1/top-5; with ``return_predictions`` also return argmax predictions and targets in loader order."""

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    start = time.perf_counter()
    for step, (images, target) in enumerate(
        tqdm(
            loader,
            desc=prefix,
            leave=False,
            disable=not progress,
            dynamic_ncols=False,
            ncols=PROGRESS_NCOLS,
            mininterval=0.5,
            bar_format=PROGRESS_BAR_FORMAT,
        )
    ):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            loss = loss_fn(output, target)
        if output.shape[1] >= 5:
            acc1, acc5 = accuracy(output, target, (1, 5))
            acc5_meter.update(acc5.item(), images.size(0))
        else:
            acc1 = accuracy(output, target, (1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
        if return_predictions:
            all_preds.append(output.argmax(dim=1).cpu())
            all_targets.append(target.cpu())
    metrics = {
        f"{prefix}_loss": loss_meter.avg,
        f"{prefix}_acc1": acc_meter.avg,
        f"{prefix}_time_s": time.perf_counter() - start,
    }
    if acc5_meter.count > 0:
        metrics[f"{prefix}_acc5"] = acc5_meter.avg
    if return_predictions:
        preds = torch.cat(all_preds).numpy() if all_preds else np.zeros(0, dtype=np.int64)
        targets = torch.cat(all_targets).numpy() if all_targets else np.zeros(0, dtype=np.int64)
        return metrics, preds, targets
    return metrics


def write_metrics(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
