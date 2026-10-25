from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / target.numel()))
    return result


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


def model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def flop_count(model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> dict[str, Any]:
    model_was_training = model.training
    model.eval()
    x = torch.randn(input_shape, device=device)
    out: dict[str, Any] = {"macs": None, "flops": None, "unsupported_ops": None, "flop_error": None}
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, x)
        out["flops"] = int(analysis.total())
        out["macs"] = int(analysis.total())
        out["unsupported_ops"] = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
    except Exception as exc:  # noqa: BLE001
        out["flop_error"] = repr(exc)
    finally:
        model.train(model_was_training)
    return out


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
