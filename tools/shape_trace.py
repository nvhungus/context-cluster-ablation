from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from lightweight_hbcc.config import apply_overrides, load_config
from lightweight_hbcc.engine import resolve_device
from lightweight_hbcc.models import build_model
from lightweight_hbcc.models.cluster import ContextClusterOp, Stage
from lightweight_hbcc.models.layers import PointReducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace model feature shapes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(args.device)
    model = build_model(cfg).to(device).eval()
    image_size = int(cfg.get("data", {}).get("image_size", 32))
    hooks = []

    def hook(name: str):
        def _hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            print(f"{name:48s} -> {tuple(output.shape)}")

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, (PointReducer, Stage, ContextClusterOp)):
            hooks.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        y = model(x)
    for handle in hooks:
        handle.remove()
    print(f"{'logits':48s} -> {tuple(y.shape)}")


if __name__ == "__main__":
    main()
