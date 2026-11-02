from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightweight_hbcc.config import apply_overrides, load_config
from lightweight_hbcc.engine import load_checkpoint, resolve_device
from lightweight_hbcc.models import build_model
from lightweight_hbcc.pruning import export_pruned_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a materialized smaller config from learned channel masks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-channels", type=int, default=8)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device, strict=False)
    pruned = export_pruned_config(
        cfg,
        model,
        Path(args.output),
        threshold=args.threshold,
        min_channels=args.min_channels,
        round_to=args.round_to,
    )
    print(pruned)


if __name__ == "__main__":
    main()
