from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightweight_hbcc.config import apply_overrides, load_config
from lightweight_hbcc.engine import load_checkpoint, resolve_device
from lightweight_hbcc.models import build_model
from lightweight_hbcc.pruning import collect_stage_masks, summarize_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect learned pruning masks and keep counts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--min-channels", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device, strict=False)
    masks = collect_stage_masks(model)
    if not masks:
        raise ValueError("No pruning masks found. Use a config with model.pruning_mask=true.")
    rows = summarize_masks(masks, args.thresholds, round_to=args.round_to, min_channels=args.min_channels)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    headers = list(rows[0].keys())
    print("\t".join(headers))
    for row in rows:
        print("\t".join(f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h]) for h in headers))


if __name__ == "__main__":
    main()
