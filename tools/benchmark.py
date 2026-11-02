from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchvision import models as tv_models

from lightweight_hbcc.config import apply_overrides, load_config
from lightweight_hbcc.engine import load_checkpoint, resolve_device
from lightweight_hbcc.models import build_model
from lightweight_hbcc.profiler import (
    append_record_csv,
    benchmark_model,
    host_info,
    model_static_metrics,
    profile_operators,
    write_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark latency (eager + CUDA Graph), throughput, memory, MACs and operator profile.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Model config (YAML) built through the model registry.")
    source.add_argument(
        "--torchvision",
        help="Standard torchvision architecture (e.g. resnet18, mobilenet_v2, shufflenet_v2_x1_0), random init.",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Classifier size for --torchvision models.")
    parser.add_argument("--image-size", type=int, default=None, help="Input resolution; defaults to data.image_size or 32.")
    parser.add_argument("--name", help="Record name; defaults to experiment.name, config stem or torchvision name.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 128])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3, help="Eager timed loops per batch size; the median is reported.")
    parser.add_argument("--graph-repeats", type=int, default=5, help="CUDA-Graph timed loops per batch size (median).")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.config:
        cfg = apply_overrides(load_config(args.config), args.override)
        model = build_model(cfg).to(device)
        image_size = args.image_size or int(cfg.get("data", {}).get("image_size", 32))
        name = args.name or cfg.get("experiment", {}).get("name", Path(args.config).stem)
        model_name = cfg.get("model", {}).get("name", cfg.get("model", {}).get("type"))
    else:
        factory = getattr(tv_models, args.torchvision)
        model = factory(weights=None, num_classes=args.num_classes or 1000).to(device)
        image_size = args.image_size or 224
        name = args.name or f"{args.torchvision}_{image_size}"
        model_name = f"torchvision.{args.torchvision}"
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device, strict=False)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"model_name": model_name, "config_id": name, "image_size": image_size, **host_info(device)}
    record.update(model_static_metrics(model, device, image_size=image_size))
    record.update(
        benchmark_model(
            model,
            device,
            args.batch_sizes,
            image_size=image_size,
            warmup=args.warmup,
            runs=args.runs,
            repeats=args.repeats,
            cuda_graph=args.cuda_graph,
            graph_repeats=args.graph_repeats,
        )
    )
    if args.profile:
        profile_path = output_dir / f"{name}_profile.txt"
        profile_operators(model, device, batch_size=1, image_size=image_size, out_path=profile_path)
        record["operator_profile"] = str(profile_path)
    out_path = output_dir / f"{name}.json"
    write_record(record, out_path)
    append_record_csv(record, output_dir / "summary.csv")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
