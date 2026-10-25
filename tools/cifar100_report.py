from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CIFAR-100 train/test and benchmark metrics into one CSV.")
    parser.add_argument("--runs", nargs="+", default=["runs_cifar100"])
    parser.add_argument("--benchmarks", default="results/benchmark_cifar100")
    parser.add_argument("--output", default="results/cifar100_summary.csv")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _benchmark_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for item in path.glob("*.json"):
        record = json.loads(item.read_text(encoding="utf-8"))
        config_id = record.get("config_id")
        if isinstance(config_id, str):
            records[config_id] = record
    return records


def _run_record(run_dir: Path, benchmarks: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    val_rows = [row for row in rows if "val_acc1" in row]
    if not val_rows:
        return None
    best = max(val_rows, key=lambda row: row["val_acc1"])
    test_path = run_dir / "test_metrics.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
    bench = benchmarks.get(run_dir.name, {})
    return {
        "experiment": run_dir.name,
        "best_epoch": best.get("epoch"),
        "best_val_acc1": best.get("val_acc1"),
        "best_val_acc5": best.get("val_acc5"),
        "best_val_loss": best.get("val_loss"),
        "test_acc1": test.get("test_acc1"),
        "test_acc5": test.get("test_acc5"),
        "test_loss": test.get("test_loss"),
        "params_total": bench.get("params_total"),
        "model_size_mb": bench.get("model_size_mb"),
        "latency_ms_b1": bench.get("latency_ms_b1"),
        "latency_ms_b16": bench.get("latency_ms_b16"),
        "latency_ms_b64": bench.get("latency_ms_b64"),
        "latency_ms_b128": bench.get("latency_ms_b128"),
        "throughput_b128": bench.get("throughput_b128"),
        "peak_memory_mb": bench.get("peak_memory_mb"),
    }


def main() -> None:
    args = parse_args()
    run_roots = [Path(item) for item in args.runs]
    benchmarks = _benchmark_records(Path(args.benchmarks))
    records: list[dict[str, Any]] = []
    for run_root in run_roots:
        if run_root.exists():
            for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
                record = _run_record(run_dir, benchmarks)
                if record is not None:
                    records.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "best_epoch",
        "best_val_acc1",
        "best_val_acc5",
        "best_val_loss",
        "test_acc1",
        "test_acc5",
        "test_loss",
        "params_total",
        "model_size_mb",
        "latency_ms_b1",
        "latency_ms_b16",
        "latency_ms_b64",
        "latency_ms_b128",
        "throughput_b128",
        "peak_memory_mb",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {output} rows={len(records)}")


if __name__ == "__main__":
    main()
