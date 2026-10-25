from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize train/val/test metrics from a run directory.")
    parser.add_argument("run_dir", help="Directory containing metrics.jsonl and optionally test_metrics.json.")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    rows = _load_jsonl(metrics_path)
    val_rows = [row for row in rows if "val_acc1" in row]
    if not val_rows:
        raise RuntimeError(f"No validation records found in {metrics_path}")
    best = max(val_rows, key=lambda row: row["val_acc1"])

    test_path = run_dir / "test_metrics.json"
    test_records = [row for row in rows if "test_acc1" in row]
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None
    if test is None and test_records:
        test = next(
            (row for row in reversed(test_records) if row.get("epoch") == best.get("epoch")),
            test_records[-1],
        )

    line = "Best val: epoch={epoch}, val_acc1={val_acc1:.2f}, val_loss={val_loss:.4f}".format(**best)
    if "val_acc5" in best:
        line += " val_acc5={val_acc5:.2f}".format(**best)
    print(line)
    if test is not None:
        line = (
            "Held-out test: checkpoint={checkpoint}, epoch={epoch}, test_acc1={test_acc1:.2f}, "
            "test_loss={test_loss:.4f}".format(**test)
        )
        if "test_acc5" in test:
            line += " test_acc5={test_acc5:.2f}".format(**test)
        print(line)
    else:
        line = "Reported validation: epoch={epoch}, acc1={val_acc1:.2f}, loss={val_loss:.4f}".format(**best)
        if "val_acc5" in best:
            line += " acc5={val_acc5:.2f}".format(**best)
        print(line)


if __name__ == "__main__":
    main()
