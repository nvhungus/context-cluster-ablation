from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightweight_hbcc.reporting import load_records, pareto_frontier, write_markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline/Pareto tables from benchmark JSON records.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="results/pareto.md")
    parser.add_argument("--acc-col", default="acc1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_records(args.inputs)
    if args.acc_col not in df.columns:
        df[args.acc_col] = df.get("test_acc1", df.get("val_acc1"))
    frontier = pareto_frontier(df, acc_col=args.acc_col)
    out = Path(args.output)
    write_markdown_table(frontier, out)
    print(out)


if __name__ == "__main__":
    main()
