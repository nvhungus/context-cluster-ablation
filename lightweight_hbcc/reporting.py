from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_records(paths: list[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files = list(p.rglob("*.json")) + list(p.rglob("*.jsonl"))
        else:
            files = [p]
        for file in files:
            if file.suffix == ".jsonl":
                for line in file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            elif file.suffix == ".json":
                rows.append(json.loads(file.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def pareto_frontier(
    df: pd.DataFrame,
    acc_col: str = "acc1",
    cost_cols: tuple[str, ...] = ("params_total", "latency_ms_b1", "peak_memory_mb_b1"),
) -> pd.DataFrame:
    if df.empty:
        return df
    valid = df.dropna(subset=[acc_col, *cost_cols], how="any").copy()
    keep = []
    for i, row in valid.iterrows():
        dominated = False
        for j, other in valid.iterrows():
            if i == j:
                continue
            better_or_equal = other[acc_col] >= row[acc_col] and all(other[c] <= row[c] for c in cost_cols)
            strictly_better = other[acc_col] > row[acc_col] or any(other[c] < row[c] for c in cost_cols)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return valid.loc[keep].sort_values([acc_col, *cost_cols], ascending=[False, True, True, True])


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_markdown(index=False) if not df.empty else "No records.\n"
    out.write_text(text + "\n", encoding="utf-8")
