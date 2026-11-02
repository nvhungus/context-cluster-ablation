"""Aggregate finished runs across seeds and compare configurations with paired tests.

Every run directory produced by tools/train.py (via tools/run_queue.py) holds metrics.jsonl,
test_metrics.json, run_info.json and test_predictions.npy/test_targets.npy. Runs are grouped by
experiment name without the ``_s{seed}`` suffix.

Examples
--------
    python tools/aggregate_seeds.py runs_a runs_b --output results/resubmission/summary.csv
    python tools/aggregate_seeds.py runs_a runs_b \
        --compare tin_hbcc_medium_ce:tin_allcluster_medium_ce \
        --compare tin_hbcc_medium_kd:tin_resnet18_ce --latex results/resubmission/table.tex

Paired comparisons use the seeds both groups share: mean/std of the per-seed difference, a paired
t-test across seeds (n >= 2), and an exact McNemar test per seed on the shared test images.
Smoke runs (any limit_*_batches set in run_info.json) are excluded unless --include-smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

SEED_SUFFIX = re.compile(r"^(?P<group>.+)_s(?P<seed>\d+)$")


@dataclass
class RunResult:
    group: str
    seed: int
    path: Path
    best_val_acc1: float | None
    best_val_acc5: float | None
    test_acc1: float | None
    test_acc5: float | None
    best_epoch: int | None
    epochs: int | None
    git_commit: str | None
    smoke: bool


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_run(run_dir: Path) -> RunResult | None:
    match = SEED_SUFFIX.match(run_dir.name)
    test = _read_json(run_dir / "test_metrics.json")
    if match is None or not test:
        return None
    info = _read_json(run_dir / "run_info.json")
    val_records = []
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "val_acc1" in record:
                    val_records.append(record)
    best = max(val_records, key=lambda r: r["val_acc1"]) if val_records else {}
    smoke = any(info.get(key) for key in ("limit_train_batches", "limit_val_batches", "limit_test_batches"))
    return RunResult(
        group=match["group"],
        seed=int(match["seed"]),
        path=run_dir,
        best_val_acc1=best.get("val_acc1"),
        best_val_acc5=best.get("val_acc5"),
        test_acc1=test.get("test_acc1"),
        test_acc5=test.get("test_acc5"),
        best_epoch=test.get("epoch"),
        epochs=info.get("epochs"),
        git_commit=info.get("git_commit"),
        smoke=bool(smoke),
    )


def collect(roots: list[Path], include_smoke: bool = False) -> dict[str, dict[int, RunResult]]:
    groups: dict[str, dict[int, RunResult]] = {}
    for root in roots:
        for test_file in sorted(root.rglob("test_metrics.json")):
            run = load_run(test_file.parent)
            if run is None or (run.smoke and not include_smoke):
                continue
            existing = groups.setdefault(run.group, {}).get(run.seed)
            if existing is not None and existing.path != run.path:
                raise ValueError(f"Duplicate run for {run.group} seed {run.seed}: {existing.path} and {run.path}")
            groups[run.group][run.seed] = run
    return groups


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def summarize(groups: dict[str, dict[int, RunResult]]) -> list[dict]:
    rows = []
    for group in sorted(groups):
        runs = [groups[group][seed] for seed in sorted(groups[group])]
        row = {"group": group, "n_seeds": len(runs), "seeds": " ".join(str(r.seed) for r in runs)}
        for key in ("best_val_acc1", "best_val_acc5", "test_acc1", "test_acc5"):
            mean, std = _mean_std([getattr(r, key) for r in runs])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        epochs = sorted({r.epochs for r in runs if r.epochs is not None})
        commits = sorted({r.git_commit for r in runs if r.git_commit})
        row["epochs"] = " ".join(str(e) for e in epochs)
        row["best_epochs"] = " ".join(str(r.best_epoch) for r in runs)
        row["git_commits"] = " ".join(c[:8] for c in commits)
        rows.append(row)
    return rows


def mcnemar_exact(pred_a: np.ndarray, pred_b: np.ndarray, targets: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test: counts of (A right, B wrong), (A wrong, B right) and two-sided p."""

    if not (len(pred_a) == len(pred_b) == len(targets)):
        raise ValueError("Prediction arrays must cover the same test images.")
    a_right = pred_a == targets
    b_right = pred_b == targets
    only_a = int(np.sum(a_right & ~b_right))
    only_b = int(np.sum(~a_right & b_right))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return only_a, only_b, p


def compare(groups: dict[str, dict[int, RunResult]], name_a: str, name_b: str, metric: str = "test_acc1") -> dict:
    if name_a not in groups or name_b not in groups:
        missing = [n for n in (name_a, name_b) if n not in groups]
        raise KeyError(f"Unknown group(s): {missing}")
    seeds = sorted(set(groups[name_a]) & set(groups[name_b]))
    diffs = [getattr(groups[name_a][s], metric) - getattr(groups[name_b][s], metric) for s in seeds]
    result = {
        "a": name_a,
        "b": name_b,
        "metric": metric,
        "seeds": " ".join(map(str, seeds)),
        "n": len(seeds),
        "diff_mean": statistics.fmean(diffs) if diffs else None,
        "diff_std": statistics.stdev(diffs) if len(diffs) > 1 else None,
        "a_wins": sum(d > 0 for d in diffs),
        "paired_t_p": None,
        "mcnemar": [],
    }
    if len(diffs) >= 2 and any(d != diffs[0] for d in diffs):
        result["paired_t_p"] = float(stats.ttest_rel(
            [getattr(groups[name_a][s], metric) for s in seeds],
            [getattr(groups[name_b][s], metric) for s in seeds],
        ).pvalue)
    for seed in seeds:
        dir_a, dir_b = groups[name_a][seed].path, groups[name_b][seed].path
        files = [dir_a / "test_predictions.npy", dir_b / "test_predictions.npy", dir_a / "test_targets.npy", dir_b / "test_targets.npy"]
        if not all(f.exists() for f in files):
            continue
        targets_a, targets_b = np.load(files[2]), np.load(files[3])
        if not np.array_equal(targets_a, targets_b):
            raise ValueError(f"Test targets differ between {dir_a} and {dir_b}; not the same test set/order.")
        only_a, only_b, p = mcnemar_exact(np.load(files[0]), np.load(files[1]), targets_a)
        result["mcnemar"].append({"seed": seed, "only_a": only_a, "only_b": only_b, "p": p})
    return result


def _fmt(mean: float | None, std: float | None, digits: int = 2) -> str:
    if mean is None:
        return "--"
    if std is None or math.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex(rows: list[dict], path: Path) -> None:
    lines = ["% group & seeds & val top-1 & test top-1 & test top-5 \\\\"]
    for row in rows:
        name = row["group"].replace("_", r"\_")
        lines.append(
            f"{name} & {row['n_seeds']} & "
            f"{_fmt(row['best_val_acc1_mean'], row['best_val_acc1_std'])} & "
            f"{_fmt(row['test_acc1_mean'], row['test_acc1_std'])} & "
            f"{_fmt(row['test_acc5_mean'], row['test_acc5_std'])} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate runs across seeds and run paired comparisons.")
    parser.add_argument("roots", nargs="+", help="Directories containing run directories (searched recursively).")
    parser.add_argument("--output", default="results/resubmission/summary.csv")
    parser.add_argument("--compare", action="append", default=[], help="GROUP_A:GROUP_B (A minus B).")
    parser.add_argument("--metric", default="test_acc1", choices=["test_acc1", "best_val_acc1", "test_acc5"])
    parser.add_argument("--latex")
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = collect([Path(r) for r in args.roots], include_smoke=args.include_smoke)
    rows = summarize(groups)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(
            f"{row['group']:42s} n={row['n_seeds']}  val={_fmt(row['best_val_acc1_mean'], row['best_val_acc1_std'])}"
            f"  test={_fmt(row['test_acc1_mean'], row['test_acc1_std'])}  epochs={row['epochs']}"
        )
    comparisons = [compare(groups, *spec.split(":", 1), metric=args.metric) for spec in args.compare]
    for c in comparisons:
        print(
            f"\n{c['a']} - {c['b']} [{c['metric']}] over seeds {c['seeds']}: "
            f"mean diff={c['diff_mean']}, std={c['diff_std']}, A wins {c['a_wins']}/{c['n']}, paired t p={c['paired_t_p']}"
        )
        for m in c["mcnemar"]:
            print(f"  seed {m['seed']}: McNemar only_a={m['only_a']} only_b={m['only_b']} p={m['p']:.4g}")
    if comparisons:
        out.with_name(out.stem + "_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    if args.latex:
        write_latex(rows, Path(args.latex))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
