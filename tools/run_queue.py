"""Run training jobs from configs/run_plan.yaml by run ID.

Examples
--------
List the plan with estimated T4 hours:
    python tools/run_queue.py --list

Run the gate job on account A (Kaggle):
    python tools/run_queue.py --ids A1 A2 A3 --tin-root /kaggle/input/tiny-imagenet/tiny-imagenet-200 --output runs

Smoke-test a set of IDs (1 epoch, a few batches):
    python tools/run_queue.py --ids A2 A6 B11 --smoke --output runs_smoke

KD students look up their teacher's best.pth (same dataset and seed) in --output first, then
recursively in every --search-dir (e.g. /kaggle/input, where earlier sessions' outputs are attached).
A run whose test_metrics.json already exists in --output is skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "run_plan.yaml"
SMOKE_OVERRIDES = ["train.epochs=1"]
SMOKE_LIMITS = ["--limit-train-batches", "5", "--limit-val-batches", "2", "--limit-test-batches", "2"]
DEADLINE_MARGIN = 1.3  # estimates are +-30%; never start a run that could hit the session limit


@dataclass
class Run:
    id: str
    dataset: str
    model: str
    seed: int
    kd_teacher: str | None = None
    conditional: bool = False
    answers: str = ""
    epochs: int | None = None
    extra_overrides: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return "kd" if self.kd_teacher else "ce"

    @property
    def experiment(self) -> str:
        return f"{self.dataset}_{self.model}_{self.mode}_s{self.seed}"


def load_plan(path: str | Path = DEFAULT_PLAN) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    runs = []
    for raw in plan["runs"]:
        raw = dict(raw)
        runs.append(
            Run(
                id=str(raw.pop("id")),
                dataset=str(raw.pop("dataset")),
                model=str(raw.pop("model")),
                seed=int(raw.pop("seed")),
                kd_teacher=raw.pop("kd_teacher", None),
                conditional=bool(raw.pop("conditional", False)),
                answers=str(raw.pop("answers", "")),
                epochs=raw.pop("epochs", None),
                extra_overrides=list(raw.pop("overrides", [])),
            )
        )
        if raw:
            raise ValueError(f"Unknown keys for run {runs[-1].id}: {sorted(raw)}")
    plan["runs"] = runs
    plan["by_id"] = {run.id: run for run in runs}
    if len(plan["by_id"]) != len(runs):
        raise ValueError("Duplicate run IDs in plan.")
    return plan


def validate_plan(plan: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty when the plan is consistent)."""

    problems = []
    by_id = plan["by_id"]
    for run in plan["runs"]:
        if run.dataset not in plan["datasets"]:
            problems.append(f"{run.id}: unknown dataset {run.dataset}")
        if run.model not in plan["models"]:
            problems.append(f"{run.id}: unknown model {run.model}")
        elif not (ROOT / plan["models"][run.model]).exists():
            problems.append(f"{run.id}: missing config {plan['models'][run.model]}")
        if run.kd_teacher:
            teacher = by_id.get(run.kd_teacher)
            if teacher is None:
                problems.append(f"{run.id}: unknown teacher {run.kd_teacher}")
            else:
                if teacher.model != "resnet18" or teacher.kd_teacher:
                    problems.append(f"{run.id}: teacher {teacher.id} is not a CE ResNet-18 run")
                if teacher.dataset != run.dataset or teacher.seed != run.seed:
                    problems.append(f"{run.id}: teacher {teacher.id} has a different dataset or seed")
    return problems


def estimated_hours(plan: dict[str, Any], run: Run) -> float | None:
    hours = plan.get("est_hours", {}).get(run.dataset, {}).get(run.model)
    if hours is None:
        return None
    overhead = float(plan.get("kd", {}).get("overhead", 1.05))
    return round(hours * (overhead if run.kd_teacher else 1.0), 2)


def find_teacher_checkpoint(teacher: Run, output: Path, search_dirs: list[Path]) -> Path | None:
    local = output / teacher.experiment / "best.pth"
    if local.exists():
        return local
    for base in search_dirs:
        if not base.exists():
            continue
        for candidate in sorted(base.rglob(f"{teacher.experiment}/best.pth")):
            return candidate
    return None


def build_command(
    plan: dict[str, Any],
    run: Run,
    output: Path,
    roots: dict[str, str],
    teacher_checkpoint: Path | None,
    smoke: bool = False,
    workers: int | None = None,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    dataset = plan["datasets"][run.dataset]
    epochs = run.epochs or dataset["epochs"]
    overrides = [item.format(**roots) for item in dataset.get("overrides", [])]
    overrides += [
        f"experiment.name={run.experiment}",
        f"train.seed={run.seed}",
        f"train.epochs={epochs}",
        *run.extra_overrides,
    ]
    if run.kd_teacher:
        overrides += list(plan["kd"]["overrides"])
    if workers is not None:
        overrides.append(f"data.workers={workers}")
    if smoke:
        overrides += SMOKE_OVERRIDES
    overrides += list(extra_overrides or [])

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "train.py"),
        "--config",
        str(ROOT / plan["models"][run.model]),
        "--output",
        str(output),
        "--no-progress",
        "--print-every",
        "10",
    ]
    for item in overrides:
        cmd += ["--override", item]
    if run.kd_teacher:
        if teacher_checkpoint is None:
            raise FileNotFoundError(f"{run.id}: teacher checkpoint not resolved")
        cmd += ["--teacher-config", str(ROOT / plan["kd"]["teacher_config"]), "--teacher-checkpoint", str(teacher_checkpoint)]
    if smoke:
        cmd += SMOKE_LIMITS
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training jobs from the run plan by ID.")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--ids", nargs="+", default=[], help="Run IDs in execution order (e.g. A1 A2 A6).")
    parser.add_argument("--list", action="store_true", help="Print the plan with estimated T4 hours and exit.")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--tin-root", default="data/tiny-imagenet-200")
    parser.add_argument("--cifar-root", default="data")
    parser.add_argument("--search-dir", action="append", default=[], help="Where to look for teacher checkpoints.")
    parser.add_argument("--gpu", help="Value for CUDA_VISIBLE_DEVICES of the training processes (e.g. 0 or 1).")
    parser.add_argument("--workers", type=int, help="Override data.workers (use 2 per job when running two jobs at once).")
    parser.add_argument("--override", action="append", default=[], help="Extra override applied to every run.")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 5 train / 2 val / 2 test batches.")
    parser.add_argument("--force", action="store_true", help="Re-run even if test_metrics.json exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue with the next ID if a run fails.")
    parser.add_argument(
        "--deadline-hours",
        type=float,
        help="Do not start a run unless elapsed + 1.3 x its estimate fits within this many hours "
        "(e.g. 11.5 on Kaggle's 12 h sessions); such runs are reported as deferred.",
    )
    return parser.parse_args()


def print_plan(plan: dict[str, Any]) -> None:
    total = {"A": 0.0, "B": 0.0, "C": 0.0}
    for run in plan["runs"]:
        hours = estimated_hours(plan, run) or 0.0
        total[run.id[0]] = total.get(run.id[0], 0.0) + hours
        teacher = f"teacher={run.kd_teacher}" if run.kd_teacher else ""
        flag = " (conditional)" if run.conditional else ""
        print(f"{run.id:>4}  {run.experiment:42s} {hours:4.1f} h  {teacher:12s} {run.answers}{flag}")
    print("estimated T4 hours: " + ", ".join(f"{k}={v:.1f}" for k, v in total.items()))


def main() -> None:
    args = parse_args()
    plan = load_plan(args.plan)
    problems = validate_plan(plan)
    if problems:
        raise SystemExit("Invalid run plan:\n" + "\n".join(problems))
    if args.list:
        print_plan(plan)
        return
    unknown = [run_id for run_id in args.ids if run_id not in plan["by_id"]]
    if unknown:
        raise SystemExit(f"Unknown run IDs: {unknown}")

    output = Path(args.output).resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    roots = {"tin_root": args.tin_root, "cifar_root": args.cifar_root}
    search_dirs = [Path(p) for p in args.search_dir]
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    summary = []
    queue_start = time.perf_counter()
    simulated_hours = 0.0
    for run_id in args.ids:
        run = plan["by_id"][run_id]
        run_dir = output / run.experiment
        if (run_dir / "test_metrics.json").exists() and not args.force:
            print(f"[skip] {run.id} {run.experiment}: test_metrics.json exists", flush=True)
            summary.append((run.id, run.experiment, "skipped", 0.0))
            continue
        estimate = estimated_hours(plan, run)
        if args.deadline_hours is not None and estimate is not None and not args.smoke:
            # A dry run simulates the session: elapsed time = estimates of the runs scheduled before.
            elapsed = simulated_hours if args.dry_run else (time.perf_counter() - queue_start) / 3600
            if elapsed + DEADLINE_MARGIN * estimate > args.deadline_hours:
                print(
                    f"[defer] {run.id} {run.experiment}: {elapsed:.2f} h elapsed + {DEADLINE_MARGIN} x {estimate} h "
                    f"estimate > {args.deadline_hours} h deadline; run it in the next session",
                    flush=True,
                )
                summary.append((run.id, run.experiment, "deferred (time)", 0.0))
                continue
        teacher_ckpt = None
        if run.kd_teacher:
            teacher = plan["by_id"][run.kd_teacher]
            teacher_ckpt = find_teacher_checkpoint(teacher, output, search_dirs)
            if teacher_ckpt is None and not args.dry_run:
                message = f"{run.id}: teacher {teacher.id} ({teacher.experiment}/best.pth) not found"
                if args.keep_going:
                    print(f"[fail] {message}", flush=True)
                    summary.append((run.id, run.experiment, "missing teacher", 0.0))
                    continue
                raise SystemExit(message)
            teacher_ckpt = teacher_ckpt or Path(f"<{teacher.experiment}/best.pth>")
        cmd = build_command(
            plan,
            run,
            output,
            roots,
            teacher_ckpt,
            smoke=args.smoke,
            workers=args.workers,
            extra_overrides=args.override,
        )
        print(f"\n=== {run.id} {run.experiment} (est. {estimated_hours(plan, run)} h) ===", flush=True)
        print(" ".join(cmd), flush=True)
        simulated_hours += estimate or 0.0
        if args.dry_run:
            summary.append((run.id, run.experiment, "dry-run", 0.0))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        elapsed_h = (time.perf_counter() - start) / 3600
        status = "ok" if code == 0 else f"exit {code}"
        summary.append((run.id, run.experiment, status, elapsed_h))
        if code != 0 and not args.keep_going:
            break

    print("\n=== queue summary ===")
    for run_id, experiment, status, hours in summary:
        print(f"{run_id:>4}  {experiment:42s} {status:16s} {hours:5.2f} h")
    if any(status not in {"ok", "skipped", "dry-run", "deferred (time)"} for _, _, status, _ in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
