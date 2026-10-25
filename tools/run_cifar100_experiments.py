from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys


CIFAR100_OVERRIDES = ("data.name=cifar100", "model.num_classes=100")


@dataclass(frozen=True)
class Experiment:
    name: str
    config: str
    kind: str
    overrides: tuple[str, ...] = field(default_factory=tuple)
    teacher: bool = False


REFERENCE = Experiment("resnet18_cifar100", "configs/baselines/resnet18_cifar.yaml", "reference")

BASELINE_EXPERIMENTS = [
    Experiment("mobilenet_v2_cifar100", "configs/baselines/mobilenet_v2_cifar.yaml", "baseline"),
    Experiment("shufflenet_v2_x1_0_cifar100", "configs/baselines/shufflenet_v2_x1_0_cifar.yaml", "baseline"),
]

COC_EXPERIMENTS = [
    Experiment("coc_cifar_baseline_cifar100", "configs/coc_cifar_baseline.yaml", "coc"),
]

STUDENT_EXPERIMENTS = [
    Experiment("hbcc_latency_tiny_cifar100", "configs/hbcc_latency_tiny.yaml", "student"),
    Experiment("hbcc_latency_small_cifar100", "configs/hbcc_latency_small.yaml", "student"),
    Experiment("hbcc_accuracy_small_cifar100", "configs/hbcc_accuracy_small.yaml", "student"),
    Experiment("hbcc_accuracy_medium_cifar100", "configs/hbcc_accuracy_medium.yaml", "student"),
]

PROXY_EXPERIMENTS = [
    Experiment(
        "proxy_dims24_depth1111_mlp2_cifar100",
        "configs/hbcc_latency_tiny.yaml",
        "proxy",
        overrides=(
            "model.embed_dims=[24,48,96,160]",
            "model.depths=[1,1,1,1]",
            "model.mlp_ratios=2.0",
        ),
    ),
    Experiment(
        "proxy_dims32_depth1121_mlp2_cifar100",
        "configs/hbcc_latency_tiny.yaml",
        "proxy",
        overrides=(
            "model.embed_dims=[32,48,96,160]",
            "model.depths=[1,1,2,1]",
            "model.mlp_ratios=2.0",
        ),
    ),
    Experiment(
        "proxy_dims32_64_depth1221_mlp3_cifar100",
        "configs/hbcc_latency_tiny.yaml",
        "proxy",
        overrides=(
            "model.embed_dims=[32,64,128,192]",
            "model.depths=[1,2,2,1]",
            "model.mlp_ratios=3.0",
            "model.heads=[1,2,4,4]",
        ),
    ),
]

ABLATION_EXPERIMENTS = [
    Experiment("hbcc_tiny_hamming_late_cifar100", "configs/ablations/hbcc_tiny_hamming_late.yaml", "ablation"),
    Experiment("hbcc_tiny_lbpconv_cifar100", "configs/ablations/hbcc_tiny_lbpconv.yaml", "ablation"),
]

KD_EXPERIMENTS = [
    Experiment(
        "hbcc_latency_tiny_kd_cifar100",
        "configs/hbcc_latency_tiny.yaml",
        "kd",
        overrides=("train.kd_alpha=0.5", "train.kd_temperature=4.0"),
        teacher=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIFAR-10-style pipeline on CIFAR-100.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs_cifar100")
    parser.add_argument("--benchmark-output", default="results/benchmark_cifar100")
    parser.add_argument("--summary-output", default="results/cifar100_summary.csv")
    parser.add_argument("--only", nargs="*", help="Run only these experiment names.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs for every training run.")
    parser.add_argument("--proxy-epochs", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-coc", action="store_true")
    parser.add_argument("--skip-students", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-kd", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Write torch operator profiles during benchmarking.")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars in child training runs.")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--benchmark-warmup", type=int, default=30)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    parser.add_argument("--benchmark-batch-sizes", nargs="+", type=int, default=[1, 16, 64, 128])
    parser.add_argument("--smoke", action="store_true", help="Run one batch/epoch and tiny benchmark settings.")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _selected(experiments: list[Experiment], only: set[str] | None) -> list[Experiment]:
    if only is None:
        return experiments
    return [exp for exp in experiments if exp.name in only]


def _base_overrides(args: argparse.Namespace, exp: Experiment) -> tuple[str, ...]:
    return (
        f"data.root={args.data_root}",
        *CIFAR100_OVERRIDES,
        f"experiment.name={exp.name}",
        *exp.overrides,
    )


def _train_args(args: argparse.Namespace, exp: Experiment) -> list[str]:
    cmd = [
        args.python,
        "tools/train.py",
        "--config",
        exp.config,
        "--output",
        args.output,
        "--print-every",
        str(args.print_every),
    ]
    if not args.progress:
        cmd.append("--no-progress")
    for override in _base_overrides(args, exp):
        cmd.extend(["--override", override])
    if args.epochs is not None:
        cmd.extend(["--override", f"train.epochs={args.epochs}"])
    elif exp.kind == "proxy":
        cmd.extend(["--override", f"train.epochs={args.proxy_epochs}"])
    if args.smoke:
        cmd.extend(
            [
                "--override",
                "train.epochs=1",
                "--limit-train-batches",
                "1",
                "--limit-val-batches",
                "1",
                "--limit-test-batches",
                "1",
            ]
        )
    if exp.teacher:
        teacher_checkpoint = Path(args.output) / REFERENCE.name / "best.pth"
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(f"Train the reference first or provide its checkpoint: {teacher_checkpoint}")
        cmd.extend(
            [
                "--teacher-config",
                REFERENCE.config,
                "--teacher-checkpoint",
                str(teacher_checkpoint),
            ]
        )
    return cmd


def _benchmark_args(args: argparse.Namespace, exp: Experiment) -> list[str]:
    cmd = [
        args.python,
        "tools/benchmark.py",
        "--config",
        exp.config,
        "--checkpoint",
        str(Path(args.output) / exp.name / "best.pth"),
        "--output",
        args.benchmark_output,
        "--warmup",
        str(1 if args.smoke else args.benchmark_warmup),
        "--runs",
        str(2 if args.smoke else args.benchmark_runs),
        "--batch-sizes",
        *[str(item) for item in ([1, 4] if args.smoke else args.benchmark_batch_sizes)],
    ]
    for override in _base_overrides(args, exp):
        cmd.extend(["--override", override])
    if args.profile:
        cmd.append("--profile")
    return cmd


def _training_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    if not args.skip_reference:
        experiments.append(REFERENCE)
    if not args.skip_baselines:
        experiments.extend(BASELINE_EXPERIMENTS)
    if not args.skip_coc:
        experiments.extend(COC_EXPERIMENTS)
    if not args.skip_students:
        experiments.extend(STUDENT_EXPERIMENTS)
    if not args.skip_proxy:
        experiments.extend(PROXY_EXPERIMENTS)
    if not args.skip_ablations:
        experiments.extend(ABLATION_EXPERIMENTS)
    if not args.skip_kd:
        experiments.extend(KD_EXPERIMENTS)
    return experiments


def main() -> None:
    args = parse_args()
    only = set(args.only) if args.only else None
    train_experiments = _selected(_training_experiments(args), only)

    if not args.skip_train:
        for exp in train_experiments:
            _run(_train_args(args, exp))

    benchmark_experiments = _selected(
        [
            REFERENCE,
            *BASELINE_EXPERIMENTS,
            *COC_EXPERIMENTS,
            *STUDENT_EXPERIMENTS,
            *PROXY_EXPERIMENTS,
            *ABLATION_EXPERIMENTS,
            *KD_EXPERIMENTS,
        ],
        only,
    )
    if not args.skip_benchmark:
        for exp in benchmark_experiments:
            checkpoint = Path(args.output) / exp.name / "best.pth"
            if not checkpoint.exists():
                print(f"skip benchmark: missing checkpoint {checkpoint}", flush=True)
                continue
            _run(_benchmark_args(args, exp))

    if not args.skip_summary:
        _run(
            [
                args.python,
                "tools/cifar100_report.py",
                "--runs",
                args.output,
                "--benchmarks",
                args.benchmark_output,
                "--output",
                args.summary_output,
            ]
        )


if __name__ == "__main__":
    main()
