from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lightweight_hbcc.config import apply_overrides, load_config, save_config
from lightweight_hbcc.data import build_loaders, num_classes_for_dataset
from lightweight_hbcc.engine import evaluate, load_checkpoint, resolve_device, save_checkpoint, train_one_epoch, write_metrics
from lightweight_hbcc.losses import DistillationLoss
from lightweight_hbcc.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HBCC/CoC/baseline image classification models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--resume")
    parser.add_argument("--teacher-config")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    parser.add_argument("--skip-test", action="store_true", help="Skip final evaluation on the held-out test split.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars for notebook/log runs.")
    parser.add_argument("--print-every", type=int, default=1, help="Print an epoch summary every N epochs. Use 0 to print only the final epoch.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def build_run_info(args: argparse.Namespace, cfg: dict, device: torch.device) -> dict:
    """Provenance record written next to every run so each reported number is traceable."""

    status = _git("status", "--porcelain")
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "seed": cfg["train"]["seed"],
        "loader_seed": cfg["data"]["loader_seed"],
        "deterministic": cfg["train"]["deterministic"],
        "epochs": int(cfg["train"].get("epochs", 200)),
        "teacher_config": args.teacher_config,
        "teacher_checkpoint": args.teacher_checkpoint,
        "resume": args.resume,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_name": _cpu_name(),
        "python_version": platform.python_version(),
    }


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    dataset_name = cfg.get("data", {}).get("name", "cifar10")
    cfg.setdefault("model", {})["num_classes"] = cfg["model"].get("num_classes", num_classes_for_dataset(dataset_name))
    train_cfg = cfg.setdefault("train", {})
    data_cfg = cfg.setdefault("data", {})
    # Every run is seeded; the resolved seeds are saved in config.yaml and run_info.json.
    seed = int(train_cfg.get("seed", 42))
    deterministic = bool(train_cfg.get("deterministic", False))
    train_cfg["seed"] = seed
    train_cfg["deterministic"] = deterministic
    data_cfg["loader_seed"] = int(data_cfg.get("loader_seed", seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"seed: {seed} loader_seed={data_cfg['loader_seed']} deterministic={deterministic}", flush=True)
    skip_test = args.skip_test or bool(train_cfg.get("skip_test", False))
    device = resolve_device(args.device)
    output_dir = Path(args.output) / cfg.get("experiment", {}).get("name", Path(args.config).stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    run_info = build_run_info(args, cfg, device)
    run_info_path = output_dir / "run_info.json"
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    test_metrics_path = output_dir / "test_metrics.json"
    if not args.resume:
        for stale_path in (metrics_path, test_metrics_path):
            if stale_path.exists():
                stale_path.unlink()

    setup_start = time.perf_counter()
    print(f"setup: dataset={dataset_name} output={output_dir} device={device}", flush=True)
    print("setup: building data loaders...", flush=True)
    train_loader, val_loader, test_loader = build_loaders(cfg.get("data", {}), include_test=not skip_test)
    print(
        "setup: data ready "
        f"train_samples={len(train_loader.dataset)} train_batches={len(train_loader)} "
        f"val_samples={len(val_loader.dataset)} val_batches={len(val_loader)} "
        f"test_enabled={test_loader is not None} elapsed={time.perf_counter() - setup_start:.1f}s",
        flush=True,
    )
    print("setup: building model...", flush=True)
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"setup: model ready params={param_count} elapsed={time.perf_counter() - setup_start:.1f}s", flush=True)
    if args.resume:
        print(f"setup: loading checkpoint {args.resume}", flush=True)
        load_checkpoint(model, args.resume, device, strict=False)

    teacher = None
    if args.teacher_config and args.teacher_checkpoint:
        teacher_cfg = load_config(args.teacher_config)
        teacher_cfg.setdefault("model", {})["num_classes"] = cfg["model"]["num_classes"]
        teacher = build_model(teacher_cfg).to(device)
        load_checkpoint(teacher, args.teacher_checkpoint, device, strict=True)
        teacher.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = DistillationLoss(
        temperature=float(train_cfg.get("kd_temperature", 4.0)),
        alpha=float(train_cfg.get("kd_alpha", 0.0 if teacher is None else 0.5)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        method=str(train_cfg.get("kd_method", "standard")),
        dkd_tckd_weight=float(train_cfg.get("dkd_tckd_weight", 1.0)),
        dkd_nckd_weight=float(train_cfg.get("dkd_nckd_weight", 4.0)),
        dkd_scale=float(train_cfg.get("dkd_scale", 1.0)),
        dkd_warmup_epochs=int(train_cfg.get("dkd_warmup_epochs", 20)),
    )
    print(f"train: distillation method={criterion.method}", flush=True)
    amp = bool(train_cfg.get("amp", True))
    pruning_reg = None
    if cfg.get("model", {}).get("pruning_mask"):
        pruning_reg = (
            float(train_cfg.get("pruning_sparsity_weight", 0.0)),
            float(train_cfg.get("pruning_crispness_weight", 0.0)),
        )

    print(f"train: starting epochs={epochs} skip_test={skip_test}", flush=True)
    best_acc = 0.0
    best_epoch = -1
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            epoch,
            teacher=teacher,
            amp=amp,
            limit_batches=args.limit_train_batches,
            pruning_regularization=pruning_reg,
            progress=not args.no_progress,
            mixup_alpha=float(train_cfg.get("mixup_alpha", 0.0)),
            cutmix_alpha=float(train_cfg.get("cutmix_alpha", 0.0)),
            cutmix_prob=float(train_cfg.get("cutmix_prob", 0.0)),
            num_classes=cfg["model"]["num_classes"],
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            amp=amp,
            limit_batches=args.limit_val_batches,
            progress=not args.no_progress,
            prefix="val",
        )
        scheduler.step()
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **train_metrics, **val_metrics}
        write_metrics(record, metrics_path)
        is_best = val_metrics["val_acc1"] >= best_acc
        if is_best:
            best_acc = val_metrics["val_acc1"]
            best_epoch = epoch
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc1": best_acc,
            "best_epoch": best_epoch,
            "config": cfg,
        }
        save_checkpoint(checkpoint, output_dir / "latest.pth")
        if is_best:
            save_checkpoint(checkpoint, output_dir / "best.pth")
        should_print = epoch == epochs - 1
        if args.print_every > 0:
            should_print = should_print or (epoch % args.print_every == 0)
        if should_print:
            if args.no_progress:
                line = (
                    "epoch={epoch} lr={lr:.6g} train_acc1={train_acc1:.2f} "
                    "train_loss={train_loss:.4f} val_acc1={val_acc1:.2f} val_loss={val_loss:.4f}".format(**record)
                )
                if "val_acc5" in record:
                    line += " val_acc5={val_acc5:.2f}".format(**record)
                print(line)
            else:
                print(json.dumps(record, indent=2))

    best_path = output_dir / "best.pth"
    if not skip_test and best_path.exists():
        if test_loader is None:
            raise RuntimeError("Final test evaluation was requested, but no test loader was created.")
        load_checkpoint(model, best_path, device, strict=True)
        test_metrics, test_preds, test_targets = evaluate(
            model,
            test_loader,
            device,
            amp=amp,
            limit_batches=args.limit_test_batches,
            progress=not args.no_progress,
            prefix="test",
            return_predictions=True,
        )
        # Per-image predictions in test-set order enable paired tests (McNemar) between runs.
        np.save(output_dir / "test_predictions.npy", test_preds)
        np.save(output_dir / "test_targets.npy", test_targets)
        test_record = {"phase": "test", "epoch": best_epoch, "checkpoint": str(best_path.name), **test_metrics}
        write_metrics(test_record, metrics_path)
        with test_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(test_record, f, indent=2)
        if args.no_progress:
            line = (
                "test checkpoint={checkpoint} epoch={epoch} test_acc1={test_acc1:.2f} "
                "test_loss={test_loss:.4f}".format(**test_record)
            )
            if "test_acc5" in test_record:
                line += " test_acc5={test_acc5:.2f}".format(**test_record)
            print(line)
        else:
            print(json.dumps(test_record, indent=2))

    run_info.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "completed_epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_acc1": best_acc,
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "limit_test_batches": args.limit_test_batches,
        }
    )
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
