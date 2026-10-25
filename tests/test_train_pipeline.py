"""End-to-end CPU run of tools/train.py on the fake dataset (configs/smoke.yaml)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _train(output: Path, *extra: str) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "train.py"),
        "--config",
        str(ROOT / "configs" / "smoke.yaml"),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--no-progress",
        *extra,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return output / "smoke_hbcc_latency_tiny_fake"


def test_train_writes_provenance_and_predictions(tmp_path) -> None:
    run_dir = _train(tmp_path / "a")
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert info["seed"] == 42 and cfg["train"]["seed"] == 42  # default seed is applied and saved
    assert cfg["data"]["loader_seed"] == 42
    assert info["completed_epochs"] == 1 and "finished_at" in info
    assert info["git_commit"]  # run inside the git checkout
    preds = np.load(run_dir / "test_predictions.npy")
    targets = np.load(run_dir / "test_targets.npy")
    assert preds.shape == targets.shape == (16,)  # fake_test_size
    test = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert abs(test["test_acc1"] - 100.0 * float(np.mean(preds == targets))) < 1e-6


def test_same_seed_reproduces_training_on_cpu(tmp_path) -> None:
    def losses(run_dir: Path) -> list[float]:
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(l)["train_loss"] for l in lines if "train_loss" in json.loads(l)]

    first = losses(_train(tmp_path / "a", "--override", "train.seed=7"))
    second = losses(_train(tmp_path / "b", "--override", "train.seed=7"))
    third = losses(_train(tmp_path / "c", "--override", "train.seed=8"))
    assert first == second
    assert first != third
