from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aggregate_seeds  # noqa: E402


def _fake_run(root: Path, name: str, val: float, test: float, preds: list[int], targets: list[int], smoke: bool = False) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"epoch": 0, "val_acc1": val - 1}) + "\n" + json.dumps({"epoch": 1, "val_acc1": val}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "test_metrics.json").write_text(json.dumps({"epoch": 1, "test_acc1": test, "test_acc5": 90.0}), encoding="utf-8")
    info = {"epochs": 2, "git_commit": "abcdef123456", "limit_train_batches": 5 if smoke else None}
    (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    np.save(run_dir / "test_predictions.npy", np.array(preds))
    np.save(run_dir / "test_targets.npy", np.array(targets))


def test_summary_and_paired_comparison(tmp_path) -> None:
    targets = [0, 1, 2, 3]
    _fake_run(tmp_path, "tin_hbcc_medium_ce_s42", 57.0, 58.0, [0, 1, 2, 0], targets)
    _fake_run(tmp_path, "tin_hbcc_medium_ce_s43", 56.0, 57.0, [0, 1, 2, 3], targets)
    _fake_run(tmp_path, "tin_allcluster_medium_ce_s42", 55.0, 56.0, [0, 0, 0, 0], targets)
    _fake_run(tmp_path, "tin_allcluster_medium_ce_s43", 55.5, 56.5, [0, 1, 0, 0], targets)
    _fake_run(tmp_path, "tin_hbcc_medium_ce_s99", 10.0, 10.0, [0, 0, 0, 0], targets, smoke=True)

    groups = aggregate_seeds.collect([tmp_path])
    assert sorted(groups["tin_hbcc_medium_ce"]) == [42, 43]  # smoke run excluded
    rows = {row["group"]: row for row in aggregate_seeds.summarize(groups)}
    assert rows["tin_hbcc_medium_ce"]["test_acc1_mean"] == pytest.approx(57.5)
    assert rows["tin_hbcc_medium_ce"]["best_val_acc1_mean"] == pytest.approx(56.5)

    result = aggregate_seeds.compare(groups, "tin_hbcc_medium_ce", "tin_allcluster_medium_ce")
    assert result["n"] == 2 and result["a_wins"] == 2
    assert result["diff_mean"] == pytest.approx(1.25)
    assert result["paired_t_p"] is not None
    seed42 = result["mcnemar"][0]
    assert (seed42["only_a"], seed42["only_b"]) == (2, 0)


def test_mcnemar_exact_no_discordant_pairs() -> None:
    preds = np.array([0, 1, 1])
    assert aggregate_seeds.mcnemar_exact(preds, preds, np.array([0, 1, 2])) == (0, 0, 1.0)


def test_latex_output_escapes_underscores(tmp_path) -> None:
    rows = [{"group": "tin_hbcc_medium_ce", "n_seeds": 3, "best_val_acc1_mean": 57.0, "best_val_acc1_std": 0.2,
             "test_acc1_mean": 58.0, "test_acc1_std": 0.3, "test_acc5_mean": 81.0, "test_acc5_std": None}]
    out = tmp_path / "t.tex"
    aggregate_seeds.write_latex(rows, out)
    text = out.read_text(encoding="utf-8")
    assert r"tin\_hbcc\_medium\_ce & 3 & 57.00 $\pm$ 0.20" in text
