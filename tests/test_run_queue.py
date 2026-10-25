from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_queue  # noqa: E402


def test_plan_is_consistent() -> None:
    plan = run_queue.load_plan()
    assert run_queue.validate_plan(plan) == []
    experiments = [run.experiment for run in plan["runs"]]
    assert len(experiments) == len(set(experiments)), "two run IDs map to the same run directory"


def test_every_kd_run_uses_same_seed_resnet_teacher() -> None:
    plan = run_queue.load_plan()
    kd_runs = [run for run in plan["runs"] if run.kd_teacher]
    assert kd_runs
    for run in kd_runs:
        teacher = plan["by_id"][run.kd_teacher]
        assert teacher.model == "resnet18" and teacher.seed == run.seed and teacher.dataset == run.dataset


def test_gate_runs_are_the_budget_matched_pair() -> None:
    plan = run_queue.load_plan()
    a1, b1 = plan["by_id"]["A1"], plan["by_id"]["B1"]
    assert (a1.model, b1.model) == ("allcluster_medium", "hbcc_medium")
    assert a1.dataset == b1.dataset == "tin" and a1.seed == b1.seed == 42


def test_build_command_for_tin_kd_run() -> None:
    plan = run_queue.load_plan()
    run = plan["by_id"]["A7"]
    cmd = run_queue.build_command(
        plan,
        run,
        Path("out"),
        {"tin_root": "/data/tin", "cifar_root": "/data/c10"},
        Path("out/tin_resnet18_ce_s42/best.pth"),
        smoke=True,
        workers=2,
    )
    overrides = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--override"]
    assert "data.root=/data/tin" in overrides and "model.num_classes=200" in overrides
    assert "experiment.name=tin_hbcc_medium_kd_s42" in overrides
    assert "train.seed=42" in overrides and "train.kd_alpha=0.5" in overrides
    assert "data.workers=2" in overrides
    assert overrides[-1] == "train.epochs=1"  # smoke epochs override the plan's 100
    assert Path(cmd[cmd.index("--teacher-checkpoint") + 1]) == Path("out/tin_resnet18_ce_s42/best.pth")
    assert "--limit-train-batches" in cmd


def test_deadline_defers_runs_that_cannot_finish(tmp_path) -> None:
    import subprocess

    plan = run_queue.load_plan()
    a1 = run_queue.estimated_hours(plan, plan["by_id"]["A1"])
    b10 = run_queue.estimated_hours(plan, plan["by_id"]["B10"])
    # A1 alone fits; B10 after A1 does not (the dry run accumulates A1's estimate).
    deadline = run_queue.DEADLINE_MARGIN * a1 + 0.01
    assert a1 + run_queue.DEADLINE_MARGIN * b10 > deadline
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_queue.py"), "--ids", "A1", "B10", "--dry-run",
         "--output", str(tmp_path / "runs"), "--deadline-hours", str(deadline)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "=== A1" in result.stdout and "[defer] A1" not in result.stdout
    assert "[defer] B10" in result.stdout
    assert not (tmp_path / "runs").exists()  # dry runs never create the output directory


def test_teacher_lookup_searches_attached_inputs(tmp_path) -> None:
    plan = run_queue.load_plan()
    teacher = plan["by_id"]["A2"]
    attached = tmp_path / "input" / "session-a-output" / "runs" / teacher.experiment
    attached.mkdir(parents=True)
    (attached / "best.pth").write_bytes(b"x")
    found = run_queue.find_teacher_checkpoint(teacher, tmp_path / "runs", [tmp_path / "input"])
    assert found == attached / "best.pth"
