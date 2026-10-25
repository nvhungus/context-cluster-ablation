"""DistillationLoss: the standard path must be unchanged, and DKD must behave correctly."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from lightweight_hbcc.losses import DistillationLoss, smooth_one_hot


def _old_standard_loss(student, target, teacher, alpha=0.5, temp=4.0, smoothing=0.0):
    """The exact formula used by every run before DKD was added."""
    if target.ndim == 1:
        ce = F.cross_entropy(student, target, label_smoothing=smoothing)
    else:
        ce = -(target * F.log_softmax(student, dim=1)).sum(dim=1).mean()
    if teacher is None or alpha <= 0.0:
        return ce
    kd = F.kl_div(
        F.log_softmax(student / temp, dim=1),
        F.softmax(teacher / temp, dim=1),
        reduction="batchmean",
    ) * (temp * temp)
    return (1.0 - alpha) * ce + alpha * kd


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("mixed", [False, True])
def test_standard_path_is_unchanged(smoothing: bool, mixed: bool) -> None:
    torch.manual_seed(0)
    student, teacher = torch.randn(8, 20), torch.randn(8, 20)
    hard = torch.randint(0, 20, (8,))
    target = smooth_one_hot(hard, 20, smoothing) if mixed else hard
    loss = DistillationLoss(temperature=4.0, alpha=0.5, label_smoothing=smoothing)
    new = loss(student, target, teacher, hard_target=hard)
    old = _old_standard_loss(student, target, teacher, 0.5, 4.0, smoothing)
    assert torch.allclose(new, old, atol=0, rtol=0), (new.item(), old.item())


def test_standard_without_teacher_is_plain_ce() -> None:
    torch.manual_seed(0)
    student, hard = torch.randn(4, 10), torch.randint(0, 10, (4,))
    loss = DistillationLoss()
    assert torch.allclose(loss(student, hard, None), F.cross_entropy(student, hard))


def test_dkd_runs_under_mixup_and_needs_hard_target() -> None:
    torch.manual_seed(0)
    student, teacher = torch.randn(8, 20), torch.randn(8, 20)
    hard = torch.randint(0, 20, (8,))
    mixed = smooth_one_hot(hard, 20, 0.1)
    loss = DistillationLoss(method="dkd", dkd_warmup_epochs=0)
    out = loss(student, mixed, teacher, hard_target=hard)
    assert torch.isfinite(out)
    assert {"ce", "tckd", "nckd", "dkd"} <= set(loss.last_components)
    with pytest.raises(ValueError, match="hard label"):
        loss(student, mixed, teacher, hard_target=None)


def test_dkd_matches_teacher_gives_near_zero_distillation_term() -> None:
    torch.manual_seed(0)
    logits, hard = torch.randn(8, 20), torch.randint(0, 20, (8,))
    loss = DistillationLoss(method="dkd", dkd_warmup_epochs=0)
    loss(logits, hard, logits.clone(), hard_target=hard)
    assert loss.last_components["dkd"] == pytest.approx(0.0, abs=1e-5)


def test_dkd_warmup_scales_the_distillation_term() -> None:
    torch.manual_seed(0)
    student, teacher = torch.randn(8, 20), torch.randn(8, 20)
    hard = torch.randint(0, 20, (8,))
    loss = DistillationLoss(method="dkd", dkd_warmup_epochs=10)
    loss.set_epoch(0)
    early = loss(student, hard, teacher, hard_target=hard)
    ce_only = loss.last_components["ce"]
    dkd = loss.last_components["dkd"]
    assert early.item() == pytest.approx(ce_only + 0.1 * dkd, rel=1e-5)
    loss.set_epoch(99)
    late = loss(student, hard, teacher, hard_target=hard)
    assert late.item() == pytest.approx(ce_only + dkd, rel=1e-5)
    assert late > early


def test_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported distillation method"):
        DistillationLoss(method="fitnet")
