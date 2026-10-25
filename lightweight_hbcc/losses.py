from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DISTILLATION_METHODS = ("standard", "dkd")


def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    out = torch.full((target.shape[0], num_classes), off_value, device=target.device, dtype=torch.float32)
    out.scatter_(1, target.reshape(-1, 1), on_value)
    return out


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _collapse_target_and_other(probabilities: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Collapse a class distribution into [p(target), p(everything else)]."""

    target_probability = (probabilities * target_mask).sum(dim=1, keepdim=True)
    other_probability = (probabilities * ~target_mask).sum(dim=1, keepdim=True)
    return torch.cat((target_probability, other_probability), dim=1)


class DistillationLoss(nn.Module):
    """Cross-entropy plus an optional distillation term.

    ``method="standard"`` is Hinton logit matching, blended as
    ``(1-alpha) * CE + alpha * T^2 * KL(teacher || student)``. This is the default and is
    bit-for-bit the behaviour used by every run up to this point.

    ``method="dkd"`` is Decoupled Knowledge Distillation (Zhao et al., CVPR 2022), which splits
    the KD term into a target-class part (TCKD) and a non-target-class part (NCKD) and reweights
    them, added to CE rather than blended: ``CE + scale * w(epoch) * (a*TCKD + b*NCKD)``.
    DKD needs the *hard* label to build its target mask, so under MixUp/CutMix the original
    label must be supplied via ``hard_target``; the CE term still uses the mixed soft target.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
        method: str = "standard",
        dkd_tckd_weight: float = 1.0,
        dkd_nckd_weight: float = 4.0,
        dkd_scale: float = 1.0,
        dkd_warmup_epochs: int = 20,
    ) -> None:
        super().__init__()
        method = str(method).lower()
        if method not in DISTILLATION_METHODS:
            raise ValueError(f"Unsupported distillation method {method!r}; expected one of {DISTILLATION_METHODS}.")
        if dkd_tckd_weight < 0.0 or dkd_nckd_weight < 0.0:
            raise ValueError("DKD weights must be non-negative.")
        if method == "dkd" and dkd_scale <= 0.0:
            raise ValueError("dkd_scale must be positive for method='dkd'.")
        if dkd_warmup_epochs < 0:
            raise ValueError("dkd_warmup_epochs must be non-negative.")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)
        self.method = method
        self.dkd_tckd_weight = float(dkd_tckd_weight)
        self.dkd_nckd_weight = float(dkd_nckd_weight)
        self.dkd_scale = float(dkd_scale)
        self.dkd_warmup_epochs = int(dkd_warmup_epochs)
        self.epoch = 0
        self.last_components: dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        """Current epoch, used only for the DKD linear warmup."""

        self.epoch = int(epoch)

    def _warmup(self) -> float:
        if self.dkd_warmup_epochs == 0:
            return 1.0
        return min((self.epoch + 1) / self.dkd_warmup_epochs, 1.0)

    def _standard_kd(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature
        return F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)

    def _dkd(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temp = self.temperature
        student_scaled = student_logits.float() / temp
        teacher_scaled = teacher_logits.detach().float() / temp
        target_mask = F.one_hot(hard_target, num_classes=student_logits.shape[1]).bool()

        student_probs = _collapse_target_and_other(F.softmax(student_scaled, dim=1), target_mask)
        teacher_probs = _collapse_target_and_other(F.softmax(teacher_scaled, dim=1), target_mask)
        tckd = F.kl_div(
            student_probs.clamp_min(1e-12).log(), teacher_probs, reduction="batchmean"
        ) * (temp * temp)

        masked_student = student_scaled.masked_fill(target_mask, -1e9)
        masked_teacher = teacher_scaled.masked_fill(target_mask, -1e9)
        nckd = F.kl_div(
            F.log_softmax(masked_student, dim=1), F.softmax(masked_teacher, dim=1), reduction="batchmean"
        ) * (temp * temp)
        return tckd, nckd

    def forward(
        self,
        student_logits: torch.Tensor,
        target: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        hard_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target.ndim == 1:
            ce = F.cross_entropy(student_logits, target, label_smoothing=self.label_smoothing)
        else:
            ce = soft_target_cross_entropy(student_logits, target)
        if teacher_logits is None or (self.method == "standard" and self.alpha <= 0.0):
            self.last_components = {"ce": float(ce.detach())}
            return ce

        if self.method == "standard":
            kd = self._standard_kd(student_logits, teacher_logits)
            self.last_components = {"ce": float(ce.detach()), "kd": float(kd.detach())}
            return (1.0 - self.alpha) * ce + self.alpha * kd

        if hard_target is None:
            if target.ndim != 1:
                raise ValueError(
                    "method='dkd' needs the hard label for its target mask. Pass hard_target= "
                    "(the pre-MixUp/CutMix label) when training with mixed targets."
                )
            hard_target = target
        tckd, nckd = self._dkd(student_logits, teacher_logits, hard_target)
        dkd = self.dkd_tckd_weight * tckd + self.dkd_nckd_weight * nckd
        total = ce + self.dkd_scale * self._warmup() * dkd
        self.last_components = {
            "ce": float(ce.detach()), "tckd": float(tckd.detach()),
            "nckd": float(nckd.detach()), "dkd": float(dkd.detach()),
        }
        return total
