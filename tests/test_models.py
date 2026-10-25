from __future__ import annotations

import torch

from lightweight_hbcc.config import load_config
from lightweight_hbcc.models import build_model
from lightweight_hbcc.models.layers import make_local_branch, sign_ste


def test_hbcc_smoke_forward() -> None:
    cfg = load_config("configs/smoke.yaml")
    model = build_model(cfg)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_all_reference_models_forward() -> None:
    for path in [
        "configs/baselines/resnet18_cifar.yaml",
        "configs/baselines/mobilenet_v2_cifar.yaml",
        "configs/baselines/shufflenet_v2_x1_0_cifar.yaml",
        "configs/hbcc_latency_tiny.yaml",
        "configs/hbcc_current_reference.yaml",
        "configs/coc_cifar_baseline.yaml",
        "configs/hbcc_accuracy_small.yaml",
        "configs/hbcc_accuracy_medium.yaml",
        "configs/ablations/hbcc_accuracy_small_pruning_mask.yaml",
    ]:
        cfg = load_config(path)
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 32, 32))
        assert y.shape == (1, 10), path


ABLATION_CONFIGS = [
    "configs/ablations/hbcc_accuracy_medium_allcluster.yaml",
    "configs/ablations/hbcc_medium_s1_dwconv.yaml",
    "configs/ablations/hbcc_medium_s1_conv3x3.yaml",
    "configs/ablations/hbcc_medium_noshuffle.yaml",
    "configs/ablations/hbcc_medium_nofold.yaml",
    "configs/ablations/hbcc_medium_s4_p2.yaml",
    "configs/ablations/hbcc_medium_r4.yaml",
    "configs/ablations/hbcc_medium_r1.yaml",
    "configs/ablations/hbcc_medium_stem2.yaml",
    "configs/ablations/hbcc_medium_alllocal.yaml",
]


def test_ablation_configs_forward_with_200_classes() -> None:
    for path in ABLATION_CONFIGS:
        cfg = load_config(path)
        cfg["model"]["num_classes"] = 200
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 200), path


def test_ablation_configs_change_exactly_one_model_field() -> None:
    base = load_config("configs/hbcc_accuracy_medium.yaml")
    for path in ABLATION_CONFIGS:
        cfg = load_config(path)
        changed = [k for k in base["model"] if cfg["model"].get(k) != base["model"][k]]
        # Removing a whole branch (rho=0 or rho=1) changes modes/branches/ratios/shuffle together.
        allowed = 4 if path.endswith(("allcluster.yaml", "alllocal.yaml")) else 1
        assert 1 <= len(changed) <= allowed, (path, changed)
        assert cfg["data"] == base["data"] and cfg["train"] == base["train"], path


DERIVED_CONFIGS = [
    # Not one-change ablations of hbcc_accuracy_medium (they also rescale widths or swap the
    # whole token mixer), so they are only checked for a correct forward pass.
    "configs/ablations/hbcc_medium_alllocal_macs_matched.yaml",
    "configs/ablations/hbcc_medium_alllocal_conv3x3.yaml",
    "configs/baselines/shufflenet_v2_x1_5_cifar.yaml",
    "configs/ablations/hbcc_alllocal_2_68m.yaml",
    "configs/ablations/hbcc_alllocal_2_68m_dw7.yaml",
]


def test_scaled_alllocal_is_near_shufflenet_x1_5_parameter_budget() -> None:
    cfg = load_config("configs/ablations/hbcc_alllocal_2_68m.yaml")
    cfg["model"]["num_classes"] = 200
    n = sum(p.numel() for p in build_model(cfg).parameters())
    assert 2.55e6 < n < 2.82e6, f"{n:,} is not close to ShuffleNetV2-x1.5's 2,683,624"


def test_dwconv7_branch_uses_a_7x7_depthwise_kernel() -> None:
    branch = make_local_branch("dwconv7", 16, 16)
    depthwise = branch[0][0]
    assert isinstance(depthwise, torch.nn.Conv2d)
    assert depthwise.kernel_size == (7, 7) and depthwise.groups == 16
    assert branch(torch.randn(1, 16, 8, 8)).shape == (1, 16, 8, 8)


def test_derived_configs_forward_with_200_classes() -> None:
    for path in DERIVED_CONFIGS:
        cfg = load_config(path)
        cfg["model"]["num_classes"] = 200
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 200), path


def test_alllocal_variants_contain_no_clustering() -> None:
    from lightweight_hbcc.models.cluster import ContextClusterOp
    for path in ["configs/ablations/hbcc_medium_alllocal.yaml",
                 "configs/ablations/hbcc_medium_alllocal_macs_matched.yaml",
                 "configs/ablations/hbcc_medium_alllocal_conv3x3.yaml"]:
        model = build_model(load_config(path))
        assert not any(isinstance(m, ContextClusterOp) for m in model.modules()), path


def test_shufflenet_x1_5_is_wider_than_x1_0() -> None:
    small = build_model(load_config("configs/baselines/shufflenet_v2_x1_0_cifar.yaml"))
    wide = build_model(load_config("configs/baselines/shufflenet_v2_x1_5_cifar.yaml"))
    n_small = sum(p.numel() for p in small.parameters())
    n_wide = sum(p.numel() for p in wide.parameters())
    assert n_wide > 1.5 * n_small, (n_small, n_wide)
    # the CIFAR stem adjustment must be applied to both
    assert wide.conv1[0].stride == (1, 1)
    assert isinstance(wide.maxpool, torch.nn.Identity)


def test_conv3x3_local_branch_is_learned_full_conv() -> None:
    branch = make_local_branch("conv3x3", 8, 8)
    conv = branch[0][0]
    assert isinstance(conv, torch.nn.Conv2d)
    assert conv.kernel_size == (3, 3) and conv.groups == 1
    assert all(p.requires_grad for p in branch.parameters())
    assert branch(torch.randn(1, 8, 16, 16)).shape == (1, 8, 16, 16)


def test_sign_ste_backward() -> None:
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    y = sign_ste(x).sum()
    y.backward()
    assert torch.equal(sign_ste(x).detach(), torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))
