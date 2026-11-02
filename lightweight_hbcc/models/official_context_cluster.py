from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from torch import nn


_OFFICIAL_MODULE: ModuleType | None = None


def _load_official_module() -> ModuleType:
    global _OFFICIAL_MODULE
    if _OFFICIAL_MODULE is not None:
        return _OFFICIAL_MODULE
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "docs" / "Context-Cluster" / "models" / "context_cluster.py",
        repo_root / "docs" / "context-cluster" / "models" / "context_cluster.py",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Official Context-Cluster code not found. Checked: {checked}")
    spec = importlib.util.spec_from_file_location("official_context_cluster", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official Context-Cluster module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The official file prints optional mmdet/mmseg install messages at import time.
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    _OFFICIAL_MODULE = module
    return module


def official_context_cluster(variant: str = "coc_tiny", num_classes: int = 1000, **kwargs: Any) -> nn.Module:
    module = _load_official_module()
    if not hasattr(module, variant):
        available = [name for name in dir(module) if name.startswith("coc_")]
        raise ValueError(f"Unknown official Context-Cluster variant '{variant}'. Available: {available}")
    factory = getattr(module, variant)
    return factory(pretrained=False, num_classes=num_classes, **kwargs)


def official_coc_tiny(num_classes: int = 1000, **kwargs: Any) -> nn.Module:
    return official_context_cluster("coc_tiny", num_classes=num_classes, **kwargs)


def official_coc_tiny_plain(num_classes: int = 1000, **kwargs: Any) -> nn.Module:
    return official_context_cluster("coc_tiny_plain", num_classes=num_classes, **kwargs)


def official_coc_small(num_classes: int = 1000, **kwargs: Any) -> nn.Module:
    return official_context_cluster("coc_small", num_classes=num_classes, **kwargs)


def official_coc_medium(num_classes: int = 1000, **kwargs: Any) -> nn.Module:
    return official_context_cluster("coc_medium", num_classes=num_classes, **kwargs)
