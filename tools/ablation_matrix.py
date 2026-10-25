from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightweight_hbcc.config import deep_update, load_config, save_config


ABLATIONS = {
    "coord_off": {"model": {"use_coord": False}},
    "coord_on": {"model": {"use_coord": True}},
    "hamming_late": {"model": {"similarities": ["cosine", "cosine", "hamming", "hamming"]}},
    "hamming_all": {"model": {"similarities": ["hamming", "hamming", "hamming", "hamming"]}},
    "local_identity": {"model": {"local_branches": ["identity", "identity", "identity", "identity"]}},
    "local_dwconv": {"model": {"local_branches": ["dwconv", "dwconv", "identity", "identity"]}},
    "local_lbpconv": {"model": {"local_branches": ["lbpconv", "lbpconv", "identity", "identity"]}},
    "shuffle_on": {"model": {"channel_shuffle": [False, True, False, False]}},
    "pruning_mask": {
        "model": {"pruning_mask": True},
        "train": {"pruning_sparsity_weight": 0.001, "pruning_crispness_weight": 0.001},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one-config-per-ablation YAML files.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", default="configs/generated_ablations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.base)
    out_dir = Path(args.output_dir)
    for name, patch in ABLATIONS.items():
        cfg = deep_update(base, patch)
        cfg.setdefault("experiment", {})["name"] = f"{base.get('experiment', {}).get('name', Path(args.base).stem)}_{name}"
        save_config(cfg, out_dir / f"{name}.yaml")
    print(out_dir)


if __name__ == "__main__":
    main()
