# Context-cluster ablation study

Code, configurations and per-run records for a controlled ablation study on
context clustering in lightweight image classification.

## What this is

Context Cluster replaces attention with a clustering token mixer and reports
competitive accuracy at 5.3M parameters on 224×224 inputs. This repository
contains experiments that test whether the mechanism survives compression to
one to three million parameters at 32×32 input.

It does not. Holding the stem, fold partitioning, block depth and width fixed and
removing the *local* convolutional branch costs 0.39 pp on Tiny-ImageNet;
removing the *clustering* branch instead gains 0.94 pp at 10% fewer parameters.
The design choices built around clustering — fold partitioning, channel shuffle,
tuned occupancy — move accuracy by less than seed variance.

Following the ablations to their conclusion gives a clustering-free model,
**All-Local (2.7M)**: a stride-1 stem, a learned 3×3 convolution at stage 1, and
7×7 depthwise plus 1×1 pointwise convolutions at stages 2–4. With logit
distillation it reaches **61.30 ± 0.39** top-1 on Tiny-ImageNet against
**59.79 ± 0.38** for a parameter-matched ShuffleNetV2-x1.5 trained identically
from the same teacher, and **58.71 ± 0.17** for the ResNet-18 teacher itself.

Two caveats belong with that number: the margin is largest **under
distillation** (with plain cross-entropy it is +0.90 pp, still significant at
p = 0.030, and the comparison against the teacher becomes a tie), and it costs
2.41× the operations of the parameter-matched baseline.

All accuracies quoted here and in the tables are **test-set**, recomputed from
the per-image predictions in `release/runs/`. Validation-set values appear in
earlier commits of this README and are superseded.

## Environment

```bash
conda env create -f environment.yml
conda activate CoC
# or:
pip install -r requirements.txt
```

Requires PyTorch with CUDA for training; `fvcore` is needed for operation counts
in `tools/benchmark.py`. All results here were produced on a single
NVIDIA Tesla T4.

Datasets are not redistributed here. CIFAR-10 downloads automatically on first
use. Tiny-ImageNet must be supplied; pass its root with `--tin-root`.

## Reproducing the results

Every reported run is declared in [`configs/run_plan.yaml`](configs/run_plan.yaml),
which is the single source of truth for the experiments: run ID, dataset, model
config, seed, teacher, and the estimated T4 hours used for scheduling.

[`docs/protocol.md`](docs/protocol.md) states the protocol every run follows:
the exact train/validation/test splits, the full training recipe, the seed
policy, how the checkpoint is selected, and how accuracy, parameters,
operation counts, latency and memory are each measured and reported.

```bash
# List every run with its experiment name and estimated cost
python tools/run_queue.py --list

# Inspect the exact command a run would execute, without running it
python tools/run_queue.py --ids B52 --dry-run

# Execute runs in order
python tools/run_queue.py --ids B52 B46 B47 \
    --tin-root /path/to/tiny-imagenet-200 \
    --output runs --search-dir /path/to/teacher/checkpoints
```

A knowledge-distillation run resolves its teacher by experiment name under
`--search-dir`; a student always uses the ResNet-18 teacher trained on the *same
dataset and the same seed*, which `tools/run_queue.py` validates before starting.

Train a single configuration directly:

```bash
python tools/train.py --config configs/ablations/hbcc_alllocal_2_68m_conv3x3_dw7.yaml \
    --override data.name=tiny_imagenet --override model.num_classes=200 \
    --override train.seed=42 --override train.epochs=100 --output runs
```

Measure cost (parameters, MACs, peak memory, eager and CUDA-Graph latency):

```bash
pip install fvcore
python tools/benchmark.py \
    --config configs/ablations/hbcc_alllocal_2_68m_conv3x3_dw7.yaml \
    --override model.num_classes=200 \
    --name all_local --output results/benchmark \
    --batch-sizes 1 16 64 128 --cuda-graph
```

Each captured CUDA Graph is verified against the eager forward pass; the record
reports `graph_matches_eager_b<N>` and `graph_max_abs_diff_b<N>`. Peak memory is
reset per batch size, and resident weights are reported separately from
activations. Latency drifts by up to 14% between sessions, so compare only
figures measured in the same session.

Aggregate seeds and run the significance tests:

```bash
python tools/aggregate_seeds.py --runs runs
```

## Regenerating the figures

```bash
python paper/figures/fig1_architecture.py   # architecture diagram
python paper/figures/fig2_pareto.py         # accuracy vs parameters and MACs
```

Both write a PDF next to the script. They need only `matplotlib`. The cost
figures they plot are hard-coded from
[`results/cost_t4_m1v3/summary.csv`](results/cost_t4_m1v3/summary.csv), which
is the single profiling session the reported memory and latency come from.

## What is in `results/`

`results/cost_t4_m1v3/` is the profiling session that every memory, latency and
throughput figure in the study is taken from: one `summary.csv` with 59 columns
per model, plus the raw per-model JSON. It includes the `graph_matches_eager_b*`
columns, which record that each CUDA-Graph capture reproduced its eager output
bitwise (`graph_max_abs_diff_b* = 0.0`), and the `*_repeats` columns holding
every timed loop behind each reported median.

The remaining CSVs in `results/` predate the current study: they profile an
earlier generation of configurations at different parameter counts, were taken
with a different torch build, and have no graph-capture columns. They are kept
because `notebooks/kaggle_smoke_test.ipynb` self-checks against them. **Do not
read the study's cost figures out of them** -- use `cost_t4_m1v3/`.

## What is in `release/runs/`

One directory per training run, 91 in total, named
`{dataset}_{model}_{ce|kd}_s{seed}`. Every run here ran its full epoch budget
(100 on Tiny-ImageNet, 250 on CIFAR-10); single-epoch smoke-test runs are not
included. Each directory contains:

| file | contents |
|---|---|
| `metrics.jsonl` | one JSON object per epoch: train loss, validation accuracy, learning rate, wall time |
| `run_info.json` | provenance — git commit, seed, resolved teacher checkpoint, torch/CUDA/cuDNN versions, GPU and CPU model, start and end time, best epoch |
| `test_metrics.json` | final test accuracy as recorded by the training script (see the caveat below) |
| `test_predictions.npy` | predicted class index for every test image, in dataset order |
| `test_targets.npy` | ground-truth class index for every test image |

Per-image predictions are included so that any comparison reported here can be
recomputed, including the paired tests, which need the pairing that aggregate
accuracies discard.

**Use the `.npy` files, not `test_metrics.json`.** In 30 of the 91 runs the two
disagree by one to four images out of 10,000 (0.01–0.04 pp). The training
script's logged top-1 comes from `output.topk(5)` while the saved predictions
come from `output.argmax`, and the two resolve exact logit ties differently
under mixed precision. Every accuracy quoted in this repository is recomputed
from the `.npy` files, which are authoritative. Validation accuracy in
`metrics.jsonl` comes from the same `topk` path, so checkpoint selection
carries the same ≤0.04 pp imprecision — far below the ~0.5 pp epoch-to-epoch
validation variation, so the selected checkpoint is unaffected in practice.

**Provenance, and its limits.** Each run's `run_info.json` records the git
commit of the working repository the run was launched from. That repository's
history is not published here, so these hashes cannot be resolved against the
code in this snapshot; they are retained because they distinguish runs
launched from different code states, and fourteen distinct commits appear
across the released runs. What this snapshot does let you verify is the
configuration of every run (`configs/run_plan.yaml` plus the referenced config
files), the resolved teacher checkpoint path, the full software and hardware
environment, the per-epoch metrics, and the per-image test predictions from
which every reported number is recomputed. What it does not let you do is
confirm that the released code is byte-identical to the code that produced a
given run.

**Teacher checkpoints are not included.** Reproducing a distilled student
therefore requires retraining its ResNet-18 teacher from the config and seed
recorded in `run_info.json`. Training runs with `deterministic: false`, so
cuDNN is non-deterministic and a retrained teacher will not be bit-identical;
the seed-to-seed variance reported in the study includes this source.

Recompute a reported accuracy:

```python
import numpy as np
p = np.load("release/runs/tin_alllocal_2_68m_conv3x3_dw7_kd_s42/test_predictions.npy")
t = np.load("release/runs/tin_alllocal_2_68m_conv3x3_dw7_kd_s42/test_targets.npy")
print(100 * (p == t).mean())
```

Reproduce a paired McNemar test between two runs:

```python
import numpy as np
from scipy import stats

def mcnemar(run_a, run_b):
    pa = np.load(f"release/runs/{run_a}/test_predictions.npy")
    pb = np.load(f"release/runs/{run_b}/test_predictions.npy")
    t  = np.load(f"release/runs/{run_a}/test_targets.npy")
    a_only = int(((pa == t) & (pb != t)).sum())
    b_only = int(((pa != t) & (pb == t)).sum())
    return a_only, b_only, stats.binomtest(a_only, a_only + b_only, 0.5).pvalue

print(mcnemar("tin_alllocal_2_68m_conv3x3_dw7_kd_s42",
              "tin_shufflenet_v2_x1_5_kd_s42"))
```

Note on the Tiny-ImageNet split: the dataset does not release test labels, so the
official 10,000-image validation split is used as the **test** set, and a further
10,000 images held out of the training split (fixed seed) serve as the validation
set for epoch selection. No model selects an epoch using the reported test set.

Some `release/runs/` directories predate the current study; the set of runs the
reported results draw on is exactly the one declared in `configs/run_plan.yaml`.

## Layout

```
lightweight_hbcc/     model, data, training engine, losses, profiler
  models/             the hybrid block, local branches, baselines
configs/              model configs; run_plan.yaml declares every reported run
  ablations/          one-change ablations used in the study
  baselines/          ResNet-18, MobileNetV2, ShuffleNetV2 x1.0 and x1.5
tools/                train.py, run_queue.py, benchmark.py, aggregate_seeds.py
tests/                unit tests for losses, profiler, run queue, models
notebooks/            the Kaggle session notebooks used to execute the runs
results/              cost_t4_m1v3/ is the profiling session behind the reported
                      cost figures; the other CSVs are from an earlier generation
release/runs/         per-run metrics, provenance and per-image predictions
paper/figures/        figure-generation scripts
```

## License

MIT, see [LICENSE](LICENSE). The Context Cluster formulation ablated here is due
to Ma et al., *Image as Set of Points*, ICLR 2023; baseline architectures come
from `torchvision`.
