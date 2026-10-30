# Experimental protocol

This file is the single source for the training recipe and for how every reported number is
produced. Every run is listed by ID in `configs/run_plan.yaml` and executed with
`tools/run_queue.py`. Every number reported comes from a run executed under this protocol.

## Datasets and splits

| Dataset | Train / val / test | Resolution | Split rule |
|---|---|---|---|
| CIFAR-10 | 45,000 / 5,000 / 10,000 | 32x32 | Val = 5k random images of the official train split (`split_seed: 42`); test = official test split |
| Tiny-ImageNet-200 | 90,000 / 10,000 / 10,000 | 64x64 resized to 32x32 | Val = 10k random images of the official train split (`split_seed: 42`); **test = the official validation split** (the official test split has no labels) |
| Food-101 (separate pipeline, not part of this study) | 68,175 / 7,575 / 25,250 | 224x224 | Val = 75 images per class, stratified, from the official train split (`split_seed: 42`); test = official test split. Pipeline: `food101/` |

The split seed is fixed at 42 for all runs; only the training seed varies.

## Training (CIFAR-10 and Tiny-ImageNet, all models identical)

| Item | Value |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 (all parameters) |
| Schedule | 5 warmup epochs (linear from 1%), then cosine to 0 |
| Epochs | **CIFAR-10: 250** for every model (CE and KD); **Tiny-ImageNet: 100** for every model |
| Batch size | 128 (train), 256 (val/test) |
| Loss | Cross-entropy with label smoothing 0.1 on MixUp/CutMix soft targets |
| Augmentation, CIFAR-10 | RandomCrop(32, padding 4), horizontal flip, RandAugment(N=2, M=9), RandomErasing(p=0.25) |
| Augmentation, Tiny-ImageNet | RandomResizedCrop(32, scale 0.7-1.0), horizontal flip, RandAugment(N=2, M=9), RandomErasing(p=0.25) |
| MixUp / CutMix | alpha 0.2 / alpha 1.0, CutMix probability 0.5 |
| Precision | AMP (fp16 autocast) for training |
| Drop path | HBCC-Small 0.05, HBCC-Medium 0.08 (and its ablations 0.08), CoC-baseline 0.05, CNN baselines 0 |
| KD (Hinton) | alpha 0.5, T = 4; teacher = ResNet-18 CE run with the **same dataset and seed**; teacher sees the same mixed batch |
| Model selection | Checkpoint with the best validation top-1 (`best.pth`); the test split is evaluated once, on that checkpoint |

## Seeds

- Training seeds: 42, 43, 44 (`train.seed`); `data.loader_seed` defaults to the training seed.
- Tiny-ImageNet main table (ResNet-18, ShuffleNetV2, HBCC-S/M CE and KD, all-cluster, ShuffleNetV2+KD): 3 seeds.
- Tiny-ImageNet MobileNetV2, CoC-baseline, MobileNetV2+KD: 1 seed (6-9 pp below the compared models).
- Tiny-ImageNet ablations: 1 seed; seeds 43 and 44 added (runs `C*`) for every ablation whose seed-42
  best validation top-1 is within 1.0 pp of HBCC-Medium CE seed 42.
- CIFAR-10: 3 seeds (42, 43, 44) for the proposed model, ShuffleNetV2-x1.5 and the teacher;
  1 seed (42) for the remaining baselines, which are marked as single-seed wherever reported.
- Food-101: 1 seed (42); stated as a limitation.
- cuDNN is non-deterministic (`train.deterministic: false`), so two runs with the same seed can differ
  slightly; the variance across seeds includes this.

## Reporting

- Accuracy: top-1 on the **test** split, mean +- std over seeds (sample std), with the number of
  seeds stated; single-seed rows are marked. Every reported value is recomputed from the
  per-image `test_predictions.npy` in `release/runs/`, not read from `test_metrics.json`: the two
  disagree in 30 of the 91 runs by one to four images out of 10,000, because the training script's
  logged top-1 comes from `output.topk(5)` while the saved predictions come from
  `output.argmax`, and the two break exact logit ties differently under mixed precision. The
  validation split is used only to select the checkpoint.
- Comparisons: per-seed differences, Welch's unequal-variance t-test on the seed means, and an
  exact paired McNemar test on the shared test images within each seed
  (`tools/aggregate_seeds.py --compare`). Welch rather than a paired t-test: the two groups are
  independently trained models, not repeated measures on one model.
- Parameters: all parameters including the fixed (non-learned) LBP filters; learned count also given.
- MACs: fvcore count plus the clustering aggregation/dispatch matmuls that fvcore does not count
  (`macs_incl_cluster`); fvcore already counts the similarity matmul.
- Latency: Kaggle T4, fp32 inference, batch sizes 1/16/64/128, in a single profiling session for
  all models on one host (`results/cost_t4_m1v3/`). Each timed loop is 100 iterations after a
  warm-up of at least one second; the reported figure is the median of **3** loops in eager mode
  and **5** loops for CUDA-Graph replay, and every individual loop is kept in the `*_repeats`
  columns. Each graph capture is checked against its eager output
  (`graph_matches_eager_b*`, `graph_max_abs_diff_b*`). GPU and CPU model are recorded.
  Re-profiling the same models in a separate session leaves parameters, operation counts and peak
  memory bit-identical but shifts batch-128 latency by up to 14%, so one session is used
  throughout.
- Peak memory: measured separately per batch size (`peak_memory_mb_b{bs}`), labelled with the batch size.
