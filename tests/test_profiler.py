from __future__ import annotations

import csv

import torch

from lightweight_hbcc.config import load_config
from lightweight_hbcc.models import build_model
from lightweight_hbcc.profiler import append_record_csv, benchmark_model, estimate_cluster_macs


def test_cluster_macs_match_fvcore_similarity_count() -> None:
    # fvcore reports 197,632 aten::matmul MACs for HBCC-Small at 32x32: exactly the similarity term.
    model = build_model(load_config("configs/hbcc_accuracy_small.yaml"))
    macs = estimate_cluster_macs(model, (1, 3, 32, 32), torch.device("cpu"))
    assert macs["cluster_similarity_macs"] == 197_632
    assert macs["cluster_extra_macs"] == 2 * 197_632


def test_cluster_macs_zero_for_cnn() -> None:
    model = build_model(load_config("configs/baselines/resnet18_cifar.yaml"))
    macs = estimate_cluster_macs(model, (1, 3, 32, 32), torch.device("cpu"))
    assert macs == {"cluster_similarity_macs": 0, "cluster_extra_macs": 0}


def test_benchmark_reports_every_metric_per_batch_size_on_cpu() -> None:
    model = build_model(load_config("configs/smoke.yaml"))
    result = benchmark_model(model, torch.device("cpu"), [1, 2], warmup=1, runs=2, repeats=2)
    for bs in (1, 2):
        assert result[f"latency_ms_b{bs}"] > 0
        assert len(result[f"latency_ms_b{bs}_repeats"]) == 2
        assert f"peak_memory_mb_b{bs}" in result
        assert result[f"throughput_b{bs}"] > 0
    assert "peak_memory_mb" not in result  # the ambiguous all-batch-sizes key is gone
    assert not any(key.startswith("latency_graph") for key in result)  # CUDA Graphs only on GPU


def test_append_record_csv_keeps_union_of_columns(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    append_record_csv({"config_id": "a", "latency_ms_b1": 1.0}, path)
    append_record_csv({"config_id": "b", "latency_ms_b1": 2.0, "graph_error_b1": "boom"}, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["config_id"] for r in rows] == ["a", "b"]
    assert rows[0]["graph_error_b1"] == "" and rows[1]["graph_error_b1"] == "boom"
