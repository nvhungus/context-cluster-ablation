from __future__ import annotations

import csv
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .metrics import count_params, flop_count, model_size_mb


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def host_info(device: torch.device) -> dict[str, Any]:
    """Hardware/software context that must be reported next to latency numbers."""

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cpu_name": _cpu_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "precision": "fp32",
    }


def _timed_ms(fn, device: torch.device, runs: int) -> float:
    """Mean wall time per call, synchronizing after every call (strict latency)."""

    start = time.perf_counter()
    for _ in range(runs):
        fn()
        synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / runs


def _cuda_graph_latency(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
    repeats: int,
    min_warmup_s: float = 1.0,
) -> dict[str, Any]:
    """Latency with the forward pass captured in a CUDA Graph (no per-kernel CPU launch overhead).

    The graph replays exactly the kernels of the eager forward pass; its output is compared
    against eager so a silent capture error cannot produce a misleading latency.
    """

    eager_out = model(x).float()
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(max(3, warmup // 10)):
            model(x)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = model(x)
    graph.replay()
    synchronize(device)
    max_abs_diff = float((static_out.float() - eager_out).abs().max().item())
    matches = bool(torch.allclose(static_out.float(), eager_out, rtol=1e-3, atol=1e-4))
    # A replay takes ~1 ms, so a fixed small replay count does not let GPU clocks settle:
    # warm up for at least `min_warmup_s` seconds and `warmup` replays.
    start = time.perf_counter()
    replays = 0
    while replays < warmup or time.perf_counter() - start < min_warmup_s:
        graph.replay()
        replays += 1
        if replays % 50 == 0:
            synchronize(device)
    synchronize(device)
    reps = [_timed_ms(graph.replay, device, runs) for _ in range(repeats)]
    del graph, static_out
    return {
        "latency_ms": statistics.median(reps),
        "latency_ms_repeats": reps,
        "max_abs_diff": max_abs_diff,
        "matches_eager": matches,
    }


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    device: torch.device,
    batch_sizes: list[int],
    image_size: int = 32,
    warmup: int = 30,
    runs: int = 100,
    throughput_runs: int | None = None,
    repeats: int = 3,
    cuda_graph: bool = True,
    graph_repeats: int = 5,
) -> dict[str, Any]:
    """Per-batch-size latency (eager and CUDA Graph), throughput and peak memory.

    Every metric is keyed by batch size. Peak memory is reset before each batch size, so
    ``peak_memory_mb_b1`` is the true batch-1 peak (weights + activations of one forward).
    Latencies are the median over ``repeats`` (eager) or ``graph_repeats`` (CUDA Graph) timed
    loops of ``runs`` iterations each.

    All eager latency/memory/throughput measurements run first; CUDA Graphs are captured only
    afterwards, because capture leaves cuBLAS workspaces allocated (~35 MB on a T4) that would
    otherwise inflate the peak memory of every later batch size.
    """

    model.eval().to(device)
    throughput_runs = throughput_runs or runs
    results: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)
        results["weights_memory_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            model(x)
        synchronize(device)
        reps = [_timed_ms(lambda: model(x), device, runs) for _ in range(repeats)]
        results[f"latency_ms_b{batch_size}"] = statistics.median(reps)
        results[f"latency_ms_b{batch_size}_repeats"] = reps
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / (1024**2)
            results[f"peak_memory_mb_b{batch_size}"] = peak
            # Peak includes the resident weights; activations are what actually scales with batch
            # size, and at large batch they dominate. Report both so the two are not conflated.
            results[f"activation_memory_mb_b{batch_size}"] = peak - results.get("weights_memory_mb", 0.0)
        else:
            results[f"peak_memory_mb_b{batch_size}"] = None
            results[f"activation_memory_mb_b{batch_size}"] = None

        start = time.perf_counter()
        for _ in range(throughput_runs):
            model(x)
        synchronize(device)
        stream_time = time.perf_counter() - start
        results[f"throughput_b{batch_size}"] = batch_size * throughput_runs / max(stream_time, 1e-12)
        del x

    if cuda_graph and device.type == "cuda":
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, image_size, image_size, device=device)
            try:
                graph = _cuda_graph_latency(model, x, device, warmup, runs, graph_repeats)
                results[f"latency_graph_ms_b{batch_size}"] = graph["latency_ms"]
                results[f"latency_graph_ms_b{batch_size}_repeats"] = graph["latency_ms_repeats"]
                results[f"graph_max_abs_diff_b{batch_size}"] = graph["max_abs_diff"]
                results[f"graph_matches_eager_b{batch_size}"] = graph["matches_eager"]
            except Exception as exc:  # noqa: BLE001 - record and continue with the other batch sizes
                results[f"graph_error_b{batch_size}"] = repr(exc)
            del x
            torch.cuda.empty_cache()
    return results


def profile_operators(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    image_size: int = 32,
    warmup: int = 5,
    active: int = 10,
    out_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    model.eval().to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=active),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(1 + warmup + active):
            model(x)
            prof.step()
    rows = []
    for item in prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=30).splitlines():
        rows.append({"row": item})
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(row["row"] for row in rows), encoding="utf-8")
    return rows


def _is_cluster_op(module: nn.Module) -> bool:
    # Duck-typed so the same profiler works with the vendored Food-101 package's ContextClusterOp.
    return type(module).__name__ == "ContextClusterOp" and all(
        hasattr(module, attr) for attr in ("proposal", "fold", "heads", "head_dim")
    )


@torch.no_grad()
def estimate_cluster_macs(model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> dict[str, int]:
    """MACs of the context-clustering matmuls, per forward pass.

    For each ContextClusterOp and head, with N points and M centers per fold tile:
    similarity = N*M*d, aggregation = N*M*d, dispatch = N*M*d (d = head_dim; summed over tiles,
    N is the full feature map). fvcore counts the similarity matmul (aten::matmul) but not the
    aggregation/dispatch, which are elementwise mul+sum; ``cluster_extra_macs`` is exactly the
    part fvcore misses, so ``macs + cluster_extra_macs`` has no double counting.
    """

    per_term = 0

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], _: torch.Tensor) -> None:
        nonlocal per_term
        b, _, h, w = inputs[0].shape
        centers_per_tile = module.proposal[0] * module.proposal[1]
        per_term += b * module.heads * h * w * centers_per_tile * module.head_dim

    handles = [m.register_forward_hook(hook) for m in model.modules() if _is_cluster_op(m)]
    was_training = model.training
    model.eval().to(device)
    try:
        model(torch.randn(input_shape, device=device))
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    return {
        "cluster_similarity_macs": int(per_term),
        "cluster_extra_macs": int(2 * per_term),
    }


def model_static_metrics(model: nn.Module, device: torch.device, image_size: int = 32) -> dict[str, Any]:
    record: dict[str, Any] = {}
    record.update(count_params(model))
    record["model_size_mb"] = model_size_mb(model)
    record.update(flop_count(model, (1, 3, image_size, image_size), device))
    record.update(estimate_cluster_macs(model, (1, 3, image_size, image_size), device))
    record["macs_incl_cluster"] = (
        record["macs"] + record["cluster_extra_macs"] if record.get("macs") is not None else None
    )
    record["bops"] = estimate_bops(model, (1, 3, image_size, image_size), device)
    return record


def write_record(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def estimate_bops(model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> int:
    total = 0
    handles = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], _: torch.Tensor) -> None:
        nonlocal total
        total += module.theoretical_bops(tuple(inputs[0].shape))

    for module in model.modules():
        if _is_cluster_op(module) and hasattr(module, "theoretical_bops"):
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval().to(device)
    model(torch.randn(input_shape, device=device))
    model.train(was_training)
    for handle in handles:
        handle.remove()
    return int(total)


def append_record_csv(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    rows: list[dict[str, Any]] = []
    if out.exists():
        with out.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(flat)
    # Records may carry different keys (e.g. graph_error_b*), so rewrite with the union of columns.
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
