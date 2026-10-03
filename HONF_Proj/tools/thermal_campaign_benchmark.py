#!/usr/bin/env python3
"""Frozen-checkpoint native wrapper and prepared-decode timings, separately.

Counts are measured in a separate forward pass, never inside latency scopes.
Input staging, checkpoint load, warm-up and organizer export are excluded.
CPU RSS is process-scoped; CUDA peaks are reset for each timed invocation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def allowed_device(value):
    device = torch.device(value)
    if device.type == "cuda":
        logical = device.index or 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = visible.split(",")[logical] if visible else str(logical)
        if physical.strip() not in {"1", "2"}:
            raise ValueError("Campaign GPU measurements are limited to physical GPU 1 or 2")
    return device


def output_directory(value):
    output = Path(value).expanduser().resolve()
    if output.is_relative_to(PROJECT_ROOT) and not any(output.is_relative_to(PROJECT_ROOT / name)
            for name in ("diagnostics", "Trained_Results")):
        raise ValueError("Numerical evidence must live under ignored diagnostics or Trained_Results")
    output.mkdir(parents=True, exist_ok=True)
    return output


def low_high_indices(dataset, *, count=2):
    entries = []
    for index, case_id in enumerate(dataset.selected_case_ids):
        if str(case_id).lstrip("0") == "273":
            continue
        group = dataset.h5["cases"][case_id]
        entries.append((int((np.asarray(group["module_present"]) > .5).sum()), str(case_id), index))
    entries.sort()
    if not entries:
        raise ValueError("No eligible timing cases")
    positions = np.linspace(0, len(entries) - 1, min(count, len(entries))).round().astype(int)
    return [entries[position][2] for position in positions]


def _cpu_rss():
    # Linux current resident memory, not an inferred per-call peak.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def latency_samples(call, device, *, repeats=9, warmup=2):
    if repeats < 1 or warmup < 0:
        raise ValueError("Positive repeats and nonnegative warm-up are required")
    for _ in range(warmup):
        result = call()
        del result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        else:
            baseline = None
        before = _cpu_rss()
        start = perf_counter()
        result = call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = perf_counter() - start
        samples.append({"seconds": seconds, "cpu_rss_before_bytes": before,
            "cpu_rss_after_bytes": _cpu_rss(), "cpu_process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_baseline_allocated_bytes": baseline,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if baseline is not None else None,
            "cuda_peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline if baseline is not None else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if baseline is not None else None})
        del result
    seconds = np.asarray([sample["seconds"] for sample in samples])
    return {"median_seconds": float(np.median(seconds)), "p90_seconds": float(np.quantile(seconds, .9)),
        "min_seconds": float(seconds.min()), "repeats": repeats, "warmup": warmup, "samples": samples,
        "memory_scope": "CUDA invocation peaks include resident model/input/prepared state; extra peak subtracts allocated baseline. CPU RSS is process-scoped, lifetime peak is not a call peak."}


def native_arguments(model, sample, query, device):
    from channelthermal.evaluation.loading import make_batch
    batch = make_batch(sample, np.asarray(query, dtype=np.float32), device)
    return {"structure": batch["structure"], "query_xy": batch["query_xy"],
        "interface_condition": batch.get("interface_condition"), "local_module_params": batch.get("local_module_params"),
        "teacher_port_tokens": batch.get("teacher_port_tokens"), "local_query_points": batch.get("module_internal_query_points"),
        "local_port_condition_mode": "predicted", "mixed_teacher_ratio": 0., "return_routing_maps": False,
        "return_prepared_state": True}


def load_native(checkpoint_path, dataset_path, device):
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model

    from honf_runtime.compat import resolve_demo_path
    model, checkpoint = load_model(Path(checkpoint_path), device)
    model.eval().requires_grad_(False)
    settings = checkpoint.get("train_config", {}).get("dataset", {})
    path = resolve_demo_path(dataset_path or settings["packed_h5_path"])
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    normalized = GlobalChannelThermalDataset(path, split="test", points_per_case=1, random_point_sampling=False,
        include_grid=True, normalizer=H5Normalizer(stats) if stats else None,
        normalize_inputs=bool(settings.get("normalize_inputs", False)), normalize_targets=bool(settings.get("normalize_targets", False)))
    raw = GlobalChannelThermalDataset(path, split="test", points_per_case=1, random_point_sampling=False, include_grid=True)
    return model, checkpoint, normalized, raw, path


def benchmark_model_case(model, sample, raw_sample, device, *, repeats, warmup, small_queries=14):
    from channelthermal.training.campaign_work import CampaignForwardWork
    from thermal_campaign_heat_inference import sensor_panel
    sensors, _, _, _, _ = sensor_panel(raw_sample)
    if small_queries <= len(sensors):
        small = sensors[:small_queries]
    else:
        grid = np.stack((sample["x_grid"].ravel(), sample["y_grid"].ravel()), -1)
        small = grid[np.linspace(0, len(grid) - 1, min(small_queries, len(grid))).round().astype(int)]
    full = np.stack((sample["x_grid"].ravel(), sample["y_grid"].ravel()), -1)
    rows = []
    with torch.no_grad():
        for panel, query in (("small_inverse", small), ("full_native_grid", full)):
            arguments = native_arguments(model, sample, query, device)
            call = lambda arguments=arguments: model(**arguments)
            wrapper = latency_samples(call, device, repeats=repeats, warmup=warmup)
            # A separate work pass avoids timing recorder synchronization and
            # supplies the reusable prepared state for decode-only repeats.
            with CampaignForwardWork(model.core) as recorded:
                result = call()
            prepared = result["prepared_state"]
            wrapper["forward_work"] = recorded.records
            decoder = lambda prepared=prepared, query=arguments["query_xy"]: model.decode_prepared(prepared, query, return_routing_maps=False)
            decoded = latency_samples(decoder, device, repeats=repeats, warmup=warmup)
            with CampaignForwardWork(model.core) as recorded_decode:
                decoder()
            decoded["forward_work"] = recorded_decode.records
            rows.extend(({"panel": panel, "queries": len(query), "scope": "complete_wrapper_P0_P1_P2", **wrapper},
                         {"panel": panel, "queries": len(query), "scope": "prepared_P2_decode_only", **decoded}))
            del prepared, result
    return rows


def evaluate(args):
    device = allowed_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model, checkpoint, normalized, raw, path = load_native(args.checkpoint, args.dataset, device)
    output = output_directory(args.output_dir)
    rows = []
    original = getattr(model.core.backend, "execution_mode", None)
    original_chunk = getattr(model.core.backend, "execution_receiver_chunk", args.executor_receiver_chunk)
    for executor in args.executors:
        if executor == "checkpoint" and original is not None:
            model.core.backend.set_execution_mode(original, receiver_chunk_size=original_chunk)
        elif executor != "checkpoint":
            setter = getattr(model.core.backend, "set_execution_mode", None)
            if setter is None:
                raise ValueError("Executor override requires a shared typed checkpoint")
            setter(executor, receiver_chunk_size=args.executor_receiver_chunk)
        for index in low_high_indices(raw, count=args.cases):
            sample, reference = normalized[index], raw[index]
            measured = benchmark_model_case(model, sample, reference, device, repeats=args.repeats,
                warmup=args.warmup, small_queries=args.small_queries)
            for row in measured:
                row.update(case_id=str(reference["case_id"]), module_count=int((reference["structure"]["module_present"] > .5).sum()),
                    executor=getattr(model.core.backend, "execution_mode", "native_checkpoint_backend"),
                    core_receiver_chunk=model.core.receiver_chunk_size,
                    executor_receiver_chunk=getattr(model.core.backend, "execution_receiver_chunk", None))
                rows.append(row)
            payload = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
                "architecture": model.config.core_honf.forward_architecture, "dataset": str(path), "partition": "exposed development",
                "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_cpu_threads": torch.get_num_threads(), "torch_version": torch.__version__,
                "scope_limit": "warm frozen model invocation only; staging/load/export and work recording excluded. Complete-wrapper and prepared decoding are independent timings and must not be added.",
                "rows": rows}
            (output / "timing.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(f"{reference['case_id']} {executor}: saved four separate timing scopes", flush=True)
    if original is not None:
        model.core.backend.set_execution_mode(original, receiver_chunk_size=original_chunk)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--small-queries", type=int, default=14)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--executor-receiver-chunk", type=int, default=128)
    parser.add_argument("--executors", nargs="+", default=["checkpoint"],
        choices=("checkpoint", "dense_masked_reference", "rectangular_subset"))
    args = parser.parse_args(argv)
    if min(args.cases, args.repeats, args.small_queries, args.cpu_threads, args.executor_receiver_chunk) <= 0 or args.warmup < 0:
        parser.error("positive timing sizes and nonnegative warm-up are required")
    return args


if __name__ == "__main__":
    print(evaluate(parse_args()))
