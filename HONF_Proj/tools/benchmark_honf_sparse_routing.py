#!/usr/bin/env python3
"""Benchmark dense and gathered HONF routing on synthetic CUDA workloads."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField


FULL_MODULE_COUNTS = (12, 32, 64, 128)
FULL_QUERY_COUNTS = (1024, 8192, 32768)
FULL_EDGE_CAPACITIES = (6, 8, 12, 16)


def _integer_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modules", type=_integer_list, default=(64,))
    parser.add_argument("--queries", type=_integer_list, default=(8192,))
    parser.add_argument("--edge-capacities", type=_integer_list, default=(8,))
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--module-limit", type=int, default=8)
    parser.add_argument("--edge-limit", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--scope", choices=["prepared_decode", "full_forward"], default="prepared_decode")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.full_matrix:
        args.modules = FULL_MODULE_COUNTS
        args.queries = FULL_QUERY_COUNTS
        args.edge_capacities = FULL_EDGE_CAPACITIES
    if args.hidden_dim <= 0 or args.module_limit < 0 or args.edge_limit < 0:
        parser.error("hidden-dim must be positive and routing limits must be nonnegative")
    if args.warmup < 1 or args.repetitions < 1:
        parser.error("warmup and repetitions must be positive")
    return args


def make_config(args: argparse.Namespace, *, capacity: int, execution: str) -> UnifiedForwardConfig:
    initial = min(6, int(capacity))
    return UnifiedForwardConfig(
        field_dim=5,
        domain_length_x=12.0,
        domain_length_y=6.0,
        coordinate_scale=[12.0, 6.0],
        periodic_axes=[],
        num_env_tokens_x=24,
        num_env_tokens_y=8,
        num_hyperedges=6,
        organizer_mode="exchangeable_slots",
        edge_capacity=int(capacity),
        initial_active_edges=initial,
        minimum_active_edges=1,
        slot_refinement_steps=2,
        edge_selection_mode="all",
        module_assignment_normalizer="entmax15",
        environment_assignment_normalizer="entmax15",
        query_assignment_normalizer="entmax15",
        environment_locality_mode="bounded_gaussian",
        environment_locality_strength=1.0,
        locality_radius_cap=3.0,
        minimum_region_scale=0.05,
        hidden_dim=int(args.hidden_dim),
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=int(args.hidden_dim),
        pairwise_kernel_num_layers=4,
        pairwise_kernel_fourier_frequencies=4,
        query_fourier_frequencies=4,
        position_fourier_frequencies=4,
        mechanism_state_mode="descriptor_first",
        field_assembly_mode="edge_additive",
        routing_execution=execution,
        query_module_limit=int(args.module_limit),
        query_edge_limit=int(args.edge_limit),
    )


def make_batch(
    *,
    modules: int,
    queries: int,
    device: torch.device,
    seed: int,
) -> BatchData:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return BatchData(
        module_centers=(torch.rand(1, modules, 2, generator=generator) * torch.tensor([12.0, 6.0])).to(device),
        module_present=torch.ones(1, modules, device=device),
        module_features=torch.randn(1, modules, 6, generator=generator).to(device),
        global_context=torch.randn(1, 6, generator=generator).to(device),
        query_xy=(torch.rand(1, queries, 2, generator=generator) * torch.tensor([12.0, 6.0])).to(device),
        query_time=None,
        target_field=None,
        case_name="routing-benchmark",
        metadata={},
    )


def build_models(
    args: argparse.Namespace,
    *,
    batch: BatchData,
    capacity: int,
    device: torch.device,
) -> dict[str, HONFNeuralField]:
    models: dict[str, HONFNeuralField] = {}
    torch.manual_seed(int(args.seed))
    dense = HONFNeuralField(make_config(args, capacity=capacity, execution="dense")).to(device).eval()
    with torch.inference_mode():
        dense(batch)
    state = copy.deepcopy(dense.state_dict())
    models["dense"] = dense
    torch.manual_seed(int(args.seed) + 1)
    gathered = HONFNeuralField(make_config(args, capacity=capacity, execution="gathered")).to(device).eval()
    with torch.inference_mode():
        gathered(batch)
    gathered.load_state_dict(state, strict=True)
    models["gathered"] = gathered
    return models


def benchmark_model(
    model: HONFNeuralField,
    batch: BatchData,
    *,
    scope: str,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    with torch.inference_mode():
        if scope == "prepared_decode":
            organized = model.encode_and_organize(batch)

            def run() -> dict[str, torch.Tensor]:
                return model.decode_queries(
                    batch.query_xy,
                    None,
                    organized,
                    organized["global_token"],
                )

        else:

            def run() -> dict[str, torch.Tensor]:
                return model(batch)

        for _ in range(int(warmup)):
            output = run()
        torch.cuda.synchronize(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(int(repetitions)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = run()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
        torch.cuda.synchronize(device)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    median_ms = statistics.median(timings)
    metrics = {
        "median_ms": median_ms,
        "p95_ms": ordered[p95_index],
        "query_throughput_per_second": int(batch.query_xy.shape[1]) / (median_ms / 1000.0),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "incremental_peak_cuda_memory_bytes": int(
            max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
        ),
        "pairwise_evaluated_pair_count": int(output["pairwise_evaluated_pair_count"].item()),
        "edge_head_evaluated_route_count": int(output["edge_head_evaluated_route_count"].item()),
        "pairwise_selection_ratio": float(output["pairwise_selection_ratio"].item()),
        "edge_head_selection_ratio": float(output["edge_head_selection_ratio"].item()),
        "mean_query_nonzero_edges": float(output["mean_query_nonzero_edges"].item()),
    }
    return metrics, output["pred_field"].detach().cpu()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device and CUDA-event timing.")
    results = []
    for modules in args.modules:
        for queries in args.queries:
            for capacity in args.edge_capacities:
                batch = make_batch(modules=modules, queries=queries, device=device, seed=args.seed)
                models = build_models(args, batch=batch, capacity=capacity, device=device)
                workload_results: dict[str, Any] = {
                    "modules": modules,
                    "queries": queries,
                    "edge_capacity": capacity,
                    "hidden_dim": args.hidden_dim,
                    "scope": args.scope,
                    "module_limit": args.module_limit,
                    "edge_limit": args.edge_limit,
                }
                fields = {}
                for execution, model in models.items():
                    metrics, field = benchmark_model(
                        model,
                        batch,
                        scope=args.scope,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        device=device,
                    )
                    workload_results[execution] = metrics
                    fields[execution] = field
                difference = fields["gathered"] - fields["dense"]
                workload_results["gathered_vs_dense"] = {
                    "maximum_absolute_error": float(difference.abs().max()),
                    "relative_l2_error": float(
                        difference.norm() / fields["dense"].norm().clamp_min(1.0e-12)
                    ),
                }
                results.append(workload_results)
                print(json.dumps(workload_results, sort_keys=True))
                del models, batch, fields
                torch.cuda.empty_cache()
    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
