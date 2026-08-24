#!/usr/bin/env python3
"""Benchmark frozen HONF checkpoints through identical evaluation-only paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "diagnostics"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer  # noqa: E402
from channelthermal.workflows.evaluate_forward import load_model, make_batch  # noqa: E402
from frozen_override_cli import (  # noqa: E402
    add_frozen_override_arguments,
    apply_label_frozen_overrides,
    resolve_frozen_overrides,
)
from honf_forward_core.config import UnifiedForwardConfig  # noqa: E402
from honf_forward_core.decoding.pairwise import HypergraphGatedPairwiseKernel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument(
        "--synthetic-scaling",
        action="store_true",
        help="Benchmark edge-explicit, fused legacy, and factorized kernels over a Q/M grid.",
    )
    parser.add_argument(
        "--synthetic-queries",
        type=int,
        nargs="+",
        default=[8192, 65536, 262144, 1_000_000],
    )
    parser.add_argument(
        "--synthetic-modules",
        type=int,
        nargs="+",
        default=[5, 12, 32, 64, 128],
    )
    parser.add_argument("--synthetic-query-chunk-size", type=int, default=8192)
    parser.add_argument("--synthetic-hidden-dim", type=int, default=256)
    parser.add_argument("--synthetic-hyperedges", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", default="0273")
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "generated" / "stage5_final_checkpoint_benchmark.json",
    )
    add_frozen_override_arguments(parser)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_call(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        with torch.inference_mode():
            output = function()
        del output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_allocated = baseline_reserved = None
    elapsed: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - started)
        del output
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    return {
        "median_seconds": float(statistics.median(elapsed)),
        "mean_seconds": float(statistics.mean(elapsed)),
        "p05_seconds": float(np.quantile(elapsed, 0.05)),
        "p95_seconds": float(np.quantile(elapsed, 0.95)),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": (
            None if peak_allocated is None else peak_allocated - int(baseline_allocated)
        ),
        "incremental_peak_reserved_bytes": (
            None if peak_reserved is None else peak_reserved - int(baseline_reserved)
        ),
    }


def evaluate_checkpoint(
    label: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()
    frozen_overrides = apply_label_frozen_overrides(model, label, args)
    dataset_cfg = checkpoint["train_config"]["dataset"]
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split=args.split,
        points_per_case=1,
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
        normalizer=H5Normalizer(stats),
    )
    matches = [
        index
        for index, case_id in enumerate(dataset.selected_case_ids)
        if str(case_id) == str(args.case_id)
    ]
    if not matches:
        raise KeyError(f"case_id={args.case_id!r} is absent from split {args.split!r}")
    sample = dataset[matches[0]]
    all_queries = np.stack(
        (np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)),
        axis=-1,
    ).astype(np.float32)
    queries = all_queries[: min(int(args.queries), len(all_queries))]
    batch = make_batch(sample, queries, device)

    def full_forward() -> dict[str, Any]:
        return model(
            batch["structure"],
            batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
        )

    with torch.inference_mode():
        prepared_output = model(
            batch["structure"],
            batch["query_xy"][:, :1],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
            return_prepared_state=True,
        )
    prepared = prepared_output["prepared_state"]
    del prepared_output

    def prepared_decode() -> dict[str, Any]:
        return model.decode_prepared(prepared, batch["query_xy"])

    state_keys_before = tuple(model.state_dict())
    decoder = benchmark_call(
        prepared_decode,
        device=device,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
    )
    full = benchmark_call(
        full_forward,
        device=device,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
    )
    state_keys_after = tuple(model.state_dict())
    if state_keys_before != state_keys_after:
        raise RuntimeError(f"{label}: benchmark changed state_dict structure")
    parameters = list(model.parameters())
    core_cfg = checkpoint.get("model_config", {}).get("core_honf", {})
    result = {
        "checkpoint": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "case_id": str(sample["case_id"]),
        "query_count": int(len(queries)),
        "organizer_mode": core_cfg.get("organizer_mode"),
        "field_assembly_mode": core_cfg.get("field_assembly_mode"),
        "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "state_dict_key_count": len(state_keys_before),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "prepared_decoder": decoder,
        "full_forward": full,
        "state_dict_structure_unchanged": state_keys_before == state_keys_after,
        "frozen_overrides": frozen_overrides,
    }
    dataset.close()
    del prepared, batch, model, checkpoint
    torch.cuda.empty_cache()
    return result


def synthetic_pairwise_scaling(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Measure prepared-decoder pairwise scaling without checkpoint or dataset I/O."""

    hidden_dim = int(args.synthetic_hidden_dim)
    num_edges = int(args.synthetic_hyperedges)
    query_chunk_size = int(args.synthetic_query_chunk_size)
    if hidden_dim <= 0 or num_edges <= 0 or query_chunk_size <= 0:
        raise ValueError(
            "Synthetic hidden dimension, hyperedge count, and query chunk size must be positive"
        )
    results: list[dict[str, Any]] = []
    for num_modules in args.synthetic_modules:
        for num_queries in args.synthetic_queries:
            if int(num_modules) <= 0 or int(num_queries) <= 0:
                raise ValueError("Synthetic query and module counts must be positive")
            generator = torch.Generator(device=device).manual_seed(
                1701 + int(num_modules) * 11 + int(num_queries)
            )
            query_xy = torch.rand(1, int(num_queries), 2, generator=generator, device=device)
            module_centers = torch.rand(1, int(num_modules), 2, generator=generator, device=device)
            module_tokens = torch.randn(
                1,
                int(num_modules),
                hidden_dim,
                generator=generator,
                device=device,
            )
            raw_features = torch.randn(
                1,
                int(num_modules),
                8,
                generator=generator,
                device=device,
            )
            A_mh = torch.softmax(
                4.0
                * torch.randn(
                    1,
                    int(num_modules),
                    num_edges,
                    generator=generator,
                    device=device,
                ),
                dim=1,
            )
            hyper_attention = torch.softmax(
                4.0
                * torch.randn(
                    1,
                    int(num_queries),
                    num_edges,
                    generator=generator,
                    device=device,
                ),
                dim=-1,
            )

            def config(
                aggregation: str,
                kernel: str,
                *,
                retained_mass_floor: float = 1.0,
            ) -> UnifiedForwardConfig:
                return UnifiedForwardConfig(
                    field_dim=5,
                    num_hyperedges=num_edges,
                    hidden_dim=hidden_dim,
                    dropout=0.0,
                    decoder_mode="enhanced_honf_pairwise",
                    field_assembly_mode="context_fusion",
                    pairwise_aggregation_mode=aggregation,
                    pairwise_kernel_mode=kernel,
                    pairwise_kernel_hidden_dim=96 if kernel == "factorized_gated" else hidden_dim,
                    pairwise_kernel_num_layers=2 if kernel == "factorized_gated" else 4,
                    routing_execution="dense",
                    query_module_retained_mass_floor=retained_mass_floor,
                )

            mode_names = (
                "edge_explicit",
                "fused_legacy",
                "fused_sparse_beta_0p98",
                "factorized_gated_r96",
                "factorized_sparse_beta_0p98_r96",
            )
            organizers = {
                name: {
                    "module_centers": module_centers,
                    "module_tokens": module_tokens,
                    "module_present": torch.ones(
                        1, int(num_modules), device=device, dtype=query_xy.dtype
                    ),
                    "module_features_raw": raw_features,
                    "A_mh": A_mh,
                }
                for name in mode_names
            }
            torch.manual_seed(1701)
            edge_explicit = HypergraphGatedPairwiseKernel(
                config("edge_explicit", "legacy_mlp")
            ).to(device).eval()
            torch.manual_seed(1701)
            fused_legacy = HypergraphGatedPairwiseKernel(
                config("fused_query_module", "legacy_mlp")
            ).to(device).eval()
            torch.manual_seed(1701)
            factorized = HypergraphGatedPairwiseKernel(
                config("fused_query_module", "factorized_gated")
            ).to(device).eval()
            torch.manual_seed(1701)
            fused_sparse = HypergraphGatedPairwiseKernel(
                config(
                    "fused_query_module",
                    "legacy_mlp",
                    retained_mass_floor=0.98,
                )
            ).to(device).eval()
            torch.manual_seed(1701)
            factorized_sparse = HypergraphGatedPairwiseKernel(
                config(
                    "fused_query_module",
                    "factorized_gated",
                    retained_mass_floor=0.98,
                )
            ).to(device).eval()
            kernels = {
                "edge_explicit": edge_explicit,
                "fused_legacy": fused_legacy,
                "fused_sparse_beta_0p98": fused_sparse,
                "factorized_gated_r96": factorized,
                "factorized_sparse_beta_0p98_r96": factorized_sparse,
            }

            def run_chunked(
                kernel: HypergraphGatedPairwiseKernel,
                name: str,
            ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
                output = None
                for start in range(0, int(num_queries), query_chunk_size):
                    output = kernel(
                        query_xy[:, start : start + query_chunk_size],
                        organizers[name],
                        hyper_attention[:, start : start + query_chunk_size],
                        gathered_execution="sparse" in name,
                    )
                assert output is not None
                return output

            with torch.inference_mode():
                initialized = {
                    name: kernel(
                        query_xy[:, : min(int(num_queries), query_chunk_size)],
                        organizers[name],
                        hyper_attention[:, : min(int(num_queries), query_chunk_size)],
                        gathered_execution="sparse" in name,
                    )[0]
                    for name, kernel in kernels.items()
                }
            fused_legacy.load_state_dict(edge_explicit.state_dict(), strict=True)
            fused_sparse.load_state_dict(edge_explicit.state_dict(), strict=True)
            factorized_sparse.load_state_dict(factorized.state_dict(), strict=True)
            with torch.inference_mode():
                initialized["edge_explicit"] = edge_explicit(
                    query_xy[:, : min(int(num_queries), query_chunk_size)],
                    organizers["edge_explicit"],
                    hyper_attention[:, : min(int(num_queries), query_chunk_size)],
                )[0]
                initialized["fused_legacy"] = fused_legacy(
                    query_xy[:, : min(int(num_queries), query_chunk_size)],
                    organizers["fused_legacy"],
                    hyper_attention[:, : min(int(num_queries), query_chunk_size)],
                )[0]
            parity = float(
                (initialized["edge_explicit"] - initialized["fused_legacy"])
                .abs()
                .amax()
                .detach()
                .cpu()
            )
            for name, kernel in kernels.items():
                timing = benchmark_call(
                    lambda kernel=kernel, name=name: run_chunked(kernel, name),
                    device=device,
                    warmup=int(args.warmup),
                    iterations=int(args.iterations),
                )
                with torch.inference_mode():
                    diagnostic_output = run_chunked(kernel, name)[2]
                selected_ratio = float(
                    diagnostic_output["pairwise_selection_ratio"].detach().cpu()
                )
                results.append(
                    {
                        "mode": name,
                        "query_count": int(num_queries),
                        "module_count": int(num_modules),
                        "query_module_pair_count": int(num_queries) * int(num_modules),
                        "query_chunk_size": min(int(num_queries), query_chunk_size),
                        "kernel_parameters": int(
                            sum(parameter.numel() for parameter in kernel.parameters())
                        ),
                        "selected_pair_ratio": selected_ratio,
                        "evaluated_query_module_pair_count": int(
                            round(selected_ratio * int(num_queries) * int(num_modules))
                        ),
                        "edge_explicit_vs_fused_max_abs": (
                            parity if name in {"edge_explicit", "fused_legacy"} else None
                        ),
                        **timing,
                    }
                )
            del (
                kernels,
                edge_explicit,
                fused_legacy,
                fused_sparse,
                factorized,
                factorized_sparse,
                initialized,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "hidden_dim": hidden_dim,
        "hyperedge_count": num_edges,
        "factorized_rank": 96,
        "query_chunk_size": query_chunk_size,
        "routing_fixture": "seeded concentrated softmax logits (scale=4) over modules and edges",
        "rows": results,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if not args.checkpoint and not args.synthetic_scaling:
        raise ValueError("Provide at least one --checkpoint or enable --synthetic-scaling")
    if args.checkpoint and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Checkpoint benchmarks require CUDA.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is unavailable")
    checkpoints: dict[str, Path] = {}
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        checkpoints[label] = Path(raw_path).expanduser().resolve()
    if checkpoints:
        resolve_frozen_overrides(args, checkpoints)
    results = {
        label: evaluate_checkpoint(label, path, args, device)
        for label, path in checkpoints.items()
    }
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "split": args.split,
        "case_id": str(args.case_id),
        "query_count": int(args.queries),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "results": results,
        "synthetic_pairwise_scaling": (
            synthetic_pairwise_scaling(args, device) if args.synthetic_scaling else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
