"""Bounded Stage-4 decoder benchmark for additive background modes.

This is a synthetic diagnostic, not a scientific training or accuracy run.
It reuses one materialized model and prepared organizer state, changes only the
parameter-free background execution mode, and caps warmup/measured iterations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField


def _batch(device: torch.device, query_count: int) -> BatchData:
    generator = torch.Generator(device="cpu").manual_seed(417)
    module_present = torch.ones(1, 16)
    return BatchData(
        module_centers=(torch.rand(1, 16, 2, generator=generator) * torch.tensor([12.0, 6.0])).to(device),
        module_present=module_present.to(device),
        module_features=torch.randn(1, 16, 12, generator=generator).to(device),
        global_context=torch.randn(1, 8, generator=generator).to(device),
        query_xy=(
            torch.rand(1, query_count, 2, generator=generator) * torch.tensor([12.0, 6.0])
        ).to(device),
        query_time=None,
        target_field=None,
        case_name="stage4-background-benchmark",
        metadata={},
    )


def _config() -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        field_dim=5,
        domain_length_x=12.0,
        domain_length_y=6.0,
        coordinate_scale=[12.0, 6.0],
        periodic_axes=[],
        num_env_tokens_x=24,
        num_env_tokens_y=8,
        num_hyperedges=8,
        organizer_mode="exchangeable_slots",
        edge_capacity=8,
        initial_active_edges=8,
        minimum_active_edges=2,
        edge_selection_mode="all",
        hidden_dim=128,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=128,
        pairwise_kernel_num_layers=3,
        mechanism_state_mode="descriptor_first",
        field_assembly_mode="edge_additive",
        additive_background_mode="dense_query_attention",
        routing_execution="dense",
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    model: HONFNeuralField,
    batch: BatchData,
    organized: dict[str, torch.Tensor],
    *,
    mode: str,
    warmups: int,
    iterations: int,
) -> dict[str, float | int | str | None]:
    model.config.additive_background_mode = mode
    model.decoder.config.additive_background_mode = mode
    for _ in range(warmups):
        model.decode_queries(batch.query_xy, None, organized, organized["global_token"])
    _sync(batch.query_xy.device)
    if batch.query_xy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(batch.query_xy.device)
        baseline_allocated = torch.cuda.memory_allocated(batch.query_xy.device)
    else:
        baseline_allocated = 0
    durations = []
    for _ in range(iterations):
        _sync(batch.query_xy.device)
        started = time.perf_counter()
        model.decode_queries(batch.query_xy, None, organized, organized["global_token"])
        _sync(batch.query_xy.device)
        durations.append(time.perf_counter() - started)
    if batch.query_xy.device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(batch.query_xy.device)
        peak_reserved = torch.cuda.max_memory_reserved(batch.query_xy.device)
        incremental_peak = max(peak_allocated - baseline_allocated, 0)
    else:
        peak_allocated = peak_reserved = incremental_peak = None
    return {
        "mode": mode,
        "median_decoder_seconds": statistics.median(durations),
        "minimum_decoder_seconds": min(durations),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": incremental_peak,
        "warmups": warmups,
        "iterations": iterations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.warmups <= 10:
        raise ValueError("--warmups must be in [0, 10].")
    if not 1 <= args.iterations <= 50:
        raise ValueError("--iterations must be in [1, 50].")
    if args.queries <= 0:
        raise ValueError("--queries must be positive.")

    device = torch.device(args.device)
    torch.manual_seed(419)
    model = HONFNeuralField(_config()).to(device).eval()
    batch = _batch(device, args.queries)
    with torch.no_grad():
        organized = model.encode_and_organize(batch, organizer_selection_override="all")
        results = [
            _measure(
                model,
                batch,
                organized,
                mode=mode,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            for mode in ("dense_query_attention", "global_pooled_attention")
        ]
    dense, pooled = results
    payload = {
        "device": str(device),
        "torch_version": torch.__version__,
        "query_count": args.queries,
        "environment_token_count": 24 * 8,
        "results": results,
        "pooled_to_dense_time_ratio": (
            float(pooled["median_decoder_seconds"]) / float(dense["median_decoder_seconds"])
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
