#!/usr/bin/env python3
"""Run the bounded A-E HONF forward-upgrade correctness ladder."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField


VARIANTS = {
    "A": {
        "organizer_mode": "fixed_projection",
        "mechanism_state_mode": "residual_concat",
        "field_assembly_mode": "context_fusion",
        "normalizer": "softmax",
        "locality": "none",
        "execution": "dense",
    },
    "B": {
        "organizer_mode": "fixed_projection",
        "mechanism_state_mode": "descriptor_first",
        "field_assembly_mode": "edge_additive",
        "normalizer": "softmax",
        "locality": "none",
        "execution": "dense",
    },
    "C": {
        "organizer_mode": "exchangeable_slots",
        "mechanism_state_mode": "descriptor_first",
        "field_assembly_mode": "edge_additive",
        "normalizer": "softmax",
        "locality": "none",
        "execution": "dense",
    },
    "D": {
        "organizer_mode": "exchangeable_slots",
        "mechanism_state_mode": "descriptor_first",
        "field_assembly_mode": "edge_additive",
        "normalizer": "entmax15",
        "locality": "compact_kernel",
        "execution": "dense",
    },
    "E": {
        "organizer_mode": "exchangeable_slots",
        "mechanism_state_mode": "descriptor_first",
        "field_assembly_mode": "edge_additive",
        "normalizer": "entmax15",
        "locality": "compact_kernel",
        "execution": "gathered",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modules", type=int, default=12)
    parser.add_argument("--active-modules", type=int, default=7)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--edge-capacity", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=263)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.modules <= 0 or not 0 < args.active_modules <= args.modules:
        parser.error("active-modules must be in [1, modules]")
    if args.queries <= 0 or args.edge_capacity < 6 or args.hidden_dim <= 0:
        parser.error("queries/hidden-dim must be positive and edge-capacity must be at least six")
    return args


def make_config(args: argparse.Namespace, variant: dict[str, str]) -> UnifiedForwardConfig:
    exchangeable = variant["organizer_mode"] == "exchangeable_slots"
    return UnifiedForwardConfig(
        field_dim=5,
        domain_length_x=12.0,
        domain_length_y=6.0,
        coordinate_scale=[12.0, 6.0],
        periodic_axes=[],
        num_env_tokens_x=16,
        num_env_tokens_y=8,
        num_hyperedges=6,
        organizer_mode=variant["organizer_mode"],
        edge_capacity=int(args.edge_capacity),
        initial_active_edges=6,
        minimum_active_edges=2,
        edge_selection_mode="quality_coverage" if exchangeable else "all",
        selection_coverage_rate=0.85,
        module_assignment_normalizer=variant["normalizer"],
        environment_assignment_normalizer=variant["normalizer"],
        query_assignment_normalizer=variant["normalizer"],
        environment_locality_mode=variant["locality"],
        environment_locality_strength=1.0,
        minimum_region_scale=0.05,
        hidden_dim=int(args.hidden_dim),
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=int(args.hidden_dim),
        pairwise_kernel_num_layers=3,
        mechanism_state_mode=variant["mechanism_state_mode"],
        field_assembly_mode=variant["field_assembly_mode"],
        routing_execution=variant["execution"],
        query_module_limit=4,
        query_edge_limit=3,
    )


def make_batch(args: argparse.Namespace, device: torch.device) -> BatchData:
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    present = torch.zeros(1, int(args.modules))
    present[:, : int(args.active_modules)] = 1.0
    return BatchData(
        module_centers=(
            torch.rand(1, int(args.modules), 2, generator=generator) * torch.tensor([12.0, 6.0])
        ).to(device),
        module_present=present.to(device),
        module_features=torch.randn(1, int(args.modules), 6, generator=generator).to(device),
        global_context=torch.randn(1, 6, generator=generator).to(device),
        query_xy=(
            torch.rand(1, int(args.queries), 2, generator=generator) * torch.tensor([12.0, 6.0])
        ).to(device),
        query_time=None,
        target_field=None,
        case_name="forward-upgrade-ladder",
        metadata={},
    )


def scalar(output: dict[str, Any], key: str) -> float | None:
    value = output.get(key)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def summarize(model: HONFNeuralField, output: dict[str, Any]) -> dict[str, Any]:
    additive = "pred_field_background" in output and "pred_field_by_edge" in output
    if additive:
        reconstructed = output["pred_field_background"] + output["pred_field_by_edge"].sum(dim=2)
        closure = float((output["pred_field"] - reconstructed).abs().max().detach().cpu())
    else:
        closure = None
    return {
        "finite": bool(torch.isfinite(output["pred_field"]).all()),
        "field_shape": list(output["pred_field"].shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "candidate_edge_count": scalar(output, "candidate_edge_count"),
        "active_edge_count": scalar(output, "active_edge_count"),
        "module_nonzero_fraction": scalar(output, "module_assignment_nonzero_fraction"),
        "environment_nonzero_fraction": scalar(output, "environment_assignment_nonzero_fraction"),
        "query_nonzero_fraction": scalar(output, "query_assignment_nonzero_fraction"),
        "mean_query_nonzero_edges": scalar(output, "mean_query_nonzero_edges"),
        "pairwise_evaluated_pair_count": scalar(output, "pairwise_evaluated_pair_count"),
        "edge_head_evaluated_route_count": scalar(output, "edge_head_evaluated_route_count"),
        "additive_closure_max_abs": closure,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    batch = make_batch(args, device)
    results: dict[str, Any] = {}
    dense_sparse_state = None
    dense_sparse_field = None
    for index, (name, variant) in enumerate(VARIANTS.items()):
        torch.manual_seed(int(args.seed) + index)
        model = HONFNeuralField(make_config(args, variant)).to(device).eval()
        with torch.inference_mode():
            output = model(batch, return_edge_fields=variant["field_assembly_mode"] == "edge_additive")
        if name == "D":
            dense_sparse_state = copy.deepcopy(model.state_dict())
            dense_sparse_field = output["pred_field"].detach()
        elif name == "E" and dense_sparse_state is not None:
            model.load_state_dict(dense_sparse_state, strict=True)
            with torch.inference_mode():
                output = model(batch, return_edge_fields=True)
        results[name] = {"modes": variant, **summarize(model, output)}
        if name == "E" and dense_sparse_field is not None:
            difference = output["pred_field"] - dense_sparse_field
            results[name]["approximation_max_abs_vs_D"] = float(difference.abs().max().cpu())
            results[name]["approximation_relative_l2_vs_D"] = float(
                (difference.norm() / dense_sparse_field.norm().clamp_min(1.0e-12)).cpu()
            )
    payload = {
        "scope": "one_batch_correctness_only",
        "device": str(device),
        "training_steps": 0,
        "modules": int(args.modules),
        "active_modules": int(args.active_modules),
        "queries": int(args.queries),
        "edge_capacity": int(args.edge_capacity),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
