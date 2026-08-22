#!/usr/bin/env python3
"""Dry-run the Stage-6 profile against the frozen Run-1302 interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer  # noqa: E402
from channelthermal.workflows.evaluate_forward import (  # noqa: E402
    apply_frozen_forward_overrides,
    load_model,
    make_batch,
)
from honf_runtime.config_loader import load_config_bundle  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if torch.is_tensor(value) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--case-id", default="0273")
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "stage6_profile_dry_run.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = torch.device(args.device)
    profile_source = (
        "project://src/config_core/forward/experiments/"
        "stage6_fixed_role_consistent_additive.json"
    )
    bundle = load_config_bundle(
        "project://src/config_core/forward/adaptive_sparse_additive.json",
        experiment_overlay=profile_source,
    )
    profile = bundle.effective["model"]["core_honf"]
    expected = {
        "organizer_mode": "fixed_projection",
        "num_hyperedges": 6,
        "edge_selection_mode": "all",
        "module_assignment_normalizer": "softmax",
        "environment_assignment_normalizer": "softmax",
        "query_assignment_normalizer": "softmax",
        "query_locality_mode": "none",
        "query_locality_strength": None,
        "mechanism_state_mode": "descriptor_first",
        "mechanism_latent_residual_scale": 0.35,
        "field_assembly_mode": "edge_additive",
        "additive_background_mode": "dense_query_attention",
        "routing_execution": "dense",
    }
    mismatches = {
        key: {"expected": value, "actual": profile.get(key)}
        for key, value in expected.items()
        if profile.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Stage-6 profile mismatch: {mismatches}")

    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()
    state_keys_before = tuple(model.state_dict())

    dataset_cfg = checkpoint["train_config"]["dataset"]
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint.get("global_normalization_stats", {}).items()
    }
    dataset = GlobalChannelThermalDataset(
        dataset_cfg["packed_h5_path"],
        split="test",
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
        raise KeyError(f"case_id={args.case_id!r} is absent from the test split")
    sample = dataset[matches[0]]
    query_xy = np.stack(
        (np.asarray(sample["x_grid"]).reshape(-1), np.asarray(sample["y_grid"]).reshape(-1)),
        axis=-1,
    ).astype(np.float32)[: int(args.queries)]
    batch = make_batch(sample, query_xy, device)
    with torch.inference_mode():
        reference_output = model(
            batch["structure"],
            batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
        )
    frozen_overrides = apply_frozen_forward_overrides(
        model,
        mechanism_latent_residual_scale=float(profile["mechanism_latent_residual_scale"]),
        query_locality_mode=str(profile["query_locality_mode"]),
        query_locality_strength=profile["query_locality_strength"],
    )
    with torch.inference_mode():
        output = model(
            batch["structure"],
            batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
            return_routing_maps=True,
            return_edge_fields=True,
            return_prepared_state=True,
        )
    background = output["pred_field_background"]
    edge_fields = output["pred_field_by_edge"]
    closure = output["pred_field"] - background - edge_fields.sum(dim=2)
    organizer = output["organizer_aux"]
    prepared = output["prepared_state"]
    required_organizer = [
        "hyper_state",
        "mechanism_descriptor_features",
        "A_mh",
        "A_eh",
        "hyper_source_coords",
        "hyper_region_coords",
        "hyper_source_scale",
        "hyper_region_scale",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_module_purity",
        "hyper_env_purity",
    ]
    routing_shapes = {
        key: _shape(value)
        for key, value in output["routing_aux"].items()
        if torch.is_tensor(value) and value.ndim > 0
    }
    inverse_organizer = prepared.organizer
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "case_id": str(sample["case_id"]),
        "device": str(device),
        "profile_source": profile_source,
        "profile_values": expected,
        "frozen_overrides": frozen_overrides,
        "field_names": list(model.config.channelthermal.field_names),
        "state_dict_key_count": len(state_keys_before),
        "state_dict_structure_unchanged": state_keys_before == tuple(model.state_dict()),
        "finite_prediction": bool(torch.isfinite(output["pred_field"]).all()),
        "checkpoint_s0_prediction_bitwise_equal": bool(
            torch.equal(reference_output["pred_field"], output["pred_field"])
        ),
        "checkpoint_s0_prediction_max_abs_difference": float(
            (reference_output["pred_field"] - output["pred_field"]).abs().max().cpu()
        ),
        "additive_closure_max_abs": float(closure.abs().max().cpu()),
        "output_shapes": {
            "pred_field": _shape(output["pred_field"]),
            "pred_field_background": _shape(background),
            "pred_field_by_edge": _shape(edge_fields),
        },
        "organizer_aux_shapes": {key: _shape(organizer.get(key)) for key in required_organizer},
        "prepared_organizer_shapes": {
            key: _shape(inverse_organizer.get(key)) for key in required_organizer
        },
        "routing_shapes": routing_shapes,
        "prepared_state": {
            "type": type(prepared).__name__,
            "global_token": _shape(prepared.global_token),
            "organizer_keys_present": {
                key: key in inverse_organizer for key in required_organizer
            },
        },
        "inverse_readiness": {
            "fixed_ordered_hyperedges": int(inverse_organizer["hyper_state"].shape[1]) == 6,
            "all_required_organizer_tensors_present": all(
                key in inverse_organizer for key in required_organizer
            ),
            "edge_resolved_field_available": "pred_field_by_edge" in output,
            "prepared_state_available": "prepared_state" in output,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    dataset.close()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
