#!/usr/bin/env python3
"""Run the bounded Run2111 WindFarm packet/direct-pair study.

All targets are gathered from the native mmap view. The source is the exact
Run2110 W-full u1500 checkpoint, and each arm is rebuilt independently from
those bytes. This entrypoint deliberately exposes only a one-update smoke or a
caller-specified bounded update count; it has no implicit continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "HONF_Proj"
CASE_SRC = PROJECT / "Case_WindFarm" / "src"
CORE_SRC = PROJECT / "src"
for _path in (str(CORE_SRC), str(CASE_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    DirectPairAccess,
    InteractionPermissionKey,
    MechanismPlan,
)
from honf_forward_core.interface_fields.budgeted_frontier import (
    BudgetConditionedDirectPairScorer,
    canonical_pair_catalog,
    enumerate_frontier_cuts,
    normalized_frontier_work_targets,
    project_direct_pair_budget_by_fraction,
    project_unique_pair_budget,
    select_frontier_by_predictions,
)
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer
from honf_forward_core.interface_fields.native_direct_pair import (
    hard_value_soft_direct_forward,
)
from honf_forward_core.interface_fields.native_joint_shadow import (
    _detach_inputs,
    hard_value_soft_organizer_forward,
)
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.paths import resolve_path

from windfarm.data import WindFarmNativeView
from windfarm.normalization import VelocityNormalizer
from windfarm.splits import make_group_split
from windfarm.workflows.joint_forward import (
    ROLE_NAMES,
    NativeRoleCatalogueCache,
    _batch_from_sample,
    _build_organizer,
    _enable_physical_trainable_scope,
    _file_sha256,
    _model_state_sha256,
    _native_rng,
    _new_model_from_source,
    _physical_from_standardized,
    _role_mse_physical,
    sample_native_role_queries,
)
from windfarm.workflows.train_forward import _compact_metadata

EXPECTED_SOURCE_SHA256 = "f69eb6924b28642a3d7671a9f76abdc3cc73d40094f9c73f6898e66022365131"
DEFAULT_CONFIG = PROJECT / "Case_WindFarm" / "configs" / "active_packet_organizer_reuse.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rows_sha256(rows: Sequence[int]) -> str:
    encoded = json.dumps([int(value) for value in rows], separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("run", {}).get("id") != "2111":
        raise ValueError("WindFarm active packet runner is locked to new Run_ID 2111")
    if config.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("Run2111 source checkpoint SHA256 does not match the audited W-full u1500")
    return config


def _native_inputs(config: Mapping[str, Any]) -> tuple[WindFarmNativeView, Any, list[int], dict[str, Any]]:
    dataset = config["dataset"]
    locations_path = resolve_path(str(dataset["locations"]))
    locations = json.loads(locations_path.read_text(encoding="utf-8"))
    try:
        compact_path = resolve_path(str(locations[dataset["compact_dataset_id"]]))
        volume_path = resolve_path(str(locations[dataset["volume_dataset_id"]]))
    except KeyError as exc:
        raise KeyError(f"dataset locations omit requested WindFarm dataset: {exc}") from exc
    view = WindFarmNativeView(
        volume_path,
        compact_metadata=_compact_metadata(compact_path),
        include_receiver_anchors=True,
    )
    if dataset.get("memory_map_volume") is not True:
        raise ValueError("Run2111 requires the WindFarm volume field to remain mmap-backed")
    if dataset.get("require_complete_volume_runs"):
        complete = np.asarray(view.volume.array("completed"), dtype=np.uint8)
        if complete.shape != (view.n_cases,) or not bool(np.all(complete == 1)):
            raise ValueError("native WindFarm volume runs are incomplete")
    canonical = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    split_path = resolve_path(str(dataset["split_indices"]))
    with np.load(split_path, allow_pickle=False) as stored:
        required = {"train", "validation", "test"}
        if not required.issubset(stored.files):
            raise ValueError(f"archived WindFarm split lacks {sorted(required - set(stored.files))}")
        for name in sorted(required):
            if not np.array_equal(np.asarray(stored[name], dtype=np.int64), getattr(canonical, name)):
                raise ValueError(f"archived seed-42 layout split differs in {name}")
    layout_indices = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    excluded_layouts = set(map(int, dataset["excluded_organizer_development_layout_indices"]))
    train_rows = [int(row) for row in canonical.train if int(layout_indices[int(row)]) not in excluded_layouts]
    if len(canonical.train) != 420 or len(train_rows) != 408:
        raise ValueError(
            f"WindFarm split/exclusion audit changed: expected 420/408 rows, "
            f"got {len(canonical.train)}/{len(train_rows)}"
        )
    if len({int(layout_indices[row]) for row in train_rows}) != 136:
        raise ValueError("post-exclusion WindFarm training set must contain 136 whole layouts")
    split_record = {
        "seed": 42,
        "source_train_rows": len(canonical.train),
        "validation_rows": len(canonical.validation),
        "test_rows": len(canonical.test),
        "excluded_organizer_development_layouts": sorted(excluded_layouts),
        "excluded_training_rows": int(len(canonical.train) - len(train_rows)),
        "student_train_rows": len(train_rows),
        "student_train_rows_sha256": _rows_sha256(train_rows),
        "student_train_layout_count": len({int(layout_indices[row]) for row in train_rows}),
        "validation_rows_sha256": _rows_sha256(canonical.validation),
        "test_rows_opened": False,
        "volume_memory_map": bool(isinstance(view.volume.array("run_cell_offsets"), np.memmap)),
        "volume_shape": list(view.volume.array("U").shape),
        "volume_dtype": str(view.volume.array("U").dtype),
    }
    return view, canonical, train_rows, split_record


def _load_source(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any], VelocityNormalizer, str]:
    checkpoint = resolve_path(str(config["source_checkpoint"]))
    digest = _file_sha256(checkpoint)
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"W-full u1500 SHA256 mismatch: expected {EXPECTED_SOURCE_SHA256}, got {digest}")
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Run2110 selected endpoint is not a checkpoint mapping")
    run = payload.get("train_config", {}).get("run", {})
    if (
        payload.get("case_id") != "WindFarm"
        or payload.get("model_family") != "honf_forward"
        or payload.get("workflow") != "forward"
        or payload.get("arm") != "w_full"
        or int(payload.get("update_count", -1)) != 1500
        or str(run.get("id", "")) != "2110"
        or str(payload.get("dataset_id", "")) != "wind_farm_volume_v1"
        or list(payload.get("channel_order", [])) != ["Ux", "Uy", "Uz"]
        or int(payload.get("field_dim", 0)) != 3
    ):
        raise ValueError("source checkpoint identity is not the exact audited Run2110 W-full u1500")
    normalization = payload.get("normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("Run2110 source lacks its training-owned normalization")
    return checkpoint, dict(payload), VelocityNormalizer.from_dict(dict(normalization)), digest


def _seeded_cut(tree: Any, rng: np.random.Generator, preference: int | None = None) -> tuple[int, ...]:
    cuts = enumerate_frontier_cuts(tree, max_depth=3)
    if not cuts:
        raise ValueError("native receiver tree exposed no complete frontier cuts")
    if preference is None:
        group = int(rng.integers(0, 3))
    else:
        group = int(preference) % 3
    sizes = np.asarray([len(cut) for cut in cuts], dtype=np.int64)
    if group == 0:
        candidates = np.flatnonzero(sizes == sizes.min())
    elif group == 2:
        candidates = np.flatnonzero(sizes == sizes.max())
    else:
        middle = np.median(sizes)
        candidates = np.flatnonzero(sizes == sizes[np.abs(sizes - middle).argmin()])
    selected = int(candidates[int(rng.integers(0, len(candidates)))])
    return tuple(cuts[selected])


def _budgets(
    *,
    selected_route: str,
    fraction: float,
) -> tuple[dict[str, float], dict[str, float]]:
    route_fractions = {"QE": float(fraction), selected_route: float(fraction)}
    # All remaining native mechanisms are intentional full-access bypasses.
    # Their absence from this mapping is recorded by MechanismPlan as such.
    return route_fractions, route_fractions.copy()


def _role_objective(
    prediction_standardized: torch.Tensor,
    target_standardized: torch.Tensor,
    role_slices: Mapping[str, slice],
    normalizer: VelocityNormalizer,
    role_scales: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    prediction_mps = _physical_from_standardized(prediction_standardized, normalizer)
    target_mps = _physical_from_standardized(target_standardized, normalizer)
    role_mse = _role_mse_physical(prediction_mps, target_mps, role_slices)
    terms = [role_mse[role] / float(role_scales[role]) ** 2 for role in ROLE_NAMES]
    total = torch.stack(terms).mean()
    mse = {role: float(role_mse[role].detach().cpu()) for role in ROLE_NAMES}
    rmse = {role: float(np.sqrt(mse[role])) for role in ROLE_NAMES}
    return total, mse, rmse


def _full_access_prediction(model: Any, batch: Any) -> tuple[Any, torch.Tensor]:
    """Use the existing Dense full-access path for the adaptation phase."""

    core = model.core
    core.backend.set_cover_mode("full_access")
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    prediction = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=batch.query_features,
        receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
    )["pred_field"]
    core.backend.set_cover_mode("external")
    return encoded, prediction


def _prediction_with_plan(
    model: Any,
    encoded: Any,
    batch: Any,
    plan: MechanismPlan,
    *,
    return_interaction_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, Any]]:
    core = model.core
    core.backend.set_cover_mode("external")
    prepared = core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(plan,))
    output = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=batch.query_features,
        receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
        return_interaction_aux=return_interaction_aux,
    )
    prediction = output["pred_field"]
    if return_interaction_aux:
        return prediction, output.get("_interaction_aux", {})
    return prediction


def _standardized_delta_to_mps(delta: torch.Tensor, normalizer: VelocityNormalizer) -> torch.Tensor:
    """Convert a standardized difference to m/s without adding its mean."""

    scale = delta.new_tensor(normalizer.safe_std * float(normalizer.u_ref_mps))
    result = delta * scale
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("standardized-difference conversion produced a non-finite m/s value")
    if bool((delta == 0).all()) and bool((result != 0).any()):
        raise AssertionError("zero standardized difference must map to zero m/s")
    return result


def _canonical_route_support(
    plan: MechanismPlan,
    encoded: Any,
    tree: Any,
    mechanism: str,
) -> dict[str, Any]:
    """Count the exact typed live support on a route's canonical panel."""

    catalog = canonical_pair_catalog(encoded, tree, mechanism)
    access = plan.access_for(
        mechanism,
        catalog.receiver_coordinates,
        int(catalog.source_validity.shape[0]),
        module_present=(encoded.module_present[0] if mechanism in {"MM", "EM", "QM"} else None),
    )
    support = (access > 0.0) & catalog.pair_validity
    full_pairs = int(catalog.pair_validity.sum().detach().cpu())
    selected_pairs = int(support.sum().detach().cpu())
    full_work = float(
        (catalog.pair_validity.to(torch.float64) * catalog.receiver_weights.to(torch.float64)[:, None])
        .sum()
        .detach()
        .cpu()
    )
    selected_work = float(
        (support.to(torch.float64) * catalog.receiver_weights.to(torch.float64)[:, None])
        .sum()
        .detach()
        .cpu()
    )
    raw_summary = plan.frontier_summary(mechanism, receivers=catalog.receiver_coordinates)
    return {
        "permission_status": plan.permission_status(mechanism),
        "eligible_unique_pairs": full_pairs,
        "actual_live_unique_pairs": selected_pairs,
        "canonical_full_work": full_work,
        "canonical_actual_work": selected_work,
        "canonical_work_fraction": selected_work / full_work if full_work > 0.0 else 0.0,
        "nonredundant_packet_count": int(raw_summary.nonredundant_packet_count),
        # This tree-summary count is retained as a raw path statistic; MM
        # diagonal entries are excluded from the exact pair totals above.
        "frontier_summary_raw_pair_count": int(raw_summary.unique_source_receiver_pairs),
        "sparse_success": bool(0 < selected_pairs < full_pairs) if full_pairs > 0 else None,
    }


def _query_qe_support(
    plan: MechanismPlan,
    encoded: Any,
    sample: Any,
    batch: Any,
) -> dict[str, Any]:
    """Measure exact QE support on a sampled live query panel."""

    receivers = batch.query_xy[0]
    source_validity = encoded.env_weights[0] > 0.0
    access = plan.access_for("QE", receivers, int(source_validity.shape[0]))
    support = (access > 0.0) & source_validity[None, :]
    receiver_weights = torch.zeros(receivers.shape[0], dtype=receivers.dtype, device=receivers.device)
    for role, role_slice in sample.role_slices.items():
        count = max(1, int(sample.role_sample_counts[role]))
        receiver_weights[role_slice] = float(sample.role_support_volume_m3[role]) / count
    full_pairs = int(source_validity.sum().detach().cpu()) * int(receivers.shape[0])
    selected_pairs = int(support.sum().detach().cpu())
    full_work = float((receiver_weights.to(torch.float64).sum() * source_validity.sum()).detach().cpu())
    selected_work = float(
        (support.to(torch.float64) * receiver_weights.to(torch.float64)[:, None]).sum().detach().cpu()
    )
    return {
        "receiver_rows": int(receivers.shape[0]),
        "source_rows": int(source_validity.sum().detach().cpu()),
        "actual_full_pairs": full_pairs,
        "actual_selected_pairs": selected_pairs,
        "actual_work": selected_work,
        "full_work": full_work,
        "work_fraction": selected_work / full_work if full_work > 0.0 else 0.0,
        "actual_receiver_panel_sparse_success": bool(0 < selected_pairs < full_pairs)
        if full_pairs > 0
        else None,
    }


def _summarize_executor_rows(auxiliary: Mapping[str, Any]) -> dict[str, Any]:
    """Persist native executor-row counters without storing bulky tensors."""

    result: dict[str, Any] = {}
    for key, value in auxiliary.items():
        name = str(key)
        if not torch.is_tensor(value):
            continue
        tensor = value.detach()
        item: dict[str, Any] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if tensor.numel() <= 32:
            item["values"] = tensor.cpu().tolist()
        elif tensor.is_floating_point():
            item.update({
                "sum": float(tensor.sum().cpu()),
                "nonzero_count": int((tensor != 0).sum().cpu()),
                "minimum": float(tensor.min().cpu()),
                "maximum": float(tensor.max().cpu()),
            })
        else:
            item.update({"sum": int(tensor.sum().cpu()), "nonzero_count": int((tensor != 0).sum().cpu())})
        result[name] = item
    return result


def _secondary_route_pilot(
    *,
    config: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    view: WindFarmNativeView,
    train_rows: Sequence[int],
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Choose MM or EM from native training-only omission and mask-gradient evidence."""

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("the WindFarm route pilot requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("authorized physical GPU 0 is unavailable for the route pilot")
    device = torch.device("cuda:0")
    pilot_config = config["forward"]["stage_a"]["secondary_route_pilot"]
    pilot_count = int(pilot_config["training_layout_count"])
    pilot_counts = {str(key): int(value) for key, value in pilot_config["role_query_counts"].items()}
    if set(pilot_counts) != set(ROLE_NAMES) or any(value <= 0 for value in pilot_counts.values()):
        raise ValueError("route pilot must sample every protected native role")
    row_by_layout: dict[int, list[int]] = {}
    layout_values = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    for row in train_rows:
        row_by_layout.setdefault(int(layout_values[int(row)]), []).append(int(row))
    eligible_layouts = sorted(row_by_layout)
    if pilot_count > len(eligible_layouts):
        raise ValueError("secondary route pilot requests more layouts than post-exclusion training contains")
    order_rng = _native_rng(seed, 0, 0, 720)
    chosen_layouts = sorted(map(int, order_rng.choice(np.asarray(eligible_layouts), size=pilot_count, replace=False)))
    pilot_rows = []
    for layout in chosen_layouts:
        candidates = sorted(row_by_layout[layout])
        row_rng = _native_rng(seed, 0, layout, 721)
        pilot_rows.append(candidates[int(row_rng.integers(0, len(candidates)))])

    cache = NativeRoleCatalogueCache()
    first_row = pilot_rows[0]
    first_case = view.run(first_row)
    first_sample = sample_native_role_queries(
        first_case,
        _native_rng(seed, 0, first_row, 722),
        pilot_counts,
        catalogue_cache=cache,
    )
    first_batch = _batch_from_sample(first_case, first_sample, normalizer, device)
    model = _new_model_from_source(source_payload, normalizer, first_batch, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scales = config["forward"]["stage_a"]["role_loss_scales_mps"]
    accum: dict[str, dict[str, Any]] = {
        route: {
            "normalized_omission_gain": [],
            "restoration_loss_gradient_rms": [],
            "full_loss": [],
            "omitted_loss": [],
            "role_mse_delta_mps2": {role: [] for role in ROLE_NAMES},
            "eligible_mask_gradient_elements": [],
        }
        for route in ("MM", "EM")
    }
    for row in pilot_rows:
        case = view.run(row)
        sample = sample_native_role_queries(
            case,
            _native_rng(seed, 0, row, 722),
            pilot_counts,
            catalogue_cache=cache,
        )
        batch = _batch_from_sample(case, sample, normalizer, device)
        with torch.no_grad():
            encoded = model.core.encode_case(batch)
            trees = model.core.backend.build_case_trees(encoded)
            full_plan = MechanismPlan.full_access(trees[0], encoded.module_present[0], int(encoded.env_coords.shape[1]))
            full_prediction = _prediction_with_plan(model, encoded, batch, full_plan)
            full_loss, full_mse, _ = _role_objective(
                full_prediction, batch.target_field, sample.role_slices, normalizer, scales
            )
        for route in ("MM", "EM"):
            zeros = torch.zeros_like(full_plan.permission_matrix(route))
            omitted_plan = full_plan.with_permission(route, zeros)
            with torch.no_grad():
                omitted_prediction = _prediction_with_plan(model, encoded, batch, omitted_plan)
                omitted_loss, omitted_mse, _ = _role_objective(
                    omitted_prediction, batch.target_field, sample.role_slices, normalizer, scales
                )
            grad_mask = full_plan.permission_matrix(route).detach().clone().requires_grad_(True)
            gradient_plan = full_plan.with_permission(route, grad_mask)
            route_key = InteractionPermissionKey(route)
            attached_mask = gradient_plan.permissions[route_key]
            gradient_prediction = _prediction_with_plan(model, encoded, batch, gradient_plan)
            gradient_loss, _gradient_mse, _gradient_rmse = _role_objective(
                gradient_prediction, batch.target_field, sample.role_slices, normalizer, scales
            )
            mask_gradient = torch.autograd.grad(gradient_loss, attached_mask, retain_graph=False)[0]
            catalog = canonical_pair_catalog(encoded, trees[0], route)
            reached_nodes = (trees[0].access(catalog.receiver_coordinates, full_plan.split_gates) > 0).any(dim=0)
            eligible_node_sources = reached_nodes[:, None] & catalog.source_validity[None, :]
            selected_gradient = mask_gradient[eligible_node_sources]
            gradient_rms = (
                float(selected_gradient.square().mean().sqrt().detach().cpu()) if selected_gradient.numel() else 0.0
            )
            normalized_gain = max(0.0, float((omitted_loss - full_loss).detach().cpu()))
            accum[route]["normalized_omission_gain"].append(normalized_gain)
            accum[route]["restoration_loss_gradient_rms"].append(gradient_rms)
            accum[route]["full_loss"].append(float(full_loss.detach().cpu()))
            accum[route]["omitted_loss"].append(float(omitted_loss.detach().cpu()))
            accum[route]["eligible_mask_gradient_elements"].append(int(selected_gradient.numel()))
            for role in ROLE_NAMES:
                accum[route]["role_mse_delta_mps2"][role].append(omitted_mse[role] - full_mse[role])

    metrics: dict[str, Any] = {}
    for route in ("MM", "EM"):
        source = accum[route]
        metrics[route] = {
            "mean_training_only_normalized_omission_loss_gain": float(np.mean(source["normalized_omission_gain"])),
            "mean_training_only_restoration_loss_gradient_rms": float(np.mean(source["restoration_loss_gradient_rms"])),
            "mean_full_role_objective": float(np.mean(source["full_loss"])),
            "mean_omitted_role_objective": float(np.mean(source["omitted_loss"])),
            "mean_role_omission_mse_delta_mps2": {
                role: float(np.mean(values)) for role, values in source["role_mse_delta_mps2"].items()
            },
            "mean_eligible_access_gradient_elements": float(np.mean(source["eligible_mask_gradient_elements"])),
        }
    selected = max(
        ("MM", "EM"),
        key=lambda route: (
            metrics[route]["mean_training_only_normalized_omission_loss_gain"],
            metrics[route]["mean_training_only_restoration_loss_gradient_rms"],
            route == "MM",
        ),
    )
    result = {
        "method": pilot_config["selection"],
        "training_only": True,
        "held_development_or_validation_rows_opened": False,
        "new_cfd_solves": 0,
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "train_rows_sha256": _rows_sha256(train_rows),
        "seed": seed,
        "pilot_rows": pilot_rows,
        "pilot_layouts": chosen_layouts,
        "role_query_counts": pilot_counts,
        "selected_route": selected,
        "route_metrics": metrics,
        "tie_policy": "higher omission loss gain, then higher restored-access loss-gradient RMS, then MM",
    }
    _write_json(run_dir / "secondary_route_pilot.json", result)
    return result


def _pad_features(values: torch.Tensor, width: int) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] > width:
        raise ValueError("direct-pair features must be a matrix no wider than the configured width")
    if values.shape[1] == width:
        return values
    return torch.nn.functional.pad(values, (0, width - int(values.shape[1])))


def _direct_feature_tables(encoded: Any) -> dict[str, Any]:
    """Build target-free, typed source and receiver feature rows for P."""

    if int(encoded.module_centers.shape[0]) != 1:
        raise ValueError("WindFarm direct scorer currently expects one case at a time")
    module_state = encoded.module_tokens[0]
    environment_state = encoded.env_tokens[0]
    global_state = encoded.global_token[0]
    module_features = torch.cat(
        (encoded.module_features[0], encoded.module_features[0].new_zeros((module_state.shape[0], 1))),
        dim=-1,
    )
    if encoded.env_features is None:
        raise ValueError("WindFarm direct control requires native environment features")
    normalized_mass = encoded.env_weights[0] / encoded.env_weights[0].mean().clamp_min(
        torch.finfo(encoded.env_weights.dtype).tiny
    )
    environment_features = torch.cat(
        (
            encoded.env_features[0],
            torch.log(normalized_mass.clamp_min(torch.finfo(normalized_mass.dtype).tiny))[:, None],
        ),
        dim=-1,
    )
    max_feature = max(int(module_features.shape[-1]), int(environment_features.shape[-1]))
    state_width = int(module_state.shape[-1])
    type_width = 3
    width = 2 * state_width + max_feature + type_width
    module_type = module_state.new_tensor((1.0, 0.0, 0.0)).reshape(1, -1).expand(module_state.shape[0], -1)
    environment_type = module_state.new_tensor((0.0, 1.0, 0.0)).reshape(1, -1).expand(environment_state.shape[0], -1)

    def pack(state: torch.Tensor, features: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        padded = _pad_features(features, max_feature)
        context = global_state.reshape(1, -1).expand(state.shape[0], -1)
        return torch.cat((state, context, padded, kind), dim=-1)

    module_table = pack(module_state, module_features, module_type)
    environment_table = pack(environment_state, environment_features, environment_type)
    return {
        "width": width,
        "module": module_table,
        "environment": environment_table,
        "module_coordinates": encoded.module_centers[0],
        "environment_coordinates": encoded.env_coords[0],
        "module_valid": encoded.module_present[0] > 0.5,
        "environment_valid": encoded.env_weights[0] > 0.0,
        "environment_state": environment_state,
        "environment_features": environment_features,
        "global_state": global_state,
        "max_feature": max_feature,
        "state_width": state_width,
        "coordinate_scale": encoded.coordinate_scale.reshape(-1, int(encoded.module_centers.shape[-1]))[-1],
    }


def _receiver_features_for_route(
    mechanism: str,
    coordinates: torch.Tensor,
    tables: Mapping[str, Any],
) -> torch.Tensor:
    if mechanism in {"MM", "ME"}:
        return tables["module"]
    if mechanism == "EM":
        return tables["environment"]
    return _query_direct_features_from_tables(coordinates, tables)


def _query_direct_features_from_tables(coordinates: torch.Tensor, tables: Mapping[str, Any]) -> torch.Tensor:
    environment_coordinates = tables["environment_coordinates"]
    coordinate_scale = tables["coordinate_scale"].to(device=coordinates.device, dtype=coordinates.dtype)
    distance = torch.cdist(
        coordinates[None] / coordinate_scale[None, None, :],
        environment_coordinates[None] / coordinate_scale[None, None, :],
    )[0]
    nearest = distance.argmin(dim=1)
    state = tables["environment_state"].index_select(0, nearest)
    features = tables["environment_features"].index_select(0, nearest)
    query_type = state.new_tensor((0.0, 0.0, 1.0)).reshape(1, -1).expand(state.shape[0], -1)
    context = tables["global_state"].reshape(1, -1).expand(state.shape[0], -1)
    return torch.cat((state, context, _pad_features(features, int(tables["max_feature"])), query_type), dim=-1)


def _direct_source_for_route(
    mechanism: str, tables: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mechanism in {"MM", "EM", "QM"}:
        return tables["module"], tables["module_coordinates"], tables["module_valid"]
    return tables["environment"], tables["environment_coordinates"], tables["environment_valid"]


def _direct_projected_access(
    *,
    scorer: BudgetConditionedDirectPairScorer,
    encoded: Any,
    tree: Any,
    mechanism: str,
    fraction: float,
    tables: Mapping[str, Any],
    receiver_coordinates: torch.Tensor,
    receiver_validity: torch.Tensor | None = None,
) -> tuple[DirectPairAccess, DirectPairAccess, Any]:
    """Project a canonical work panel, then score exact current receivers."""

    catalog = canonical_pair_catalog(encoded, tree, mechanism)
    source_features, source_coordinates, source_valid = _direct_source_for_route(mechanism, tables)
    catalog_receivers = catalog.receiver_coordinates
    canonical_scores = scorer(
        _receiver_features_for_route(mechanism, catalog_receivers, tables),
        source_features,
        catalog_receivers,
        source_coordinates,
        mechanism=mechanism,
        budget_fraction=fraction,
    )
    projection = project_direct_pair_budget_by_fraction(
        canonical_scores,
        budget_fraction=fraction,
        eligible_pairs=catalog.pair_validity,
        receiver_weights=catalog.receiver_weights,
    )
    actual_scores = scorer(
        _receiver_features_for_route(mechanism, receiver_coordinates, tables),
        source_features,
        receiver_coordinates,
        source_coordinates,
        mechanism=mechanism,
        budget_fraction=fraction,
    )
    threshold = projection.threshold.to(device=actual_scores.device, dtype=actual_scores.dtype)
    current_receiver_validity = (
        torch.ones(receiver_coordinates.shape[0], device=receiver_coordinates.device, dtype=torch.bool)
        if receiver_validity is None
        else receiver_validity.to(device=receiver_coordinates.device, dtype=torch.bool)
    )
    source_mask = source_valid.to(device=actual_scores.device, dtype=actual_scores.dtype)
    if bool(torch.isneginf(threshold)):
        hard_weights = (current_receiver_validity.to(actual_scores.dtype)[:, None] * source_mask[None, :]).to(
            actual_scores.dtype
        )
        soft_weights = hard_weights.clone()
    else:
        if bool(torch.isposinf(threshold)):
            eligible_scores = canonical_scores.detach()[catalog.pair_validity]
            threshold = (
                eligible_scores.max() + 1.0 if eligible_scores.numel() else canonical_scores.detach().new_tensor(1.0e6)
            )
        threshold = threshold.to(device=actual_scores.device, dtype=actual_scores.dtype)
        hard_weights = ((actual_scores >= threshold) & (source_mask[None, :] > 0.5)).to(actual_scores.dtype)
        soft_weights = torch.sigmoid(actual_scores - threshold) * source_mask[None, :]
    if bool(torch.isposinf(projection.threshold)):
        eligible_scores = canonical_scores.detach()[catalog.pair_validity]
        if eligible_scores.numel() == 0:
            soft_weights = actual_scores.new_zeros(actual_scores.shape)
    if mechanism == "MM" and receiver_coordinates.shape[0] == source_coordinates.shape[0]:
        diagonal = torch.eye(receiver_coordinates.shape[0], device=actual_scores.device, dtype=torch.bool)
        hard_weights = hard_weights.masked_fill(diagonal, 0.0)
        soft_weights = soft_weights.masked_fill(diagonal, 0.0)
    return (
        DirectPairAccess(receiver_coordinates.detach(), hard_weights.detach(), current_receiver_validity),
        DirectPairAccess(receiver_coordinates.detach(), soft_weights, current_receiver_validity),
        projection,
    )


def _direct_plan_pair(
    *,
    model: Any,
    encoded: Any,
    batch: Any,
    scorer: BudgetConditionedDirectPairScorer,
    secondary_route: str,
    fraction: float,
) -> tuple[MechanismPlan, MechanismPlan, dict[str, Any], dict[str, Any]]:
    """Build separate hard and soft P plans from detached current inputs."""

    scoring_encoded = _detach_inputs(encoded)
    trees = model.core.backend.build_case_trees(scoring_encoded)
    if len(trees) != 1:
        raise ValueError("WindFarm direct-pair plan builder expects one case")
    tree = trees[0]
    tables = _direct_feature_tables(scoring_encoded)
    base = MechanismPlan(
        tree,
        tree.universe.coordinates.new_zeros((len(tree.nodes),)),
        scoring_encoded.module_present[0],
        int(scoring_encoded.env_coords.shape[1]),
    )
    hard_plan = base
    soft_plan = base
    projections: dict[str, Any] = {}
    actual_access: dict[str, Any] = {}
    for mechanism in ("QE", secondary_route):
        catalog = canonical_pair_catalog(scoring_encoded, tree, mechanism)
        if mechanism == "QE":
            receivers = batch.query_xy[0].detach()
            receiver_validity = torch.ones(receivers.shape[0], dtype=torch.bool, device=receivers.device)
        else:
            receivers = catalog.receiver_coordinates
            receiver_validity = catalog.receiver_validity
        hard_access, soft_access, projection = _direct_projected_access(
            scorer=scorer,
            encoded=scoring_encoded,
            tree=tree,
            mechanism=mechanism,
            fraction=fraction,
            tables=tables,
            receiver_coordinates=receivers,
            receiver_validity=receiver_validity,
        )
        hard_plan = hard_plan.with_direct_pair_access(
            mechanism,
            hard_access.receiver_coordinates,
            hard_access.weights,
            receiver_validity=hard_access.receiver_validity,
        )
        soft_plan = soft_plan.with_direct_pair_access(
            mechanism,
            soft_access.receiver_coordinates,
            soft_access.weights,
            receiver_validity=soft_access.receiver_validity,
        )
        projections[mechanism] = projection
        if mechanism == "QE":
            actual_full_pairs = int(receiver_validity.sum().detach().cpu()) * int(
                catalog.source_validity.sum().detach().cpu()
            )
        else:
            actual_full_pairs = int(catalog.pair_validity.sum().detach().cpu())
        actual_access[mechanism] = {
            "receiver_rows": int(hard_access.weights.shape[0]),
            "source_rows": int(hard_access.weights.shape[1]),
            "actual_selected_pairs": int((hard_access.weights > 0).sum().detach().cpu()),
            "actual_full_pairs": actual_full_pairs,
            "canonical_eligible_pairs": int(catalog.pair_validity.sum().detach().cpu()),
            "canonical_work_fraction": (
                float(projection.achieved_work / projection.full_access_work)
                if projection.full_access_work > 0.0
                else 0.0
            ),
            "canonical_projected_pairs": int(projection.selected_unique_pairs),
            "canonical_sparse_success": (
                bool(
                    projection.sparse_success
                    and 0 < int(projection.selected_unique_pairs) < int(projection.full_unique_pairs)
                )
                if projection.full_unique_pairs > 0
                else None
            ),
            "actual_receiver_panel_sparse_success": (
                bool(0 < int((hard_access.weights > 0).sum().detach().cpu()) < actual_full_pairs)
                if actual_full_pairs > 0
                else None
            ),
            "numerically_empty_route": projection.full_unique_pairs == 0,
        }
    return hard_plan, soft_plan, projections, actual_access


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_resume_checkpoint(
    path: Path | None,
    *,
    arm: str,
    source_sha256: str,
    train_rows_sha256: str,
    secondary_route: str,
    seed: int,
) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint_path = path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Run2111 resume checkpoint must be a mapping")
    expected = {
        "run_id": "2111",
        "arm": arm,
        "source_sha256": source_sha256,
        "train_rows_sha256": train_rows_sha256,
        "secondary_route": secondary_route,
        "seed": int(seed),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"resume checkpoint {key}={payload.get(key)!r} does not match {value!r}")
    update = int(payload.get("update_count", -1))
    if update < 1 or not isinstance(payload.get("optimizer_state_dict"), Mapping):
        raise ValueError("resume checkpoint lacks a completed update or optimizer state")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise TypeError("resume checkpoint lacks the physical model state mapping")
    return dict(payload)


def _validate_resume_history(arm_dir: Path, checkpoint: Mapping[str, Any]) -> None:
    """Refuse resume when the append-only update ledger is not at this checkpoint."""

    ledger = arm_dir / "updates.jsonl"
    if not ledger.is_file():
        raise FileNotFoundError(f"resume arm has no update ledger: {ledger}")
    last: dict[str, Any] | None = None
    with ledger.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    expected_update = int(checkpoint["update_count"])
    if last is None or int(last.get("update", -1)) != expected_update:
        raise ValueError(
            "resume update ledger must end exactly at the selected checkpoint "
            f"(ledger={None if last is None else last.get('update')}, checkpoint={expected_update})"
        )


def _set_global_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _set_update_seed(seed: int, update: int) -> None:
    step_seed = (int(seed) * 1_000_003 + int(update) * 9_176 + 37) % (2**32)
    _set_global_seed(step_seed)


def _new_run_directory(config: Mapping[str, Any], *, smoke: bool = False) -> Path:
    root = resolve_path(str(config["run"]["output_root"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "active_budgeted_packet_smoke" if smoke else "active_budgeted_packet_reuse"
    path = root / f"Run_2111_{stamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _build_frozen_utility_organizer(
    core: Any,
    encoded: Any,
    tree: Any,
    organizer_state_dict: Mapping[str, Any],
    *,
    forward_config: Mapping[str, Any],
    device: torch.device,
    frontier_role_count: int = len(ROLE_NAMES),
) -> InputOnlyCoverOrganizer:
    """Rebuild the saved Stage-C organizer, including its utility head."""

    base = _build_organizer(core, encoded, tree, forward_config, device)
    utility = InputOnlyCoverOrganizer(
        state_dim=base.state_dim,
        module_feature_dim=base.module_feature_dim,
        environment_feature_dim=base.environment_feature_dim,
        hidden_dim=int(base.node_encoder[0].out_features),
        role_count=base.role_count,
        source_chunk_size=base.source_chunk_size,
        split_threshold=base.split_threshold,
        source_threshold=base.source_threshold,
        quadrature_invariant_source_measure=base.quadrature_invariant_source_measure,
        frontier_utility_enabled=True,
        frontier_role_count=int(frontier_role_count),
    ).to(device)
    missing, unexpected = utility.load_state_dict(organizer_state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"frozen utility organizer state mismatch (missing={list(missing)}, unexpected={list(unexpected)})"
        )
    utility.eval()
    for parameter in utility.parameters():
        parameter.requires_grad_(False)
    return utility


def _initialize_budgeted_route_score_heads(
    organizer: InputOnlyCoverOrganizer,
    routes: Sequence[str],
    *,
    seed: int,
    weight_std: float = 1.0e-3,
    permission_bias: float = 2.0,
) -> dict[str, Any]:
    """Break all-access init ties without changing the full-access bias."""

    if weight_std <= 0.0:
        raise ValueError("budgeted score-head initialization needs positive variance")
    route_records: dict[str, Any] = {}
    with torch.no_grad():
        for route_index, raw_route in enumerate(routes):
            route = str(raw_route).upper()
            if route not in organizer.pair_scorers:
                raise ValueError(f"unsupported budgeted native route {route!r}")
            scorer = organizer.pair_scorers[route]
            final = next(module for module in reversed(list(scorer.modules())) if isinstance(module, torch.nn.Linear))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + 10_007 * (route_index + 1))
            values = torch.randn(final.weight.shape, generator=generator, dtype=torch.float32)
            final.weight.copy_(values.to(device=final.weight.device, dtype=final.weight.dtype) * float(weight_std))
            final.bias.fill_(float(permission_bias))
            route_records[route] = {
                "final_weight_std": float(final.weight.std(unbiased=False).detach().cpu()),
                "final_bias": float(final.bias.detach().mean().cpu()),
            }
    return {
        "method": "centered_seeded_normal_final_pair_scorer_weights",
        "weight_std_target": float(weight_std),
        "permission_bias": float(permission_bias),
        "routes": route_records,
        "purpose": "avoid tied all-access initial scores while preserving positive full-access logits",
    }


def _verify_organizer_source_permutation(
    *,
    organizer: InputOnlyCoverOrganizer,
    score: Any,
    encoded: Any,
    tree: Any,
    mechanism: str,
    split_gates: torch.Tensor,
    budget_fraction: float,
) -> dict[str, Any]:
    """Check source equivariance and exact tie-atomic projection invariance."""

    source_embeddings = (
        score.module_embeddings if mechanism in {"MM", "EM", "QM"} else score.environment_embeddings
    )
    source_coordinates = (
        encoded.module_centers[0] if mechanism in {"MM", "EM", "QM"} else encoded.env_coords[0]
    )
    if score.node_embeddings is None or source_embeddings is None:
        raise RuntimeError("organizer did not return embeddings needed for permutation check")
    _node, node_centers, _extent = organizer._node_embeddings(tree, encoded.global_token[0])
    scale = organizer._case_scale(encoded.coordinate_scale, 0).to(
        device=node_centers.device,
        dtype=node_centers.dtype,
    )
    relative = (source_coordinates[None, :, :] - node_centers[:, None, :] * scale) / scale
    validity = (
        encoded.module_present[0] > 0.5
        if mechanism in {"MM", "EM", "QM"}
        else encoded.env_weights[0] > 0.0
    )
    with torch.no_grad():
        reference = organizer._score_sources(
            score.node_embeddings,
            source_embeddings,
            relative,
            scorer=organizer.pair_scorers[mechanism],
            mechanism=mechanism,
            budget_fraction=budget_fraction,
        )
        reference = reference.masked_fill(~validity[None, :], -30.0) if mechanism in {"MM", "EM", "QM"} else reference
        permutation = torch.arange(source_embeddings.shape[0] - 1, -1, -1, device=source_embeddings.device)
        permuted = organizer._score_sources(
            score.node_embeddings,
            source_embeddings.index_select(0, permutation),
            relative.index_select(1, permutation),
            scorer=organizer.pair_scorers[mechanism],
            mechanism=mechanism,
            budget_fraction=budget_fraction,
        )
        permuted_validity = validity.index_select(0, permutation)
        if mechanism in {"MM", "EM", "QM"}:
            permuted = permuted.masked_fill(~permuted_validity[None, :], -30.0)
        score_error = float((permuted - reference.index_select(1, permutation)).abs().max().detach().cpu())

        catalog = canonical_pair_catalog(encoded, tree, mechanism)
        projection = project_unique_pair_budget(
            reference,
            tree,
            split_gates,
            budget_fraction=budget_fraction,
            source_validity=catalog.source_validity,
            receiver_validity=catalog.receiver_validity,
            pair_validity=catalog.pair_validity,
            receiver_weights=catalog.receiver_weights,
            receiver_coordinates=catalog.receiver_coordinates,
        )
        permuted_projection = project_unique_pair_budget(
            reference.index_select(1, permutation),
            tree,
            split_gates,
            budget_fraction=budget_fraction,
            source_validity=catalog.source_validity.index_select(0, permutation),
            receiver_validity=catalog.receiver_validity,
            pair_validity=catalog.pair_validity.index_select(1, permutation),
            receiver_weights=catalog.receiver_weights,
            receiver_coordinates=catalog.receiver_coordinates,
        )
        inverse_permutation = torch.argsort(permutation)
        support_matches = torch.equal(
            projection.membership,
            permuted_projection.membership.index_select(1, inverse_permutation),
        )
        work_delta = abs(float(projection.achieved_work) - float(permuted_projection.achieved_work))
    if score_error > 2.0e-6 or not support_matches or work_delta > 1.0e-9:
        raise RuntimeError(
            f"{mechanism} source permutation/tie projection check failed "
            f"(score error={score_error}, support match={support_matches}, work delta={work_delta})"
        )
    return {
        "route": mechanism,
        "source_permutation": "reverse source order; receiver rows retained",
        "pair_score_equivariance_max_abs": score_error,
        "tie_atomic_projected_support_invariant": support_matches,
        "projected_work_abs_delta": work_delta,
        "selected_unique_pairs": int(projection.selected_unique_pairs),
        "full_unique_pairs": int(projection.full_unique_pairs),
    }


def _gradient_group_l2(parameters: Sequence[torch.nn.Parameter]) -> float:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.sqrt(torch.stack([gradient.square().sum() for gradient in gradients]).sum()).cpu())


def _train_g_packet(
    *,
    config: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    view: WindFarmNativeView,
    train_rows: Sequence[int],
    run_dir: Path,
    updates: int,
    seed: int,
    secondary_route: str,
    fixed_budget: float | None,
    stop_after_seconds: int | None,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    if secondary_route not in {"MM", "EM"}:
        raise ValueError("WindFarm packet routing requires a pilot-selected MM or EM secondary route")
    if updates < 1 or updates > int(config["forward"]["max_physical_updates_per_arm"]):
        raise ValueError("requested WindFarm update count is outside the Run2111 physical-update cap")
    device = torch.device("cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("WindFarm GPU execution requires CUDA_VISIBLE_DEVICES=0 (physical GPU 0)")
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("authorized physical GPU 0 is not available as logical cuda:0")
    _set_global_seed(seed)
    if torch.cuda.get_device_name(0) != "NVIDIA RTX 6000 Ada Generation":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 resolved to an unexpected physical GPU")
    resume_state = _load_resume_checkpoint(
        resume_checkpoint,
        arm="g_packet",
        source_sha256=EXPECTED_SOURCE_SHA256,
        train_rows_sha256=_rows_sha256(train_rows),
        secondary_route=secondary_route,
        seed=seed,
    )

    forward_cfg = config["forward"]
    role_counts = {str(key): int(value) for key, value in forward_cfg["stage_a"]["role_query_counts"].items()}
    sampler = NativeRoleCatalogueCache()
    layout_indices = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    preflight_row = 12 if 12 in train_rows else int(train_rows[0])
    preflight_case = view.run(preflight_row)
    preflight_sample = sample_native_role_queries(
        preflight_case,
        _native_rng(seed, 0, preflight_row, 710),
        role_counts,
        catalogue_cache=sampler,
    )
    preflight_batch = _batch_from_sample(preflight_case, preflight_sample, normalizer, device)
    model = _new_model_from_source(source_payload, normalizer, preflight_batch, device)
    model.eval()
    with torch.inference_mode():
        model.core.backend.set_cover_mode("full_access")
        parity_encoded = model.core.encode_case(preflight_batch)
        native_full = model.core.decode_queries(
            model.core.prepare(parity_encoded, parity_encoded.module_tokens),
            preflight_batch.query_xy,
            query_features=preflight_batch.query_features,
            receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
        )["pred_field"]
        model.core.backend.set_cover_mode("external")
        parity_tree = model.core.backend.build_case_trees(parity_encoded)[0]
        explicit_full_plan = MechanismPlan.full_access(
            parity_tree,
            parity_encoded.module_present[0],
            int(parity_encoded.env_coords.shape[1]),
        )
        explicit_full = model.core.decode_queries(
            model.core.prepare(
                parity_encoded,
                parity_encoded.module_tokens,
                fixed_cover_plans=(explicit_full_plan,),
            ),
            preflight_batch.query_xy,
            query_features=preflight_batch.query_features,
            receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
        )["pred_field"]
        parity_error = float((native_full - explicit_full).abs().max().cpu())
        parity_ok = bool(torch.allclose(native_full, explicit_full, atol=5.0e-6, rtol=5.0e-6))
    if not parity_ok:
        raise RuntimeError(f"u0 full-access route parity failed: max abs={parity_error:.8g}")
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    model.train()
    trainable = _enable_physical_trainable_scope(model)
    encoded = model.core.encode_case(preflight_batch)
    tree = model.core.backend.build_case_trees(encoded)[0]
    organizer = _build_organizer(model.core, encoded, tree, config["forward"], device)
    if resume_state is None:
        score_head_initialization = _initialize_budgeted_route_score_heads(
            organizer,
            ("QE", secondary_route),
            seed=seed,
        )
    else:
        score_head_initialization = resume_state.get("score_head_initialization", {})
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable, "lr": 1.0e-5, "weight_decay": 1.0e-5},
            {"params": list(organizer.parameters()), "lr": 2.0e-4, "weight_decay": 1.0e-5},
        ]
    )
    model_start_hash = _model_state_sha256(model)
    role_scales = {str(key): float(value) for key, value in forward_cfg["stage_a"]["role_loss_scales_mps"].items()}
    if set(role_scales) != set(ROLE_NAMES):
        raise ValueError("Run2111 role-loss scales do not match the native role sampler")
    if role_counts != dict(forward_cfg["stage_a"]["role_query_counts"]):
        raise ValueError("Run2111 native role counts must be explicit integers")

    arm_dir = run_dir / "arms" / "g_packet"
    checkpoint_dir = arm_dir / "checkpoints"
    if resume_state is None:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            arm_dir / "initial_state.json",
            {
                "source_checkpoint": str(config["source_checkpoint"]),
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "model_state_sha256_after_materialization": model_start_hash,
                "u0_policy_free_vs_explicit_full_max_abs_standardized": parity_error,
                "u0_full_access_parity_passed": parity_ok,
                "source_update_count": 1500,
                "initialization": "same exact W-full u1500 tensors; lazy materialization does not modify learned tensors",
                "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
                "organizer_parameter_count": int(sum(parameter.numel() for parameter in organizer.parameters())),
                "secondary_route": secondary_route,
                "budgeted_score_head_initialization": score_head_initialization,
                "device": "cuda:0 physical GPU 0",
                "train_rows_sha256": _rows_sha256(train_rows),
            },
        )
    else:
        if resume_checkpoint is None or not resume_checkpoint.resolve().is_relative_to(checkpoint_dir.resolve()):
            raise ValueError("G resume checkpoint must belong to this Run2111 arm directory")
        _validate_resume_history(arm_dir, resume_state)
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        organizer.load_state_dict(resume_state["organizer_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if _model_state_sha256(model) != resume_state.get("model_state_sha256"):
            raise ValueError("G resume model state hash does not match its checkpoint record")
        model_start_hash = str(resume_state["model_state_sha256"])

    review_updates = {int(value) for value in forward_cfg["review_updates"] if int(value) <= updates}
    review_updates.add(updates)
    start_time = time.monotonic()
    elapsed_before = 0.0 if resume_state is None else float(resume_state.get("elapsed_seconds_total", 0.0))
    start_update = 0 if resume_state is None else int(resume_state["update_count"])
    if updates <= start_update:
        raise ValueError(f"requested cumulative update target {updates} is not after resume point {start_update}")
    physical_optimizer_calls = 0 if resume_state is None else int(resume_state.get("physical_optimizer_calls", 0))
    organizer_optimizer_calls = 0 if resume_state is None else int(resume_state.get("organizer_optimizer_calls", 0))
    all_access_warm_updates = int(forward_cfg["stage_a"]["all_access_updates"])
    completed_update = start_update
    last_record: dict[str, Any] | None = None
    score_symmetry_checks: dict[str, Any] | None = None

    def save_checkpoint(update: int, record: Mapping[str, Any]) -> None:
        elapsed_total = elapsed_before + time.monotonic() - start_time
        state_payload = {
            "schema_version": 2,
            "run_id": "2111",
            "arm": "g_packet",
            "source_checkpoint": str(config["source_checkpoint"]),
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "source_update_count": 1500,
            "update_count": update,
            "model_state_sha256": _model_state_sha256(model),
            "model_state_dict": model.state_dict(),
            "organizer_state_dict": organizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalization": normalizer.to_dict(),
            "train_rows_sha256": _rows_sha256(train_rows),
            "secondary_route": secondary_route,
            "seed": seed,
            "sampler_identity": "counter_rng(seed, absolute_update, row, stream); deterministic native role catalogue",
            "score_head_initialization": score_head_initialization,
            "physical_optimizer_calls": physical_optimizer_calls,
            "organizer_optimizer_calls": organizer_optimizer_calls,
            "elapsed_seconds_total": elapsed_total,
            "budget_policy": {
                "mandatory_route": "QE",
                "secondary_route": secondary_route,
                "budgeted_score_head_initialization": score_head_initialization,
                "unbudgeted_routes": "explicit full-access bypass",
                "no_fixed_k_histogram": True,
            },
        }
        _atomic_checkpoint(checkpoint_dir / f"updates_{update:06d}.pt", state_payload)
        _write_json(arm_dir / "latest_review.json", {key: value for key, value in record.items() if key != "elapsed_seconds"})

    for update in range(start_update + 1, updates + 1):
        if stop_after_seconds is not None and time.monotonic() - start_time >= stop_after_seconds:
            if completed_update > start_update and last_record is not None:
                save_checkpoint(completed_update, last_record)
            break
        row_rng = _native_rng(seed, update, 0, 711)
        row = int(train_rows[int(row_rng.integers(0, len(train_rows)))])
        case = view.run(row)
        sample = sample_native_role_queries(
            case,
            _native_rng(seed, update, row, 712),
            role_counts,
            catalogue_cache=sampler,
        )
        batch = _batch_from_sample(case, sample, normalizer, device)
        packet_update = fixed_budget is not None or update > all_access_warm_updates
        _set_update_seed(seed, update)
        if not packet_update:
            encoded, prediction = _full_access_prediction(model, batch)
            cut = None
            fraction = 1.0
            budget_kind = "all_access_physical_only"
            route_budgets: dict[str, float] = {}
            plan_budgets: dict[str, float] = {}
        else:
            model.core.backend.set_cover_mode("external")
            encoded = model.core.encode_case(batch)
            trees = model.core.backend.build_case_trees(encoded)
            tree = trees[0]
            cut_rng = _native_rng(seed, update, row, 713)
            cut = _seeded_cut(tree, cut_rng)
            if fixed_budget is not None:
                fraction = float(fixed_budget)
                budget_kind = "explicit_smoke_budget"
            elif update <= (all_access_warm_updates + int(forward_cfg["stage_a"]["primary_budget_updates"])):
                fraction = float(forward_cfg["budgets"]["primary"])
                budget_kind = "primary_budget"
            else:
                mix = forward_cfg["stage_a"]["subsequent_budget_mix"]
                regime_rng = _native_rng(seed, update, row, 714)
                draw = float(regime_rng.random())
                cumulative = 0.0
                picked = "full_access"
                for name in ("full_access", "primary", "stress"):
                    cumulative += float(mix[name])
                    if draw < cumulative:
                        picked = name
                        break
                fraction = 1.0 if picked == "full_access" else float(config["forward"]["budgets"][picked])
                budget_kind = picked
            route_budgets, plan_budgets = _budgets(selected_route=secondary_route, fraction=fraction)

        if fixed_budget is not None:
            fraction = float(fixed_budget)
            budget_kind = "explicit_smoke_budget"

        optimizer.zero_grad(set_to_none=True)
        hard_plan = None
        route_projections: dict[str, Any] = {}
        if packet_update:
            result = hard_value_soft_organizer_forward(
                model.core,
                encoded,
                encoded.module_tokens,
                organizer,
                batch.query_xy,
                batch.query_features,
                receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
                budgets=route_budgets,
                frontier_cuts=(cut,),
                budget_fractions=plan_budgets,
            )
            prediction = result.prediction
            hard_plan = result.hard_plans[0]
            score_inputs = {
                "module_states": encoded.module_tokens,
                "environment_states": encoded.env_tokens,
                "global_state": encoded.global_token,
            }
            organizer_scores = organizer.score_cases(encoded, score_inputs, (tree,), budgets=route_budgets)
            if score_symmetry_checks is None:
                score_symmetry_checks = {
                    mechanism: _verify_organizer_source_permutation(
                        organizer=organizer,
                        score=organizer_scores[0],
                        encoded=encoded,
                        tree=tree,
                        mechanism=mechanism,
                        split_gates=hard_plan.split_gates,
                        budget_fraction=fraction,
                    )
                    for mechanism in ("QE", secondary_route)
                }
            for mechanism in ("QE", secondary_route):
                catalog = canonical_pair_catalog(encoded, tree, mechanism)
                route_projections[mechanism] = project_unique_pair_budget(
                    organizer_scores[0].mechanism_logits[mechanism],
                    tree,
                    hard_plan.split_gates,
                    budget_fraction=fraction,
                    source_validity=catalog.source_validity,
                    receiver_validity=catalog.receiver_validity,
                    pair_validity=catalog.pair_validity,
                    receiver_weights=catalog.receiver_weights,
                    receiver_coordinates=catalog.receiver_coordinates,
                )
        target = batch.target_field
        if target is None:
            raise RuntimeError("native optimizer sample unexpectedly lacks standardized velocity targets")
        loss, role_mse, role_rmse = _role_objective(prediction, target, sample.role_slices, normalizer, role_scales)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Run2111 G packet loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable + list(organizer.parameters()), max_norm=1.0)
        organizer_route_gradient_norms = {
            mechanism: _gradient_group_l2(tuple(organizer.pair_scorers[mechanism].parameters()))
            for mechanism in ("QE", secondary_route)
        }
        optimizer.step()
        physical_optimizer_calls += 1
        if packet_update:
            organizer_optimizer_calls += 1
        completed_update = update
        route_work_records: dict[str, Any] = {}
        for mechanism, projection in route_projections.items():
            plan_tree = hard_plan.tree
            catalog = canonical_pair_catalog(encoded, plan_tree, mechanism)
            live = hard_plan.frontier_summary(mechanism, receivers=catalog.receiver_coordinates)
            actual_pairs = int(
                (((hard_plan.access_for(
                    mechanism,
                    catalog.receiver_coordinates,
                    int(catalog.source_validity.shape[0]),
                    module_present=(
                        encoded.module_present[0] if mechanism in {"MM", "EM", "QM"} else None
                    ),
                ) > 0.0) & catalog.pair_validity).sum()).detach().cpu()
            )
            full_pairs = int(projection.full_unique_pairs)
            live_access = hard_plan.access_for(
                mechanism,
                catalog.receiver_coordinates,
                int(catalog.source_validity.shape[0]),
                module_present=(encoded.module_present[0] if mechanism in {"MM", "EM", "QM"} else None),
            )
            live_support = (live_access > 0.0) & catalog.pair_validity
            live_work = float(
                (live_support.to(torch.float64) * catalog.receiver_weights.to(torch.float64)[:, None])
                .sum()
                .detach()
                .cpu()
            )
            route_work_records[mechanism] = {
                "budget_fraction": float(projection.budget_fraction),
                "requested_work": float(projection.requested_work),
                "achieved_unique_work": live_work,
                "full_access_unique_work": float(projection.full_access_work),
                "work_fraction": live_work / float(projection.full_access_work)
                if projection.full_access_work > 0.0
                else 0.0,
                "selected_unique_pairs": actual_pairs,
                "projection_unique_pairs": int(projection.selected_unique_pairs),
                "full_unique_pairs": full_pairs,
                "nonredundant_packets": int(live.nonredundant_packet_count),
                "frontier_summary_raw_pair_count": int(live.unique_source_receiver_pairs),
                "empty_support": actual_pairs == 0,
                "within_canonical_cap": live_work <= float(projection.requested_work) + 1.0e-9,
                "projection_live_count_match": actual_pairs == int(projection.selected_unique_pairs),
                "projection_achieved_work": float(projection.achieved_work),
                "sparse_success": bool(
                    0 < actual_pairs < full_pairs
                    and live_work < float(projection.full_access_work)
                    and live_work <= float(projection.requested_work) + 1.0e-9
                ) if full_pairs > 0 else None,
            }
        record = {
            "update": update,
            "row": row,
            "layout_index": int(layout_indices[row]),
            "wind_direction_deg": float(case.wind_direction_deg),
            "budget_kind": budget_kind,
            "budget_fraction_qe": fraction,
            "budget_fraction_secondary": fraction,
            "secondary_route": secondary_route,
            "frontier_cut": None if cut is None else list(cut),
            "frontier_size": None if cut is None else len(cut),
            "objective_role_normalized_mse_mean": float(loss.detach().cpu()),
            "role_mse_mps2": role_mse,
            "role_rmse_mps": role_rmse,
            "physical_optimizer_calls": physical_optimizer_calls,
            "organizer_optimizer_calls": organizer_optimizer_calls,
            "permission_statuses": (
                {key: "native_full_access" for key in ("MM", "ME", "EM", "QM", "QE")}
                if hard_plan is None
                else {key: hard_plan.permission_status(key) for key in ("MM", "ME", "EM", "QM", "QE")}
            ),
            "budgeted_route_work": route_work_records,
            "organizer_route_gradient_l2": organizer_route_gradient_norms,
            "source_permutation_tie_checks": score_symmetry_checks,
            "full_access_bypass_routes": (
                ["MM", "ME", "EM", "QM", "QE"] if hard_plan is None else list(hard_plan.explicit_bypass_keys)
            ),
            "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
        }
        last_record = record
        _jsonl(arm_dir / "updates.jsonl", record)
        if update in review_updates:
            save_checkpoint(update, record)
        if update % 10 == 0 or update == 1:
            cut_summary = "full_access" if cut is None else str(len(cut))
            print(
                f"G packet update={update}/{updates} row={row} b={fraction:.2f} "
                f"loss={float(loss.detach().cpu()):.5g} cut={cut_summary}",
                flush=True,
            )

    return {
        "arm": "g_packet",
        "updates": completed_update,
        "updates_this_invocation": completed_update - start_update,
        "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
        "physical_optimizer_calls": physical_optimizer_calls,
        "organizer_optimizer_calls": organizer_optimizer_calls,
        "final_model_state_sha256": _model_state_sha256(model),
        "secondary_route": secondary_route,
        "run_dir": str(arm_dir),
    }


def _run_direct_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run an independent direct-pair control on exact receiver×source masks."""

    config = kwargs["config"]
    source_payload = kwargs["source_payload"]
    normalizer = kwargs["normalizer"]
    view = kwargs["view"]
    train_rows = kwargs["train_rows"]
    run_dir = kwargs["run_dir"]
    updates = int(kwargs["updates"])
    seed = int(kwargs["seed"])
    secondary_route = str(kwargs["secondary_route"])
    fixed_budget = kwargs["fixed_budget"]
    stop_after_seconds = kwargs["stop_after_seconds"]
    resume_checkpoint = kwargs["resume_checkpoint"]
    if secondary_route not in {"MM", "EM"}:
        raise ValueError("WindFarm direct control requires the pilot-selected MM or EM route")
    if updates < 1 or updates > int(config["forward"]["max_physical_updates_per_arm"]):
        raise ValueError("requested WindFarm update count is outside the Run2111 physical-update cap")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("WindFarm GPU execution requires CUDA_VISIBLE_DEVICES=0 (physical GPU 0)")
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("authorized physical GPU 0 is not available as logical cuda:0")
    resume_state = _load_resume_checkpoint(
        resume_checkpoint,
        arm="direct_pair",
        source_sha256=EXPECTED_SOURCE_SHA256,
        train_rows_sha256=_rows_sha256(train_rows),
        secondary_route=secondary_route,
        seed=seed,
    )
    _set_global_seed(seed)
    device = torch.device("cuda:0")
    forward_cfg = config["forward"]
    role_counts = {str(key): int(value) for key, value in forward_cfg["stage_a"]["role_query_counts"].items()}
    sampler = NativeRoleCatalogueCache()
    preflight_row = 12 if 12 in train_rows else int(train_rows[0])
    preflight_case = view.run(preflight_row)
    preflight_sample = sample_native_role_queries(
        preflight_case,
        _native_rng(seed, 0, preflight_row, 710),
        role_counts,
        catalogue_cache=sampler,
    )
    preflight_batch = _batch_from_sample(preflight_case, preflight_sample, normalizer, device)
    model = _new_model_from_source(source_payload, normalizer, preflight_batch, device)
    model.eval()
    with torch.inference_mode():
        model.core.backend.set_cover_mode("full_access")
        parity_encoded = model.core.encode_case(preflight_batch)
        native_full = model.core.decode_queries(
            model.core.prepare(parity_encoded, parity_encoded.module_tokens),
            preflight_batch.query_xy,
            query_features=preflight_batch.query_features,
            receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
        )["pred_field"]
        model.core.backend.set_cover_mode("external")
        parity_tree = model.core.backend.build_case_trees(parity_encoded)[0]
        explicit_plan = MechanismPlan.full_access(
            parity_tree,
            parity_encoded.module_present[0],
            int(parity_encoded.env_coords.shape[1]),
        )
        explicit_full = model.core.decode_queries(
            model.core.prepare(
                parity_encoded,
                parity_encoded.module_tokens,
                fixed_cover_plans=(explicit_plan,),
            ),
            preflight_batch.query_xy,
            query_features=preflight_batch.query_features,
            receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
        )["pred_field"]
        parity_error = float((native_full - explicit_full).abs().max().cpu())
        parity_ok = bool(torch.allclose(native_full, explicit_full, atol=5.0e-6, rtol=5.0e-6))
    if not parity_ok:
        raise RuntimeError(f"P u0 full-access route parity failed: max abs={parity_error:.8g}")
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    model.train()
    trainable = _enable_physical_trainable_scope(model)
    encoded_preflight = model.core.encode_case(preflight_batch)
    feature_width = int(_direct_feature_tables(_detach_inputs(encoded_preflight))["width"])
    scorer = BudgetConditionedDirectPairScorer(
        receiver_feature_dim=feature_width,
        source_feature_dim=feature_width,
        hidden_dim=96,
        source_chunk_size=128,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable, "lr": 1.0e-5, "weight_decay": 1.0e-5},
            {"params": list(scorer.parameters()), "lr": 2.0e-4, "weight_decay": 1.0e-5},
        ]
    )
    model_start_hash = _model_state_sha256(model)
    role_scales = {str(key): float(value) for key, value in forward_cfg["stage_a"]["role_loss_scales_mps"].items()}
    arm_dir = run_dir / "arms" / "direct_pair"
    checkpoint_dir = arm_dir / "checkpoints"
    if resume_state is None:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            arm_dir / "initial_state.json",
            {
                "source_checkpoint": str(config["source_checkpoint"]),
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "model_state_sha256_after_materialization": model_start_hash,
                "u0_policy_free_vs_explicit_full_max_abs_standardized": parity_error,
                "u0_full_access_parity_passed": parity_ok,
                "source_update_count": 1500,
                "initialization": "same exact W-full u1500 tensors; lazy materialization does not modify learned tensors",
                "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
                "direct_scorer_parameter_count": int(sum(parameter.numel() for parameter in scorer.parameters())),
                "direct_feature_width": feature_width,
                "secondary_route": secondary_route,
                "device": "cuda:0 physical GPU 0",
                "train_rows_sha256": _rows_sha256(train_rows),
            },
        )
    else:
        if resume_checkpoint is None or not resume_checkpoint.resolve().is_relative_to(checkpoint_dir.resolve()):
            raise ValueError("P resume checkpoint must belong to this Run2111 arm directory")
        _validate_resume_history(arm_dir, resume_state)
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        scorer.load_state_dict(resume_state["direct_scorer_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if _model_state_sha256(model) != resume_state.get("model_state_sha256"):
            raise ValueError("P resume model state hash does not match its checkpoint record")
        model_start_hash = str(resume_state["model_state_sha256"])
    review_updates = {int(value) for value in forward_cfg["review_updates"] if int(value) <= updates}
    review_updates.add(updates)
    start_time = time.monotonic()
    elapsed_before = 0.0 if resume_state is None else float(resume_state.get("elapsed_seconds_total", 0.0))
    start_update = 0 if resume_state is None else int(resume_state["update_count"])
    if updates <= start_update:
        raise ValueError(f"requested cumulative update target {updates} is not after resume point {start_update}")
    physical_optimizer_calls = 0 if resume_state is None else int(resume_state.get("physical_optimizer_calls", 0))
    scorer_optimizer_calls = 0 if resume_state is None else int(resume_state.get("direct_scorer_optimizer_calls", 0))
    gradient_separation = None if resume_state is None else resume_state.get("gradient_separation")
    warm_updates = int(forward_cfg["stage_a"]["all_access_updates"])
    completed_update = start_update
    last_record: dict[str, Any] | None = None

    def save_checkpoint(update: int, record: Mapping[str, Any]) -> None:
        elapsed_total = elapsed_before + time.monotonic() - start_time
        state_payload = {
            "schema_version": 2,
            "run_id": "2111",
            "arm": "direct_pair",
            "source_checkpoint": str(config["source_checkpoint"]),
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "source_update_count": 1500,
            "update_count": update,
            "model_state_sha256": _model_state_sha256(model),
            "model_state_dict": model.state_dict(),
            "direct_scorer_state_dict": scorer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalization": normalizer.to_dict(),
            "train_rows_sha256": _rows_sha256(train_rows),
            "secondary_route": secondary_route,
            "seed": seed,
            "sampler_identity": "counter_rng(seed, absolute_update, row, stream); deterministic native role catalogue",
            "physical_optimizer_calls": physical_optimizer_calls,
            "direct_scorer_optimizer_calls": scorer_optimizer_calls,
            "elapsed_seconds_total": elapsed_total,
            "budget_policy": {
                "mandatory_route": "QE",
                "secondary_route": secondary_route,
                "unbudgeted_routes": "explicit full-access bypass",
                "no_packet_approximation": True,
            },
            "gradient_separation": gradient_separation,
        }
        _atomic_checkpoint(checkpoint_dir / f"updates_{update:06d}.pt", state_payload)
        _write_json(arm_dir / "latest_review.json", {key: value for key, value in record.items() if key != "elapsed_seconds"})

    for update in range(start_update + 1, updates + 1):
        if stop_after_seconds is not None and time.monotonic() - start_time >= stop_after_seconds:
            if completed_update > start_update and last_record is not None:
                save_checkpoint(completed_update, last_record)
            break
        row_rng = _native_rng(seed, update, 0, 711)
        row = int(train_rows[int(row_rng.integers(0, len(train_rows)))])
        case = view.run(row)
        sample = sample_native_role_queries(
            case,
            _native_rng(seed, update, row, 712),
            role_counts,
            catalogue_cache=sampler,
        )
        batch = _batch_from_sample(case, sample, normalizer, device)
        packet_update = fixed_budget is not None or update > warm_updates
        _set_update_seed(seed, update)
        if not packet_update:
            _encoded, prediction = _full_access_prediction(model, batch)
            fraction = 1.0
            budget_kind = "all_access_physical_only"
        else:
            model.core.backend.set_cover_mode("external")
            encoded = model.core.encode_case(batch)
            if gradient_separation is None:
                encoded = replace(
                    encoded,
                    module_centers=encoded.module_centers.detach().clone().requires_grad_(True),
                )
            if fixed_budget is not None:
                fraction = float(fixed_budget)
                budget_kind = "explicit_smoke_budget"
            elif update <= warm_updates + int(forward_cfg["stage_a"]["primary_budget_updates"]):
                fraction = float(config["forward"]["budgets"]["primary"])
                budget_kind = "primary_budget"
            else:
                mix = forward_cfg["stage_a"]["subsequent_budget_mix"]
                regime_rng = _native_rng(seed, update, row, 714)
                draw = float(regime_rng.random())
                cumulative = 0.0
                picked = "full_access"
                for name in ("full_access", "primary", "stress"):
                    cumulative += float(mix[name])
                    if draw < cumulative:
                        picked = name
                        break
                fraction = 1.0 if picked == "full_access" else float(config["forward"]["budgets"][picked])
                budget_kind = picked
            hard_plan, soft_plan, route_projections, actual_access = _direct_plan_pair(
                model=model,
                encoded=encoded,
                batch=batch,
                scorer=scorer,
                secondary_route=secondary_route,
                fraction=fraction,
            )
            result = hard_value_soft_direct_forward(
                model.core,
                encoded,
                encoded.module_tokens,
                (hard_plan,),
                (soft_plan,),
                batch.query_xy,
                batch.query_features,
                # Direct QE access is bound to this exact sampled receiver
                # panel; one native read preserves receiver order for the
                # hard and soft route masks.
                receiver_chunk_size=int(batch.query_xy.shape[1]),
            )
            prediction = result.prediction

        optimizer.zero_grad(set_to_none=True)
        target = batch.target_field
        if target is None:
            raise RuntimeError("native optimizer sample unexpectedly lacks standardized velocity targets")
        loss, role_mse, role_rmse = _role_objective(prediction, target, sample.role_slices, normalizer, role_scales)
        gradient_probe_record = None
        if packet_update and gradient_separation is None:
            hard_loss, _hard_mse, _hard_rmse = _role_objective(
                result.hard_prediction, target, sample.role_slices, normalizer, role_scales
            )
            physical_and_design_inputs = [*trainable, encoded.module_centers]
            hard_gradients = torch.autograd.grad(
                hard_loss,
                physical_and_design_inputs,
                retain_graph=True,
                allow_unused=True,
            )
            combined_gradients = torch.autograd.grad(
                loss,
                physical_and_design_inputs,
                retain_graph=True,
                allow_unused=True,
            )
            max_physical_abs = 0.0
            max_physical_reference = 0.0
            design_hard = hard_gradients[-1]
            design_combined = combined_gradients[-1]
            if design_hard is None:
                design_hard = torch.zeros_like(encoded.module_centers)
            if design_combined is None:
                design_combined = torch.zeros_like(encoded.module_centers)
            design_max_abs = float((design_hard - design_combined).abs().max().detach().cpu())
            design_reference = float(design_hard.abs().max().detach().cpu())
            for parameter, hard_gradient, combined_gradient in zip(
                trainable, hard_gradients[:-1], combined_gradients[:-1], strict=True
            ):
                hard_value = torch.zeros_like(parameter) if hard_gradient is None else hard_gradient
                combined_value = torch.zeros_like(parameter) if combined_gradient is None else combined_gradient
                max_physical_abs = max(
                    max_physical_abs,
                    float((hard_value - combined_value).abs().max().detach().cpu()),
                )
                max_physical_reference = max(
                    max_physical_reference,
                    float(hard_value.abs().max().detach().cpu()),
                )
            scorer_gradients = torch.autograd.grad(
                loss, tuple(scorer.parameters()), retain_graph=True, allow_unused=True
            )
            nonempty_scorer_gradients = [gradient.detach() for gradient in scorer_gradients if gradient is not None]
            scorer_gradient_norm = (
                float(torch.sqrt(torch.stack([gradient.square().sum() for gradient in nonempty_scorer_gradients]).sum()).cpu())
                if nonempty_scorer_gradients
                else 0.0
            )
            physical_pass = max_physical_abs <= 1.0e-5 + 1.0e-4 * max_physical_reference
            design_pass = design_max_abs <= 1.0e-5 + 1.0e-4 * design_reference
            passed = physical_pass and design_pass
            if not passed:
                raise RuntimeError(
                    "P soft-shadow path changed physical-model gradients; encoded/scorer inputs must stay detached"
                )
            gradient_separation = {
                "passed": passed,
                "physical_gradient_max_abs_delta": max_physical_abs,
                "hard_physical_gradient_max_abs": max_physical_reference,
                "design_gradient_max_abs_delta": design_max_abs,
                "hard_design_gradient_max_abs": design_reference,
                "direct_scorer_gradient_l2": scorer_gradient_norm,
                "physical_and_design_inputs_detached_from_scorer": True,
            }
            gradient_probe_record = gradient_separation
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Run2111 P direct-pair loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable + list(scorer.parameters()), max_norm=1.0)
        optimizer.step()
        physical_optimizer_calls += 1
        if packet_update:
            scorer_optimizer_calls += 1
        completed_update = update
        record = {
            "update": update,
            "row": row,
            "layout_index": int(view.metadata["layout_index"][row]),
            "wind_direction_deg": float(case.wind_direction_deg),
            "budget_kind": budget_kind,
            "budget_fraction_qe": fraction,
            "budget_fraction_secondary": fraction,
            "secondary_route": secondary_route,
            "objective_role_normalized_mse_mean": float(loss.detach().cpu()),
            "role_mse_mps2": role_mse,
            "role_rmse_mps": role_rmse,
            "physical_optimizer_calls": physical_optimizer_calls,
            "direct_scorer_optimizer_calls": scorer_optimizer_calls,
            "budgeted_route_work": {
                key: {
                    "budget_fraction": float(value.budget_fraction),
                    "requested_work": float(value.requested_work),
                    "achieved_unique_work": float(value.achieved_work),
                    "full_access_unique_work": float(value.full_access_work),
                    "work_fraction": (
                        float(value.achieved_work / value.full_access_work)
                        if value.full_access_work > 0.0
                        else 0.0
                    ),
                    "selected_unique_pairs": int(value.selected_unique_pairs),
                    "full_unique_pairs": int(value.full_unique_pairs),
                    "canonical_sparse_success": (
                        bool(
                            value.sparse_success
                            and 0 < int(value.selected_unique_pairs) < int(value.full_unique_pairs)
                        )
                        if value.full_unique_pairs > 0
                        else None
                    ),
                    "actual_receiver_panel_sparse_success": actual_access[key][
                        "actual_receiver_panel_sparse_success"
                    ],
                }
                for key, value in (route_projections.items() if packet_update else [])
            },
            "direct_receiver_access": actual_access if packet_update else {},
            "full_access_bypass_routes": (
                [route for route in ("MM", "ME", "EM", "QM", "QE") if route not in {"QE", secondary_route}]
                if packet_update
                else ["MM", "ME", "EM", "QM", "QE"]
            ),
            "physical_gradient_separation": gradient_probe_record,
            "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
        }
        last_record = record
        _jsonl(arm_dir / "updates.jsonl", record)
        if update in review_updates:
            save_checkpoint(update, record)
        if update % 10 == 0 or update == 1:
            print(
                f"P direct update={update}/{updates} row={row} b={fraction:.2f} loss={float(loss.detach().cpu()):.5g}",
                flush=True,
            )
    return {
        "arm": "direct_pair",
        "updates": completed_update,
        "updates_this_invocation": completed_update - start_update,
        "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
        "physical_optimizer_calls": physical_optimizer_calls,
        "direct_scorer_optimizer_calls": scorer_optimizer_calls,
        "final_model_state_sha256": _model_state_sha256(model),
        "gradient_separation": gradient_separation,
        "secondary_route": secondary_route,
        "run_dir": str(arm_dir),
    }


def _stage_b_rows(
    *,
    config: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    view: WindFarmNativeView,
    train_rows: Sequence[int],
    run_dir: Path,
    stage_a_checkpoint: Path,
    output_path: Path | None,
    seed: int,
    secondary_route: str,
) -> dict[str, Any]:
    """Measure bounded train-only frontier fidelity and unique-pair work."""

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("WindFarm Stage B requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("authorized physical GPU 0 is unavailable for Stage B")
    if secondary_route not in {"MM", "EM"}:
        raise ValueError("Stage B requires the training-only pilot route")
    stage_a_checkpoint = stage_a_checkpoint.resolve()
    if not stage_a_checkpoint.is_file():
        raise FileNotFoundError(f"Stage-A G checkpoint does not exist: {stage_a_checkpoint}")
    checkpoint_sha = _file_sha256(stage_a_checkpoint)
    checkpoint = load_trusted_checkpoint(stage_a_checkpoint, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("arm") != "g_packet":
        raise ValueError("Stage B requires a trusted Run2111 G packet checkpoint")
    if checkpoint.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("Stage-A checkpoint does not originate from the audited Run2110 source")
    if checkpoint.get("train_rows_sha256") != _rows_sha256(train_rows):
        raise ValueError("Stage-A checkpoint was trained with a different WindFarm training split")
    if checkpoint.get("secondary_route") != secondary_route:
        raise ValueError("Stage-A checkpoint route differs from the training-only route pilot")

    device = torch.device("cuda:0")
    stage_cfg = config["forward"]["stage_b"]
    query_counts = {str(key): int(value) for key, value in stage_cfg["role_query_counts"].items()}
    if set(query_counts) != set(ROLE_NAMES) or any(value < 1 for value in query_counts.values()):
        raise ValueError("Stage-B query panel must include positive counts for all protected roles")
    max_rows = int(stage_cfg["initial_training_rows"])
    if max_rows > int(stage_cfg["maximum_training_rows"]):
        raise ValueError("Stage-B initial row count exceeds the declared Stage-B cap")
    layout_indices = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    rows_by_layout: dict[int, list[int]] = {}
    for row in train_rows:
        rows_by_layout.setdefault(int(layout_indices[int(row)]), []).append(int(row))
    eligible_layouts = sorted(rows_by_layout)
    if len(eligible_layouts) < max_rows:
        raise ValueError("Stage B requests more unique layouts than the post-exclusion train split")
    row_rng = _native_rng(seed, 0, 0, 731)
    selected_layouts = sorted(
        map(int, row_rng.choice(np.asarray(eligible_layouts, dtype=np.int64), size=max_rows, replace=False))
    )
    selected_rows: list[int] = []
    for layout in selected_layouts:
        candidates = sorted(rows_by_layout[layout])
        selected_rows.append(candidates[int(_native_rng(seed, 0, layout, 732).integers(0, len(candidates)))])

    scales = config["forward"]["stage_a"]["role_loss_scales_mps"]
    sampler = NativeRoleCatalogueCache()

    def make_batch(row: int) -> tuple[Any, Any, Any]:
        case = view.run(int(row))
        sample = sample_native_role_queries(
            case,
            _native_rng(seed, 0, int(row), 733),
            query_counts,
            catalogue_cache=sampler,
        )
        batch = _batch_from_sample(case, sample, normalizer, device)
        return case, sample, batch

    _first_case, _first_sample, first_batch = make_batch(selected_rows[0])
    model = _new_model_from_source(source_payload, normalizer, first_batch, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference_rmse: dict[int, dict[str, float]] = {}
    # Measure Run2110's incumbent on the exact query panels used by Stage B.
    for row in selected_rows:
        _case, sample, batch = make_batch(row)
        with torch.no_grad():
            _encoded, reference_prediction = _full_access_prediction(model, batch)
            _reference_loss, _reference_mse, reference_role_rmse = _role_objective(
                reference_prediction, batch.target_field, sample.role_slices, normalizer, scales
            )
        reference_rmse[int(row)] = reference_role_rmse

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    first_encoded, _first_full = _full_access_prediction(model, first_batch)
    first_tree = model.core.backend.build_case_trees(first_encoded)[0]
    organizer = _build_organizer(model.core, first_encoded, first_tree, config["forward"]["stage_a"], device)
    organizer.load_state_dict(checkpoint["organizer_state_dict"], strict=True)
    organizer.eval()
    for parameter in organizer.parameters():
        parameter.requires_grad_(False)
    route_budgets, plan_budgets = _budgets(
        selected_route=secondary_route,
        fraction=float(stage_cfg["budget_fraction"]),
    )
    chunk_size = int(model.config.interface_model.receiver_chunk_size)
    all_mechanisms = ("MM", "ME", "EM", "QM", "QE")
    result_rows: list[dict[str, Any]] = []

    for row in selected_rows:
        case, sample, batch = make_batch(row)
        with torch.no_grad():
            encoded, full_policy_prediction = _full_access_prediction(model, batch)
            trees = model.core.backend.build_case_trees(encoded)
            if len(trees) != 1:
                raise ValueError("WindFarm Stage B expects one current case tree")
            tree = trees[0]
            full_plan = MechanismPlan.full_access(
                tree,
                encoded.module_present[0],
                int(encoded.env_coords.shape[1]),
            )
            full_plan_prediction = _prediction_with_plan(model, encoded, batch, full_plan)
            prepared_full = model.core.prepare(
                encoded,
                encoded.module_tokens,
                fixed_cover_plans=(full_plan,),
            )
            alternate_chunk = max(1, chunk_size // 2)
            alternate_chunk_prediction = model.core.decode_queries(
                prepared_full,
                batch.query_xy,
                query_features=batch.query_features,
                receiver_chunk_size=alternate_chunk,
            )["pred_field"]
            full_loss, full_mse, full_rmse = _role_objective(
                full_policy_prediction, batch.target_field, sample.role_slices, normalizer, scales
            )
            del full_loss, full_mse
            full_pair_repeats = torch.stack(
                (
                    full_policy_prediction - full_plan_prediction,
                    full_plan_prediction - alternate_chunk_prediction,
                    full_policy_prediction - alternate_chunk_prediction,
                ),
                dim=0,
            )
            repeat_mps = _standardized_delta_to_mps(full_pair_repeats, normalizer)
            numerical_allowance: dict[str, float] = {}
            for role in ROLE_NAMES:
                role_values = repeat_mps[:, 0, sample.role_slices[role], :]
                numerical_allowance[role] = float(role_values.abs().max().detach().cpu())

            prepared_state = {
                "module_states": encoded.module_tokens,
                "environment_states": encoded.env_tokens,
                "global_state": encoded.global_token,
            }
            scores = organizer.score_cases(encoded, prepared_state, trees, budgets=route_budgets)
            cuts = enumerate_frontier_cuts(tree, max_depth=3)
            if len(cuts) > int(stage_cfg["maximum_frontier_cuts_per_case"]):
                raise ValueError("Stage-B bounded tree produced too many candidate cuts")
            candidate_rows: list[dict[str, Any]] = []
            role_error_table: list[list[float]] = []
            for cut in cuts:
                plans = organizer.plans_from_scores(
                    scores,
                    encoded,
                    trees,
                    hard=True,
                    frontier_cuts=(cut,),
                    budget_fractions=plan_budgets,
                )
                plan = plans[0]
                prediction = _prediction_with_plan(model, encoded, batch, plan)
                _loss, _mse, candidate_rmse = _role_objective(
                    prediction, batch.target_field, sample.role_slices, normalizer, scales
                )
                role_error_table.append([float(candidate_rmse[role]) for role in ROLE_NAMES])
                adequate_by_role = {
                    role: bool(
                        candidate_rmse[role]
                        <= float(stage_cfg["adequacy_reference_ratio"]) * full_rmse[role]
                        + numerical_allowance[role]
                    )
                    for role in ROLE_NAMES
                }
                route_metrics: dict[str, Any] = {}
                total_work = 0.0
                full_total_work = 0.0
                packet_total = 0
                for mechanism in all_mechanisms:
                    catalog = canonical_pair_catalog(encoded, tree, mechanism)
                    support_metrics = _canonical_route_support(plan, encoded, tree, mechanism)
                    full_work = float(support_metrics["canonical_full_work"])
                    selected_work = float(support_metrics["canonical_actual_work"])
                    full_pairs = int(support_metrics["eligible_unique_pairs"])
                    selected_pairs = int(support_metrics["actual_live_unique_pairs"])
                    is_budgeted = mechanism in {"QE", secondary_route}
                    if is_budgeted:
                        projection = project_unique_pair_budget(
                            scores[0].mechanism_logits[mechanism],
                            tree,
                            plan.split_gates,
                            budget_fraction=float(stage_cfg["budget_fraction"]),
                            source_validity=catalog.source_validity,
                            receiver_validity=catalog.receiver_validity,
                            pair_validity=catalog.pair_validity,
                            receiver_weights=catalog.receiver_weights,
                            receiver_coordinates=catalog.receiver_coordinates,
                        )
                        achieved_work = selected_work
                        if abs(achieved_work - float(projection.achieved_work)) > 1.0e-8 + 1.0e-6 * max(
                            1.0, float(projection.achieved_work)
                        ):
                            raise RuntimeError(
                                f"Stage-B {mechanism} hard access disagrees with its canonical projection"
                            )
                        sparse_success = bool(
                            projection.sparse_success
                            and 0 < selected_pairs < full_pairs
                            and plan.permission_status(mechanism) != "full_access_bypass_missing_key"
                        ) if projection.full_unique_pairs > 0 else None
                        requested_work = float(projection.requested_work)
                        nonredundant_packets = int(support_metrics["nonredundant_packet_count"])
                    else:
                        # Missing typed keys are intentional full-access bypasses.
                        achieved_work = full_work
                        sparse_success = None
                        requested_work = full_work
                        nonredundant_packets = int(support_metrics["nonredundant_packet_count"])
                    total_work += achieved_work
                    full_total_work += full_work
                    packet_total += nonredundant_packets
                    route_metrics[mechanism] = {
                        "permission_status": plan.permission_status(mechanism),
                        "budgeted": is_budgeted,
                        "eligible_unique_pairs": full_pairs,
                        "actual_live_unique_pairs": selected_pairs,
                        "canonical_full_work": full_work,
                        "canonical_requested_work": requested_work,
                        "canonical_achieved_work": achieved_work,
                        "canonical_work_fraction": achieved_work / full_work if full_work > 0 else 0.0,
                        "nonredundant_packet_count": nonredundant_packets,
                        "frontier_summary_raw_pair_count": int(
                            support_metrics["frontier_summary_raw_pair_count"]
                        ),
                        "sparse_success": sparse_success,
                    }
                work_fraction = total_work / full_total_work if full_total_work > 0.0 else 0.0
                candidate_rows.append(
                    {
                        "frontier_cut": list(cut),
                        "frontier_size": len(cut),
                        "nonredundant_packet_count": packet_total,
                        "role_rmse_mps": {role: float(candidate_rmse[role]) for role in ROLE_NAMES},
                        "role_positive_excess_from_same_student_full": {
                            role: abs(float(candidate_rmse[role]) - float(full_rmse[role]))
                            / max(float(full_rmse[role]), 1.0e-8)
                            if float(candidate_rmse[role]) > float(full_rmse[role])
                            else 0.0
                            for role in ROLE_NAMES
                        },
                        "adequate_by_role": adequate_by_role,
                        "adequate": all(adequate_by_role.values()),
                        "canonical_total_work": total_work,
                        "canonical_full_access_work": full_total_work,
                        "canonical_work_fraction": work_fraction,
                        "route_work": route_metrics,
                        "full_access_bypass_routes": list(plan.explicit_bypass_keys),
                    }
                )

            adequate_indices = [i for i, item in enumerate(candidate_rows) if item["adequate"]]
            if adequate_indices:
                selected_index: int | None = min(
                    adequate_indices,
                    key=lambda i: (
                        float(candidate_rows[i]["canonical_total_work"]),
                        int(candidate_rows[i]["nonredundant_packet_count"]),
                        i,
                    ),
                )
            else:
                selected_index = None
            risk_values = []
            for candidate in candidate_rows:
                per_role_excess = []
                for role in ROLE_NAMES:
                    cap = (
                        float(stage_cfg["adequacy_reference_ratio"]) * full_rmse[role]
                        + numerical_allowance[role]
                    )
                    per_role_excess.append(
                        max(0.0, candidate["role_rmse_mps"][role] - cap) / max(cap, 1.0e-8)
                    )
                risk_values.append(max(per_role_excess))
            least_risk_index = min(range(len(candidate_rows)), key=lambda i: (risk_values[i], i))
            pareto_indices = []
            for candidate_index, candidate in enumerate(candidate_rows):
                errors = [candidate["role_rmse_mps"][role] for role in ROLE_NAMES]
                dominated = False
                for other_index, other in enumerate(candidate_rows):
                    if other_index == candidate_index:
                        continue
                    other_errors = [other["role_rmse_mps"][role] for role in ROLE_NAMES]
                    no_worse = (
                        other["canonical_total_work"] <= candidate["canonical_total_work"]
                        and all(a <= b for a, b in zip(other_errors, errors, strict=True))
                    )
                    strictly_better = (
                        other["canonical_total_work"] < candidate["canonical_total_work"]
                        or any(a < b for a, b in zip(other_errors, errors, strict=True))
                    )
                    if no_worse and strictly_better:
                        dominated = True
                        break
                if not dominated:
                    pareto_indices.append(candidate_index)
            selected_cut = None if selected_index is None else tuple(cuts[selected_index])
            if selected_cut is None:
                first_selected_plan = full_plan
                first_selection_fallback = True
            else:
                first_selected_plan = organizer.plans_from_scores(
                    scores,
                    encoded,
                    trees,
                    hard=True,
                    frontier_cuts=(selected_cut,),
                    budget_fractions=plan_budgets,
                )[0]
                first_selection_fallback = False
            first_selected_prediction, first_aux = _prediction_with_plan(
                model,
                encoded,
                batch,
                first_selected_plan,
                return_interaction_aux=True,
            )
            _first_selected_loss, _first_selected_mse, first_selected_rmse = _role_objective(
                first_selected_prediction,
                batch.target_field,
                sample.role_slices,
                normalizer,
                scales,
            )
            first_query_routes = {
                mechanism: _canonical_route_support(first_selected_plan, encoded, tree, mechanism)
                for mechanism in all_mechanisms
            }
            first_query_routes["QE"]["sampled_live_query_panel"] = _query_qe_support(
                first_selected_plan,
                encoded,
                sample,
                batch,
            )
            first_query_record = {
                "query_stream_id": 733,
                "selected_cut": None if selected_cut is None else list(selected_cut),
                "unsupported_at_budget": selected_index is None,
                "explicit_full_access_fallback": first_selection_fallback,
                "role_rmse_mps": first_selected_rmse,
                "route_support": first_query_routes,
                "native_executor_rows": _summarize_executor_rows(first_aux),
            }

            # The selected cut is fixed by the first measured panel. A second
            # independent sample checks it without reselecting on held target
            # values. Repeated full passes establish an m/s allowance on that
            # exact second panel.
            second_sample = sample_native_role_queries(
                case,
                _native_rng(seed, 0, int(row), 734),
                query_counts,
                catalogue_cache=sampler,
            )
            second_batch = _batch_from_sample(case, second_sample, normalizer, device)
            with torch.no_grad():
                second_encoded, second_full_policy = _full_access_prediction(model, second_batch)
                second_trees = model.core.backend.build_case_trees(second_encoded)
                second_tree = second_trees[0]
                second_full_plan = MechanismPlan.full_access(
                    second_tree,
                    second_encoded.module_present[0],
                    int(second_encoded.env_coords.shape[1]),
                )
                second_full_plan_prediction = _prediction_with_plan(
                    model,
                    second_encoded,
                    second_batch,
                    second_full_plan,
                )
                second_prepared_full = model.core.prepare(
                    second_encoded,
                    second_encoded.module_tokens,
                    fixed_cover_plans=(second_full_plan,),
                )
                second_alternate_chunk = max(1, chunk_size // 2)
                second_alternate_prediction = model.core.decode_queries(
                    second_prepared_full,
                    second_batch.query_xy,
                    query_features=second_batch.query_features,
                    receiver_chunk_size=second_alternate_chunk,
                )["pred_field"]
                _second_full_loss, _second_full_mse, second_full_rmse = _role_objective(
                    second_full_policy,
                    second_batch.target_field,
                    second_sample.role_slices,
                    normalizer,
                    scales,
                )
                second_repeats = torch.stack(
                    (
                        second_full_policy - second_full_plan_prediction,
                        second_full_plan_prediction - second_alternate_prediction,
                        second_full_policy - second_alternate_prediction,
                    ),
                    dim=0,
                )
                second_repeat_mps = _standardized_delta_to_mps(second_repeats, normalizer)
                second_allowance = {
                    role: float(
                        second_repeat_mps[:, 0, second_sample.role_slices[role], :]
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    for role in ROLE_NAMES
                }
                second_prepared_state = {
                    "module_states": second_encoded.module_tokens,
                    "environment_states": second_encoded.env_tokens,
                    "global_state": second_encoded.global_token,
                }
                second_scores = organizer.score_cases(
                    second_encoded,
                    second_prepared_state,
                    second_trees,
                    budgets=route_budgets,
                )
                second_plan = (
                    second_full_plan
                    if selected_cut is None
                    else organizer.plans_from_scores(
                        second_scores,
                        second_encoded,
                        second_trees,
                        hard=True,
                        frontier_cuts=(selected_cut,),
                        budget_fractions=plan_budgets,
                    )[0]
                )
                second_selected_prediction, second_aux = _prediction_with_plan(
                    model,
                    second_encoded,
                    second_batch,
                    second_plan,
                    return_interaction_aux=True,
                )
                _second_selected_loss, _second_selected_mse, second_selected_rmse = _role_objective(
                    second_selected_prediction,
                    second_batch.target_field,
                    second_sample.role_slices,
                    normalizer,
                    scales,
                )
            second_adequate = {
                role: bool(
                    second_selected_rmse[role]
                    <= float(stage_cfg["adequacy_reference_ratio"]) * second_full_rmse[role]
                    + second_allowance[role]
                )
                for role in ROLE_NAMES
            }
            second_query_routes = {
                mechanism: _canonical_route_support(second_plan, second_encoded, second_tree, mechanism)
                for mechanism in all_mechanisms
            }
            second_query_routes["QE"]["sampled_live_query_panel"] = _query_qe_support(
                second_plan,
                second_encoded,
                second_sample,
                second_batch,
            )
            second_query_record = {
                "query_stream_id": 734,
                "selection_locked_to_first_panel": True,
                "selected_cut": None if selected_cut is None else list(selected_cut),
                "unsupported_at_budget": selected_index is None,
                "explicit_full_access_fallback": selected_cut is None,
                "same_student_full_role_rmse_mps": second_full_rmse,
                "selected_role_rmse_mps": second_selected_rmse,
                "numerical_allowance_mps": second_allowance,
                "adequate_by_role": second_adequate,
                "full_access_fallback_adequate": (
                    all(second_adequate.values()) if selected_cut is None else None
                ),
                "selected_sparse_cut_adequate": (
                    all(second_adequate.values()) if selected_cut is not None else False
                ),
                "adequate": bool(selected_cut is not None and all(second_adequate.values())),
                "route_support": second_query_routes,
                "native_executor_rows": _summarize_executor_rows(second_aux),
            }
            if selected_cut is None:
                least_risk_cut = tuple(cuts[least_risk_index])
                least_risk_plan = organizer.plans_from_scores(
                    second_scores,
                    second_encoded,
                    second_trees,
                    hard=True,
                    frontier_cuts=(least_risk_cut,),
                    budget_fractions=plan_budgets,
                )[0]
                least_risk_prediction, least_risk_aux = _prediction_with_plan(
                    model,
                    second_encoded,
                    second_batch,
                    least_risk_plan,
                    return_interaction_aux=True,
                )
                _least_risk_loss, _least_risk_mse, least_risk_rmse = _role_objective(
                    least_risk_prediction,
                    second_batch.target_field,
                    second_sample.role_slices,
                    normalizer,
                    scales,
                )
                least_risk_adequate = {
                    role: bool(
                        least_risk_rmse[role]
                        <= float(stage_cfg["adequacy_reference_ratio"]) * second_full_rmse[role]
                        + second_allowance[role]
                    )
                    for role in ROLE_NAMES
                }
                second_query_record["least_risk_research_only"] = {
                    "cut": list(least_risk_cut),
                    "role_rmse_mps": least_risk_rmse,
                    "adequate_by_role": least_risk_adequate,
                    "adequate": all(least_risk_adequate.values()),
                    "never_deployment_or_sparse_success": True,
                    "route_support": {
                        mechanism: _canonical_route_support(
                            least_risk_plan,
                            second_encoded,
                            second_tree,
                            mechanism,
                        )
                        for mechanism in all_mechanisms
                    },
                    "sampled_live_qe_query_panel": _query_qe_support(
                        least_risk_plan,
                        second_encoded,
                        second_sample,
                        second_batch,
                    ),
                    "native_executor_rows": _summarize_executor_rows(least_risk_aux),
                }
            result_rows.append(
                {
                    "row": int(row),
                    "layout_index": int(layout_indices[row]),
                    "wind_direction_deg": float(case.wind_direction_deg),
                    "query_seed": seed,
                    "role_query_counts": query_counts,
                    "source_reference_role_rmse_mps": reference_rmse[int(row)],
                    "same_student_full_role_rmse_mps": full_rmse,
                    "numerical_allowance_mps": numerical_allowance,
                    "numerical_allowance_method": stage_cfg["numerical_allowance"],
                    "full_access_parity_chunk_sizes": [chunk_size, alternate_chunk],
                    "candidate_count": len(candidate_rows),
                    "candidate_table": candidate_rows,
                    "pareto_candidate_indices": pareto_indices,
                    "unsupported_at_budget": selected_index is None,
                    "selected_measured_index": selected_index,
                    "least_risk_index": least_risk_index,
                    "least_risk_is_not_deployment_selection": True,
                    "selected_cut_first_query_panel": first_query_record,
                    "selected_cut_second_query_panel": second_query_record,
                }
            )
    candidate_prediction_calls = int(sum(item["candidate_count"] for item in result_rows))
    unsupported_rows = sum(bool(item["unsupported_at_budget"]) for item in result_rows)
    native_forward_calls = {
        "retained_w_full_source_reference": len(result_rows),
        "same_student_full_access_policy": 2 * len(result_rows),
        "same_student_explicit_full_plan": 2 * len(result_rows),
        "same_student_alternate_chunk_parity": 2 * len(result_rows),
        "all_frontier_candidate_predictions_first_panel": candidate_prediction_calls,
        "selected_first_panel_replay": len(result_rows),
        "selected_second_panel_replay": len(result_rows),
        "least_risk_second_panel_research_only": unsupported_rows,
    }
    total_native_forward_calls = int(sum(native_forward_calls.values()))
    if total_native_forward_calls > 2048:
        raise ValueError(
            f"Stage-B complete native prediction calls {total_native_forward_calls} exceed the 2048 train-only cap"
        )
    table = {
        "schema_version": 1,
        "run_id": "2111",
        "stage": "B_measured_frontier_fidelity_work",
        "training_only": True,
        "validation_or_test_rows_opened": False,
        "new_cfd_solves": 0,
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "source_checkpoint": str(resolve_path(str(config["source_checkpoint"])).resolve()),
        "stage_a_checkpoint": str(stage_a_checkpoint),
        "stage_a_checkpoint_sha256": checkpoint_sha,
        "stage_a_update_count": int(checkpoint["update_count"]),
        "config_path": str(DEFAULT_CONFIG.resolve()),
        "config_sha256_for_stage_b": _file_sha256(DEFAULT_CONFIG),
        "dataset_manifest": str(resolve_path(str(config["dataset"]["manifest"])).resolve()),
        "dataset_manifest_sha256": _file_sha256(resolve_path(str(config["dataset"]["manifest"]))),
        "train_rows_sha256": _rows_sha256(train_rows),
        "secondary_route": secondary_route,
        "budget_fraction": float(stage_cfg["budget_fraction"]),
        "frontier_max_depth": 3,
        "candidate_evaluations": candidate_prediction_calls,
        "candidate_evaluation_cap": 2048,
        "native_forward_call_counts": native_forward_calls,
        "total_complete_native_forward_calls": total_native_forward_calls,
        "total_complete_native_forward_call_cap": 2048,
        "native_forward_call_cap_passed": total_native_forward_calls <= 2048,
        "initial_training_layout_count": len(selected_layouts),
        "selected_layouts": selected_layouts,
        "selected_rows": selected_rows,
        "adequacy": "per-role candidate RMSE <= 1.10 * same-student full RMSE + repeated-parity allowance in m/s",
        "candidate_rows": result_rows,
    }
    table_path = (run_dir / "stage_b_frontier_table.json") if output_path is None else output_path
    _write_json(table_path, table)
    return table


def _stage_c_fit(
    *,
    config: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    view: WindFarmNativeView,
    train_rows: Sequence[int],
    run_dir: Path,
    stage_b_path: Path,
    seed: int,
    requested_updates: int,
) -> dict[str, Any]:
    """Fit the input-only case-dependent frontier utility head from Stage B."""

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
        raise RuntimeError("WindFarm Stage C requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("authorized physical GPU 0 is unavailable for Stage C")
    if not stage_b_path.is_file():
        raise FileNotFoundError(f"Stage-B measured table does not exist: {stage_b_path}")
    table = json.loads(stage_b_path.read_text(encoding="utf-8"))
    if (
        table.get("run_id") != "2111"
        or table.get("stage") != "B_measured_frontier_fidelity_work"
        or table.get("training_only") is not True
        or table.get("validation_or_test_rows_opened") is not False
        or table.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256
        or table.get("train_rows_sha256") != _rows_sha256(train_rows)
    ):
        raise ValueError("Stage-B table is not a matching train-only Run2111 artifact")
    update_cap = int(config["forward"]["stage_c"]["maximum_frontier_supervision_updates"])
    if requested_updates < 1 or requested_updates > update_cap:
        raise ValueError(f"Stage-C utility updates must be between 1 and the configured cap {update_cap}")
    stage_a_checkpoint = Path(table["stage_a_checkpoint"]).resolve()
    if _file_sha256(stage_a_checkpoint) != table.get("stage_a_checkpoint_sha256"):
        raise ValueError("Stage-A checkpoint changed after Stage-B measurement")
    stage_a = load_trusted_checkpoint(stage_a_checkpoint, map_location="cpu")
    if not isinstance(stage_a, Mapping) or stage_a.get("arm") != "g_packet":
        raise ValueError("Stage-C fitting requires its exact Stage-A G checkpoint")
    secondary_route = str(table["secondary_route"])
    if secondary_route not in {"MM", "EM"}:
        raise ValueError("Stage-B table has an invalid secondary route")

    device = torch.device("cuda:0")
    rows = [int(value) for value in table["selected_rows"]]
    if not rows or any(row not in train_rows for row in rows):
        raise ValueError("Stage-C supervision rows must all belong to the post-exclusion training split")
    query_counts = {str(k): int(v) for k, v in table["candidate_rows"][0]["role_query_counts"].items()}
    sampler = NativeRoleCatalogueCache()
    first_case = view.run(rows[0])
    first_sample = sample_native_role_queries(
        first_case,
        _native_rng(seed, 0, rows[0], 733),
        query_counts,
        catalogue_cache=sampler,
    )
    first_batch = _batch_from_sample(first_case, first_sample, normalizer, device)
    model = _new_model_from_source(source_payload, normalizer, first_batch, device)
    model.load_state_dict(stage_a["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.core.backend.set_cover_mode("external")
    with torch.no_grad():
        encoded = model.core.encode_case(first_batch)
        tree = model.core.backend.build_case_trees(encoded)[0]
    base_organizer = _build_organizer(
        model.core,
        encoded,
        tree,
        config["forward"]["stage_a"],
        device,
    )
    base_organizer.load_state_dict(stage_a["organizer_state_dict"], strict=True)
    utility_organizer = InputOnlyCoverOrganizer(
        state_dim=base_organizer.state_dim,
        module_feature_dim=base_organizer.module_feature_dim,
        environment_feature_dim=base_organizer.environment_feature_dim,
        hidden_dim=int(base_organizer.node_encoder[0].out_features),
        role_count=base_organizer.role_count,
        source_chunk_size=base_organizer.source_chunk_size,
        split_threshold=base_organizer.split_threshold,
        source_threshold=base_organizer.source_threshold,
        quadrature_invariant_source_measure=base_organizer.quadrature_invariant_source_measure,
        frontier_utility_enabled=True,
        frontier_role_count=len(ROLE_NAMES),
    ).to(device)
    missing, unexpected = utility_organizer.load_state_dict(base_organizer.state_dict(), strict=False)
    expected_missing = {key for key in utility_organizer.state_dict() if key.startswith("frontier_utility_head.")}
    if set(missing) != expected_missing or unexpected:
        raise ValueError("Stage-C utility organizer did not preserve the complete frozen Stage-A scorer")
    for parameter in utility_organizer.parameters():
        parameter.requires_grad_(False)
    utility_head = utility_organizer.frontier_utility_head
    if utility_head is None:
        raise RuntimeError("Stage-C utility head was not constructed")
    for parameter in utility_head.parameters():
        parameter.requires_grad_(True)
    utility_organizer.train()

    rows_by_id = {int(item["row"]): item for item in table["candidate_rows"]}
    if set(rows_by_id) != set(rows):
        raise ValueError("Stage-B row/candidate table does not match the declared selected rows")
    labels: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]] = {}
    for row in rows:
        item = rows_by_id[row]
        candidates = item["candidate_table"]
        cuts = tuple(tuple(map(int, candidate["frontier_cut"])) for candidate in candidates)
        if len(cuts) != len(item["candidate_table"]) or not cuts:
            raise ValueError("Stage-B candidate table contains invalid frontier cuts")
        candidate_errors = torch.tensor(
            [[float(candidate["role_rmse_mps"][role]) for role in ROLE_NAMES] for candidate in candidates],
            dtype=torch.float32,
            device=device,
        )
        full_errors = torch.tensor(
            [float(item["same_student_full_role_rmse_mps"][role]) for role in ROLE_NAMES],
            dtype=torch.float32,
            device=device,
        )
        # The measured deployment gate is one-sided: improvements over full
        # access are acceptable, while only positive degradation consumes the
        # dimensionless role allowance. Keep Stage-C labels algebraically
        # aligned with Stage-B's 1.10 * full + measured numerical allowance.
        distortion_targets = (candidate_errors - full_errors[None, :]).clamp_min(0.0) / full_errors[
            None, :
        ].clamp_min(1.0e-8)
        work_targets = normalized_frontier_work_targets(
            torch.tensor(
                [float(candidate["canonical_total_work"]) for candidate in candidates],
                dtype=torch.float32,
                device=device,
            ),
            float(item["candidate_table"][0]["canonical_full_access_work"]),
        )
        packet_counts = torch.tensor(
            [int(candidate["nonredundant_packet_count"]) for candidate in candidates],
            dtype=torch.long,
            device=device,
        )
        labels[row] = (distortion_targets, work_targets, full_errors, packet_counts, cuts)

    route_budgets, plan_budgets = _budgets(
        selected_route=secondary_route,
        fraction=float(table["budget_fraction"]),
    )
    scales = config["forward"]["stage_a"]["role_loss_scales_mps"]
    adequacy_ratio_excess = float(config["forward"]["stage_b"]["adequacy_reference_ratio"]) - 1.0
    row_role_tolerances: dict[int, torch.Tensor] = {}
    for row in rows:
        full_values = torch.tensor(
            [float(rows_by_id[row]["same_student_full_role_rmse_mps"][role]) for role in ROLE_NAMES],
            device=device,
            dtype=torch.float32,
        )
        allowance_values = torch.tensor(
            [float(rows_by_id[row]["numerical_allowance_mps"][role]) for role in ROLE_NAMES],
            device=device,
            dtype=torch.float32,
        )
        row_role_tolerances[row] = adequacy_ratio_excess + allowance_values / full_values.clamp_min(1.0e-8)
    global_role_tolerance = torch.stack([row_role_tolerances[row] for row in rows], dim=0).median(dim=0).values
    if not bool(torch.isfinite(global_role_tolerance).all()) or bool((global_role_tolerance < 0).any()):
        raise ValueError("Stage-C train-only role tolerances must be finite and nonnegative")
    optimizer = torch.optim.AdamW(utility_head.parameters(), lr=2.0e-4, weight_decay=1.0e-5)
    updates_path = run_dir / "stage_c_utility_updates.jsonl"
    update_rows = sorted(rows)
    losses: list[float] = []
    for update in range(1, requested_updates + 1):
        row = update_rows[(update - 1) % len(update_rows)]
        case = view.run(row)
        sample = sample_native_role_queries(
            case,
            _native_rng(seed, 0, row, 733),
            query_counts,
            catalogue_cache=sampler,
        )
        batch = _batch_from_sample(case, sample, normalizer, device)
        model.core.backend.set_cover_mode("external")
        with torch.no_grad():
            current_encoded = model.core.encode_case(batch)
            trees = model.core.backend.build_case_trees(current_encoded)
        prepared_state = {
            "module_states": current_encoded.module_tokens,
            "environment_states": current_encoded.env_tokens,
            "global_state": current_encoded.global_token,
        }
        scores = utility_organizer.score_cases(
            current_encoded,
            prepared_state,
            trees,
            budgets=route_budgets,
        )
        targets, work_targets, _full_errors, _packet_counts, cuts = labels[row]
        prediction = utility_organizer.score_frontiers(scores[0], trees[0], cuts=cuts)
        distortion_loss = (prediction.role_distortion - targets).square().mean()
        work_loss = (prediction.predicted_work_fraction - work_targets).square().mean()
        loss = distortion_loss + work_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Run2111 Stage-C utility loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(utility_head.parameters(), max_norm=1.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        _jsonl(
            updates_path,
            {
                "update": update,
                "row": row,
                "layout_index": int(view.metadata["layout_index"][row]),
                "distortion_mse": float(distortion_loss.detach().cpu()),
                "work_fraction_mse": float(work_loss.detach().cpu()),
                "loss": loss_value,
                "optimizer_calls": update,
                "training_only": True,
            },
        )

    # Apply the learned selector to the same train-only rows for auditable fit
    # diagnostics. Unsupported predictions remain separate from the research
    # least-risk frontier and are not treated as sparse deployment success.
    selection_rows: list[dict[str, Any]] = []
    utility_organizer.eval()
    for row in update_rows:
        case = view.run(row)
        sample = sample_native_role_queries(
            case,
            _native_rng(seed, 0, row, 733),
            query_counts,
            catalogue_cache=sampler,
        )
        batch = _batch_from_sample(case, sample, normalizer, device)
        with torch.no_grad():
            current_encoded = model.core.encode_case(batch)
            trees = model.core.backend.build_case_trees(current_encoded)
            prepared_state = {
                "module_states": current_encoded.module_tokens,
                "environment_states": current_encoded.env_tokens,
                "global_state": current_encoded.global_token,
            }
            scores = utility_organizer.score_cases(
                current_encoded,
                prepared_state,
                trees,
                budgets=route_budgets,
            )
            _distortion_targets, _work_targets, full_errors, packet_counts, cuts = labels[row]
            prediction = utility_organizer.score_frontiers(scores[0], trees[0], cuts=cuts)
            tolerance = row_role_tolerances[row].to(dtype=prediction.role_distortion.dtype)
            selection = select_frontier_by_predictions(
                prediction,
                role_tolerance=tolerance,
                packet_counts=packet_counts,
            )
            selected = selection.selected_frontier
            selection_rows.append(
                {
                    "row": row,
                    "layout_index": int(view.metadata["layout_index"][row]),
                    "selected_frontier": None if selected is None else list(selected),
                    "selected_index": selection.selected_index,
                    "unsupported_at_budget": selection.unsupported_at_budget,
                    "least_risk_frontier": list(selection.least_risk_frontier),
                    "least_risk_is_research_fallback_only": True,
                    "role_tolerance_dimensionless": [float(value) for value in tolerance.detach().cpu()],
                    "role_tolerance_source": "0.10 + measured per-role second-panel numerical allowance_mps / same-student full-access RMSE_mps; train-only Stage-B rows",
                    "predicted_role_positive_excess": (
                        []
                        if selection.selected_index is None
                        else [
                            float(value)
                            for value in prediction.role_distortion[selection.selected_index].detach().cpu()
                        ]
                    ),
                    "selected_predicted_work_fraction": (
                        None
                        if selection.selected_index is None
                        else float(prediction.predicted_work_fraction[selection.selected_index].cpu())
                    ),
                }
            )
    stage_c_cfg = config["forward"]["stage_c"]
    refinement_updates = int(stage_c_cfg["selected_hard_cut_joint_refinement_updates"])
    replay_every = int(stage_c_cfg["full_access_replay_every"])
    if refinement_updates < 1 or replay_every < 1:
        raise ValueError("Stage-C selected-cut refinement and full-replay cadence must be positive")
    if refinement_updates > int(config["forward"]["max_physical_updates_per_arm"]):
        raise ValueError("Stage-C physical refinement exceeds the per-arm physical-update cap")
    model_hash_before_refinement = _model_state_sha256(model)
    frozen_organizer_state = {
        key: value.detach().clone()
        for key, value in utility_organizer.state_dict().items()
        if not key.startswith("pair_scorers.QE.")
        and not key.startswith(f"pair_scorers.{secondary_route}.")
    }
    for parameter in utility_organizer.parameters():
        parameter.requires_grad_(False)
    for route in ("QE", secondary_route):
        for parameter in utility_organizer.pair_scorers[route].parameters():
            parameter.requires_grad_(True)
    utility_organizer.eval()
    model.core.backend.set_cover_executor("dense_masked")
    model.core.backend.set_cover_mode("external")
    model.train()
    physical_parameters = _enable_physical_trainable_scope(model)
    permission_parameters = [
        parameter
        for route in ("QE", secondary_route)
        for parameter in utility_organizer.pair_scorers[route].parameters()
    ]
    refinement_optimizer = torch.optim.AdamW(
        [
            {
                "params": physical_parameters,
                "lr": float(stage_c_cfg["physical_learning_rate"]),
                "weight_decay": 1.0e-5,
            },
            {
                "params": permission_parameters,
                "lr": float(stage_c_cfg["permission_learning_rate"]),
                "weight_decay": 1.0e-5,
            },
        ]
    )
    refinement_path = run_dir / "stage_c_refinement_updates.jsonl"
    refinement_physical_calls = 0
    refinement_permission_calls = 0
    refinement_selected_cut_calls = 0
    refinement_research_cut_calls = 0
    refinement_predicted_adequate_calls = 0
    refinement_replay_calls = 0
    refinement_unsupported_fallback_calls = 0
    for update in range(1, refinement_updates + 1):
        row = update_rows[(update - 1) % len(update_rows)]
        case = view.run(row)
        sample = sample_native_role_queries(
            case,
            _native_rng(seed, update, row, 735),
            query_counts,
            catalogue_cache=sampler,
        )
        batch = _batch_from_sample(case, sample, normalizer, device)
        model.core.backend.set_cover_mode("external")
        with torch.no_grad():
            selector_encoded = model.core.encode_case(batch)
            selector_trees = model.core.backend.build_case_trees(selector_encoded)
            selector_state = {
                "module_states": selector_encoded.module_tokens,
                "environment_states": selector_encoded.env_tokens,
                "global_state": selector_encoded.global_token,
            }
            selector_scores = utility_organizer.score_cases(
                selector_encoded,
                selector_state,
                selector_trees,
                budgets=route_budgets,
            )
            _targets, _work_targets, _full_errors, packet_counts, cuts = labels[row]
            selector_prediction = utility_organizer.score_frontiers(
                selector_scores[0], selector_trees[0], cuts=cuts
            )
            frontier_selection = select_frontier_by_predictions(
                selector_prediction,
                role_tolerance=row_role_tolerances[row],
                packet_counts=packet_counts,
            )
            selected_cut = frontier_selection.selected_frontier
            research_cut = frontier_selection.least_risk_frontier
            refinement_cut = selected_cut if selected_cut is not None else research_cut
            refinement_predicted_adequate_calls += int(selected_cut is not None)

        scheduled_replay = update % replay_every == 0
        if scheduled_replay:
            _encoded, prediction = _full_access_prediction(model, batch)
            update_kind = "full_access_replay"
            selected_hard_plan = None
            refinement_replay_calls += int(scheduled_replay)
            permission_gradient_norms = {route: 0.0 for route in ("QE", secondary_route)}
            executor_rows: dict[str, Any] = {}
            route_work: dict[str, Any] = {
                route: {
                    "permission_status": "native_full_access",
                    "eligible_unique_pairs": None,
                    "actual_live_unique_pairs": None,
                    "sparse_success": False,
                }
                for route in ("QE", secondary_route)
            }
        else:
            model.core.backend.set_cover_mode("external")
            encoded = model.core.encode_case(batch)
            _set_update_seed(seed, 10_000 + update)
            collect_aux = update == 1 or update % 10 == 0 or update == refinement_updates
            joint_result = hard_value_soft_organizer_forward(
                model.core,
                encoded,
                encoded.module_tokens,
                utility_organizer,
                batch.query_xy,
                batch.query_features,
                receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
                collect_hard_aux=collect_aux,
                budgets=route_budgets,
                frontier_cuts=(refinement_cut,),
                budget_fractions=plan_budgets,
            )
            prediction = joint_result.prediction
            selected_hard_plan = joint_result.hard_plans[0]
            update_kind = (
                "selected_hard_cut_joint_refinement"
                if selected_cut is not None
                else "least_risk_budget_feasible_research_refinement"
            )
            executor_rows = _summarize_executor_rows(joint_result.hard_interaction_aux)
            route_work = {
                route: _canonical_route_support(
                    selected_hard_plan,
                    encoded,
                    selected_hard_plan.tree,
                    route,
                )
                for route in ("QE", secondary_route)
            }
            route_work["QE"]["sampled_live_query_panel"] = _query_qe_support(
                selected_hard_plan,
                encoded,
                sample,
                batch,
            )
            if selected_cut is not None:
                refinement_selected_cut_calls += 1
            else:
                refinement_research_cut_calls += 1

        target = batch.target_field
        if target is None:
            raise RuntimeError("Stage-C refinement sample has no native velocity target")
        loss, role_mse, role_rmse = _role_objective(prediction, target, sample.role_slices, normalizer, scales)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Stage-C joint-refinement loss at update {update}")
        refinement_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(physical_parameters + permission_parameters, max_norm=1.0)
        if selected_hard_plan is not None:
            permission_gradient_norms = {
                route: _gradient_group_l2(tuple(utility_organizer.pair_scorers[route].parameters()))
                for route in ("QE", secondary_route)
            }
        refinement_optimizer.step()
        refinement_physical_calls += 1
        if selected_hard_plan is not None:
            refinement_permission_calls += 1
        _jsonl(
            refinement_path,
            {
                "update": update,
                "row": row,
                "layout_index": int(view.metadata["layout_index"][row]),
                "update_kind": update_kind,
                "selected_frontier": None if selected_cut is None else list(selected_cut),
                "refinement_frontier": None if scheduled_replay else list(refinement_cut),
                "refinement_frontier_is_least_risk_research_only": bool(
                    not scheduled_replay and selected_cut is None
                ),
                "selected_index": frontier_selection.selected_index,
                "unsupported_at_budget": frontier_selection.unsupported_at_budget,
                "least_risk_frontier_research_only": list(frontier_selection.least_risk_frontier),
                "objective_role_normalized_mse_mean": float(loss.detach().cpu()),
                "role_mse_mps2": role_mse,
                "role_rmse_mps": role_rmse,
                "physical_optimizer_calls": refinement_physical_calls,
                "permission_optimizer_calls": refinement_permission_calls,
                "permission_gradient_l2": permission_gradient_norms,
                "route_work": route_work,
                "native_executor_rows": executor_rows,
                "training_only": True,
            },
        )
        if update % 10 == 0 or update == 1:
            print(
                f"Stage-C refine update={update}/{refinement_updates} kind={update_kind} "
                f"loss={float(loss.detach().cpu()):.5g}",
                flush=True,
            )

    for key, prior_value in frozen_organizer_state.items():
        current_value = utility_organizer.state_dict()[key].detach()
        if not torch.equal(prior_value.to(device=current_value.device), current_value):
            raise RuntimeError(f"Stage-C refinement unexpectedly modified frozen organizer component {key}")
    utility_organizer.eval()
    for parameter in utility_organizer.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model_hash_after_refinement = _model_state_sha256(model)
    frozen_checkpoint = {
        "schema_version": 1,
        "run_id": "2111",
        "stage": "C_frozen_selected_forward_organizer",
        "source_checkpoint": str(config["source_checkpoint"]),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_update_count": 1500,
        "stage_a_checkpoint": str(stage_a_checkpoint),
        "stage_a_checkpoint_sha256": table["stage_a_checkpoint_sha256"],
        "stage_b_table": str(stage_b_path.resolve()),
        "stage_b_table_sha256": _file_sha256(stage_b_path),
        "train_rows_sha256": _rows_sha256(train_rows),
        "secondary_route": secondary_route,
        "budget_fraction": float(table["budget_fraction"]),
        "frontier_role_count": len(ROLE_NAMES),
        "role_tolerance": [float(value) for value in global_role_tolerance.detach().cpu()],
        "role_tolerance_source": "per-role median of train-only Stage-B row tolerances: (adequacy_reference_ratio - 1) + measured numerical allowance_mps / same-student full-access RMSE_mps",
        "role_tolerance_rows": {
            str(row): [float(value) for value in row_role_tolerances[row].detach().cpu()]
            for row in rows
        },
        "role_tolerance_units": "dimensionless positive RMSE excess relative to same-student full-access role RMSE",
        "role_tolerance_validation_scope": "train-only Stage-B panels; no validation/test outcomes opened",
        "utility_updates": requested_updates,
        "utility_loss_first": losses[0],
        "utility_loss_last": losses[-1],
        "model_state_sha256_before_refinement": model_hash_before_refinement,
        "model_state_sha256": model_hash_after_refinement,
        "model_state_dict": model.state_dict(),
        "organizer_state_dict": utility_organizer.state_dict(),
        "selected_training_rows": selection_rows,
        "unsupported_is_explicit": True,
        "validation_or_test_rows_opened_for_supervision": False,
        "refinement_updates": refinement_updates,
        "refinement_replay_every": replay_every,
        "refinement_full_access_replay_calls": refinement_replay_calls,
        "refinement_unsupported_full_access_fallback_calls": refinement_unsupported_fallback_calls,
        "refinement_selected_hard_cut_calls": refinement_selected_cut_calls,
        "refinement_least_risk_research_cut_calls": refinement_research_cut_calls,
        "refinement_predicted_adequate_calls": refinement_predicted_adequate_calls,
        "physical_optimizer_calls_in_stage_c": refinement_physical_calls,
        "permission_optimizer_calls_in_stage_c": refinement_permission_calls,
        "utility_optimizer_calls": requested_updates,
        "utility_head_frozen_during_refinement": True,
        "frozen_node_source_and_split_organizer_state_verified": True,
    }
    frozen_path = run_dir / "stage_c_frozen_forward_organizer.pt"
    _atomic_checkpoint(frozen_path, frozen_checkpoint)
    result = {
        "stage": "C_frontier_utility_fit_and_frozen_organizer",
        "updates": requested_updates,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "unsupported_training_rows": sum(bool(item["unsupported_at_budget"]) for item in selection_rows),
        "training_rows": len(selection_rows),
        "frozen_checkpoint": str(frozen_path),
        "frozen_checkpoint_sha256": _file_sha256(frozen_path),
        "model_state_sha256_before_refinement": model_hash_before_refinement,
        "model_state_sha256": model_hash_after_refinement,
        "secondary_route": secondary_route,
        "role_tolerance": [float(value) for value in global_role_tolerance.detach().cpu()],
        "role_tolerance_source": "train-only Stage-B per-role median tolerance",
        "refinement_updates": refinement_updates,
        "refinement_selected_hard_cut_calls": refinement_selected_cut_calls,
        "refinement_least_risk_research_cut_calls": refinement_research_cut_calls,
        "refinement_predicted_adequate_calls": refinement_predicted_adequate_calls,
        "refinement_full_access_replay_calls": refinement_replay_calls,
        "refinement_unsupported_full_access_fallback_calls": refinement_unsupported_fallback_calls,
        "physical_optimizer_calls_in_stage_c": refinement_physical_calls,
        "permission_optimizer_calls_in_stage_c": refinement_permission_calls,
        "validation_or_test_rows_opened": False,
    }
    _write_json(run_dir / "stage_c_result.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("stage_a", "stage_b", "stage_c"), default="stage_a")
    parser.add_argument("--arm", choices=("g_packet", "direct_pair"), default="g_packet")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2111)
    parser.add_argument(
        "--secondary-route",
        choices=("MM", "EM"),
        default=None,
        help="optional assertion against the train-only omission/restoration pilot",
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=None,
        help="fixed b for one-update smoke; omit for the bounded curriculum",
    )
    parser.add_argument("--run-dir", type=Path, help="reuse the exact Run2111 directory for a matched second arm")
    parser.add_argument("--stage-a-checkpoint", type=Path, help="G packet checkpoint for the train-only Stage-B table")
    parser.add_argument(
        "--stage-b-output",
        type=Path,
        help="write a new measured Stage-B table at this path instead of replacing Run2111's default table",
    )
    parser.add_argument("--stage-b-table", type=Path, help="measured train-only Stage-B table for Stage C")
    parser.add_argument("--stage-c-updates", type=int, default=400, help="bounded frontier-utility optimizer updates")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume this arm from its exact Run2111 review checkpoint; --updates is the cumulative target",
    )
    parser.add_argument("--max-seconds", type=int, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify source, mmap view, exclusions, and device mapping without training",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="require exactly one update; use --budget-fraction 0.90 for a sparse smoke"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.smoke and args.updates != 1:
        raise ValueError("--smoke is exactly one optimizer update")
    if args.budget_fraction is not None and not 0.0 <= args.budget_fraction <= 1.0:
        raise ValueError("--budget-fraction must be in [0,1]")
    if args.resume_checkpoint is not None and args.run_dir is None:
        raise ValueError("--resume-checkpoint requires --run-dir to retain the original Run2111 identity")
    if args.stage in {"stage_b", "stage_c"} and args.run_dir is None:
        raise ValueError("Stage B/C must extend the existing Run2111 identity via --run-dir")
    if args.stage != "stage_b" and args.stage_b_output is not None:
        raise ValueError("--stage-b-output is only valid for a Stage-B measurement")
    view, _split, train_rows, split_record = _native_inputs(config)
    checkpoint, payload, normalizer, source_sha = _load_source(config)
    if source_sha != str(config["source_checkpoint_sha256"]):
        raise ValueError("source identity changed after config validation")
    if args.preflight_only:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        record = {
            "run_id": "2111",
            "source_checkpoint": str(checkpoint),
            "source_sha256": source_sha,
            "source_update_count": int(payload["update_count"]),
            "source_arm": str(payload["arm"]),
            "split": split_record,
            "normalization": normalizer.to_dict(),
            "cuda_visible_devices": visible,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    if args.run_dir is None:
        run_dir = _new_run_directory(config, smoke=bool(args.smoke))
        _write_json(
            run_dir / "run_identity.json",
            {
                "run_id": "2111",
                "name": "active_budgeted_packet_reuse",
                "config": str(args.config.resolve()),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "source_checkpoint": str(checkpoint),
                "source_sha256": source_sha,
                "source_update_count": 1500,
                "source_arm": "w_full",
                "dataset_split": split_record,
                "zero_new_cfd": True,
                "physical_gpu": 0,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    else:
        run_dir = args.run_dir.resolve()
        identity_path = run_dir / "run_identity.json"
        if not identity_path.is_file():
            raise FileNotFoundError("matched Run2111 directory lacks its identity record")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("run_id") != "2111" or identity.get("source_sha256") != source_sha:
            raise ValueError("matched-arm output directory has a different run/source identity")
    pilot_path = run_dir / "secondary_route_pilot.json"
    if pilot_path.is_file():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if (
            pilot.get("source_checkpoint_sha256") != source_sha
            or pilot.get("training_only") is not True
            or pilot.get("selected_route") not in {"MM", "EM"}
        ):
            raise ValueError("Run2111 secondary-route pilot identity is invalid")
        if pilot.get("train_rows_sha256") != split_record["student_train_rows_sha256"]:
            raise ValueError("Run2111 route pilot was measured on another training-row set")
    else:
        pilot = _secondary_route_pilot(
            config=config,
            source_payload=payload,
            normalizer=normalizer,
            view=view,
            train_rows=train_rows,
            seed=int(args.seed),
            run_dir=run_dir,
        )
        pilot["train_rows_sha256"] = split_record["student_train_rows_sha256"]
        _write_json(pilot_path, pilot)
    secondary_route = str(pilot["selected_route"])
    if args.secondary_route is not None and args.secondary_route != secondary_route:
        raise ValueError(
            f"caller route {args.secondary_route} disagrees with training-only pilot choice {secondary_route}"
        )
    if args.stage == "stage_b":
        stage_a_checkpoint = args.stage_a_checkpoint
        if stage_a_checkpoint is None:
            stage_a_checkpoint = run_dir / "arms" / "g_packet" / "checkpoints" / "updates_000100.pt"
        stage_b_output = (
            run_dir / "stage_b_frontier_table.json"
            if args.stage_b_output is None
            else args.stage_b_output.resolve()
        )
        if stage_b_output.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing Stage-B table: {stage_b_output}"
            )
        table = _stage_b_rows(
            config=config,
            source_payload=payload,
            normalizer=normalizer,
            view=view,
            train_rows=train_rows,
            run_dir=run_dir,
            stage_a_checkpoint=stage_a_checkpoint,
            output_path=None if args.stage_b_output is None else stage_b_output,
            seed=int(args.seed),
            secondary_route=secondary_route,
        )
        result = {
            "stage": "B_measured_frontier_fidelity_work",
            "run_dir": str(run_dir),
            "stage_b_table": str(stage_b_output.resolve()),
            "training_layouts": table["initial_training_layout_count"],
            "candidate_evaluations": table["candidate_evaluations"],
            "unsupported_rows": sum(bool(row["unsupported_at_budget"]) for row in table["candidate_rows"]),
            "new_cfd_solves": 0,
            "training_only": True,
        }
        _write_json(run_dir / "last_command_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.stage == "stage_c":
        stage_b_path = args.stage_b_table or (run_dir / "stage_b_frontier_table.json")
        result = _stage_c_fit(
            config=config,
            source_payload=payload,
            normalizer=normalizer,
            view=view,
            train_rows=train_rows,
            run_dir=run_dir,
            stage_b_path=stage_b_path,
            seed=int(args.seed),
            requested_updates=int(args.stage_c_updates),
        )
        result["run_dir"] = str(run_dir)
        _write_json(run_dir / "last_command_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.arm == "g_packet":
        result = _train_g_packet(
            config=config,
            source_payload=payload,
            normalizer=normalizer,
            view=view,
            train_rows=train_rows,
            run_dir=run_dir,
            updates=int(args.updates),
            seed=int(args.seed),
            secondary_route=secondary_route,
            fixed_budget=args.budget_fraction,
            stop_after_seconds=args.max_seconds,
            resume_checkpoint=args.resume_checkpoint,
        )
    else:
        result = _run_direct_pair(
            config=config,
            source_payload=payload,
            normalizer=normalizer,
            view=view,
            train_rows=train_rows,
            run_dir=run_dir,
            updates=int(args.updates),
            seed=int(args.seed),
            secondary_route=secondary_route,
            fixed_budget=args.budget_fraction,
            stop_after_seconds=args.max_seconds,
            resume_checkpoint=args.resume_checkpoint,
        )
    _write_json(run_dir / "last_command_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
