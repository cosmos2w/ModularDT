"""Bounded evaluation support for Round13 spatial interaction groups.

The legacy diagnostic deliberately reuses the retained Round12 input panel and
the maintained native task. It reports a small set of tree, latent-fit and
same-target parent/descendant measurements without training or timing jobs.
The Round13 endpoint evaluator is added below the same maintained boundaries.
"""

from __future__ import annotations

import json
import io
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contextual_frontier_evaluation import (
    _checkpoint_selection,
    _filesystem_free_bytes,
    _jsonable,
    _native_export_arrays,
    _summarize_nine_observables,
    load_contextual_checkpoint,
)

_LEGACY_DIAGNOSTIC_CASE_LIMIT = 12
_LEGACY_SAMPLE_REPEATS = 8
_LEGACY_OUTPUT_LIMIT = 32 * 1024**2
_SPATIAL_OUTPUT_LIMIT = 256 * 1024**2
_OUTPUT_HEADROOM = 200 * 1024**3
_SURFACE_TRAIN_IDS = ("0303", "0322", "0337", "0348")


def _retained_round12_panels() -> dict[str, Any]:
    """Read the already retained panel file; do not select cases again."""
    project_root = Path(__file__).resolve().parents[2]
    panel_path = (
        project_root
        / "diagnostics"
        / "round12_3_execution_20261024"
        / "input_panels.json"
    )
    try:
        panels = json.loads(panel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(f"Retained Round12 input panel is unavailable: {panel_path}") from exc
    required = ("risk_train_ids", "organizer_train_ids", "detailed_dev_ids", "intervention_train_ids")
    if not isinstance(panels, Mapping) or any(key not in panels for key in required):
        raise ValueError("Retained Round12 input panel file lacks its required fixed case lists.")
    return dict(panels)


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _tree_shape(frontier: Any, leaf_count: int) -> dict[str, Any]:
    """Describe the complete binary candidate tree and its active cut."""
    children = tuple(frontier.children)
    if len(children) < leaf_count or len(children) > 2 * leaf_count - 1:
        raise ValueError("Legacy frontier has an invalid child table.")
    sizes = [1] * len(children)
    heights = [0] * len(children)
    imbalances = []
    for node in range(leaf_count, len(children)):
        pair = children[node]
        if pair is None:
            raise ValueError("Legacy internal frontier node has no children.")
        left, right = map(int, pair)
        if left >= node or right >= node or left < 0 or right < 0:
            raise ValueError("Legacy frontier children are not in postorder.")
        sizes[node] = sizes[left] + sizes[right]
        heights[node] = 1 + max(heights[left], heights[right])
        imbalances.append(
            {
                "node": node,
                "left_leaf_count": sizes[left],
                "right_leaf_count": sizes[right],
                "absolute_leaf_count_difference": abs(sizes[left] - sizes[right]),
                "larger_to_smaller_leaf_ratio": sizes[max((left, right), key=lambda v: sizes[v])]
                / max(1, sizes[min((left, right), key=lambda v: sizes[v])]),
            }
        )

    probabilities = torch.as_tensor(frontier.probabilities).detach().cpu().numpy().reshape(-1)
    reached = tuple(map(int, frontier.reached))
    if probabilities.size not in (0, len(children) - leaf_count):
        raise ValueError("Legacy frontier policy probabilities do not align with its internal nodes.")
    root = len(children) - 1
    depths = [0] * len(children)
    stack = [root]
    while stack:
        parent = stack.pop()
        pair = children[parent]
        if pair is None:
            continue
        for child in pair:
            depths[int(child)] = depths[parent] + 1
            stack.append(int(child))
    reached_rows = []
    for node in reached:
        index = node - leaf_count
        probability = float(probabilities[index]) if 0 <= index < probabilities.size else None
        logit = None
        if probability is not None:
            clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
            logit = math.log(clipped / (1.0 - clipped))
        reached_rows.append(
            {
                "node": node,
                "depth": depths[node],
                "subtree_leaf_count": sizes[node],
                "stop_probability": probability,
                "signed_stop_margin_probability": None if probability is None else probability - 0.5,
                "temperature_scaled_stop_logit_margin": logit,
            }
        )
    active = tuple(map(int, getattr(frontier, "nodes", ())))
    return {
        "candidate_leaf_count": leaf_count,
        "candidate_tree_node_count": len(children),
        "candidate_internal_node_count": max(0, len(children) - leaf_count),
        "maximum_tree_depth": heights[root] if children else 0,
        "active_K": len(active),
        "active_node_ids": list(active),
        "active_subtree_leaf_counts": [sizes[node] for node in active],
        "reached_internal_node_count": len(reached),
        "reached_nodes": reached_rows,
        "mean_absolute_child_size_difference": (
            float(np.mean([row["absolute_leaf_count_difference"] for row in imbalances]))
            if imbalances
            else 0.0
        ),
        "maximum_absolute_child_size_difference": (
            max((row["absolute_leaf_count_difference"] for row in imbalances), default=0)
        ),
        "mean_larger_to_smaller_child_ratio": (
            float(np.mean([row["larger_to_smaller_leaf_ratio"] for row in imbalances]))
            if imbalances
            else 1.0
        ),
        "maximum_larger_to_smaller_child_ratio": (
            max((row["larger_to_smaller_leaf_ratio"] for row in imbalances), default=1.0)
        ),
    }


def _latent_group_residuals(
    states: torch.Tensor,
    coordinates: torch.Tensor,
    point_mass: torch.Tensor,
    labels: torch.Tensor,
    *,
    ridge: float = 1.0e-6,
) -> dict[str, Any]:
    """Fit weighted constant and first-order latent functions on one E state."""
    states = states.detach().to(dtype=torch.float64)
    coordinates = coordinates.detach().to(device=states.device, dtype=torch.float64)
    point_mass = point_mass.detach().to(device=states.device, dtype=torch.float64).reshape(-1)
    labels = labels.detach().to(device=states.device, dtype=torch.long).reshape(-1)
    if states.ndim != 2 or states.shape[0] != coordinates.shape[0] or labels.numel() != states.shape[0]:
        raise ValueError("Latent residual inputs must align over original E records.")
    if point_mass.numel() != states.shape[0] or coordinates.shape[-1] != 2:
        raise ValueError("Latent residual inputs require one mass and 2D coordinate per E record.")
    if not torch.isfinite(states).all() or not torch.isfinite(coordinates).all():
        raise ValueError("Latent residual inputs contain non-finite states or coordinates.")
    positive = torch.isfinite(point_mass) & (point_mass > 0)
    if not bool(positive.any()):
        return {"active_group_count": 0, "positive_mass": 0.0, "groups": []}

    spacing = []
    for axis in range(2):
        unique = torch.unique(coordinates[positive, axis], sorted=True)
        diffs = unique[1:] - unique[:-1]
        diffs = diffs[diffs > 0]
        spacing.append(float(diffs.min()) if diffs.numel() else 0.0)

    group_ids = torch.unique(labels[positive], sorted=True)
    group_rows = []
    total_mass = 0.0
    mean_square_sum = 0.0
    affine_square_sum = 0.0
    scalar_count = 0
    for group_id in group_ids.tolist():
        members = (labels == int(group_id)) & positive
        x = coordinates[members]
        z = states[members]
        mass = point_mass[members]
        mass_sum = mass.sum()
        weights = mass / mass_sum
        center = (weights[:, None] * x).sum(0)
        mean = (weights[:, None] * z).sum(0)
        centered_state = z - mean
        mean_mse = (weights[:, None] * centered_state.square()).sum() / z.shape[-1]

        half_width = torch.stack(
            [
                torch.maximum((x[:, axis] - center[axis]).abs().max(), x.new_tensor(0.5 * spacing[axis]))
                for axis in range(2)
            ]
        ).clamp_min(torch.finfo(x.dtype).eps)
        normalized_x = (x - center) / half_width
        covariance = normalized_x.T @ (weights[:, None] * centered_state)
        scatter = normalized_x.T @ (weights[:, None] * normalized_x)
        coefficient = torch.linalg.solve(
            scatter + ridge * torch.eye(2, dtype=x.dtype, device=x.device), covariance
        )
        affine_error = centered_state - normalized_x @ coefficient
        affine_mse = (weights[:, None] * affine_error.square()).sum() / z.shape[-1]
        group_mass = float(mass_sum)
        count = int(z.numel())
        total_mass += group_mass
        mean_square_sum += group_mass * float(mean_mse)
        affine_square_sum += group_mass * float(affine_mse)
        scalar_count += count
        group_rows.append(
            {
                "group_id": int(group_id),
                "member_count": int(members.sum()),
                "positive_mass": group_mass,
                "mean_only_latent_rmse": math.sqrt(max(0.0, float(mean_mse))),
                "first_order_latent_rmse": math.sqrt(max(0.0, float(affine_mse))),
                "latent_residual_reduction_fraction": (
                    1.0 - float(affine_mse) / float(mean_mse) if float(mean_mse) > 0 else 0.0
                ),
            }
        )
    return {
        "active_group_count": len(group_rows),
        "positive_mass": total_mass,
        "mean_only_latent_rmse_mass_pooled": math.sqrt(mean_square_sum / max(total_mass, 1e-300)),
        "first_order_latent_rmse_mass_pooled": math.sqrt(affine_square_sum / max(total_mass, 1e-300)),
        "groups": group_rows,
        "fit": "weighted local least squares in normalized coordinates; ridge=1e-6",
        "residual_definition": "weighted RMS over latent channels; not a physical-error certificate",
    }


def _sampled_frontier_records(model: Any, scene: Any, *, epoch: int, repeats: int) -> list[dict[str, Any]]:
    core = model.core
    rows: list[dict[str, Any]] = []
    model.train()
    try:
        with torch.no_grad():
            for update in range(repeats):
                core.set_organization_clock(epoch, update)
                prepared = core.prepare(scene)
                rows.append(
                    {
                        "update_key": update,
                        "layers": [
                            {
                                "K": int(groups[0].frontier.K),
                                "active_nodes": list(map(int, groups[0].frontier.nodes)),
                                "reached_nodes": list(map(int, groups[0].frontier.reached)),
                            }
                            for groups in prepared.layers
                        ],
                    }
                )
    finally:
        model.eval()
        core.set_organization_clock(epoch, 0)
    return rows


def _native_component_losses(provider: Any, prediction: Any, batch: Any) -> dict[str, Any]:
    parts = provider._role_losses(prediction, batch.targets)
    total = provider._weighted_native(prediction, batch.targets)
    weights = getattr(provider, "native_weights", {})
    return {
        "raw_weighted_native_total": _scalar(total),
        "raw_native_components": {str(name): _scalar(value) for name, value in parts.items()},
        "native_component_weights": {str(name): float(weights.get(name, 1.0)) for name in parts},
    }


def _validate_new_output(output: str | Path, *, max_bytes: int) -> Path:
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"Spatial interaction-group output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Spatial interaction-group output parent must exist: {output_path.parent}")
    free = _filesystem_free_bytes(output_path.parent)
    if free - max_bytes < _OUTPUT_HEADROOM:
        raise OSError("Spatial interaction-group output reservation would violate DATA2 headroom.")
    return output_path


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _surface_residual_detail(prediction: Any, batch: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Retain per-module surface bias and circumferential residual detail."""
    main = prediction.native_main
    prepared = prediction.native_prepared
    pred = _array(main["pred_interface"])[0].astype(np.float64, copy=False)
    target = _array(batch.targets.interface_target)[0].astype(np.float64, copy=False)
    present = _array(prepared.source_present)[0].astype(bool, copy=False)
    valid = _array(prepared.stencils["surface"].valid).reshape(main["pred_interface"].shape[:-1])[0]
    valid = valid.astype(bool, copy=False) & (_array(batch.targets.interface_valid_mask)[0] > 0.5)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError("Surface detail requires aligned [module,theta,2] native prediction and target arrays.")
    if present.shape != (pred.shape[0],) or valid.shape != pred.shape[:2]:
        raise ValueError("Surface validity and active-module masks do not align with native interface outputs.")

    module_ids_value = batch.scene_inputs.structure.get("module_ids")
    if module_ids_value is None:
        module_ids = np.asarray([f"slot_{index:02d}" for index in range(pred.shape[0])], dtype="U")
    else:
        module_ids = _array(module_ids_value)[0]
        if module_ids.shape != (pred.shape[0],):
            module_ids = np.asarray([f"slot_{index:02d}" for index in range(pred.shape[0])], dtype="U")
        else:
            module_ids = module_ids.astype("U", copy=False)

    theta_count = pred.shape[1]
    angles = np.arange(theta_count, dtype=np.float64) * (2.0 * math.pi / theta_count)
    channel_names = ("temperature_surface", "q_proxy")
    details: dict[str, Any] = {
        "case_id": str(batch.case_keys[0]),
        "angle_rad": angles.tolist(),
        "angle_definition": "native module-local theta=2*pi*j/ntheta, in preserved receiver order",
        "channels": {},
    }
    arrays: dict[str, np.ndarray] = {
        "surface_module_ids": module_ids,
        "surface_angles_rad": angles,
        "surface_valid": valid & present[:, None],
    }
    for channel, name in enumerate(channel_names):
        physical_valid = valid & present[:, None] & np.isfinite(target[..., channel])
        nonfinite_prediction_count = int((physical_valid & ~np.isfinite(pred[..., channel])).sum())
        channel_valid = physical_valid & np.isfinite(pred[..., channel])
        residual = pred[..., channel] - np.nan_to_num(target[..., channel], nan=0.0)
        residual = np.where(channel_valid, residual, np.nan)
        by_module = []
        means = []
        demeaned_rows = []
        for module_index, module_id in enumerate(module_ids):
            mask = channel_valid[module_index]
            if not mask.any():
                continue
            mean_residual = float(residual[module_index, mask].mean())
            demeaned = residual[module_index, mask] - mean_residual
            means.append(mean_residual)
            demeaned_rows.extend(demeaned.tolist())
            by_module.append(
                {
                    "module_id": str(module_id),
                    "module_slot": module_index,
                    "valid_angle_count": int(mask.sum()),
                    "mean_signed_residual": mean_residual,
                    "demeaned_angular_rmse": float(np.sqrt(np.mean(np.square(demeaned)))),
                    "residual_rmse": float(np.sqrt(np.mean(np.square(residual[module_index, mask])))),
                }
            )
        mean_bias_rms = float(np.sqrt(np.mean(np.square(means)))) if means else None
        angular_rmse = (
            float(np.sqrt(np.mean(np.square(demeaned_rows)))) if demeaned_rows else None
        )
        residual_rmse = (
            float(np.sqrt(np.nanmean(np.square(residual)))) if channel_valid.any() else None
        )
        details["channels"][name] = {
            "module_count_with_valid_angles": len(by_module),
            "valid_angle_count": int(channel_valid.sum()),
            "nonfinite_prediction_count_on_physical_support": nonfinite_prediction_count,
            "module_weighted_bias_rms": mean_bias_rms,
            "valid_angle_pooled_demeaned_rmse": angular_rmse,
            "valid_angle_pooled_residual_rmse": residual_rmse,
            "unit": "packed_dataset_native_temperature" if channel == 0 else "packed_dataset_native_q_proxy",
            "per_module": by_module,
        }
        arrays[f"{name}_prediction"] = pred[..., channel].astype(np.float32)
        arrays[f"{name}_target"] = target[..., channel].astype(np.float32)
        arrays[f"{name}_residual"] = residual.astype(np.float32)
        arrays[f"{name}_physical_valid"] = physical_valid
        arrays[f"{name}_valid"] = channel_valid
    return details, arrays


def _native_coordinate_arrays(prediction: Any, batch: Any) -> dict[str, np.ndarray]:
    """Retain exact physical receiver coordinates and their native mapping."""
    prepared = prediction.native_prepared
    scene = prepared.scene_batch
    fluid_xy = _array(batch.receivers.fluid_xy)[0].astype(np.float32, copy=False)
    local_xy = _array(batch.receivers.local_query_points)[0].astype(np.float32, copy=False)
    module_xy = _array(scene.module_coordinates)[0].astype(np.float32, copy=False)
    module_lengths = scene.module_characteristic_lengths
    if module_lengths is None:
        raise ValueError("Round13 coordinate exports require explicit physical module characteristic lengths.")
    module_lengths = _array(module_lengths)[0].astype(np.float32, copy=False)
    if local_xy.ndim == 2:
        local_xy = np.broadcast_to(local_xy[None], (module_xy.shape[0], *local_xy.shape)).copy()
    if (module_xy.ndim != 2 or module_xy.shape[-1] != 2
            or module_lengths.shape != module_xy.shape
            or local_xy.ndim != 3 or local_xy.shape[0] != module_xy.shape[0]
            or local_xy.shape[-1] != 2):
        raise ValueError("Native module geometry and local material receiver coordinates do not align.")
    material_xy = module_xy[:, None, :] + module_lengths[:, None, :] * local_xy
    module_ids = _array(scene.module_ids)[0]
    material_module_ids = np.broadcast_to(module_ids[:, None], local_xy.shape[:2])
    arrays = {
        "fluid_query_coordinates": fluid_xy,
        "fluid_query_indices": np.arange(fluid_xy.shape[0], dtype=np.int32),
        "material_local_query_coordinates": local_xy,
        "material_receiver_coordinates": material_xy.astype(np.float32, copy=False),
        "material_receiver_module_ids": material_module_ids,
        "material_receiver_local_indices": np.broadcast_to(
            np.arange(local_xy.shape[1], dtype=np.int32)[None], local_xy.shape[:2],
        ),
        "module_coordinates": module_xy,
        "module_characteristic_lengths": module_lengths,
        "module_ids": module_ids,
        "module_mask": _array(scene.module_mask)[0].astype(bool, copy=False),
        "source_present": _array(prepared.source_present)[0].astype(bool, copy=False),
        "environment_coordinates": _array(scene.environment_coordinates)[0].astype(np.float32, copy=False),
        "environment_mask": _array(scene.environment_mask)[0].astype(bool, copy=False),
        "coordinate_origin": _array(scene.coordinate_origin)[0].astype(np.float32, copy=False),
        "coordinate_scale": _array(scene.coordinate_scale)[0].astype(np.float32, copy=False),
        "coordinate_frame_key": np.asarray(str(scene.coordinate_frame_key)),
    }
    return arrays


def _score_spatial_variant(
    model: Any,
    provider: Any,
    base_prediction: Any,
    batch: Any,
    prepared: Any,
    *,
    include_losses: bool = True,
    zero_slopes: bool | None = None,
) -> dict[str, Any]:
    native = base_prediction.native_prepared
    decode_options = {} if zero_slopes is None else {"zero_slopes": bool(zero_slopes)}
    decoded = model.core.decode(prepared, native.query_batch, **decode_options)
    native_main = model._extract_native_roles(
        native, decoded.thermal_temperature, decoded.nonlinear_field
    )
    scored_prediction = replace(base_prediction, native_main=native_main)
    metric = provider.validation_metrics(scored_prediction, batch.targets, {})["case_rows"][0]
    result = {
        "native_metrics": metric,
        "prepare_work": dict(prepared.work_counts),
        "decode_work": dict(decoded.work_counts),
        "downstream_stage_recomputed": True,
    }
    if include_losses:
        result["native_losses"] = _native_component_losses(provider, scored_prediction, batch)
    if zero_slopes is not None:
        result["decode_zero_slopes"] = bool(zero_slopes)
    return result


def _distribution_summary(values: Any, *, near_zero_threshold: float = 0.05) -> dict[str, Any]:
    numbers = _array(values).astype(np.float64, copy=False).reshape(-1)
    finite = numbers[np.isfinite(numbers)]
    if not finite.size:
        return {
            "count": int(numbers.size),
            "finite_count": 0,
            "near_zero_threshold": float(near_zero_threshold),
            "near_zero_count": 0,
            "mean": None,
            "std": None,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
            "positive_count": 0,
            "negative_count": 0,
        }
    return {
        "count": int(numbers.size),
        "finite_count": int(finite.size),
        "near_zero_threshold": float(near_zero_threshold),
        "near_zero_count": int((np.abs(finite) <= near_zero_threshold).sum()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "minimum": float(finite.min()),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "maximum": float(finite.max()),
        "positive_count": int((finite > 0).sum()),
        "negative_count": int((finite < 0).sum()),
    }


def _stratified_organizer_candidate(
    metadata: Sequence[Mapping[str, Any]],
    *,
    panel_index: int,
    panel_count: int,
) -> tuple[int, Mapping[str, Any]]:
    """Choose a deterministic candidate across input-only depth and size strata."""
    if not metadata:
        raise ValueError("Organizer candidate metadata cannot be empty.")
    if panel_count < 1 or panel_index < 0 or panel_index >= panel_count:
        raise ValueError("Organizer panel index must be within a non-empty panel.")
    ordered = sorted(
        metadata,
        key=lambda item: (int(item["depth"]), int(item["size"]), int(item["node_id"])),
    )
    if panel_count == 1:
        selected_index = (len(ordered) - 1) // 2
    else:
        # Include both ends of the legal depth/size ordering and spread the fixed
        # panel evenly through it. The choice uses no targets or prediction errors.
        numerator = panel_index * (len(ordered) - 1)
        denominator = panel_count - 1
        selected_index = (numerator + denominator // 2) // denominator
    return selected_index, ordered[selected_index]


def _frontier_cut_summary(group: Any) -> dict[str, Any]:
    frontier = group.frontier
    tree = group.hierarchy
    active = tuple(map(int, frontier.nodes))
    candidate_count = len(frontier.candidate_nodes)
    value_mode_count = int(group.value_coefficients.shape[1])
    active_rows = [
        {
            "node_id": node,
            "depth": int(tree.depths[node]),
            "subtree_leaf_count": int(tree.sizes[node]),
        }
        for node in active
    ]
    utilities = _array(frontier.utilities).astype(np.float64, copy=False).reshape(-1)
    return {
        "K": len(active),
        "P": value_mode_count,
        "K_times_P": int(len(active) * value_mode_count),
        "allocated_group_count": len(active),
        "positive_mass_group_count": int(group.live_group_count),
        "candidate_policy_rows": int(frontier.policy_rows),
        "candidate_internal_node_count": candidate_count,
        "selection_mode": str(frontier.selection_mode),
        "exploration_depth": frontier.exploration_depth,
        "active_node_ids": list(active),
        "active_nodes_with_ancestry_sizes": active_rows,
        "candidate_utility_distribution": _distribution_summary(utilities),
    }


def _latent_residual_summary(prepared: Any, *, batch_index: int = 0) -> dict[str, Any]:
    rows = []
    for stage, groups in enumerate(prepared.layers):
        group = groups[batch_index]
        if not hasattr(group, "latent_constant_residual"):
            raise ValueError("Spatial core must expose compact latent constant and first-order residuals.")
        constant = _array(group.latent_constant_residual).astype(np.float64, copy=False).reshape(-1)
        affine = _array(group.latent_residual).astype(np.float64, copy=False).reshape(-1)
        mass = _array(group.mass).astype(np.float64, copy=False).reshape(-1)
        latent_width = int(group.group_means.shape[-1])
        if constant.shape != mass.shape or affine.shape != mass.shape or latent_width < 1:
            raise ValueError("Compact latent residuals and selected group masses do not align.")
        positive = np.isfinite(mass) & (mass > 0)
        total_mass = float(mass[positive].sum())
        if not positive.any() or total_mass <= 0:
            rows.append({"stage": stage, "active_group_count": 0, "positive_mass": 0.0})
            continue
        constant_mse = float(np.sum(mass[positive] * constant[positive]) / total_mass / latent_width)
        affine_mse = float(np.sum(mass[positive] * affine[positive]) / total_mass / latent_width)
        rows.append(
            {
                "stage": stage,
                "active_group_count": int(positive.sum()),
                "positive_mass": total_mass,
                "latent_width": latent_width,
                "mean_only_latent_rmse_mass_pooled": math.sqrt(max(0.0, constant_mse)),
                "first_order_latent_rmse_mass_pooled": math.sqrt(max(0.0, affine_mse)),
                "per_group": [
                    {
                        "node_id": int(node),
                        "positive_mass": float(mass[index]),
                        "mean_only_latent_rmse": math.sqrt(max(0.0, float(constant[index]) / latent_width)),
                        "first_order_latent_rmse": math.sqrt(max(0.0, float(affine[index]) / latent_width)),
                    }
                    for index, node in enumerate(group.group_nodes)
                    if positive[index]
                ],
                "source": "core-computed compact selected-group residual moments; no per-E latent state retained",
            }
        )
    return {
        "stages": rows,
        "fit_comparison": "core-computed mass-weighted latent residual after group-constant mean versus local first-order fit, using the same prepared E statistics",
        "interpretation_limit": "latent representation diagnostic only; not a physical field error",
    }


def _spatial_prepared_tensor_bytes(prepared: Any) -> int:
    """Count unique tensor storages retained by one prepared spatial case."""
    unique: dict[tuple[str, int, int], int] = {}
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            size = int(storage.nbytes())
            key = (str(value.device), int(storage.data_ptr()), size)
            unique[key] = size
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, np.ndarray, np.generic)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for child in value:
                visit(child)
        elif hasattr(value, "__dict__"):
            for child in vars(value).values():
                visit(child)

    visit(prepared)
    return int(sum(unique.values()))


def _spatial_support_arrays(
    prepared: Any,
    case_id: str,
    *,
    physical_heat: Any | None = None,
) -> dict[str, np.ndarray]:
    """Export exact four-case E-to-group, tree, moment, and relation support."""
    scene = prepared.scene
    tree = prepared.layers[0][0].hierarchy
    prefix = f"case_{case_id}__spatial__"
    children = np.asarray(
        [(-1, -1) if pair is None else pair for pair in tree.children], dtype=np.int32,
    )
    arrays: dict[str, np.ndarray] = {
        prefix + "environment_coordinates": _array(scene.environment_coordinates[0]).astype(np.float32),
        prefix + "environment_coordinates_normalized": _array(
            (scene.environment_coordinates[0] - prepared.frame_origin[0]) / prepared.frame_scale[0]
        ).astype(np.float32),
        prefix + "environment_mask": _array(scene.environment_mask[0]).astype(bool),
        prefix + "environment_measure": _array(scene.environment_measure[0]).astype(np.float32),
        prefix + "environment_measure_reference": _array(
            scene.environment_measure_reference[0]
        ).astype(np.float32),
        prefix + "coordinate_origin": _array(scene.coordinate_origin[0]).astype(np.float32),
        prefix + "coordinate_scale": _array(scene.coordinate_scale[0]).astype(np.float32),
        prefix + "coordinate_frame_key": np.asarray(str(scene.coordinate_frame_key)),
        prefix + "environment_mass_normalized": _array(
            scene.environment_measure[0] / scene.environment_measure_reference[0]
        ).astype(np.float32),
        prefix + "tree_children": children,
        prefix + "tree_parents": np.asarray(tree.parents, dtype=np.int32),
        prefix + "tree_depths": np.asarray(tree.depths, dtype=np.int16),
        prefix + "tree_subtree_leaf_counts": np.asarray(tree.sizes, dtype=np.int16),
        prefix + "tree_leaf_order": np.asarray(tree.leaf_order, dtype=np.int16),
        prefix + "tree_leaf_cells": np.asarray(tree.template.leaf_cells, dtype=np.int16),
        prefix + "tree_box_min": np.asarray(tree.box_min, dtype=np.float32),
        prefix + "tree_box_max": np.asarray(tree.box_max, dtype=np.float32),
        prefix + "module_coordinates": _array(scene.module_coordinates[0]).astype(np.float32),
        prefix + "module_characteristic_lengths": _array(
            scene.module_characteristic_lengths[0]
        ).astype(np.float32),
        prefix + "module_mask": _array(scene.module_mask[0]).astype(bool),
        prefix + "module_ids": _array(scene.module_ids[0]).astype(np.int64),
    }
    if physical_heat is not None:
        arrays[prefix + "module_physical_heat"] = _array(physical_heat[0]).astype(np.float32)
    for stage, groups in enumerate(prepared.layers):
        group = groups[0]
        stage_prefix = prefix + f"stage{stage}__"
        arrays.update({
            stage_prefix + "environment_group_labels": _array(group.frontier.labels).astype(np.int16),
            stage_prefix + "active_node_ids": np.asarray(group.frontier.nodes, dtype=np.int32),
            stage_prefix + "active_node_depths": np.asarray(
                [tree.depths[node] for node in group.frontier.nodes], dtype=np.int16,
            ),
            stage_prefix + "active_node_leaf_counts": np.asarray(
                [tree.sizes[node] for node in group.frontier.nodes], dtype=np.int16,
            ),
            stage_prefix + "candidate_node_ids": np.asarray(group.frontier.candidate_nodes, dtype=np.int32),
            stage_prefix + "candidate_learned_utilities": _array(group.frontier.utilities).astype(np.float32),
            stage_prefix + "group_means": _array(group.group_means).astype(np.float32),
            stage_prefix + "group_slopes": _array(group.group_slopes).astype(np.float32),
            stage_prefix + "group_latent_constant_residual": _array(group.latent_constant_residual).astype(np.float32),
            stage_prefix + "group_latent_residual": _array(group.latent_residual).astype(np.float32),
            stage_prefix + "group_mass": _array(group.mass).astype(np.float32),
            stage_prefix + "group_centers": _array(group.centers).astype(np.float32),
            stage_prefix + "group_scales": _array(group.scales).astype(np.float32),
            stage_prefix + "group_relation_mean_headwise": _array(group.relation_mean).astype(np.float32),
            stage_prefix + "all_node_group_means": _array(group.all_node_means).astype(np.float32),
            stage_prefix + "all_node_group_slopes": _array(group.all_node_slopes).astype(np.float32),
            stage_prefix + "all_node_group_latent_constant_residual": _array(group.all_node_constant_residual).astype(np.float32),
            stage_prefix + "all_node_group_latent_residual": _array(group.all_node_residual).astype(np.float32),
            stage_prefix + "all_node_group_mass": _array(group.all_node_mass).astype(np.float32),
            stage_prefix + "all_node_group_centers": _array(group.all_node_centers).astype(np.float32),
            stage_prefix + "all_node_group_scales": _array(group.all_node_scales).astype(np.float32),
            stage_prefix + "all_node_relation_mean_headwise": _array(group.all_node_relation_mean).astype(np.float32),
            stage_prefix + "all_node_relation_magnitude": _array(group.all_node_relation_magnitude).astype(np.float32),
        })
    return arrays


def _fixed_spatial_covers(prepared: Any) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    frontier = prepared.layers[0][0].frontier
    tree = frontier.tree
    candidates = {
        "root_K1": (int(tree.root),),
        "nearest_K8": tuple(frontier.cover_nearest_k(8)),
        "nearest_K32": tuple(frontier.cover_nearest_k(32)),
        "nearest_K128": tuple(frontier.cover_nearest_k(128)),
        "leaf_K192": tuple(range(int(tree.leaf_count))),
    }
    if len(set(candidates.values())) != 5:
        raise ValueError("The five declared Round13 fixed covers must be distinct on the 24x8 tree.")
    checked = {name: tree.validate_cover(nodes) for name, nodes in candidates.items()}
    detail = {
        name: {
            "K": len(nodes),
            "node_ids": list(nodes),
            "selection": (
                "root singleton" if name == "root_K1"
                else "deterministic legal hierarchy cover nearest target K"
                if name.startswith("nearest_") else "all 192 leaf cells"
            ),
        }
        for name, nodes in checked.items()
    }
    return checked, detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"


def _spatial_control_probe(
    model: Any,
    stencil: Any,
    panel: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one input-column AD/FD and homogeneous-boundary zero-heat probe."""
    from .thermal_geometry_supervision import _native_inputs

    family_id = str(stencil.anchor_id)
    record = stencil.baseline
    module_ids = tuple(record.design.active_module_ids)
    structure, heat = _native_inputs(record, module_ids, device, dtype)
    boundary_inputs = {
        name: _array(structure[name]).astype(np.float64, copy=False)
        for name in ("t_in", "t_wall")
    }
    if any(not np.isfinite(value).all() or not np.equal(value, 0.0).all() for value in boundary_inputs.values()):
        raise ValueError("The homogeneous-boundary zero-heat probe requires exact zero inlet and wall temperatures.")
    fluid = torch.as_tensor(panel["fluid_query_features"][None], device=device, dtype=dtype)
    local = torch.as_tensor(panel["local_query_features"][None], device=device, dtype=dtype)
    scene_key = ("round13_bounded_source_probe", family_id)
    readout_names = (
        "u", "v", "p", "omega", "temperature_fluid", "temperature_surface", "q_proxy",
        "temperature_material",
    )

    def prepare_and_apply(controls: torch.Tensor):
        prepared = model.prepare_native(
            structure,
            fluid,
            physical_heat=controls,
            local_query_points=local,
            ntheta=16,
            scene_key=scene_key,
        )
        output = model.apply_native(prepared, controls)
        return output, prepared

    def readouts_with_prepared(controls: torch.Tensor) -> tuple[torch.Tensor, Any]:
        output, prepared = prepare_and_apply(controls)
        field = output["pred_field"][0]
        fluid_valid = prepared.stencils["fluid"].valid[0]
        if not bool(fluid_valid.any()):
            raise ValueError("Bounded control panel has no valid fluid receivers.")
        fluid_means = [field[fluid_valid, index].mean() for index in range(5)]
        interface = output["pred_interface"][0]
        surface_shape = interface.shape[:-1]
        present = prepared.source_present[0, :, None].to(torch.bool)
        surface_valid = prepared.stencils["surface"].valid[0].reshape(surface_shape) & present
        outside_valid = prepared.stencils["outside"].valid[0].reshape(surface_shape) & surface_valid
        material = output["pred_internal_temperature"][0, ..., 0]
        material_valid = (
            prepared.stencils["material"].valid.reshape(output["pred_internal_temperature"].shape[:-1])[0]
            & prepared.source_present[0, :, None].to(torch.bool)
        )
        if not bool(surface_valid.any()) or not bool(outside_valid.any()) or not bool(material_valid.any()):
            raise ValueError("Bounded control panel has no valid surface, outside, or material receiver.")
        surface_mean = interface[..., 0][surface_valid].mean()
        q_mean = interface[..., 1][outside_valid].mean()
        material_mean = material[material_valid].mean()
        return torch.stack((*fluid_means, surface_mean, q_mean, material_mean)), prepared

    def readouts(controls: torch.Tensor) -> torch.Tensor:
        return readouts_with_prepared(controls)[0]

    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        base_heat = heat.detach()
        source_slot = int(torch.argmax(base_heat.abs()[0]))
        source_step = min(
            1.0e-3 * float(getattr(model, "heat_scale", 1.0)),
            max(1.0e-8, 0.25 * float(base_heat[0, source_slot].abs())),
        )
        direction = torch.zeros_like(base_heat)
        direction[0, source_slot] = 1.0
        differentiable_heat = base_heat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            _base_values, jvp = torch.autograd.functional.jvp(
                readouts,
                differentiable_heat,
                direction,
                create_graph=False,
                strict=False,
            )
        with torch.no_grad():
            base_values, base_prepared = readouts_with_prepared(base_heat)
            plus_heat = base_heat.clone()
            minus_heat = base_heat.clone()
            plus_heat[0, source_slot] += source_step
            minus_heat[0, source_slot] -= source_step
            plus_values, plus_prepared = readouts_with_prepared(plus_heat)
            minus_values, minus_prepared = readouts_with_prepared(minus_heat)
            finite_difference = (plus_values - minus_values) / (2.0 * source_step)

            zero_heat = torch.zeros_like(base_heat)
            zero_output, zero_prepared = prepare_and_apply(zero_heat)
            zero_field = zero_output["pred_field"][0]
            zero_interface = zero_output["pred_interface"][0]
            zero_material = zero_output["pred_internal_temperature"][0, ..., 0]
            zero_fluid_mask = zero_prepared.stencils["fluid"].valid[0]
            zero_surface_shape = zero_interface.shape[:-1]
            zero_present = zero_prepared.source_present[0, :, None].to(torch.bool)
            zero_surface_mask = (
                zero_prepared.stencils["surface"].valid[0].reshape(zero_surface_shape) & zero_present
            )
            zero_outside_mask = (
                zero_prepared.stencils["outside"].valid[0].reshape(zero_surface_shape) & zero_surface_mask
            )
            zero_material_mask = (
                zero_prepared.stencils["material"].valid.reshape(zero_output["pred_internal_temperature"].shape[:-1])[0]
                & zero_prepared.source_present[0, :, None].to(torch.bool)
            )
            zero_components = {
                "temperature_fluid": (zero_field[:, 4], zero_fluid_mask),
                "temperature_surface": (zero_interface[..., 0], zero_surface_mask),
                "q_proxy": (zero_interface[..., 1], zero_outside_mask),
                "temperature_material": (zero_material, zero_material_mask),
            }

        zero_summary = {}
        zero_arrays = {}
        for name, (values, valid) in zero_components.items():
            selected = values[valid]
            if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
                zero_summary[name] = {
                    "status": "invalid_no_finite_physical_support",
                    "valid_row_count": int(selected.numel()),
                    "rms": None,
                    "max_absolute": None,
                }
            else:
                zero_summary[name] = {
                    "status": "valid",
                    "valid_row_count": int(selected.numel()),
                    "rms": float(selected.double().square().mean().sqrt()),
                    "max_absolute": float(selected.double().abs().max()),
                    "unit": "packed_dataset_native_temperature" if name.startswith("temperature_") else "packed_dataset_native_q_proxy",
                }
            zero_arrays[f"zero_heat_{name}_prediction"] = _array(values).astype(np.float32)
            zero_arrays[f"zero_heat_{name}_valid"] = _array(valid).astype(bool)

        def active_nodes(prepared: Any) -> list[int]:
            return list(map(int, prepared.organization.layers[0][0].frontier.nodes))

        summary = {
            "schema": "thermal_spatial_interaction_groups_bounded_control_probe_v1",
            "reference_family_id": family_id,
            "scope": "one existing TRAIN0348/M10 saved-reference geometry; no new physical solve",
            "original_module_ids": list(module_ids),
            "source_column_probe": {
                "varied_module_id": str(module_ids[source_slot]),
                "source_column_index": source_slot,
                "all_original_heat_columns_retained": True,
                "other_heat_columns_held_fixed": True,
                "baseline_heat": _array(base_heat)[0].tolist(),
                "step": source_step,
                "scheme": "central finite difference",
                "readout_names": list(readout_names),
                "fixed_cut_jvp": _array(jvp).tolist(),
                "reprepared_finite_difference": _array(finite_difference).tolist(),
                "absolute_ad_fd_error": _array((jvp - finite_difference).abs()).tolist(),
                "baseline_stage0_cut": active_nodes(base_prepared),
                "plus_stage0_cut": active_nodes(plus_prepared),
                "minus_stage0_cut": active_nodes(minus_prepared),
                "cut_crossed_on_plus": active_nodes(plus_prepared) != active_nodes(base_prepared),
                "cut_crossed_on_minus": active_nodes(minus_prepared) != active_nodes(base_prepared),
                "interpretation": "JVP differentiates the selected baseline cut; finite differences fully reprepare and may include a cut crossing; AD/FD agreement is not physical-gradient validation",
            },
            "homogeneous_boundary_zero_heat": {
                "all_original_heat_columns_set_to_zero": True,
                "heat_input_vector": _array(zero_heat)[0].tolist(),
            "explicit_boundary_inputs": {
                name: value.tolist() for name, value in boundary_inputs.items()
            },
                "analytic_temperature_rise_target": 0.0,
                "measured_components": zero_summary,
                "stage0_cut": active_nodes(zero_prepared),
                "interpretation": "temperature and q readouts are compared with the analytic homogeneous-boundary zero target; flow magnitude is not treated as a zero-flow error",
            },
            "optimizer_updates": 0,
            "solver_attempts": 0,
        }
        arrays = {
            "probe_source_module_ids": np.asarray(module_ids, dtype="U"),
            "probe_baseline_physical_heat": _array(base_heat).astype(np.float64),
            "probe_homogeneous_boundary_zero_heat_input": _array(zero_heat).astype(np.float64),
            "probe_homogeneous_boundary_t_in": boundary_inputs["t_in"],
            "probe_homogeneous_boundary_t_wall": boundary_inputs["t_wall"],
            "probe_fixed_cut_jvp": _array(jvp).astype(np.float64),
            "probe_reprepared_finite_difference": _array(finite_difference).astype(np.float64),
            **zero_arrays,
        }
        return summary, arrays
    finally:
        for module, was_training in modes.items():
            module.training = was_training


def evaluate_legacy_group_diagnostic(
    checkpoint: str | Path,
    device: str | torch.device,
    output: str | Path,
    *,
    case_ids: Sequence[str] | None = None,
    sample_repeats: int = _LEGACY_SAMPLE_REPEATS,
) -> dict[str, Any]:
    """Measure bounded Round12 tree, latent-fit and signed native utility evidence.

    Defaults to four retained input-selected TRAIN cases spanning M1/M2/M10/M12.
    At most one reached stage-1 node is replayed per case; descendants trigger
    a complete dependent reprepare before native decoding and loss evaluation.
    """
    if not 1 <= int(sample_repeats) <= 16:
        raise ValueError("Legacy sampled-frontier repetitions must be in [1,16].")
    output_path = _validate_new_output(output, max_bytes=_LEGACY_OUTPUT_LIMIT)
    model, provider, payload = load_contextual_checkpoint(checkpoint, device)
    if model.architecture not in ("G-H", "G-EE-H"):
        raise ValueError("Legacy spatial diagnostics require an adaptive G-H or G-EE-H checkpoint.")
    panels = _retained_round12_panels()
    selected = tuple(map(str, case_ids if case_ids is not None else panels["intervention_train_ids"]))
    if not selected or len(selected) > _LEGACY_DIAGNOSTIC_CASE_LIMIT or len(set(selected)) != len(selected):
        raise ValueError("Legacy diagnostic case IDs must be unique and contain at most twelve TRAIN cases.")
    if not set(selected).issubset(set(map(str, provider.training_case_ids))):
        raise ValueError("Legacy spatial diagnostic accepts TRAIN case IDs only.")
    epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    model.core.set_organization_clock(epoch, 0)
    model.eval()
    case_rows = []
    with torch.no_grad():
        batches = provider.native_evaluation_batches(split="train", case_keys=selected, microbatch_size=1)
        observed = []
        for batch in batches:
            case_id = str(batch.case_keys[0])
            if case_id not in selected or len(batch.case_keys) != 1:
                raise ValueError("Legacy diagnostic loader changed the explicit one-case TRAIN panel.")
            observed.append(case_id)
            prediction, auxiliary = provider.predict_native(
                model, batch.scene_inputs, batch.receivers, model.architecture, "validation", epoch=epoch
            )
            native = prediction.native_prepared
            prepared = native.organization
            scene = prepared.scene
            query = native.query_batch
            deterministic_layers = []
            residual_layers = []
            for layer, groups in enumerate(prepared.layers):
                group = groups[0]
                deterministic_layers.append(_tree_shape(group.frontier, int(scene.environment_mask[0].sum())))
                mass = scene.environment_measure[0] / scene.environment_measure_reference[0]
                mass = mass * scene.environment_mask[0]
                points = (scene.environment_coordinates[0] - prepared.frame_origin[0]) / prepared.frame_scale[0]
                residual_layers.append(
                    _latent_group_residuals(
                        group.environment_states,
                        points,
                        mass,
                        group.frontier.labels,
                    )
                )

            # Select one highest-confidence reached stage-1 decision by policy
            # margin only. This selection is independent of target errors.
            first_frontier = prepared.layers[0][0].frontier
            leaf_count = int(scene.environment_mask[0].sum())
            probabilities = torch.as_tensor(first_frontier.probabilities).detach().cpu().numpy().reshape(-1)
            reached = tuple(map(int, first_frontier.reached))
            if reached:
                node = max(
                    reached,
                    key=lambda value: (abs(float(probabilities[value - leaf_count] - 0.5)), -value),
                )
                chosen_probability = float(probabilities[node - leaf_count])
                replay = {}
                variant_predictions = {}
                variant_losses = {}
                for mode in ("parent", "children"):
                    changed = model.core.prepare(scene, forced_frontier={0: (node, mode)})
                    decoded = model.core.decode(changed, query)
                    main = model._extract_native_roles(
                        native, decoded.thermal_temperature, decoded.nonlinear_field
                    )
                    changed_prediction = replace(prediction, native_main=main)
                    variant_predictions[mode] = changed_prediction
                    metrics = provider.validation_metrics(changed_prediction, batch.targets, {})["case_rows"][0]
                    losses = _native_component_losses(provider, changed_prediction, batch)
                    variant_losses[mode] = losses
                    replay[mode] = {
                        "K_by_layer": [int(layer[0].frontier.K) for layer in changed.layers],
                        "native_metrics": metrics,
                        "native_losses": losses,
                        "reprepare_work": changed.work_counts,
                        "decode_work": decoded.work_counts,
                        "dependent_stage_recomputed": True,
                    }
                component_changes = {
                    name: float(
                        variant_losses["parent"]["raw_native_components"][name]
                        - variant_losses["children"]["raw_native_components"][name]
                    )
                    for name in variant_losses["parent"]["raw_native_components"]
                }
                replay.update(
                    {
                        "chosen_layer": 0,
                        "chosen_node": node,
                        "selection": "maximum absolute reached-node stop probability margin in stage 1; input/model policy only",
                        "stop_probability": chosen_probability,
                        "parent_minus_children_weighted_native_gain": (
                            variant_losses["parent"]["raw_weighted_native_total"]
                            - variant_losses["children"]["raw_weighted_native_total"]
                        ),
                        "parent_minus_children_raw_component_gains": component_changes,
                        "positive_gain_means_descendants_reduce_native_loss": True,
                        "raw_signed_losses_retained": True,
                    }
                )
            else:
                replay = {"status": "no_reached_internal_node", "chosen_nodes": 0}

            sampled = _sampled_frontier_records(
                model, scene, epoch=epoch, repeats=int(sample_repeats)
            )
            sampled_by_layer = []
            for layer in range(len(prepared.layers)):
                draws = [row["layers"][layer] for row in sampled]
                sampled_by_layer.append(
                    {
                        "sample_count": len(draws),
                        "K_by_draw": [row["K"] for row in draws],
                        "mean_K": float(np.mean([row["K"] for row in draws])),
                        "active_node_frequency": {
                            str(node_id): sum(node_id in row["active_nodes"] for row in draws) / len(draws)
                            for node_id in sorted({node for row in draws for node in row["active_nodes"]})
                        },
                        "reached_node_frequency": {
                            str(node_id): sum(node_id in row["reached_nodes"] for row in draws) / len(draws)
                            for node_id in sorted({node for row in draws for node in row["reached_nodes"]})
                        },
                    }
                )
            ordinary_metrics = provider.validation_metrics(prediction, batch.targets, auxiliary)["case_rows"][0]
            case_rows.append(
                {
                    "case_id": case_id,
                    "module_count": int(scene.module_mask[0].sum()),
                    "deterministic_frontiers": deterministic_layers,
                    "sampled_frontiers": sampled_by_layer,
                    "sampled_draws": sampled,
                    "latent_residuals_on_identical_prepared_E_states": residual_layers,
                    "ordinary_native_metrics": ordinary_metrics,
                    "one_signed_parent_descendant_replay": replay,
                }
            )
            model.core.set_organization_clock(epoch, 0)
    if tuple(observed) != selected:
        raise ValueError("Legacy diagnostic did not evaluate the explicit TRAIN panel exactly once in order.")
    result = {
        "schema": "legacy_spatial_interaction_group_diagnostic_v1",
        "status": "completed",
        "checkpoint": str(Path(checkpoint).expanduser().resolve(strict=True)),
        "epoch": epoch,
        "arm": model.architecture,
        "partition": "source TRAIN, selected fixed25 TRAIN150",
        "case_ids": list(selected),
        "panel_selection": "Round12 retained intervention panel from input_panels.json; no error-based selection",
        "sample_repeats_per_case": int(sample_repeats),
        "case_count": len(case_rows),
        "case_rows": case_rows,
        "native_loss_definition": "maintained provider._weighted_native and per-role _role_losses, identical target/mask batch for each alternative",
        "tree_margin_definition": "stop probability minus 0.5 and temperature-scaled logit(probability), evaluated only at reached internal nodes",
        "latent_fit_limit": "latent representation diagnostic only; it does not certify physical field fidelity",
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "test_access": False,
        "plots": "not generated",
    }
    payload_bytes = json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if len(payload_bytes) > _LEGACY_OUTPUT_LIMIT:
        raise OSError("Legacy spatial diagnostic exceeded its 32 MiB output cap.")
    output_path.mkdir(parents=False, exist_ok=False)
    (output_path / "result.json").write_bytes(payload_bytes)
    actual_bytes = (output_path / "result.json").stat().st_size
    if actual_bytes != len(payload_bytes):
        raise OSError("Legacy spatial diagnostic output size changed while writing.")
    return {"output_path": str(output_path), "output_bytes": actual_bytes, **_jsonable(result)}


def evaluate_spatial_groups(
    checkpoint: str | Path,
    device: str | torch.device,
    output: str | Path,
    *,
    include_cost: bool = False,
) -> dict[str, Any]:
    """Measure bounded Round13 native, response, cut, and organizer evidence.

    The fixed panel comes from the retained Round12 input selections and report.
    This evaluates the exposed DEV22 panel, twelve M1/M2 TRAIN cases, the
    retained TRAIN4/DEV4 surface panels, eight saved response atlases, the
    organizer24 same-target probes, and four TRAIN scenes under five legal
    covers plus the bounded mode, relay, and organizer ablations. It never
    trains or invokes a solver.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = _validate_new_output(output, max_bytes=_SPATIAL_OUTPUT_LIMIT)
    model, provider, payload = load_contextual_checkpoint(checkpoint_path, device)
    identity = payload.get("experiment_identity", {})
    if (not isinstance(identity, Mapping)
            or identity.get("workflow") != "spatial_interaction_groups_joint_training_v1"
            or model.architecture not in ("F13-0", "F13-1")):
        raise ValueError("Spatial-group evaluation requires a maintained F13-0 or F13-1 checkpoint.")
    provider_identity = identity.get("provider_identity", {})
    panels = _retained_round12_panels()
    risk_train_ids = tuple(map(str, panels["risk_train_ids"]))
    organizer_ids = tuple(map(str, panels["organizer_train_ids"]))
    dev_detail_ids = tuple(map(str, panels["detailed_dev_ids"]))
    intervention_ids = tuple(map(str, panels["intervention_train_ids"]))
    surface_train_ids = _SURFACE_TRAIN_IDS
    train_ids = tuple(dict.fromkeys((*organizer_ids, *risk_train_ids)))
    dev_ids = tuple(map(str, getattr(provider, "validation_case_ids", ())))
    if len(dev_ids) != 22 or len(set(dev_ids)) != 22:
        raise ValueError("Round13 exposed native validation must be the fixed DEV22 panel.")
    if not set(dev_detail_ids).issubset(dev_ids):
        raise ValueError("Retained DEV4 surface-detail IDs are outside the exposed DEV22 panel.")
    training_membership = set(map(str, getattr(provider, "training_case_ids", ())))
    if not set(train_ids).issubset(training_membership):
        raise ValueError("A retained Round12 TRAIN panel ID is outside the rebuilt provider's training membership.")
    if (len(risk_train_ids) != 12 or len(organizer_ids) != 24
            or len(intervention_ids) != 4 or len(surface_train_ids) != 4):
        raise ValueError("Round13 panel cardinalities differ from their retained Round12 definitions.")

    epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    arm = str(payload.get("arm", provider_identity.get("architecture", "")))
    if arm != model.architecture:
        raise ValueError("Saved checkpoint arm and rebuilt spatial adapter architecture differ.")
    device_obj = torch.device(device)
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("Spatial interaction-group checkpoint has no parameters.")
    model.eval()
    model.core.set_organization_clock(epoch, 0)

    from .interaction_group_evaluation import mean_prediction, selected_train_mean
    train_means = selected_train_mean(provider)
    train_rows_by_id: dict[str, Mapping[str, Any]] = {}
    train_records_by_id: dict[str, Mapping[str, Any]] = {}
    train_baseline_rows_by_id: dict[str, Mapping[str, Any]] = {}
    train_baseline_records_by_id: dict[str, Mapping[str, Any]] = {}
    dev_rows_by_id: dict[str, Mapping[str, Any]] = {}
    dev_records_by_id: dict[str, Mapping[str, Any]] = {}
    dev_baseline_rows_by_id: dict[str, Mapping[str, Any]] = {}
    dev_baseline_records_by_id: dict[str, Mapping[str, Any]] = {}
    detail_rows: dict[str, Any] = {}
    detail_arrays: dict[str, np.ndarray] = {}
    organizer_rows: list[dict[str, Any]] = []
    dev_organizer_rows: list[dict[str, Any]] = []
    cover_rows: list[dict[str, Any]] = []
    fixed_cover_mode_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    def score_detail(case_id: str, prediction: Any, batch: Any) -> None:
        export = _native_export_arrays(prediction, batch, case_id)
        export.update(_native_coordinate_arrays(prediction, batch))
        for name, value in export.items():
            detail_arrays[f"case_{case_id}__{name}"] = np.asarray(value)
        surface, surface_arrays = _surface_residual_detail(prediction, batch)
        detail_rows[case_id] = surface
        for name, value in surface_arrays.items():
            detail_arrays[f"case_{case_id}__{name}"] = np.asarray(value)

    def timed_native_case(batch: Any) -> dict[str, Any]:
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        values = []
        last_auxiliary = {}
        # One bounded warmup followed by three synchronized complete-native repeats.
        for iteration in range(4):
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            start = time.perf_counter()
            _prediction, last_auxiliary = provider.predict_native(
                model, batch.scene_inputs, batch.receivers, arm, "validation", epoch=epoch,
            )
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            elapsed = time.perf_counter() - start
            if iteration:
                values.append(elapsed)
        work = dict(last_auxiliary.get("work", {})) if isinstance(last_auxiliary, Mapping) else {}
        return {
            "case_id": str(batch.case_keys[0]),
            "complete_native_seconds": values,
            "median_complete_native_seconds": float(np.median(values)),
            "repeat_count": len(values),
            "warmup_count": 1,
            "measured_work_counts": work,
            "boundary": "provider.predict_native including core prepare, native union, and native extraction",
        }

    def measured_record(batch: Any) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
        prediction, auxiliary = provider.predict_native(
            model, batch.scene_inputs, batch.receivers, arm, "validation", epoch=epoch,
        )
        record = provider.validation_metrics(prediction, batch.targets, auxiliary)
        rows = record.get("case_rows") if isinstance(record, Mapping) else None
        if not isinstance(rows, Sequence) or len(rows) != 1:
            raise ValueError("Round13 native evaluation must preserve exactly one complete case per batch.")
        return prediction, auxiliary, record, rows[0]

    observed_train = []
    with torch.no_grad():
        for batch in provider.native_evaluation_batches(
            split="train", case_keys=train_ids, microbatch_size=1,
        ):
            case_id = str(batch.case_keys[0])
            if case_id not in train_ids or len(batch.case_keys) != 1:
                raise ValueError("Round13 selected TRAIN loader changed the exact one-case panel.")
            observed_train.append(case_id)
            prediction, auxiliary, record, row = measured_record(batch)
            train_rows_by_id[case_id] = row
            train_records_by_id[case_id] = record
            baseline = mean_prediction(prediction, train_means)
            train_baseline_records_by_id[case_id] = provider.validation_metrics(
                baseline, batch.targets, auxiliary,
            )
            train_baseline_rows_by_id[case_id] = train_baseline_records_by_id[case_id]["case_rows"][0]

            if case_id in surface_train_ids:
                score_detail(case_id, prediction, batch)

            if case_id in organizer_ids:
                native = prediction.native_prepared
                organized = native.organization
                core = model.core
                candidates = tuple(map(int, core.utility_candidates(organized, batch_index=0)))
                if not candidates:
                    raise ValueError("Organizer diagnostic found no internal hierarchy candidates.")
                organizer_index = organizer_ids.index(case_id)
                candidate_metadata_rows = tuple(
                    core.utility_candidate_metadata(organized, batch_index=0)
                )
                candidate_index, candidate_metadata = _stratified_organizer_candidate(
                    candidate_metadata_rows,
                    panel_index=organizer_index,
                    panel_count=len(organizer_ids),
                )
                candidate_metadata = dict(candidate_metadata)
                node = int(candidate_metadata["node_id"])
                if node not in candidates:
                    raise ValueError("Organizer candidate metadata does not match the utility candidate set.")
                features = core.utility_features(organized, node, batch_index=0, detach=True)
                utility_proxy = _scalar(core.predict_utility(features))
                frontier = organized.layers[0][0].frontier
                candidate_utilities = _array(frontier.utilities).astype(np.float64, copy=False).reshape(-1)
                candidate_positions = {candidate: index for index, candidate in enumerate(candidates)}
                selected_frontier_utility = float(candidate_utilities[candidate_positions[node]])
                candidate_tree = frontier.tree
                ancestry = []
                ancestor = node
                while True:
                    ancestry.append(
                        {
                            "node_id": int(ancestor),
                            "depth": int(candidate_tree.depths[ancestor]),
                            "subtree_leaf_count": int(candidate_tree.sizes[ancestor]),
                        }
                    )
                    if ancestor == candidate_tree.root:
                        break
                    ancestor = int(candidate_tree.parents[ancestor])
                ancestry.reverse()
                alternatives = {}
                raw_losses = {}
                for mode in ("parent", "children"):
                    changed = core.prepare(native.organization.scene, forced_frontier={0: (node, mode)})
                    alternatives[mode] = _score_spatial_variant(
                        model, provider, prediction, batch, changed,
                    )
                    raw_losses[mode] = alternatives[mode]["native_losses"]
                    alternatives[mode]["K_by_layer"] = [
                        int(layer[0].frontier.K) for layer in changed.layers
                    ]
                raw_gain = (
                    raw_losses["parent"]["raw_weighted_native_total"]
                    - raw_losses["children"]["raw_weighted_native_total"]
                )
                component_gains = {
                    name: float(
                        raw_losses["parent"]["raw_native_components"][name]
                        - raw_losses["children"]["raw_native_components"][name]
                    )
                    for name in raw_losses["parent"]["raw_native_components"]
                }
                predicted_descendant_action = utility_proxy > 0.0
                observed_descendant_benefit = raw_gain > 0.0
                proxy_regret = (
                    abs(raw_gain)
                    if predicted_descendant_action != observed_descendant_benefit
                    else 0.0
                )
                organizer_rows.append(
                    {
                        "case_id": case_id,
                        "module_count": int(row["module_count"]),
                        "base_native_metrics": row,
                        "actual_cut_summary": {
                            "stage0": _frontier_cut_summary(organized.layers[0][0]),
                            "stage1": _frontier_cut_summary(organized.layers[1][0]),
                            "chosen_candidate_ancestry": ancestry,
                        },
                        "latent_mean_vs_first_order_residuals": _latent_residual_summary(organized),
                        "persistent_spatial_prepared_tensor_storage_bytes": _spatial_prepared_tensor_bytes(organized),
                        "work_counts": {
                            "spatial_prepare": dict(organized.work_counts),
                            "native_prediction": dict(auxiliary.get("work", {})),
                        },
                        "target_free_candidate_selection": {
                            "candidate_count": len(candidates),
                            "depth_size_stratified_rank": candidate_index,
                            "candidate_policy_index": candidate_positions[node],
                            "candidate": candidate_metadata,
                            "selection": "fixed organizer24 panel index selects evenly spaced rank after sorting input-only node depth, subtree leaf count, and node id; no target or error access",
                        },
                        "predicted_parent_minus_children_signed_gain_proxy": utility_proxy,
                        "selected_frontier_learned_utility": selected_frontier_utility,
                        "learned_utility_distribution": _distribution_summary(candidate_utilities),
                        "learned_utility_near_zero": abs(utility_proxy) <= 0.05,
                        "learned_utility_near_zero_threshold": 0.05,
                        "parent_minus_children_raw_weighted_native_gain": raw_gain,
                        "parent_minus_children_raw_component_gains": component_gains,
                        "utility_proxy_sign_agrees_with_observed_gain": (
                            bool(np.sign(utility_proxy) == np.sign(raw_gain))
                            if utility_proxy != 0.0 and raw_gain != 0.0 else None
                        ),
                        "proxy_action": "children" if predicted_descendant_action else "parent",
                        "observed_native_better_action": "children" if observed_descendant_benefit else "parent",
                        "bounded_single_candidate_native_regret": proxy_regret,
                        "regret_definition": "absolute raw weighted native gain when the selected candidate proxy chooses the worse of parent versus descendants; zero otherwise",
                        "positive_raw_gain_means_children_reduce_native_loss": True,
                        "raw_same_target_losses": raw_losses,
                        "alternatives": alternatives,
                        "optimizer_updates": 0,
                    }
                )

            if case_id in intervention_ids:
                native = prediction.native_prepared
                core = model.core
                detail_arrays.update(
                    _spatial_support_arrays(
                        native.organization, case_id, physical_heat=batch.receivers.heat,
                    )
                )
                fixed, fixed_description = _fixed_spatial_covers(native.organization)
                for name, nodes in fixed.items():
                    changed = core.prepare(
                        native.organization.scene, forced_frontier={0: {"cover": nodes}},
                    )
                    row_variant = _score_spatial_variant(
                        model, provider, prediction, batch, changed,
                    )
                    fixed_cover_zero_mode = None
                    if name == "nearest_K32":
                        fixed_cover_zero_mode = _score_spatial_variant(
                            model, provider, prediction, batch, changed, zero_slopes=True,
                        )
                        fixed_cover_mode_rows.append(
                            {
                                "case_id": case_id,
                                "cover_name": name,
                                "K_by_layer": [int(layer[0].frontier.K) for layer in changed.layers],
                                "same_prepared_cut_and_full_relay": True,
                                **fixed_cover_zero_mode,
                            }
                        )
                    cover_rows.append(
                        {
                            "case_id": case_id,
                            "cover_name": name,
                            "cover_definition": fixed_description[name],
                            "K_by_layer": [int(layer[0].frontier.K) for layer in changed.layers],
                            "zero_final_read_mode_same_cut": fixed_cover_zero_mode,
                            **row_variant,
                        }
                    )
                ablations = (
                    ("zero_final_read_slopes", None, True),
                    ("zero_spatial_slopes_relay_and_final", {"zero_slopes": True}, None),
                    ("bypass_group_to_M_relay", {"clamp_layer": 0}, None),
                    ("remove_organizer_relation_features", {"relation_features": False}, None),
                )
                baseline_nodes = tuple(map(int, native.organization.layers[0][0].frontier.nodes))
                for name, prepare_options, decode_zero_slopes in ablations:
                    changed = (
                        native.organization
                        if prepare_options is None
                        else core.prepare(native.organization.scene, **prepare_options)
                    )
                    row_variant = _score_spatial_variant(
                        model, provider, prediction, batch, changed,
                        zero_slopes=decode_zero_slopes,
                    )
                    changed_nodes = tuple(map(int, changed.layers[0][0].frontier.nodes))
                    ablation_rows.append(
                        {
                            "case_id": case_id,
                            "ablation": name,
                            "prepare_options": {} if prepare_options is None else prepare_options,
                            "decode_zero_slopes": decode_zero_slopes,
                            "same_prepared_cut": changed is native.organization,
                            "full_relay_preserved_for_final_read_mode": name == "zero_final_read_slopes",
                            "baseline_cut_node_ids": list(baseline_nodes),
                            "ablation_cut_node_ids": list(changed_nodes),
                            "cut_unchanged": changed_nodes == baseline_nodes,
                            "K_by_layer": [int(layer[0].frontier.K) for layer in changed.layers],
                            **row_variant,
                        }
                    )

            if include_cost and case_id in ("0006", "0051"):
                cost_rows.append(timed_native_case(batch))

    if tuple(observed_train) != train_ids:
        raise ValueError("Round13 selected TRAIN panel was not evaluated exactly once in requested order.")
    if tuple(case_id for case_id in organizer_ids if case_id in {row["case_id"] for row in organizer_rows}) != organizer_ids:
        raise ValueError("Round13 organizer diagnostic did not cover every fixed organizer24 case.")
    if tuple(row["case_id"] for row in organizer_rows) != organizer_ids:
        raise ValueError("Round13 organizer diagnostic changed the retained case order.")
    if tuple(row["case_id"] for row in cover_rows if row["cover_name"] == "root_K1") != intervention_ids:
        raise ValueError("Round13 cover diagnostic did not cover all four retained interaction cases.")
    if len(cover_rows) != 4 * 5 or len(ablation_rows) != 4 * 4:
        raise ValueError("Round13 fixed-cover/ablation evidence cardinality differs from its bounded plan.")
    if len(fixed_cover_mode_rows) != 4:
        raise ValueError("Round13 common-cover zero-mode probe must cover each retained interaction case once.")

    observed_dev = []
    with torch.no_grad():
        for batch in provider.native_evaluation_batches(
            split="validation", case_keys=dev_ids, microbatch_size=1,
        ):
            case_id = str(batch.case_keys[0])
            if case_id not in dev_ids or len(batch.case_keys) != 1:
                raise ValueError("Round13 DEV22 loader changed the exact one-case panel.")
            observed_dev.append(case_id)
            prediction, auxiliary, record, row = measured_record(batch)
            dev_rows_by_id[case_id] = row
            dev_records_by_id[case_id] = record
            baseline = mean_prediction(prediction, train_means)
            dev_baseline_records_by_id[case_id] = provider.validation_metrics(
                baseline, batch.targets, auxiliary,
            )
            dev_baseline_rows_by_id[case_id] = dev_baseline_records_by_id[case_id]["case_rows"][0]
            if case_id in dev_detail_ids:
                score_detail(case_id, prediction, batch)
                native = prediction.native_prepared
                organized = native.organization
                core = model.core
                candidates = tuple(map(int, core.utility_candidates(organized, batch_index=0)))
                if not candidates:
                    raise ValueError("DEV4 utility diagnostic found no internal hierarchy candidates.")
                candidate_index = dev_detail_ids.index(case_id) % len(candidates)
                node = candidates[candidate_index]
                metadata = {
                    int(item["node_id"]): item
                    for item in core.utility_candidate_metadata(organized, batch_index=0)
                }[node]
                utility_proxy = _scalar(core.predict_utility(
                    core.utility_features(organized, node, batch_index=0, detach=True)
                ))
                alternatives = {}
                for mode in ("parent", "children"):
                    changed = core.prepare(native.organization.scene, forced_frontier={0: (node, mode)})
                    alternatives[mode] = _score_spatial_variant(
                        model, provider, prediction, batch, changed,
                    )
                parent_loss = alternatives["parent"]["native_losses"]["raw_weighted_native_total"]
                child_loss = alternatives["children"]["native_losses"]["raw_weighted_native_total"]
                raw_gain = float(parent_loss - child_loss)
                proxy_action = "children" if utility_proxy > 0.0 else "parent"
                observed_action = "children" if raw_gain > 0.0 else "parent"
                dev_organizer_rows.append(
                    {
                        "case_id": case_id,
                        "module_count": int(row["module_count"]),
                        "base_native_metrics": row,
                        "selection": "one deterministic internal node per retained DEV4 order; no target/error-based node selection",
                        "selected_candidate": dict(metadata),
                        "candidate_count": len(candidates),
                        "predicted_parent_minus_children_signed_gain_proxy": utility_proxy,
                        "parent_minus_children_raw_weighted_native_gain": raw_gain,
                        "proxy_action": proxy_action,
                        "observed_native_better_action": observed_action,
                        "bounded_single_candidate_native_regret": abs(raw_gain) if proxy_action != observed_action else 0.0,
                        "same_target_masks_queries_and_dependent_reprepare": True,
                        "alternatives": alternatives,
                        "actual_cut_summary": {
                            "stage0": _frontier_cut_summary(organized.layers[0][0]),
                            "stage1": _frontier_cut_summary(organized.layers[1][0]),
                        },
                        "persistent_spatial_prepared_tensor_storage_bytes": _spatial_prepared_tensor_bytes(organized),
                        "work_counts": {
                            "spatial_prepare": dict(organized.work_counts),
                            "native_prediction": dict(auxiliary.get("work", {})),
                        },
                        "optimizer_updates": 0,
                    }
                )
    if tuple(observed_dev) != dev_ids:
        raise ValueError("Round13 DEV22 panel was not evaluated exactly once in provider membership order.")
    if tuple(row["case_id"] for row in dev_organizer_rows) != dev_detail_ids:
        raise ValueError("Round13 DEV4 utility panel was not evaluated once per retained detail case.")
    if set(detail_rows) != {*surface_train_ids, *dev_detail_ids}:
        raise ValueError("Round13 retained TRAIN4/DEV4 surface detail was not fully populated.")

    risk_rows = [train_rows_by_id[case_id] for case_id in risk_train_ids]
    risk_records = [train_records_by_id[case_id] for case_id in risk_train_ids]
    risk_baseline_rows = [train_baseline_rows_by_id[case_id] for case_id in risk_train_ids]
    dev_rows = [dev_rows_by_id[case_id] for case_id in dev_ids]
    dev_records = [dev_records_by_id[case_id] for case_id in dev_ids]
    dev_baseline_rows = [dev_baseline_rows_by_id[case_id] for case_id in dev_ids]
    module_counts = {row["case_id"]: int(row["module_count"]) for row in (*risk_rows, *dev_rows)}
    dev_module_counts = tuple(sorted({module_counts[case_id] for case_id in dev_ids}))
    for case_id in risk_train_ids[:6]:
        if module_counts[case_id] != 1:
            raise ValueError(f"Retained TRAIN risk panel expected {case_id} to be M1.")
    for case_id in risk_train_ids[6:]:
        if module_counts[case_id] != 2:
            raise ValueError(f"Retained TRAIN risk panel expected {case_id} to be M2.")

    reducer = getattr(provider, "reduce_native_evaluation_metrics", None)
    if not callable(reducer):
        raise TypeError("Spatial native provider must expose its maintained evaluation reducer.")
    train_reduction = dict(reducer(
        risk_records, split="train", case_keys=risk_train_ids,
    ))
    dev_reduction = dict(reducer(
        dev_records, split="validation", case_keys=dev_ids,
    ))
    train_baseline_records = [train_baseline_records_by_id[case_id] for case_id in risk_train_ids]
    dev_baseline_records = [dev_baseline_records_by_id[case_id] for case_id in dev_ids]
    train_metrics = {
        "TRAIN_M1": _summarize_nine_observables(
            [row for row in risk_rows if int(row["module_count"]) == 1], known_module_counts=(1, 2),
        ),
        "TRAIN_M2": _summarize_nine_observables(
            [row for row in risk_rows if int(row["module_count"]) == 2], known_module_counts=(1, 2),
        ),
        "TRAIN_M1_M2_combined": _summarize_nine_observables(risk_rows, known_module_counts=(1, 2)),
        "DEV22": _summarize_nine_observables(dev_rows, known_module_counts=dev_module_counts),
    }
    baseline_metrics = {
        "TRAIN_M1": _summarize_nine_observables(
            [row for row in risk_baseline_rows if int(row["module_count"]) == 1], known_module_counts=(1, 2),
        ),
        "TRAIN_M2": _summarize_nine_observables(
            [row for row in risk_baseline_rows if int(row["module_count"]) == 2], known_module_counts=(1, 2),
        ),
        "TRAIN_M1_M2_combined": _summarize_nine_observables(
            risk_baseline_rows, known_module_counts=(1, 2),
        ),
        "DEV22": _summarize_nine_observables(dev_baseline_rows, known_module_counts=dev_module_counts),
    }

    from . import contextual_frontier_responses as responses
    stencils, atlas_metadata, atlas_root = responses._load_stencils(provider)
    if len(stencils) != 8:
        raise ValueError("Round13 physical response evaluation must reuse all eight saved atlas families.")
    first_roles = stencils[responses._ALL_FAMILIES[0]].baseline.output.roles
    floors, fit_families, floor_source = responses._response_native_floors(model, provider, first_roles)
    cohort_plan = responses._cohort_plan(fit_families)
    response_results: dict[str, Any] = {}
    response_metrics: dict[str, Any] = {}
    response_panels: dict[str, Any] = {}
    with torch.no_grad():
        for family_id in responses._ALL_FAMILIES:
            rows, metrics, panel_data = responses._evaluate_family(
                model,
                stencils[family_id],
                family_id,
                fit_families=fit_families,
                device=device_obj,
                dtype=parameter.dtype,
                floors=floors,
            )
            response_results[family_id] = rows
            response_metrics[family_id] = metrics
            response_panels[family_id] = panel_data
    response_representatives = responses._representative_families(
        response_results, fit_families, limit=3,
    )
    response_panel_bytes = responses._panel_export_bytes(
        response_representatives, response_panels, response_results,
    )
    response_invalid = [
        {
            "family_id": family_id,
            "response": label,
            "component": component,
            "status": metric.get("status"),
            "nonfinite_prediction_count": metric.get("nonfinite_prediction_count", 0),
            "nonfinite_reference_count": metric.get("nonfinite_reference_count", 0),
        }
        for family_id, labels in response_metrics.items()
        for label, components in labels.items()
        for component, metric in components.items()
        if metric.get("status") != "valid"
    ]
    response_summary = {
        "schema": "thermal_spatial_interaction_group_response_evaluation_v1",
        "status": "completed_with_invalid_component_metrics" if response_invalid else "completed",
        "atlas_root": atlas_root,
        "atlas_family_ids": list(responses._ALL_FAMILIES),
        "response_cohort_contract": {
            "fit_train_family_ids": list(fit_families),
            "train_reference_only_family_ids": list(cohort_plan.get("train_reference_only", ())),
            "response_validation_only_family_ids": list(cohort_plan["response_validation_only"]),
            "cohort_labels": {cohort: list(ids) for cohort, ids in cohort_plan.items()},
            "original_source_partition": "all eight saved atlas anchors are original source TRAIN families",
            "response_validation_scope": "held out from response fitting; not source DEV or TEST",
        },
        "native_train_floors": floors,
        "native_floor_source": floor_source,
        "metric_validity": {
            "invalid_component_count": len(response_invalid),
            "invalid_components": response_invalid,
        },
        "response_summaries": responses._summarize_families(
            response_results, response_metrics, fit_families,
        ),
        "families": response_results,
        "atlas_families": {
            family_id: {
                "atlas_anchor_id": atlas_metadata[family_id].get("anchor_id"),
                "physical_family_id": stencils[family_id].physical_family_id,
                "source_partition": getattr(stencils[family_id].split, "value", stencils[family_id].split),
                "response_cohort": responses._cohort_for_family(family_id, fit_families),
            }
            for family_id in responses._ALL_FAMILIES
        },
        "representative_panel_family_ids": list(response_representatives),
        "plots": "not generated",
    }
    control_probe, control_arrays = _spatial_control_probe(
        model,
        stencils["0348"],
        response_panels["0348"]["panel"],
        device=device_obj,
        dtype=parameter.dtype,
    )
    response_summary["bounded_control_probe"] = control_probe["source_column_probe"]
    response_summary["homogeneous_boundary_zero_heat"] = control_probe["homogeneous_boundary_zero_heat"]
    for name, value in control_arrays.items():
        detail_arrays[f"control_0348__{name}"] = np.asarray(value)

    detail_buffer = io.BytesIO()
    np.savez_compressed(detail_buffer, **detail_arrays)
    detail_bytes = detail_buffer.getvalue()
    dev_ids_by_count = {
        f"M{count}": [case_id for case_id in dev_ids if module_counts.get(case_id) == count]
        for count in dev_module_counts
    }
    result = {
        "schema": "thermal_spatial_interaction_groups_round13_evaluation_v1",
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": _checkpoint_selection(checkpoint_path, payload),
        "checkpoint_epoch": epoch,
        "arm": arm,
        "experiment_identity": {
            "workflow": identity.get("workflow"),
            "config_name": identity.get("config_name"),
            "training_membership": identity.get("training_membership"),
            "validation_scope": provider_identity.get("validation_scope"),
            "normalization_fit_scope": provider_identity.get("normalization_fit_scope"),
        },
        "native_evaluation": {
            "metrics": train_metrics,
            "fixed_train_risk_case_ids": list(risk_train_ids),
            "train_case_ids_evaluated_for_organizer_and_detail": list(train_ids),
            "dev22_case_ids": list(dev_ids),
            "dev22_case_ids_by_module_count": dev_ids_by_count,
            "dev22_module_counts": list(dev_module_counts),
            "fixed_train_risk_case_ids_by_module_count": {
                "M1": list(risk_train_ids[:6]), "M2": list(risk_train_ids[6:]),
            },
            "maintained_reductions": {
                "TRAIN_M1_M2_panel": train_reduction,
                "DEV22": dev_reduction,
            },
            "selected_train_mean_baseline": {
                "means": train_means,
                "metrics": baseline_metrics,
                "scope": "constant selected-TRAIN physical means, evaluated with identical native batches and masks",
            },
            "case_rows": {
                "TRAIN_M1_M2_panel": risk_rows,
                "DEV22": dev_rows,
            },
        },
        "metrics": train_metrics,
        "surface_residual_detail": {
            "case_ids": [*surface_train_ids, *dev_detail_ids],
            "TRAIN4_case_ids": list(surface_train_ids),
            "DEV4_case_ids": list(dev_detail_ids),
            "definition": "per-module signed residual mean plus demeaned circumferential residual on identical valid native angles",
            "coordinate_export_policy": {
                "coordinate_frame": "prepared Thermal physical rectangular xy frame",
                "fluid": "direct batch.receivers.fluid_xy coordinates in receiver row order",
                "material": "exact reconstruction: module_coordinates + module_characteristic_lengths * material_local_query_coordinates; Thermal v2 characteristic lengths are the physical module radius on both axes",
                "module_support": "module coordinates, IDs, masks, characteristic lengths, and physical heat are saved for the four interaction intervention cases",
            },
            "arrays_file": "native_detail.npz",
            "cases": detail_rows,
        },
        "native_detail_arrays_file": "native_detail.npz",
        "response_panel_arrays_file": "response_delta_panels.npz",
        "organizer24_same_target_parent_descendant": {
            "case_ids": list(organizer_ids),
            "selection": "Round12 retained input-only organizer panel; one deterministic internal target per case",
            "candidate_proxy_definition": "core.predict_utility fixed-width target-free proxy; positive means descendant split benefit",
            "measured_gain_definition": "raw weighted native parent loss minus child-cover weighted native loss on identical target, masks, query points and downstream reprepare",
            "cases": organizer_rows,
        },
        "dev4_same_target_utility_probe": {
            "case_ids": list(dev_detail_ids),
            "selection": "one deterministic internal node per retained DEV4 order; same native batch targets, masks and queries for parent/descendants",
            "cases": dev_organizer_rows,
        },
        "fixed_cover_interventions": {
            "case_ids": list(intervention_ids),
            "cover_names": ["root_K1", "nearest_K8", "nearest_K32", "nearest_K128", "leaf_K192"],
            "selection": "five fixed, deterministic legal hierarchy covers, independent of target errors",
            "cases": cover_rows,
        },
        "mode_relay_organizer_relation_ablations": {
            "case_ids": list(intervention_ids),
            "ablations": [
                "zero_final_read_slopes",
                "zero_spatial_slopes_relay_and_final",
                "bypass_group_to_M_relay",
                "remove_organizer_relation_features",
            ],
            "cases": ablation_rows,
        },
        "fixed_cover_zero_mode_probe": {
            "case_ids": list(intervention_ids),
            "cover_name": "nearest_K32",
            "cover_cardinality_unchanged": True,
            "same_prepared_cut_and_full_group_to_M_relay": True,
            "cases": fixed_cover_mode_rows,
        },
        "cost": {
            "status": "measured" if include_cost else "not_requested",
            "cases": cost_rows,
            "bounded_method": "one warmup plus three synchronized complete provider.predict_native repeats on TRAIN0006/M1 and TRAIN0051/M2",
        },
        "physical_response_evaluation": response_summary,
        "response_summary": response_summary,
        "optional_ranking": {
            "status": "omitted",
            "reason": "No bounded candidate-ranking panel was run in this terminal evaluation.",
        },
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "test_access": False,
        "plots": "not generated",
        "storage_reservation": {
            "max_output_bytes": _SPATIAL_OUTPUT_LIMIT,
            "free_bytes_before": _filesystem_free_bytes(output_path.parent),
            "peak_temporary_disk_bytes": 0,
        },
    }
    result_bytes = _json_bytes(result)
    planned_bytes = len(result_bytes) + len(detail_bytes) + len(response_panel_bytes)
    if planned_bytes > _SPATIAL_OUTPUT_LIMIT:
        raise OSError(
            f"Round13 numerical evidence needs {planned_bytes} bytes; cap is {_SPATIAL_OUTPUT_LIMIT}."
        )
    output_path.mkdir(parents=False, exist_ok=False)
    files = {
        "result.json": result_bytes,
        "native_detail.npz": detail_bytes,
        "response_delta_panels.npz": response_panel_bytes,
    }
    for name, data in files.items():
        (output_path / name).write_bytes(data)
    actual_bytes = sum((output_path / name).stat().st_size for name in files)
    if actual_bytes != planned_bytes:
        raise OSError("Round13 numerical evidence byte count changed between preflight and save.")
    return {
        "output_path": str(output_path),
        "output_bytes": actual_bytes,
        "max_output_bytes": _SPATIAL_OUTPUT_LIMIT,
        **_json_safe(result),
    }


__all__ = ["evaluate_legacy_group_diagnostic", "evaluate_spatial_groups"]
