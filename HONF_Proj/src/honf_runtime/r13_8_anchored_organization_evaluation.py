"""Evaluation support for Round13-8 anchored organization reads.

The native predictor and Thermal locator remain owned by their maintained
providers. This module holds small, deterministic reductions used to select
organization rows and summarize decoder versus native boundary probes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import traceback
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _live_probe_coordinates_match(returned: Any, expected: Any) -> bool:
    """Compare a live probe tensor with its plan at the tensor's precision."""
    actual = _numpy(returned)
    target = _numpy(expected).astype(np.float64, copy=False)
    if actual.shape != target.shape:
        return False
    epsilon = np.finfo(actual.dtype).eps if actual.dtype.kind == "f" else np.finfo(np.float64).eps
    scale = max(1.0, float(np.max(np.abs(target))) if target.size else 1.0)
    tolerance = min(1.0e-5, max(1.0e-12, 8.0 * epsilon * scale))
    return bool(np.allclose(actual.astype(np.float64, copy=False), target, rtol=0.0, atol=tolerance))


def _scene_geometry_payload(scene: Any) -> dict[str, Any]:
    """Keep physical coordinates, identities and measures beside E-row traces."""
    def rows(name: str) -> Any:
        value = getattr(scene, name, None)
        if value is None:
            return None
        array = _numpy(value)
        if array.ndim and array.shape[0] == 1:
            array = array[0]
        return array.tolist()

    return {
        "coordinate_frame_key": getattr(scene, "coordinate_frame_key", None),
        "coordinate_origin": rows("coordinate_origin"),
        "coordinate_scale": rows("coordinate_scale"),
        "module": {
            "original_ids": rows("module_ids"),
            "coordinates": rows("module_coordinates"),
            "valid_mask": rows("module_mask"),
            "physical_measure": rows("module_measure"),
            "measure_reference": rows("module_measure_reference"),
            "local_radius": rows("local_radius"),
            "characteristic_lengths": rows("module_characteristic_lengths"),
            "train_normalized_input_features": rows("module_features"),
        },
        "environment": {
            "original_ids": rows("environment_ids"),
            "coordinates": rows("environment_coordinates"),
            "valid_mask": rows("environment_mask"),
            "physical_measure": rows("environment_measure"),
            "measure_reference": rows("environment_measure_reference"),
            "characteristic_lengths": rows("environment_characteristic_lengths"),
            "train_normalized_input_features": rows("environment_features"),
        },
        "operating_features_train_normalized": rows("operating_features"),
    }


def _stable_id_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if np.isfinite(numeric):
            return (0, numeric)
    return (1, str(value))


def select_deterministic_anchor_panel(
    environment_ids: Any,
    environment_coordinates: Any,
    environment_mask: Any | None = None,
    *,
    count: int = 4,
) -> tuple[Any, ...]:
    """Choose a spatially spread anchor panel using only IDs and coordinates.

    The first anchor is the smallest stable original ID; later anchors use
    farthest-point selection in per-axis normalized coordinates, with original
    ID as the deterministic tie break. Padded or non-finite rows are omitted.
    Input row order therefore does not affect the selected original IDs.
    """
    ids = _numpy(environment_ids)
    coordinates = _numpy(environment_coordinates)
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError("Anchor panel selection accepts one scene at a time.")
        ids = ids[0]
    if coordinates.ndim == 3:
        if coordinates.shape[0] != 1:
            raise ValueError("Anchor panel selection accepts one scene at a time.")
        coordinates = coordinates[0]
    if ids.ndim != 1 or coordinates.ndim != 2 or coordinates.shape[0] != ids.shape[0]:
        raise ValueError("Environment IDs and coordinates must have shapes [E] and [E,D].")
    if coordinates.shape[1] == 0:
        raise ValueError("Environment coordinates must have at least one dimension.")
    if count <= 0:
        raise ValueError("Anchor panel count must be positive.")

    if environment_mask is None:
        valid = np.ones(ids.shape[0], dtype=bool)
    else:
        valid = _numpy(environment_mask)
        if valid.ndim == 2:
            if valid.shape[0] != 1:
                raise ValueError("Anchor panel selection accepts one scene at a time.")
            valid = valid[0]
        if valid.shape != ids.shape:
            raise ValueError("Environment mask must have shape [E].")
        valid = valid.astype(bool)
    valid &= np.isfinite(coordinates).all(axis=1)
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        raise ValueError("Anchor panel selection needs at least one valid environment record.")

    keys = [_stable_id_key(ids[int(row)]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Valid environment original IDs must be unique.")
    low = coordinates[rows].min(axis=0)
    span = coordinates[rows].max(axis=0) - low
    span = np.where(span > 0, span, 1.0)
    normalized = (coordinates[rows] - low) / span
    stable_order = sorted(range(len(rows)), key=lambda position: keys[position])

    selected_positions = [stable_order[0]]
    target_count = min(int(count), len(rows))
    while len(selected_positions) < target_count:
        remaining = [position for position in stable_order if position not in selected_positions]
        distances = {
            position: min(
                float(np.square(normalized[position] - normalized[chosen]).sum())
                for chosen in selected_positions
            )
            for position in remaining
        }
        farthest = max(distances.values())
        selected_positions.append(min(
            (position for position in remaining if distances[position] == farthest),
            key=lambda position: keys[position],
        ))
    return tuple(ids[int(rows[position])].item() if isinstance(ids[int(rows[position])], np.generic)
                 else ids[int(rows[position])] for position in selected_positions)


def summarize_boundary_sweep(
    offsets_in_spacing: Sequence[float],
    values: Any,
    *,
    output_scale: Any = 1.0,
    evaluation_level: str,
) -> dict[str, Any]:
    """Measure paired continuity and first-derivative gaps around a face.

    ``values`` must come from probes that rebuilt the active locator from each
    probe's live coordinate tensor. Offsets are signed distances in local E
    spacing units. One-sided affine fits estimate the field limits and slopes
    at the face from multiple probes. Call separately for decoder output and
    native extraction so their regularity evidence cannot be conflated.
    """
    if evaluation_level not in ("decoder", "native_extraction"):
        raise ValueError("evaluation_level must be 'decoder' or 'native_extraction'.")
    offsets = _numpy(offsets_in_spacing).astype(np.float64, copy=False)
    samples = _numpy(values).astype(np.float64, copy=False)
    if offsets.ndim != 1 or samples.ndim < 2 or samples.shape[0] != offsets.size:
        raise ValueError("Boundary offsets must be [S] and values must have a matching leading [S] axis.")
    if offsets.size < 3 or not np.isfinite(offsets).all() or not np.isfinite(samples).all():
        raise ValueError("Boundary probes need at least three finite offsets and finite values.")
    if len(np.unique(offsets)) != offsets.size:
        raise ValueError("Boundary probe offsets must be unique.")
    zero_rows = np.flatnonzero(np.isclose(offsets, 0.0, rtol=0.0, atol=1e-12))
    if zero_rows.size != 1:
        raise ValueError("Boundary probe set must contain exactly one face sample at offset zero.")

    flat = samples.reshape(offsets.size, -1)
    scale = _numpy(output_scale).astype(np.float64, copy=False)
    try:
        scale = np.broadcast_to(scale, samples.shape).reshape(offsets.size, -1)
    except ValueError as exc:
        raise ValueError("output_scale must broadcast to the boundary output shape.") from exc
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Boundary output scales must be finite and positive.")

    zero_index = int(zero_rows[0])
    pairs = []
    for positive_index in np.flatnonzero(offsets > 0):
        distance = float(offsets[positive_index])
        negative = np.flatnonzero(np.isclose(offsets, -distance, rtol=1e-7, atol=1e-10))
        if negative.size != 1:
            raise ValueError("Every positive boundary offset needs one symmetric negative probe.")
        negative_index = int(negative[0])
        jump = flat[positive_index] - flat[negative_index]
        normalized_jump = jump / scale[zero_index]
        positive_slope = (flat[positive_index] - flat[zero_index]) / distance
        negative_slope = (flat[zero_index] - flat[negative_index]) / distance
        slope_gap = positive_slope - negative_slope
        pairs.append({
            "distance_in_E_spacing": distance,
            "paired_output_change_rmse": float(np.sqrt(np.mean(np.square(jump)))) if jump.size else 0.0,
            "paired_output_change_max_abs": float(np.max(np.abs(jump))) if jump.size else 0.0,
            "scaled_continuity_jump_rmse": float(np.sqrt(np.mean(np.square(normalized_jump)))) if jump.size else 0.0,
            "one_sided_slope_gap_rmse_per_spacing": (
                float(np.sqrt(np.mean(np.square(slope_gap)))) if slope_gap.size else 0.0
            ),
            "one_sided_slope_gap_max_abs_per_spacing": (
                float(np.max(np.abs(slope_gap))) if slope_gap.size else 0.0
            ),
        })
    if len(pairs) < 2:
        raise ValueError("Boundary probe set must include at least two symmetric offset pairs.")
    positive_indices = np.flatnonzero(offsets > 0)
    negative_indices = np.flatnonzero(offsets < 0)

    def fit_side(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = np.column_stack((np.ones(indices.size), offsets[indices]))
        coefficients, *_ = np.linalg.lstsq(design, flat[indices], rcond=None)
        residual = flat[indices] - design @ coefficients
        return coefficients, residual

    positive_fit, positive_residual = fit_side(positive_indices)
    negative_fit, negative_residual = fit_side(negative_indices)
    limit_gap = positive_fit[0] - negative_fit[0]
    scaled_limit_gap = limit_gap / scale[zero_index]
    slope_gap = positive_fit[1] - negative_fit[1]
    ordered_positive = positive_indices[np.argsort(offsets[positive_indices])]
    local_positive = ordered_positive[:min(3, ordered_positive.size)]
    local_negative = np.asarray([
        int(np.flatnonzero(np.isclose(offsets, -offsets[position], rtol=1e-7, atol=1e-10))[0])
        for position in local_positive
    ], dtype=np.int64)
    local_positive_fit, local_positive_residual = fit_side(local_positive)
    local_negative_fit, local_negative_residual = fit_side(local_negative)
    local_limit_gap = local_positive_fit[0] - local_negative_fit[0]
    local_scaled_limit_gap = local_limit_gap / scale[zero_index]
    local_slope_gap = local_positive_fit[1] - local_negative_fit[1]
    return {
        "schema": "honf_r13_8_boundary_sweep_summary_v1",
        "evaluation_level": evaluation_level,
        "offset_unit": "local E spacing",
        "probe_offsets_in_spacing": offsets.tolist(),
        "output_shape": list(samples.shape[1:]),
        "output_scale": scale[zero_index].reshape(samples.shape[1:]).tolist(),
        "sampled_values": samples.tolist(),
        "locator_rebuild_required": True,
        "one_sided_affine_fit": {
            "positive_limit": positive_fit[0].reshape(samples.shape[1:]).tolist(),
            "negative_limit": negative_fit[0].reshape(samples.shape[1:]).tolist(),
            "estimated_limit_gap_rmse": float(np.sqrt(np.mean(np.square(limit_gap)))) if limit_gap.size else 0.0,
            "estimated_limit_gap_max_abs": float(np.max(np.abs(limit_gap))) if limit_gap.size else 0.0,
            "scaled_limit_gap_rmse": (
                float(np.sqrt(np.mean(np.square(scaled_limit_gap)))) if scaled_limit_gap.size else 0.0
            ),
            "positive_slope_per_spacing": positive_fit[1].reshape(samples.shape[1:]).tolist(),
            "negative_slope_per_spacing": negative_fit[1].reshape(samples.shape[1:]).tolist(),
            "slope_gap_rmse_per_spacing": float(np.sqrt(np.mean(np.square(slope_gap))) if slope_gap.size else 0.0),
            "slope_gap_max_abs_per_spacing": float(np.max(np.abs(slope_gap))) if slope_gap.size else 0.0,
            "positive_fit_rmse": float(np.sqrt(np.mean(np.square(positive_residual)))) if positive_residual.size else 0.0,
            "negative_fit_rmse": float(np.sqrt(np.mean(np.square(negative_residual)))) if negative_residual.size else 0.0,
            "fit_scope": "broad diagnostic using all nonzero paired offsets; nonlinear curvature can bias extrapolated limits",
            "fit": "independent least-squares affine fit on each side using all nonzero paired offsets",
        },
        "near_face_affine_fit": {
            "positive_probe_offsets_in_spacing": offsets[local_positive].tolist(),
            "negative_probe_offsets_in_spacing": offsets[local_negative].tolist(),
            "symmetric_pair_count": int(local_positive.size),
            "positive_limit": local_positive_fit[0].reshape(samples.shape[1:]).tolist(),
            "negative_limit": local_negative_fit[0].reshape(samples.shape[1:]).tolist(),
            "estimated_limit_gap_rmse": (
                float(np.sqrt(np.mean(np.square(local_limit_gap)))) if local_limit_gap.size else 0.0
            ),
            "estimated_limit_gap_max_abs": (
                float(np.max(np.abs(local_limit_gap))) if local_limit_gap.size else 0.0
            ),
            "scaled_limit_gap_rmse": (
                float(np.sqrt(np.mean(np.square(local_scaled_limit_gap)))) if local_scaled_limit_gap.size else 0.0
            ),
            "positive_slope_per_spacing": local_positive_fit[1].reshape(samples.shape[1:]).tolist(),
            "negative_slope_per_spacing": local_negative_fit[1].reshape(samples.shape[1:]).tolist(),
            "slope_gap_rmse_per_spacing": (
                float(np.sqrt(np.mean(np.square(local_slope_gap)))) if local_slope_gap.size else 0.0
            ),
            "positive_fit_rmse": (
                float(np.sqrt(np.mean(np.square(local_positive_residual)))) if local_positive_residual.size else 0.0
            ),
            "negative_fit_rmse": (
                float(np.sqrt(np.mean(np.square(local_negative_residual)))) if local_negative_residual.size else 0.0
            ),
            "fit_scope": "closest up to three symmetric nonzero offset pairs, with actual offsets retained",
            "fit": "independent least-squares affine fit on each side using the nearest symmetric probe pairs",
        },
        "continuity_and_derivative_pairs": pairs,
    }


_ANCHORED_WORK_SUFFIXES = {
    "directory_rows": ("prepared_e_directory_rows",),
    "directory_full_entries": ("prepared_e_directory_full_entries",),
    "directory_joint95_entries": ("prepared_e_directory_joint95_entries",),
    "directory_sort_entries": ("prepared_e_directory_sort_entries",),
    "directory_compile_pair_rows": ("prepared_e_directory_compile_pair_rows",),
    "directory_positive_exchange_entries": ("prepared_e_directory_positive_exchange_entries",),
    "locator_rows": ("query_anchor_locator_rows",),
    "anchor_index_rows": ("query_anchor_index_rows",),
    "unique_raw_score_pairs_M": ("query_unique_union_raw_score_pairs_M",),
    "unique_raw_score_pairs_E": ("query_unique_union_raw_score_pairs_E",),
    "anchor_member_pairs_M": ("query_anchor_member_pairs_M",),
    "anchor_member_pairs_E": ("query_anchor_member_pairs_E",),
    "anchor_normalization_rows_M": ("query_anchor_normalization_rows_M",),
    "anchor_normalization_rows_E": ("query_anchor_normalization_rows_E",),
    "anchor_typed_gate_rows_M": ("query_anchor_typed_gate_rows_M",),
    "anchor_typed_gate_rows_E": ("query_anchor_typed_gate_rows_E",),
    "anchor_coalesced_value_pairs_M": ("query_anchor_coalesced_value_pairs_M",),
    "anchor_coalesced_value_pairs_E": ("query_anchor_coalesced_value_pairs_E",),
    "anchor_value_reduction_pairs_M": ("query_anchor_value_reduction_pairs_M",),
    "anchor_value_reduction_pairs_E": ("query_anchor_value_reduction_pairs_E",),
    "M_member_query_affine_rows": ("query_M_member_query_affine_rows",),
    "E_member_query_affine_rows": ("query_E_member_query_affine_rows",),
    "joint_field_head_rows": ("query_joint_field_head_rows",),
}


def summarize_anchored_work_counts(work_counts: Mapping[str, Any]) -> dict[str, Any]:
    """Keep unique union scoring distinct from anchor-local read work."""
    if not isinstance(work_counts, Mapping):
        raise TypeError("Anchored work counts must be a mapping.")
    numeric: dict[str, float] = {}
    for raw_name, raw_value in work_counts.items():
        if isinstance(raw_value, bool):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            if value < 0:
                raise ValueError(f"Anchored work counter {raw_name!r} must be nonnegative.")
            numeric[str(raw_name)] = value

    summary: dict[str, Any] = {}
    missing = []
    for output_name, suffixes in _ANCHORED_WORK_SUFFIXES.items():
        matched = [value for name, value in numeric.items()
                   if any(name == suffix or name.endswith(suffix) for suffix in suffixes)]
        if not matched:
            missing.append(output_name)
            continue
        summary[output_name] = int(sum(matched)) if all(float(value).is_integer() for value in matched) else sum(matched)
    return {
        "schema": "honf_r13_8_separated_anchored_work_v1",
        "observed": summary,
        "missing_counters": missing,
        "source_work_counts": dict(work_counts),
        "semantics": {
            "unique_raw_score_pairs": "unique query/original-member scores shared across active anchors within this call",
            "anchor_member_pairs": "sum of row-local query/member memberships across anchors; may exceed unique scores",
            "anchor_normalization_rows": "separate per-anchor role distributions",
            "coalesced_value_pairs": "member coefficients after anchor-weighted coalescing",
            "anchor_value_reduction_pairs": "actual per-anchor value reduction operations",
        },
    }


def require_complete_anchored_work_counts(work_counts: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize actual A/L counters and reject incomplete measured work."""
    summary = summarize_anchored_work_counts(work_counts)
    missing = summary["missing_counters"]
    if missing:
        raise ValueError("Anchored evaluation is incomplete; missing measured work counters: "
                         + ", ".join(missing))
    return summary


def _synchronize(device: Any) -> None:
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def _mode_context(model: Any, mode: str):
    core = getattr(model, "core", None)
    read_execution = getattr(core, "read_execution", None)
    return read_execution(mode) if callable(read_execution) else nullcontext()


def _stable_case_name(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(case_id))
    return safe or "case"


def _write_representative_arrays(split_result: dict[str, Any], directory: Path) -> dict[str, Any]:
    arrays_by_id = split_result.pop("representative_arrays")
    exports = []
    raw_bytes = 0
    for case_id, arrays in arrays_by_id.items():
        name = f"case_{_stable_case_name(case_id)}.npz"
        arrays = {name: _numpy(value) for name, value in arrays.items()}
        raw_bytes += sum(int(value.nbytes) for value in arrays.values())
        with (directory / name).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        exports.append({"case_id": str(case_id), "path": name})
    split_result["representative_exports"] = exports
    return {"case_count": len(exports), "uncompressed_bytes": raw_bytes,
            "compressed_bytes": sum((directory / item["path"]).stat().st_size for item in exports)}


def _paired_read_penalties(full: Mapping[str, Any], restricted: Mapping[str, Any]) -> dict[str, Any]:
    """Pair native per-case metrics and retain signed admission penalties."""
    full_metrics = full.get("metrics")
    restricted_metrics = restricted.get("metrics")
    if not isinstance(full_metrics, Mapping) or not isinstance(restricted_metrics, Mapping):
        raise TypeError("Full/restricted native metrics must be mappings.")
    if set(full_metrics) != set(restricted_metrics):
        raise ValueError("Full/restricted native metric names differ.")
    result = {}
    for metric_name in sorted(full_metrics):
        full_rows = full_metrics[metric_name].get("per_case", ())
        restricted_rows = restricted_metrics[metric_name].get("per_case", ())
        restricted_by_id = {str(row["case_id"]): row for row in restricted_rows}
        if len(restricted_by_id) != len(restricted_rows):
            raise ValueError(f"Restricted metric {metric_name} contains duplicate case IDs.")
        paired = []
        for row in full_rows:
            case_id = str(row["case_id"])
            if case_id not in restricted_by_id:
                raise ValueError(f"Restricted metric {metric_name} is missing case {case_id}.")
            other = restricted_by_id[case_id]
            if int(other["module_count"]) != int(row["module_count"]):
                raise ValueError(f"Full/restricted module-count mismatch for case {case_id}.")
            paired.append({
                "case_id": case_id,
                "module_count": int(row["module_count"]),
                "restricted_minus_full_rmse": float(other["value"]) - float(row["value"]),
                "full_rmse": float(row["value"]),
                "restricted_rmse": float(other["value"]),
            })
        if len(paired) != len(restricted_rows):
            raise ValueError(f"Full/restricted metric {metric_name} case panels differ.")
        deltas = np.asarray([row["restricted_minus_full_rmse"] for row in paired], dtype=np.float64)
        result[metric_name] = {
            "case_count": len(paired),
            "unit": full_metrics[metric_name].get("unit"),
            "mean_restricted_minus_full": float(deltas.mean()) if deltas.size else None,
            "p90_restricted_minus_full": float(np.percentile(deltas, 90)) if deltas.size else None,
            "worst_positive_penalty": float(max(0.0, float(deltas.max()))) if deltas.size else None,
            "improved_case_count": int((deltas < 0).sum()),
            "regressed_case_count": int((deltas > 0).sum()),
            "per_case": paired,
        }
    return result


def _accumulate_numeric_counts(total: dict[str, float], values: Any) -> None:
    def visit(raw_values: Any, prefix: str) -> None:
        if isinstance(raw_values, Mapping):
            for name, raw in raw_values.items():
                visit(raw, f"{prefix}{name}__")
            return
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes, bytearray)):
            for index, raw in enumerate(raw_values):
                visit(raw, f"{prefix}{index:04d}__")
            return
        if isinstance(raw_values, (bool, np.bool_)):
            return
        try:
            value = float(raw_values)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            name = prefix[:-2] if prefix.endswith("__") else prefix
            total[name] = total.get(name, 0.0) + value

    visit(values, "")


def _native_metric_deltas(baseline: Mapping[str, Any], intervention: Mapping[str, Any]) -> dict[str, Any]:
    if set(baseline) != set(intervention):
        raise ValueError("Baseline and intervention native observable sets differ.")
    deltas: dict[str, Any] = {}
    for name in sorted(baseline):
        left, right = baseline[name], intervention[name]
        if left.get("unit") != right.get("unit"):
            raise ValueError(f"Baseline and intervention units differ for {name}.")
        left_rows = left.get("per_case", ())
        right_rows = right.get("per_case", ())
        if len(left_rows) != 1 or len(right_rows) != 1:
            raise ValueError(f"Same-state intervention {name} must retain one paired native case.")
        if str(left_rows[0]["case_id"]) != str(right_rows[0]["case_id"]):
            raise ValueError(f"Baseline and intervention case IDs differ for {name}.")
        deltas[name] = {
            "case_id": str(left_rows[0]["case_id"]),
            "unit": left.get("unit"),
            "intervention_minus_baseline_rmse": (
                float(right_rows[0]["value"]) - float(left_rows[0]["value"])
            ),
            "baseline_rmse": float(left_rows[0]["value"]),
            "intervention_rmse": float(right_rows[0]["value"]),
        }
    return deltas


def _thermal_boundary_coordinates(scene: Any, locator_mode: str) -> tuple[dict[str, Any], ...]:
    """Choose two central artificial grid faces in the prepared physical domain."""
    if locator_mode not in ("e_anchor", "local_e_cover"):
        raise ValueError(f"Unsupported thermal boundary locator mode: {locator_mode!r}.")
    coordinates = _numpy(scene.environment_coordinates)
    origins = _numpy(scene.coordinate_origin)
    scales = _numpy(scene.coordinate_scale)
    if coordinates.ndim != 3 or coordinates.shape[0] != 1 or coordinates.shape[1] != 24 * 8:
        raise ValueError("Thermal boundary probes require one complete 24x8 E lattice.")
    if origins.shape != (1, 2) or scales.shape != (1, 2):
        raise ValueError("Thermal boundary probes require saved coordinate_origin and coordinate_scale.")
    origin = _numpy(origins[0]).astype(np.float64, copy=False)
    extent = _numpy(scales[0]).astype(np.float64, copy=False)
    if not np.isfinite(origin).all() or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("Thermal coordinate bounds must be finite with positive extents.")
    dx, dy = extent[0] / 24.0, extent[1] / 8.0
    centers_x = origin[0] + (np.arange(24, dtype=np.float64) + 0.5) * dx
    centers_y = origin[1] + (np.arange(8, dtype=np.float64) + 0.5) * dy
    offsets = np.asarray(
        [-0.5, -0.2, -0.05, -0.005, -0.0005, 0.0, 0.0005, 0.005, 0.05, 0.2, 0.5],
        dtype=np.float64,
    )
    if locator_mode == "e_anchor":
        x_face, x_offset = origin[0] + 12.0 * dx, 12.0
        y_face, y_offset = origin[1] + 4.0 * dy, 4.0
        center_x, center_y = centers_x[12], centers_y[4]
        probes = (
            ("x_hard_ownership_face", "x", "hard_cell_ownership_face", x_face, centers_y[4], dx, offsets, 12.0),
            ("y_hard_ownership_face", "y", "hard_cell_ownership_face", y_face, centers_x[12], dy, offsets, 4.0),
        )
    else:
        center_x, center_y = centers_x[12], centers_y[4]
        ownership_x = origin[0] + 12.0 * dx
        ownership_y = origin[1] + 4.0 * dy
        probes = (
            ("x_center_knot", "x", "smooth_cover_center_knot", center_x, center_y, dx, offsets, 12.5),
            ("y_center_knot", "y", "smooth_cover_center_knot", center_y, center_x, dy, offsets, 4.5),
            ("x_former_ownership_face", "x", "smooth_cover_former_ownership_face",
             ownership_x, center_y, dx, offsets, 12.0),
            ("y_former_ownership_face", "y", "smooth_cover_former_ownership_face",
             ownership_y, center_x, dy, offsets, 4.0),
            ("x_outer_strip_junction", "x", "smooth_cover_outer_strip_junction",
             centers_x[0], center_y, dx, offsets, 0.5),
            ("y_outer_strip_junction", "y", "smooth_cover_outer_strip_junction",
             centers_y[0], center_x, dy, offsets, 0.5),
        )
    result = []
    for name, axis, kind, face, orthogonal, spacing, probe_offsets, grid_index in probes:
        if axis == "x":
            coordinates_for_probe = np.column_stack((
                np.full(probe_offsets.shape, face) + probe_offsets * spacing,
                np.full(probe_offsets.shape, orthogonal),
            ))
        else:
            coordinates_for_probe = np.column_stack((
                np.full(probe_offsets.shape, orthogonal),
                np.full(probe_offsets.shape, face) + probe_offsets * spacing,
            ))
        result.append({
            "boundary_name": name,
            "boundary_axis": axis,
            "boundary_kind": kind,
            "offsets_in_spacing": probe_offsets.tolist(),
            "probe_coordinates": coordinates_for_probe.tolist(),
            "face_coordinate": float(face),
            "orthogonal_coordinate": float(orthogonal),
            "grid_index": float(grid_index),
            "center_point_coordinates": [float(center_x), float(center_y)],
        })
    return tuple(result)


def _native_metric_summary(provider: Any, batch: Any, prediction: Any, auxiliary: Any) -> dict[str, Any]:
    from . import contextual_frontier_evaluation as native

    record = provider.validation_metrics(prediction, batch.targets, auxiliary)
    rows = record.get("case_rows") if isinstance(record, Mapping) else None
    if not isinstance(rows, Sequence) or len(rows) != 1:
        raise ValueError("Anchored native diagnostics need one maintained per-case metric row.")
    return native._summarize_nine_observables(rows)


def _reassignment_intervention(scene: Any, trace_response: Mapping[str, Any]) -> dict[str, Any]:
    """Choose a stable same-row member move from actual full-read trace support."""
    trace = trace_response.get("trace")
    if not isinstance(trace, Mapping):
        raise TypeError("Anchored trace response must expose its packed core trace mapping.")
    slots = _numpy(trace.get("anchor_indices"))
    weights = _numpy(trace.get("anchor_weights"))
    read_rows = _numpy(trace.get("anchor_read_indices"))
    member_read_ids = _numpy(trace.get("anchor_read_member_read_index"))
    anchor_slots = _numpy(trace.get("anchor_read_E_slots"))
    original_ids = _numpy(trace.get("anchor_read_original_id"))
    is_module = _numpy(trace.get("anchor_read_is_module"))
    if (slots.ndim != 3 or weights.shape != slots.shape or read_rows.ndim != 2
            or read_rows.shape[1] != 3 or anchor_slots.shape != (read_rows.shape[0],)
            or member_read_ids.ndim != 1 or original_ids.shape != member_read_ids.shape
            or is_module.shape != member_read_ids.shape
            or (member_read_ids.size and ((member_read_ids < 0).any()
                                          or (member_read_ids >= read_rows.shape[0]).any()))):
        shapes = {
            "anchor_indices": slots.shape,
            "anchor_weights": weights.shape,
            "anchor_read_indices": read_rows.shape,
            "anchor_read_E_slots": anchor_slots.shape,
            "anchor_read_member_read_index": member_read_ids.shape,
            "anchor_read_original_id": original_ids.shape,
            "anchor_read_is_module": is_module.shape,
        }
        raise ValueError(f"Packed anchor trace lacks the fields needed for deterministic reassignment: {shapes}.")
    if slots.shape[0] != 1 or slots.shape[1] < 1:
        raise ValueError("Reassignment selection requires one scene and a selected query row.")
    active_slots = np.flatnonzero((weights[0, 0] > 0) & (slots[0, 0] >= 0))
    if active_slots.size == 0:
        raise ValueError("Selected query has no active E anchor for reassignment.")
    read_k = int(active_slots[0])
    anchor_slot = int(slots[0, 0, read_k])
    environment_ids = _numpy(scene.environment_ids)[0]
    module_ids = _numpy(scene.module_ids)[0]
    environment_mask = _numpy(scene.environment_mask)[0].astype(bool)
    module_mask = _numpy(scene.module_mask)[0].astype(bool)
    anchor_id_value = environment_ids[anchor_slot]
    anchor_id = anchor_id_value.item() if isinstance(anchor_id_value, np.generic) else anchor_id_value
    read_rows_for_anchor = np.flatnonzero(
        (read_rows[:, 0] == 0) & (read_rows[:, 1] == 0) & (read_rows[:, 2] == read_k)
    )
    if read_rows_for_anchor.size != 1:
        raise ValueError("Trace must contain exactly one packed read for the deterministic query/anchor pair.")
    member_rows = np.flatnonzero(member_read_ids == read_rows_for_anchor[0])
    if member_rows.size == 0 or int(anchor_slots[read_rows_for_anchor[0]]) != anchor_slot:
        raise ValueError("Trace has no member entries for the deterministic query/anchor pair.")
    source_candidates = []
    current_row_members = set()
    for row in member_rows:
        member_id_value = original_ids[row]
        member_id = member_id_value.item() if isinstance(member_id_value, np.generic) else member_id_value
        role = "M" if bool(is_module[row]) else "E"
        current_row_members.add((role, member_id))
        if role == "E" and member_id == anchor_id:
            continue
        source_candidates.append((role, _stable_id_key(member_id), member_id))
    if not source_candidates:
        raise ValueError("Selected full E row has no non-self member eligible for reassignment.")
    source_role, _source_key, source_id = min(source_candidates, key=lambda item: (item[0] != "M", item[1]))

    inventories = (("M", module_ids, module_mask), ("E", environment_ids, environment_mask))
    target_candidates = []
    for role, ids, mask in inventories:
        for identity in ids[mask]:
            value = identity.item() if isinstance(identity, np.generic) else identity
            if role == source_role and value == source_id:
                continue
            if role == "E" and value == anchor_id:
                continue
            if (role, value) in current_row_members:
                continue
            target_candidates.append((role, _stable_id_key(value), value))
    if not target_candidates:
        return {
            "kind": "remove_member",
            "requested_kind": "reassign_member",
            "fallback_reason": (
                "the selected full E row already contains every distinct eligible present target member"
            ),
            "member_role": source_role,
            "member_id": int(source_id),
            "anchor_id": int(anchor_id),
        }
    same_role = [item for item in target_candidates if item[0] == source_role]
    target_role, _target_key, target_id = min(same_role or target_candidates, key=lambda item: item[1])
    return {
        "kind": "reassign_member",
        "source_role": source_role,
        "source_id": int(source_id),
        "target_role": target_role,
        "target_id": int(target_id),
        "anchor_id": int(anchor_id),
    }


def _invoke_anchored_boundary_sweep(
    boundary_fn: Any,
    model: Any,
    batch: Any,
    query_index: int,
    probe: Mapping[str, Any],
    read_mode: str,
) -> Any:
    """Pass the evaluated read branch into the provider-owned boundary decoder."""
    if read_mode not in ("full", "restricted"):
        raise ValueError("Boundary read mode must be 'full' or 'restricted'.")
    return boundary_fn(
        model,
        batch,
        query_index,
        probe["probe_coordinates"],
        offsets_in_spacing=probe["offsets_in_spacing"],
        read_mode=read_mode,
    )


def _select_validation_panel(provider: Any, requested: Sequence[str] | None = None) -> tuple[str, ...]:
    ids = tuple(map(str, getattr(provider, "validation_case_ids", ())))
    if not ids:
        raise ValueError("Anchored native evaluation requires exposed validation case IDs.")
    if requested is not None:
        selected = tuple(map(str, requested))
        if not selected or len(selected) != len(set(selected)) or any(case_id not in ids for case_id in selected):
            raise ValueError("Requested representative cases must be unique members of the exposed validation panel.")
        return selected
    from .member_routing_evaluation import select_validation_representatives

    return select_validation_representatives(provider, limit=4)


def _validate_checkpoint_population(model: Any, provider: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Anchored native evaluation requires saved experiment_identity metadata.")
    workflow = str(identity.get("workflow", ""))
    architecture = str(getattr(model, "architecture", ""))
    if architecture == "R13-7-U-ME-SingleDistributionJoint95":
        arm = "U"
        if workflow != "interaction_reused_single_distribution_member_routing_training_v1":
            raise ValueError("The U evaluation checkpoint has an unexpected saved workflow identity.")
    elif architecture in ("R13-8-A-EAnchor", "R13-8-L-LocalECover"):
        arm = "A" if architecture == "R13-8-A-EAnchor" else "L"
        if workflow != "interaction_reused_e_anchored_member_routing_training_v1":
            raise ValueError("The A/L evaluation checkpoint has an unexpected saved workflow identity.")
    else:
        raise ValueError(f"Unsupported R13-8 comparison architecture: {architecture!r}.")

    membership = str(identity.get("training_membership", ""))
    training_ids = tuple(map(str, getattr(provider, "training_case_ids", ())))
    validation_ids = tuple(map(str, getattr(provider, "validation_case_ids", ())))
    if membership in ("fixed25_train150_dev22", "subset_train150"):
        population = "fixed25_VAL22"
        expected_train, expected_validation = 150, 22
    elif membership in ("full_train600", "full_train600_formal"):
        population = "canonical_VAL89"
        expected_train, expected_validation = 600, 89
    else:
        raise ValueError(f"Unsupported R13-8 comparison training membership: {membership!r}.")
    if len(training_ids) != expected_train or len(validation_ids) != expected_validation:
        raise ValueError(
            f"Checkpoint membership {membership!r} requires TRAIN{expected_train}/VAL{expected_validation}; "
            f"provider exposes TRAIN{len(training_ids)}/VAL{len(validation_ids)}."
        )
    if len(set(training_ids)) != len(training_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Anchored native evaluator provider case IDs must be unique.")
    if arm in ("A", "L"):
        core = getattr(model, "core", None)
        if getattr(core, "schema", None) != "interaction_reused_e_anchored_member_routing_v1":
            raise ValueError("A/L checkpoint does not contain the maintained E-anchored core schema.")
    return {
        "arm": arm,
        "architecture": architecture,
        "workflow": workflow,
        "training_membership": membership,
        "validation_population": population,
        "training_case_count": len(training_ids),
        "validation_case_count": len(validation_ids),
        "validation_case_ids": list(validation_ids),
        "validation_scope": getattr(provider, "validation_scope", None),
    }


def evaluate(
    checkpoint: str | Path,
    device: str,
    output: str | Path,
    *,
    representative_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run paired native evaluation and bounded E-organization diagnostics.

    Native metrics and exports are delegated to the maintained provider and
    `_evaluate_native_split`. A/L-specific traces and interventions are
    collected only on the fixed representative panel; U remains an explicit
    global-entry comparison with no E-anchored counter claim.
    """
    import torch

    from .contextual_frontier_evaluation import _checkpoint_selection, load_contextual_checkpoint
    from .member_routing_evaluation import _evaluate_native_split, _jsonable, _storage_snapshot

    started = time.perf_counter()
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError("Keep earlier scientific evidence; select a fresh R13-8 evaluation output.")

    load_started = time.perf_counter()
    model, provider, payload = load_contextual_checkpoint(checkpoint_path, device)
    load_seconds = time.perf_counter() - load_started
    identity = _validate_checkpoint_population(model, provider, payload)
    validation_ids = tuple(identity["validation_case_ids"])
    representative_ids = _select_validation_panel(provider, representative_case_ids)
    fixed_train_candidates = ("0006", "0051", "0322", "0231")
    training_ids = tuple(map(str, provider.training_case_ids))
    train_detail_ids = tuple(case_id for case_id in fixed_train_candidates if case_id in training_ids)
    if len(train_detail_ids) != len(fixed_train_candidates):
        raise ValueError("R13-8 fixed TRAIN health/detail panel is not present in this saved training population.")

    recipe = getattr(provider, "recipe", None)
    if not isinstance(recipe, Mapping):
        raise TypeError("Maintained anchored provider must expose its saved recipe.")
    output_root = Path(recipe["storage"]["output_root"]).expanduser().resolve(strict=True)
    if output_path == output_root or not output_path.is_relative_to(output_root):
        raise ValueError(f"R13-8 evaluation output must be beneath saved DATA2 root {output_root}.")
    storage_options = {
        "storage_output_root": output_root,
        "output_budget_bytes": int(recipe["evaluation"]["max_output_bytes"]),
        "payload_budget_bytes": int(recipe["storage"]["max_output_bytes"]),
        "temporary_budget_bytes": int(recipe["storage"]["max_peak_temporary_bytes"]),
        "minimum_free_headroom_bytes": int(recipe["storage"]["minimum_free_headroom_bytes"]),
    }
    _storage_snapshot(output_root, **storage_options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = _storage_snapshot(output_path.parent, **storage_options)
    output_path.mkdir(parents=False)

    executions: dict[str, Any] = {}
    arrays_written: dict[str, Any] = {}
    array_write_seconds = 0.0
    try:
        for mode in ("full", "restricted"):
            mode_dir = output_path / mode
            mode_dir.mkdir()
            validation_dir = mode_dir / "validation"
            train_dir = mode_dir / "train_detail"
            validation_dir.mkdir()
            train_dir.mkdir()
            _synchronize(device)
            mode_started = time.perf_counter()
            with _mode_context(model, mode):
                validation_started = time.perf_counter()
                validation = _evaluate_native_split(
                    model, provider, payload, "validation", validation_ids,
                    export_ids=set(representative_ids),
                    # R13-7 U's actual applied E rows and global query pi are
                    # retained below by _collect_u_organization_diagnostics.
                    # The older generic member route exporter assumes B_M/B_E
                    # in the decoder trace, which this H-based U trace does not emit.
                    collect_routes=False,
                )
                _synchronize(device)
                validation_seconds = time.perf_counter() - validation_started
                train_started = time.perf_counter()
                train = _evaluate_native_split(
                    model, provider, payload, "train", train_detail_ids,
                    export_ids=set(train_detail_ids),
                )
                _synchronize(device)
                train_seconds = time.perf_counter() - train_started
            _synchronize(device)
            mode_seconds = time.perf_counter() - mode_started
            write_started = time.perf_counter()
            arrays_written[f"{mode}.validation"] = _write_representative_arrays(validation, validation_dir)
            arrays_written[f"{mode}.train_detail"] = _write_representative_arrays(train, train_dir)
            array_write_seconds += time.perf_counter() - write_started

            work = {"validation": validation["executed_work_counts"],
                    "train_detail": train["executed_work_counts"]}
            if identity["arm"] in ("A", "L"):
                validation["separated_anchored_work"] = require_complete_anchored_work_counts(
                    validation["executed_work_counts"]
                )
                train["separated_anchored_work"] = require_complete_anchored_work_counts(
                    train["executed_work_counts"]
                )
            else:
                validation["separated_anchored_work"] = {
                    "status": "not_applicable_legacy_global_entry",
                    "reason": "U uses the R13-7 global entrance path and does not emit R13-8 E-directory counters.",
                }
                train["separated_anchored_work"] = dict(validation["separated_anchored_work"])
            executions[mode] = {
                "validation": validation,
                "train_detail": train,
                "wall_seconds": mode_seconds,
                "split_wall_seconds": {
                    "validation": validation_seconds,
                    "train_detail": train_seconds,
                },
                "work_counts_by_split": work,
                "runtime_semantics": (
                    "synchronized complete native calls including provider preparation, decode and native extraction; "
                    "fixed case membership, one case per batch; exports timed separately"
                ),
            }

        anchored_diagnostics: dict[str, Any]
        diagnostics_started = time.perf_counter()
        if identity["arm"] in ("A", "L"):
            anchored_diagnostics = _collect_anchored_diagnostics(
                model, provider, payload, representative_ids, device,
            )
        else:
            anchored_diagnostics = _collect_u_organization_diagnostics(
                model, provider, payload, representative_ids, device,
            )
        _synchronize(device)
        diagnostics_seconds = time.perf_counter() - diagnostics_started

        result = {
            "schema": "honf_r13_8_e_anchored_native_evaluation_v1",
            "status": "completed",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch", payload.get("current_epoch", 0))),
            "checkpoint_selection": _checkpoint_selection(checkpoint_path, payload),
            "checkpoint_identity": identity,
            "arm": identity["arm"],
            "core_schema": getattr(getattr(model, "core", None), "schema", None),
            "locator_mode": getattr(getattr(model, "core", None), "locator_mode", None),
            "retained_mass": getattr(getattr(model, "core", None), "retained_mass", None),
            "validation_population": identity["validation_population"],
            "case_ids": {"validation": list(validation_ids), "train_detail": list(train_detail_ids)},
            "fixed_detail_panel": {
                "validation": list(representative_ids),
                "train": list(train_detail_ids),
                "selection": "input-only module-count strata and stable original case IDs; no predictions or errors",
                "comparison_note": "The matched TRAIN150/VAL22 recipes share this input partition; pass these IDs to pin the same panel on a broader population.",
            },
            "paired_read_penalties": {
                "validation": _paired_read_penalties(
                    executions["full"]["validation"], executions["restricted"]["validation"],
                ),
                "train_detail": _paired_read_penalties(
                    executions["full"]["train_detail"], executions["restricted"]["train_detail"],
                ),
            },
            "executions": executions,
            "organization_and_boundary_diagnostics": anchored_diagnostics,
            "timing": {
                "checkpoint_load_seconds": load_seconds,
                "native_full_and_restricted_seconds": sum(
                    float(executions[mode]["wall_seconds"]) for mode in ("full", "restricted")
                ),
                "diagnostic_seconds": diagnostics_seconds,
                "representative_array_write_seconds": array_write_seconds,
                "process_seconds_before_result_json": time.perf_counter() - started,
                "measurement": "wall-clock with CUDA synchronization at measured native and diagnostic boundaries",
            },
            "fit_epochs": 0,
            "optimizer_updates": 0,
            "solver_attempts": 0,
            "TEST_access": False,
            "inverse_tested": False,
            "storage": storage,
            "representative_array_sizes": arrays_written,
        }
        jsonable = _jsonable(result)
        json_bytes = len(json.dumps(jsonable, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        array_bytes = sum(int(item["compressed_bytes"]) for item in arrays_written.values())
        estimated_bytes = json_bytes + array_bytes + 4096
        output_budget = storage_options["output_budget_bytes"]
        if estimated_bytes > output_budget:
            raise OSError(f"R13-8 evaluation estimate {estimated_bytes} exceeds output budget {output_budget}.")
        if storage["free_before_bytes"] - estimated_bytes < storage_options["minimum_free_headroom_bytes"]:
            raise OSError("R13-8 evaluation output would violate configured DATA2 free-space headroom.")
        result["storage"].update({
            "estimated_output_bytes": estimated_bytes,
            "representative_uncompressed_bytes": sum(
                int(item["uncompressed_bytes"]) for item in arrays_written.values()
            ),
            "representative_compressed_bytes": array_bytes,
            "free_after_representatives_bytes": int(shutil.disk_usage(output_path).free),
        })
        for _ in range(3):
            result_path = output_path / "result.json"
            serialized = json.dumps(
                _jsonable(result), separators=(",", ":"), allow_nan=False,
            ) + "\n"
            with result_path.open("w", encoding="utf-8") as stream:
                stream.write(serialized)
            actual_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
            if result["storage"].get("actual_output_bytes") == actual_bytes:
                break
            result["storage"]["actual_output_bytes"] = actual_bytes
            result["storage"]["free_after_bytes"] = int(shutil.disk_usage(output_path).free)
        actual_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
        if actual_bytes > output_budget:
            raise OSError("R13-8 native evaluation exceeded its declared output budget.")
    except BaseException as exc:
        partial_files = sorted(
            (path for path in output_path.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(output_path).as_posix(),
        )
        failure_receipt = {
            "schema": "honf_r13_8_native_evaluation_failure_receipt_v1",
            "status": "failed_partial_output_preserved",
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "partial_files_before_receipt": [
                {
                    "relative_path": path.relative_to(output_path).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                }
                for path in partial_files
            ],
            "partial_bytes_before_receipt": sum(int(path.stat().st_size) for path in partial_files),
            "note": "Incomplete evaluation output is retained for diagnosis; it is not a completed result.",
        }
        (output_path / "failure.json").write_text(
            json.dumps(failure_receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        raise

    return {
        "status": "completed",
        "output": str(output_path),
        "checkpoint_epoch": result["checkpoint_epoch"],
        "arm": result["arm"],
        "validation_case_count": len(validation_ids),
        "fixed_validation_detail_ids": list(representative_ids),
        "actual_bytes": actual_bytes,
        "process_seconds": result["timing"]["process_seconds_before_result_json"],
    }


def _collect_anchored_diagnostics(
    model: Any,
    provider: Any,
    payload: Mapping[str, Any],
    representative_ids: Sequence[str],
    device: Any,
) -> dict[str, Any]:
    """Collect anchored traces/probes through the adapter-owned native hooks."""
    import torch

    from .member_routing_evaluation import _jsonable, _query_coordinates, _query_indices

    trace_fn = getattr(provider, "anchored_organization_trace", None)
    intervention_fn = getattr(provider, "anchored_intervention", None)
    boundary_fn = getattr(provider, "anchored_boundary_sweep", None)
    for name, fn in (("anchored_organization_trace", trace_fn),
                     ("anchored_intervention", intervention_fn),
                     ("anchored_boundary_sweep", boundary_fn)):
        if not callable(fn):
            raise TypeError(f"A/L provider must expose {name} for R13-8 native evaluation.")

    all_traces: dict[str, Any] = {"full": {}, "restricted": {}}
    interventions: dict[str, Any] = {}
    boundary_sweeps: dict[str, Any] = {"full": {}, "restricted": {}}
    diagnostic_work: dict[str, dict[str, float]] = {
        "organization_trace": {}, "self_only": {}, "reassign_member": {}, "boundary_sweep": {},
    }
    diagnostic_timing: dict[str, float] = {
        "native_prediction_replay": 0.0, "organization_trace": 0.0,
        "same_state_interventions": 0.0, "boundary_sweeps": 0.0,
    }
    observed: list[str] = []
    epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    for mode in ("full", "restricted"):
        with _mode_context(model, mode), torch.no_grad():
            batches = provider.native_evaluation_batches(
                split="validation", case_keys=representative_ids, microbatch_size=1,
            )
            seen = []
            for batch in batches:
                case_ids = tuple(map(str, batch.targets.case_ids))
                if len(case_ids) != 1 or tuple(map(str, batch.case_keys)) != case_ids:
                    raise ValueError("Anchored diagnostics require one complete fixed case per native batch.")
                case_id = case_ids[0]
                expected = str(representative_ids[len(seen)])
                if case_id != expected:
                    raise ValueError("Anchored diagnostic provider changed the fixed representative case order.")
                _synchronize(device)
                prediction_started = time.perf_counter()
                prediction, auxiliary = provider.predict_native(
                    model, batch.scene_inputs, batch.receivers, model.architecture, "validation", epoch=epoch,
                )
                _synchronize(device)
                diagnostic_timing["native_prediction_replay"] += time.perf_counter() - prediction_started
                query_indices = _query_indices(batch.receivers, prediction)
                _synchronize(device)
                trace_started = time.perf_counter()
                trace = trace_fn(model, prediction, query_indices, read_mode=mode)
                _synchronize(device)
                diagnostic_timing["organization_trace"] += time.perf_counter() - trace_started
                if not isinstance(trace, Mapping):
                    raise TypeError("anchored_organization_trace must return a mapping.")
                packed_trace = trace.get("trace")
                if not isinstance(packed_trace, Mapping):
                    raise TypeError("anchored_organization_trace must retain its packed core trace mapping.")
                _accumulate_numeric_counts(diagnostic_work["organization_trace"], trace.get("work_counts"))
                scene = getattr(getattr(prediction, "native_prepared", None), "scene_batch", None)
                if scene is None:
                    raise TypeError("Anchored native prediction has no prepared scene for fixed anchor selection.")
                environment_ids = _numpy(scene.environment_ids)[0]
                environment_coordinates = _numpy(scene.environment_coordinates)[0]
                environment_mask = _numpy(scene.environment_mask)[0]
                anchors = select_deterministic_anchor_panel(
                    environment_ids, environment_coordinates, environment_mask, count=4,
                )
                id_to_index = {int(value): i for i, value in enumerate(environment_ids.tolist())}
                anchor_rows = []
                for original_id in anchors:
                    row = id_to_index[int(original_id)]
                    anchor_rows.append({
                        "environment_original_id": int(original_id),
                        "environment_slot": int(row),
                        "coordinates": np.asarray(environment_coordinates[row], dtype=np.float64).tolist(),
                    })
                packed_shape_keys = (
                    "anchor_indices", "anchor_weights", "anchor_read_indices", "anchor_read_E_slots",
                    "anchor_read_member_read_index", "anchor_read_member_index", "anchor_read_original_id",
                    "anchor_read_is_module", "anchor_read_access_prior", "anchor_read_probabilities",
                    "anchor_read_calibrated_scores", "anchor_read_strength", "e_access_full_offsets",
                    "e_access_restricted_offsets", "e_access_full_member_index",
                    "e_access_restricted_member_index", "member_score_union_indices_M",
                    "member_score_union_indices_E",
                )
                trace_row = {
                    "query_indices": list(query_indices),
                    "query_coordinates": _query_coordinates(prediction, query_indices),
                    "deterministic_anchor_panel": anchor_rows,
                    "scene_geometry": _scene_geometry_payload(scene),
                    "packed_trace_tensor_shapes": {
                        key: list(_numpy(packed_trace[key]).shape)
                        for key in packed_shape_keys if key in packed_trace
                    },
                    "trace": trace,
                }
                all_traces[mode][case_id] = trace_row

                if mode == "full":
                    baseline_metrics = _native_metric_summary(
                        provider, batch, prediction, auxiliary,
                    )
                    intervention_specs = (
                        {"kind": "self_only"},
                        _reassignment_intervention(scene, trace),
                    )
                    for intervention in intervention_specs:
                        actual_kind = str(intervention["kind"])
                        probe_name = str(intervention.get("requested_kind", actual_kind))
                        fallback_reason = intervention.get("fallback_reason")
                        provider_intervention = {
                            key: value for key, value in intervention.items()
                            if key not in ("requested_kind", "fallback_reason")
                        }
                        _synchronize(device)
                        intervention_started = time.perf_counter()
                        measured = intervention_fn(
                            model, batch, prediction, query_indices,
                            directory_intervention=provider_intervention,
                            read_mode=mode,
                        )
                        _synchronize(device)
                        diagnostic_timing["same_state_interventions"] += time.perf_counter() - intervention_started
                        if not isinstance(measured, Mapping):
                            raise TypeError(f"anchored_intervention({probe_name}) must return a mapping.")
                        if measured.get("same_prepared_context") is not True:
                            raise ValueError("Access-only intervention did not preserve the prepared contextual state.")
                        selection = measured.get("directory_intervention")
                        if not isinstance(selection, Mapping) or selection.get("eligible") is not True:
                            raise ValueError(
                                f"anchored_intervention({probe_name}) did not execute an eligible directory change."
                            )
                        intervention_prediction = measured.get("prediction")
                        if intervention_prediction is None:
                            raise ValueError(f"anchored_intervention({probe_name}) returned no native prediction.")
                        _accumulate_numeric_counts(diagnostic_work[probe_name], measured.get("work_counts"))
                        intervention_metrics = _native_metric_summary(
                            provider, batch, intervention_prediction, measured.get("auxiliary"),
                        )
                        expected = provider_intervention
                        if actual_kind == "self_only":
                            if selection.get("kind") != "self_only":
                                raise ValueError("Provider changed the requested self-only directory control.")
                        elif actual_kind == "reassign_member":
                            selected_ids = selection.get("selected_ids")
                            if not isinstance(selected_ids, Mapping):
                                raise ValueError("Reassignment result must return its exact selected source and target IDs.")
                            for identity_name in ("source_role", "source_id", "target_role", "target_id"):
                                if selected_ids.get(identity_name) != expected[identity_name]:
                                    raise ValueError(
                                        f"Provider changed the requested reassignment {identity_name}."
                                    )
                            if selected_ids.get("anchor_id") != expected["anchor_id"]:
                                raise ValueError("Provider changed the selected E-directory anchor ID.")
                        elif actual_kind == "remove_member":
                            selected_ids = selection.get("selected_ids")
                            if not isinstance(selected_ids, Mapping):
                                raise ValueError("Removal fallback must return its exact selected source ID.")
                            for identity_name in ("member_role", "member_id"):
                                if selected_ids.get(identity_name) != expected[identity_name]:
                                    raise ValueError(f"Provider changed the removal fallback {identity_name}.")
                            if selected_ids.get("anchor_id") != expected["anchor_id"]:
                                raise ValueError("Provider changed the removal fallback E-directory anchor ID.")
                        baseline_trace = measured.get("baseline_trace")
                        intervention_trace = measured.get("intervention_trace", measured.get("trace"))
                        if not isinstance(baseline_trace, Mapping) or not isinstance(intervention_trace, Mapping):
                            raise ValueError(f"anchored_intervention({probe_name}) must retain both actual read traces.")
                        interventions.setdefault(case_id, {})[probe_name] = {
                            "same_prepared_context": True,
                            "requested_control": probe_name,
                            "actual_control": actual_kind,
                            "fallback_reason": fallback_reason,
                            "submitted_directory_intervention": provider_intervention,
                            "directory_intervention": selection,
                            "selected_query_indices": list(query_indices),
                            "baseline_native_metrics": baseline_metrics,
                            "native_metrics": intervention_metrics,
                            "native_metric_deltas": _native_metric_deltas(
                                baseline_metrics, intervention_metrics,
                            ),
                            "baseline_work_counts": measured.get(
                                "baseline_work_counts",
                                measured.get("work_counts", {}).get("baseline")
                                if isinstance(measured.get("work_counts"), Mapping) else None,
                            ),
                            "intervention_work_counts": measured.get(
                                "intervention_work_counts",
                                measured.get("work_counts", {}).get("intervention")
                                if isinstance(measured.get("work_counts"), Mapping) else None,
                            ),
                            "baseline_trace": baseline_trace,
                            "intervention_trace": intervention_trace,
                            "reassignment": measured.get("reassignment"),
                        }

                case_sweeps = []
                for probe in _thermal_boundary_coordinates(scene, model.core.locator_mode):
                    _synchronize(device)
                    boundary_started = time.perf_counter()
                    sweep = _invoke_anchored_boundary_sweep(
                        boundary_fn, model, batch, query_indices[0], probe, mode,
                    )
                    _synchronize(device)
                    boundary_seconds = time.perf_counter() - boundary_started
                    diagnostic_timing["boundary_sweeps"] += boundary_seconds
                    if not isinstance(sweep, Mapping):
                        raise TypeError("anchored_boundary_sweep must return a mapping.")
                    enriched = dict(sweep)
                    expected_offsets = _numpy(probe["offsets_in_spacing"]).astype(np.float64, copy=False)
                    returned_offsets = _numpy(
                        enriched.get("offsets_in_spacing", expected_offsets)
                    ).astype(np.float64, copy=False)
                    if not np.array_equal(returned_offsets, expected_offsets):
                        raise ValueError("Boundary hook changed the declared symmetric E-spacing offsets.")
                    returned_coordinate_value = enriched.get(
                        "probe_coordinates", probe["probe_coordinates"]
                    )
                    expected_coordinates = _numpy(probe["probe_coordinates"]).astype(np.float64, copy=False)
                    if not _live_probe_coordinates_match(returned_coordinate_value, expected_coordinates):
                        raise ValueError("Boundary hook did not evaluate the declared live probe coordinates.")
                    enriched.update({
                        "offsets_in_spacing": probe["offsets_in_spacing"],
                        "boundary_name": probe["boundary_name"],
                        "boundary_axis": probe["boundary_axis"],
                        "boundary_kind": probe["boundary_kind"],
                        "probe_coordinates": probe["probe_coordinates"],
                        "center_point_coordinates": probe["center_point_coordinates"],
                        "environment_original_ids": environment_ids.tolist(),
                    })
                    summary = _summarize_provider_boundary(enriched)
                    summary["hook_wall_seconds"] = boundary_seconds
                    case_sweeps.append(summary)
                    _accumulate_numeric_counts(
                        diagnostic_work["boundary_sweep"],
                        enriched.get("work_counts_by_probe"),
                    )
                boundary_sweeps[mode][case_id] = case_sweeps
                seen.append(case_id)
            if tuple(seen) != tuple(map(str, representative_ids)):
                raise ValueError("Anchored diagnostics did not cover the fixed representative panel exactly once.")
        observed.extend(seen)
    return {
        "status": "completed",
        "locator_mode": getattr(getattr(model, "core", None), "locator_mode", None),
        "organization_traces": all_traces,
        "same_state_interventions": interventions,
        "boundary_sweeps": boundary_sweeps,
        "diagnostic_work_counts_by_kind": {
            kind: {name: int(value) if float(value).is_integer() else value
                   for name, value in sorted(counts.items())}
            for kind, counts in diagnostic_work.items()
        },
        "diagnostic_timing_seconds_by_kind": diagnostic_timing,
        "cases_evaluated_per_mode": list(map(str, representative_ids)),
        "case_evaluations_across_modes": len(observed),
        "semantics": {
            "directory_trace": "actual applied incoming EM/EE plus explicit structural self and compiled full/joint95 E rows",
            "access_interventions": "same-prepared-context read-time changes; no contextual state recomputation",
            "boundary": "decoder and native extraction are separately summarized; provider must rebuild live locator for every probe",
        },
    }


def _collect_u_organization_diagnostics(
    model: Any,
    provider: Any,
    payload: Mapping[str, Any],
    representative_ids: Sequence[str],
    device: Any,
) -> dict[str, Any]:
    """Retain comparable input-selected E-row views and U's actual global pi reads."""
    import torch

    from .member_routing_evaluation import _jsonable, _query_coordinates, _query_indices
    from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1

    epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    row_views: dict[str, Any] = {}
    query_reads: dict[str, dict[str, Any]] = {"full": {}, "restricted": {}}
    work_counts: dict[str, dict[str, float]] = {"full_query_trace": {}, "restricted_query_trace": {}}
    timings = {"native_prediction_replay": 0.0, "full_global_query_trace": 0.0,
               "restricted_global_query_trace": 0.0}

    for expected_case_id in map(str, representative_ids):
        with torch.no_grad():
            batches = provider.native_evaluation_batches(
                split="validation", case_keys=(expected_case_id,), microbatch_size=1,
            )
            batch = next(iter(batches), None)
            if batch is None:
                raise ValueError(f"U organization diagnostics could not rebuild case {expected_case_id}.")
            case_ids = tuple(map(str, batch.targets.case_ids))
            if case_ids != (expected_case_id,) or tuple(map(str, batch.case_keys)) != case_ids:
                raise ValueError("U organization diagnostics changed the fixed case ID or order.")
            _synchronize(device)
            prediction_started = time.perf_counter()
            with _mode_context(model, "full"):
                prediction, auxiliary = provider.predict_native(
                    model, batch.scene_inputs, batch.receivers, model.architecture,
                    "validation", epoch=epoch,
                )
            _synchronize(device)
            timings["native_prediction_replay"] += time.perf_counter() - prediction_started

            prepared = getattr(prediction, "native_prepared", None)
            if prepared is None:
                raise TypeError("U native prediction did not retain its prepared contextual state.")
            organization = prepared.organization
            scene = prepared.scene_batch
            required = ("H_EM_applied", "H_EE_applied", "H", "retrieval_R")
            missing = [name for name in required if getattr(organization, name, None) is None]
            if missing:
                raise ValueError(f"U prepared organization is missing actual E-row state: {missing}.")

            module_ids = _numpy(scene.module_ids)[0]
            environment_ids = _numpy(scene.environment_ids)[0]
            module_mask = _numpy(scene.module_mask)[0].astype(bool)
            environment_mask = _numpy(scene.environment_mask)[0].astype(bool)
            environment_coordinates = _numpy(scene.environment_coordinates)[0]
            e_panel = select_deterministic_anchor_panel(
                environment_ids, environment_coordinates, environment_mask, count=4,
            )
            environment_slot_by_id = {
                (value.item() if isinstance(value, np.generic) else value): index
                for index, value in enumerate(environment_ids)
            }
            module_count = int(scene.module_ids.shape[1])
            member_ids = [value.item() if isinstance(value, np.generic) else value for value in module_ids]
            member_ids.extend(
                value.item() if isinstance(value, np.generic) else value for value in environment_ids
            )
            member_roles = ["M"] * len(module_ids) + ["E"] * len(environment_ids)
            member_valid = np.concatenate((module_mask, environment_mask))
            applied_em = _numpy(organization.H_EM_applied)
            applied_ee = _numpy(organization.H_EE_applied)
            h_global = _numpy(organization.H)
            retrieval = _numpy(organization.retrieval_R)
            h_self = getattr(organization, "H_self", None)
            h_self = None if h_self is None else _numpy(h_self)
            e_count = len(environment_ids)
            if (applied_em.shape[:2] != (1, e_count) or applied_em.shape[-1] != len(module_ids)
                    or applied_ee.shape != (1, e_count, e_count)
                    or h_global.ndim != 3 or retrieval.ndim != 3
                    or h_global.shape[0] != 1 or retrieval.shape[0] != 1
                    or h_global.shape[-2:] != (len(member_ids), len(member_ids))
                    or retrieval.shape[-2:] != (len(member_ids), len(member_ids))):
                raise ValueError("U prepared E rows lost their actual M/E address axes.")
            e_rows = []
            for environment_id in e_panel:
                e_id = environment_id.item() if isinstance(environment_id, np.generic) else environment_id
                e_slot = environment_slot_by_id[e_id]
                global_row = module_count + e_slot
                h_self_row = None
                if h_self is not None and h_self.ndim == 3 and h_self.shape[-2:] == h_global.shape[-2:]:
                    h_self_row = h_self[0, global_row].tolist()
                e_rows.append({
                    "environment_original_id": int(e_id),
                    "environment_slot": int(e_slot),
                    "coordinates": np.asarray(environment_coordinates[e_slot], dtype=np.float64).tolist(),
                    "view_semantics": (
                        "input-selected E row view, not an anchor choice; no structural-self member is added"
                    ),
                    "applied_H_EM_row": applied_em[0, e_slot].tolist(),
                    "applied_H_EE_row": applied_ee[0, e_slot].tolist(),
                    "H_global_row_at_M_plus_E_slot": h_global[0, global_row].tolist(),
                    "retrieval_R_global_address_row_at_M_plus_E_slot": retrieval[0, global_row].tolist(),
                    "H_self_global_row_if_exposed": h_self_row,
                    "member_address_original_ids": member_ids,
                    "member_address_roles": member_roles,
                    "member_address_present": member_valid.tolist(),
                })
            row_views[expected_case_id] = {
                "deterministic_E_row_panel": [int(value) for value in e_panel],
                "row_selection": "input-only stable E original IDs and coordinates; shared panel rule with A/L",
                "scene_geometry": _scene_geometry_payload(scene),
                "rows": e_rows,
                "source_prepared_work_counts": dict(getattr(prediction, "work", {}) or {}),
            }

            query_indices = tuple(_query_indices(batch.receivers, prediction))
            query = select_query_rows_v1(prepared.query_batch, query_indices)
            for mode in ("full", "restricted"):
                _synchronize(device)
                trace_started = time.perf_counter()
                output = model.core.decode(
                    prepared.organization, query,
                    intervention={"read_mode": mode}, return_trace=True,
                )
                _synchronize(device)
                timings[f"{mode}_global_query_trace"] += time.perf_counter() - trace_started
                trace = output.trace
                if not isinstance(trace, Mapping):
                    raise ValueError(f"U {mode} query decoder omitted its actual global read trace.")
                pi = trace.get("proposal_weights_joint", trace.get("pi_joint"))
                if pi is None:
                    raise ValueError("U query trace omitted the actual global address proposal pi.")
                pi_array = _numpy(pi)
                if pi_array.ndim != 3 or pi_array.shape[:2] != (1, len(query_indices)):
                    raise ValueError("U global proposal trace does not align with the selected query panel.")
                keep_keys = (
                    "proposal_logits_joint", "proposal_scores_joint", "proposal_weights_joint", "pi_joint",
                    "proposal_null", "address_original_ids", "address_role_codes", "address_eligible_mask",
                    "candidate_mask_M", "candidate_mask_E", "member_weights_M", "member_weights_E",
                    "member_logits_M", "member_logits_E", "prior_P_M", "prior_P_E",
                    "member_read_strength_M", "member_read_strength_E",
                )
                query_reads[mode][expected_case_id] = {
                    "query_indices": list(query_indices),
                    "query_coordinates": _query_coordinates(prediction, query_indices),
                    "actual_global_proposal_pi": _jsonable(pi),
                    "trace": {key: _jsonable(trace[key]) for key in keep_keys if key in trace},
                    "work_counts": {str(name): int(value) for name, value in output.work_counts.items()},
                    "organization_rows_shared_across_read_modes": True,
                }
                _accumulate_numeric_counts(work_counts[f"{mode}_query_trace"], output.work_counts)

    return {
        "status": "completed_global_entry_E_row_views",
        "locator_mode": None,
        "deterministic_E_row_views": row_views,
        "global_query_distribution_traces": query_reads,
        "anchored_work_counters": {
            "status": "not_applicable_global_entry",
            "reason": "U has no compiled E directory, live E anchor carrier, or anchor-local decoder work.",
        },
        "anchored_boundary_sweeps": {
            "status": "not_applicable_global_entry",
            "reason": "U does not select a live E anchor or E cover at each query coordinate.",
        },
        "diagnostic_work_counts_by_kind": work_counts,
        "diagnostic_timing_seconds_by_kind": timings,
        "cases_evaluated": list(map(str, representative_ids)),
        "semantics": {
            "E_row_view": (
                "input-selected original E row of actual applied H_EM/H_EE, global H and retrieval_R; "
                "the panel does not make the row an address anchor"
            ),
            "query_read": "actual U global address proposal pi and decoder support for the matched fixed query panel",
            "structural_self": "no additional structural-self member is injected by the E-row panel",
        },
    }


def _summarize_provider_boundary(boundary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize one provider-produced live-locator boundary sweep."""
    if boundary.get("locator_rebuilt_for_each_probe") is not True:
        raise ValueError("Boundary sweep must rebuild the actual locator from every probe coordinate.")
    read_mode = boundary.get("read_mode")
    if read_mode not in ("full", "restricted"):
        raise ValueError("Boundary hook must report the actual full or restricted read mode.")
    offsets = boundary.get("offsets_in_spacing")
    decoder = boundary.get("decoder")
    native = boundary.get("native_extraction")
    if not isinstance(decoder, Mapping) or not isinstance(native, Mapping):
        raise TypeError("Boundary hook must return separate decoder and native_extraction measurements.")
    if "output_scale" not in decoder or "output_scale" not in native:
        raise ValueError("Boundary outputs must declare decoder and native-extraction output scales.")
    offset_values = _numpy(offsets).astype(np.float64, copy=False)
    work_counts_by_probe = boundary.get("work_counts_by_probe")
    if (not isinstance(work_counts_by_probe, Sequence)
            or isinstance(work_counts_by_probe, (str, bytes, bytearray))
            or offset_values.ndim != 1
            or len(work_counts_by_probe) != offset_values.size
            or any(not isinstance(row, Mapping) for row in work_counts_by_probe)):
        raise ValueError("Boundary hook must retain actual work counts for every live locator probe.")
    shared_fixed_work_counts = boundary.get("shared_fixed_work_counts")
    if not isinstance(shared_fixed_work_counts, Mapping):
        raise ValueError("Boundary hook must retain the once-per-sweep fixed preparation/native work counts.")
    separated_shared_work = require_complete_anchored_work_counts(shared_fixed_work_counts)
    interface_control = boundary.get("physical_interface_control")
    if not isinstance(interface_control, Mapping):
        raise ValueError("Boundary hook must identify the fixed-control coverage-face probe semantics.")
    if interface_control.get("kind") != "artificial_coverage_face":
        raise ValueError("R13-8 boundary probes must be labelled as artificial E-coverage faces.")
    anchor_indices = boundary.get("anchor_indices")
    anchor_weights = boundary.get("anchor_weights")
    environment_ids = boundary.get("environment_original_ids")
    if anchor_indices is None or anchor_weights is None or environment_ids is None:
        raise ValueError("Boundary hook must retain live anchor slots, weights, and original environment IDs.")
    slots = _numpy(anchor_indices)
    weights = _numpy(anchor_weights).astype(np.float64, copy=False)
    ids = _numpy(environment_ids).reshape(-1)
    if slots.ndim != 2 or weights.shape != slots.shape or slots.shape[0] != offset_values.size:
        raise ValueError("Boundary live anchor carriers must preserve one [K] row per probe.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Boundary live anchor weights must be finite and nonnegative.")
    anchor_rows = []
    for probe_index, (slot_row, weight_row) in enumerate(zip(slots, weights, strict=True)):
        active = weight_row > 0
        if not np.isclose(weight_row.sum(), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("Every live boundary probe must have normalized anchor weights.")
        active_slots = slot_row[active].astype(np.int64, copy=False)
        if ((active_slots < 0).any() or (active_slots >= ids.size).any()
                or len(np.unique(active_slots)) != active_slots.size):
            raise ValueError("Boundary probe assigned an invalid or duplicate positive-weight E anchor.")
        if np.any(slot_row[~active] != -1):
            raise ValueError("Zero-weight boundary anchor slots must be padded with -1.")
        anchor_rows.append({
            "probe_index": probe_index,
            "offset_in_spacing": float(offset_values[probe_index]),
            "environment_slots": active_slots.astype(np.int64).tolist(),
            "environment_original_ids": [
                ids[int(slot)].item() if isinstance(ids[int(slot)], np.generic) else ids[int(slot)]
                for slot in active_slots
            ],
            "weights": weight_row[active].tolist(),
        })
    anchor_usage: dict[str, dict[str, Any]] = {}
    for row in anchor_rows:
        original_ids = row["environment_original_ids"]
        if len(set(map(str, original_ids))) != len(original_ids):
            raise ValueError("A live boundary query contains duplicate positive-weight E IDs.")
        for original_id in original_ids:
            key = str(original_id)
            entry = anchor_usage.setdefault(key, {
                "environment_original_id": original_id,
                "probe_indices": [],
            })
            entry["probe_indices"].append(int(row["probe_index"]))
    native_validity = native.get("interpolation_valid_mask")
    if native_validity is None:
        raise ValueError("Native boundary extraction must retain the fixed-grid interpolation validity mask.")
    valid_mask = _numpy(native_validity).astype(bool, copy=False).reshape(-1)
    native_values = _numpy(native["values"])
    if valid_mask.shape != offset_values.shape or native_values.ndim < 2 \
            or native_values.shape[0] != offset_values.size:
        raise ValueError("Native interpolation validity and sampled values must align with every probe.")
    if valid_mask.all():
        native_summary = summarize_boundary_sweep(
            offsets, native_values, output_scale=native.get("output_scale", 1.0),
            evaluation_level="native_extraction",
        )
        native_summary.update({
            "regularity_assessment": "assessed_all_probes_within_native_cell_center_hull",
            "native_grid_nx_ny": native.get("native_grid_nx_ny"),
            "native_grid_cell_count": native.get("native_grid_cell_count"),
            "interpolation_valid_mask": valid_mask.tolist(),
            "sampling_method": native.get("sampling_method"),
            "off_grid_curl_evaluated": native.get("off_grid_curl_evaluated"),
        })
    else:
        scale = _numpy(native.get("output_scale", 1.0))
        native_summary = {
            "schema": "honf_r13_8_native_boundary_partial_hull_v1",
            "evaluation_level": "native_extraction",
            "regularity_assessment": "not_assessed_partial_native_center_hull",
            "offset_unit": "local E spacing",
            "probe_offsets_in_spacing": offset_values.tolist(),
            "output_shape": list(native_values.shape[1:]),
            "output_scale": scale.tolist(),
            "native_grid_nx_ny": native.get("native_grid_nx_ny"),
            "native_grid_cell_count": native.get("native_grid_cell_count"),
            "locator_rebuild_required": True,
            "interpolation_valid_mask": valid_mask.tolist(),
            "valid_probe_indices": np.flatnonzero(valid_mask).astype(int).tolist(),
            "outside_native_cell_center_hull_probe_indices": np.flatnonzero(~valid_mask).astype(int).tolist(),
            "sampled_values": native_values.tolist(),
            "sampling_method": native.get("sampling_method"),
            "off_grid_curl_evaluated": native.get("off_grid_curl_evaluated"),
            "reason": (
                "At least one probe lies outside the native cell-centre interpolation hull; sampled values and "
                "validity are retained, but native continuity and derivative fits are not assessed."
            ),
        }
    locator_mode = str(boundary.get("locator_mode", ""))
    if locator_mode not in ("e_anchor", "local_e_cover"):
        raise ValueError("Boundary hook must report its active R13-8 locator mode.")
    cover_regularities = (
        "piecewise-constant E ownership with an expected cover jump at a hard face"
        if locator_mode == "e_anchor" else
        "smoothstep cover is expected to be C1 at center knots, former cell-ownership faces, and outer-strip junctions"
    )
    result = {
        "locator_rebuilt_for_each_probe": True,
        "read_mode": read_mode,
        "locator_mode": locator_mode,
        "boundary_name": boundary.get("boundary_name"),
        "probe_coordinates": boundary.get("probe_coordinates"),
        "center_point_coordinates": boundary.get("center_point_coordinates"),
        "boundary_axis": boundary.get("boundary_axis"),
        "boundary_kind": boundary.get("boundary_kind"),
        "live_anchor_carrier_by_probe": anchor_rows,
        "active_anchor_reuse_across_probes": {
            "unique_original_ids": len(anchor_usage),
            "duplicate_positive_ids_within_probe": 0,
            "reused_original_ids": [
                {"environment_original_id": row["environment_original_id"],
                 "probe_indices": row["probe_indices"],
                 "probe_count": len(row["probe_indices"])}
                for row in anchor_usage.values() if len(row["probe_indices"]) > 1
            ],
        },
        "decoder": summarize_boundary_sweep(
            offsets, decoder["values"], output_scale=decoder.get("output_scale", 1.0),
            evaluation_level="decoder",
        ),
        "native_extraction": native_summary,
        "physical_interface_control": interface_control,
        "physical_interface_claim": (
            "artificial E-coverage boundary with controls held fixed; this is not a material-interface probe"
        ),
        "locator_cover_regularity_expectation": cover_regularities,
        "decoder_reader_regularity": (
            "measured from decoder/native one-sided limits and slopes; C1 locator coverage alone does not imply C1 reads"
        ),
        "member_score_geometry_law": (
            "learned member scores receive normalized query-member displacement, its Euclidean vector norm, "
            "and asinh displacement; a norm feature may have a center cusp when its learned sensitivity is nonzero"
        ),
        "raw_score_trace_by_probe": boundary.get(
            "score_trace_by_probe", boundary.get("raw_score_trace_by_probe")
        ),
        "work_counts_by_probe": [dict(row) for row in work_counts_by_probe],
        "separated_work_counts_by_probe": [
            summarize_anchored_work_counts(row) for row in work_counts_by_probe
        ],
        "shared_fixed_work_counts": dict(shared_fixed_work_counts),
        "separated_shared_fixed_work_counts": separated_shared_work,
    }
    return result


__all__ = [
    "evaluate",
    "require_complete_anchored_work_counts",
    "select_deterministic_anchor_panel",
    "summarize_anchored_work_counts",
    "summarize_boundary_sweep",
]
