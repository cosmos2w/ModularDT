"""Small, strict reducers and identity checks for multiview native evidence.

This module does not load checkpoints or datasets. Callers provide arrays from
an already authorized evaluator and bind them to the complete evaluation
identity before reusing cached numerical results.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CACHE_BINDING_FIELDS = (
    "dataset_id",
    "partition",
    "sample_ids",
    "checkpoint_sha256",
    "normalizer_sha256",
    "query_sha256",
    "mask_sha256",
    "target_sha256",
    "output_schema",
    "units",
)

THERMAL_NATIVE_FIELDS = ("u", "v", "p", "omega", "temperature")
THERMAL_NATIVE_UNIT_SCOPE = "benchmark_native_source_pack_units; no SI conversion asserted"
THERMAL_SAMPLED_AUXILIARIES = (
    "surface_temperature",
    "material_temperature",
    "q_normal_proxy",
    "sampled_module_peak_temperature",
)
WIND_NATIVE_COMPONENTS = ("Ux", "Uy", "Uz")
WIND_NATIVE_UNIT = "m/s"


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible semantic object with a stable encoding."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_binding_mismatches(
    cached: Mapping[str, Any], requested: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return every missing or unequal identity field; an incomplete binding never matches."""

    mismatches: dict[str, tuple[Any, Any]] = {}
    for key in CACHE_BINDING_FIELDS:
        old = cached.get(key)
        new = requested.get(key)
        if old is None or new is None or old != new:
            mismatches[key] = (old, new)
    return mismatches


def cache_binding_matches(cached: Mapping[str, Any], requested: Mapping[str, Any]) -> bool:
    """True only when all case, model, normalizer, query, mask and target bindings match."""

    return not cache_binding_mismatches(cached, requested)


def apply_native_validation_policy(model: Any, policy: str) -> str:
    """Apply only the explicitly pinned fresh-model validation policy.

    Classic references retain their historical default path. The consolidated
    path is enabled only for a recipe that names the qualified policy.
    """

    if policy == "classic_native_default":
        return "classic_native_default_unchanged"
    if policy != "consolidated_route_flags_v1":
        raise ValueError(f"Unsupported native validation policy: {policy!r}")
    core = getattr(model, "core", None)
    setter = getattr(core, "set_consolidated_native_validation", None)
    if not callable(setter):
        raise TypeError("The pinned consolidated policy requires model.core.set_consolidated_native_validation.")
    setter(True)
    return "consolidated_route_flags_v1_applied"


def _selected_values(values: Any, mask: Any) -> np.ndarray:
    values_array = np.asarray(values)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape == values_array.shape or values_array.ndim > mask_array.ndim and values_array.shape[: mask_array.ndim] == mask_array.shape:
        selected = values_array[mask_array]
    elif values_array.ndim == mask_array.ndim + 1 and values_array.shape[1:] == mask_array.shape:
        selected = values_array[:, mask_array]
    else:
        raise ValueError(f"Mask shape {mask_array.shape} cannot select values shaped {values_array.shape}.")
    if selected.size == 0:
        raise ValueError("A physical metric cannot use an empty fixed mask.")
    selected = np.asarray(selected, dtype=np.float64)
    if not np.all(np.isfinite(selected)):
        raise ValueError("Selected physical values contain NaN or infinity.")
    return selected


def masked_field_metrics(prediction: Any, reference: Any, mask: Any) -> dict[str, Any]:
    """Compute unclipped physical RMSE, MAE, max error and relative L2 on one fixed mask."""

    predicted = np.asarray(prediction)
    expected = np.asarray(reference)
    if predicted.shape != expected.shape:
        raise ValueError(f"Prediction/reference shapes differ: {predicted.shape} != {expected.shape}.")
    selected_prediction = _selected_values(predicted, mask).reshape(-1)
    selected_reference = _selected_values(expected, mask).reshape(-1)
    if selected_prediction.shape != selected_reference.shape:
        raise ValueError("The supplied mask selected different numbers of prediction and reference values.")
    residual = selected_prediction - selected_reference
    reference_norm = float(np.linalg.norm(selected_reference))
    return {
        "count": int(residual.size),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "signed_bias": float(np.mean(residual)),
        "reference_l2_norm": reference_norm,
        "reference_rms": float(reference_norm / math.sqrt(selected_reference.size)),
        "relative_l2": float(np.linalg.norm(residual) / reference_norm) if reference_norm > 0.0 else None,
        "zero_reference_norm": reference_norm == 0.0,
    }


def equal_case_summary(case_values: Mapping[str, float]) -> dict[str, Any]:
    """Summarize one scalar per case so large masks do not silently upweight cases."""

    if not case_values:
        raise ValueError("At least one per-case value is required.")
    ordered = sorted((str(key), float(value)) for key, value in case_values.items())
    values = np.asarray([value for _, value in ordered], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Per-case values must be finite.")
    worst_value = float(np.max(values))
    return {
        "case_count": len(ordered),
        "equal_case_mean": float(np.mean(values)),
        "case_p50": float(np.quantile(values, 0.50)),
        "case_p90": float(np.quantile(values, 0.90)),
        "worst": worst_value,
        "worst_case_ids": [key for key, value in ordered if value == worst_value],
        "per_case": {key: value for key, value in ordered},
    }


def stratified_case_summary(
    case_values: Mapping[str, float],
    stratum_by_case: Mapping[str, str | int],
    *,
    required_strata: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Summarize equal-case means and tails overall and in fixed input strata."""

    if set(case_values) != set(stratum_by_case):
        raise ValueError("Every scored case must have exactly one fixed input stratum.")
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for case, value in case_values.items():
        grouped[str(stratum_by_case[case])][str(case)] = float(value)
    expected = {str(value) for value in required_strata} if required_strata is not None else set(grouped)
    if set(grouped) != expected:
        raise ValueError(f"Observed strata {sorted(grouped)} differ from required strata {sorted(expected)}.")
    return {
        "overall": equal_case_summary(case_values),
        "strata": {stratum: equal_case_summary(grouped[stratum]) for stratum in sorted(grouped)},
    }


def equal_cluster_summary(row_values: Mapping[str, float], cluster_by_row: Mapping[str, str]) -> dict[str, Any]:
    """Average rows inside each cluster first, then give every cluster equal weight."""

    if set(row_values) != set(cluster_by_row):
        raise ValueError("Every row value must have exactly one explicit cluster binding.")
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row, value in row_values.items():
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("Row values must be finite.")
        by_cluster[str(cluster_by_row[row])].append(scalar)
    cluster_values = {cluster: float(np.mean(values)) for cluster, values in sorted(by_cluster.items())}
    summary = equal_case_summary(cluster_values)
    return {
        "cluster_count": len(cluster_values),
        "row_count": len(row_values),
        "equal_cluster_mean": summary["equal_case_mean"],
        "cluster_p50": summary["case_p50"],
        "cluster_p90": summary["case_p90"],
        "worst_cluster_mean": summary["worst"],
        "worst_cluster_ids": summary["worst_case_ids"],
        "per_cluster_row_mean": cluster_values,
        "per_cluster_row_count": {cluster: len(values) for cluster, values in sorted(by_cluster.items())},
    }


def wind_role_metrics(
    prediction: Any,
    reference: Any,
    role_ids: Sequence[str],
    valid_mask: Any,
    *,
    expected_total_queries: int = 8192,
    required_roles: Sequence[str] = (
        "volume", "hub_slab", "downstream_envelope", "near_turbine", "background",
    ),
) -> dict[str, Any]:
    """Compute native Wind component and vector errors with Q shared across roles.

    Vector RMSE is ``sqrt(mean_q(sum_component(error**2)))`` in m/s. It is
    intentionally not divided by the three vector components.
    """

    predicted = np.asarray(prediction)
    expected = np.asarray(reference)
    roles = np.asarray(role_ids, dtype=str)
    mask = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != expected.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("Wind predictions and references must have matching [query, 3] shapes.")
    if roles.shape != (predicted.shape[0],) or mask.shape != roles.shape:
        raise ValueError("Wind role IDs and fixed mask must have one value per query.")
    if predicted.shape[0] != expected_total_queries:
        raise ValueError(f"Wind Q must total {expected_total_queries} across roles; got {predicted.shape[0]}.")
    observed_roles = set(roles.tolist())
    if observed_roles != set(required_roles):
        raise ValueError(f"Wind roles differ from the declared five-role family: {sorted(observed_roles)}.")
    if not np.all(np.isfinite(predicted[mask])) or not np.all(np.isfinite(expected[mask])):
        raise ValueError("Selected Wind vector values contain NaN or infinity.")
    difference = np.asarray(predicted - expected, dtype=np.float64)
    all_truth = np.asarray(expected[mask], dtype=np.float64)
    role_results: dict[str, Any] = {}
    for role in required_roles:
        selected = mask & (roles == role)
        if not np.any(selected):
            raise ValueError(f"Declared Wind role {role!r} has no valid fixed receivers.")
        residual = difference[selected]
        truth = np.asarray(expected[selected], dtype=np.float64)
        comp_norm = np.linalg.norm(truth, axis=0)
        role_results[str(role)] = {
            "query_count": int(np.count_nonzero(roles == role)),
            "valid_count": int(np.count_nonzero(selected)),
            "component_rmse_m_s": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
            "component_signed_bias_m_s": np.mean(residual, axis=0).tolist(),
            "component_relative_l2": [
                float(np.linalg.norm(residual[:, index]) / comp_norm[index]) if comp_norm[index] > 0.0 else None
                for index in range(3)
            ],
            "component_reference_l2_norm_m_s": comp_norm.tolist(),
            "component_reference_rms_m_s": np.sqrt(np.mean(np.square(truth), axis=0)).tolist(),
            "vector_reference_rms_m_s": float(np.sqrt(np.mean(np.sum(np.square(truth), axis=1)))),
            "vector_rmse_m_s": float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1)))),
        }
    return {
        "total_query_count": int(predicted.shape[0]),
        "valid_query_count": int(np.count_nonzero(mask)),
        "vector_rmse_m_s": float(np.sqrt(np.mean(np.sum(np.square(difference[mask]), axis=1)))),
        "component_rmse_m_s": np.sqrt(np.mean(np.square(difference[mask]), axis=0)).tolist(),
        "component_signed_bias_m_s": np.mean(difference[mask], axis=0).tolist(),
        "component_reference_l2_norm_m_s": np.linalg.norm(all_truth, axis=0).tolist(),
        "component_reference_rms_m_s": np.sqrt(np.mean(np.square(all_truth), axis=0)).tolist(),
        "vector_reference_rms_m_s": float(np.sqrt(np.mean(np.sum(np.square(all_truth), axis=1)))),
        "role_metrics": role_results,
        "vector_formula": "sqrt(mean_over_valid_queries(sum_over_Ux_Uy_Uz(error_component^2)))",
    }


def signed_response_metrics(
    predicted_baseline: Any,
    predicted_perturbed: Any,
    reference_baseline: Any,
    reference_perturbed: Any,
    mask: Any,
) -> dict[str, Any]:
    """Score one saved signed endpoint pair without conflating endpoint and delta peaks."""

    pred0, pred1 = np.asarray(predicted_baseline), np.asarray(predicted_perturbed)
    ref0, ref1 = np.asarray(reference_baseline), np.asarray(reference_perturbed)
    if not (pred0.shape == pred1.shape == ref0.shape == ref1.shape):
        raise ValueError("Baseline and perturbed arrays must share one native shape.")
    mask_array = np.asarray(mask, dtype=bool)
    pred0_selected = _selected_values(pred0, mask_array).reshape(-1)
    pred1_selected = _selected_values(pred1, mask_array).reshape(-1)
    ref0_selected = _selected_values(ref0, mask_array).reshape(-1)
    ref1_selected = _selected_values(ref1, mask_array).reshape(-1)
    pred_delta_selected = pred1_selected - pred0_selected
    ref_delta_selected = ref1_selected - ref0_selected
    delta_error = pred_delta_selected - ref_delta_selected
    reference_norm = float(np.linalg.norm(ref_delta_selected))
    return {
        "count": int(ref_delta_selected.size),
        "delta_rmse": float(np.sqrt(np.mean(np.square(delta_error)))),
        "delta_mae": float(np.mean(np.abs(delta_error))),
        "delta_relative_l2": float(np.linalg.norm(delta_error) / reference_norm) if reference_norm > 0 else None,
        "measured_signed_mean_delta": float(np.mean(ref_delta_selected)),
        "predicted_signed_mean_delta": float(np.mean(pred_delta_selected)),
        "measured_endpoint_peak_change": float(np.max(ref1_selected) - np.max(ref0_selected)),
        "predicted_endpoint_peak_change": float(np.max(pred1_selected) - np.max(pred0_selected)),
        "measured_max_delta": float(np.max(ref_delta_selected)),
        "predicted_max_delta": float(np.max(pred_delta_selected)),
        "peak_definition": "endpoint_peak_change=max(perturbed)-max(baseline); max_delta=max(perturbed-baseline)",
    }


def _as_cpu_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_multiview_native_trace(trace: Mapping[str, Any], *, expected_views: int | None = None) -> dict[str, Any]:
    """Validate native R1 legacy-width or R2 split-view trace and executed aggregates.

    R1 deliberately retains scalar `[B,D,J]` routes and one full-width 128
    message. R2 exposes two `[B,D,J,2]` route tensors and two 64-wide message
    blocks. In either case `head_aggregate` is the route-specific sum across
    donors; `aggregate` remains the flattened 128-wide value passed to the
    round update.
    """

    required = {
        "s", "rho", "w", "view_ids", "weighted_message_content",
        "head_aggregate", "aggregate",
    }
    missing = required - set(trace)
    if missing:
        raise ValueError(f"Native multiview trace is missing required fields: {sorted(missing)}.")
    s, rho, weights = (_as_cpu_array(trace[key]) for key in ("s", "rho", "w"))
    if s.ndim not in (3, 4) or rho.shape != s.shape or weights.shape != s.shape:
        raise ValueError("Trace s/rho/w must share R1 `[B,D,J]` or R2 `[B,D,J,V]` shape.")
    view_ids = _as_cpu_array(trace["view_ids"])
    views = 1 if s.ndim == 3 else int(s.shape[-1])
    if view_ids.ndim != 1 or view_ids.shape[0] != views:
        raise ValueError("Trace view_ids must provide one ID per explicit or implicit view.")
    if len(set(view_ids.tolist())) != view_ids.size:
        raise ValueError("Trace view IDs must be unique.")
    if expected_views is not None and view_ids.size != expected_views:
        raise ValueError(f"Expected {expected_views} views; trace has {view_ids.size}.")
    content = _as_cpu_array(trace["weighted_message_content"])
    head_aggregate = _as_cpu_array(trace["head_aggregate"])
    aggregate = _as_cpu_array(trace["aggregate"])
    if s.ndim == 3:
        b, d, j = s.shape
        message_width = 128
        if content.shape != (b, d, j, message_width):
            raise ValueError("R1 weighted message content must retain shape `[B,D,J,128]`.")
        expected_head_shape = (b, d, 1, message_width)
    else:
        b, d, j, v = s.shape
        if v != 2:
            raise ValueError("R2 must expose exactly two learned routing views.")
        message_width = 64
        if content.shape != (b, d, j, v, message_width):
            raise ValueError("R2 weighted message content must have shape `[B,D,J,2,64]`.")
        expected_head_shape = (b, d, v, message_width)
    if aggregate.shape != (b, d, 128):
        raise ValueError("The legacy/full-message aggregate must have shape `[B,D,128]`.")
    if head_aggregate.shape != expected_head_shape:
        raise ValueError(f"Head aggregate must have native shape {expected_head_shape}.")
    for name, array in (("s", s), ("rho", rho), ("w", weights),
                        ("weighted_message_content", content),
                        ("head_aggregate", head_aggregate), ("aggregate", aggregate)):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Native trace field {name!r} contains NaN or infinity.")
    content_value = trace["weighted_message_content"]
    head_value = trace["head_aggregate"]
    full_value = trace["aggregate"]
    if hasattr(content_value, "detach") and hasattr(content_value, "sum"):
        # Preserve the saved tensor dtype and native reduction implementation.
        rebuilt_value = content_value.detach().cpu().sum(dim=2)
        head_value_cpu = head_value.detach().cpu() if hasattr(head_value, "detach") else head_value
        full_value_cpu = full_value.detach().cpu() if hasattr(full_value, "detach") else full_value
        if s.ndim == 3:
            # The historical R1 aggregate keeps an explicit singleton view axis.
            rebuilt_value = rebuilt_value.unsqueeze(-2)
        flattened_head = head_value_cpu.squeeze(-2) if s.ndim == 3 else head_value_cpu.reshape(b, d, 128)
        exact_head = bool((rebuilt_value == head_value_cpu).all())
        exact_full = bool((flattened_head == full_value_cpu).all())
        rebuilt = _as_cpu_array(rebuilt_value)
        head_aggregate = _as_cpu_array(head_value_cpu)
    else:
        rebuilt = np.sum(content, axis=2)
        head_aggregate = np.asarray(head_aggregate)
        if s.ndim == 3:
            # Match the native trace's [B,D,1,128] R1 head shape.
            rebuilt = np.expand_dims(rebuilt, axis=-2)
        flattened_head = (np.squeeze(head_aggregate, axis=-2) if s.ndim == 3
                          else np.reshape(head_aggregate, (b, d, 128)))
        exact_head = bool(np.array_equal(rebuilt, head_aggregate))
        exact_full = bool(np.array_equal(flattened_head, aggregate))
    if not exact_head:
        raise ValueError("Native-dtype donor reduction does not exactly reproduce head_aggregate.")
    if not exact_full:
        raise ValueError("Flattened head_aggregate does not exactly reproduce legacy aggregate.")
    # Report a diagnostic residual in FP64 without using it as the equality
    # criterion; the equality check above follows native dtype and reduction.
    max_abs = float(np.max(np.abs(np.asarray(rebuilt, dtype=np.float64)
                                  - np.asarray(head_aggregate, dtype=np.float64)))) if rebuilt.size else 0.0
    return {
        "view_count": views,
        "view_ids": [str(value) for value in view_ids.tolist()],
        "shape_s_rho_w": list(s.shape),
        "weighted_message_shape": list(content.shape),
        "message_width_per_view": message_width,
        "head_aggregate_shape": list(head_aggregate.shape),
        "full_aggregate_shape": list(aggregate.shape),
        "native_dtype_exact_head_reduction": exact_head,
        "native_dtype_exact_flattened_aggregate": exact_full,
        "weighted_content_reconstruction_max_abs": max_abs,
    }
