"""Physical metrics, organizer exports, and canonical result paths."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation_tools.plots import (
    error_metrics,
    masked_error_metrics,
    module_and_fluid_masks,
)
from honf_runtime.artifact_layout import EvaluationArtifactLayout, default_evaluation_root
from honf_runtime.compat import current_timestamp, resolve_demo_path


def denormalize_predictions(predictions: Dict[str, Any], dataset: GlobalChannelThermalDataset, normalize_targets: bool) -> Dict[str, Any]:
    """Convert normalized predictions."""

    if not normalize_targets:
        return predictions
    out = dict(predictions)
    out["pred_field_grid"] = dataset.normalizer.denormalize_fields(out["pred_field_grid"])
    if np.asarray(out["pred_internal_temperature"]).size:
        out["pred_internal_temperature"] = dataset.normalizer.denormalize_internal_temperature(out["pred_internal_temperature"])
    if np.asarray(out["pred_interface"]).size:
        out["pred_interface"] = dataset.normalizer.denormalize_interface_targets(out["pred_interface"])
    return out


def safe_path_name(value: object) -> str:
    """Perform the safe path name operation used by this module."""

    raw = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw) or "case"


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for artifact provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_output_dir(base_dir_arg: str | None, checkpoint_path: Path, case_id: object) -> Path:
    """Return one timestamped job under the canonical single-case evaluation root."""

    base_dir = (
        Path(base_dir_arg)
        if base_dir_arg
        else default_evaluation_root(checkpoint_path, evaluation_kind="single_case")
    )
    return resolve_demo_path(base_dir) / f"{safe_path_name(case_id)}_{current_timestamp()}"


def _as_numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
    """Convert tensors and array-like values without retaining a computation graph."""

    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _aux_array(
    aux: Dict[str, Any],
    key: str,
    default: Any,
    *,
    ndim: int | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Read one auxiliary array and remove only an explicit singleton batch axis."""

    value = aux.get(key, default)
    array = _as_numpy(value, dtype=dtype)
    if ndim is not None and array.ndim == ndim + 1 and array.shape[0] == 1:
        array = array[0]
    return array


def extract_organization_arrays(sample: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract presentation and case-adaptive organization arrays.

    The returned arrays are always a single case (the evaluation workflow
    removes its batch axis before calling this helper).  A singleton batch axis
    is nevertheless accepted so direct diagnostics can use the helper too.
    Residual fields are optional for historical organizers and receive explicit
    shape-compatible defaults in that case.
    """

    centers = _as_numpy(sample["structure"]["module_centers"])
    if centers.ndim == 3 and centers.shape[0] == 1:
        centers = centers[0]
    present = _as_numpy(sample["structure"]["module_present"]) > 0.5
    if present.ndim == 2 and present.shape[0] == 1:
        present = present[0]
    env_coords = _aux_array(aux, "env_coords", np.zeros((0, 2)), ndim=2)
    if env_coords.ndim != 2:
        env_coords = env_coords.reshape(-1, 2) if env_coords.size else np.zeros((0, 2), dtype=np.float32)
    A_eh = _aux_array(aux, "A_eh", np.zeros((env_coords.shape[0], 1)), ndim=2)
    if A_eh.ndim != 2:
        A_eh = np.zeros((env_coords.shape[0], 1), dtype=np.float32)
    edge_count = int(A_eh.shape[-1])
    A_mh = _aux_array(aux, "A_mh", np.zeros((centers.shape[0], edge_count)), ndim=2)
    if A_mh.ndim != 2:
        A_mh = np.zeros((centers.shape[0], edge_count), dtype=np.float32)
    if A_mh.shape[-1] != edge_count:
        edge_count = int(A_mh.shape[-1])
        if A_eh.shape[-1] != edge_count:
            A_eh = A_eh[:, :edge_count] if A_eh.shape[-1] > edge_count else np.pad(
                A_eh, ((0, 0), (0, edge_count - A_eh.shape[-1]))
            )
    strength = _aux_array(aux, "hyper_strength", np.ones((edge_count,), dtype=np.float32), ndim=1)
    if strength.ndim != 1:
        strength = strength.reshape(-1)
    edge_count = min(edge_count, int(strength.shape[0])) if strength.size else edge_count
    if edge_count <= 0:
        edge_count = 1
        strength = np.zeros((1,), dtype=np.float32)
        A_mh = np.zeros((centers.shape[0], 1), dtype=np.float32)
        A_eh = np.zeros((env_coords.shape[0], 1), dtype=np.float32)
    else:
        strength = strength[:edge_count]
        A_mh = A_mh[:, :edge_count]
        A_eh = A_eh[:, :edge_count]
    # Hard support is the presentation/topology contract for both adaptive
    # organizers.  Phase 2 also exports a differentiable soft support, but it
    # must never reactivate padded candidates in standard evaluation views.
    active_source = aux.get(
        "hard_case_edge_mask",
        aux.get("effective_edge_mask", aux.get("edge_active_mask")),
    )
    if active_source is None:
        active_source = np.ones((edge_count,), dtype=np.float32)
    active_mask = _as_numpy(active_source).reshape(-1)[:edge_count]
    if active_mask.size < edge_count:
        active_mask = np.pad(active_mask, (0, edge_count - active_mask.size), constant_values=0.0)
    A_me = _aux_array(
        aux,
        "A_me",
        np.zeros((centers.shape[0], env_coords.shape[0]), dtype=np.float32),
        ndim=2,
    )
    if A_me.ndim != 2:
        A_me = np.zeros((centers.shape[0], env_coords.shape[0]), dtype=np.float32)
    source = _aux_array(aux, "hyper_source_coords", np.zeros((edge_count, 2)), ndim=2)
    region = _aux_array(
        aux,
        "hyper_thermal_region_coords",
        aux.get("hyper_region_coords", np.zeros((edge_count, 2))),
        ndim=2,
    )
    if source.ndim != 2 or source.shape[-1] != 2:
        source = np.zeros((edge_count, 2), dtype=np.float32)
    if region.ndim != 2 or region.shape[-1] != 2:
        region = np.zeros((edge_count, 2), dtype=np.float32)

    def edge_array(key: str, default: Any) -> np.ndarray:
        values = _aux_array(aux, key, default, ndim=1)
        values = values.reshape(-1)
        if values.size < edge_count:
            values = np.pad(values, (0, edge_count - values.size), constant_values=0.0)
        return values[:edge_count]

    def matrix_array(key: str, default: Any, rows: int) -> np.ndarray:
        values = _aux_array(aux, key, default, ndim=2)
        if values.ndim != 2:
            return np.zeros((rows, edge_count), dtype=np.float32)
        out = np.zeros((rows, edge_count), dtype=np.float32)
        out[: min(rows, values.shape[0]), : min(edge_count, values.shape[1])] = values[:rows, :edge_count]
        return out

    content_factor = _aux_array(
        aux,
        "residual_content_factor",
        np.zeros((0, edge_count), dtype=np.float32),
        ndim=2,
    )
    if content_factor.ndim != 2:
        content_factor = np.zeros((0, edge_count), dtype=np.float32)
    elif content_factor.shape[1] != edge_count:
        if content_factor.shape[1] > edge_count:
            content_factor = content_factor[:, :edge_count]
        else:
            content_factor = np.pad(
                content_factor,
                ((0, 0), (0, edge_count - content_factor.shape[1])),
                constant_values=0.0,
            )

    residual_trace = _aux_array(
        aux,
        "residual_fraction_trace",
        np.ones((edge_count + 1,), dtype=np.float32),
        ndim=1,
    ).reshape(-1)
    if residual_trace.size < edge_count + 1:
        residual_trace = np.pad(
            residual_trace,
            (0, edge_count + 1 - residual_trace.size),
            constant_values=float(residual_trace[-1]) if residual_trace.size else 1.0,
        )
    residual_trace = residual_trace[: edge_count + 1]
    cap = int(np.sum(present))
    case_count = _aux_array(aux, "case_adaptive_edge_count", np.asarray(np.sum(active_mask > 0.5)), ndim=0).reshape(-1)
    case_cap = _aux_array(aux, "case_adaptive_edge_cap", np.asarray(cap), ndim=0).reshape(-1)
    cap_hit = _aux_array(aux, "case_adaptive_cap_hit", np.asarray(0.0), ndim=0).reshape(-1)
    stop_reached = _aux_array(aux, "case_adaptive_stop_reached", np.asarray(0.0), ndim=0).reshape(-1)
    stop_margin = _aux_array(aux, "case_adaptive_stop_margin", np.asarray(0.0), ndim=0).reshape(-1)
    return {
        "centers": centers,
        "present": present,
        "heat": _as_numpy(sample["structure"].get("heat_powers", np.zeros((centers.shape[0],)))),
        "env_coords": env_coords,
        "A_me": A_me,
        "A_eh": A_eh,
        "A_mh": A_mh,
        "strength": strength,
        "active_hyperedge_mask": active_mask,
        "effective_edge_mask": active_mask,
        "hard_case_edge_mask": edge_array("hard_case_edge_mask", active_mask),
        "edge_survival_soft": edge_array(
            "edge_survival_soft",
            aux.get("edge_survival_weight", active_mask),
        ),
        "edge_survival_weight": edge_array("edge_survival_weight", active_mask),
        "module_mass": edge_array("hyper_module_mass", np.zeros_like(strength)),
        "env_mass": edge_array("hyper_env_mass", np.zeros_like(strength)),
        "src": source[:edge_count],
        "dst": region[:edge_count],
        "case_adaptive_edge_count": case_count,
        "case_adaptive_edge_cap": case_cap,
        "case_adaptive_stop_reached": stop_reached,
        "case_adaptive_cap_hit": cap_hit,
        "case_adaptive_stop_margin": stop_margin,
        "residual_fraction_trace": residual_trace,
        "residual_marginal_explained_fraction": edge_array(
            "residual_marginal_explained_fraction", np.zeros((edge_count,))
        ),
        "residual_mechanism_strength": edge_array("residual_mechanism_strength", strength),
        "residual_module_factor": matrix_array(
            "residual_module_factor", np.zeros((centers.shape[0], edge_count)), centers.shape[0]
        ),
        "residual_environment_factor": matrix_array(
            "residual_environment_factor", np.zeros((env_coords.shape[0], edge_count)), env_coords.shape[0]
        ),
        "residual_content_factor": content_factor,
        "residual_interaction_tensor": _aux_array(
            aux,
            "residual_interaction_tensor",
            np.zeros((0, 0, 0), dtype=np.float32),
            ndim=3,
        ),
        "residual_coupling_row_mass": _aux_array(
            aux,
            "residual_coupling_row_mass",
            np.zeros((centers.shape[0],), dtype=np.float32),
            ndim=1,
        ).reshape(-1)[: centers.shape[0]],
        "residual_stop_fraction": _aux_array(aux, "residual_stop_fraction", np.asarray(0.02), ndim=0).reshape(-1),
        "residual_soft_stop_temperature": _aux_array(
            aux, "residual_soft_stop_temperature", np.asarray(0.05), ndim=0
        ).reshape(-1),
        "case_adaptive_soft_edge_count": _aux_array(
            aux,
            "case_adaptive_soft_edge_count",
            np.asarray(np.sum(edge_array("edge_survival_soft", active_mask)), dtype=np.float32),
            ndim=0,
        ).reshape(-1),
        # Keep the short alias consumed by historical diagnostics while the
        # explicit case-adaptive name remains the canonical export.
        "soft_edge_count": np.asarray(np.sum(edge_array("edge_survival_soft", active_mask)), dtype=np.float32),
    }


def _entropy(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Perform the entropy operation used by this module."""

    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 1.0e-12, None)
    return -np.sum(arr * np.log(arr), axis=axis)


def effective_rank(values: Any) -> float:
    """Return entropy effective rank for an evaluation-only matrix diagnostic."""

    array = _as_numpy(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0:
        return 0.0
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    singular = np.linalg.svd(array, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    if total <= 1.0e-12:
        return 0.0
    probability = energy / total
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1.0e-12)))))


def mean_column_cosine(values: Any) -> float:
    """Return the mean off-diagonal cosine between nonzero matrix columns."""

    array = _as_numpy(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return 1.0 if array.ndim == 2 and array.shape[1] == 1 else 0.0
    norms = np.linalg.norm(array, axis=0)
    valid = norms > 1.0e-12
    if int(valid.sum()) < 2:
        return 1.0 if int(valid.sum()) == 1 else 0.0
    normalized = array[:, valid] / norms[valid][None, :]
    cosine = normalized.T @ normalized
    upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def interaction_tensor_effective_ranks(interaction_tensor: Any) -> Dict[str, float]:
    """Calculate effective ranks of the three Phase-2 tensor unfoldings.

    The input is the optional single-case tensor ``[M,E,D_I]``.  This helper
    intentionally performs SVD only in evaluation/reporting code; training
    diagnostics should consume the organizer's cheap scalar summaries.
    """

    tensor = _as_numpy(interaction_tensor, dtype=np.float64)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or 0 in tensor.shape:
        return {
            "interaction_tensor_module_effective_rank": 0.0,
            "interaction_tensor_environment_effective_rank": 0.0,
            "interaction_tensor_content_effective_rank": 0.0,
        }
    module_unfolding = tensor.reshape(tensor.shape[0], -1)
    environment_unfolding = np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1)
    content_unfolding = np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1)
    return {
        "interaction_tensor_module_effective_rank": effective_rank(module_unfolding),
        "interaction_tensor_environment_effective_rank": effective_rank(environment_unfolding),
        "interaction_tensor_content_effective_rank": effective_rank(content_unfolding),
    }


def interaction_tensor_diagnostics(
    interaction_tensor: Any,
    module_present: Any = None,
) -> Dict[str, Any]:
    """Return Phase-2 tensor shape, positivity, inactive-padding, and rank facts."""

    tensor = _as_numpy(interaction_tensor, dtype=np.float64)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        return {}
    ranks = interaction_tensor_effective_ranks(tensor)
    finite = bool(np.isfinite(tensor).all())
    finite_values = tensor[np.isfinite(tensor)]
    if finite_values.size:
        tensor_min = float(np.min(finite_values))
        tensor_max = float(np.max(finite_values))
    else:
        tensor_min = 0.0
        tensor_max = 0.0
    result: Dict[str, Any] = {
        "interaction_tensor_shape": [int(value) for value in tensor.shape],
        "interaction_tensor_finite": finite,
        "interaction_tensor_nonnegative": bool(tensor_min >= -1.0e-7) if finite_values.size else finite,
        "interaction_tensor_min": tensor_min,
        "interaction_tensor_max": tensor_max,
        **ranks,
    }
    present = _as_numpy(module_present, dtype=np.float64) if module_present is not None else np.ones((tensor.shape[0],), dtype=np.float64)
    if present.ndim == 2 and present.shape[0] == 1:
        present = present[0]
    present = present.reshape(-1)
    inactive = np.flatnonzero(present[: tensor.shape[0]] <= 0.5)
    result["interaction_tensor_inactive_module_max"] = (
        float(np.max(np.abs(tensor[inactive]))) if inactive.size else 0.0
    )
    result["interaction_tensor_zero_baseline"] = bool(np.isclose(tensor_min, 0.0)) if finite_values.size else finite
    return result


def support_diagnostics(aux: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize Phase-2 hard/soft support consistency without changing outputs."""

    hard_value = aux.get("hard_case_edge_mask", aux.get("hard_selected_edge_mask", aux.get("edge_active_mask")))
    if hard_value is None:
        return {}
    hard = _as_numpy(hard_value, dtype=np.float64)
    soft_value = aux.get("edge_survival_soft", aux.get("edge_survival_weight"))
    effective_value = aux.get("effective_edge_mask", aux.get("edge_active_mask"))
    result: Dict[str, Any] = {}
    if soft_value is not None:
        soft = _as_numpy(soft_value, dtype=np.float64)
        if soft.shape == hard.shape:
            result["soft_hard_support_gap_mean"] = float(np.mean(np.abs(soft - hard))) if hard.size else 0.0
            result["soft_hard_support_gap_max"] = float(np.max(np.abs(soft - hard))) if hard.size else 0.0
            result["soft_support_effective_k"] = float(np.sum(soft)) if soft.size else 0.0
    if effective_value is not None:
        effective = _as_numpy(effective_value, dtype=np.float64)
        if effective.shape == hard.shape:
            gap = np.abs(effective - hard)
            result["hard_forward_support_gap_mean"] = float(np.mean(gap)) if gap.size else 0.0
            result["hard_forward_support_gap_max"] = float(np.max(gap)) if gap.size else 0.0
            result["hard_forward_support_exact"] = bool(np.all(gap <= 1.0e-6))
    result["hard_support_count"] = float(np.sum(hard > 0.5)) if hard.size else 0.0
    return result


def _mean_aux(aux: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Return a finite scalar mean for optional organizer diagnostics."""

    value = aux.get(key)
    if value is None:
        return float(default)
    values = _as_numpy(value, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float(default)


def _max_aux(aux: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Return a finite scalar maximum for optional organizer diagnostics."""

    value = aux.get(key)
    if value is None:
        return float(default)
    values = _as_numpy(value, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float(default)


def _residual_summary(aux: Dict[str, Any]) -> Dict[str, float]:
    """Summarize case-adaptive counts and residual traces without guessing K."""

    trace = _as_numpy(aux.get("residual_fraction_trace"), dtype=np.float64)
    if trace.size == 0:
        return {}
    if trace.ndim == 1:
        trace = trace[None, :]
    count = _as_numpy(aux.get("case_adaptive_edge_count"), dtype=np.float64).reshape(-1)
    if count.size == 0:
        count = np.full((trace.shape[0],), trace.shape[1] - 1, dtype=np.float64)
    if count.size == 1 and trace.shape[0] > 1:
        count = np.repeat(count, trace.shape[0])
    count = np.clip(np.rint(count[: trace.shape[0]]).astype(np.int64), 0, trace.shape[1] - 1)
    rows = np.arange(trace.shape[0])
    final = trace[rows, count]
    margins = _as_numpy(aux.get("case_adaptive_stop_margin"), dtype=np.float64)
    margins = margins[np.isfinite(margins)]
    cap_values = _as_numpy(aux.get("case_adaptive_edge_cap"), dtype=np.float64).reshape(-1)
    if cap_values.size == 1 and count.size > 1:
        cap_values = np.repeat(cap_values, count.size)
    summary = {
        "case_adaptive_edge_count_mean": float(np.mean(count)) if count.size else 0.0,
        "case_adaptive_edge_cap_mean": _mean_aux(aux, "case_adaptive_edge_cap"),
        "case_adaptive_soft_edge_count_mean": _mean_aux(
            aux,
            "case_adaptive_soft_edge_count",
            _mean_aux(aux, "soft_edge_count"),
        ),
        "case_adaptive_stop_reached_fraction": _mean_aux(aux, "case_adaptive_stop_reached"),
        "case_adaptive_cap_hit_fraction": _mean_aux(aux, "case_adaptive_cap_hit"),
        "residual_fraction_final_mean": float(np.mean(final)) if final.size else 0.0,
        "residual_fraction_final_max": float(np.max(final)) if final.size else 0.0,
        "case_adaptive_stop_margin_mean": _mean_aux(aux, "case_adaptive_stop_margin"),
        "case_adaptive_stop_margin_min": float(np.min(margins)) if margins.size else 0.0,
        "residual_monotonic_violation_max": _max_aux(aux, "residual_monotonic_violation_max"),
        "case_adaptive_k_over_cap_mean": float(np.mean(count / np.maximum(cap_values[: count.size], 1.0)))
        if cap_values.size >= count.size
        else 0.0,
    }
    final_values = final[np.isfinite(final)]
    summary["residual_fraction_final_p95"] = (
        float(np.quantile(final_values, 0.95)) if final_values.size else 0.0
    )
    marginal = _as_numpy(aux.get("residual_marginal_explained_fraction"), dtype=np.float64)
    if marginal.size:
        if marginal.ndim == 1:
            marginal = marginal[None, :]
        first = marginal[:, 0]
        last_indices = np.clip(count - 1, 0, marginal.shape[1] - 1)
        last = marginal[np.arange(min(marginal.shape[0], last_indices.size)), last_indices[: marginal.shape[0]]]
        summary["residual_first_marginal_mean"] = float(np.mean(first))
        summary["residual_last_active_marginal_mean"] = float(np.mean(last))
        for index in range(min(3, marginal.shape[1])):
            values = marginal[:, index]
            summary[f"residual_marginal_{index + 1}_mean"] = float(np.mean(values))
    else:
        summary["residual_first_marginal_mean"] = 0.0
        summary["residual_last_active_marginal_mean"] = 0.0
    summary.update(support_diagnostics(aux))
    return summary


def hypergraph_diagnostics(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Perform the hypergraph diagnostics operation used by this module."""

    aux = predictions["organizer_aux"]
    base = predictions.get("base_organizer_aux", {})
    A_mh = np.asarray(aux.get("A_mh", np.zeros((0, 0))), dtype=np.float64)
    A_eh = np.asarray(aux.get("A_eh", np.zeros((0, 0))), dtype=np.float64)
    strength = np.asarray(aux.get("hyper_strength", np.zeros((0,))), dtype=np.float64)
    module_mass = np.asarray(aux.get("hyper_module_mass", np.zeros_like(strength)), dtype=np.float64)
    env_mass = np.asarray(aux.get("hyper_env_mass", np.zeros_like(strength)), dtype=np.float64)
    num_h = max(int(strength.shape[0]), 1)
    residual_summary = _residual_summary(aux)
    case_adaptive = bool(residual_summary) or "case_adaptive_edge_count" in aux
    direct_active_count = _mean_aux(
        aux,
        "case_adaptive_edge_count",
        residual_summary.get("case_adaptive_edge_count_mean", 0.0),
    )
    static = {
        "active_edge_count": direct_active_count if case_adaptive else float(np.sum(strength > 0.05)),
        "A_mh_entropy": float(np.mean(_entropy(A_mh, axis=-1))) if A_mh.size else 0.0,
        "A_eh_entropy": float(np.mean(_entropy(A_eh, axis=-1))) if A_eh.size else 0.0,
        "module_mass_entropy_norm": float(_entropy(module_mass, axis=-1) / np.log(max(num_h, 2))) if module_mass.size else 0.0,
        "env_mass_entropy_norm": float(_entropy(env_mass, axis=-1) / np.log(max(num_h, 2))) if env_mass.size else 0.0,
        "module_mass_max": float(np.max(module_mass)) if module_mass.size else 0.0,
        "env_mass_max": float(np.max(env_mass)) if env_mass.size else 0.0,
        "hyper_strength_mean": float(np.mean(strength)) if strength.size else 0.0,
        "hyper_strength_max": float(np.max(strength)) if strength.size else 0.0,
    }
    for key in (
        "pre_fallback_zero_support_module_rows",
        "post_fallback_zero_support_module_rows",
        "pre_fallback_zero_support_environment_rows",
        "post_fallback_zero_support_environment_rows",
    ):
        if key in aux:
            static[key] = float(np.mean(_as_numpy(aux[key], dtype=np.float64)))
    static.update(residual_summary)
    tensor = aux.get("residual_interaction_tensor")
    if tensor is not None:
        module_present = aux.get("module_present")
        static.update(interaction_tensor_diagnostics(tensor, module_present))
    content_factor = aux.get("residual_content_factor")
    if content_factor is not None:
        content_array = _as_numpy(content_factor, dtype=np.float64)
        if content_array.ndim == 3 and content_array.shape[0] == 1:
            content_array = content_array[0]
        static["residual_content_factor_effective_rank"] = effective_rank(content_array)
        static["residual_content_factor_nonzero_fraction"] = (
            float(np.mean(np.abs(content_array) > 1.0e-8)) if content_array.size else 0.0
        )
        static["residual_content_factor_column_cosine"] = mean_column_cosine(content_array)
    routing_maps = predictions.get("routing_maps", {})
    routing_aux = predictions.get("routing_aux", {})
    routing = {
        key: float(routing_aux[key])
        for key in (
            "routed_module_retained_mass_mean",
            "routed_module_retained_mass_p05",
            "routed_module_retained_mass_min",
            "routed_query_edge_pair_count",
        )
        if key in routing_aux
    }
    if routing_maps:
        alpha = np.asarray(routing_maps.get("query_hyper_attention", np.zeros((0, 0, 0))), dtype=np.float64)
        pair = np.asarray(routing_maps.get("pairwise_edge_contribution", np.zeros_like(alpha)), dtype=np.float64)
        c_h = np.asarray(routing_maps.get("c_H_norm", np.zeros((0, 0))), dtype=np.float64)
        c_pair = np.asarray(routing_maps.get("c_pair_norm", np.zeros((0, 0))), dtype=np.float64)
        if alpha.size:
            entropy = _entropy(alpha, axis=-1)
            routing.update(
                {
                    "query_attention_entropy": float(np.mean(entropy)),
                    "query_attention_effective_edges": float(np.mean(np.exp(entropy))),
                    "query_attention_max": float(np.mean(np.max(alpha, axis=-1))),
                    "pairwise_edge_contribution_mean": float(np.mean(pair)) if pair.size else 0.0,
                    "c_H_norm_mean": float(np.mean(c_h)) if c_h.size else 0.0,
                    "c_pair_norm_mean": float(np.mean(c_pair)) if c_pair.size else 0.0,
                }
            )
    changes = {}
    for key, out_key in (
        ("A_mh", "A_mh_change_norm"),
        ("A_eh", "A_eh_change_norm"),
        ("hyper_source_coords", "source_coordinate_shift"),
        ("hyper_region_coords", "region_coordinate_shift"),
        ("hyper_module_mass", "module_mass_shift"),
        ("hyper_env_mass", "env_mass_shift"),
        ("hyper_strength", "strength_shift"),
    ):
        if key in aux and key in base:
            final_arr = np.asarray(aux[key], dtype=np.float64)
            base_arr = np.asarray(base[key], dtype=np.float64)
            if final_arr.shape == base_arr.shape:
                changes[out_key] = float(np.linalg.norm(final_arr - base_arr))
    return {
        "static_organization": static,
        "routing": routing,
        "base_vs_final": changes,
        "note": "Organization/routing/plan diagnostics are computed from predicted mode when both teacher and predicted modes are evaluated.",
    }


def summarize(
    raw_sample: Dict[str, Any],
    predictions: Dict[str, Any],
    checkpoint_path: Path,
    layout: EvaluationArtifactLayout,
    channel_order: list[str],
) -> Dict[str, Any]:
    """Perform the summarize operation used by this module."""

    pred = predictions["pred_field_grid"]
    gt = raw_sample["steady_field"][..., : pred.shape[-1]]
    _, fluid_mask = module_and_fluid_masks(raw_sample, pred)
    suffix = str(predictions.get("suffix", "predicted"))
    layout.ensure("arrays", "metrics")
    npz_path = layout.arrays / f"evaluation_outputs_{suffix}.npz"
    array_payload: Dict[str, Any] = {
        "pred_field_grid": pred.astype(np.float32),
        "gt_field_grid": gt.astype(np.float32),
        "pred_internal_temperature": predictions["pred_internal_temperature"].astype(np.float32),
        "pred_interface": predictions["pred_interface"].astype(np.float32),
        "pred_port_condition": predictions["pred_port_condition"].astype(np.float32),
    }
    # Phase-2's full interaction tensor is expensive and therefore only
    # arrives when explicitly requested by the caller.  Persist it in the
    # existing managed array artifact when present; ordinary evaluations keep
    # the historical compact payload.
    organizer_aux = predictions.get("organizer_aux", {})
    for key in (
        "residual_interaction_tensor",
        "residual_content_factor",
        "edge_survival_soft",
        "hard_case_edge_mask",
    ):
        if key in organizer_aux:
            value = _as_numpy(organizer_aux[key], dtype=np.float32)
            if value.size:
                array_payload[key] = value
    np.savez_compressed(npz_path, **array_payload)
    channel_metrics = {
        str(name): masked_error_metrics(pred[..., idx], gt[..., idx], fluid_mask)
        for idx, name in enumerate(channel_order[: pred.shape[-1]])
    }
    field_metrics = error_metrics(pred, gt)
    field_metrics_fluid = masked_error_metrics(pred, gt, fluid_mask)
    temperature_metrics_fluid = masked_error_metrics(pred[..., 4], gt[..., 4], fluid_mask) if pred.shape[-1] >= 5 else None
    metrics_csv_path = layout.metrics / f"metrics_{suffix}.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "suffix",
                "field_mse",
                "field_rmse",
                "field_mae",
                "field_relative_l2",
                "fluid_mse",
                "fluid_rmse",
                "fluid_mae",
                "fluid_relative_l2",
                "temperature_fluid_mse",
                "temperature_fluid_rmse",
                "temperature_fluid_mae",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": str(raw_sample["case_id"]),
                "suffix": suffix,
                "field_mse": field_metrics.get("mse"),
                "field_rmse": field_metrics.get("rmse"),
                "field_mae": field_metrics.get("mae"),
                "field_relative_l2": field_metrics.get("relative_l2"),
                "fluid_mse": field_metrics_fluid.get("mse"),
                "fluid_rmse": field_metrics_fluid.get("rmse"),
                "fluid_mae": field_metrics_fluid.get("mae"),
                "fluid_relative_l2": field_metrics_fluid.get("relative_l2"),
                "temperature_fluid_mse": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("mse"),
                "temperature_fluid_rmse": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("rmse"),
                "temperature_fluid_mae": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("mae"),
            }
        )
    return {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "phase": "prompt3_physical_coupling",
        "field_metrics": field_metrics,
        "field_metrics_fluid": field_metrics_fluid,
        "temperature_metrics_fluid": temperature_metrics_fluid,
        "field_channel_metrics_fluid": channel_metrics,
        "internal_interface_note": "Skipped only when internal/interface tensors are empty.",
        "interface_flux_mode": str(predictions.get("interface_flux_mode", "unknown")),
        "outputs": {
            "global_field_quicklook": str(layout.fields / f"global_field_quicklook_{suffix}.png"),
            "npz": str(npz_path),
            "metrics_csv": str(metrics_csv_path),
        },
    }
