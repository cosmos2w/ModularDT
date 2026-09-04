#!/usr/bin/env python3
"""Evaluate case-adaptive residual organization on a deterministic case subset.

This is the maintained multi-case evaluator for the residual organizer.  It
restores each checkpoint's model and normalization, runs the existing prepared
decoder, and writes one per-case CSV plus a JSON summary.  The summary includes
evaluation-only support/organizer-reliance discrepancies, K-by-module-count
summaries, and optional Phase-2 interaction-tensor unfolding ranks.
Residual arrays are serialized as JSON strings in the CSV so a row remains
self-contained; the single-case evaluator remains the place for PNG
organization views.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import inspect
import json
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Case_ThermalChannel" / "src"))

from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
from channelthermal.evaluation.loading import load_model, make_batch
from channelthermal.evaluation.prepared import predict_case
from channelthermal.evaluation.results import (
    extract_organization_arrays,
    interaction_tensor_diagnostics,
    support_diagnostics,
)
from channelthermal.evaluation_tools.plots import (
    error_metrics,
    masked_error_metrics,
    module_and_fluid_masks,
    module_radius_from_sample,
)
from channelthermal.training.epoch import (
    effective_local_loss_weights,
    effective_port_condition_settings,
    effective_port_global_weight,
    interface_loss,
    internal_loss,
    organizer_regularization,
    port_condition_loss,
    port_cyclic_smoothness_loss,
    port_global_consistency_loss,
    predicted_consistency_weight_for_epoch,
)
from channelthermal.training_tools.losses import channelthermal_field_mse

from honf_runtime.artifact_layout import (
    EvaluationArtifactLayout,
    default_evaluation_root,
    finalize_evaluation_job,
)

EPS = 1.0e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the focused multi-case evaluator options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Canonical evaluation job directory; defaults to a timestamped managed job beside the first checkpoint.",
    )
    parser.add_argument(
        "--stability-perturbations",
        action="store_true",
        help="Repeat selected cases with a deterministic module permutation and query chunk size.",
    )
    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=1,
        help="Number of deterministic permutation/chunk repeats when stability diagnostics are enabled.",
    )
    parser.add_argument(
        "--return-routing-maps",
        action="store_true",
        help="Request query-routing maps and report query/pairwise effective ranks.",
    )
    parser.add_argument(
        "--tensor-diagnostics",
        "--return-interaction-tensor",
        dest="tensor_diagnostics",
        action="store_true",
        help="Explicitly request Phase-2 interaction tensors and report unfolding ranks/content support.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Include wall-clock and CUDA memory measurements in rows and the summary.",
    )
    parser.add_argument(
        "--benchmark-query-count",
        action="append",
        type=int,
        default=[],
        help="Prepared-decode query volume to benchmark; repeat for multiple Q values.",
    )
    parser.add_argument(
        "--forensic-audit",
        action="store_true",
        help=(
            "Run the bounded Run-1701 collapse audit: tensor concentration, "
            "hard/full/H0 support accuracy, soft-gate derivatives, and one "
            "read-only pre-clip gradient audit per checkpoint."
        ),
    )
    parser.add_argument(
        "--gradient-audit",
        action="store_true",
        help="Run one read-only real-data training-loss backward pass per checkpoint.",
    )
    parser.add_argument(
        "--gradient-audit-query-count",
        type=int,
        default=8192,
        help="Deterministic query count for the read-only forensic backward pass.",
    )
    parser.add_argument(
        "--case-edge-selection-mode",
        choices=("none", "probe_fidelity"),
        default="none",
        help="Evaluation-only query-edge selection over a fixed prepared bank.",
    )
    parser.add_argument(
        "--probe-relative-rms-tolerance",
        type=float,
        default=0.005,
        help="Overall relative-RMS fidelity tolerance for probe_fidelity selection.",
    )
    parser.add_argument(
        "--probe-channel-tolerance",
        type=float,
        default=0.01,
        help="Maximum per-channel relative-RMS fidelity tolerance on probes.",
    )
    return parser.parse_args(argv)


def checkpoint_specs(values: Iterable[str]) -> dict[str, Path]:
    """Parse unique ``LABEL=PATH`` checkpoint specifications."""

    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, value = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Checkpoint label cannot be empty: {item!r}")
        if label in result:
            raise ValueError(f"Duplicate checkpoint label: {label!r}")
        result[label] = Path(value).expanduser().resolve()
    return result


def sha256_file(path: Path) -> str:
    """Return a streaming file digest for evaluator provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_rank(values: np.ndarray) -> float:
    """Compute entropy effective rank for one two-dimensional diagnostic map."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0:
        return 0.0
    singular = np.linalg.svd(array, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    if total <= EPS:
        return 0.0
    probability = energy / total
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, EPS)))))


def _mean_column_cosine(values: np.ndarray) -> float:
    """Return mean off-diagonal cosine similarity between matrix columns."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return 1.0 if array.ndim == 2 and array.shape[1] == 1 else 0.0
    norms = np.linalg.norm(array, axis=0)
    valid = norms > EPS
    if int(valid.sum()) < 2:
        return 1.0 if int(valid.sum()) == 1 else 0.0
    normalized = array[:, valid] / norms[valid][None, :]
    cosine = normalized.T @ normalized
    upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _normalized_separation(points: np.ndarray, sample: dict[str, Any]) -> float:
    """Return pairwise center separation normalized by the physical diagonal."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return 0.0
    deltas = values[:, None, :2] - values[None, :, :2]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    upper = distances[np.triu_indices(values.shape[0], k=1)]
    if not upper.size:
        return 0.0
    structure = sample.get("structure", {})
    length_x = _as_float(structure.get("domain_length_x"), 0.0)
    length_y = _as_float(structure.get("domain_length_y"), 0.0)
    diagonal = float(np.hypot(length_x, length_y)) if length_x > 0 and length_y > 0 else 1.0
    return float(np.mean(upper) / max(diagonal, EPS))


def _active_matrix(values: Any, active_mask: np.ndarray) -> np.ndarray:
    """Select hard-active mechanism columns from one assignment/factor matrix."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        return np.zeros((0, 0), dtype=np.float64)
    mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return np.zeros((array.shape[0], 0), dtype=np.float64)
    if mask.shape[0] < array.shape[1]:
        return array
    return array[:, mask[: array.shape[1]]]


def _phase2_case_metrics(
    sample: dict[str, Any],
    arrays: dict[str, np.ndarray],
    aux: dict[str, Any],
    hard_mask: np.ndarray,
    count: float,
) -> dict[str, Any]:
    """Collect tensor/rank/support/K-by-M diagnostics for one Phase-2 case."""

    metrics: dict[str, Any] = {
        "interaction_tensor_shape": None,
        "interaction_tensor_module_effective_rank": None,
        "interaction_tensor_environment_effective_rank": None,
        "interaction_tensor_content_effective_rank": None,
        "interaction_tensor_finite": None,
        "interaction_tensor_nonnegative": None,
        "interaction_tensor_inactive_module_max": None,
        "interaction_tensor_min": None,
        "interaction_tensor_max": None,
        "interaction_tensor_total_mass": None,
        "interaction_tensor_max_channel_mass_fraction": None,
        "interaction_tensor_channel_effective_count": None,
        "first_component_explained_fraction": None,
        "soft_gate_derivative_mean": None,
        "soft_gate_derivative_p95": None,
        "soft_gate_derivative_max": None,
        "residual_content_factor_effective_rank": None,
        "residual_content_factor_nonzero_fraction": None,
        "residual_content_factor_column_cosine": None,
        "selected_module_effective_rank": None,
        "selected_environment_effective_rank": None,
        "selected_environment_edge_column_cosine": None,
        "selected_region_separation_normalized": None,
        "selected_environment_rank_over_hard_k": None,
        "soft_hard_support_gap_mean": None,
        "soft_hard_support_gap_max": None,
        "hard_forward_support_gap_mean": None,
        "hard_forward_support_gap_max": None,
        "hard_forward_support_exact": None,
        "hard_support_count": None,
        "soft_support_effective_k": None,
        "residual_monotonic_violation_max": None,
        "empty_selected_edge_count": None,
        "post_fallback_zero_support_module_rows": None,
        "post_fallback_zero_support_environment_rows": None,
        "case_adaptive_k_over_module_count": None,
        "case_adaptive_k_over_cap": None,
    }
    tensor = arrays.get("residual_interaction_tensor")
    if tensor is not None and np.asarray(tensor).size:
        tensor_metrics = interaction_tensor_diagnostics(
            tensor,
            arrays.get("present"),
        )
        metrics.update(tensor_metrics)
        tensor_array = np.asarray(tensor, dtype=np.float64)
        if tensor_array.ndim == 4 and tensor_array.shape[0] == 1:
            tensor_array = tensor_array[0]
        if tensor_array.ndim == 3:
            channel_mass = np.abs(tensor_array).sum(axis=(0, 1))
            total_mass = float(channel_mass.sum())
            metrics["interaction_tensor_total_mass"] = total_mass
            if total_mass > EPS:
                probability = channel_mass / total_mass
                metrics["interaction_tensor_max_channel_mass_fraction"] = float(probability.max())
                metrics["interaction_tensor_channel_effective_count"] = float(
                    np.exp(-np.sum(probability * np.log(np.maximum(probability, EPS))))
                )
            else:
                metrics["interaction_tensor_max_channel_mass_fraction"] = 0.0
                metrics["interaction_tensor_channel_effective_count"] = 0.0
    marginal = np.asarray(
        aux.get("residual_marginal_explained_fraction", []), dtype=np.float64
    ).reshape(-1)
    if marginal.size:
        metrics["first_component_explained_fraction"] = float(marginal[0])
    trace = np.asarray(aux.get("residual_fraction_trace", []), dtype=np.float64).reshape(-1)
    cap = max(int(round(_as_float(aux.get("case_adaptive_edge_cap"), count))), 1)
    if trace.size >= 2:
        previous = np.empty(cap, dtype=np.float64)
        available = min(cap, trace.size - 1)
        previous[:available] = trace[:available]
        if available < cap:
            previous[available:] = trace[-1]
        stop = _as_float(aux.get("residual_stop_fraction"), 0.01)
        temperature = max(_as_float(aux.get("residual_soft_stop_temperature"), 0.002), EPS)
        logits = np.clip((previous - stop) / temperature, -80.0, 80.0)
        survival = 1.0 / (1.0 + np.exp(-logits))
        derivative = survival * (1.0 - survival) / temperature
        # The required first mechanism is not controlled by a sigmoid gate.
        derivative[0] = 0.0
        metrics["soft_gate_derivative_mean"] = float(np.mean(derivative))
        metrics["soft_gate_derivative_p95"] = float(np.quantile(derivative, 0.95))
        metrics["soft_gate_derivative_max"] = float(np.max(derivative))
    content = np.asarray(arrays.get("residual_content_factor", np.zeros((0, 0))), dtype=np.float64)
    if content.ndim == 3 and content.shape[0] == 1:
        content = content[0]
    if content.ndim == 2 and content.size:
        content_mask = np.asarray(hard_mask, dtype=bool).reshape(-1)
        active_columns = (
            content[:, content_mask[: content.shape[1]]]
            if content_mask.size >= content.shape[1]
            else content
        )
        metrics["residual_content_factor_effective_rank"] = effective_rank(active_columns)
        metrics["residual_content_factor_nonzero_fraction"] = (
            float(np.mean(np.abs(active_columns) > 1.0e-8)) if active_columns.size else 0.0
        )
        metrics["residual_content_factor_column_cosine"] = _mean_column_cosine(active_columns)

    support = support_diagnostics(aux)
    metrics.update({key: value for key, value in support.items() if key in metrics})
    metrics["residual_monotonic_violation_max"] = _as_max_float(
        aux.get("residual_monotonic_violation_max")
    )
    metrics["empty_selected_edge_count"] = _as_max_float(
        aux.get("empty_selected_edge_count")
    )
    metrics["post_fallback_zero_support_module_rows"] = _as_max_float(
        aux.get("post_fallback_zero_support_module_rows")
    )
    metrics["post_fallback_zero_support_environment_rows"] = _as_max_float(
        aux.get("post_fallback_zero_support_environment_rows")
    )
    module_assignment = _active_matrix(arrays.get("A_mh"), hard_mask)
    environment_assignment = _active_matrix(arrays.get("A_eh"), hard_mask)
    if module_assignment.size:
        metrics["selected_module_effective_rank"] = effective_rank(module_assignment)
    if environment_assignment.size:
        metrics["selected_environment_effective_rank"] = effective_rank(environment_assignment)
        metrics["selected_environment_edge_column_cosine"] = _mean_column_cosine(environment_assignment)
    active_indices = np.flatnonzero(np.asarray(hard_mask, dtype=bool))
    regions = np.asarray(arrays.get("dst", np.zeros((0, 2))), dtype=np.float64)
    if regions.ndim == 2 and active_indices.size and regions.shape[0] > int(active_indices.max()):
        metrics["selected_region_separation_normalized"] = _normalized_separation(regions[active_indices], sample)
    module_count = int(np.sum(np.asarray(arrays.get("present", []), dtype=np.float64) > 0.5))
    cap = _as_float(aux.get("case_adaptive_edge_cap"), float(max(module_count, 1)))
    metrics["case_adaptive_k_over_module_count"] = float(count / max(module_count, 1))
    metrics["case_adaptive_k_over_cap"] = float(count / max(cap, 1.0))
    if metrics.get("selected_environment_effective_rank") is not None:
        metrics["selected_environment_rank_over_hard_k"] = float(
            metrics["selected_environment_effective_rank"] / max(count, 1.0)
        )
    return metrics


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays into JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    """Reduce an optional scalar/array to a finite float."""

    if value is None:
        return float(default)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float(default)


def _as_max_float(value: Any, default: float = 0.0) -> float:
    """Reduce an optional scalar/array to its finite maximum."""

    if value is None:
        return float(default)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else float(default)


def _near_interface_mask(sample: dict[str, Any], fluid_mask: np.ndarray) -> np.ndarray:
    """Build a small geometry-only near-interface mask for reporting."""

    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float32) > 0.5
    radius = module_radius_from_sample(sample)
    if not np.any(present):
        return np.zeros_like(fluid_mask, dtype=bool)
    distance = np.min(
        np.stack([np.hypot(x_grid - cx, y_grid - cy) - radius for cx, cy in centers[present]], axis=0),
        axis=0,
    )
    return np.asarray(fluid_mask, dtype=bool) & (distance >= 0.0) & (distance <= 0.25)


def _far_field_mask(sample: dict[str, Any], fluid_mask: np.ndarray) -> np.ndarray:
    """Build the established geometry-only far-field mask for reporting."""

    x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
    y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"], dtype=np.float32) > 0.5
    radius = module_radius_from_sample(sample)
    if not np.any(present):
        return np.asarray(fluid_mask, dtype=bool)
    distance = np.min(
        np.stack(
            [np.hypot(x_grid - cx, y_grid - cy) - radius for cx, cy in centers[present]],
            axis=0,
        ),
        axis=0,
    )
    return np.asarray(fluid_mask, dtype=bool) & (distance >= 1.0)


def _target_field(sample: dict[str, Any], dataset: GlobalChannelThermalDataset) -> np.ndarray:
    """Return the grid target in the same normalization space as prediction."""

    target = np.asarray(sample["steady_field"], dtype=np.float32)
    if bool(dataset.normalize_targets):
        target = dataset.normalizer.normalize_fields(target)
    return np.asarray(target, dtype=np.float32)


def _permuted_sample(sample: dict[str, Any], permutation: np.ndarray) -> dict[str, Any]:
    """Permute all known module-axis tensors while preserving a case sample."""

    out = copy.deepcopy(sample)
    structure = out["structure"]
    for key in ("module_centers", "heat_powers", "module_present"):
        if key in structure:
            structure[key] = np.asarray(structure[key])[permutation].copy()
    for key in (
        "interface_condition",
        "interface_condition_valid_mask",
        "interface_target",
        "teacher_port_tokens",
        "local_module_params",
        "module_internal_temperature_points",
        "module_internal_mask",
        "module_internal_query_points",
    ):
        if key in out and np.asarray(out[key]).ndim > 0 and np.asarray(out[key]).shape[0] == permutation.size:
            out[key] = np.asarray(out[key])[permutation].copy()
    return out


def _clone_organizer(organizer: dict[str, Any]) -> dict[str, Any]:
    """Clone a prepared organizer without sharing mutable tensor storage."""

    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in organizer.items()
    }


def _organizer_edge_mask(organizer: dict[str, Any]) -> torch.Tensor | None:
    """Return the explicit runtime edge mask as ``[B,K]`` when available."""

    value = organizer.get("effective_edge_mask")
    if not torch.is_tensor(value):
        value = organizer.get("edge_active_mask")
    if not torch.is_tensor(value):
        return None
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        return None
    return value > 0


def _interaction_tensor_request_kwargs(model: Any, requested: bool) -> dict[str, bool]:
    """Return the optional wrapper flag used to request Phase-2 tensors.

    Tensor export is deliberately opt-in because the full ``[M,E,D_I]``
    object is only needed for evaluation ranks and figures.  The aliases make
    this evaluator tolerant of the wrapper/core spelling selected by the
    implementation branch while ordinary checkpoints remain unchanged.
    """

    if not requested:
        return {}
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual proxy models
        return {}
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    for name in (
        "return_interaction_tensor",
        "return_residual_interaction_tensor",
        "return_tensor_diagnostics",
        # The ThermalChannel facade currently exposes its explicit tensor
        # request as the broader organizer-diagnostics switch.
        "return_organizer_diagnostics",
    ):
        if name in parameters or accepts_kwargs:
            return {name: True}
    return {}


def _prepare_case(
    model: Any,
    sample: dict[str, Any],
    device: torch.device,
    *,
    query_batch_size: int,
    tensor_diagnostics: bool = False,
) -> tuple[np.ndarray, Any]:
    """Prepare one case once and retain the wrapper's prepared decoder state."""

    x_grid = np.asarray(sample["x_grid"])
    y_grid = np.asarray(sample["y_grid"])
    query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
    if query_xy.shape[0] == 0:
        raise ValueError("Cannot prepare a case with an empty evaluation grid.")
    chunk = query_xy[: max(1, int(query_batch_size))]
    with torch.no_grad():
        batch = make_batch(sample, chunk, device)
        output = model(
            batch["structure"],
            batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.5,
            return_routing_maps=False,
            return_edge_fields=False,
            return_prepared_state=True,
            **_interaction_tensor_request_kwargs(model, tensor_diagnostics),
        )
    prepared = output.get("prepared_state")
    if prepared is None or not hasattr(prepared, "organizer"):
        raise RuntimeError("The model did not return a PreparedChannelThermalCase for ablation diagnostics.")
    # Some wrappers expose requested diagnostics in ``organizer_aux`` or the
    # top-level output but only retain the standard organizer dictionary in
    # ``PreparedChannelThermalCase``.  Copy just the opt-in tensor keys into
    # that retained state so downstream chunked rank/visualization code can
    # reuse the same prepared case.
    if tensor_diagnostics:
        for source in (output.get("organizer_aux", {}), output):
            if not isinstance(source, dict):
                continue
            for key in (
                "residual_interaction_tensor",
                "residual_content_factor",
            ):
                if key in source:
                    prepared.organizer[key] = source[key]
    return query_xy, prepared


def _decode_prepared_grid(
    model: Any,
    prepared: Any,
    query_xy: np.ndarray,
    device: torch.device,
    *,
    query_batch_size: int,
    organizer: dict[str, Any] | None = None,
    disable_hyper_value: bool = False,
    grid_shape: tuple[int, ...],
) -> np.ndarray:
    """Decode a prepared case with an evaluation-only organizer/config override."""

    target_organizer = prepared.organizer if organizer is None else organizer
    decoder_config = model.core.decoder.config
    previous_hyper_value = bool(decoder_config.use_hyper_value_context)
    if disable_hyper_value:
        # The config object is mutable for checkpoint-owned evaluation
        # overrides.  Restore it in ``finally`` so an ablation can never leak
        # into the next case/checkpoint.
        decoder_config.use_hyper_value_context = False
    chunks: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, query_xy.shape[0], max(1, int(query_batch_size))):
                chunk = torch.from_numpy(query_xy[start : start + max(1, int(query_batch_size))]).unsqueeze(0).to(device)
                output = model.core.decode_queries(
                    query_xy=chunk,
                    query_time=None,
                    organizer_output=target_organizer,
                    global_token=prepared.global_token,
                    query_features=model._query_features(chunk),
                    return_routing_maps=False,
                    return_edge_fields=False,
                )
                chunks.append(output["pred_field"].detach().cpu().numpy()[0])
    finally:
        decoder_config.use_hyper_value_context = previous_hyper_value
    if not chunks:
        return np.zeros((*grid_shape, int(model.config.field_dim)), dtype=np.float32)
    return np.concatenate(chunks, axis=0).reshape(*grid_shape, int(model.config.field_dim)).astype(np.float32)


def _benchmark_prepared_queries(
    model: Any,
    prepared: Any,
    sample: dict[str, Any],
    device: torch.device,
    *,
    query_batch_size: int,
    query_counts: Sequence[int],
    selection_seconds: float,
) -> dict[str, Any]:
    """Benchmark cached selected/full decodes at explicit query volumes."""

    if not query_counts:
        return {}
    base = np.stack(
        [
            np.asarray(sample["x_grid"]).reshape(-1),
            np.asarray(sample["y_grid"]).reshape(-1),
        ],
        axis=-1,
    ).astype(np.float32)
    if base.shape[0] == 0:
        raise ValueError("Cannot benchmark an empty query grid.")
    full_organizer = dict(prepared.organizer)
    full_organizer.pop("predictive_edge_mask", None)
    organizers = {
        "selected": prepared.organizer,
        "full": full_organizer,
    }
    metrics: dict[str, Any] = {}
    with torch.no_grad():
        for query_count in query_counts:
            count = int(query_count)
            indices = np.arange(count, dtype=np.int64) % int(base.shape[0])
            queries = torch.from_numpy(base[indices]).unsqueeze(0).to(device=device)

            def decode_all(organizer: dict[str, Any]) -> None:
                for start in range(0, count, max(1, int(query_batch_size))):
                    chunk = queries[:, start : start + max(1, int(query_batch_size))]
                    model.core.decode_queries(
                        query_xy=chunk,
                        query_time=None,
                        organizer_output=organizer,
                        global_token=prepared.global_token,
                        query_features=model._query_features(chunk),
                        return_routing_maps=False,
                        return_edge_fields=False,
                    )

            # Warm both dictionary variants, then alternate measurement order
            # to avoid giving the full-bank path a systematic cache advantage.
            for organizer in organizers.values():
                warm = queries[:, : min(count, max(1, int(query_batch_size)))]
                model.core.decode_queries(
                    query_xy=warm,
                    query_time=None,
                    organizer_output=organizer,
                    global_token=prepared.global_token,
                    query_features=model._query_features(warm),
                    return_routing_maps=False,
                    return_edge_fields=False,
                )
            timings: dict[str, list[float]] = {label: [] for label in organizers}
            peaks: dict[str, list[int]] = {label: [] for label in organizers}
            for repeat in range(3):
                labels = list(organizers)
                if repeat % 2:
                    labels.reverse()
                for label in labels:
                    organizer = organizers[label]
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    decode_all(organizer)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings[label].append(float(time.perf_counter() - started))
                    peaks[label].append(
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    )
            for label in organizers:
                elapsed = float(np.median(np.asarray(timings[label], dtype=np.float64)))
                metrics[f"benchmark_{label}_prepared_seconds_q{count}"] = elapsed
                metrics[f"benchmark_{label}_peak_allocated_bytes_q{count}"] = (
                    int(max(peaks[label]))
                    if device.type == "cuda"
                    else None
                )
            metrics[f"benchmark_selected_total_seconds_q{count}"] = float(
                selection_seconds
                + metrics[f"benchmark_selected_prepared_seconds_q{count}"]
            )
    return metrics


def _uniform_incidence_organizer(organizer: dict[str, Any]) -> dict[str, Any] | None:
    """Replace active incidence rows with row-preserving uniform values."""

    active = _organizer_edge_mask(organizer)
    if active is None or active.shape[-1] == 0:
        return None
    output = _clone_organizer(organizer)
    active_count = active.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=torch.float32)
    uniform = active.to(dtype=torch.float32).unsqueeze(1) / active_count.unsqueeze(1)
    changed = False
    for key in ("A_mh", "A_eh"):
        incidence = output.get(key)
        if not torch.is_tensor(incidence) or incidence.ndim != 3 or incidence.shape[0] != active.shape[0]:
            continue
        if incidence.shape[-1] != active.shape[-1]:
            continue
        row_mass = incidence.sum(dim=-1, keepdim=True)
        output[key] = uniform.to(dtype=incidence.dtype) * row_mass
        changed = True
    return output if changed else None


def _soft_support_organizer(model: Any, organizer: dict[str, Any]) -> dict[str, Any] | None:
    """Reconstruct the training-style soft residual support for evaluation."""

    trace = organizer.get("residual_fraction_trace")
    strength = organizer.get("residual_mechanism_strength")
    if not torch.is_tensor(trace) or not torch.is_tensor(strength):
        return None
    if trace.ndim == 1:
        trace = trace.unsqueeze(0)
    if strength.ndim == 1:
        strength = strength.unsqueeze(0)
    if trace.ndim != 2 or strength.ndim != 2 or trace.shape[1] != strength.shape[1] + 1:
        return None
    active = _organizer_edge_mask(organizer)
    if active is None or active.shape != strength.shape:
        return None
    viable = organizer.get("candidate_edge_viable_mask")
    if not torch.is_tensor(viable) or viable.shape != strength.shape:
        viable = torch.ones_like(strength, dtype=torch.bool)
    module_present = organizer.get("module_present")
    if torch.is_tensor(module_present):
        if module_present.ndim == 1:
            module_present = module_present.unsqueeze(0)
        module_count = (module_present > 0).sum(dim=-1)
    else:
        module_count = active.sum(dim=-1)
    config = model.core.config
    minimum = torch.minimum(
        module_count.to(device=strength.device, dtype=torch.long),
        torch.full_like(module_count, max(int(config.minimum_active_edges), 1)),
    )
    steps = torch.arange(1, strength.shape[1] + 1, device=strength.device).unsqueeze(0)
    stop_fraction = float(config.residual_stop_fraction)
    temperature = max(float(config.residual_soft_stop_temperature), EPS)
    soft = torch.where(
        steps <= minimum.unsqueeze(-1),
        torch.ones_like(strength),
        torch.sigmoid((trace[:, :-1] - stop_fraction) / temperature),
    )
    soft = soft * viable.to(dtype=strength.dtype)
    output = _clone_organizer(organizer)
    output["edge_survival_weight"] = soft
    output["effective_edge_mask"] = soft
    output["edge_transition_gate"] = soft
    output["case_adaptive_soft_edge_count"] = soft.sum(dim=-1)
    for factor_key, incidence_key in (
        ("residual_module_factor", "A_mh"),
        ("residual_environment_factor", "A_eh"),
    ):
        factor = output.get(factor_key)
        original = output.get(incidence_key)
        if not torch.is_tensor(factor) or not torch.is_tensor(original) or factor.ndim != 3:
            continue
        # Match the residual organizer contract: survival and mechanism
        # strength are applied before each token's incidence is normalized
        # across the packed mechanism axis (not across module/environment
        # tokens).  Keep its exact zero-support fallback as well: for an
        # active token, the first viable extracted mechanism receives the
        # factor mass; padded module tokens remain exact zeros.
        support = soft * strength
        weighted = factor * support.unsqueeze(1).to(dtype=factor.dtype)
        retained_mass = weighted.sum(dim=-1)
        if incidence_key == "A_mh":
            active_tokens = organizer.get("module_present")
            if torch.is_tensor(active_tokens):
                if active_tokens.ndim == 1:
                    active_tokens = active_tokens.unsqueeze(0)
                active_tokens = active_tokens.to(device=factor.device, dtype=torch.bool)
            else:
                active_tokens = torch.ones_like(retained_mass, dtype=torch.bool)
        else:
            active_tokens = torch.ones_like(retained_mass, dtype=torch.bool)
        if active_tokens.shape != retained_mass.shape:
            active_tokens = torch.ones_like(retained_mass, dtype=torch.bool)
        zero_rows = active_tokens & (retained_mass <= EPS)
        first_index = (support > 0).to(dtype=factor.dtype).argmax(dim=-1)
        first_hot = torch.nn.functional.one_hot(first_index, num_classes=factor.shape[-1]).to(dtype=factor.dtype)
        fallback = factor * first_hot.unsqueeze(1)
        weighted = weighted + fallback * zero_rows.unsqueeze(-1).to(dtype=factor.dtype)
        normalized = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(EPS)
        output[incidence_key] = normalized * active_tokens.unsqueeze(-1).to(dtype=factor.dtype)
    return output


def _pooled_mechanism_organizer(organizer: dict[str, Any]) -> dict[str, Any] | None:
    """Collapse active mechanisms into one pooled packed mechanism."""

    active = _organizer_edge_mask(organizer)
    hyper_state = organizer.get("hyper_state")
    if active is None or not torch.is_tensor(hyper_state):
        return None
    if hyper_state.ndim != 3 or hyper_state.shape[1] != active.shape[1] or not bool(active.any()):
        return None
    output = _clone_organizer(organizer)
    batch_size, packed_count = active.shape
    weights = active.to(dtype=hyper_state.dtype)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def pool_mechanism_axis(value: Any) -> Any:
        if not torch.is_tensor(value) or value.ndim < 2 or value.shape[0] != batch_size or value.shape[1] != packed_count:
            return value
        view = weights.reshape(batch_size, packed_count, *([1] * (value.ndim - 2)))
        pooled = (value * view).sum(dim=1) / denominator.reshape(batch_size, *([1] * (value.ndim - 2)))
        result = torch.zeros_like(value)
        result[:, 0] = pooled
        return result

    def sum_edge_axis(value: Any) -> Any:
        if not torch.is_tensor(value) or value.ndim < 3 or value.shape[0] != batch_size or value.shape[-1] != packed_count:
            return value
        view = weights.reshape(batch_size, *([1] * (value.ndim - 2)), packed_count)
        pooled = (value * view).sum(dim=-1, keepdim=True)
        result = torch.zeros_like(value)
        result[..., :1] = pooled
        return result

    for key in (
        "hyper_state",
        "candidate_hyper_state",
        "hyper_source_coords",
        "hyper_source_variance",
        "hyper_source_scale",
        "hyper_region_coords",
        "hyper_region_variance",
        "hyper_region_scale",
        "hyper_module_mass_raw",
        "hyper_env_mass_raw",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_module_purity",
        "hyper_env_purity",
        "candidate_module_mass_fraction",
        "candidate_environment_mass_fraction",
        "candidate_module_purity",
        "candidate_environment_purity",
        "candidate_source_coords",
        "candidate_source_scale",
        "candidate_region_coords",
        "candidate_region_scale",
        "hyper_strength",
        "edge_quality",
        "mechanism_geometry_features",
        "mechanism_mass_features",
        "mechanism_raw_features",
        "mechanism_descriptor_features",
        "hyper_source_region_distance",
        "hyper_source_region_downstream",
        "hyper_source_region_lateral",
    ):
        if key in output:
            output[key] = pool_mechanism_axis(output[key])
    for key in ("A_mh", "A_eh", "candidate_A_mh", "candidate_A_eh", "residual_module_factor", "residual_environment_factor"):
        if key in output:
            output[key] = sum_edge_axis(output[key])
    one = torch.zeros_like(active, dtype=hyper_state.dtype)
    one[:, 0] = 1.0
    for key in (
        "edge_active_mask",
        "effective_edge_mask",
        "edge_transition_gate",
        "hard_case_edge_mask",
        "hard_selected_edge_mask",
        "edge_survival_weight",
        "edge_viable_mask",
        "candidate_edge_viable_mask",
    ):
        value = output.get(key)
        if torch.is_tensor(value) and value.shape == active.shape:
            output[key] = one.to(dtype=value.dtype)
    output["case_adaptive_edge_count"] = one.sum(dim=-1)
    output["case_adaptive_soft_edge_count"] = one.sum(dim=-1)
    return output


@contextlib.contextmanager
def _temporary_core_setting(model: Any, name: str, value: Any) -> Iterable[None]:
    """Temporarily change one shared core setting and restore every reference."""

    candidates = (
        getattr(getattr(model, "config", None), "core_honf", None),
        getattr(getattr(model, "core", None), "config", None),
        getattr(getattr(getattr(model, "core", None), "organizer", None), "config", None),
        getattr(getattr(getattr(model, "core", None), "decoder", None), "config", None),
    )
    changed: list[tuple[Any, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen or not hasattr(candidate, name):
            continue
        seen.add(id(candidate))
        changed.append((candidate, getattr(candidate, name)))
        setattr(candidate, name, value)
    try:
        yield
    finally:
        for candidate, previous in changed:
            setattr(candidate, name, previous)


def _routing_support_organizer(
    organizer: dict[str, Any],
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Apply an evaluation-only query-routing mask without changing bank tensors."""

    output = _clone_organizer(organizer)
    hyper_state = output.get("hyper_state")
    if not torch.is_tensor(hyper_state) or hyper_state.ndim != 3:
        raise ValueError("Prepared organizer does not expose hyper_state [B,K,H].")
    target = mask.to(device=hyper_state.device, dtype=hyper_state.dtype)
    if target.shape != hyper_state.shape[:2]:
        raise ValueError("Routing mask must have shape [B,K] matching hyper_state.")
    if bool((target.sum(dim=-1) <= 0).any()):
        raise ValueError("Routing mask must retain at least one edge per case.")
    for key in (
        "edge_active_mask",
        "effective_edge_mask",
        "edge_transition_gate",
        "hard_case_edge_mask",
        "hard_selected_edge_mask",
        "edge_survival_weight",
        "edge_survival_soft",
    ):
        current = output.get(key)
        if torch.is_tensor(current) and current.shape == target.shape:
            output[key] = target.to(dtype=current.dtype)
    output["case_adaptive_edge_count"] = target.sum(dim=-1)
    output["case_adaptive_soft_edge_count"] = target.sum(dim=-1)
    return output


def _forensic_support_ablations(
    model: Any,
    sample: dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    hard_prediction: np.ndarray,
    device: torch.device,
    *,
    query_batch_size: int,
) -> dict[str, Any]:
    """Compare stored hard support with forced full-bank and H0-only predictions."""

    with _temporary_core_setting(model, "residual_stop_fraction", -1.0):
        full_prediction = predict_case(
            model,
            sample,
            device,
            query_batch_size=int(query_batch_size),
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.5,
            return_routing_maps=False,
            return_topology_signature=False,
            return_prepared_state=True,
            return_interaction_tensor=False,
        )
    full_field = np.asarray(full_prediction["pred_field_grid"], dtype=np.float32)
    prepared = full_prediction.get("_prepared_state")
    if prepared is None:
        raise RuntimeError("Forced-full forensic decode did not retain prepared state.")
    hyper_state = prepared.organizer.get("hyper_state")
    if not torch.is_tensor(hyper_state) or hyper_state.ndim != 3:
        raise RuntimeError("Forced-full prepared state lacks hyper_state [B,K,H].")
    h0_mask = torch.zeros(hyper_state.shape[:2], device=hyper_state.device, dtype=hyper_state.dtype)
    h0_mask[:, 0] = 1.0
    x_grid = np.asarray(sample["x_grid"])
    y_grid = np.asarray(sample["y_grid"])
    query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
    h0_field = _decode_prepared_grid(
        model,
        prepared,
        query_xy,
        device,
        query_batch_size=int(query_batch_size),
        organizer=_routing_support_organizer(prepared.organizer, h0_mask),
        grid_shape=tuple(x_grid.shape),
    )
    target = _target_field(sample, dataset)[..., : hard_prediction.shape[-1]]
    result: dict[str, Any] = {
        "forensic_forced_full_edge_count": int(hyper_state.shape[1]),
        "forensic_forced_full_field_mse": error_metrics(full_field, target)["mse"],
        "forensic_h0_only_field_mse": error_metrics(h0_field, target)["mse"],
        "forensic_forced_full_vs_hard_discrepancy": _field_discrepancy(full_field, hard_prediction),
        "forensic_h0_only_vs_hard_discrepancy": _field_discrepancy(h0_field, hard_prediction),
    }
    for channel_index, channel in enumerate(dataset.channel_order[: hard_prediction.shape[-1]]):
        result[f"forensic_forced_full_{channel}_mse"] = error_metrics(
            full_field[..., channel_index], target[..., channel_index]
        )["mse"]
        result[f"forensic_h0_only_{channel}_mse"] = error_metrics(
            h0_field[..., channel_index], target[..., channel_index]
        )["mse"]
    return result


def _gradient_group_norms(model: Any) -> dict[str, float]:
    """Return pre-clip L2 gradient norms for disjoint model ownership groups."""

    squared = {"total": 0.0, "organizer": 0.0, "tensor_organizer": 0.0, "decoder": 0.0, "other": 0.0}
    maximum = 0.0
    finite = True
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach()
        finite = finite and bool(torch.isfinite(detached).all())
        # Accumulate in float64: the forensic purpose is to expose extreme
        # but finite pre-clip gradients, not to turn a float32 square overflow
        # into an apparent missing/NaN norm in the JSON report.
        detached64 = detached.double()
        value = float(torch.sum(detached64 * detached64).cpu())
        squared["total"] += value
        maximum = max(maximum, float(detached64.abs().max().cpu()))
        if name.startswith("core.organizer.case_adaptive_tensor_residual"):
            squared["tensor_organizer"] += value
            squared["organizer"] += value
        elif name.startswith("core.organizer"):
            squared["organizer"] += value
        elif name.startswith("core.decoder"):
            squared["decoder"] += value
        else:
            squared["other"] += value
    result = {f"{name}_grad_norm": float(np.sqrt(value)) for name, value in squared.items()}
    result["gradient_max_abs"] = maximum
    result["gradients_finite"] = float(finite)
    total_square = max(squared["total"], EPS)
    for name in ("organizer", "tensor_organizer", "decoder", "other"):
        result[f"{name}_gradient_energy_fraction"] = float(squared[name] / total_square)
    return result


def forensic_gradient_audit(
    model: Any,
    checkpoint: dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    index: int,
    device: torch.device,
    *,
    query_count: int,
) -> dict[str, Any]:
    """Run one deterministic training-loss backward pass without an optimizer step."""

    sample = dataset[index]
    x_grid = np.asarray(sample["x_grid"])
    y_grid = np.asarray(sample["y_grid"])
    all_queries = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
    all_targets = _target_field(sample, dataset).reshape(-1, int(model.config.field_dim))
    count = min(max(int(query_count), 1), int(all_queries.shape[0]))
    selected = np.linspace(0, all_queries.shape[0] - 1, num=count, dtype=np.int64)
    batch = make_batch(sample, all_queries[selected], device)
    target = torch.from_numpy(all_targets[selected].astype(np.float32)).unsqueeze(0).to(device)
    loss_cfg = dict(checkpoint.get("train_config", {}).get("loss", {}))
    training_cfg = dict(checkpoint.get("train_config", {}).get("training", {}))
    epoch = int(checkpoint.get("epoch", checkpoint.get("current_epoch", 0)))
    port_mode, mixed_ratio = effective_port_condition_settings(epoch, training_cfg)
    internal_weight, interface_weight = effective_local_loss_weights(loss_cfg, port_mode, mixed_ratio)
    predicted_weight = predicted_consistency_weight_for_epoch(epoch, loss_cfg)
    port_global_weight = effective_port_global_weight(loss_cfg, port_mode, mixed_ratio)
    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(
            batch["structure"],
            batch["query_xy"],
            interface_condition=batch.get("interface_condition"),
            local_module_params=batch.get("local_module_params"),
            teacher_port_tokens=batch.get("teacher_port_tokens"),
            local_query_points=batch.get("module_internal_query_points"),
            local_port_condition_mode=port_mode,
            mixed_teacher_ratio=float(mixed_ratio),
            return_predicted_port_outputs=bool(predicted_weight > 0.0),
            return_port_global_consistency=bool(port_global_weight != 0.0),
            return_organizer_diagnostics=True,
        )
        field = channelthermal_field_mse(
            output["pred_field"],
            target,
            loss_cfg,
            field_names=model.config.channelthermal.field_names,
            point_weights=None,
        )
        zero = output["pred_field"].new_zeros(())
        loss_internal = internal_loss(output, batch) if internal_weight != 0.0 else zero
        loss_interface = interface_loss(output, batch, loss_cfg) if interface_weight != 0.0 else zero
        port_weight = float(loss_cfg.get("port_supervised_weight", loss_cfg.get("port_condition_weight", 0.0)))
        smooth_weight = float(loss_cfg.get("port_smoothness_weight", 0.0))
        loss_port = port_condition_loss(output, batch, loss_cfg) if port_weight != 0.0 else zero
        loss_smooth = port_cyclic_smoothness_loss(output, batch) if smooth_weight != 0.0 else zero
        loss_port_global = port_global_consistency_loss(output) if port_global_weight != 0.0 else zero
        if "predicted_port_internal_temperature" in output and "predicted_port_interface" in output:
            predicted_internal = internal_loss(
                {"pred_internal_temperature": output["predicted_port_internal_temperature"], "pred_field": output["pred_field"]},
                batch,
            )
            predicted_interface = interface_loss(
                {"pred_interface": output["predicted_port_interface"], "pred_field": output["pred_field"]},
                batch,
                loss_cfg,
            )
            loss_predicted = predicted_internal + predicted_interface
        else:
            loss_predicted = zero
        loss_org = organizer_regularization(output, loss_cfg)
        total = (
            float(loss_cfg.get("field_mse_weight", 1.0)) * field
            + float(internal_weight) * loss_internal
            + float(interface_weight) * loss_interface
            + port_weight * loss_port
            + smooth_weight * loss_smooth
            + float(port_global_weight) * loss_port_global
            + float(predicted_weight) * loss_predicted
            + loss_org
        )
        total.backward()
        result: dict[str, Any] = {
            "case_id": str(sample["case_id"]),
            "query_count": count,
            "epoch": epoch,
            "loss_total": float(total.detach().cpu()),
            "loss_field": float(field.detach().cpu()),
            "loss_internal": float(loss_internal.detach().cpu()),
            "loss_interface": float(loss_interface.detach().cpu()),
            "loss_port": float(loss_port.detach().cpu()),
            "loss_port_smoothness": float(loss_smooth.detach().cpu()),
            "loss_port_global": float(loss_port_global.detach().cpu()),
            "loss_predicted_consistency": float(loss_predicted.detach().cpu()),
            "loss_organizer": float(loss_org.detach().cpu()),
            "gradient_clip_norm": float(training_cfg.get("gradient_clip_norm", 0.0) or 0.0),
        }
        result.update(_gradient_group_norms(model))
        clip = result["gradient_clip_norm"]
        result["implied_clip_scale"] = float(
            min(1.0, clip / max(result["total_grad_norm"], EPS)) if clip > 0.0 else 1.0
        )
        return result
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)


def _field_discrepancy(candidate: np.ndarray, baseline: np.ndarray) -> float | None:
    """Return normalized RMS field sensitivity for an evaluation-only ablation."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if candidate_array.shape != baseline_array.shape or candidate_array.size == 0:
        return None
    difference = candidate_array - baseline_array
    return float(np.sqrt(np.mean(difference * difference)) / max(np.sqrt(np.mean(baseline_array * baseline_array)), EPS))


def _hard_train_eval_consistency(
    model: Any,
    sample: dict[str, Any],
    baseline: dict[str, Any],
    baseline_aux: dict[str, Any],
    device: torch.device,
    *,
    query_batch_size: int,
) -> dict[str, Any]:
    """Check Phase-2 hard forward equality under train/eval module state."""

    was_training = bool(model.training)
    try:
        model.train()
        with torch.no_grad():
            training_prediction = predict_case(
                model,
                sample,
                device,
                query_batch_size=int(query_batch_size),
                local_port_condition_mode="predicted",
                mixed_teacher_ratio=0.5,
                return_routing_maps=False,
                return_topology_signature=False,
            )
    finally:
        model.train(was_training)
    result = {
        "hard_train_eval_field_discrepancy": _field_discrepancy(
            np.asarray(training_prediction["pred_field_grid"]),
            np.asarray(baseline["pred_field_grid"]),
        ),
        "hard_train_eval_support_gap": None,
        "hard_train_eval_support_exact": None,
    }
    training_aux = training_prediction.get("organizer_aux", {})
    base_hard = baseline_aux.get(
        "hard_case_edge_mask",
        baseline_aux.get("hard_selected_edge_mask", baseline_aux.get("edge_active_mask")),
    )
    train_hard = training_aux.get(
        "hard_case_edge_mask",
        training_aux.get("hard_selected_edge_mask", training_aux.get("edge_active_mask")),
    )
    if base_hard is not None and train_hard is not None:
        base_array = np.asarray(base_hard, dtype=np.float64)
        train_array = np.asarray(train_hard, dtype=np.float64)
        if base_array.shape == train_array.shape:
            gap = np.abs(base_array - train_array)
            result["hard_train_eval_support_gap"] = float(np.max(gap)) if gap.size else 0.0
            result["hard_train_eval_support_exact"] = bool(np.all(gap <= 1.0e-6))
    return result


def _stability_row(
    model: Any,
    sample: dict[str, Any],
    baseline: dict[str, Any],
    device: torch.device,
    *,
    query_batch_size: int,
    repeat: int,
    case_edge_selection_mode: str = "none",
    probe_relative_rms_tolerance: float = 0.005,
    probe_channel_tolerance: float = 0.01,
) -> dict[str, float]:
    """Compare one deterministic permutation/chunk repeat with its baseline."""

    count = int(np.asarray(sample["structure"]["module_present"] > 0.5).sum())
    if count > 1:
        width = int(np.asarray(sample["structure"]["module_present"]).shape[0])
        permutation = np.concatenate(
            [
                np.roll(np.arange(count, dtype=np.int64), (repeat + 1) % count),
                np.arange(count, width, dtype=np.int64),
            ]
        )
        perturbed = _permuted_sample(sample, permutation)
    else:
        perturbed = sample
    alternate_batch = max(1, int(query_batch_size // (repeat + 2)))
    prediction = predict_case(
        model,
        perturbed,
        device,
        query_batch_size=alternate_batch,
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.5,
        return_routing_maps=False,
        return_topology_signature=False,
        case_edge_selection_mode=case_edge_selection_mode,
        case_edge_probe_relative_rms_tolerance=probe_relative_rms_tolerance,
        case_edge_probe_channel_tolerance=probe_channel_tolerance,
    )
    field_delta = np.asarray(prediction["pred_field_grid"], dtype=np.float64) - np.asarray(
        baseline["pred_field_grid"], dtype=np.float64
    )
    base_aux = baseline["organizer_aux"]
    test_aux = prediction["organizer_aux"]
    base_count = _as_float(
        base_aux.get("predictive_edge_count", base_aux.get("case_adaptive_edge_count")),
        _as_float(base_aux.get("active_edge_count")),
    )
    test_count = _as_float(
        test_aux.get("predictive_edge_count", test_aux.get("case_adaptive_edge_count")),
        _as_float(test_aux.get("active_edge_count")),
    )
    base_trace = np.asarray(base_aux.get("residual_fraction_trace", []), dtype=np.float64).reshape(-1)
    test_trace = np.asarray(test_aux.get("residual_fraction_trace", []), dtype=np.float64).reshape(-1)
    trace_delta = float(np.max(np.abs(base_trace - test_trace))) if base_trace.shape == test_trace.shape and base_trace.size else float("nan")
    return {
        "stability_field_max_abs": float(np.max(np.abs(field_delta))) if field_delta.size else 0.0,
        "stability_field_rmse": float(np.sqrt(np.mean(field_delta * field_delta))) if field_delta.size else 0.0,
        "stability_count_match": float(base_count == test_count),
        "stability_count_delta": float(test_count - base_count),
        "stability_trace_max_abs": trace_delta,
    }


def evaluate_case(
    label: str,
    model: Any,
    checkpoint: dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    index: int,
    device: torch.device,
    *,
    query_batch_size: int,
    return_routing_maps: bool,
    benchmark: bool,
    stability_perturbations: bool,
    stability_repeats: int,
    tensor_diagnostics: bool = False,
    forensic_audit: bool = False,
    case_edge_selection_mode: str = "none",
    probe_relative_rms_tolerance: float = 0.005,
    probe_channel_tolerance: float = 0.01,
    benchmark_query_counts: Sequence[int] = (),
) -> dict[str, Any]:
    """Evaluate and summarize one case without changing model state."""

    sample = dataset[index]
    start_time = time.perf_counter()
    if benchmark and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prediction = predict_case(
        model,
        sample,
        device,
        query_batch_size=int(query_batch_size),
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.5,
        return_routing_maps=bool(return_routing_maps),
        return_topology_signature=False,
        return_prepared_state=True,
        return_interaction_tensor=bool(tensor_diagnostics),
        case_edge_selection_mode=case_edge_selection_mode,
        case_edge_probe_relative_rms_tolerance=probe_relative_rms_tolerance,
        case_edge_probe_channel_tolerance=probe_channel_tolerance,
    )
    elapsed = time.perf_counter() - start_time
    pred = np.asarray(prediction["pred_field_grid"], dtype=np.float32)
    target = _target_field(sample, dataset)[..., : pred.shape[-1]]
    _, fluid_mask = module_and_fluid_masks(sample, pred)
    near_mask = _near_interface_mask(sample, fluid_mask)
    far_mask = _far_field_mask(sample, fluid_mask)
    aux = dict(prediction["organizer_aux"])
    prepared_for_ablation = prediction.get("_prepared_state")
    tensor_request_note = ""
    if tensor_diagnostics and "residual_interaction_tensor" not in aux:
        try:
            _, requested_prepared = _prepare_case(
                model,
                sample,
                device,
                query_batch_size=int(query_batch_size),
                tensor_diagnostics=True,
            )
            prepared_for_ablation = requested_prepared
            for key, value in requested_prepared.organizer.items():
                # The retained organizer is the authoritative source for
                # optional Phase-2 tensors and support fields.  Do not replace
                # wrapper-derived legacy values when they already exist.
                aux.setdefault(key, value)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            tensor_request_note = f" Interaction tensor request unavailable: {type(exc).__name__}: {exc}."
    arrays = extract_organization_arrays(sample, aux)
    if tensor_diagnostics and not np.asarray(
        arrays.get("residual_interaction_tensor", np.zeros((0, 0, 0)))
    ).size and not tensor_request_note:
        tensor_request_note = " Interaction tensor request was accepted but no tensor was exposed by this checkpoint/wrapper."
    predictive_mode = str(case_edge_selection_mode) == "probe_fidelity"
    count = _as_float(
        aux.get("predictive_edge_count", aux.get("case_adaptive_edge_count")),
        float(np.sum(arrays["active_hyperedge_mask"] > 0.5)),
    )
    cap = _as_float(
        aux.get("case_adaptive_edge_cap"),
        float(np.asarray(arrays.get("strength", [])).reshape(-1).size),
    )
    hard_mask = np.asarray(arrays["active_hyperedge_mask"], dtype=bool).reshape(-1)
    initial_coupling = np.asarray(arrays.get("A_me", np.zeros((0, 0))), dtype=np.float64)
    row_mass = np.asarray(arrays.get("residual_coupling_row_mass", []), dtype=np.float64).reshape(-1)
    if initial_coupling.ndim == 2 and initial_coupling.shape[0] == row_mass.size:
        initial_coupling = initial_coupling * row_mass[:, None]
    else:
        initial_coupling = np.zeros((0, 0), dtype=np.float64)
    routing_maps = prediction.get("routing_maps", {})
    alpha = np.asarray(routing_maps.get("query_hyper_attention", np.zeros((0, 0))), dtype=np.float64)
    pairwise = np.asarray(routing_maps.get("pairwise_edge_contribution", np.zeros((0, 0))), dtype=np.float64)
    configured_mode = str(getattr(model.config.core_honf, "organizer_mode", ""))
    phase2_mode = configured_mode == "case_adaptive_tensor_residual" or any(
        key in aux for key in ("residual_interaction_tensor", "residual_content_factor", "edge_survival_soft")
    )
    phase2_metrics = (
        _phase2_case_metrics(sample, arrays, aux, hard_mask, count)
        if phase2_mode
        else {}
    )
    full_bank_field: np.ndarray | None = None
    if predictive_mode:
        if prepared_for_ablation is None:
            raise RuntimeError("probe_fidelity evaluation requires retained prepared state.")
        full_organizer = dict(prepared_for_ablation.organizer)
        full_organizer.pop("predictive_edge_mask", None)
        query_xy_full = np.stack(
            [
                np.asarray(sample["x_grid"]).reshape(-1),
                np.asarray(sample["y_grid"]).reshape(-1),
            ],
            axis=-1,
        ).astype(np.float32)
        full_bank_field = _decode_prepared_grid(
            model,
            prepared_for_ablation,
            query_xy_full,
            device,
            query_batch_size=int(query_batch_size),
            organizer=full_organizer,
            grid_shape=tuple(np.asarray(sample["x_grid"]).shape),
        )
    row: dict[str, Any] = {
        "checkpoint": label,
        "case_id": str(sample["case_id"]),
        "module_count": int(np.sum(arrays["present"] > 0.5)),
        "case_adaptive_edge_count": count,
        "case_adaptive_edge_cap": cap,
        "case_adaptive_soft_edge_count": _as_float(
            aux.get("case_adaptive_soft_edge_count", aux.get("soft_edge_count")),
            float(np.sum(arrays["edge_survival_weight"])),
        ),
        "case_adaptive_stop_reached": _as_float(aux.get("case_adaptive_stop_reached")),
        "case_adaptive_cap_hit": _as_float(aux.get("case_adaptive_cap_hit")),
        "case_adaptive_stop_margin": _as_float(aux.get("case_adaptive_stop_margin")),
        "residual_fraction_final": float(
            arrays["residual_fraction_trace"][min(max(round(count), 0), arrays["residual_fraction_trace"].size - 1)]
        )
        if arrays["residual_fraction_trace"].size
        else 0.0,
        "residual_coupling_effective_rank": effective_rank(initial_coupling),
        "residual_trace": json.dumps(_jsonable(arrays["residual_fraction_trace"])),
        "residual_marginal_explained_fraction": json.dumps(_jsonable(arrays["residual_marginal_explained_fraction"])),
        "residual_mechanism_strength": json.dumps(_jsonable(arrays["residual_mechanism_strength"])),
        "residual_content_factor": json.dumps(_jsonable(arrays["residual_content_factor"])),
        "hard_active_mechanism_indices": json.dumps(np.flatnonzero(hard_mask).astype(int).tolist()),
        "field_mse": error_metrics(pred, target)["mse"],
        "fluid_mse": masked_error_metrics(pred, target, fluid_mask)["mse"],
        "near_interface_mse": masked_error_metrics(pred, target, near_mask)["mse"],
        "far_field_mse": masked_error_metrics(pred, target, far_mask)["mse"],
        "field_relative_l2": error_metrics(pred, target)["relative_l2"],
        "case_edge_selection_mode": str(case_edge_selection_mode),
        "predictive_edge_count": _as_float(aux.get("predictive_edge_count"), count)
        if predictive_mode
        else None,
        "predictive_edge_cap": cap if predictive_mode else None,
        "predictive_probe_relative_rms": _as_float(
            aux.get("predictive_probe_relative_rms")
        )
        if predictive_mode
        else None,
        "predictive_probe_channel_relative_rms_max": _as_max_float(
            aux.get("predictive_probe_channel_relative_rms")
        )
        if predictive_mode
        else None,
        "predictive_probe_count": _as_float(aux.get("predictive_probe_count"))
        if predictive_mode
        else None,
        "predictive_candidate_count": _as_float(aux.get("predictive_candidate_count"))
        if predictive_mode
        else None,
        "predictive_selection_seconds": _as_float(aux.get("predictive_selection_seconds"))
        if predictive_mode
        else None,
        "predictive_full_grid_discrepancy": _field_discrepancy(full_bank_field, pred)
        if full_bank_field is not None
        else None,
        "predictive_full_bank_field_mse": error_metrics(full_bank_field, target)["mse"]
        if full_bank_field is not None
        else None,
        "soft_support_vs_hard_field_discrepancy": None,
        "organizer_ablation_uniform_incidence_discrepancy": None,
        "organizer_ablation_pooled_mechanism_discrepancy": None,
        "organizer_ablation_no_hyper_value_discrepancy": None,
        "tensor_diagnostics_requested": bool(tensor_diagnostics),
        "interaction_tensor_available": bool(
            np.asarray(arrays.get("residual_interaction_tensor", np.zeros((0, 0, 0)))).size
        ),
        "interaction_tensor_shape": None,
        "hard_train_eval_field_discrepancy": None,
        "hard_train_eval_support_gap": None,
        "hard_train_eval_support_exact": None,
        "query_effective_rank": effective_rank(alpha.reshape(-1, alpha.shape[-1])) if alpha.ndim == 3 and alpha.size else 0.0,
        "pairwise_effective_rank": effective_rank(pairwise.reshape(-1, pairwise.shape[-1])) if pairwise.ndim == 3 and pairwise.size else 0.0,
        "soft_hard_ablation_note": "",
        "forensic_audit_requested": bool(forensic_audit),
    }
    if phase2_metrics:
        row.update(phase2_metrics)
        shape = phase2_metrics.get("interaction_tensor_shape")
        if shape is not None:
            row["interaction_tensor_shape"] = json.dumps(shape)
    if predictive_mode:
        module_count = max(int(np.sum(arrays["present"] > 0.5)), 1)
        row["case_adaptive_k_over_module_count"] = float(count / module_count)
        row["case_adaptive_k_over_cap"] = float(count / max(cap, 1.0))
    is_residual = configured_mode in {"case_adaptive_residual", "case_adaptive_tensor_residual"} or "residual_fraction_trace" in aux
    if is_residual:
        diagnostic_note = "Evaluation-only organizer-reliance ablations were decoded from one prepared case."
        if tensor_request_note:
            diagnostic_note += tensor_request_note
        try:
            prepared = prepared_for_ablation
            if prepared is None:
                query_xy, prepared = _prepare_case(
                    model,
                    sample,
                    device,
                    query_batch_size=int(query_batch_size),
                    tensor_diagnostics=bool(tensor_diagnostics and phase2_mode),
                )
            else:
                x_grid = np.asarray(sample["x_grid"])
                y_grid = np.asarray(sample["y_grid"])
                query_xy = np.stack(
                    [x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1
                ).astype(np.float32)
            grid_shape = tuple(np.asarray(sample["x_grid"]).shape)
            if not phase2_mode:
                soft_organizer = _soft_support_organizer(model, prepared.organizer)
                if soft_organizer is not None:
                    soft_field = _decode_prepared_grid(
                        model,
                        prepared,
                        query_xy,
                        device,
                        query_batch_size=int(query_batch_size),
                        organizer=soft_organizer,
                        grid_shape=grid_shape,
                    )
                    row["soft_support_vs_hard_field_discrepancy"] = _field_discrepancy(soft_field, pred)
                else:
                    diagnostic_note += " Soft-support override unavailable for this checkpoint."
            else:
                # Phase 2's soft support is a backward-only surrogate.  Its
                # forward incidences and query attention are hard-supported;
                # compare train/eval hard outputs instead of decoding a soft
                # alternate function.
                row.update(
                    _hard_train_eval_consistency(
                        model,
                        sample,
                        prediction,
                        aux,
                        device,
                        query_batch_size=int(query_batch_size),
                    )
                )
            uniform_organizer = _uniform_incidence_organizer(prepared.organizer)
            if uniform_organizer is not None:
                uniform_field = _decode_prepared_grid(
                    model,
                    prepared,
                    query_xy,
                    device,
                    query_batch_size=int(query_batch_size),
                    organizer=uniform_organizer,
                    grid_shape=grid_shape,
                )
                row["organizer_ablation_uniform_incidence_discrepancy"] = _field_discrepancy(uniform_field, pred)
            pooled_organizer = _pooled_mechanism_organizer(prepared.organizer)
            if pooled_organizer is not None:
                pooled_field = _decode_prepared_grid(
                    model,
                    prepared,
                    query_xy,
                    device,
                    query_batch_size=int(query_batch_size),
                    organizer=pooled_organizer,
                    grid_shape=grid_shape,
                )
                row["organizer_ablation_pooled_mechanism_discrepancy"] = _field_discrepancy(pooled_field, pred)
            no_hyper_field = _decode_prepared_grid(
                model,
                prepared,
                query_xy,
                device,
                query_batch_size=int(query_batch_size),
                disable_hyper_value=True,
                grid_shape=grid_shape,
            )
            row["organizer_ablation_no_hyper_value_discrepancy"] = _field_discrepancy(no_hyper_field, pred)
            if forensic_audit and phase2_mode:
                row.update(
                    _forensic_support_ablations(
                        model,
                        sample,
                        dataset,
                        pred,
                        device,
                        query_batch_size=int(query_batch_size),
                    )
                )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - checkpoint/device-specific fallback
            diagnostic_note = (
                "Evaluation-only ablation diagnostics unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        row["soft_hard_ablation_note"] = diagnostic_note
    else:
        row["soft_hard_ablation_note"] = "Skipped: checkpoint is not an adaptive residual organizer."
    for channel_index, channel in enumerate(dataset.channel_order[: pred.shape[-1]]):
        pred_channel = pred[..., channel_index]
        target_channel = target[..., channel_index]
        row[f"{channel}_mse"] = masked_error_metrics(
            pred_channel, target_channel, fluid_mask
        )["mse"]
        row[f"whole_{channel}_mse"] = error_metrics(pred_channel, target_channel)["mse"]
        row[f"near_interface_{channel}_mse"] = masked_error_metrics(
            pred_channel, target_channel, near_mask
        )["mse"]
        row[f"far_field_{channel}_mse"] = masked_error_metrics(
            pred_channel, target_channel, far_mask
        )["mse"]
        if full_bank_field is not None:
            row[f"predictive_full_grid_{channel}_discrepancy"] = _field_discrepancy(
                full_bank_field[..., channel_index], pred_channel
            )
    if benchmark:
        row["forward_seconds"] = float(elapsed)
        row["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        row["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        if prepared_for_ablation is not None:
            row.update(
                _benchmark_prepared_queries(
                    model,
                    prepared_for_ablation,
                    sample,
                    device,
                    query_batch_size=int(query_batch_size),
                    query_counts=benchmark_query_counts,
                    selection_seconds=float(row.get("predictive_selection_seconds") or 0.0),
                )
            )
    if stability_perturbations:
        for repeat in range(max(int(stability_repeats), 1)):
            stability = _stability_row(
                model,
                sample,
                prediction,
                device,
                query_batch_size=int(query_batch_size),
                repeat=repeat,
                case_edge_selection_mode=case_edge_selection_mode,
                probe_relative_rms_tolerance=probe_relative_rms_tolerance,
                probe_channel_tolerance=probe_channel_tolerance,
            )
            for key, value in stability.items():
                row[f"{key}_repeat_{repeat + 1}"] = value
    return row


def _selected_indices(dataset: GlobalChannelThermalDataset, case_ids: Sequence[str], max_cases: int | None) -> list[int]:
    """Resolve deterministic case selection in dataset order."""

    requested = {str(case_id) for case_id in case_ids}
    indices = [idx for idx, case_id in enumerate(dataset.selected_case_ids) if not requested or str(case_id) in requested]
    if max_cases is not None:
        indices = indices[: max(int(max_cases), 0)]
    missing = requested - {str(dataset.selected_case_ids[idx]) for idx in indices}
    if missing:
        raise KeyError(f"Requested case IDs are absent from split {dataset.split!r}: {sorted(missing)}")
    return indices


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with a deterministic union of field names."""

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], *, include_by_checkpoint: bool = True) -> dict[str, Any]:
    """Build count/residual/accuracy summaries from per-case rows."""

    summary: dict[str, Any] = {"case_count": len(rows)}
    if not rows:
        return summary
    numeric_keys = [
        "module_count",
        "case_adaptive_edge_count",
        "case_adaptive_edge_cap",
        "case_adaptive_soft_edge_count",
        "case_adaptive_stop_reached",
        "case_adaptive_cap_hit",
        "case_adaptive_stop_margin",
        "residual_fraction_final",
        "residual_coupling_effective_rank",
        "field_mse",
        "fluid_mse",
        "near_interface_mse",
        "far_field_mse",
        "field_relative_l2",
        "soft_support_vs_hard_field_discrepancy",
        "organizer_ablation_uniform_incidence_discrepancy",
        "organizer_ablation_pooled_mechanism_discrepancy",
        "organizer_ablation_no_hyper_value_discrepancy",
        "hard_train_eval_field_discrepancy",
        "hard_train_eval_support_gap",
        "case_adaptive_k_over_module_count",
        "case_adaptive_k_over_cap",
        "interaction_tensor_module_effective_rank",
        "interaction_tensor_environment_effective_rank",
        "interaction_tensor_content_effective_rank",
        "interaction_tensor_inactive_module_max",
        "interaction_tensor_min",
        "interaction_tensor_max",
        "interaction_tensor_total_mass",
        "interaction_tensor_max_channel_mass_fraction",
        "interaction_tensor_channel_effective_count",
        "first_component_explained_fraction",
        "soft_gate_derivative_mean",
        "soft_gate_derivative_p95",
        "soft_gate_derivative_max",
        "residual_content_factor_effective_rank",
        "residual_content_factor_nonzero_fraction",
        "residual_content_factor_column_cosine",
        "selected_module_effective_rank",
        "selected_environment_effective_rank",
        "selected_environment_edge_column_cosine",
        "selected_region_separation_normalized",
        "selected_environment_rank_over_hard_k",
        "soft_hard_support_gap_mean",
        "soft_hard_support_gap_max",
        "hard_forward_support_gap_mean",
        "hard_forward_support_gap_max",
        "hard_support_count",
        "soft_support_effective_k",
        "residual_monotonic_violation_max",
        "empty_selected_edge_count",
        "post_fallback_zero_support_module_rows",
        "post_fallback_zero_support_environment_rows",
        "forensic_forced_full_edge_count",
        "forensic_forced_full_field_mse",
        "forensic_h0_only_field_mse",
        "forensic_forced_full_vs_hard_discrepancy",
        "forensic_h0_only_vs_hard_discrepancy",
        "predictive_edge_count",
        "predictive_edge_cap",
        "predictive_probe_relative_rms",
        "predictive_probe_channel_relative_rms_max",
        "predictive_probe_count",
        "predictive_candidate_count",
        "predictive_selection_seconds",
        "predictive_full_grid_discrepancy",
        "predictive_full_bank_field_mse",
    ]
    numeric_keys.extend(
        key
        for key in sorted({key for row in rows for key in row})
        if key.startswith("benchmark_") and key not in numeric_keys
    )
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))], dtype=np.float64)
        if values.size:
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_p95"] = float(np.quantile(values, 0.95))
            summary[f"{key}_max"] = float(values.max())
    counts = np.asarray([round(float(row["case_adaptive_edge_count"])) for row in rows], dtype=np.int64)
    summary["case_adaptive_count_histogram"] = {str(int(value)): int(np.sum(counts == value)) for value in sorted(set(counts.tolist()))}
    if any(row.get("predictive_edge_count") is not None for row in rows):
        summary["predictive_k_histogram"] = dict(summary["case_adaptive_count_histogram"])
    summary["interaction_tensor_available_count"] = int(
        sum(bool(row.get("interaction_tensor_available")) for row in rows)
    )
    for key, output_key in (
        ("hard_forward_support_exact", "hard_forward_support_exact_fraction"),
        ("hard_train_eval_support_exact", "hard_train_eval_support_exact_fraction"),
        ("interaction_tensor_finite", "interaction_tensor_finite_fraction"),
        ("interaction_tensor_nonnegative", "interaction_tensor_nonnegative_fraction"),
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            summary[output_key] = float(np.mean(values))
    k_by_module: dict[str, dict[str, Any]] = {}
    for module_count in sorted({int(row["module_count"]) for row in rows}):
        members = [row for row in rows if int(row["module_count"]) == module_count]
        member_k = np.asarray([float(row["case_adaptive_edge_count"]) for row in members], dtype=np.float64)
        member_caps = np.asarray([float(row["case_adaptive_edge_cap"]) for row in members], dtype=np.float64)
        member_ratios = np.asarray(
            [
                float(row["case_adaptive_k_over_module_count"])
                for row in members
                if row.get("case_adaptive_k_over_module_count") is not None
            ],
            dtype=np.float64,
        )
        member_cap_ratios = np.asarray(
            [
                float(row["case_adaptive_k_over_cap"])
                for row in members
                if row.get("case_adaptive_k_over_cap") is not None
            ],
            dtype=np.float64,
        )
        k_by_module[str(module_count)] = {
            "case_count": len(members),
            "hard_k_histogram": {
                str(int(value)): int(np.sum(member_k == value))
                for value in sorted(set(member_k.astype(np.int64).tolist()))
            },
            "hard_k_mean": float(np.mean(member_k)) if member_k.size else 0.0,
            "hard_k_median": float(np.median(member_k)) if member_k.size else 0.0,
            "hard_k_min": float(np.min(member_k)) if member_k.size else 0.0,
            "hard_k_max": float(np.max(member_k)) if member_k.size else 0.0,
            "cap_mean": float(np.mean(member_caps)) if member_caps.size else 0.0,
            "k_over_module_count_mean": float(np.mean(member_ratios)) if member_ratios.size else 0.0,
            "k_over_cap_mean": float(np.mean(member_cap_ratios)) if member_cap_ratios.size else 0.0,
        }
    summary["case_adaptive_k_by_module_count"] = k_by_module
    # Keep the short spelling convenient for downstream report consumers while
    # retaining the canonical case-adaptive name above.
    summary["k_by_module_count"] = k_by_module
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_checkpoint.setdefault(str(row["checkpoint"]), []).append(row)
    # A grouped summary must not recurse into another same-label grouped
    # summary. Keep one top-level by-checkpoint view while each member holds
    # the ordinary numeric/count aggregates.
    if include_by_checkpoint:
        summary["by_checkpoint"] = {
            label: summarize_rows(members, include_by_checkpoint=False)
            for label, members in by_checkpoint.items()
        }
    return summary


def _checkpoint_dataset(checkpoint: dict[str, Any], split: str) -> GlobalChannelThermalDataset:
    """Construct the checkpoint-owned normalized global dataset."""

    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    dataset_path = dataset_config.get("packed_h5_path")
    if not dataset_path:
        raise ValueError("Checkpoint does not contain train_config.dataset.packed_h5_path")
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    normalizer = H5Normalizer(stats) if stats else None
    return GlobalChannelThermalDataset(
        dataset_path,
        split=split,
        points_per_case=1,
        normalize_inputs=bool(dataset_config.get("normalize_inputs", False)),
        normalize_targets=bool(dataset_config.get("normalize_targets", False)),
        random_point_sampling=False,
        include_grid=True,
        include_structure_targets=False,
        normalizer=normalizer,
    )


def run_checkpoint(label: str, path: Path, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one checkpoint and return provenance plus rows."""

    model, checkpoint = load_model(path, device)
    dataset = _checkpoint_dataset(checkpoint, args.split)
    indices = _selected_indices(dataset, args.case_id, args.max_cases)
    rows: list[dict[str, Any]] = []
    try:
        for position, index in enumerate(indices, start=1):
            row = evaluate_case(
                label,
                model,
                checkpoint,
                dataset,
                index,
                device,
                query_batch_size=args.query_batch_size,
                return_routing_maps=args.return_routing_maps,
                tensor_diagnostics=args.tensor_diagnostics,
                benchmark=args.benchmark,
                stability_perturbations=args.stability_perturbations,
                stability_repeats=args.stability_repeats,
                forensic_audit=args.forensic_audit,
                case_edge_selection_mode=args.case_edge_selection_mode,
                probe_relative_rms_tolerance=args.probe_relative_rms_tolerance,
                probe_channel_tolerance=args.probe_channel_tolerance,
                benchmark_query_counts=args.benchmark_query_count,
            )
            rows.append(row)
            print(f"[{label}] {position}/{len(indices)} case={row['case_id']}", flush=True)
        gradient_audit = (
            forensic_gradient_audit(
                model,
                checkpoint,
                dataset,
                indices[0],
                device,
                query_count=int(args.gradient_audit_query_count),
            )
            if (args.forensic_audit or args.gradient_audit) and indices
            else None
        )
    finally:
        dataset.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model_config = checkpoint.get("model_config", {})
    core_config = model_config.get("core_honf", {})
    provenance = {
        "label": label,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "epoch": checkpoint.get("epoch", checkpoint.get("current_epoch", -1)),
        "split": args.split,
        "evaluated_case_count": len(rows),
        "organizer_mode": core_config.get("organizer_mode"),
        "num_hyperedges": core_config.get("num_hyperedges"),
        "residual_stop_fraction": core_config.get("residual_stop_fraction"),
        "residual_soft_stop_temperature": core_config.get("residual_soft_stop_temperature"),
        "residual_interaction_dim": core_config.get("residual_interaction_dim"),
        "residual_mechanism_cap_multiplier": core_config.get("residual_mechanism_cap_multiplier"),
        "query_batch_size": args.query_batch_size,
        "tensor_diagnostics_requested": bool(args.tensor_diagnostics),
        "forensic_audit_requested": bool(args.forensic_audit),
        "gradient_audit_requested": bool(args.gradient_audit or args.forensic_audit),
        "forensic_gradient_audit": gradient_audit,
        "case_edge_selection_mode": args.case_edge_selection_mode,
        "probe_relative_rms_tolerance": args.probe_relative_rms_tolerance,
        "probe_channel_tolerance": args.probe_channel_tolerance,
    }
    return provenance, rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluator and write canonical CSV/JSON artifacts."""

    args = parse_args(argv)
    if int(args.query_batch_size) <= 0:
        raise ValueError("--query-batch-size must be positive")
    if int(args.gradient_audit_query_count) <= 0:
        raise ValueError("--gradient-audit-query-count must be positive")
    if float(args.probe_relative_rms_tolerance) < 0.0:
        raise ValueError("--probe-relative-rms-tolerance must be nonnegative")
    if float(args.probe_channel_tolerance) < 0.0:
        raise ValueError("--probe-channel-tolerance must be nonnegative")
    if any(int(value) <= 0 for value in args.benchmark_query_count):
        raise ValueError("--benchmark-query-count values must be positive")
    if args.forensic_audit:
        args.tensor_diagnostics = True
    specs = checkpoint_specs(args.checkpoint)
    for path in specs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    first_checkpoint = next(iter(specs.values()))
    if args.output_dir is None:
        output_dir = (
            default_evaluation_root(
                first_checkpoint,
                evaluation_kind="multi_case_case_adaptive_residual",
            )
            / time.strftime("%Y%m%d_%H%M%S")
        )
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    layout = EvaluationArtifactLayout.at(output_dir)
    layout.ensure("metrics", "diagnostics")
    device = torch.device(args.device)
    all_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for label, path in specs.items():
        info, rows = run_checkpoint(label, path, args, device)
        provenance[label] = info
        all_rows.extend(rows)
    per_case_path = layout.metrics / "case_adaptive_residual_per_case.csv"
    summary_path = layout.diagnostics / "case_adaptive_residual_summary.json"
    write_csv(per_case_path, all_rows)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "script": str(Path(__file__).resolve()),
        "device": str(device),
        "split": args.split,
        "provenance": provenance,
        "summary": summarize_rows(all_rows),
        "artifacts": {
            "per_case_csv": str(per_case_path),
            "summary_json": str(summary_path),
            "evaluation_manifest": str(layout.root / "evaluation_manifest.json"),
        },
        "notes": {
            "soft_support_vs_hard_field_discrepancy": "Normalized RMS field difference from a training-style soft-support decode versus normal hard evaluation.",
            "phase2_tensor_diagnostics": "Requested only with --tensor-diagnostics/--return-interaction-tensor; reports M/E/content unfolding ranks and does not affect ordinary inference.",
            "phase2_support_consistency": "Phase-2 soft survival is a backward-only surrogate; hard train/eval field and support discrepancies are reported separately.",
            "organizer_reliance_ablations": "Normalized RMS field differences for row-uniform incidences, pooled active mechanisms, and suppressed hyper-value context; all are evaluation-only.",
            "k_by_module_count": "Hard case-adaptive K grouped by active module count; K_cap is a case-size safety budget, not a global scientific rank.",
            "runtime_memory": "included only with --benchmark",
            "forensic_audit": "Run-1701-only read-only collapse timeline, support ablations, gate-derivative statistics, and one pre-clip training-loss backward pass per checkpoint; no optimizer step is taken.",
            "probe_fidelity": "Evaluation-only exhaustive support search over the prepared fixed bank; the selected mask changes query routing only and is cached in prepared state.",
        },
    }
    summary_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    finalize_evaluation_job(
        layout.root,
        kind="multi_case_case_adaptive_residual",
        checkpoint_path=first_checkpoint,
        requested_checkpoint=",".join(args.checkpoint),
    )
    print(f"[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
