#!/usr/bin/env python3
"""Bounded Run-2100 mean-shift candidate and routed-support diagnostics.

This driver is evaluation-only.  It runs fixed query probes through the
ordinary ThermalChannel forward path and records the finite candidate motion,
the actual common-router support, and a frozen module-hub candidate
comparison.  It does not train, reserve a run, write a checkpoint, or mutate
model parameters.

The mean-shift backend exposes one deliberately small diagnostic contract on
``PreparedRoutingIndex.diagnostics`` when ``return_routing_maps=True``::

    {
        "candidate_trajectory": {
            "coords": Tensor[B, T + 1, K, d],
            "descriptors": Tensor[B, T + 1, K, D],
        }
    }

The coordinates are physical.  Row zero is the module-seeded state and the
remaining rows are the three finite updates.  The trajectory is metadata for
this report: it is never merged, rounded, or fed back into routing execution.

Run from ``HONF_Proj/`` with an explicitly labelled checkpoint::

    python tools/diagnostics/run_mean_shift_routing_diagnostics.py run \
        --checkpoint epoch0500=path/to/epoch_0500_model.pt \
        --case-id 0273 --query-count 32 \
        --output diagnostics/mean_shift_0273.json
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "tools" / "diagnostics",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "Case_ThermalChannel" / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Loading, query selection, the routing summary, and the physical ledger all
# remain owned by the maintained Goal-1 diagnostic driver.
from channelthermal.evaluation.prepared import select_sample
from run_dynamic_sparse_routing_study import (  # type: ignore[import-not-found]
    ANCHOR_CASE_IDS,
    DiagnosticUnavailable,
    _build_fresh_profile_model,
    _canonical_ground_truth_errors,
    _forward_batch,
    _ledger_row,
    _load_dataset,
    _load_model_spec,
    _load_raw_sample,
    _profile_args,
    _query_points,
    _relative_difference,
    _routing_index_summary,
    parse_checkpoint_specs,
    select_device,
    write_json,
)

DEFAULT_QUERY_COUNT = 32
DEFAULT_MODE_TOLERANCE = 0.05
DEFAULT_MAX_TRAJECTORY_VALUES = 200_000
DEFAULT_TIMING_WARMUP = 1
DEFAULT_TIMING_REPEATS = 3
DEFAULT_PROFILE = "project://src/config_core/forward/routing_mean_shift_context.json"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(value: torch.Tensor | np.ndarray | Sequence[float]) -> dict[str, Any]:
    """Return finite distribution statistics with an explicit empty contract."""

    if torch.is_tensor(value):
        array = value.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _per_batch_summary(values: torch.Tensor, mask: torch.Tensor | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_index in range(int(values.shape[0])):
        selected = values[batch_index]
        if mask is not None:
            selected = selected[mask[batch_index]]
        rows.append(_summary(selected))
    return rows


def _candidate_fields(candidate: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the typed ``RoutingCandidates`` contract used by the backend."""

    if candidate is None:
        raise DiagnosticUnavailable("prepared routing state has no candidates")
    coords = getattr(candidate, "coords", None)
    descriptors = getattr(candidate, "descriptors", None)
    valid = getattr(candidate, "valid", None)
    if not torch.is_tensor(coords) or coords.ndim != 3:
        raise DiagnosticUnavailable("RoutingCandidates.coords is not [B,K,d]")
    if not torch.is_tensor(descriptors) or descriptors.ndim != 3:
        raise DiagnosticUnavailable("RoutingCandidates.descriptors is not [B,K,D]")
    if not torch.is_tensor(valid) or valid.ndim != 2:
        raise DiagnosticUnavailable("RoutingCandidates.valid is not [B,K]")
    batch_count, candidate_count = (int(coords.shape[0]), int(coords.shape[1]))
    if tuple(descriptors.shape[:2]) != (batch_count, candidate_count):
        raise ValueError("candidate descriptors do not align with candidate coordinates")
    if tuple(valid.shape) != (batch_count, candidate_count):
        raise ValueError("candidate validity does not align with candidate coordinates")
    return coords, descriptors.to(device=coords.device, dtype=coords.dtype), valid.to(
        device=coords.device, dtype=torch.bool
    )


def _candidate_trajectory(
    state: Mapping[str, Any],
    final_candidate: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read and validate the exact prepared-state trajectory contract."""

    routing_index = state.get("routing_index")
    diagnostics = getattr(routing_index, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise DiagnosticUnavailable("prepared routing index exposes no diagnostics mapping")
    payload = diagnostics.get("candidate_trajectory")
    if not isinstance(payload, Mapping):
        raise DiagnosticUnavailable(
            "mean-shift candidate trajectory is absent; request return_routing_maps=True"
        )
    coords = payload.get("coords")
    descriptors = payload.get("descriptors")
    if not torch.is_tensor(coords) or coords.ndim != 4:
        raise DiagnosticUnavailable("candidate_trajectory.coords must be [B,T+1,K,d]")
    if not torch.is_tensor(descriptors) or descriptors.ndim != 4:
        raise DiagnosticUnavailable("candidate_trajectory.descriptors must be [B,T+1,K,D]")
    final_coords, final_descriptors, valid = _candidate_fields(final_candidate)
    expected = (int(final_coords.shape[0]), int(final_coords.shape[1]))
    if tuple(coords.shape[:1]) != (expected[0],) or int(coords.shape[2]) != expected[1]:
        raise ValueError(
            "candidate trajectory coordinates do not align with final candidates: "
            f"{tuple(coords.shape)} vs B={expected[0]}, K={expected[1]}"
        )
    if tuple(descriptors.shape[:3]) != (
        expected[0],
        int(coords.shape[1]),
        expected[1],
    ):
        raise ValueError("candidate trajectory descriptors do not align with coordinates")
    if int(coords.shape[-1]) != int(final_coords.shape[-1]):
        raise ValueError("candidate trajectory coordinate dimension differs from final candidates")
    if int(descriptors.shape[-1]) != int(final_descriptors.shape[-1]):
        raise ValueError("candidate trajectory descriptor dimension differs from final candidates")
    return (
        coords.to(device=final_coords.device, dtype=final_coords.dtype),
        descriptors.to(device=final_coords.device, dtype=final_coords.dtype),
        valid,
    )


def _routing_length_scale(backend: Any, encoded: Any, coords: torch.Tensor) -> torch.Tensor:
    """Use the adapter's routing scale ``ell``, rather than input normalization."""

    method = getattr(backend, "_length_scale", None)
    if not callable(method):
        raise DiagnosticUnavailable("routed backend exposes no routing length-scale method")
    scale = method(encoded, coords)
    if not torch.is_tensor(scale):
        scale = coords.new_tensor(scale)
    scale = scale.to(device=coords.device, dtype=coords.dtype).reshape(-1)
    if scale.numel() != int(coords.shape[-1]):
        raise ValueError(
            f"routing length scale has {scale.numel()} values; expected {coords.shape[-1]}"
        )
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
        raise ValueError("routing length scale must be finite and positive")
    return scale


def _routing_feature_bandwidth(backend: Any) -> float:
    bandwidth = float(getattr(backend, "mean_shift_feature_bandwidth", 1.0))
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("backend mean-shift feature bandwidth must be finite and positive")
    return bandwidth


def _pairwise_minimum(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return the minimum finite pair distance per ``[B,T]`` row."""

    batch_count, time_count, candidate_count, coordinate_dim = (int(item) for item in values.shape)
    result = values.new_full((batch_count, time_count), float("nan"))
    if candidate_count < 2 or candidate_count > 1024:
        return result
    finite = torch.isfinite(values).all(dim=-1)
    row_valid = valid[:, None, :].expand(batch_count, time_count, candidate_count) & finite
    flat_values = values.reshape(batch_count * time_count, candidate_count, coordinate_dim)
    distances = torch.cdist(flat_values, flat_values)
    pair_mask = torch.triu(
        torch.ones(
            (candidate_count, candidate_count),
            device=values.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    flat_valid = row_valid.reshape(batch_count * time_count, candidate_count)
    pair_valid = pair_mask[None] & flat_valid[:, :, None] & flat_valid[:, None, :]
    distances = distances.masked_fill(~pair_valid, float("inf"))
    minimum = distances.amin(dim=(-2, -1))
    minimum = torch.where(
        torch.isfinite(minimum),
        minimum,
        minimum.new_full((), float("nan")),
    )
    return minimum.reshape(batch_count, time_count)


def _connected_component_count(
    points: np.ndarray,
    active: np.ndarray,
    tolerance: float,
) -> int:
    parent = {int(index): int(index) for index in active}

    def find(index: int, parent_map: dict[int, int] = parent) -> int:
        while parent_map[index] != index:
            parent_map[index] = parent_map[parent_map[index]]
            index = parent_map[index]
        return index

    for left_offset, left in enumerate(active[:-1]):
        other = active[left_offset + 1 :]
        distances = np.linalg.norm(points[other] - points[left], axis=-1)
        for right in other[distances <= tolerance]:
            root_left, root_right = find(int(left)), find(int(right))
            if root_left != root_right:
                parent[root_right] = root_left
    return len({find(int(index)) for index in active})


def _approximate_mode_count(
    values: torch.Tensor,
    valid: torch.Tensor,
    tolerance: float,
) -> list[int | None]:
    """Count connected components under a stated tolerance for diagnostics only."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("mode tolerance must be finite and positive")
    points = values.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    counts: list[int | None] = []
    for batch_index, row in enumerate(points):
        mask = valid_np[batch_index] & np.isfinite(row).all(axis=-1)
        active = np.flatnonzero(mask)
        counts.append(
            None if active.size == 0 else _connected_component_count(row, active, tolerance)
        )
    return counts


def _descriptor_concentration(
    descriptors: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    batch_count, time_count, candidate_count, descriptor_dim = (int(item) for item in descriptors.shape)
    finite = torch.isfinite(descriptors).all(dim=-1)
    mask = valid[:, None, :].expand(batch_count, time_count, candidate_count) & finite
    denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(descriptors.dtype)
    safe = torch.where(mask[..., None], descriptors, torch.zeros_like(descriptors))
    centroid = safe.sum(dim=-2) / denominator
    distance = torch.linalg.vector_norm(descriptors - centroid[:, :, None, :], dim=-1)
    norm = torch.linalg.vector_norm(descriptors, dim=-1)
    per_iteration: list[dict[str, Any]] = []
    for time_index in range(time_count):
        row_mask = mask[:, time_index]
        pairwise = _pairwise_minimum(
            descriptors[:, time_index : time_index + 1],
            valid,
        )[:, 0]
        per_iteration.append(
            {
                "iteration": time_index,
                "candidate_count": int(row_mask.sum().detach().cpu()),
                "distance_to_centroid_l2": _summary(distance[:, time_index][row_mask]),
                "descriptor_norm": _summary(norm[:, time_index][row_mask]),
                "minimum_pairwise_l2": _summary(pairwise),
                "per_batch_distance_to_centroid_l2": _per_batch_summary(
                    distance[:, time_index], row_mask
                ),
            }
        )
    return {
        "status": "ok",
        "descriptor_width": descriptor_dim,
        "lower_distance_means_indicate_more_concentration": True,
        "per_iteration": per_iteration,
        "final": per_iteration[-1] if per_iteration else None,
    }


def candidate_trajectory_diagnostics(
    trajectory: torch.Tensor,
    *,
    valid: torch.Tensor,
    descriptors: torch.Tensor,
    length_scale: torch.Tensor | Sequence[float],
    feature_bandwidth: float = 1.0,
    mode_tolerance: float = DEFAULT_MODE_TOLERANCE,
    retain_trajectory_values: bool = True,
    max_trajectory_values: int = DEFAULT_MAX_TRAJECTORY_VALUES,
) -> dict[str, Any]:
    """Compute bounded physical/scaled motion and separation diagnostics."""

    if trajectory.ndim != 4:
        raise ValueError("trajectory must have shape [B,T,K,d]")
    batch_count, time_count, candidate_count, coordinate_dim = (int(item) for item in trajectory.shape)
    if time_count <= 0 or candidate_count <= 0 or coordinate_dim <= 0:
        raise ValueError("trajectory dimensions must be positive")
    if tuple(valid.shape) != (batch_count, candidate_count):
        raise ValueError("valid must have shape [B,K]")
    if tuple(descriptors.shape[:3]) != (batch_count, time_count, candidate_count):
        raise ValueError("descriptors must align with trajectory on [B,T,K]")
    if not math.isfinite(float(feature_bandwidth)) or float(feature_bandwidth) <= 0.0:
        raise ValueError("feature_bandwidth must be finite and positive")
    scale = torch.as_tensor(length_scale, device=trajectory.device, dtype=trajectory.dtype).reshape(-1)
    if scale.numel() != coordinate_dim or not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
        raise ValueError("length_scale must contain finite positive values for every coordinate")
    scaled = trajectory / scale.reshape(1, 1, 1, -1)
    finite_physical = torch.isfinite(trajectory).all(dim=-1)
    finite_scaled = torch.isfinite(scaled).all(dim=-1)
    valid_physical = valid[:, None, :].expand(batch_count, time_count, candidate_count) & finite_physical
    valid_scaled = valid[:, None, :].expand(batch_count, time_count, candidate_count) & finite_scaled

    def motion(values: torch.Tensor, row_valid: torch.Tensor) -> dict[str, Any]:
        movement = torch.linalg.vector_norm(values[:, 1:] - values[:, :-1], dim=-1)
        drift = torch.linalg.vector_norm(values - values[:, :1], dim=-1)
        movement_rows = []
        for time_index in range(max(time_count - 1, 0)):
            mask = row_valid[:, time_index] & row_valid[:, time_index + 1]
            movement_rows.append(
                {
                    "iteration": time_index + 1,
                    "from_iteration": time_index,
                    "to_iteration": time_index + 1,
                    "summary": _summary(movement[:, time_index][mask]),
                    "per_batch": _per_batch_summary(movement[:, time_index], mask),
                }
            )
        drift_rows = []
        for time_index in range(time_count):
            mask = row_valid[:, time_index] & row_valid[:, 0]
            drift_rows.append(
                {
                    "iteration": time_index,
                    "summary": _summary(drift[:, time_index][mask]),
                    "per_batch": _per_batch_summary(drift[:, time_index], mask),
                }
            )
        return {
            "drift_from_seed": drift_rows,
            "final_drift": drift_rows[-1] if drift_rows else None,
            "iteration_movement": movement_rows,
        }

    def separation(values: torch.Tensor) -> dict[str, Any]:
        minimum = _pairwise_minimum(values, valid)
        rows = [
            {
                "iteration": time_index,
                "summary": _summary(minimum[:, time_index]),
                "per_batch": _per_batch_summary(minimum[:, time_index]),
            }
            for time_index in range(time_count)
        ]
        return {
            "per_iteration": rows,
            "final": rows[-1] if rows else None,
            "undefined_when_fewer_than_two_valid_candidates": True,
        }

    scaled_tolerance = float(mode_tolerance / float(scale.max().detach().cpu()))
    joint = torch.cat((scaled, descriptors / float(feature_bandwidth)), dim=-1)
    joint_valid = valid_physical & torch.isfinite(descriptors).all(dim=-1)
    physical_motion = motion(trajectory, valid_physical)
    scaled_motion = motion(scaled, valid_scaled)
    joint_motion = motion(joint, joint_valid)
    physical_separation = separation(trajectory)
    scaled_separation = separation(scaled)
    joint_separation = separation(joint)
    physical_modes = [
        _approximate_mode_count(trajectory[:, time_index], valid, mode_tolerance)
        for time_index in range(time_count)
    ]
    scaled_modes = [
        _approximate_mode_count(scaled[:, time_index], valid, scaled_tolerance)
        for time_index in range(time_count)
    ]
    joint_modes = [
        _approximate_mode_count(joint[:, time_index], joint_valid[:, time_index], mode_tolerance)
        for time_index in range(time_count)
    ]
    result: dict[str, Any] = {
        "status": "ok",
        "shape": [batch_count, time_count, candidate_count, coordinate_dim],
        "iteration_count": max(time_count - 1, 0),
        "candidate_count": candidate_count,
        "valid_candidate_count": valid.sum(dim=-1).detach().cpu().tolist(),
        "valid_candidate_mask": valid.detach().cpu().tolist(),
        "length_scale_ell": scale.detach().cpu().tolist(),
        "joint_feature_bandwidth": float(feature_bandwidth),
        "physical": {
            "drift_from_seed": physical_motion["drift_from_seed"],
            "final_drift": physical_motion["final_drift"],
            "iteration_movement": physical_motion["iteration_movement"],
            "minimum_separation": physical_separation,
            "approximate_mode_count": {
                "counts_per_iteration_per_batch": physical_modes,
                "tolerance": float(mode_tolerance),
                "space": "physical candidate coordinates",
                "method": "connected components of pairwise distances <= tolerance",
                "diagnostic_only": True,
                "changes_candidate_execution": False,
            },
        },
        "scaled": {
            "drift_from_seed": scaled_motion["drift_from_seed"],
            "final_drift": scaled_motion["final_drift"],
            "iteration_movement": scaled_motion["iteration_movement"],
            "minimum_separation": scaled_separation,
            "approximate_mode_count": {
                "counts_per_iteration_per_batch": scaled_modes,
                "tolerance": scaled_tolerance,
                "space": "coordinates divided by routing length scale ell",
                "tolerance_conversion": "conservative scalar reporting convention: physical tolerance divided by max ell; inspect ell per axis for anisotropic scales",
                "method": "connected components of pairwise distances <= tolerance",
                "diagnostic_only": True,
                "changes_candidate_execution": False,
            },
        },
        "joint": {
            "drift_from_seed": joint_motion["drift_from_seed"],
            "final_drift": joint_motion["final_drift"],
            "iteration_movement": joint_motion["iteration_movement"],
            "minimum_separation": joint_separation,
            "approximate_mode_count": {
                "counts_per_iteration_per_batch": joint_modes,
                "tolerance": float(mode_tolerance),
                "space": "dimensionless joint state [candidate coordinates / ell, descriptor / feature_bandwidth] used by mean shift",
                "method": "connected components of pairwise distances <= tolerance",
                "diagnostic_only": True,
                "changes_candidate_execution": False,
            },
        },
        "descriptor_concentration": _descriptor_concentration(descriptors, valid),
    }
    if retain_trajectory_values:
        value_count = int(trajectory.numel())
        if value_count <= int(max_trajectory_values):
            result["trajectory_physical"] = trajectory.detach().cpu().tolist()
            result["trajectory_scaled"] = scaled.detach().cpu().tolist()
        else:
            result["trajectory_values_omitted"] = {
                "physical_shape": list(trajectory.shape),
                "scaled_shape": list(scaled.shape),
                "value_count": value_count,
                "max_trajectory_values": int(max_trajectory_values),
            }
    return result


def _candidate_snapshot(
    candidate: Any,
    backend: Any,
    encoded: Any,
    *,
    mode_tolerance: float,
) -> dict[str, Any]:
    coords, descriptors, valid = _candidate_fields(candidate)
    scale = _routing_length_scale(backend, encoded, coords)
    return candidate_trajectory_diagnostics(
        coords.unsqueeze(1),
        valid=valid,
        descriptors=descriptors.unsqueeze(1),
        length_scale=scale,
        feature_bandwidth=_routing_feature_bandwidth(backend),
        mode_tolerance=mode_tolerance,
        retain_trajectory_values=False,
    )


def _benchmark_candidate_generation(
    original_build: Any,
    call_args: tuple[Any, ...],
    call_kwargs: Mapping[str, Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Time candidate construction without trajectory metadata overhead."""

    kwargs = dict(call_kwargs)
    kwargs["return_trajectory"] = False
    for _ in range(int(warmup)):
        original_build(*call_args, **kwargs)
    _sync(device)
    started = time.perf_counter()
    for _ in range(int(repeats)):
        original_build(*call_args, **kwargs)
    _sync(device)
    elapsed = time.perf_counter() - started
    return {
        "seconds_total": float(elapsed),
        "seconds_mean": float(elapsed / int(repeats)),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "return_trajectory": False,
    }


@contextlib.contextmanager
def _temporary_strategy(backend: Any, strategy: str) -> Iterator[None]:
    previous = getattr(backend, "routing_strategy", None)
    if previous is None:
        raise DiagnosticUnavailable("routed backend exposes no routing_strategy")
    backend.routing_strategy = str(strategy)
    try:
        yield
    finally:
        backend.routing_strategy = previous


@contextlib.contextmanager
def _capture_preparations(
    model: Any,
    *,
    device: torch.device,
    mode_tolerance: float,
    compare_module_hubs: bool,
    timing_warmup: int,
    timing_repeats: int,
) -> Iterator[list[dict[str, Any]]]:
    """Capture exact mean-shift preparation metadata and candidate timings."""

    core = getattr(model, "core", None)
    backend = getattr(core, "backend", None)
    original_prepare = getattr(backend, "prepare", None)
    original_build = getattr(backend, "_build_candidates_with_sources", None)
    if not callable(original_prepare) or not callable(original_build):
        raise DiagnosticUnavailable(
            "routed backend must expose prepare and _build_candidates_with_sources hooks"
        )
    build_records: list[dict[str, Any]] = []
    preparations: list[dict[str, Any]] = []

    def wrapped_build(*call_args: Any, **call_kwargs: Any) -> Any:
        phase = str(getattr(core, "_interface_read_role", "unscoped"))
        _sync(device)
        started = time.perf_counter()
        result = original_build(*call_args, **call_kwargs)
        _sync(device)
        actual_seconds = time.perf_counter() - started
        record: dict[str, Any] = {
            "phase": phase,
            "result": result,
            "candidate_generation_seconds_in_forward": float(actual_seconds),
            "candidate_generation_timing": _benchmark_candidate_generation(
                original_build,
                tuple(call_args),
                call_kwargs,
                device=device,
                warmup=timing_warmup,
                repeats=timing_repeats,
            ),
        }
        if compare_module_hubs:
            reference_kwargs = dict(call_kwargs)
            reference_kwargs["return_trajectory"] = False
            try:
                with _temporary_strategy(backend, "module_hubs"):
                    _sync(device)
                    reference_started = time.perf_counter()
                    reference_result = original_build(*call_args, **reference_kwargs)
                    _sync(device)
                    reference_actual = time.perf_counter() - reference_started
                    reference_timing = _benchmark_candidate_generation(
                        original_build,
                        tuple(call_args),
                        reference_kwargs,
                        device=device,
                        warmup=timing_warmup,
                        repeats=timing_repeats,
                    )
                record["module_hubs_reference"] = {
                    "status": "ok",
                    "candidate_generation_seconds_in_reference_call": float(reference_actual),
                    "candidate_generation_timing": reference_timing,
                    "candidate": reference_result[0],
                }
            except Exception as error:  # noqa: BLE001 - retain optional comparison status
                record["module_hubs_reference"] = {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
        build_records.append(record)
        return result

    def wrapped_prepare(*call_args: Any, **call_kwargs: Any) -> Any:
        build_start = len(build_records)
        state = original_prepare(*call_args, **call_kwargs)
        encoded = call_args[0] if call_args else call_kwargs.get("encoded")
        phase = str(getattr(core, "_interface_read_role", "unscoped"))
        current_build = build_records[build_start:]
        if not isinstance(state, Mapping) or encoded is None or not current_build:
            return state
        routing_index = state.get("routing_index")
        final_candidate = getattr(routing_index, "candidates", None)
        build_record = current_build[-1]
        final_candidate_metrics: dict[str, Any]
        try:
            coords, _, _ = _candidate_fields(final_candidate)
            scale = _routing_length_scale(backend, encoded, coords)
            final_candidate_metrics = _candidate_snapshot(
                final_candidate,
                backend,
                encoded,
                mode_tolerance=mode_tolerance,
            )
            try:
                trajectory, trajectory_descriptors, trajectory_valid = _candidate_trajectory(
                    state,
                    final_candidate,
                )
                candidate_metrics = candidate_trajectory_diagnostics(
                    trajectory,
                    valid=trajectory_valid,
                descriptors=trajectory_descriptors,
                length_scale=scale,
                feature_bandwidth=_routing_feature_bandwidth(backend),
                mode_tolerance=mode_tolerance,
                    retain_trajectory_values=True,
                )
            except DiagnosticUnavailable as error:
                candidate_metrics = {
                    "status": "unavailable",
                    "reason": f"{type(error).__name__}: {error}",
                }
            except Exception as error:  # noqa: BLE001 - preserve final candidate summary
                candidate_metrics = {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
        except DiagnosticUnavailable as error:
            candidate_metrics = {
                "status": "unavailable",
                "reason": f"{type(error).__name__}: {error}",
            }
            final_candidate_metrics = {}
        except Exception as error:  # noqa: BLE001 - preserve support ledger on metric failure
            candidate_metrics = {
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            final_candidate_metrics = {}
        common_summary = _routing_index_summary(state, encoded, phase=phase)
        reference = build_record.get("module_hubs_reference")
        reference_payload: dict[str, Any]
        if not isinstance(reference, Mapping):
            reference_payload = {"status": "unavailable"}
        elif reference.get("status") != "ok":
            reference_payload = {
                "status": reference.get("status"),
                "reason": reference.get("reason"),
            }
        else:
            reference_payload = {
                "status": "ok",
                "candidate_generation_seconds_in_reference_call": reference.get(
                    "candidate_generation_seconds_in_reference_call"
                ),
                "candidate_generation_timing": reference.get("candidate_generation_timing"),
                "candidate": _candidate_snapshot(
                    reference.get("candidate"),
                    backend,
                    encoded,
                    mode_tolerance=mode_tolerance,
                ),
            }
        preparations.append(
            {
                "phase": phase,
                "candidate_generation": {
                    "seconds_in_forward": build_record.get(
                        "candidate_generation_seconds_in_forward"
                    ),
                    "timing": build_record.get("candidate_generation_timing"),
                    "module_hubs_reference": reference_payload,
                },
                "candidates": candidate_metrics,
                "final_candidate": final_candidate_metrics,
                "routing_index": common_summary,
            }
        )
        return state

    backend._build_candidates_with_sources = wrapped_build
    backend.prepare = wrapped_prepare
    try:
        yield preparations
    finally:
        backend.prepare = original_prepare
        backend._build_candidates_with_sources = original_build


def _fine_support_statistics(ledger: Mapping[str, Any]) -> dict[str, Any]:
    phases = ledger.get("phase_metrics", {})
    result: dict[str, Any] = {}
    if isinstance(phases, Mapping):
        for phase, metrics in phases.items():
            if not isinstance(metrics, Mapping):
                continue
            result[str(phase)] = {
                key: metrics.get(key)
                for key in (
                    "executed_receiver_count",
                    "raw_path_count",
                    "unique_pair_count",
                    "duplicate_expansion",
                    "fine_pair_count",
                    "query_hub_support",
                    "query_hub_union",
                    "dense_active_module_pair_reference",
                    "dense_environment_pair_reference",
                )
                if key in metrics
            }
    result["source_and_candidate_occupancy"] = ledger.get("final_routing_index", {}).get(
        "incidence", {}
    )
    result["semantics"] = (
        "Actual common-router positive supports and deduplicated fine pairs; "
        "learned route weights are not physical influence."
    )
    return result


def _query_grid_indices(
    raw_sample: Mapping[str, Any],
    query_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_grid = np.asarray(raw_sample["x_grid"], dtype=np.float64)
    y_grid = np.asarray(raw_sample["y_grid"], dtype=np.float64)
    grid_points = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    query = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(query[:, None, :] - grid_points[None, :, :], axis=-1)
    indices = np.argmin(distances, axis=1).astype(np.int64)
    return indices, distances[np.arange(len(query)), indices]


def _fluid_query_mask(
    raw_sample: Mapping[str, Any],
    query_indices: np.ndarray,
    model: Any,
) -> tuple[np.ndarray, str]:
    module_mask = raw_sample.get("module_mask")
    if module_mask is not None:
        return ~np.asarray(module_mask, dtype=bool).reshape(-1)[query_indices], "raw_sample.module_mask"
    x_grid = np.asarray(raw_sample["x_grid"], dtype=np.float64)
    y_grid = np.asarray(raw_sample["y_grid"], dtype=np.float64)
    centers = np.asarray(raw_sample["structure"]["module_centers"], dtype=np.float64)
    present = np.asarray(raw_sample["structure"]["module_present"], dtype=np.float64) > 0.5
    material = np.asarray(raw_sample["structure"].get("material_params", []), dtype=np.float64).reshape(-1)
    fallback_radius = float(getattr(model.config.core_honf, "module_radius", 0.0))
    radius = float(material[5]) if material.size > 5 and material[5] > 0.0 else fallback_radius
    points = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    inside = np.zeros((points.shape[0],), dtype=bool)
    for center in centers[present]:
        inside |= np.linalg.norm(points - center.reshape(1, 2), axis=-1) <= radius
    return ~inside[query_indices], "geometry_fallback_module_radius"


def _query_field_metrics(
    outputs: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
    query_xy: np.ndarray,
    dataset: Any,
    checkpoint: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    """Compare query-batch fields to masked, dataset-normalized grid targets."""

    prediction = outputs.get("pred_field")
    if not torch.is_tensor(prediction):
        return {"status": "unavailable", "reason": "forward output has no pred_field tensor"}
    prediction_np = prediction.detach().cpu().numpy()
    if prediction_np.ndim == 3 and prediction_np.shape[0] == 1:
        prediction_np = prediction_np[0]
    if prediction_np.ndim != 2:
        return {
            "status": "unavailable",
            "reason": f"pred_field must be [Q,F] after batch squeeze, got {prediction_np.shape}",
        }
    query_indices, distances = _query_grid_indices(raw_sample, query_xy)
    target_grid = np.asarray(raw_sample["steady_field"], dtype=np.float64).reshape(
        -1, np.asarray(raw_sample["steady_field"]).shape[-1]
    )
    field_width = min(int(prediction_np.shape[-1]), int(target_grid.shape[-1]))
    if field_width <= 0 or int(prediction_np.shape[0]) != int(query_indices.size):
        return {
            "status": "unavailable",
            "reason": "prediction and query-aligned steady_field widths/counts differ",
        }
    prediction_np = np.asarray(prediction_np[:, :field_width], dtype=np.float64)
    target_np = target_grid[query_indices, :field_width]
    normalized_targets = bool(
        checkpoint.get("train_config", {}).get("dataset", {}).get("normalize_targets", False)
    )
    normalizer = getattr(dataset, "normalizer", None)
    if normalizer is not None and hasattr(normalizer, "normalize_fields"):
        target_norm = np.asarray(normalizer.normalize_fields(target_np), dtype=np.float64)
        prediction_norm = (
            prediction_np
            if normalized_targets
            else np.asarray(normalizer.normalize_fields(prediction_np), dtype=np.float64)
        )
        target_space = "dataset_normalized"
    else:
        target_norm = target_np
        prediction_norm = prediction_np
        target_space = "raw_checkpoint_space"
    fluid_mask, mask_source = _fluid_query_mask(raw_sample, query_indices, model)
    finite = np.isfinite(prediction_norm).all(axis=-1) & np.isfinite(target_norm).all(axis=-1)
    selected = fluid_mask & finite
    diff = prediction_norm[selected] - target_norm[selected]
    truth = target_norm[selected]
    sse = float(np.dot(diff.reshape(-1), diff.reshape(-1))) if diff.size else 0.0
    target_sse = float(np.dot(truth.reshape(-1), truth.reshape(-1))) if truth.size else 0.0
    return {
        "status": "ok",
        "target_space": target_space,
        "prediction_space_was_checkpoint_normalized": normalized_targets,
        "mask": "fluid_only",
        "mask_source": mask_source,
        "query_count": int(query_indices.size),
        "matched_grid_count": int(np.sum(distances <= 1.0e-5)),
        "max_query_to_grid_distance": float(np.max(distances)) if distances.size else 0.0,
        "fluid_query_count": int(np.sum(fluid_mask)),
        "finite_fluid_query_count": int(np.sum(selected)),
        "sse": sse,
        "target_sse": target_sse,
        "count": int(diff.size),
        "relative_l2": float(math.sqrt(sse / max(target_sse, 1.0e-12))),
        "normalization_definition": "checkpoint dataset field mean/std applied to both prediction and target",
    }


def _prediction_discrepancy(
    normal: Mapping[str, Any],
    module_hubs: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "pred_field",
        "pred_interface",
        "pred_internal_temperature",
        "pred_port_condition",
    ):
        if torch.is_tensor(normal.get(key)) and torch.is_tensor(module_hubs.get(key)):
            result[key] = _relative_difference(module_hubs[key], normal[key])
    return result


def _comparison_error_delta(
    normal: Mapping[str, Any],
    module_hubs: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("sse", "target_sse", "count", "relative_l2"):
        left, right = module_hubs.get(key), normal.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[key] = float(left) - float(right)
    return result


def _run_forward_with_strategy(
    model: Any,
    sample: Mapping[str, Any],
    query_np: np.ndarray,
    device: torch.device,
    *,
    strategy: str,
    mode_tolerance: float,
    compare_module_hubs: bool,
    timing_warmup: int,
    timing_repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    backend = model.core.backend
    with _temporary_strategy(backend, strategy), _capture_preparations(
        model,
        device=device,
        mode_tolerance=mode_tolerance,
        compare_module_hubs=compare_module_hubs,
        timing_warmup=timing_warmup,
        timing_repeats=timing_repeats,
    ) as preparations, torch.no_grad():
        outputs = _forward_batch(
            model,
            sample,
            query_np,
            device,
            return_prepared_state=True,
            return_routing_maps=True,
            return_organizer_passes=True,
        )
    return outputs, preparations


def _checkpoint_record(spec: Any, checkpoint: Mapping[str, Any], model: Any) -> dict[str, Any]:
    core = getattr(getattr(model, "config", None), "core_honf", None)
    interface = getattr(core, "interface_model", None)
    routing = getattr(interface, "routing", None)
    return {
        "label": str(getattr(spec, "label", "fresh_profile")),
        "path": str(getattr(spec, "path", "fresh_profile_unsaved")),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", 0))),
        "forward_architecture": str(getattr(core, "forward_architecture", "unknown")),
        "routing_strategy": str(getattr(routing, "strategy", "unknown")),
    }


def _run_one_case(
    model: Any,
    dataset: Any,
    dataset_path: Path,
    raw_sample: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    case_id: str,
    device: torch.device,
    split: str,
    query_count: int,
    mode_tolerance: float,
    compare_module_hubs: bool,
    timing_warmup: int,
    timing_repeats: int,
) -> dict[str, Any]:
    sample = select_sample(dataset, str(case_id), 0)
    query_np = _query_points(sample, int(query_count))
    normal_outputs, preparations = _run_forward_with_strategy(
        model,
        sample,
        query_np,
        device,
        strategy="mean_shift",
        mode_tolerance=mode_tolerance,
        compare_module_hubs=compare_module_hubs,
        timing_warmup=timing_warmup,
        timing_repeats=timing_repeats,
    )
    normal_ledger = _ledger_row(normal_outputs, sample, str(case_id), len(query_np))
    normal_field_metrics = _query_field_metrics(
        normal_outputs,
        raw_sample,
        query_np,
        dataset,
        checkpoint,
        model,
    )
    row: dict[str, Any] = {
        "case_id": str(case_id),
        "query_count": len(query_np),
        "split": str(split),
        "dataset": str(dataset_path),
        "preparations": preparations,
        "fine_support_statistics": _fine_support_statistics(normal_ledger),
        "common_router_ledger": normal_ledger,
        "query_batch_ground_truth": normal_field_metrics,
        "canonical_ground_truth_errors": _canonical_ground_truth_errors(
            normal_outputs,
            sample,
            raw_sample,
            dataset,
            model,
            checkpoint,
        ),
    }
    if not compare_module_hubs:
        row["candidate_intervention_comparison"] = {"status": "disabled"}
        return row

    module_outputs, module_preparations = _run_forward_with_strategy(
        model,
        sample,
        query_np,
        device,
        strategy="module_hubs",
        mode_tolerance=mode_tolerance,
        compare_module_hubs=False,
        timing_warmup=timing_warmup,
        timing_repeats=timing_repeats,
    )
    module_ledger = _ledger_row(module_outputs, sample, str(case_id), len(query_np))
    module_field_metrics = _query_field_metrics(
        module_outputs,
        raw_sample,
        query_np,
        dataset,
        checkpoint,
        model,
    )
    row["candidate_intervention_comparison"] = {
        "status": "ok",
        "intervention": "temporary backend.routing_strategy='module_hubs'",
        "same_model_weights": True,
        "trained_accuracy_comparison": False,
        "prediction_discrepancy_module_hubs_minus_mean_shift": _prediction_discrepancy(
            normal_outputs,
            module_outputs,
        ),
        "query_batch_ground_truth": {
            "mean_shift": normal_field_metrics,
            "module_hubs": module_field_metrics,
            "module_hubs_minus_mean_shift": _comparison_error_delta(
                normal_field_metrics,
                module_field_metrics,
            ),
        },
        "fine_support_statistics": {
            "mean_shift": row["fine_support_statistics"],
            "module_hubs": _fine_support_statistics(module_ledger),
        },
        "module_hubs_forward": {
            "preparations": module_preparations,
            "common_router_ledger": module_ledger,
            "canonical_ground_truth_errors": _canonical_ground_truth_errors(
                module_outputs,
                sample,
                raw_sample,
                dataset,
                model,
                checkpoint,
            ),
        },
        "interpretation": (
            "Frozen candidate-generation intervention under the same model weights; "
            "it does not establish trained Run-2100 accuracy."
        ),
    }
    return row


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    raw_specs = list(args.checkpoint or [])
    device = select_device(args.device)
    if int(args.query_count) <= 0:
        raise ValueError("query count must be positive")
    if int(args.timing_warmup) < 0 or int(args.timing_repeats) <= 0:
        raise ValueError("timing warmup must be nonnegative and repeats must be positive")
    if raw_specs:
        specs = parse_checkpoint_specs(raw_specs)
        if len(specs) != 1:
            raise ValueError("run requires exactly one LABEL=PATH checkpoint")
        spec = specs[0]
        model, checkpoint = _load_model_spec(spec, device)
        dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
    else:
        model, train_dataset, checkpoint, dataset_path = _build_fresh_profile_model(args, device)
        spec = SimpleNamespace(label="fresh_profile", path=Path("fresh_profile_unsaved"))
        close_train = getattr(train_dataset, "close", None)
        if callable(close_train):
            close_train()
        dataset, dataset_path = _load_dataset(checkpoint, _profile_args(args))
    architecture = str(model.config.core_honf.forward_architecture)
    routing = getattr(getattr(model.config.core_honf, "interface_model", None), "routing", None)
    strategy = str(getattr(routing, "strategy", "unknown"))
    if architecture != "routed_pairwise_honf":
        raise ValueError(f"mean-shift diagnostics require routed_pairwise_honf, got {architecture!r}")
    if strategy != "mean_shift":
        raise ValueError(f"mean-shift diagnostics require routing.strategy='mean_shift', got {strategy!r}")
    case_ids = [str(value) for value in (args.case_id or ANCHOR_CASE_IDS)]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    raw_samples = {
        case_id: _load_raw_sample(dataset_path, args.split, case_id)
        for case_id in case_ids
    }
    rows: list[dict[str, Any]] = []
    try:
        for case_id in case_ids:
            rows.append(
                _run_one_case(
                    model,
                    dataset,
                    dataset_path,
                    raw_samples[case_id],
                    checkpoint,
                    case_id=case_id,
                    device=device,
                    split=str(args.split),
                    query_count=int(args.query_count),
                    mode_tolerance=float(args.mode_tolerance),
                    compare_module_hubs=bool(args.compare_module_hubs),
                    timing_warmup=int(args.timing_warmup),
                    timing_repeats=int(args.timing_repeats),
                )
            )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return {
        "schema_version": 2,
        "task": "mean_shift_routing_diagnostics",
        "status": "ok",
        "device": str(device),
        "checkpoint": _checkpoint_record(spec, checkpoint, model),
        "case_ids": case_ids,
        "query_count": int(args.query_count),
        "mode_tolerance_physical": float(args.mode_tolerance),
        "module_hubs_reference": bool(args.compare_module_hubs),
        "candidate_timing": {
            "warmup": int(args.timing_warmup),
            "repeats": int(args.timing_repeats),
            "timed_scope": "backend._build_candidates_with_sources with return_trajectory=False",
            "forward_scope": "actual candidate builder call, including requested trajectory metadata",
        },
        "results": rows,
        "interpretation": {
            "candidate_trajectory": "Finite mean-shift motion in physical coordinates; trajectory arrays are diagnostic metadata only.",
            "scaled_coordinates": "Physical coordinates divided by adapter routing length scale ell, matching mean-shift geometry; scalar mode tolerance is conservative when ell is anisotropic.",
            "joint_coordinates": "Dimensionless [x/ell, descriptor/feature_bandwidth] state used by the fixed-data mean-shift kernel.",
            "approximate_mode_count": "Tolerance-labelled connected-component diagnostic only; it never merges candidates or controls routing execution.",
            "fine_support": "Captured from the common routed pair compiler after positive query/source support and before fine QM/QE evaluation.",
            "candidate_comparison": "Mean-shift and module-hub forwards use the same model weights; the module-hub run is a temporary frozen candidate intervention, not a separately trained accuracy result.",
            "barrier_evidence": "ThermalChannel neutral resistance is an implementation contract; physical barrier benefit remains Evidence Missing without solver-labelled layouts.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    run = subparsers.add_parser("run", help="evaluate bounded mean-shift candidate and common-router diagnostics")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    run.add_argument("--profile", default=DEFAULT_PROFILE)
    run.add_argument("--dataset", default=None)
    run.add_argument("--split", default="test")
    run.add_argument("--case-id", action="append", default=None)
    run.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    run.add_argument("--device", default="cpu")
    run.add_argument("--mode-tolerance", type=float, default=DEFAULT_MODE_TOLERANCE)
    run.add_argument("--timing-warmup", type=int, default=DEFAULT_TIMING_WARMUP)
    run.add_argument("--timing-repeats", type=int, default=DEFAULT_TIMING_REPEATS)
    run.add_argument(
        "--compare-module-hubs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the same-weight temporary module-hub forward and candidate reference",
    )
    run.set_defaults(handler=run_diagnostics)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.handler(args)
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
