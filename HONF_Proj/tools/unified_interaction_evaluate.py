"""Maintained development evaluator for unified interaction-refinement checkpoints.

The evaluator reconstructs the sealed Thermal/Wind task providers, validates
checkpoint ages and normalization/source identities, and keeps inference inputs
separate from native targets. Detailed maps are limited to the fixed
representatives; complete development metrics use the providers' exact
DEV22/DEV24 reducers. Outputs are one-time evidence under the ignored
``diagnostics/generated/unified_refinement_20261007`` tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "diagnostics/generated/unified_refinement_20261007/evaluation"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "diagnostics/generated/unified_refinement_20261007/runs"
ROUTE_FRONTIER_THRESHOLDS = (0.25, 0.5, 0.75)
CONTROL_MODES = ("adaptive", "all_base", "nearest", "upstream", "shuffle")
THERMAL_REPRESENTATIVES = ("0277", "0291", "0294", "0687")
WIND_REPRESENTATIVE_ROWS = ((69, 8), (426, 30))
WIND_DIRECTION_PAIR_ROWS = (6, 7)
WIND_DIRECTION_PAIR_RECEIVERS = 9


def _add_import_paths() -> None:
    paths = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "Case_ThermalChannel/src",
        PROJECT_ROOT / "Case_WindFarm/src",
    )
    for path in reversed(paths):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _as_numpy(value: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _synchronize(value: Any) -> None:
    """Include completed device work and CPU transfer in timed call scopes."""
    import torch

    device = value.device if torch.is_tensor(value) else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tail_summary(residual: Any) -> dict[str, Any]:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "available": False}
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "available": True,
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(np.max(absolute)),
    }


def _dependency_payload(dependency: Any) -> dict[str, Any] | None:
    if dependency is None:
        return None
    if is_dataclass(dependency):
        return asdict(dependency)
    if isinstance(dependency, Mapping):
        return dict(dependency)
    return {name: getattr(dependency, name) for name in getattr(dependency, "__dataclass_fields__", {})}


def _thermal_native_metrics(case: Mapping[str, Any], output: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Summarize unweighted, point-pooled residuals on a representative native grid.

    ``point_weights`` do not enter this display-oriented summary. The fixed25
    DEV fluid panel has positive but nonuniform quadrature weights, so these
    residual tails are distinct from the provider's point-weighted per-case
    fluid RMSE and equal-case DEV summary.
    """
    from channelthermal.training.unified_task import RESPONSE_SURFACE_STRIDE

    target_fluid = np.asarray(case["field_targets"][..., 4:5], dtype=np.float64)
    target_interface = np.asarray(case["interface_target"], dtype=np.float64)
    target_material = np.asarray(case["module_internal_temperature_points"], dtype=np.float64)[..., None]
    present = np.asarray(case["structure"]["module_present"], dtype=bool)
    interface_mask = np.squeeze(np.asarray(case["interface_condition_valid_mask"], dtype=bool))
    if "pred_interface" in output and output["pred_interface"].shape[-2] != target_interface.shape[-2]:
        target_count = target_interface.shape[-2]
        prediction_count = output["pred_interface"].shape[-2]
        if (
            target_count % prediction_count != 0
            or target_count // prediction_count != RESPONSE_SURFACE_STRIDE
        ):
            raise ValueError("Thermal native interface target count does not match the sealed surface stride.")
        stride = int(RESPONSE_SURFACE_STRIDE)
        target_interface = target_interface[..., ::stride, :]
        if interface_mask.ndim and interface_mask.shape[-1] == target_count:
            interface_mask = interface_mask[..., ::stride]
    if interface_mask.shape != target_interface.shape[:-1]:
        interface_mask = np.isfinite(target_interface[..., 0])
    result: dict[str, Any] = {"fluid_temperature": _tail_summary(output["fluid_temperature"][0] - target_fluid)}
    if "pred_interface" in output:
        predicted = output["pred_interface"][0]
        mask = present[:, None] & interface_mask
        result["interface_temperature"] = _tail_summary((predicted[..., 0] - target_interface[..., 0])[mask])
        result["interface_q_proxy"] = _tail_summary((predicted[..., 1] - target_interface[..., 1])[mask])
    if "pred_internal_temperature" in output:
        predicted = output["pred_internal_temperature"][0]
        material_mask = np.broadcast_to(present[:, None], target_material.shape[:-1])
        result["material_temperature"] = _tail_summary(
            (predicted[..., 0] - target_material[..., 0])[material_mask]
        )
    return result


def _wind_native_metrics(prediction: Any, reference: Any) -> dict[str, Any]:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "components_mps": [_tail_summary(residual[..., index]) for index in range(3)],
        "vector_magnitude_mps": _tail_summary(np.linalg.norm(residual, axis=-1)),
    }


def _thermal_geometry_region_masks(
    query_xy: Any,
    centers_xy: Any,
    present: Any,
    radius: float,
    inlet_speed: float,
    fluid_valid: Any,
) -> dict[str, np.ndarray]:
    """Build input-only near/downstream masks on native Thermal receivers.

    Near means within two module radii of any active center. Downstream uses
    the prescribed inlet sign and the existing geometry-only band
    ``2r < dx <= 6r, |dy| <= 2r``, excluding the near mask. Coordinates and
    radius remain in packed dataset-native geometry units.
    """
    query = np.asarray(query_xy, dtype=np.float64)
    centers = np.asarray(centers_xy, dtype=np.float64)
    active = np.asarray(present, dtype=bool).reshape(-1)
    valid = np.asarray(fluid_valid, dtype=bool).reshape(-1)
    if query.ndim != 2 or query.shape[-1] != 2 or valid.shape != (query.shape[0],):
        raise ValueError("Thermal region masks require [Q,2] queries and an aligned validity mask.")
    if centers.ndim != 2 or centers.shape[-1] != 2 or active.shape != (centers.shape[0],):
        raise ValueError("Thermal region masks require [M,2] centers and an aligned presence mask.")
    if not np.isfinite(query).all() or not np.isfinite(centers[active]).all():
        raise ValueError("Thermal geometry region inputs must be finite for active sources and receivers.")
    if not math.isfinite(float(radius)) or radius <= 0.0:
        raise ValueError("Thermal native module radius must be finite and positive.")
    near = np.zeros(query.shape[0], dtype=bool)
    downstream = np.zeros_like(near)
    active_centers = centers[active]
    if active_centers.size:
        relative = query[:, None, :] - active_centers[None, :, :]
        distances = np.linalg.norm(relative, axis=-1)
        near = valid & (distances.min(axis=1) <= 2.0 * radius)
        if math.isfinite(float(inlet_speed)) and float(inlet_speed) != 0.0:
            direction = math.copysign(1.0, float(inlet_speed))
            dx = direction * relative[..., 0]
            dy = np.abs(relative[..., 1])
            downstream = valid & np.any(
                (dx > 2.0 * radius) & (dx <= 6.0 * radius) & (dy <= 2.0 * radius),
                axis=1,
            )
            downstream &= ~near
    return {"fluid": valid.copy(), "near": near, "downstream": downstream}


def _equal_case_region_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    available = [float(row[key]["rmse"]) for row in rows if row[key].get("available", False)]
    return {
        "available_case_count": len(available),
        "empty_case_ids": [str(row["case_id"]) for row in rows if not row[key].get("available", False)],
        "equal_case_rmse_mean": float(np.mean(available)) if available else None,
        "equal_case_rmse_p90": float(np.quantile(available, 0.9)) if available else None,
    }


def _wind_plane_geometry(run: Any, requested_height_m: float = 70.0) -> dict[str, Any]:
    """Build native z-plane coordinates without reading any velocity values."""
    from windfarm.geometry import D_M
    from windfarm.study_spatial import native_coordinates

    index = int(np.argmin(np.abs(np.asarray(run.z_m) - requested_height_m)))
    iy, ix = np.meshgrid(np.arange(run.ny), np.arange(run.nx), indexing="ij")
    flat = (ix + run.nx * (iy + run.ny * index)).reshape(-1).astype(np.int64)
    return {
        "flat_indices": flat,
        "coords_D": native_coordinates(run, flat, D_M),
        "shape": (int(run.ny), int(run.nx)),
        "fixed_axis": "z",
        "requested_m": float(requested_height_m),
        "actual_m": float(np.asarray(run.z_m)[index]),
        "index": index,
        "horizontal_axis": "x",
        "vertical_axis": "y",
        "horizontal_D": np.asarray(run.x_m, dtype=np.float64) / D_M,
        "vertical_D": np.asarray(run.y_m, dtype=np.float64) / D_M,
    }


def _fit_wind_source_rotation(
    source_ids_from: Any,
    centers_from_D: Any,
    source_ids_to: Any,
    centers_to_D: Any,
    *,
    expected_degrees_ccw: float = 15.0,
    max_center_error_D: float = 1.0e-5,
) -> dict[str, Any]:
    """Fit and verify the input-only rigid frame map for paired turbine IDs."""
    ids_a = np.asarray(source_ids_from, dtype=np.int64).reshape(-1)
    ids_b = np.asarray(source_ids_to, dtype=np.int64).reshape(-1)
    centers_a = np.asarray(centers_from_D, dtype=np.float64)
    centers_b = np.asarray(centers_to_D, dtype=np.float64)
    if centers_a.shape != (ids_a.size, 3) or centers_b.shape != (ids_b.size, 3):
        raise ValueError("Wind source centers must align with their physical source-ID catalogs.")
    if not np.array_equal(ids_a, ids_b) or np.unique(ids_a).size != ids_a.size:
        raise ValueError("Wind direction-pair source IDs must be unique and in identical physical order.")
    if ids_a.size < 3 or not (np.isfinite(centers_a).all() and np.isfinite(centers_b).all()):
        raise ValueError("Wind direction-pair geometry needs at least three finite matched sources.")
    centered_a = centers_a[:, :2] - centers_a[:, :2].mean(axis=0)
    centered_b = centers_b[:, :2] - centers_b[:, :2].mean(axis=0)
    left, _, right_t = np.linalg.svd(centered_a.T @ centered_b)
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    rotation_row = left @ correction @ right_t
    rotation_column = rotation_row.T
    translation = centers_b[:, :2].mean(axis=0) - rotation_column @ centers_a[:, :2].mean(axis=0)
    predicted_b = centers_a[:, :2] @ rotation_row + translation
    errors = np.linalg.norm(predicted_b - centers_b[:, :2], axis=1)
    angle = math.degrees(math.atan2(rotation_column[1, 0], rotation_column[0, 0]))
    z_error = float(np.max(np.abs(centers_a[:, 2] - centers_b[:, 2])))
    max_error = float(np.max(errors))
    angle_error = (angle - expected_degrees_ccw + 180.0) % 360.0 - 180.0
    if max_error > max_center_error_D or abs(z_error) > 1.0e-7 or abs(angle_error) > 1.0e-3:
        raise ValueError(
            "Wind same-layout source correspondence failed the declared rotation/center checks "
            f"(max_xy_error_D={max_error:.3g}, max_z_error_D={z_error:.3g}, "
            f"rotation_error_deg={angle_error:.3g})."
        )
    return {
        "source_ids": ids_a.copy(),
        "rotation_column_xy": rotation_column,
        "translation_xy_D": translation,
        "rotation_degrees_ccw": float(angle),
        "max_center_error_D": max_error,
        "max_z_error_D": z_error,
        "matched_source_count": int(ids_a.size),
    }


def _map_wind_direction_coordinates(coordinates_D: Any, transform: Mapping[str, Any]) -> np.ndarray:
    coordinates = np.asarray(coordinates_D, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ValueError("Wind direction-pair receivers must be a finite [Q,3] coordinate array.")
    rotation = np.asarray(transform["rotation_column_xy"], dtype=np.float64)
    translation = np.asarray(transform["translation_xy_D"], dtype=np.float64)
    if rotation.shape != (2, 2) or translation.shape != (2,):
        raise ValueError("Wind direction-pair frame transform has an invalid shape.")
    result = coordinates.copy()
    result[:, :2] = coordinates[:, :2] @ rotation.T + translation
    return result


def _wind_trilinear_reference(run: Any, coordinates_D: Any, diameter_m: float = 80.0) -> dict[str, np.ndarray]:
    """Read target velocity only at the eight native interpolation neighbors."""
    coordinates = np.asarray(coordinates_D, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ValueError("Wind reference interpolation requires finite [Q,3] physical coordinates.")
    axes = tuple(np.asarray(axis, dtype=np.float64) for axis in (run.x_m, run.y_m, run.z_m))
    if any(axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0) for axis in axes):
        raise ValueError("Wind native interpolation axes must be strictly increasing vectors.")
    query_m = coordinates * float(diameter_m)
    if any(np.any(query_m[:, axis_index] < axis[0]) or np.any(query_m[:, axis_index] > axis[-1])
           for axis_index, axis in enumerate(axes)):
        raise ValueError("A mapped direction-pair receiver lies outside the native reference support.")

    bracket_indices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for dimension, axis in enumerate(axes):
        upper = np.searchsorted(axis, query_m[:, dimension], side="right")
        upper = np.clip(upper, 1, axis.size - 1)
        lower = upper - 1
        fraction = (query_m[:, dimension] - axis[lower]) / (axis[upper] - axis[lower])
        bracket_indices.append((lower.astype(np.int64), upper.astype(np.int64), fraction))

    q_count = coordinates.shape[0]
    flat_neighbors = np.empty((q_count, 8), dtype=np.int64)
    weights = np.empty((q_count, 8), dtype=np.float64)
    nx, ny = int(run.nx), int(run.ny)
    corner = 0
    for iz_side in (0, 1):
        iz = bracket_indices[2][iz_side]
        wz = bracket_indices[2][2] if iz_side else 1.0 - bracket_indices[2][2]
        for iy_side in (0, 1):
            iy = bracket_indices[1][iy_side]
            wy = bracket_indices[1][2] if iy_side else 1.0 - bracket_indices[1][2]
            for ix_side in (0, 1):
                ix = bracket_indices[0][ix_side]
                wx = bracket_indices[0][2] if ix_side else 1.0 - bracket_indices[0][2]
                flat_neighbors[:, corner] = ix + nx * (iy + ny * iz)
                weights[:, corner] = wx * wy * wz
                corner += 1
    samples = np.asarray(run.U[flat_neighbors.reshape(-1)], dtype=np.float64).reshape(q_count, 8, 3)
    values = np.sum(samples * weights[..., None], axis=1).astype(np.float32)
    return {
        "velocity_mps": values,
        "flat_neighbor_indices": flat_neighbors,
        "interpolation_weights": weights,
    }


def farthest_receiver_indices(coordinates: Any, count: int = 9) -> np.ndarray:
    """Choose a deterministic geometry-only, space-filling receiver subset."""
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < count or not np.isfinite(points).all():
        raise ValueError("Receiver coordinates must be finite [Q,D] with at least the requested count.")
    lower = points.min(axis=0)
    span = points.max(axis=0) - lower
    normalized = (points - lower) / np.where(span > 0.0, span, 1.0)
    lex = np.lexsort(tuple(normalized[:, index] for index in reversed(range(normalized.shape[1]))))
    chosen = [int(lex[0])]
    min_distance_sq = np.sum((normalized - normalized[chosen[0]]) ** 2, axis=1)
    min_distance_sq[chosen[0]] = -1.0
    while len(chosen) < count:
        index = int(np.argmax(min_distance_sq))
        chosen.append(index)
        distance_sq = np.sum((normalized - normalized[index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[np.asarray(chosen, dtype=np.int64)] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def split_native_receiver_interface(
    coordinates: Any,
    prediction: Any,
    reference: Any,
    *,
    observed_count: int = 6,
    held_count: int = 3,
) -> dict[str, np.ndarray]:
    """Expose a reusable six-observed/three-held native receiver view.

    Selection depends only on physical receiver coordinates. Reference values
    are attached after predictions have already been produced and are never
    passed to the model.
    """
    coords = np.asarray(coordinates)
    predicted = np.asarray(prediction)
    truth = np.asarray(reference)
    total = int(observed_count) + int(held_count)
    if coords.ndim != 2 or predicted.shape[0] != coords.shape[0] or truth.shape[0] != coords.shape[0]:
        raise ValueError("Native receiver coordinates, predictions and references must share Q rows.")
    if observed_count < 1 or held_count < 1 or total > coords.shape[0]:
        raise ValueError("The observed/held native receiver split must be nonempty and fit within Q.")
    indices = farthest_receiver_indices(coords, total)
    observed = indices[:observed_count]
    held = indices[observed_count:]
    return {
        "observed_indices": observed,
        "held_indices": held,
        "observed_coordinates": coords[observed].copy(),
        "held_coordinates": coords[held].copy(),
        "observed_prediction": predicted[observed].copy(),
        "held_prediction": predicted[held].copy(),
        "observed_reference": truth[observed].copy(),
        "held_reference": truth[held].copy(),
    }


def parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    """Parse ``LABEL=PATH`` without inferring epoch from the filename."""
    if "=" not in value:
        raise ValueError("Each checkpoint must be written LABEL=PATH.")
    label, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", label):
        raise ValueError(f"Unsafe checkpoint label {label!r}.")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return label, path


def _expected_literal_age(label: str) -> int | None:
    normalized = label.lower().replace("_", "").replace("-", "")
    if normalized.endswith(("literal1000", "epoch1000")):
        return 1000
    if normalized.endswith(("literal2500", "epoch2500")):
        return 2500
    return None


def _load_task(task: str, device: str):
    _add_import_paths()
    from unified_interaction_train import _factory, _model_state_sha256, _profile

    from honf_runtime.reproducibility import seed_all

    seed = 0 if task == "thermal" else 42
    seed_all(seed)
    model, provider, _ = _factory(task)({"seed": seed, "device": device})
    engine_config, selection, _ = _profile(task, seed)
    initial_state_sha256 = _model_state_sha256(model)
    model.to(device)
    model.eval()
    return model, provider, engine_config, selection, initial_state_sha256


def validate_checkpoint_binding(
    payload: Mapping[str, Any],
    *,
    task: str,
    provider_identity: Mapping[str, Any],
    initial_state_sha256: str,
    engine_config: Any,
    selection: Any,
    label: str,
) -> dict[str, Any]:
    """Reject wrong task/profile/provider or a misleading literal checkpoint."""
    if payload.get("workflow") != "unified_interaction_refinement":
        raise ValueError("Checkpoint is not from the maintained unified interaction workflow.")
    if payload.get("checkpoint_schema_version") != 1:
        raise ValueError("Unsupported unified-interaction checkpoint schema.")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("Checkpoint must carry a nonnegative literal integer epoch.")
    if payload.get("current_epoch") != epoch:
        raise ValueError("Checkpoint current_epoch differs from its saved literal epoch.")
    expected_age = _expected_literal_age(label)
    if expected_age is not None and epoch != expected_age:
        raise ValueError(f"Checkpoint label {label!r} requires literal payload age {expected_age}, got {epoch}.")
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Checkpoint omitted its sealed experiment identity mapping.")
    expected_task = "ThermalChannel" if task == "thermal" else "WindFarm"
    expected_profile = "fixed25_v1" if task == "thermal" else "fixed24_v1"
    expected_seed = 0 if task == "thermal" else 42
    checks = {
        "workflow": "unified_interaction_refinement",
        "task": expected_task,
        "development_profile": expected_profile,
        "seed": expected_seed,
        "initial_model_state_sha256": initial_state_sha256,
        "engine_profile": "warmup500_open600_soft800_total2500",
    }
    for key, expected in checks.items():
        if identity.get(key) != expected:
            raise ValueError(f"Checkpoint experiment identity mismatch for {key!r}.")
    saved_provider_identity = identity.get("provider_identity")
    if not isinstance(saved_provider_identity, Mapping):
        raise TypeError("Checkpoint provider identity must be a mapping.")
    saved_provider_identity = dict(saved_provider_identity)
    active_provider_identity = dict(provider_identity)
    saved_device = saved_provider_identity.pop("device", None)
    evaluation_device = active_provider_identity.pop("device", None)
    if saved_provider_identity != active_provider_identity:
        raise ValueError("Checkpoint dataset membership, source, normalization or provider identity differs.")
    if identity.get("engine_config") != engine_config.__dict__:
        raise ValueError("Checkpoint shared-engine configuration differs from the sealed development profile.")
    if identity.get("selection_policy") != selection.__dict__:
        raise ValueError("Checkpoint selector differs from the sealed development profile.")
    if not isinstance(identity.get("optimizer_schedule_contract"), list):
        raise TypeError("Checkpoint optimizer schedule contract must be a list.")
    arm = payload.get("arm")
    if arm not in {"warmup", "full_detail", "adaptive_detail"}:
        raise ValueError(f"Unknown unified-interaction checkpoint arm {arm!r}.")
    return {
        "label": label,
        "epoch": epoch,
        "requested_age": expected_age,
        "age_semantics": "literal checkpoint payload epoch; selected aliases are reported separately",
        "arm": arm,
        "run_id": identity.get("run_id"),
        "saved_training_device": saved_device,
        "evaluation_provider_device": evaluation_device,
    }


def _load_training_validation_cache(
    payload: Mapping[str, Any],
    *,
    task: str,
    provider: Any,
    checkpoint_path: Path,
    runs_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Reuse a monitor receipt only from the exact bound run, arm and epoch."""
    identity = payload.get("experiment_identity")
    if not isinstance(identity, Mapping):
        return None
    run_id = identity.get("run_id")
    arm = payload.get("arm")
    epoch = payload.get("epoch")
    if not isinstance(run_id, str) or not run_id or not isinstance(arm, str) or not isinstance(epoch, int):
        return None
    expected_directory = (runs_root / task / run_id / arm).resolve()
    resolved_checkpoint = checkpoint_path.resolve()
    if resolved_checkpoint.parent != expected_directory or not resolved_checkpoint.is_file():
        return None
    cache_path = expected_directory / f"validation_epoch_{epoch:04d}.json"
    if not cache_path.is_file() or cache_path.stat().st_mtime_ns < resolved_checkpoint.stat().st_mtime_ns:
        return None
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(cached, dict):
        raise TypeError(f"Monitor validation receipt is not an object: {cache_path}")
    expected_count = 22 if task == "thermal" else 24
    if cached.get("validation_execution_mode") != arm or cached.get("validation_route_phase") != "hard":
        return None
    if task == "thermal":
        expected_ids = [str(case["case_id"]) for case in provider.validation_cases]
        if cached.get("case_count") != expected_count or cached.get("case_ids") != expected_ids:
            return None
    else:
        expected_rows = np.asarray(provider.validation_rows, dtype=np.int64).reshape(-1).tolist()
        rows = cached.get("rows")
        if (
            cached.get("row_count") != expected_count
            or not isinstance(rows, list)
            or [int(row.get("row_index", -1)) for row in rows] != expected_rows
        ):
            return None
    matching_history = [
        row for row in payload.get("history", [])
        if isinstance(row, Mapping) and int(row.get("epoch", -1)) == epoch
    ]
    if len(matching_history) != 1 or matching_history[0].get("validation") != cached:
        return None
    return cached, {
        "source": "same-run monitoring validation JSON",
        "cache_path": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "literal_epoch": epoch,
        "arm": arm,
        "run_id": run_id,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_sha256": _sha256(resolved_checkpoint),
        "membership_rows": expected_count,
        "matching": "exact run directory, arm, literal monitor epoch, hard route, and ordered DEV membership",
        "checkpoint_embeds_exact_validation_metrics": True,
    }


def _extract_refinement_arrays(task: str, predictions: Any) -> dict[str, np.ndarray]:
    names = ("base", "fine", "probability", "keep", "protected")
    if task == "thermal":
        response = predictions.native_prepared.response
        auxiliary = getattr(response, "refinement_aux", {})
    else:
        auxiliary = predictions.auxiliary
    if not isinstance(auxiliary, Mapping):
        return {}
    result = {name: _as_numpy(auxiliary[name]) for name in names if name in auxiliary}
    if "complete_fine_values" in auxiliary:
        result["complete_fine_values"] = np.asarray([bool(auxiliary["complete_fine_values"])])
    return result


def _collect_validation(
    task: str,
    model: Any,
    provider: Any,
    *,
    mode: str,
    threshold: float,
    epoch: int,
    temperature: float,
) -> dict[str, Any]:
    import torch

    batches = list(provider.validation_batches())
    if not batches:
        raise ValueError("The sealed development provider returned no validation batches.")
    records: list[Mapping[str, Any]] = []
    work_totals: dict[str, int] = {}
    inference_seconds = 0.0
    post_forward_seconds = 0.0
    case_ids: list[str] = []
    tail_values: dict[str, list[np.ndarray]] = {}
    thermal_region_rows: list[dict[str, Any]] = []
    thermal_fluid_data_boundary = {
        "sampled_case_rows": 0,
        "sampled_query_rows": 0,
        "positive_point_weight_rows": 0,
        "nonpositive_point_weight_rows": 0,
        "nonfinite_point_weight_rows": 0,
        "finite_temperature_target_rows": 0,
        "valid_native_fluid_stencil_rows": 0,
        "eligible_residual_tail_rows": 0,
        "observed_finite_point_weight_min": None,
        "observed_finite_point_weight_max": None,
    }
    route_mode = "adaptive_detail" if mode == "adaptive" else ("full_detail" if mode == "all_fine" else mode)
    for batch in batches:
        # The provider receives only its whitelisted scene_inputs and receiver
        # geometry. Native supervision is touched only after this call returns.
        _synchronize(batch.receivers.coordinates_D if task == "wind" else batch.receivers.fluid_xy)
        inference_started = time.perf_counter()
        scene = provider.make_scene(batch.scene_inputs)
        with torch.no_grad():
            predictions, auxiliary = provider.predict_native(
                model,
                scene,
                batch.receivers,
                route_mode,
                "hard",
                epoch=epoch,
                temperature=temperature,
                **({"threshold": threshold} if task == "thermal" else {}),
            )
        _synchronize(batch.receivers.coordinates_D if task == "wind" else batch.receivers.fluid_xy)
        inference_seconds += time.perf_counter() - inference_started
        post_forward_started = time.perf_counter()
        records.append(provider.validation_metrics(predictions, batch.targets, auxiliary))
        if task == "thermal":
            case_ids.extend(str(value) for value in batch.targets.case_ids)
            work = predictions.work
            prepared = predictions.native_prepared
            native = predictions.native_main
            point_weight_values = _as_numpy(batch.targets.point_weights)
            finite_point_weights = np.isfinite(point_weight_values)
            positive_point_weights = finite_point_weights & (point_weight_values > 0)
            point_weights = positive_point_weights
            fluid_valid = _as_numpy(prepared.stencils["fluid"].valid).astype(bool)
            fluid_target = _as_numpy(batch.targets.field_targets[..., 4])
            fluid_prediction = _as_numpy(native["fluid_temperature"][..., 0])
            finite_temperature_target = np.isfinite(fluid_target)
            fluid_mask = fluid_valid & point_weights & finite_temperature_target
            fluid_residual = fluid_prediction - fluid_target
            tail_values.setdefault("fluid_temperature", []).append(fluid_residual[fluid_mask].reshape(-1))
            thermal_fluid_data_boundary["sampled_case_rows"] += len(batch.targets.case_ids)
            thermal_fluid_data_boundary["sampled_query_rows"] += int(fluid_target.size)
            thermal_fluid_data_boundary["positive_point_weight_rows"] += int(positive_point_weights.sum())
            thermal_fluid_data_boundary["nonpositive_point_weight_rows"] += int(
                (finite_point_weights & ~positive_point_weights).sum()
            )
            thermal_fluid_data_boundary["nonfinite_point_weight_rows"] += int((~finite_point_weights).sum())
            thermal_fluid_data_boundary["finite_temperature_target_rows"] += int(finite_temperature_target.sum())
            thermal_fluid_data_boundary["valid_native_fluid_stencil_rows"] += int(fluid_valid.sum())
            thermal_fluid_data_boundary["eligible_residual_tail_rows"] += int(fluid_mask.sum())
            finite_weight_values = point_weight_values[finite_point_weights]
            if finite_weight_values.size:
                weight_min = float(finite_weight_values.min())
                weight_max = float(finite_weight_values.max())
                current_min = thermal_fluid_data_boundary["observed_finite_point_weight_min"]
                current_max = thermal_fluid_data_boundary["observed_finite_point_weight_max"]
                thermal_fluid_data_boundary["observed_finite_point_weight_min"] = (
                    weight_min if current_min is None else min(current_min, weight_min)
                )
                thermal_fluid_data_boundary["observed_finite_point_weight_max"] = (
                    weight_max if current_max is None else max(current_max, weight_max)
                )
            raw_structure = (
                batch.scene_inputs.structure
                if hasattr(batch.scene_inputs, "structure")
                else batch.scene_inputs.get("structure") if isinstance(batch.scene_inputs, Mapping) else None
            )
            if not isinstance(raw_structure, Mapping):
                raise TypeError("Thermal validation scene inputs must expose the whitelisted physical structure.")
            centers = _as_numpy(raw_structure["module_centers"])
            source_present = _as_numpy(prepared.source_present) > 0.5
            material = _as_numpy(raw_structure["material_params"])
            inlet_speed = _as_numpy(raw_structure["u_in"])
            coordinates = _as_numpy(batch.receivers.fluid_xy)
            if centers.ndim == 2:
                centers = centers[None]
            if material.ndim == 1:
                material = material[None]
            if inlet_speed.ndim == 0:
                inlet_speed = inlet_speed.reshape(1, 1)
            if inlet_speed.ndim == 1:
                inlet_speed = inlet_speed[:, None]
            if coordinates.ndim == 2:
                coordinates = coordinates[None]
            if centers.shape[0] != fluid_residual.shape[0] or material.shape[-1] <= 5:
                raise ValueError("Thermal geometry metadata is not aligned with its validation batch or radius field.")
            for batch_index, case_id in enumerate(batch.targets.case_ids):
                masks = _thermal_geometry_region_masks(
                    coordinates[batch_index],
                    centers[batch_index],
                    source_present[batch_index],
                    float(material[batch_index, 5]),
                    float(inlet_speed[batch_index, 0]),
                    fluid_mask[batch_index],
                )
                case_region_row: dict[str, Any] = {"case_id": str(case_id)}
                for region in ("near", "downstream"):
                    values = fluid_residual[batch_index][masks[region]]
                    summary = _tail_summary(values)
                    tail_values.setdefault(f"{region}/temperature", []).append(values.reshape(-1))
                    case_region_row[f"{region}_temperature"] = summary
                thermal_region_rows.append(case_region_row)
            present = _as_numpy(prepared.source_present) > 0.5
            surface_target = _as_numpy(batch.targets.interface_target[..., 0])
            surface_prediction = _as_numpy(native["pred_interface"][..., 0])
            surface_valid = _as_numpy(prepared.stencils["surface"].valid).reshape(surface_target.shape).astype(bool)
            surface_mask = present[:, :, None] & surface_valid & np.isfinite(surface_target)
            tail_values.setdefault("interface", []).append(
                (surface_prediction - surface_target)[surface_mask].reshape(-1)
            )
            material_target = _as_numpy(batch.targets.material_targets)
            material_prediction = _as_numpy(native["pred_internal_temperature"][..., 0])
            material_valid = _as_numpy(prepared.stencils["material"].valid).reshape(material_target.shape).astype(bool)
            material_mask = present[:, :, None] & material_valid & np.isfinite(material_target)
            tail_values.setdefault("material_temperature", []).append(
                (material_prediction - material_target)[material_mask].reshape(-1)
            )
        else:
            case_ids.extend(str(item["case"]) for item in batch.targets.case_metadata)
            work = predictions.auxiliary
            from windfarm.training.unified_task import _ROLE_INDEX, ROLE_NAMES

            residual = _as_numpy(predictions.main_mps) - _as_numpy(batch.targets.velocity_mps)
            role_ids = _as_numpy(predictions.role_ids)
            for role in ROLE_NAMES:
                selected = role_ids == _ROLE_INDEX[role]
                for component in range(3):
                    tail_values.setdefault(f"{role}/component{component}", []).append(
                        residual[..., component][selected].reshape(-1)
                    )
                tail_values.setdefault(f"{role}/vector_magnitude", []).append(
                    np.linalg.norm(residual, axis=-1)[selected].reshape(-1)
                )
        for name in ("fine_rows", "cheap_rows", "gate_rows", "near_rows", "selected_detail_rows", "active_pairs"):
            value = work.get(name, 0)
            if isinstance(value, (int, float)):
                work_totals[name] = work_totals.get(name, 0) + int(value)
        post_forward_seconds += time.perf_counter() - post_forward_started
    metrics = dict(provider.reduce_native_metrics(records))
    expected_count = 22 if task == "thermal" else 24
    if len(case_ids) != expected_count or len(set(case_ids)) != expected_count:
        raise ValueError(f"Expected the exact DEV{expected_count} panel once, got {len(case_ids)} rows.")
    tails: dict[str, Any] = {}
    if task == "thermal":
        for name, parts in tail_values.items():
            tails[name] = _tail_summary(np.concatenate(parts))
    else:
        from windfarm.training.unified_task import _ROLE_INDEX, ROLE_NAMES

        for role in ROLE_NAMES:
            tails[role] = {
                "components": [_tail_summary(np.concatenate(tail_values[f"{role}/component{index}"]))
                               for index in range(3)],
                "vector_magnitude": _tail_summary(np.concatenate(tail_values[f"{role}/vector_magnitude"])),
            }
    metrics["residual_tails"] = tails
    if task == "thermal":
        metrics["residual_tail_definition"] = {
            "aggregation": "unweighted point-pooled residuals across the provider's validation samples",
            "fluid_temperature": {
                "eligibility": "valid native fluid stencil, positive point weight, and finite temperature target",
                "point_weight_application": "positive weights are an inclusion mask only; weight magnitudes do not weight residual tails",
                "companion_primary_metric": "point-weighted fluid MSE per case, then equal-case DEV22 aggregation",
            },
            "interface_and_material_temperature": (
                "unweighted pooled residuals over valid active-source rows"
            ),
        }
        metrics["data_boundary"] = {
            "partition": "fixed25_v1 exposed DEV22",
            "thermal_fluid_validation_sample": {
                **thermal_fluid_data_boundary,
                "point_weight_magnitudes_applied_to_residual_tails": False,
            },
        }
        metrics["near_downstream_diagnostics"] = {
            "definition": {
                "near": "valid native fluid receivers within <=2r of any active module center",
                "downstream": "valid native fluid receivers with signed dx in (2r,6r], |dy|<=2r from any active center, excluding near",
                "flow_direction": "sign of prescribed input u_in along native +x; u_in=0 gives an empty downstream region",
                "radius": "material_params[...,5] in packed dataset-native geometry units",
                "eligibility": "native fluid stencil valid, positive point weight, finite temperature reference",
                "error_unit": "packed_dataset_native_temperature",
            },
            "per_case": thermal_region_rows,
            "equal_case": {
                region: _equal_case_region_summary(thermal_region_rows, f"{region}_temperature")
                for region in ("near", "downstream")
            },
        }
    metrics["route_work"] = work_totals
    metrics["deployed_route"] = {"mode": mode, "threshold": threshold, "temperature": temperature}
    metrics["measured_scope"] = {
        "inference_batches": len(batches),
        "target_free_scene_and_model_seconds": inference_seconds,
        "includes_cuda_synchronization": True,
        "includes_model_output_host_transfer": False,
        "post_forward_target_metrics_and_host_transfer_seconds": post_forward_seconds,
        "artifact_write_excluded": True,
    }
    return metrics


def _receiver_interface_arrays(
    coordinates: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, np.ndarray]:
    return split_native_receiver_interface(coordinates, prediction, reference)


def _thermal_context_tensors(structure: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Batch and tensorize only the native Thermal input whitelist."""
    import torch
    from channelthermal.source_response import CONTEXT_KEYS

    result: dict[str, Any] = {}
    for name in CONTEXT_KEYS:
        if name not in structure:
            continue
        dtype = torch.long if name == "module_source_ids" else torch.float32
        value = torch.as_tensor(structure[name], dtype=dtype, device=device)
        expected_rank = {"module_centers": 2, "module_present": 1, "module_source_ids": 1}.get(name)
        if expected_rank is not None and value.ndim == expected_rank:
            value = value.unsqueeze(0)
        result[name] = value
    return result


def _thermal_representative(
    provider: Any,
    case: Mapping[str, Any],
    *,
    model: Any,
    route: str,
    threshold: float,
    epoch: int,
    temperature: float,
) -> tuple[dict[str, np.ndarray], Any, Any, Any]:
    import torch
    from channelthermal.training.unified_task import ThermalReceivers, ThermalSceneInputs

    context_structure = _thermal_context_tensors(case["structure"], provider.device)
    scene = provider.make_scene(ThermalSceneInputs(context_structure))
    receivers = ThermalReceivers(
        fluid_xy=torch.as_tensor(np.asarray(case["query_xy"], dtype=np.float32)[None], device=provider.device),
        local_query_points=torch.as_tensor(
            np.asarray(case["module_internal_query_points"], dtype=np.float32)[None], device=provider.device
        ),
        heat=torch.as_tensor(
            np.asarray(case["structure"]["heat_powers"], dtype=np.float32).reshape(1, -1),
            device=provider.device,
        ),
    )
    mode = "adaptive_detail" if route == "adaptive" else ("full_detail" if route == "all_fine" else route)
    with torch.no_grad():
        predictions, _auxiliary = provider.predict_native(
            model, scene, receivers, mode, "hard", epoch=epoch, temperature=temperature, threshold=threshold
        )
    result = {name: _as_numpy(value) for name, value in predictions.native_main.items() if torch.is_tensor(value)}
    result.update({f"route/{name}": value for name, value in _extract_refinement_arrays("thermal", predictions).items()})
    result["query_xy"] = _as_numpy(receivers.fluid_xy)
    result["physical_heat"] = _as_numpy(receivers.heat)
    result["source_present"] = _as_numpy(predictions.native_prepared.source_present)
    result["source_ids"] = _as_numpy(predictions.native_prepared.context.source_ids)
    physical_source_ids = case["structure"].get(
        "module_source_ids", np.arange(result["source_ids"].shape[-1], dtype=np.int64)
    )
    result["source_identity_catalogue"] = (
        _as_numpy(physical_source_ids) if hasattr(physical_source_ids, "detach")
        else np.asarray(physical_source_ids).copy()
    )
    result["source_centers"] = _as_numpy(predictions.native_prepared.context.centers)
    result["source_lengths"] = _as_numpy(predictions.native_prepared.context.source_lengths)
    result["source_states"] = _as_numpy(predictions.native_prepared.context.source_states)
    result["global_state"] = _as_numpy(predictions.native_prepared.context.global_state)
    result["material_params"] = _as_numpy(scene.structure["material_params"])
    result["module_radius_native"] = result["material_params"][..., 5:6].copy()
    if predictions.native_prepared.context.environment_states is not None:
        result["environment_states"] = _as_numpy(predictions.native_prepared.context.environment_states)
    result["environment_coords"] = _as_numpy(predictions.native_prepared.context.environment_coords)
    result["environment_measures"] = _as_numpy(predictions.native_prepared.context.environment_measure)
    result["temperature_target"] = np.asarray(case["field_targets"][..., 4:5], dtype=np.float32)[None]
    result["interface_target"] = np.asarray(case["interface_target"], dtype=np.float32)[None]
    result["material_target"] = np.asarray(case["module_internal_temperature_points"], dtype=np.float32)[None, ..., None]
    result["route_work_json"] = np.asarray(json.dumps(predictions.work, sort_keys=True))
    result["dependency_json"] = np.asarray(json.dumps(
        _dependency_payload(predictions.native_prepared.context.dependency), sort_keys=True, default=_json_default
    ))
    return result, predictions, receivers, scene


def _thermal_precise_heat_demo(
    model: Any,
    provider: Any,
    scene: Any,
    receivers: Any,
    *,
    route: str,
    threshold: float,
    temperature: float,
) -> dict[str, Any]:
    import torch

    # Rebuild outside inference_mode so the physical heat input remains a live
    # autograd leaf. The route is fixed before this heat-only affine probe.
    provider._set_execution(model, route, "hard", temperature, False, threshold=threshold)
    prepared = model.prepare_native(
        scene.structure,
        receivers.fluid_xy,
        local_query_points=receivers.local_query_points,
        ntheta=16,
    )
    heat = receivers.heat.detach().to(dtype=torch.float64)
    active = prepared.source_present.to(dtype=torch.bool)
    active_heat = heat.masked_select(active)
    if active_heat.numel() == 0:
        raise ValueError("Thermal representative has no active physical heat sources.")
    scale = active_heat.abs().mean()
    delta = torch.zeros_like(heat)
    active_count = int(active.sum().item())
    first_active = int(torch.nonzero(active[0], as_tuple=False).flatten()[0].item())
    delta[0, first_active] = scale * 0.01
    if not bool(torch.isfinite(delta).all()) or float(delta.abs().sum()) == 0:
        raise ValueError("Thermal model-only perturbation must be finite and nonzero.")
    kernels_before = model.export_native_kernels(prepared, accumulation_dtype=torch.float64)
    baseline = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)
    increment = model.apply_native(prepared, delta, increment=True, accumulation_dtype=torch.float64)
    endpoint = model.apply_native(prepared, heat + delta, accumulation_dtype=torch.float64)
    zero_heat = model.apply_native(prepared, torch.zeros_like(heat), accumulation_dtype=torch.float64)
    error = endpoint["fluid_temperature"] - baseline["fluid_temperature"] - increment["fluid_temperature"]
    query_index = 0
    heat_leaf = heat.detach().clone().requires_grad_(True)
    differentiable = model.apply_native(prepared, heat_leaf, accumulation_dtype=torch.float64)
    selected = differentiable["fluid_temperature"][0, query_index, 0]
    gradient = torch.autograd.grad(selected, heat_leaf)[0]
    kernels_after = model.export_native_kernels(prepared, accumulation_dtype=torch.float64)
    expected = kernels_after["fluid"][0, query_index, :, 0]
    kernel_heat_independence_error = max(
        float((kernels_before[name] - kernels_after[name]).abs().max().detach().cpu())
        for name in kernels_before
    )
    thermal_adapter = model.thermal if hasattr(model, "thermal") else model
    environment_flow_context = bool(getattr(thermal_adapter, "environment_flow_context", False))
    return {
        "semantics": "fixed affine route; packed native heating-rate inputs and packed native temperature output",
        "units": {"heat": "packed_dataset_native_heating_rate", "temperature": "packed_dataset_native_temperature"},
        "action": f"model-only +1% of the active-source mean absolute heat at source slot {first_active}; other slots unchanged",
        "active_source_count": active_count,
        "precise_accumulation_dtype": "float64",
        "increment_max_abs_closure_error": float(error.abs().max().detach().cpu()),
        "increment_rmse_closure_error": float(error.square().mean().sqrt().detach().cpu()),
        "heat_vjp_query_index": query_index,
        "heat_vjp": _as_numpy(gradient).reshape(-1).tolist(),
        "direct_kernel_at_receiver": _as_numpy(expected).reshape(-1).tolist(),
        "heat_vjp_kernel_max_abs_error": float((gradient[0] - expected).abs().max().detach().cpu()),
        "zero_heat_temperature_max_abs": float(zero_heat["fluid_temperature"].abs().max().detach().cpu()),
        "prepared_kernel_heat_independence_max_abs_error": kernel_heat_independence_error,
        "environment_flow_context_enabled": environment_flow_context,
        "predicted_flow_output_or_target_entered_heat_application": False,
        "input_output_arrays": {
            "delta_heat": _as_numpy(delta).reshape(-1).tolist(),
            "baseline_fluid_temperature": _as_numpy(baseline["fluid_temperature"]),
            "increment_fluid_temperature": _as_numpy(increment["fluid_temperature"]),
            "endpoint_fluid_temperature": _as_numpy(endpoint["fluid_temperature"]),
        },
    }


def _thermal_effective_all_base_far(base_far: Any, fine_far: Any, protected: Any):
    """Use the retained fine path on protected near pairs in an all-base route."""
    import torch

    if base_far.shape != fine_far.shape or protected.shape != base_far.shape[:-1]:
        raise ValueError("Thermal base/fine/protected arrays must share [B,Q,M] source order.")
    return torch.where(protected[..., None].bool(), fine_far, base_far)


def _thermal_complete_fine_export(
    model: Any,
    provider: Any,
    scene: Any,
    receivers: Any,
    *,
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pay once for and export actual Thermal all-source fine coefficients.

    This diagnostic export is deliberately outside the ordinary inference
    timing and runs with the model in evaluation mode and gradients disabled.
    The ``training_signal`` switch requests complete source-level F arrays; it
    does not enable the organizer surrogate for an all-fine policy.
    """
    import torch

    model.eval()
    provider._set_execution(model, "all_fine", "hard", temperature, True, threshold=0.5)
    model.core.reset_auxiliary()
    with torch.no_grad():
        prepared = model.prepare_native(
            scene.structure,
            receivers.fluid_xy,
            local_query_points=receivers.local_query_points,
            ntheta=16,
        )
        physical = model.apply_native(prepared, receivers.heat)
        thermal_prepared = prepared.get("thermal", prepared) if isinstance(prepared, Mapping) else prepared
        response = thermal_prepared.response
        auxiliary = getattr(response, "refinement_aux", {})
        if not isinstance(auxiliary, Mapping) or not auxiliary.get("complete_fine_values", False):
            raise RuntimeError("Thermal all-fine export did not pay for complete source coefficients.")
        thermal_model = model.thermal if hasattr(model, "thermal") else model
        kernels = thermal_model.export_native_kernels(thermal_prepared, accumulation_dtype=torch.float64)
        base_far = auxiliary["base"].to(dtype=torch.float64)
        fine_far = auxiliary["fine"].to(dtype=torch.float64)
        effective_base_far = _thermal_effective_all_base_far(
            base_far, fine_far, auxiliary["protected"]
        )
        near_weight = response.near_weight.to(dtype=torch.float64)
        base_neural_kernel = effective_base_far * (1.0 - near_weight[..., None])
        if response.near_indices.numel():
            batch_index, query_index, source_index = response.near_indices
            correction = (
                near_weight[batch_index, query_index, source_index, None]
                * response.near_values.to(dtype=torch.float64)
            )
            base_neural_kernel = base_neural_kernel.index_put(
                (batch_index, query_index, source_index),
                base_neural_kernel[batch_index, query_index, source_index] + correction,
            )
        base_neural_kernel = base_neural_kernel / float(response.forcing_scale)
        base_native_kernel = thermal_model._interpolate(
            base_neural_kernel, thermal_prepared.stencils["fluid"]
        )
        fine_native_kernel = kernels["fluid"]
        heat = receivers.heat.to(dtype=torch.float64)[:, None, :, None]
        delta_by_source = (fine_native_kernel - base_native_kernel) * heat
        neural_delta_by_source = (
            (fine_far - effective_base_far)
            * (1.0 - near_weight[..., None])
            * heat
            / float(response.forcing_scale)
        )
        delta_from_source_sum = delta_by_source.sum(dim=2)
        arrays: dict[str, np.ndarray] = {
            "neural_receiver_xy": _as_numpy(response.receivers),
            "native_grid_indices": _as_numpy(thermal_prepared.grid_indices),
            "source_ids": _as_numpy(thermal_prepared.context.source_ids),
            "source_present": _as_numpy(thermal_prepared.source_present),
            "fine_far_coefficients": _as_numpy(auxiliary["fine"]),
            "base_far_coefficients": _as_numpy(auxiliary["base"]),
            "router_probability": _as_numpy(auxiliary["probability"]),
            "all_fine_keep": _as_numpy(auxiliary["keep"]),
            "near_protected": _as_numpy(auxiliary["protected"]),
            "complete_fine_fluid_kernel": _as_numpy(kernels["fluid"]),
            "complete_base_fluid_kernel": _as_numpy(base_native_kernel),
            "fine_minus_base_by_source_times_heat": _as_numpy(delta_by_source),
            "fine_minus_base_neural_by_source_times_heat": _as_numpy(neural_delta_by_source),
            "fine_minus_base_heat_sum": _as_numpy(delta_from_source_sum),
            "all_fine_fluid_temperature": _as_numpy(physical["fluid_temperature"]),
        }
        fine_rows = int(auxiliary.get(
            "fine_rows", arrays["fine_far_coefficients"].shape[0]
            * arrays["fine_far_coefficients"].shape[1]
            * arrays["fine_far_coefficients"].shape[2],
        ))
        active_pairs = int(auxiliary.get("active_pairs", 0))
        neural_rows = int(response.receivers.shape[1])
    model.core.reset_auxiliary()
    return arrays, {
        "mode": "all_fine",
        "phase": "hard",
        "training_signal": True,
        "organizer_surrogate": False,
        "model_training": bool(model.training),
        "complete_fine_values": True,
        "fine_rows_paid": fine_rows,
        "active_source_receiver_pairs": active_pairs,
        "neural_receiver_rows": neural_rows,
        "native_fluid_receiver_rows": int(receivers.fluid_xy.shape[1]),
        "base_route_source_semantics": (
            "protected near pairs retain the original fine far path and exact near term; "
            "only unprotected pairs substitute the learned base far coefficient"
        ),
        "scope": "separate diagnostic coefficient export; excluded from normal-route inference timing",
    }


def _thermal_geometry_ad_fd_demo(
    model: Any,
    provider: Any,
    scene: Any,
    heat: Any,
    neural_coordinates: np.ndarray,
    native_grid_indices: np.ndarray,
    adaptive_route: Mapping[str, np.ndarray],
    *,
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compare one fixed-route Thermal geometry VJP with rebuilt central FD."""
    import torch

    probability = np.asarray(adaptive_route["route/probability"], dtype=np.float64)
    protected = np.asarray(adaptive_route["route/protected"], dtype=bool)
    keep = np.asarray(adaptive_route["route/keep"], dtype=bool)
    present = np.asarray(adaptive_route["source_present"], dtype=bool)
    if probability.ndim != 3 or probability.shape[1] != neural_coordinates.shape[1]:
        raise ValueError("Thermal native geometry check needs receiver-aligned .5 route probabilities.")
    eligible = present[:, None, :] & ~protected
    margin = np.where(eligible, np.abs(probability - 0.5), np.inf)
    finite_margin = np.isfinite(margin).any(axis=-1)
    if finite_margin.any():
        row_margin = np.where(finite_margin, np.min(margin, axis=-1, initial=np.inf), -np.inf)
        receiver_index = int(np.argmax(row_margin[0]))
        minimum_margin = float(row_margin[0, receiver_index])
    else:
        receiver_index = 0
        minimum_margin = float("inf")
    eligible_selected = eligible[0, receiver_index] & keep[0, receiver_index]
    active_slots = np.flatnonzero(present[0])
    if active_slots.size == 0:
        raise ValueError("Thermal geometry check has no active source.")
    source_slot = int(np.flatnonzero(eligible_selected)[0]) if eligible_selected.any() else int(active_slots[0])
    neural_catalogue = np.asarray(neural_coordinates)
    grid_catalogue = np.asarray(native_grid_indices, dtype=np.int64)
    if neural_catalogue.ndim != 3 or grid_catalogue.shape != neural_catalogue.shape[:2]:
        raise ValueError("Thermal geometry check requires receiver coordinates and native grid IDs in exact row order.")
    selected_grid_id = int(grid_catalogue[0, receiver_index])
    coordinate = torch.as_tensor(
        neural_catalogue[:, receiver_index:receiver_index + 1], dtype=torch.float32,
        device=scene.structure["module_centers"].device,
    )
    provider._set_execution(model, "adaptive_detail", "hard", temperature, False, threshold=0.5)

    def evaluate(center_value: Any, *, differentiate: bool):
        context = torch.enable_grad() if differentiate else torch.no_grad()
        with context:
            structure = dict(scene.structure)
            structure["module_centers"] = center_value
            prepared = model.prepare_native(structure, coordinate, ntheta=16)
            output = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)["fluid_temperature"][0, 0, 0]
            native_prepared = prepared.get("thermal", prepared) if isinstance(prepared, Mapping) else prepared
            auxiliary = getattr(native_prepared.response, "refinement_aux", {})
            if not isinstance(auxiliary, Mapping):
                raise TypeError("Thermal geometry check lost its fixed-route refinement metadata.")
            route_mask = _as_numpy(auxiliary["keep"]).astype(bool)
            protected_mask = _as_numpy(auxiliary["protected"]).astype(bool)
            receiver_ids = _as_numpy(native_prepared.response.receiver_ids).astype(np.int64)
            if receiver_ids.ndim != 2 or route_mask.shape[:2] != receiver_ids.shape:
                raise ValueError("Rebuilt Thermal route masks are not aligned with their native receiver IDs.")
            matching = np.flatnonzero(receiver_ids[0] == selected_grid_id)
            if matching.size != 1:
                raise ValueError(
                    f"Requested native fluid receiver ID {selected_grid_id} maps to {matching.size} rebuilt rows."
                )
            selected_row = int(matching[0])
            if differentiate:
                gradient = torch.autograd.grad(output, center_value, retain_graph=False)[0]
            else:
                gradient = None
        return output, receiver_ids, route_mask, protected_mask, selected_row, gradient

    center_leaf = scene.structure["module_centers"].detach().clone().requires_grad_(True)
    base_value, base_ids, base_keep_all, base_protected_all, base_row, gradient = evaluate(
        center_leaf, differentiate=True
    )
    material_params = scene.structure["material_params"]
    if material_params.ndim == 1:
        material_params = material_params.unsqueeze(0)
    if material_params.ndim != 2 or material_params.shape[0] != 1 or material_params.shape[-1] <= 5:
        raise ValueError("Thermal geometry check expects one native material-parameter row with a source radius.")
    radius = float(material_params[0, 5].detach().abs().cpu())
    center_scale = max(1.0, float(center_leaf[0, source_slot, 0].detach().abs().cpu()))
    step = max(radius * 1.0e-3, 16.0 * torch.finfo(center_leaf.dtype).eps * center_scale)
    axis = 0
    positive = center_leaf.detach().clone()
    negative = center_leaf.detach().clone()
    positive[0, source_slot, axis] += step
    negative[0, source_slot, axis] -= step
    value_plus, ids_plus, keep_plus_all, protected_plus_all, plus_row, _ = evaluate(
        positive, differentiate=False
    )
    value_minus, ids_minus, keep_minus_all, protected_minus_all, minus_row, _ = evaluate(
        negative, differentiate=False
    )
    catalogue_unchanged = np.array_equal(base_ids, ids_plus) and np.array_equal(base_ids, ids_minus)
    unchanged = catalogue_unchanged and (
        np.array_equal(base_keep_all, keep_plus_all)
        and np.array_equal(base_keep_all, keep_minus_all)
        and np.array_equal(base_protected_all, protected_plus_all)
        and np.array_equal(base_protected_all, protected_minus_all)
    )
    base_keep = base_keep_all[:, base_row:base_row + 1]
    base_protected = base_protected_all[:, base_row:base_row + 1]
    keep_plus = keep_plus_all[:, plus_row:plus_row + 1]
    keep_minus = keep_minus_all[:, minus_row:minus_row + 1]
    protected_plus = protected_plus_all[:, plus_row:plus_row + 1]
    protected_minus = protected_minus_all[:, minus_row:minus_row + 1]
    saved_route_matches = np.array_equal(base_keep, keep[:, receiver_index:receiver_index + 1, :])
    autodiff = float(gradient[0, source_slot, axis].detach().cpu())
    finite_difference = float(((value_plus - value_minus) / (2.0 * step)).detach().cpu())
    arrays = {
        "receiver_coordinate": _as_numpy(coordinate),
        "native_grid_id": np.asarray([selected_grid_id], dtype=np.int64),
        "rebuilt_receiver_ids_base": base_ids,
        "rebuilt_receiver_ids_positive": ids_plus,
        "rebuilt_receiver_ids_negative": ids_minus,
        "route_keep_base": base_keep,
        "route_keep_positive": keep_plus,
        "route_keep_negative": keep_minus,
        "route_protected_base": base_protected,
        "route_protected_positive": protected_plus,
        "route_protected_negative": protected_minus,
        "source_center_base": _as_numpy(center_leaf[0, source_slot]),
        "source_center_positive": _as_numpy(positive[0, source_slot]),
        "source_center_negative": _as_numpy(negative[0, source_slot]),
    }
    return arrays, {
        "semantics": "local adaptive hard-route geometry derivative; context is rebuilt for both finite-difference endpoints",
        "wrt": "module center x coordinate in native geometry units",
        "source_slot": source_slot,
        "source_id": _as_numpy(scene.structure["module_source_ids"])[0, source_slot].item()
        if torch.is_tensor(scene.structure["module_source_ids"])
        else str(scene.structure["module_source_ids"][source_slot]),
        "receiver_index_in_native_neural_catalogue": receiver_index,
        "minimum_unprotected_gate_margin_to_threshold": minimum_margin,
        "rebuilt_base_route_matches_saved_route": saved_route_matches,
        "rebuilt_native_receiver_catalogue_unchanged_at_both_endpoints": catalogue_unchanged,
        "rebuilt_full_route_and_near_masks_unchanged_at_both_endpoints": unchanged,
        "step_native_geometry_units": step,
        "base_output_native_temperature": float(base_value.detach().cpu()),
        "positive_output_native_temperature": float(value_plus.detach().cpu()),
        "negative_output_native_temperature": float(value_minus.detach().cpu()),
        "autodiff": autodiff,
        "central_finite_difference": finite_difference,
        "absolute_difference": abs(autodiff - finite_difference),
        "relative_difference": abs(autodiff - finite_difference) / max(abs(autodiff), abs(finite_difference), 1.0e-12),
        "discrete_route_unchanged_at_both_endpoints": unchanged,
        "local_fixed_route_fd_check_valid": bool(saved_route_matches and unchanged),
    }


def _collect_thermal_response_development(
    model: Any,
    provider: Any,
    *,
    arm: str,
    epoch: int,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate only the provider's four existing response-development records."""
    import torch
    from channelthermal.training.unified_task import ThermalSceneInputs

    samples = getattr(provider, "_development_response_inputs", None)
    expected_ids = tuple(
        str(value["family_id"] if isinstance(value, Mapping) else value)
        for value in getattr(provider, "development_families", ())
    )
    if not samples or len(samples) != 4:
        raise ValueError("Thermal provider must expose the four predeclared response-development samples.")
    route_specs = ("all_fine", "adaptive_detail")
    if arm == "warmup":
        # The shared warmup is all-fine by contract; an adaptive route is still
        # an explicit fixed-scene control on these same weights.
        pass
    report: dict[str, Any] = {
        "scope": "four existing response-development families; no new solves or labels",
        "family_ids": [str(sample["family_id"]) for sample in samples],
        "route_modes": list(route_specs),
        "records": [],
    }
    if expected_ids and tuple(report["family_ids"]) != expected_ids:
        raise ValueError("Thermal response-development sample order differs from its provider membership.")
    arrays: dict[str, np.ndarray] = {}
    for sample in samples:
        family_id = str(sample["family_id"])
        structure = _thermal_context_tensors(sample["structure"], provider.device)
        scene = provider.make_scene(ThermalSceneInputs(structure))
        fluid_xy = sample["fluid_xy"].to(provider.device)
        local = sample["local"].to(provider.device)
        heat = sample["heat"].to(provider.device)
        route_outputs: dict[str, dict[str, np.ndarray]] = {}
        route_times: dict[str, float] = {}
        route_work: dict[str, Any] = {}
        for route in route_specs:
            provider._set_execution(model, route, "hard", temperature, False, threshold=0.5)
            model.core.reset_auxiliary()
            _synchronize(fluid_xy)
            started = time.perf_counter()
            with torch.no_grad():
                prepared = model.prepare_native(scene.structure, fluid_xy, local_query_points=local, ntheta=16)
                prediction = model.apply_native(prepared, heat, increment=True)
                output = {
                    "fluid_temperature": _as_numpy(prediction["fluid_temperature"]),
                    "interface": _as_numpy(prediction["pred_interface"][..., :1]),
                    "material_temperature": _as_numpy(prediction["pred_internal_temperature"]),
                }
            _synchronize(fluid_xy)
            route_times[route] = time.perf_counter() - started
            route_outputs[route] = output
            thermal_prepared = prepared["thermal"] if isinstance(prepared, Mapping) else prepared
            response = thermal_prepared.response
            auxiliary = getattr(response, "refinement_aux", {})
            neural_rows = int(thermal_prepared.grid_indices.numel())
            padded_sources = int(thermal_prepared.source_present.shape[1])
            active_sources = int((thermal_prepared.source_present > 0).sum().item())
            route_work[route] = {
                "fine_rows": int(auxiliary.get("fine_rows", neural_rows * padded_sources))
                if isinstance(auxiliary, Mapping) else neural_rows * padded_sources,
                "active_source_receiver_pairs": int(auxiliary.get("active_pairs", neural_rows * active_sources))
                if isinstance(auxiliary, Mapping) else neural_rows * active_sources,
                "near_rows": int(auxiliary.get("near_rows", 0)) if isinstance(auxiliary, Mapping) else 0,
                "selected_detail_rows": int(auxiliary.get("selected_detail_rows", 0))
                if isinstance(auxiliary, Mapping) else neural_rows * active_sources,
                "native_neural_receiver_rows": neural_rows,
            }
        # Response supervision is consumed after both target-free route reads.
        targets = {
            "fluid_temperature": _as_numpy(sample["fluid_target"])[..., None],
            "interface": _as_numpy(sample["surface_target"])[..., None],
            "material_temperature": _as_numpy(sample["material_target"])[..., None],
        }
        masks = {
            "fluid_temperature": np.isfinite(targets["fluid_temperature"][..., 0]),
            "interface": _as_numpy(sample["surface_mask"]).astype(bool),
            "material_temperature": _as_numpy(sample["material_mask"]).astype(bool),
        }
        record: dict[str, Any] = {
            "family_id": family_id,
            "route_seconds": route_times,
            "route_work": route_work,
            "receiver_counts": {
                "fluid": int(fluid_xy.shape[1]),
                "surface_per_source": int(targets["interface"].shape[-2]),
                "material_per_source": int(targets["material_temperature"].shape[-2]),
            },
            "route_metrics": {},
        }
        arrays[f"{family_id}/fluid_ids"] = _as_numpy(sample["fluid_ids"])
        arrays[f"{family_id}/material_ids"] = _as_numpy(sample["material_ids"])
        arrays[f"{family_id}/fluid_coordinates"] = _as_numpy(fluid_xy)
        arrays[f"{family_id}/source_ids"] = _as_numpy(prepared["thermal"].context.source_ids
                                                       if isinstance(prepared, Mapping) else prepared.context.source_ids)
        arrays[f"{family_id}/source_centers"] = _as_numpy(prepared["thermal"].context.centers
                                                           if isinstance(prepared, Mapping) else prepared.context.centers)
        arrays[f"{family_id}/physical_heat_increment"] = _as_numpy(heat)
        for role, target in targets.items():
            arrays[f"{family_id}/target/{role}"] = target
            for route, output in route_outputs.items():
                predicted = output[role]
                arrays[f"{family_id}/{route}/{role}"] = predicted
                residual = predicted - target
                mask = masks[role]
                if residual.ndim > mask.ndim:
                    mask = np.broadcast_to(mask[..., None], residual.shape)
                record["route_metrics"].setdefault(route, {})[role] = _tail_summary(residual[mask])
                arrays[f"{family_id}/{route}/{role}_residual"] = residual
        report["records"].append(record)
    return report, arrays


def _wind_predict_plane(model: Any, scene: Any, coordinates: np.ndarray, *, mode: str,
                        threshold: float, temperature: float) -> tuple[dict[str, Any], dict[str, int]]:
    import torch
    from windfarm.shared_interaction import _wind_receiver_features_from_scene

    query = torch.as_tensor(np.asarray(coordinates, dtype=np.float32)[None], device=scene.centers.device)
    # The provider's public prediction method fixes the shared default
    # threshold. This controlled read uses the same maintained adapter/core
    # route while explicitly setting the predeclared frontier threshold.
    model.core.reset_auxiliary()
    model.core.set_execution(
        mode=mode, phase="hard", threshold=float(threshold),
        temperature=float(temperature), training_signal=False,
    )
    with torch.no_grad():
        context = model.core.prepare(scene)
        features = _wind_receiver_features_from_scene(scene, query)
        prediction = model.core.read_refinement(
            context, query, receiver_features=features, training_signal=False
        )
        auxiliary = dict(prediction.auxiliary)
        auxiliary.update(model.core.auxiliary_terms())
        values = prediction.values
        physical = model._physical_from_standardized(values, query)
        result = {"velocity_mps": _as_numpy(physical), "query_coordinates_D": _as_numpy(query)}
        route_work = {
            name: int(value)
            for name in ("fine_rows", "selected_detail_rows", "near_rows", "cheap_rows", "gate_rows", "active_pairs")
            if isinstance((value := auxiliary.get(name)), (int, float, bool))
        }
        if mode == "adaptive" and math.isclose(float(threshold), 0.5, rel_tol=0.0, abs_tol=1.0e-12):
            for name in ("base", "fine", "probability", "keep", "protected", "complete_fine_values"):
                value = auxiliary.get(name)
                if torch.is_tensor(value):
                    result[f"route/{name}"] = _as_numpy(value)
                elif isinstance(value, (int, float, bool)):
                    result[f"route/{name}"] = np.asarray(value)
    return result, route_work


def _wind_single_correction_removal(
    model: Any,
    scene: Any,
    coordinates: np.ndarray,
    adaptive_route: Mapping[str, np.ndarray],
    *,
    base_velocity_mps: np.ndarray,
    full_velocity_mps: np.ndarray,
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Remove one selected nonprotected source correction at one input receiver."""
    import torch
    from windfarm.shared_interaction import _wind_receiver_features_from_scene

    points = np.asarray(coordinates, dtype=np.float64)
    center = points.mean(axis=0)
    receiver_index = int(np.argmin(np.sum((points - center) ** 2, axis=-1)))
    keep = np.asarray(adaptive_route["route/keep"], dtype=bool)
    probability = np.asarray(adaptive_route["route/probability"], dtype=np.float64)
    protected = np.asarray(adaptive_route["route/protected"], dtype=bool)
    present = _as_numpy(scene.present).astype(bool)
    selected = keep[0, receiver_index] & ~protected[0, receiver_index] & present[0]
    if not selected.any():
        return {"query_coordinate_D": points[receiver_index:receiver_index + 1]}, {
            "status": "no_selected_nonprotected_source_at_fixed_receiver",
            "receiver_index": receiver_index,
            "selected_nonprotected_count": 0,
            "scope": "one fixed-route receiver; no source effect was fabricated",
        }
    selected_slots = np.flatnonzero(selected)
    source_slot = int(selected_slots[np.argmax(probability[0, receiver_index, selected_slots])])
    query = torch.as_tensor(
        points[receiver_index:receiver_index + 1][None], dtype=torch.float32, device=scene.centers.device
    )
    model.eval()
    model.core.reset_auxiliary()
    model.core.set_execution(
        mode="adaptive", phase="hard", threshold=0.5,
        temperature=float(temperature), training_signal=False,
    )
    with torch.no_grad():
        context = model.core.prepare(scene)
        features = _wind_receiver_features_from_scene(scene, query)
        fixed_keep = torch.as_tensor(keep[:, receiver_index:receiver_index + 1],
                                     dtype=torch.bool, device=query.device)
        removed_keep = fixed_keep.clone()
        removed_keep[0, 0, source_slot] = False
        adaptive = model.core.read_refinement(
            context, query, receiver_features=features, training_signal=False, fixed_route=fixed_keep
        )
        removed = model.core.read_refinement(
            context, query, receiver_features=features, training_signal=False, fixed_route=removed_keep
        )
        adaptive_mps = model._physical_from_standardized(adaptive.values, query)
        removed_mps = model._physical_from_standardized(removed.values, query)
        change = adaptive_mps - removed_mps
    arrays = {
        "query_coordinate_D": _as_numpy(query),
        "source_id": _as_numpy(scene.source_ids[:, source_slot]),
        "source_slot": np.asarray([source_slot], dtype=np.int64),
        "source_probability": np.asarray([probability[0, receiver_index, source_slot]], dtype=np.float64),
        "all_base_velocity_mps": np.asarray(base_velocity_mps[receiver_index]).copy(),
        "all_fine_velocity_mps": np.asarray(full_velocity_mps[receiver_index]).copy(),
        "fixed_keep_before": keep[:, receiver_index:receiver_index + 1],
        "fixed_keep_after": _as_numpy(removed_keep),
        "adaptive_corrected_velocity_mps": _as_numpy(adaptive_mps),
        "one_source_correction_removed_velocity_mps": _as_numpy(removed_mps),
        "one_source_correction_physical_change_mps": _as_numpy(change),
    }
    return arrays, {
        "status": "measured",
        "receiver_index": receiver_index,
        "source_slot": source_slot,
        "source_id": _json_default(arrays["source_id"].reshape(-1)[0]),
        "source_probability": float(probability[0, receiver_index, source_slot]),
        "protected": False,
        "selected_nonprotected_count": int(selected.sum()),
        "all_base_velocity_mps": np.asarray(base_velocity_mps[receiver_index]).tolist(),
        "all_fine_velocity_mps": np.asarray(full_velocity_mps[receiver_index]).tolist(),
        "adaptive_corrected_velocity_mps": _as_numpy(adaptive_mps).reshape(-1).tolist(),
        "one_source_correction_removed_velocity_mps": _as_numpy(removed_mps).reshape(-1).tolist(),
        "one_source_correction_physical_change_mps": _as_numpy(change).reshape(-1).tolist(),
        "scope": "one fixed-route receiver; same prepared context, every other route bit held fixed",
    }


def _wind_vjp(model: Any, prepared: Any, receivers: Any, cotangent: Any, *, wrt: str) -> Any:
    import torch
    from windfarm.shared_interaction import _wind_receiver_features_from_scene

    model._assert_prepared_owned(prepared)
    if wrt not in {"centers", "sources", "context", "receivers"}:
        raise ValueError("Wind VJP supports the adapter's physical centers/sources/context/receivers variables.")
    value = receivers if wrt == "receivers" else getattr(prepared.scene, wrt)
    leaf = value.detach().clone().requires_grad_(True)
    scene = prepared.scene if wrt == "receivers" else replace(prepared.scene, **{wrt: leaf})
    query = leaf if wrt == "receivers" else receivers
    context = model.core.prepare(scene)
    features = _wind_receiver_features_from_scene(scene, query)
    prediction = model.core.read_refinement(context, query, receiver_features=features, training_signal=False)
    physical = model._physical_from_standardized(prediction.values, query)
    scalar = (physical * cotangent).sum()
    gradient = torch.autograd.grad(scalar, leaf)[0]
    return gradient


def _wind_derivative_demo(model: Any, prepared: Any, coordinates: np.ndarray) -> dict[str, Any]:
    import torch

    query = torch.as_tensor(np.asarray(coordinates, dtype=np.float32)[None], device=prepared.scene.centers.device)
    tangent = torch.zeros_like(prepared.scene.centers)
    active_slots = torch.nonzero(prepared.scene.present[0] > 0.5, as_tuple=False).flatten()
    if active_slots.numel() == 0:
        raise ValueError("Wind representative has no active turbine source.")
    tangent[0, active_slots[0], 0] = 0.01
    linearized = model.linearize_case(prepared, query, tangent, wrt="centers")
    cotangent = torch.ones_like(linearized["values"]) / math.sqrt(linearized["values"].numel())
    vjp = _wind_vjp(model, prepared, query, cotangent, wrt="centers")
    lhs = (linearized["jvp"] * cotangent).sum()
    rhs = (vjp * tangent).sum()
    return {
        "units": "m/s output per rotor-diameter geometry input; physical denormalization and height profile included",
        "semantics": "local hard-route AD; discrete route membership is fixed piecewise, not differentiated",
        "wrt": "centers",
        "tangent": _as_numpy(tangent),
        "cotangent": _as_numpy(cotangent),
        "jvp": _as_numpy(linearized["jvp"]),
        "vjp": _as_numpy(vjp),
        "jvp_vjp_dot_error": float((lhs - rhs).abs().detach().cpu()),
        "affine_finite_increment": "unsupported; no affine apply_increment was called",
    }


def _wind_complete_fine_export(
    model: Any,
    scene: Any,
    coordinates: np.ndarray,
    *,
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pay once for an actual Wind all-source fine-message export."""
    import torch
    from windfarm.shared_interaction import _wind_receiver_features_from_scene

    model.eval()
    query = torch.as_tensor(np.asarray(coordinates, dtype=np.float32)[None], device=scene.centers.device)
    model.core.reset_auxiliary()
    model.core.set_execution(
        mode="all_fine", phase="hard", threshold=0.5,
        temperature=float(temperature), training_signal=True,
    )
    with torch.no_grad():
        context = model.core.prepare(scene)
        features = _wind_receiver_features_from_scene(scene, query)
        prediction = model.core.read_refinement(
            context, query, receiver_features=features, training_signal=True
        )
        auxiliary = prediction.auxiliary
        if not auxiliary.get("complete_fine_values", False) or prediction.full_values is None:
            raise RuntimeError("Wind all-fine export did not pay for complete source messages.")
        physical = model._physical_from_standardized(prediction.values, query)
        arrays = {
            "query_coordinates_D": _as_numpy(query),
            "source_ids": _as_numpy(context.source_ids),
            "source_present": _as_numpy(context.present),
            "source_measure": _as_numpy(context.source_measure),
            "fine_source_messages": _as_numpy(auxiliary["fine"]),
            "base_source_messages": _as_numpy(auxiliary["base"]),
            "router_probability": _as_numpy(auxiliary["probability"]),
            "all_fine_keep": _as_numpy(auxiliary["keep"]),
            "near_protected": _as_numpy(auxiliary["protected"]),
            "all_fine_standardized_values": _as_numpy(prediction.full_values),
            "all_fine_velocity_mps": _as_numpy(physical),
        }
        fine_rows = int(auxiliary.get("fine_rows", 0))
        active_pairs = int(auxiliary.get("active_pairs", 0))
        receiver_rows = int(query.shape[1])
    model.core.reset_auxiliary()
    return arrays, {
        "mode": "all_fine",
        "phase": "hard",
        "training_signal": True,
        "organizer_surrogate": False,
        "model_training": bool(model.training),
        "complete_fine_values": True,
        "fine_rows_paid": fine_rows,
        "active_source_receiver_pairs": active_pairs,
        "receiver_rows": receiver_rows,
        "scope": "separate diagnostic coefficient export; excluded from normal-route inference timing",
    }


def _wind_partition(provider: Any, row: int) -> dict[str, Any]:
    from windfarm.training.unified_task import DEFAULT_DERIVED_ROOT, _load_original_split

    split = _load_original_split(provider.view, DEFAULT_DERIVED_ROOT)
    if row in set(map(int, split.train.tolist())):
        original = "original_train"
    elif row in set(map(int, split.validation.tolist())):
        original = "original_validation"
    elif row in set(map(int, split.test.tolist())):
        original = "original_test"
    else:
        raise ValueError(f"Wind row {row} is absent from the sealed source split.")
    fixed_train = row in set(map(int, provider.train_rows.tolist()))
    fixed_dev = row in set(map(int, provider.validation_rows.tolist()))
    return {
        "original_split": original,
        "fixed24_v1_membership": "TRAIN" if fixed_train else "DEV" if fixed_dev else "outside_fixed24_v1_selection",
        "development_metrics_population": False,
    }


def _evaluate_wind_same_layout_direction_pair(
    model: Any,
    provider: Any,
    *,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure one target-free-selected, source-matched TRAIN direction pair."""
    from windfarm.geometry import D_M
    from windfarm.training.unified_task import _scene_inputs

    row_from, row_to = WIND_DIRECTION_PAIR_ROWS
    train_rows = set(map(int, np.asarray(provider.train_rows).tolist()))
    if row_from not in train_rows or row_to not in train_rows:
        raise ValueError("Wind direction-pair rows must both belong to the sealed fixed24_v1 TRAIN panel.")
    cases = (provider.view.run(row_from), provider.view.run(row_to))
    expected = ((2, 23, 270.0), (2, 23, 285.0))
    for case, (layout, count, direction) in zip(cases, expected):
        if (
            int(case.layout_index) != layout
            or int(case.n_turbines) != count
            or not math.isclose(float(case.wind_direction_deg), direction, rel_tol=0.0, abs_tol=1.0e-6)
        ):
            raise ValueError(f"Frozen same-layout direction probe row {case.index} changed its input identity.")

    scenes = tuple(provider.make_scene((_scene_inputs(case),)) for case in cases)
    active_ids = []
    active_centers = []
    for scene in scenes:
        present = _as_numpy(scene.present).reshape(-1) > 0.5
        ids = _as_numpy(scene.source_ids).reshape(-1)[present].astype(np.int64)
        centers = _as_numpy(scene.centers).reshape(-1, 3)[present].astype(np.float64)
        active_ids.append(ids)
        active_centers.append(centers)
    transform = _fit_wind_source_rotation(active_ids[0], active_centers[0], active_ids[1], active_centers[1])

    plane_from = _wind_plane_geometry(cases[0].run, 70.0)
    candidate_from = np.asarray(plane_from["coords_D"], dtype=np.float32)
    mapped_candidates = _map_wind_direction_coordinates(candidate_from, transform)
    x_bounds = np.asarray(cases[1].run.x_m, dtype=np.float64) / float(D_M)
    y_bounds = np.asarray(cases[1].run.y_m, dtype=np.float64) / float(D_M)
    z_bounds = np.asarray(cases[1].run.z_m, dtype=np.float64) / float(D_M)
    inside_to = (
        (mapped_candidates[:, 0] >= x_bounds[0]) & (mapped_candidates[:, 0] <= x_bounds[-1])
        & (mapped_candidates[:, 1] >= y_bounds[0]) & (mapped_candidates[:, 1] <= y_bounds[-1])
        & (mapped_candidates[:, 2] >= z_bounds[0]) & (mapped_candidates[:, 2] <= z_bounds[-1])
    )
    candidate_indices = np.flatnonzero(inside_to)
    if candidate_indices.size < WIND_DIRECTION_PAIR_RECEIVERS:
        raise ValueError("Fewer than nine row-6 native-plane receivers map inside row-7 reference support.")
    selected_local = farthest_receiver_indices(
        candidate_from[candidate_indices], count=WIND_DIRECTION_PAIR_RECEIVERS
    )
    selected_candidates = candidate_indices[selected_local]
    receivers_from = candidate_from[selected_candidates].astype(np.float32, copy=True)
    receivers_to = _map_wind_direction_coordinates(receivers_from, transform).astype(np.float32)
    selected_native_flat_from = np.asarray(plane_from["flat_indices"], dtype=np.int64)[selected_candidates]

    query_pairs = (receivers_from, receivers_to)
    route_names = ("all_fine", "all_base", "adaptive")
    row_routes: list[dict[str, dict[str, Any]]] = []
    row_work: list[dict[str, dict[str, int]]] = []
    for scene, coordinates in zip(scenes, query_pairs):
        predictions: dict[str, dict[str, Any]] = {}
        work: dict[str, dict[str, int]] = {}
        for route in route_names:
            result, counts = _wind_predict_plane(
                model,
                scene,
                coordinates,
                mode=route,
                threshold=0.5,
                temperature=temperature,
            )
            predictions[route] = result
            work[route] = counts
        row_routes.append(predictions)
        row_work.append(work)

    # The paired native targets are not read until both target-free model
    # prediction sets have completed.
    references = tuple(
        _wind_trilinear_reference(case.run, coordinates, diameter_m=float(D_M))
        for case, coordinates in zip(cases, query_pairs)
    )

    prefix = "layout2_direction_pair"
    arrays: dict[str, np.ndarray] = {
        f"{prefix}/receiver_index": np.arange(WIND_DIRECTION_PAIR_RECEIVERS, dtype=np.int64),
        f"{prefix}/row{row_from}/query_coordinates_D": receivers_from,
        f"{prefix}/row{row_to}/query_coordinates_D": receivers_to,
        f"{prefix}/row{row_from}/native_plane_flat_indices": selected_native_flat_from,
        f"{prefix}/source_ids": active_ids[0].copy(),
        f"{prefix}/row{row_from}/source_centers_D": active_centers[0].copy(),
        f"{prefix}/row{row_to}/source_centers_D": active_centers[1].copy(),
        f"{prefix}/row{row_from}/environment_coords_D": _as_numpy(scenes[0].environment_coords),
        f"{prefix}/row{row_to}/environment_coords_D": _as_numpy(scenes[1].environment_coords),
        f"{prefix}/source_rotation_column_xy": np.asarray(transform["rotation_column_xy"], dtype=np.float64),
        f"{prefix}/source_translation_xy_D": np.asarray(transform["translation_xy_D"], dtype=np.float64),
    }
    row_records = []
    for row, case, predictions, work, reference in zip(
        WIND_DIRECTION_PAIR_ROWS, cases, row_routes, row_work, references
    ):
        row_prefix = f"{prefix}/row{row}"
        target = reference["velocity_mps"]
        arrays[f"{row_prefix}/target/velocity_mps"] = target
        arrays[f"{row_prefix}/target/interpolation_flat_indices"] = reference["flat_neighbor_indices"]
        arrays[f"{row_prefix}/target/interpolation_weights"] = reference["interpolation_weights"]
        metrics: dict[str, Any] = {}
        for route in route_names:
            physical = np.asarray(predictions[route]["velocity_mps"][0], dtype=np.float32)
            arrays[f"{row_prefix}/{route}/velocity_mps"] = physical
            arrays[f"{row_prefix}/{route}/residual_mps"] = physical - target
            metrics[route] = _wind_native_metrics(physical, target)
            if route == "adaptive":
                for route_key in ("keep", "protected", "probability"):
                    value = predictions[route].get(f"route/{route_key}")
                    if value is None:
                        raise RuntimeError(f"Wind adaptive direction pair omitted the {route_key} mask.")
                    arrays[f"{row_prefix}/adaptive/{route_key}"] = np.asarray(value).copy()
        full = np.asarray(predictions["all_fine"]["velocity_mps"][0], dtype=np.float32)
        base = np.asarray(predictions["all_base"]["velocity_mps"][0], dtype=np.float32)
        adaptive = np.asarray(predictions["adaptive"]["velocity_mps"][0], dtype=np.float32)
        arrays[f"{row_prefix}/F_minus_B/velocity_mps"] = full - base
        arrays[f"{row_prefix}/adaptive_minus_full/velocity_mps"] = adaptive - full
        row_records.append({
            "row_index": int(row),
            "case": str(case.case),
            "layout_index": int(case.layout_index),
            "wind_direction_deg": float(case.wind_direction_deg),
            "module_count": int(case.n_turbines),
            "partition": "fixed24_v1 TRAIN; illustrative changed-direction probe, excluded from DEV24 statistics",
            "source_membership": _wind_partition(provider, row),
            "route_work": work,
            "physical_metrics_on_nine_receivers": metrics,
            "target_interpolation": "trilinear native-cell-center interpolation at the exact mapped query coordinates",
            "environment_context_token_count": int(_as_numpy(scenes[row - row_from].environment_coords).shape[-2]),
            "dependency": _dependency_payload(scenes[row - row_from].dependency),
        })
    return {
        "scope": "one fixed input-only same-layout direction pair; no complete plane, fit, or new solver call",
        "selection": {
            "method": "nine farthest-point native receiver coordinates on row 6's 70 m z-plane, filtered to row 7 native support before selection",
            "uses_target_values": False,
            "receiver_count": WIND_DIRECTION_PAIR_RECEIVERS,
            "row6_native_flat_indices_saved": True,
        },
        "fixed24_manifest_sha256": str(provider.manifest["manifest_sha256"]),
        "source_correspondence": {
            "source_ids_identical_in_order": True,
            "matched_source_count": transform["matched_source_count"],
            "rotation_degrees_ccw": transform["rotation_degrees_ccw"],
            "rotation_column_xy": np.asarray(transform["rotation_column_xy"]).tolist(),
            "translation_xy_D": np.asarray(transform["translation_xy_D"]).tolist(),
            "max_center_error_D": transform["max_center_error_D"],
            "max_z_error_D": transform["max_z_error_D"],
            "transform_direction": f"row {row_from} coordinates to row {row_to} coordinates",
        },
        "paired_receivers_share_mapped_physical_coordinates": True,
        "physical_outputs": ["all_fine", "all_base", "adaptive_hard_threshold_0.5"],
        "rows": row_records,
    }, arrays


def _evaluate_representatives(
    task: str,
    model: Any,
    provider: Any,
    *,
    arm: str,
    epoch: int,
    temperature: float,
    field_only: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if field_only and arm != "full_detail":
        raise ValueError("Field-only representative exports are restricted to a Full-detail checkpoint.")
    report: dict[str, Any] = {
        "scope": "fixed representative native maps; targets are loaded/saved only after target-free prediction",
        "route_frontier_thresholds": [] if field_only else list(ROUTE_FRONTIER_THRESHOLDS),
        "controls_on_fixed_representatives_for_each_checkpoint": not field_only,
        "export_mode": "all_fine_field_only" if field_only else "adaptive_controls_and_complete_fine_diagnostic",
        "records": [],
    }
    arrays: dict[str, np.ndarray] = {}
    complete_fine_arrays: dict[str, np.ndarray] = {}
    adaptive_controls = not field_only
    if task == "thermal":
        from channelthermal.source_response import THERMAL_INTERACTION_CONTROL_UNITS

        representatives = []
        by_id = {str(case["case_id"]): case for case in provider.validation_cases}
        if not set(THERMAL_REPRESENTATIVES).issubset(by_id):
            raise ValueError("Thermal fixed representative IDs are not in the sealed DEV22 provider.")
        for case_id in THERMAL_REPRESENTATIVES:
            case = by_id[case_id]
            representatives.append((case_id, case))
        report["thermal_units"] = dict(THERMAL_INTERACTION_CONTROL_UNITS)
        report["temperature_unit"] = "packed_dataset_native_temperature"
        for case_id, case in representatives:
            scene_meta = {
                "case_id": case_id,
                "partition": "fixed25_v1 exposed DEV22",
                "module_count": int(np.asarray(case["structure"]["module_present"]).sum()),
            }
            case_point_weights = np.asarray(case["point_weights"], dtype=np.float64)
            case_temperature_target = np.asarray(case["field_targets"][..., 4], dtype=np.float64)
            finite_point_weights = np.isfinite(case_point_weights)
            finite_temperature_target = np.isfinite(case_temperature_target)
            finite_weights = case_point_weights[finite_point_weights]
            scene_meta["data_boundary"] = {
                "scope": "all native fluid query rows for this fixed25_v1 DEV case",
                "native_fluid_query_rows": int(case_temperature_target.size),
                "positive_point_weight_rows": int(np.count_nonzero(finite_point_weights & (case_point_weights > 0))),
                "nonpositive_point_weight_rows": int(np.count_nonzero(finite_point_weights & (case_point_weights <= 0))),
                "nonfinite_point_weight_rows": int(np.count_nonzero(~finite_point_weights)),
                "finite_temperature_target_rows": int(finite_temperature_target.sum()),
                "point_weight_min": float(finite_weights.min()) if finite_weights.size else None,
                "point_weight_max": float(finite_weights.max()) if finite_weights.size else None,
                "reference_and_residual_summary": "unweighted native-grid point pooling",
                "point_weight_magnitudes_applied_to_reference_or_tail_summary": False,
            }
            coordinates = np.asarray(case["query_xy"], dtype=np.float32)
            route_records: dict[str, np.ndarray] = {}
            route_predictions: dict[str, Any] = {}
            adaptive_receivers = None
            scene_array_names = {
                "query_xy", "source_present", "source_ids", "source_identity_catalogue",
                "source_centers", "source_states", "global_state",
                "source_lengths", "material_params",
                "module_radius_native",
                "physical_heat", "environment_states", "environment_coords", "environment_measures", "temperature_target",
                "interface_target", "material_target", "dependency_json",
            }
            modes = [("all_fine", None)]
            if adaptive_controls:
                modes.extend((mode, threshold) for threshold in ROUTE_FRONTIER_THRESHOLDS for mode in CONTROL_MODES)
            for mode, threshold in modes:
                active_threshold = 0.5 if threshold is None else float(threshold)
                _synchronize(model.parameters().__next__())
                route_started = time.perf_counter()
                result, _, receivers, scene = _thermal_representative(
                    provider, case, model=model, route=mode, threshold=active_threshold,
                    epoch=epoch, temperature=temperature,
                )
                _synchronize(model.parameters().__next__())
                suffix = mode if threshold is None else f"{mode}_thr{active_threshold:.2f}"
                scene_meta.setdefault("route_seconds", {})[suffix] = time.perf_counter() - route_started
                for name, value in result.items():
                    if name not in scene_array_names and name != "route_work_json":
                        if name.startswith("route/") and suffix != "adaptive_thr0.50":
                            continue
                        route_records[f"{case_id}/{suffix}/{name}"] = value
                route_predictions[suffix] = result
                if mode == "adaptive" and active_threshold == 0.5:
                    adaptive_receivers = receivers
            # Attach native supervision only after every target-free route
            # prediction for this case has completed.
            reference = np.asarray(case["field_targets"][..., 4:5], dtype=np.float32)
            full = route_predictions["all_fine"]
            arrays[f"{case_id}/query_xy"] = coordinates
            arrays[f"{case_id}/target/fluid_temperature"] = reference
            for name in scene_array_names - {"query_xy"}:
                if name in full:
                    arrays[f"{case_id}/scene/{name}"] = full[name]
            arrays.update(route_records)
            route_metrics = {}
            for suffix, result in route_predictions.items():
                if "fluid_temperature" in result:
                    arrays[f"{case_id}/{suffix}/fluid_residual"] = result["fluid_temperature"] - reference[None]
                    route_metrics[suffix] = _thermal_native_metrics(case, result)
            interface_suffix = "adaptive_thr0.50" if "adaptive_thr0.50" in route_predictions else "all_fine"
            interface = _receiver_interface_arrays(
                coordinates,
                route_predictions[interface_suffix]["fluid_temperature"][0],
                reference,
            )
            for name, value in interface.items():
                arrays[f"{case_id}/receiver_interface/{name}"] = value
            scene_meta["receiver_interface_route"] = interface_suffix
            if adaptive_controls:
                base_key = "all_base_thr0.50"
                arrays[f"{case_id}/F_minus_B/fluid_temperature"] = (
                    full["fluid_temperature"] - route_predictions[base_key]["fluid_temperature"]
                )
                scene_meta["F_minus_B_fluid_temperature"] = _tail_summary(
                    full["fluid_temperature"] - route_predictions[base_key]["fluid_temperature"]
                )
                if adaptive_receivers is None:
                    raise RuntimeError("Thermal adaptive .5 route did not retain its native receiver inputs.")
            action_receivers = adaptive_receivers if adaptive_controls else receivers
            _synchronize(model.parameters().__next__())
            fine_started = time.perf_counter()
            fine_arrays: dict[str, np.ndarray] = {}
            fine_receipt: dict[str, Any] = {"status": "not_requested_for_full_field_export"}
            if not field_only:
                fine_arrays, fine_receipt = _thermal_complete_fine_export(
                    model, provider, scene, action_receivers, temperature=temperature
                )
                _synchronize(model.parameters().__next__())
                fine_receipt["elapsed_seconds"] = time.perf_counter() - fine_started
            if adaptive_controls:
                exact_sum = fine_arrays["fine_minus_base_heat_sum"]
                observed_delta = (
                    route_predictions["all_fine"]["fluid_temperature"]
                    - route_predictions["all_base_thr0.50"]["fluid_temperature"]
                )
                if exact_sum.shape != observed_delta.shape:
                    raise ValueError("Thermal exported source correction and native route delta shapes differ.")
                closure = exact_sum - observed_delta
                fine_arrays["observed_allfine_minus_allbase_temperature"] = observed_delta
                fine_arrays["exact_sum_closure_residual"] = closure
                fine_receipt["fine_minus_base_heat_sum_max_abs_closure_error"] = float(
                    np.max(np.abs(closure))
                )
                fine_receipt["fine_minus_base_heat_sum_rmse_closure_error"] = float(
                    np.sqrt(np.mean(closure * closure))
                )
            if case_id == "0291" and not field_only:
                _synchronize(model.parameters().__next__())
                geometry_started = time.perf_counter()
                geometry_arrays, geometry_receipt = _thermal_geometry_ad_fd_demo(
                    model,
                    provider,
                    scene,
                    action_receivers.heat,
                    fine_arrays["neural_receiver_xy"],
                    fine_arrays["native_grid_indices"],
                    route_predictions["adaptive_thr0.50"],
                    temperature=temperature,
                )
                _synchronize(model.parameters().__next__())
                geometry_receipt["geometry_ad_fd_scope_seconds"] = time.perf_counter() - geometry_started
                for name, value in geometry_arrays.items():
                    route_records[f"{case_id}/geometry_ad_fd/{name}"] = value
                scene_meta["geometry_ad_fd"] = geometry_receipt
            for name, value in fine_arrays.items():
                complete_fine_arrays[f"{case_id}/{name}"] = value
            if not field_only:
                action_route = "adaptive_detail" if adaptive_controls else "full_detail"
                _synchronize(model.parameters().__next__())
                action_started = time.perf_counter()
                action = _thermal_precise_heat_demo(
                    model, provider, scene, action_receivers,
                    route=action_route, threshold=0.5, temperature=temperature,
                )
                _synchronize(model.parameters().__next__())
                action["derivative_scope_seconds"] = time.perf_counter() - action_started
                for name, value in action.pop("input_output_arrays").items():
                    arrays[f"{case_id}/thermal_action/{name}"] = value
                scene_meta["fixed_route_heat_demo"] = action
            scene_meta["complete_fine_export"] = fine_receipt
            report["records"].append({
                **scene_meta,
                "receiver_count": int(coordinates.shape[0]),
                "native_query_coordinates": "benchmark-native physical coordinates; model saw geometry and heating only",
                "complete_fine_field": True,
                "complete_fine_source_coefficients": not field_only,
                "route_metrics": route_metrics,
                "route_work": {
                    suffix: json.loads(str(result["route_work_json"]))
                    for suffix, result in route_predictions.items() if "route_work_json" in result
                },
                "dependency": json.loads(str(full["dependency_json"])),
            })
        return report, arrays, complete_fine_arrays

    from windfarm.training.unified_task import _scene_inputs

    representatives = []
    for row, expected_m in WIND_REPRESENTATIVE_ROWS:
        case = provider.view.run(row)
        if int(case.n_turbines) != expected_m or not math.isclose(
            float(case.wind_direction_deg), 270.0, abs_tol=1.0e-6
        ):
            raise ValueError(f"Frozen Wind representative row {row} no longer matches M={expected_m}, WD=270.")
        representatives.append((row, expected_m, case, _scene_inputs(case), _wind_plane_geometry(case.run, 70.0)))
    report["reference"] = "stored OpenFOAM data; no new solver calls"
    report["velocity_unit"] = "m/s"

    # Wind representative planes are the exact previously used rows 69 and 426.
    for row, expected_m, case, scene_input, plane in representatives:
        scene = provider.make_scene((scene_input,))
        coordinates = np.asarray(plane["coords_D"], dtype=np.float32)
        arrays[f"row{row}/source_ids"] = _as_numpy(scene.source_ids)
        arrays[f"row{row}/source_present"] = _as_numpy(scene.present)
        arrays[f"row{row}/source_centers_D"] = _as_numpy(scene.centers)
        arrays[f"row{row}/source_features"] = _as_numpy(scene.sources)
        arrays[f"row{row}/global_context"] = _as_numpy(scene.context)
        arrays[f"row{row}/environment_coords_D"] = _as_numpy(scene.environment_coords)
        arrays[f"row{row}/environment_measures_D3"] = _as_numpy(scene.environment_measures)
        arrays[f"row{row}/environment_features"] = _as_numpy(scene.environment_tokens)
        metadata = {
            "row_index": row,
            "case": str(case.case),
            "layout_index": int(case.layout_index),
            "wind_direction_deg": float(case.wind_direction_deg),
            "module_count": int(case.n_turbines),
            "plane": {key: value for key, value in plane.items() if key not in {"coords_D", "flat_indices"}},
            **_wind_partition(provider, row),
        }
        mode_specs: list[tuple[str, float | None]] = [("all_fine", None)]
        if adaptive_controls:
            mode_specs.extend((mode, threshold) for threshold in ROUTE_FRONTIER_THRESHOLDS for mode in CONTROL_MODES)
        route_predictions: dict[str, dict[str, Any]] = {}
        for mode, threshold in mode_specs:
            active_threshold = 0.5 if threshold is None else float(threshold)
            _synchronize(model.parameters().__next__())
            route_started = time.perf_counter()
            result, route_work = _wind_predict_plane(
                model, scene, coordinates, mode=mode, threshold=active_threshold, temperature=temperature
            )
            _synchronize(model.parameters().__next__())
            suffix = mode if threshold is None else f"{mode}_thr{active_threshold:.2f}"
            route_predictions[suffix] = result
            metadata.setdefault("route_work", {})[suffix] = route_work
            metadata.setdefault("route_seconds", {})[suffix] = time.perf_counter() - route_started
            for name, value in result.items():
                arrays[f"row{row}/{suffix}/{name}"] = value
        _synchronize(model.parameters().__next__())
        fine_started = time.perf_counter()
        fine_receipt: dict[str, Any] = {"status": "not_requested_for_full_field_export"}
        if not field_only:
            fine_arrays, fine_receipt = _wind_complete_fine_export(
                model, scene, coordinates, temperature=temperature
            )
            _synchronize(model.parameters().__next__())
            fine_receipt["elapsed_seconds"] = time.perf_counter() - fine_started
            for name, value in fine_arrays.items():
                complete_fine_arrays[f"row{row}/{name}"] = value
            _synchronize(model.parameters().__next__())
            correction_started = time.perf_counter()
            correction_arrays, correction_receipt = _wind_single_correction_removal(
                model,
                scene,
                coordinates,
                route_predictions["adaptive_thr0.50"],
                base_velocity_mps=route_predictions["all_base_thr0.50"]["velocity_mps"][0],
                full_velocity_mps=route_predictions["all_fine"]["velocity_mps"][0],
                temperature=temperature,
            )
            _synchronize(model.parameters().__next__())
            correction_receipt["elapsed_seconds"] = time.perf_counter() - correction_started
            metadata["single_correction_removal"] = correction_receipt
            for name, value in correction_arrays.items():
                arrays[f"row{row}/single_correction/{name}"] = value
        arrays[f"row{row}/query_coordinates_D"] = coordinates
        # Read the stored OpenFOAM reference only after all target-free model
        # predictions for this scene are complete.
        reference = np.asarray(case.run.U[plane["flat_indices"]], dtype=np.float32).copy().reshape(
            *plane["shape"], 3
        ).reshape(-1, 3)
        arrays[f"row{row}/target/velocity_mps"] = reference.reshape(*plane["shape"], 3)
        for suffix, result in route_predictions.items():
            predicted = result["velocity_mps"][0]
            arrays[f"row{row}/{suffix}/residual_mps"] = predicted - reference
        if adaptive_controls:
            arrays[f"row{row}/F_minus_B/velocity_mps"] = (
                route_predictions["all_fine"]["velocity_mps"][0]
                - route_predictions["all_base_thr0.50"]["velocity_mps"][0]
            )
            metadata["F_minus_B_velocity_mps"] = _wind_native_metrics(
                route_predictions["all_fine"]["velocity_mps"][0],
                route_predictions["all_base_thr0.50"]["velocity_mps"][0],
            )
        interface_suffix = "adaptive_thr0.50" if "adaptive_thr0.50" in route_predictions else "all_fine"
        interface = _receiver_interface_arrays(
            coordinates,
            route_predictions[interface_suffix]["velocity_mps"][0],
            reference,
        )
        for name, value in interface.items():
            arrays[f"row{row}/receiver_interface/{name}"] = value
        if not field_only:
            active_mode = "adaptive" if adaptive_controls else "all_fine"
            model.core.set_execution(mode=active_mode, phase="hard", threshold=0.5,
                                     temperature=temperature, training_signal=False)
            prepared = model.prepare_case(case, device=provider.device)
            derivative_coords = coordinates[interface["held_indices"]]
            _synchronize(model.parameters().__next__())
            derivative_started = time.perf_counter()
            action = _wind_derivative_demo(model, prepared, derivative_coords)
            _synchronize(model.parameters().__next__())
            action["derivative_scope_seconds"] = time.perf_counter() - derivative_started
            for name in ("tangent", "cotangent", "jvp", "vjp"):
                arrays[f"row{row}/local_derivative/{name}"] = action.pop(name)
            metadata["local_derivative"] = action
        metadata["route_metrics"] = {
            suffix: _wind_native_metrics(result["velocity_mps"][0], reference)
            for suffix, result in route_predictions.items()
        }
        metadata["dependency"] = _dependency_payload(scene.dependency)
        metadata["complete_fine_export"] = fine_receipt
        metadata["complete_fine_field"] = True
        metadata["complete_fine_source_messages"] = not field_only
        report["records"].append(metadata)
    if adaptive_controls:
        paired_report, paired_arrays = _evaluate_wind_same_layout_direction_pair(
            model, provider, temperature=temperature
        )
        report["same_layout_direction_pair"] = paired_report
        arrays.update(paired_arrays)
    return report, arrays, complete_fine_arrays


def _validate_export_scope(arm: str, *, detailed: bool, field_only: bool) -> None:
    if detailed and field_only:
        raise ValueError("A checkpoint cannot request both adaptive controls and Full-detail field-only export.")
    if detailed and arm != "adaptive_detail":
        raise ValueError("Detailed organizer exports are restricted to the selected adaptive-detail checkpoint.")
    if field_only and arm != "full_detail":
        raise ValueError("Field-only representative exports are restricted to a Full-detail checkpoint.")


def evaluate_checkpoint(
    task: str,
    label: str,
    checkpoint_path: Path,
    *,
    model: Any,
    provider: Any,
    engine_config: Any,
    selection: Any,
    initial_state_sha256: str,
    output_root: Path,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    detailed: bool = False,
    field_only: bool = False,
    recompute_validation: bool = False,
) -> dict[str, Any]:
    from honf_runtime.compat import load_trusted_checkpoint

    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Trusted checkpoint loader returned a non-mapping payload.")
    provider_state = payload.get("provider_training_state", {})
    prefit_provider_state = provider.training_state_dict() if callable(
        getattr(provider, "training_state_dict", None)
    ) else {}
    restore_provider_state = getattr(provider, "load_training_state_dict", None)
    provider_state_restore: dict[str, Any] = {"status": "not_supported"}
    if callable(restore_provider_state):
        if task == "wind":
            restore_provider_state(provider_state, allow_prefit_calibration_replace=True)
        else:
            restore_provider_state(provider_state)
        provider_state_restore = {"status": "restored_from_checkpoint"}
        local_scale = prefit_provider_state.get("message_scale")
        saved_scale = provider_state.get("message_scale")
        if task == "wind":
            provider_state_restore.update({
                "prefit_device_message_rms": local_scale,
                "checkpoint_message_rms": saved_scale,
                "prefit_to_checkpoint_message_rms_delta": (
                    None if local_scale is None or saved_scale is None
                    else float(local_scale) - float(saved_scale)
                ),
                "checkpoint_calibration_is_authoritative": True,
            })
    binding = validate_checkpoint_binding(
        payload,
        task=task,
        provider_identity=provider.identity_payload(),
        initial_state_sha256=initial_state_sha256,
        engine_config=engine_config,
        selection=selection,
        label=label,
    )
    _validate_export_scope(binding["arm"], detailed=detailed, field_only=field_only)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(provider.device if hasattr(provider, "device") else "cpu")
    model.eval()
    age = int(binding["epoch"])
    temperature = engine_config.temperature_for_epoch(max(age, 1))
    arm = str(binding["arm"])
    normal_mode = "adaptive" if arm == "adaptive_detail" else "all_fine"
    validation_cache = None if recompute_validation else _load_training_validation_cache(
        payload, task=task, provider=provider, checkpoint_path=checkpoint_path, runs_root=runs_root
    )
    started = time.perf_counter()
    if validation_cache is None:
        metrics = _collect_validation(
            task,
            model,
            provider,
            mode=normal_mode,
            threshold=0.5,
            epoch=age,
            temperature=temperature,
        )
        validation_source = {
            "source": "fresh evaluator target-free inference over the complete fixed DEV panel",
            "development_rows": 22 if task == "thermal" else 24,
        }
    else:
        metrics, validation_source = validation_cache
    if detailed or field_only:
        representative_report, representative_arrays, complete_fine_arrays = _evaluate_representatives(
            task,
            model,
            provider,
            arm=arm,
            epoch=age,
            temperature=temperature,
            field_only=field_only,
        )
    else:
        representative_report = {"scope": "not requested; representative controls are emitted only for the selected adaptive checkpoint",
                                 "records": []}
        representative_arrays = {}
        complete_fine_arrays = {}
    response_report = None
    response_arrays: dict[str, np.ndarray] = {}
    if task == "thermal" and detailed:
        response_report, response_arrays = _collect_thermal_response_development(
            model, provider, arm=arm, epoch=age, temperature=temperature
        )
    elapsed = time.perf_counter() - started
    output_dir = output_root / task / label
    validation_path = output_dir / "development_validation_metrics.json"
    _atomic_json(validation_path, {"metrics": metrics, "source": validation_source})
    representative_path = output_dir / "representative_native_arrays.npz"
    complete_fine_path = output_dir / "representative_complete_fine_arrays.npz"
    response_path = output_dir / "thermal_response_development_arrays.npz"
    if detailed or field_only:
        _atomic_npz(representative_path, representative_arrays)
    if detailed:
        _atomic_npz(complete_fine_path, complete_fine_arrays)
    if task == "thermal" and detailed:
        _atomic_npz(response_path, response_arrays)
    summary = {
        "status": "completed",
        "task": task,
        "development_profile": "fixed25_v1" if task == "thermal" else "fixed24_v1",
        "checkpoint": {
            **binding,
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "dataset_identity": dict(provider.identity_payload()),
        "provider_state_restore": provider_state_restore,
        "normal_route_metric_source": validation_source,
        "normal_route_metrics": metrics,
        "representative_native": representative_report,
        "thermal_response_development": response_report,
        "arrays": {
            "development_metrics": str(validation_path),
            "development_metrics_sha256": _sha256(validation_path),
            **({
                "representatives": str(representative_path),
                "representatives_sha256": _sha256(representative_path),
            } if detailed or field_only else {}),
            **({
                "complete_fine_export": str(complete_fine_path),
                "complete_fine_export_sha256": _sha256(complete_fine_path),
            } if detailed else {}),
            **({
                "thermal_response_development": str(response_path),
                "thermal_response_development_sha256": _sha256(response_path),
            } if task == "thermal" and detailed else {}),
        },
        "evaluation_seconds": elapsed,
        "data_boundary": {
            "targets_entered_inference": False,
            "development_metric_rows": 22 if task == "thermal" else 24,
            "representative_panels": 4 if task == "thermal" else 2,
            "route_controls_on_representatives_only": True,
            "complete_fine_export_is_separate_diagnostic_cost": True,
            "detailed_representatives_requested": detailed,
            "full_detail_field_only_export_requested": field_only,
            "development_metrics_reused_from_monitor": validation_cache is not None,
            "thermal_units": "packed dataset-native temperature and heating-rate units",
            "wind_units": "m/s with rotor-diameter geometry coordinates",
            "wind_reference": "stored OpenFOAM data; no new solver calls",
        },
    }
    summary_path = output_dir / "evaluation_summary.json"
    _atomic_json(summary_path, summary)
    return {"summary_path": str(summary_path), **summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("thermal", "wind"), required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH",
                        help="Use labels selected, literal1000 and/or literal2500; epoch is read from payload.")
    parser.add_argument("--detailed-label", action="append", default=[], metavar="LABEL",
                        help="Emit fixed-representative controls and complete fine-source exports for this adaptive checkpoint label only.")
    parser.add_argument("--field-label", action="append", default=[], metavar="LABEL",
                        help="Emit fixed-representative native fields for this Full-detail checkpoint only; no controls or complete fine-source export.")
    parser.add_argument("--recompute-validation", action="store_true",
                        help="Re-evaluate all22/all24 even when the exact same-run hard-route monitor cache is available.")
    parser.add_argument("--device", default="cpu", help="Default is CPU; GPU use requires separate authorization.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("Evaluator device must be cpu or an explicit cuda device.")
    specs = [parse_checkpoint_spec(value) for value in args.checkpoint]
    if len({label for label, _ in specs}) != len(specs):
        raise ValueError("Checkpoint labels must be unique.")
    if len({str(path) for _, path in specs}) != len(specs):
        raise ValueError("A checkpoint path cannot be relabeled as multiple ages.")
    labels = {label for label, _ in specs}
    if len(set(args.detailed_label)) != len(args.detailed_label) or not set(args.detailed_label).issubset(labels):
        raise ValueError("Every detailed-label must identify one unique supplied checkpoint label.")
    if len(set(args.field_label)) != len(args.field_label) or not set(args.field_label).issubset(labels):
        raise ValueError("Every field-label must identify one unique supplied checkpoint label.")
    if set(args.detailed_label) & set(args.field_label):
        raise ValueError("A checkpoint label cannot request both detailed and field-only exports.")
    model, provider, engine_config, selection, initial_sha = _load_task(args.task, args.device)
    output_root = args.output_dir.expanduser().resolve()
    results = []
    for label, path in specs:
        results.append(evaluate_checkpoint(
            args.task,
            label,
            path,
            model=model,
            provider=provider,
            engine_config=engine_config,
            selection=selection,
            initial_state_sha256=initial_sha,
            output_root=output_root,
            runs_root=args.runs_root.expanduser().resolve(),
            detailed=label in set(args.detailed_label),
            field_only=label in set(args.field_label),
            recompute_validation=(
                args.recompute_validation
                or label in set(args.detailed_label)
                or label in set(args.field_label)
            ),
        ))
    index = {
        "status": "completed",
        "task": args.task,
        "device": args.device,
        "checkpoint_results": [
            {"label": result["checkpoint"]["label"], "epoch": result["checkpoint"]["epoch"],
             "arm": result["checkpoint"]["arm"], "summary_path": result["summary_path"]}
            for result in results
        ],
    }
    index_path = output_root / args.task / "evaluation_index.json"
    _atomic_json(index_path, index)
    print(json.dumps({"index": str(index_path), **index}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the maintained CLI
    raise SystemExit(main())
