#!/usr/bin/env python3
"""Evaluate Wind checkpoints on an exact native validation panel and hub planes.

This CPU-only evaluator keeps the stored mesh coordinates and each checkpoint's
own transform. Plane predictions are decoded in bounded tiles from one
prepared organization per row. Development DEV24 and full_train1000 VALID90
boards have distinct strict identity guards and output scopes. It never trains,
invokes a solver, or reads source TEST target values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (
    PROJECT / "src",
    PROJECT / "tools",
    PROJECT / "Case_WindFarm/src",
):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from content_hypergraph_evaluate import load_evaluation_components
from windfarm.content_hypergraph import WindContentReceivers
from windfarm.data import WindFarmNativeView, case_batch
from windfarm.geometry import ENV_TOKEN_SHAPE, support_features
from windfarm.study_spatial import native_plane
from windfarm.workflows.evaluate_forward import load_checkpoint as load_retained_checkpoint


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def endpoint_role_query_counts(
    role_ids: np.ndarray,
    role_names: Sequence[str],
    *,
    expected_query_count: int,
) -> dict[str, int]:
    """Count each endpoint role from the sealed receiver IDs for every row."""
    role_values = np.asarray(role_ids)
    names = tuple(str(name) for name in role_names)
    if role_values.ndim != 2 or role_values.shape[0] == 0:
        raise ValueError("Endpoint role IDs must have shape [rows, queries] with at least one row.")
    if not names or len(set(names)) != len(names):
        raise ValueError("Endpoint role names must be nonempty and unique.")
    if role_values.shape[1] != int(expected_query_count):
        raise ValueError("Endpoint role IDs do not match the declared per-row query count.")
    if not np.issubdtype(role_values.dtype, np.integer):
        raise TypeError("Endpoint role IDs must use an integer dtype.")
    if role_values.size and (int(role_values.min()) < 0 or int(role_values.max()) >= len(names)):
        raise ValueError("Endpoint role IDs contain an index with no role name.")
    counts_by_role = np.stack(
        [np.count_nonzero(role_values == role_index, axis=1) for role_index in range(len(names))],
        axis=1,
    )
    if not np.all(counts_by_role.sum(axis=1) == int(expected_query_count)):
        raise ValueError("Endpoint role counts do not sum to the declared per-row query count.")
    if not np.all(counts_by_role == counts_by_role[0]):
        raise ValueError("The fixed endpoint panel must use the same role counts for every row.")
    return {name: int(counts_by_role[0, index]) for index, name in enumerate(names)}


def parse_named_paths(values: Sequence[str], option_name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{option_name} entries must have the form LABEL=PATH")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
            raise ValueError(f"Invalid {option_name} label {label!r}")
        if label in result:
            raise ValueError(f"Duplicate {option_name} label {label!r}")
        result[label] = Path(raw_path).expanduser().resolve()
    return result


def select_low_high_development_rows(
    rows: Sequence[int],
    turbine_count_by_row: Mapping[int, int],
    *,
    preferred_low: int = 69,
    preferred_high: int = 426,
) -> dict[str, int]:
    """Choose input-only minimum/maximum-M exposed validation rows with stable tie breaks."""
    canonical = tuple(int(row) for row in rows)
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("The frozen development row list must be nonempty and unique.")
    missing = sorted(set(canonical) - set(map(int, turbine_count_by_row)))
    if missing:
        raise ValueError(f"Missing frozen turbine counts for development rows {missing}")
    counts = {row: int(turbine_count_by_row[row]) for row in canonical}
    if any(count <= 0 for count in counts.values()):
        raise ValueError("Development turbine counts must be positive.")
    low_count = min(counts.values())
    high_count = max(counts.values())
    low_candidates = [row for row in canonical if counts[row] == low_count]
    high_candidates = [row for row in canonical if counts[row] == high_count]
    low = preferred_low if preferred_low in low_candidates else min(low_candidates)
    high = preferred_high if preferred_high in high_candidates else max(high_candidates)
    if low == high and len(canonical) > 1:
        other = [row for row in canonical if row != low]
        high = preferred_high if preferred_high in other else max(other)
    return {"low_M": int(low), "high_M": int(high)}


def native_hub_plane(case: Any) -> dict[str, Any]:
    """Return the stored z-plane nearest hub height and its exact flat IDs."""
    run = case.run
    z_index = int(np.argmin(np.abs(np.asarray(run.z_m, dtype=np.float64) - float(case.hub_height_m))))
    plane = native_plane(run, "z", float(case.hub_height_m), diameter=float(case.diameter_m))
    iy, ix = np.indices((run.ny, run.nx), dtype=np.int64)
    flat = (z_index * run.ny * run.nx + iy * run.nx + ix).astype(np.int64)
    target = np.asarray(run.U[flat.reshape(-1)], dtype=np.float32).reshape(run.ny, run.nx, 3).copy()
    if not np.array_equal(target, np.asarray(plane["target"], dtype=np.float32)):
        raise ValueError("Native hub plane target and C-order flat IDs do not align.")
    coordinates = np.asarray(plane["coords_D"], dtype=np.float32).reshape(run.ny, run.nx, 3)
    if not np.allclose(coordinates[..., 2], float(run.z_m[z_index]) / float(case.diameter_m), rtol=0, atol=1e-7):
        raise ValueError("Hub plane coordinates do not lie on the selected stored z plane.")
    return {
        "flat_indices": flat,
        "coordinates_D": coordinates.copy(),
        "target_mps": target,
        "z_index": z_index,
        "requested_z_m": float(case.hub_height_m),
        "actual_z_m": float(run.z_m[z_index]),
        "y_m": np.asarray(run.y_m, dtype=np.float64).copy(),
        "x_m": np.asarray(run.x_m, dtype=np.float64).copy(),
    }


def choose_upstream_source(scene_inputs: Any) -> tuple[int, int]:
    """Choose the input-only most-upstream active source in native x/D."""
    centers = np.asarray(scene_inputs.module_centers_D, dtype=np.float64)
    present = np.asarray(scene_inputs.module_present) > 0.5
    ids = np.asarray(scene_inputs.module_ids, dtype=np.int64)
    active = np.flatnonzero(present)
    if active.size == 0 or ids.shape != present.shape or centers.shape != (ids.size, 3):
        raise ValueError("Wind source geometry must contain aligned active literal source IDs.")
    chosen = min(active.tolist(), key=lambda index: (float(centers[index, 0]), int(ids[index])))
    return int(chosen), int(ids[chosen])


def native_downstream_crosswind_cuts(case: Any, scene_inputs: Any) -> dict[str, Any]:
    """Choose exact stored x/z cuts at 2D, 4D, and 8D past the upstream hub."""
    source_index, source_id = choose_upstream_source(scene_inputs)
    source = np.asarray(scene_inputs.module_centers_D[source_index], dtype=np.float64)
    run = case.run
    z_index = int(np.argmin(np.abs(np.asarray(run.z_m, dtype=np.float64) - float(case.hub_height_m))))
    rows: list[dict[str, Any]] = []
    for requested_offset in (2.0, 4.0, 8.0):
        requested_x_m = (float(source[0]) + requested_offset) * float(case.diameter_m)
        x_index = int(np.argmin(np.abs(np.asarray(run.x_m, dtype=np.float64) - requested_x_m)))
        iy = np.arange(run.ny, dtype=np.int64)
        flat = (z_index * run.ny * run.nx + iy * run.nx + x_index).astype(np.int64)
        coordinates = np.column_stack((
            np.full(run.ny, float(run.x_m[x_index]) / float(case.diameter_m)),
            np.asarray(run.y_m, dtype=np.float64) / float(case.diameter_m),
            np.full(run.ny, float(run.z_m[z_index]) / float(case.diameter_m)),
        )).astype(np.float32)
        target = np.asarray(run.U[flat], dtype=np.float32).copy()
        rows.append({
            "requested_offset_D": float(requested_offset),
            "requested_x_m": float(requested_x_m),
            "x_index": x_index,
            "actual_x_m": float(run.x_m[x_index]),
            "actual_offset_D": float(run.x_m[x_index] / float(case.diameter_m) - source[0]),
            "z_index": z_index,
            "actual_z_m": float(run.z_m[z_index]),
            "flat_indices": flat,
            "coordinates_D": coordinates,
            "target_mps": target,
        })
    return {
        "source_index": source_index,
        "source_id": source_id,
        "source_center_D": source.astype(np.float32),
        "cuts": rows,
    }


def _receivers_for_tile(adapter: Any, scene_inputs: Any, support: Any, coordinates_D: np.ndarray,
                        flat_indices: np.ndarray) -> WindContentReceivers:
    coordinates = np.asarray(coordinates_D, dtype=np.float32)
    flat = np.asarray(flat_indices, dtype=np.int64)
    features = adapter.query_features(coordinates, support)
    local_query, local_module = adapter.packed_local_pair_indices(scene_inputs, coordinates)
    return WindContentReceivers(
        coordinates_D=coordinates[None],
        features=np.asarray(features, dtype=np.float32)[None],
        flat_indices=flat[None],
        role_ids=np.zeros((1, len(flat)), dtype=np.int8),
        scene_keys=(scene_inputs.scene_key,),
        local_batch_indices=np.zeros(local_query.size, dtype=np.int64),
        local_query_indices=local_query,
        local_module_indices=local_module,
    )


def decode_content_query_tiles(
    model: Any,
    adapter: Any,
    scene_inputs: Any,
    scene: Any,
    organization: Any,
    coordinates_D: np.ndarray,
    flat_indices: np.ndarray,
    *,
    tile_size: int,
    support: Any,
) -> np.ndarray:
    """Decode exact native queries from the caller-owned prepared organization."""
    coordinates = np.asarray(coordinates_D, dtype=np.float32)
    flat = np.asarray(flat_indices, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or flat.shape != (len(coordinates),):
        raise ValueError("Content-native query coordinates and IDs must align as [Q,3] and [Q].")
    if tile_size < 1:
        raise ValueError("Native query tile size must be positive.")
    prediction = np.empty((len(coordinates), 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(coordinates), tile_size):
            stop = min(start + tile_size, len(coordinates))
            receivers = _receivers_for_tile(
                adapter, scene_inputs, support, coordinates[start:stop], flat[start:stop]
            )
            query = adapter.make_query_batch(scene, receivers)
            output = model.decode(organization, query)
            physical, _, _ = adapter.physical_velocity(output.nonlinear_field, query.coordinates)
            prediction[start:stop] = physical[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return prediction


def decode_retained_query_tiles(model: Any, prepared: Any, coordinates_D: np.ndarray, features: np.ndarray,
                                *, tile_size: int) -> np.ndarray:
    coordinates = torch.tensor(np.asarray(coordinates_D, dtype=np.float32)[None], dtype=torch.float32)
    query_features = torch.tensor(np.asarray(features, dtype=np.float32)[None], dtype=torch.float32)
    output = np.empty((coordinates.shape[1], 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, coordinates.shape[1], tile_size):
            stop = min(start + tile_size, coordinates.shape[1])
            values = model.predict_physical(
                prepared,
                coordinates[:, start:stop],
                query_features[:, start:stop],
                receiver_chunk_size=tile_size,
            )
            output[start:stop] = values[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return output


def _role_metrics_row(provider: Any, metadata: Mapping[str, Any], prediction: np.ndarray,
                      target: np.ndarray, role_ids: np.ndarray, coordinates_D: np.ndarray) -> dict[str, Any]:
    profile = provider.adapter.profile_at(torch.tensor(coordinates_D[None], dtype=torch.float32))[0]
    profile_np = profile.detach().cpu().numpy().astype(np.float64, copy=False)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    role_ids = np.asarray(role_ids, dtype=np.int64)
    roles: dict[str, Any] = {}
    objective: list[float] = []
    profile_objective: list[float] = []
    role_order = tuple(provider.role_query_counts)
    role_lookup = {name: index for index, name in enumerate(role_order)}
    for name in role_order:
        selected = role_ids == role_lookup[name]
        if not np.any(selected):
            raise ValueError(f"Endpoint panel has no queries in role {name!r}.")
        delta = prediction[selected] - target[selected]
        profile_delta = profile_np[selected] - target[selected]
        mse = np.mean(delta * delta, axis=0)
        profile_mse = np.mean(profile_delta * profile_delta, axis=0)
        scales = np.asarray(provider.role_component_scales[name], dtype=np.float64)
        objective.append(float(np.mean(mse / np.square(scales))))
        profile_objective.append(float(np.mean(profile_mse / np.square(scales))))
        roles[name] = {
            "component_rmse_mps": np.sqrt(mse).astype(float).tolist(),
            "component_rmse_mean_mps": float(np.sqrt(np.mean(mse))),
            "vector_rmse_mps": float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1)))),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(target[selected]), 1.0e-12)),
            "height_profile_relative_l2": float(
                np.linalg.norm(profile_delta) / max(np.linalg.norm(target[selected]), 1.0e-12)
            ),
            "component_scales_mps": scales.astype(float).tolist(),
            "query_count": int(selected.sum()),
        }
    return {
        **dict(metadata),
        "roles": roles,
        "field_score": float(np.mean(objective)),
        "height_profile_score": float(np.mean(profile_objective)),
    }


def summarize_role_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("A role summary requires at least one development record.")
    role_names = tuple(records[0]["roles"])
    result: dict[str, Any] = {}
    for role in role_names:
        components = np.asarray([record["roles"][role]["component_rmse_mps"] for record in records], dtype=np.float64)
        vectors = np.asarray([record["roles"][role]["vector_rmse_mps"] for record in records], dtype=np.float64)
        row_ids = [int(record["row_index"]) for record in records]
        layouts = np.asarray([int(record["layout_index"]) for record in records], dtype=np.int64)
        layout_values: dict[int, list[float]] = defaultdict(list)
        for layout, value in zip(layouts, vectors):
            layout_values[int(layout)].append(float(value))
        per_layout = {layout: float(np.mean(values)) for layout, values in layout_values.items()}
        layout_ids = np.asarray(sorted(per_layout), dtype=np.int64)
        layout_rmse = np.asarray([per_layout[int(layout)] for layout in layout_ids], dtype=np.float64)
        worst_rows = [row_ids[int(index)] for index in np.argmax(components, axis=0)]
        vector_worst = int(np.argmax(vectors))
        layout_worst = int(np.argmax(layout_rmse))
        result[role] = {
            "component_rmse_mps_mean_across_rows": components.mean(axis=0).astype(float).tolist(),
            "component_rmse_mps_p95_across_rows": np.quantile(components, 0.95, axis=0).astype(float).tolist(),
            "component_rmse_mps_worst_across_rows": components.max(axis=0).astype(float).tolist(),
            "worst_row_id_by_component": [int(value) for value in worst_rows],
            "vector_rmse_mps_mean_across_rows": float(vectors.mean()),
            "vector_rmse_mps_p95_across_rows": float(np.quantile(vectors, 0.95)),
            "vector_rmse_mps_worst_across_rows": float(vectors[vector_worst]),
            "worst_vector_row_id": int(row_ids[vector_worst]),
            "layout_tail": {
                "layout_count": len(per_layout),
                "mean_of_layout_mean_vector_rmse_mps": float(layout_rmse.mean()),
                "p95_layout_mean_vector_rmse_mps": float(np.quantile(layout_rmse, 0.95)),
                "worst_layout_mean_vector_rmse_mps": float(layout_rmse[layout_worst]),
                "worst_layout_index": int(layout_ids[layout_worst]),
                "per_layout_mean_vector_rmse_mps": {str(key): value for key, value in per_layout.items()},
            },
        }
    return result


def summarize_cpu_cost_trials(warmup_seconds: Sequence[float], repeat_seconds: Sequence[float]) -> dict[str, Any]:
    """Summarize the fixed two-warmup, five-repeat native CPU timing contract."""
    values = np.asarray(repeat_seconds, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Native CPU cost summaries require exactly five finite positive repeats.")
    warmups = np.asarray(warmup_seconds, dtype=np.float64)
    if warmups.shape != (2,) or not np.isfinite(warmups).all() or np.any(warmups <= 0):
        raise ValueError("Native CPU cost summaries require exactly two finite positive warmups.")
    return {
        "warmup_count": int(warmups.size),
        "warmup_seconds": warmups.astype(float).tolist(),
        "repeat_count": int(values.size),
        "repeat_seconds": values.astype(float).tolist(),
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(np.mean(values)),
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
        "sample_std_seconds": float(np.std(values, ddof=1)),
    }


def _measure_alternating_cpu_cost_trials(calls: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    """Run two warmup and five measured rounds with rotated model order."""
    labels = tuple(calls)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("CPU cost trials require at least one uniquely named model call.")
    warmup_seconds = {label: [] for label in labels}
    repeat_seconds = {label: [] for label in labels}
    digests = {label: [] for label in labels}
    output_shapes: dict[str, tuple[int, ...]] = {}
    execution_orders: list[list[str]] = []

    def order_for_round(round_index: int) -> tuple[str, ...]:
        shift = (round_index // 2) % len(labels)
        rotated = labels[shift:] + labels[:shift]
        return rotated if round_index % 2 == 0 else tuple(reversed(rotated))

    for round_index in range(7):
        order = order_for_round(round_index)
        execution_orders.append(list(order))
        is_warmup = round_index < 2
        for label in order:
            started = time.perf_counter()
            prediction = np.asarray(calls[label](), dtype=np.float32)
            elapsed = time.perf_counter() - started
            if prediction.ndim != 2 or prediction.shape[1] != 3 or not np.isfinite(prediction).all():
                raise ValueError(f"Native CPU cost call {label} returned malformed/nonfinite [Q,3] values.")
            if label in output_shapes and prediction.shape != output_shapes[label]:
                raise ValueError(f"Native CPU cost call {label} changed output shape across repeats.")
            output_shapes[label] = prediction.shape
            digest = _array_digest(prediction)
            if is_warmup:
                warmup_seconds[label].append(elapsed)
            else:
                repeat_seconds[label].append(elapsed)
            digests[label].append(digest)

    result: dict[str, dict[str, Any]] = {}
    for label in labels:
        warmup_digests = digests[label][:2]
        repeat_digests = digests[label][2:]
        summary = summarize_cpu_cost_trials(warmup_seconds[label], repeat_seconds[label])
        reference_digest = warmup_digests[0]
        summary.update({
            "warmup_prediction_sha256": warmup_digests,
            "repeat_prediction_sha256": repeat_digests,
            "repeat_predictions_bitwise_identical": all(value == reference_digest for value in repeat_digests),
            "all_predictions_bitwise_identical": all(value == reference_digest for value in digests[label]),
            "execution_order_by_round": [order.index(label) for order in execution_orders],
        })
        result[label] = summary
    return result, execution_orders


def _content_cpu_prediction_once(
    model: Any,
    adapter: Any,
    case: Any,
    scene_inputs: Any,
    scene: Any,
    coordinates_D: np.ndarray,
    flat_indices: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    with torch.no_grad():
        organization = model.prepare(scene)
    return decode_content_query_tiles(
        model,
        adapter,
        scene_inputs,
        scene,
        organization,
        coordinates_D,
        flat_indices,
        tile_size=tile_size,
        support=case.support,
    )


def _retained_cpu_prediction_once(
    model: Any,
    case: Any,
    materialization_batch: Any,
    coordinates_D: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    features = support_features(coordinates_D, case.support)
    with torch.no_grad():
        prepared = model.prepare_case(materialization_batch)
    return decode_retained_query_tiles(
        model, prepared, coordinates_D, features, tile_size=tile_size
    )


def _run2202_cpu_prediction_once(
    task_type: Any,
    task_shell: Any,
    model: Any,
    scene: Any,
    receivers: Any,
    source_epoch: int,
) -> np.ndarray:
    with torch.no_grad():
        prediction, _auxiliary = task_type.predict_native(
            task_shell,
            model,
            scene,
            receivers,
            execution_mode="full_detail",
            phase="hard",
            epoch=int(source_epoch),
            temperature=1.0,
        )
    return prediction.main_mps[0].detach().cpu().numpy().astype(np.float32, copy=True)


def _content_cost_call_for_row(
    model: Any,
    adapter: Any,
    scenes: Mapping[int, tuple[Any, Any, Any, float]],
    tile_size: int,
    row: int,
    coordinates_D: np.ndarray,
    flat_indices: np.ndarray,
    _role_ids: np.ndarray,
) -> Any:
    case, scene_inputs, scene, _setup = scenes[row]
    return partial(
        _content_cpu_prediction_once,
        model,
        adapter,
        case,
        scene_inputs,
        scene,
        coordinates_D,
        flat_indices,
        tile_size,
    )


def _retained_cost_call_for_row(
    model: Any,
    cases: Mapping[int, tuple[Any, Any, np.ndarray, float]],
    tile_size: int,
    row: int,
    coordinates_D: np.ndarray,
    _flat_indices: np.ndarray,
    _role_ids: np.ndarray,
) -> Any:
    case, materialization_batch, _unused, _setup = cases[row]
    return partial(
        _retained_cpu_prediction_once,
        model,
        case,
        materialization_batch,
        coordinates_D,
        tile_size,
    )


def _run2202_cost_call_for_row(
    task_type: Any,
    task_shell: Any,
    model: Any,
    scenes: Mapping[int, tuple[Any, Any, float]],
    source_epoch: int,
    row: int,
    coordinates_D: np.ndarray,
    _flat_indices: np.ndarray,
    role_ids: np.ndarray,
) -> Any:
    _case, scene, _setup = scenes[row]
    from windfarm.training.unified_task import WindReceiverInputs

    receiver_batch = WindReceiverInputs(
        coordinates_D=np.asarray(coordinates_D, dtype=np.float32)[None],
        role_ids=np.asarray(role_ids, dtype=np.int8)[None],
    )
    return partial(
        _run2202_cpu_prediction_once,
        task_type,
        task_shell,
        model,
        scene,
        receiver_batch,
        source_epoch,
    )


def _load_run2202_native_checkpoint(path: Path) -> tuple[Any, Mapping[str, Any], dict[str, Any]]:
    """Strictly load Run2202's original native refined model and its saved transforms."""
    from windfarm.migration import (
        RUN2202_EPOCH_CHECKPOINT_SHA256,
        RUN2202_NORMALIZATION_SHA256,
        _validate_source_identity,
    )
    from windfarm.model import build_windfarm_model
    from windfarm.normalization import VerticalProfileBaseline, read_normalization_json

    from honf_runtime.compat import load_trusted_checkpoint

    checkpoint = load_trusted_checkpoint(path, map_location="cpu")
    identity, provider_identity, epoch = _validate_source_identity(checkpoint)
    file_hash = _sha256_file(path)
    expected_file_hash = RUN2202_EPOCH_CHECKPOINT_SHA256.get(int(epoch))
    if expected_file_hash is None or file_hash != expected_file_hash:
        raise ValueError("Run2202 cost checkpoint is not a literal, checksum-bound saved epoch.")
    normalization_path = Path(str(provider_identity["normalization_path"])).expanduser()
    if not normalization_path.is_file():
        # The historical path may have moved; this fallback is accepted only
        # when it is byte-identical to the normalizer hash sealed in Run2202.
        normalization_path = PROJECT / "Case_WindFarm/Dataset/derived/forward_velocity_v1/normalization.json"
    if not normalization_path.is_file():
        raise FileNotFoundError("Run2202's exact TRAIN normalizer is unavailable at its recorded or verified path.")
    normalization_hash = _sha256_file(normalization_path)
    if (
        normalization_hash != RUN2202_NORMALIZATION_SHA256
        or normalization_hash != provider_identity.get("normalization_sha256")
    ):
        raise ValueError("Run2202 native loading requires its exact checksum-bound TRAIN normalizer.")
    normalizer, file_profile = read_normalization_json(normalization_path)
    profile = VerticalProfileBaseline.from_dict(dict(provider_identity["background_profile"]))
    if file_profile is None or file_profile.to_dict() != profile.to_dict():
        raise ValueError("Run2202 embedded and retained TRAIN background profiles differ.")
    model_config = dict(provider_identity["model"])
    model_config.pop("environment_token_shape", None)
    model = build_windfarm_model(
        model_config,
        velocity_transform=normalizer,
        background_profile=profile,
    ).cpu().eval()
    loaded = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Run2202 original native model did not load the saved state strictly.")
    model.requires_grad_(False)
    return model, checkpoint, {
        "loader": "windfarm.model.build_windfarm_model plus strict checkpoint state load",
        "workflow": str(checkpoint["workflow"]),
        "model_config": model_config,
        "normalization_path_used": str(normalization_path.resolve()),
        "normalization_sha256": normalization_hash,
        "background_profile_sha256": hashlib.sha256(
            json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "source_train_rows_sha256": str(provider_identity["train_rows_sha256"]),
        "source_validation_rows_sha256": str(provider_identity["validation_rows_sha256"]),
        "dataset_scope": str(provider_identity["dataset_scope"]),
        "source_epoch": int(epoch),
        "checkpoint_sha256": file_hash,
        "source_identity_run_id": str(identity["run_identity"]),
    }


def run_native_cpu_cost_measurements(
    new_paths: Mapping[str, Path],
    retained_paths: Mapping[str, Path],
    run2202_paths: Mapping[str, Path],
    panel_result_path: Path,
    output: Path,
    *,
    query_tile_size: int = 1024,
) -> dict[str, Any]:
    """Measure matched low/high-M Q8192 native CPU inference for saved models.

    The source panel's stored coordinates, flat IDs, and role IDs are the only
    receiver inputs read. Target arrays are neither loaded nor used. Scene
    materialization and checkpoint loading are recorded outside the timed
    inference repeats; every timed call includes organization/preparation,
    the full native prediction and conversion of all Q predictions to NumPy on
    CPU. Two warmups and five alternating-order repeats are run for each
    model and row after every model family has been loaded strictly.
    """
    if output.exists():
        raise FileExistsError(f"CPU cost output already exists: {output}")
    if not new_paths:
        raise ValueError("CPU cost mode needs a corrected content checkpoint to bind the fixed DEV24 provider.")
    all_labels = tuple(new_paths) + tuple(retained_paths) + tuple(run2202_paths)
    if len(set(all_labels)) != len(all_labels):
        raise ValueError("CPU cost checkpoint labels must be unique across model families.")
    if query_tile_size <= 0:
        raise ValueError("CPU cost query tile size must be positive.")
    panel_result_path = panel_result_path.expanduser().resolve()
    if not panel_result_path.is_file():
        raise FileNotFoundError(f"Fixed DEV24 panel result is unavailable: {panel_result_path}")
    panel_result = json.loads(panel_result_path.read_text(encoding="utf-8"))
    if panel_result.get("status") != "completed":
        raise ValueError("CPU cost mode requires a completed, immutable source panel result.")
    panel_description = panel_result.get("endpoint_panel")
    if not isinstance(panel_description, Mapping):
        raise TypeError("CPU cost panel result lacks a frozen endpoint panel description mapping.")
    panel_path = Path(str(panel_description["path"])).expanduser().resolve()
    if not panel_path.is_file() or _sha256_file(panel_path) != panel_description.get("sha256"):
        raise ValueError("CPU cost endpoint panel file does not match its stored checksum.")
    with np.load(panel_path, allow_pickle=False) as archive:
        # Deliberately do not access `targets_mps` in this cost-only path.
        row_indices = np.asarray(archive["row_indices"], dtype=np.int64).copy()
        flat_indices = np.asarray(archive["native_flat_indices"], dtype=np.int64).copy()
        role_ids = np.asarray(archive["role_ids"], dtype=np.int8).copy()
        coordinates = np.asarray(archive["coordinates_D"], dtype=np.float32).copy()
    if (
        row_indices.shape != (24,)
        or flat_indices.shape != (24, 8192)
        or role_ids.shape != (24, 8192)
        or coordinates.shape != (24, 8192, 3)
        or len(set(row_indices.tolist())) != 24
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("CPU cost requires the exact saved fixed DEV24 × Q8192 native receiver panel.")
    if tuple(panel_description.get("row_order", ())) != tuple(map(int, row_indices)):
        raise ValueError("CPU cost receiver panel row order differs from its result binding.")
    row_offset = {int(row): index for index, row in enumerate(row_indices)}
    selected_rows = {"low_M": 69, "high_M": 426}
    if not set(selected_rows.values()).issubset(row_offset):
        raise ValueError("The frozen endpoint panel does not contain DEV low/high-M rows 69 and 426.")

    first_label = next(iter(new_paths))
    first_model, provider, first_payload = load_evaluation_components(new_paths[first_label], "cpu")
    if torch.cuda.is_initialized() or next(first_model.parameters()).device.type != "cpu":
        raise RuntimeError("Native CPU cost mode must remain CPU-only.")
    _assert_dev24_provider(provider)
    provider_identity = provider.identity_payload()
    if (
        panel_result.get("provider_identity", {}).get("subset_manifest_sha256")
        != provider_identity.get("subset_manifest_sha256")
        or panel_result.get("dataset_binding", {}).get("validation_row_indices_sha256")
        != provider_identity.get("validation_row_indices_sha256")
    ):
        raise ValueError("CPU cost checkpoints and the saved native receiver panel have different DEV24 bindings.")
    input_panel_hash = str(panel_description["panel_sha256"])
    records: dict[str, Any] = {}
    pending_calls: dict[str, Any] = {}

    def record_cost(
        label: str,
        checkpoint_path: Path,
        payload: Mapping[str, Any],
        family: str,
        loader_info: Mapping[str, Any],
        scene_setup_seconds: Mapping[int, float],
        call_for_row: Any,
        *,
        tile: int,
    ) -> None:
        checkpoint = _checkpoint_metadata(label, checkpoint_path, payload, model, family)
        row_records: dict[str, Any] = {}
        for row_name, row in selected_rows.items():
            panel_i = row_offset[row]
            coords = coordinates[panel_i]
            row_roles = role_ids[panel_i]
            row_flat = flat_indices[panel_i]
            unique_roles, role_counts = np.unique(row_roles, return_counts=True)
            if not np.array_equal(unique_roles, np.arange(5, dtype=np.int8)) or np.any(role_counts <= 0):
                raise ValueError(f"Cost receiver row {row} does not retain all five native roles.")
            row_metadata = provider.view.metadata
            case_name_key = next(
                (name for name in ("case_name", "case", "case_id") if name in row_metadata), None
            )
            row_records[row_name] = {
                "row_index": int(row),
                "case": (
                    str(row_metadata[case_name_key][row])
                    if case_name_key is not None else f"native_row_{row}"
                ),
                "layout_index": int(row_metadata["layout_index"][row]),
                "n_turbines": int(row_metadata["n_turbines"][row]),
                "query_count": 8192,
                "role_counts": {str(int(k)): int(v) for k, v in zip(unique_roles, role_counts)},
                "native_receiver_panel_sha256": _array_digest(row_flat, row_roles, coords),
                "scene_materialization_seconds_excluded_from_timed_repeats": float(scene_setup_seconds[row]),
            }
        pending_calls[label] = call_for_row
        records[label] = {
            "checkpoint": checkpoint,
            "family": family,
            "native_loader": dict(loader_info),
            "timing_scope": loader_info.get("timing_scope"),
            "query_tile_size": int(tile),
            "rows": row_records,
        }

    content_models: dict[str, tuple[Any, Path, Mapping[str, Any]]] = {
        first_label: (first_model, new_paths[first_label], first_payload)
    }
    expected_adapter = json.dumps(provider_identity["adapter_state"], sort_keys=True)
    for label, path in list(new_paths.items())[1:]:
        model, other_provider, payload = load_evaluation_components(path, "cpu")
        if (
            other_provider.identity_payload().get("subset_manifest_sha256") != provider_identity.get("subset_manifest_sha256")
            or json.dumps(other_provider.identity_payload()["adapter_state"], sort_keys=True) != expected_adapter
        ):
            raise ValueError("Current content cost checkpoints do not share the exact provider transforms.")
        content_models[label] = (model, path, payload)
        del other_provider
    content_cases: dict[int, tuple[Any, Any, Any, float]] = {}
    for row in selected_rows.values():
        setup_started = time.perf_counter()
        case = provider.native_case_for_development_row(int(row))
        scene_inputs = provider.adapter.scene_inputs(case)
        scene = provider.adapter.make_scene_batch((scene_inputs,))
        content_cases[int(row)] = (case, scene_inputs, scene, time.perf_counter() - setup_started)
    for label, (model, checkpoint_path, payload) in content_models.items():
        model.eval()
        if torch.cuda.is_initialized() or next(model.parameters()).device.type != "cpu":
            raise RuntimeError("A native content checkpoint escaped CPU placement.")
        setup_seconds = {row: content_cases[row][3] for row in selected_rows.values()}
        content_call = partial(
            _content_cost_call_for_row,
            model,
            provider.adapter,
            content_cases,
            query_tile_size,
        )

        record_cost(
            label,
            checkpoint_path,
            payload,
            "native_content_hypergraph",
            {
                "loader": "honf_runtime.content_hypergraph.load_content_hypergraph_checkpoint",
                "normalizer_sha256": provider_identity.get("normalizer_sha256"),
                "background_profile_sha256": provider_identity.get("background_profile_sha256"),
                "timing_scope": "core.prepare + all Q8192 core.decode tiles + TRAIN-transform inversion + full CPU NumPy return",
            },
            setup_seconds,
            content_call,
            tile=query_tile_size,
        )
        del model

    retained_view = WindFarmNativeView(
        provider.view.volume,
        compact_metadata=provider.view.metadata,
        diameter_m=provider.view.diameter_m,
        hub_height_m=provider.view.hub_height_m,
        reference_speed_mps=provider.view.reference_speed_mps,
        token_shape=ENV_TOKEN_SHAPE,
    )
    retained_cases: dict[int, tuple[Any, Any, np.ndarray, float]] = {}
    for row in selected_rows.values():
        setup_started = time.perf_counter()
        case = retained_view.run(int(row))
        materialization_batch = case_batch(case, case.module_centers)
        retained_cases[int(row)] = (case, materialization_batch, np.empty((0,), dtype=np.float32), time.perf_counter() - setup_started)
    for label, path in retained_paths.items():
        model, payload = load_retained_checkpoint(
            path,
            device="cpu",
            materialization_batch=retained_cases[69][1],
        )
        model.eval()
        if torch.cuda.is_initialized() or next(model.parameters()).device.type != "cpu":
            raise RuntimeError("A retained forward checkpoint escaped CPU placement.")
        setup_seconds = {row: retained_cases[row][3] for row in selected_rows.values()}
        retained_call = partial(
            _retained_cost_call_for_row,
            model,
            retained_cases,
            query_tile_size,
        )

        normalizer_payload = payload.get("normalization")
        record_cost(
            label,
            path,
            payload,
            "retained_forward_model",
            {
                "loader": "windfarm.workflows.evaluate_forward.load_checkpoint",
                "normalizer_payload_sha256": hashlib.sha256(
                    json.dumps(normalizer_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                ).hexdigest(),
                "timing_scope": "model.prepare_case + all Q8192 physical prediction chunks + full CPU NumPy return",
            },
            setup_seconds,
            retained_call,
            tile=query_tile_size,
        )
        del model, payload

    if run2202_paths:
        from windfarm.migration import _validate_source_identity
        from windfarm.normalization import read_normalization_json
        from windfarm.training.unified_task import (
            WindRefinementTask,
            _as_scene_batch,
            _scene_inputs,
        )

        # Use the exact full-TRAIN normalizer retained under its fixed hash.
        normalization_path = PROJECT / "Case_WindFarm/Dataset/derived/forward_velocity_v1/normalization.json"
        normalizer, _full_profile = read_normalization_json(normalization_path)
        source_provider_identity: Mapping[str, Any] | None = None
        run2202_view = WindFarmNativeView(
            provider.view.volume,
            compact_metadata=provider.view.metadata,
            diameter_m=provider.view.diameter_m,
            hub_height_m=provider.view.hub_height_m,
            reference_speed_mps=provider.view.reference_speed_mps,
            token_shape=(2, 2, 2),
        )
        run2202_cases: dict[int, tuple[Any, Any, float]] = {}
        for row in selected_rows.values():
            setup_started = time.perf_counter()
            case = run2202_view.run(int(row))
            scene_inputs = _scene_inputs(case)
            scene = _as_scene_batch((scene_inputs,), torch.device("cpu"))
            run2202_cases[int(row)] = (case, scene, time.perf_counter() - setup_started)
        for label, path in run2202_paths.items():
            model, payload, loader_info = _load_run2202_native_checkpoint(path)
            _identity, saved_provider, source_epoch = _validate_source_identity(payload)
            if source_provider_identity is None:
                source_provider_identity = saved_provider
            elif (
                saved_provider.get("normalization_sha256") != source_provider_identity.get("normalization_sha256")
                or saved_provider.get("train_rows_sha256") != source_provider_identity.get("train_rows_sha256")
            ):
                raise ValueError("Run2202 checkpoint ages do not share the same full-TRAIN source binding.")
            if normalizer.to_dict() != model.velocity_transform.to_dict():
                raise ValueError("Run2202 provider normalizer and strict native model normalizer differ.")
            model.eval()
            task_shell = object.__new__(WindRefinementTask)
            task_shell.normalizer = model.velocity_transform
            recipe = dict(saved_provider["resolved_recipe"])
            task_shell.execution_backend = str(saved_provider["execution_backend"])
            task_shell.query_tile_size = int(saved_provider["query_tile_size"])
            task_shell.gate_version = str(recipe.get("gate_version", "hard_v1"))
            task_shell.gate_transition = tuple(recipe.get("gate_transition", (0.35, 0.65)))
            setup_seconds = {row: run2202_cases[row][2] for row in selected_rows.values()}
            run2202_call = partial(
                _run2202_cost_call_for_row,
                WindRefinementTask,
                task_shell,
                model,
                run2202_cases,
                int(source_epoch),
            )

            loader_info["timing_scope"] = (
                "WindRefinementTask.predict_native(full_detail, hard) including core organization, "
                "all Q8192 native reads, physical inversion and full CPU NumPy return"
            )
            loader_info["full_train_recipe_sha256"] = saved_provider.get("resolved_recipe_sha256")
            loader_info["source_normalization_sha256"] = saved_provider.get("normalization_sha256")
            record_cost(
                label,
                path,
                payload,
                "Run2202_original_refined_native",
                loader_info,
                setup_seconds,
                run2202_call,
                tile=int(saved_provider["query_tile_size"]),
            )
            del model, payload

    model_order_by_sweep: dict[str, list[list[str]]] = {}
    for row_name, row in selected_rows.items():
        panel_i = row_offset[row]
        coords = coordinates[panel_i]
        row_roles = role_ids[panel_i]
        row_flat = flat_indices[panel_i]
        calls = {
            label: call_for_row(row, coords, row_flat, row_roles)
            for label, call_for_row in pending_calls.items()
        }
        timing_by_model, execution_orders = _measure_alternating_cpu_cost_trials(calls)
        model_order_by_sweep[row_name] = execution_orders
        for label, timing in timing_by_model.items():
            records[label]["rows"][row_name]["timing"] = timing

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU cost evaluation unexpectedly initialized CUDA.")
    result = {
        "schema": "wind_native_cpu_cost_measurement_v1",
        "status": "completed",
        "dataset_scope": "wind_shared_fixed24_v1 exposed DEV24 only; low/high-M rows 69 and 426",
        "source_panel": {
            "result_json": str(panel_result_path),
            "archive": str(panel_path),
            "archive_sha256": str(panel_description["sha256"]),
            "panel_sha256": input_panel_hash,
            "row_indices": list(map(int, row_indices)),
            "receiver_count_per_row": 8192,
            "timed_rows": selected_rows,
            "target_values_loaded_or_used": False,
            "coordinate_frame": str(panel_description["coordinate_frame"]),
            "velocity_units": str(panel_description["target_units"]),
        },
        "timing_contract": {
            "device": "cpu",
            "warmups_per_model_row": 2,
            "timed_repeats_per_model_row": 5,
            "model_order_rotated_between_sweeps": True,
            "model_order_by_sweep_by_row": model_order_by_sweep,
            "clock": "time.perf_counter wall seconds",
        "includes": "organization/preparation, complete Q8192 native prediction, transform to physical m/s, and NumPy CPU return",
            "excludes": "checkpoint loading, native scene materialization, and panel archive loading",
            "optimizer_updates": 0,
            "solver_attempts": 0,
            "wind_test_target_values_read": False,
        },
        "runtime": {
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
        },
        "models": records,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "cost_result.json").write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _assert_dev24_provider(provider: Any) -> None:
    if provider.profile_id != "wind_shared_fixed24_v1" or len(provider.validation_rows) != 24:
        raise ValueError("Wind physical reference evaluation requires the sealed fixed DEV24 provider.")
    if len({int(provider.view.metadata["layout_index"][row]) for row in provider.validation_rows}) != 8:
        raise ValueError("Wind fixed DEV24 must contain exactly eight development layouts.")


def _int64_indices_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def _full_followup_scope_contract() -> dict[str, Any]:
    from windfarm.training.joint_task import (
        FULL_TRAIN_FOLLOWUP_TEST_SHA256,
        FULL_TRAIN_FOLLOWUP_TRAIN_SHA256,
        FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256,
        JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID,
    )

    return {
        "profile_id": JOINT_WIND_FULL_FOLLOWUP_PROFILE_ID,
        "protocol": "full_train1000",
        "train_count": 420,
        "validation_count": 90,
        "test_count": 90,
        "train_layout_count": 140,
        "validation_layout_count": 30,
        "test_layout_count": 30,
        "total_row_count": 600,
        "train_rows_sha256": FULL_TRAIN_FOLLOWUP_TRAIN_SHA256,
        "validation_rows_sha256": FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256,
        "test_rows_sha256": FULL_TRAIN_FOLLOWUP_TEST_SHA256,
    }


def _validate_full_train1000_provider_payload(
    identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    train_rows: Sequence[int],
    validation_rows: Sequence[int],
    *,
    expected_contract: Mapping[str, Any],
) -> None:
    """Validate only sealed membership/identity metadata; never read TEST targets."""
    profile_id = str(expected_contract["profile_id"])
    if (
        identity.get("dataset_profile") != profile_id
        or identity.get("subset_id") != profile_id
        or manifest.get("subset_id") != profile_id
        or identity.get("protocol") != expected_contract["protocol"]
        or identity.get("subset_manifest_sha256") != manifest.get("manifest_sha256")
    ):
        raise ValueError("Full VALID90 board requires its exact full_train1000 profile and manifest binding.")

    train = np.asarray(train_rows, dtype=np.int64)
    validation = np.asarray(validation_rows, dtype=np.int64)
    manifest_train = np.asarray(manifest.get("train_row_indices", ()), dtype=np.int64)
    manifest_validation = np.asarray(manifest.get("validation_row_indices", ()), dtype=np.int64)
    test = np.asarray(manifest.get("test_row_indices", ()), dtype=np.int64)
    if (
        manifest.get("full_train_followup1000") is not True
        or manifest.get("test_row_count") != int(expected_contract["test_count"])
    ):
        raise ValueError("Full VALID90 board manifest is missing its locked original TEST metadata binding.")
    memberships = (
        ("TRAIN", train, manifest_train, "train_rows", "train_row_indices_sha256", "train_rows_sha256", "train_layout_indices"),
        ("VALID", validation, manifest_validation, "validation_rows", "validation_row_indices_sha256", "validation_rows_sha256", "validation_layout_indices"),
        ("TEST metadata", test, test, None, "test_row_indices_sha256", "test_rows_sha256", "test_layout_indices"),
    )
    combined: list[np.ndarray] = []
    for name, rows, manifest_rows, identity_rows_key, identity_hash_key, contract_hash_key, layout_key in memberships:
        expected_count = int(expected_contract[f"{contract_hash_key.removesuffix('_rows_sha256')}_count"])
        expected_hash = str(expected_contract[contract_hash_key])
        if (
            rows.ndim != 1
            or rows.size != expected_count
            or not np.array_equal(rows, np.sort(rows))
            or not np.array_equal(rows, manifest_rows)
            or _int64_indices_sha256(rows) != expected_hash
            or manifest.get(identity_hash_key) != expected_hash
            or identity.get(identity_hash_key) != expected_hash
        ):
            raise ValueError(f"Full VALID90 board has an unexpected {name} membership binding.")
        if identity_rows_key is not None and identity.get(identity_rows_key) != rows.astype(int).tolist():
            raise ValueError(f"Full VALID90 board provider {name} rows differ from the manifest.")
        layout_count = int(expected_contract[f"{contract_hash_key.removesuffix('_rows_sha256')}_layout_count"])
        expected_layouts = manifest.get(layout_key, ())
        if len(expected_layouts) != layout_count:
            raise ValueError(f"Full VALID90 board {name} must retain {layout_count} layouts.")
        if identity.get(layout_key) != expected_layouts:
            raise ValueError(f"Full VALID90 board provider {name} layout binding differs from the manifest.")
        combined.append(rows)

    all_rows = np.concatenate(combined)
    total_rows = int(expected_contract["total_row_count"])
    if all_rows.size != total_rows or np.unique(all_rows).size != total_rows:
        raise ValueError("Full VALID90 board TRAIN/VALID/TEST metadata memberships overlap or are incomplete.")
    if not np.array_equal(np.sort(all_rows), np.arange(total_rows, dtype=np.int64)):
        raise ValueError("Full VALID90 board memberships do not cover the canonical native rows.")

    recipe = identity.get("recipe", {})
    source = identity.get("normalizer_and_background_source", {})
    training = identity.get("training", {})
    if (
        recipe.get("protocol") != "full_train1000"
        or recipe.get("approved_stop_after") != 1000
        or identity.get("architecture") != "A-H6"
        or identity.get("model_config", {}).get("architecture") != "A-H6"
        or training.get("total_epochs") != 5000
        or source.get("kind") != "verified_full_train_followup_train_fit_cache"
        or source.get("training_rows_sha256") != expected_contract["train_rows_sha256"]
    ):
        raise ValueError("Full VALID90 board requires the A-H6 full_train1000 model and TRAIN-only transforms.")
    if (
        identity.get("wind_test_target_values_read") is not False
        or identity.get("unexposed_test_target_values_read") is not False
        or manifest.get("wind_test_target_values_read") is not False
    ):
        raise ValueError("WindTEST target values must remain locked for the full VALID90 board.")


def _assert_full_train1000_provider(provider: Any) -> Mapping[str, Any]:
    identity = provider.identity_payload()
    _validate_full_train1000_provider_payload(
        identity,
        provider.manifest,
        provider.train_rows,
        provider.validation_rows,
        expected_contract=_full_followup_scope_contract(),
    )
    return identity


def _prepare_retained_models(reference_paths: Mapping[str, Path], provider: Any,
                             rows: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any], WindFarmNativeView]:
    if not reference_paths:
        return {}, {}, WindFarmNativeView.__new__(WindFarmNativeView)
    source_view = WindFarmNativeView(
        provider.view.volume,
        compact_metadata=provider.view.metadata,
        diameter_m=provider.view.diameter_m,
        hub_height_m=provider.view.hub_height_m,
        reference_speed_mps=provider.view.reference_speed_mps,
        token_shape=ENV_TOKEN_SHAPE,
    )
    material_case = source_view.run(int(rows[0]))
    materialization_batch = case_batch(material_case, material_case.module_centers)
    models: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    for label, path in reference_paths.items():
        model, payload = load_retained_checkpoint(
            path,
            device="cpu",
            materialization_batch=materialization_batch,
        )
        models[label] = model
        payloads[label] = payload
    return models, payloads, source_view


def _checkpoint_metadata(label: str, path: Path, payload: Mapping[str, Any], model: Any,
                         family: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256_file(path),
        "completed_epoch": int(payload.get("epoch", payload.get("current_epoch", -1))),
        "architecture": str(getattr(model, "architecture", model.get_config().get("architecture", "unknown")
                                     if hasattr(model, "get_config") else "unknown")),
        "family": family,
    }


def _evaluate_endpoints(
    provider: Any,
    new_models: Mapping[str, tuple[Any, Path, Mapping[str, Any]]],
    reference_models: Mapping[str, tuple[Any, Path, Mapping[str, Any]]],
    reference_view: WindFarmNativeView,
    output: Path,
    *,
    board_scope: str,
    microbatch_cases: int,
    query_tile_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batches = tuple(provider.native_evaluation_batches(
        split="validation", microbatch_size=microbatch_cases
    ))
    expected_row_count = 24 if board_scope == "development" else 90
    if len(provider.validation_rows) != expected_row_count or len(batches) * microbatch_cases < expected_row_count:
        raise ValueError(f"Native {board_scope} endpoint iterator has the wrong exposed validation size.")
    endpoint_rows = [int(row) for batch in batches for row in batch.case_keys]
    if endpoint_rows != [int(row) for row in provider.validation_rows]:
        raise ValueError(f"Native Q8192 endpoint order differs from the sealed {board_scope} validation membership.")
    row_indices: list[int] = []
    flat_ids: list[np.ndarray] = []
    role_ids: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    row_hashes: dict[str, str] = {}
    for batch in batches:
        receivers = batch.receivers
        for index, row in enumerate(batch.case_keys):
            row_index = int(row)
            flat = np.asarray(receivers.flat_indices[index], dtype=np.int64)
            role = np.asarray(receivers.role_ids[index], dtype=np.int8)
            coords = np.asarray(receivers.coordinates_D[index], dtype=np.float32)
            target = np.asarray(batch.targets.velocity_mps[index], dtype=np.float32)
            row_indices.append(row_index)
            flat_ids.append(flat.copy())
            role_ids.append(role.copy())
            coordinates.append(coords.copy())
            targets.append(target.copy())
            row_hashes[str(row_index)] = _array_digest(flat, role, coords, target)
    if not all(np.isfinite(target).all() for target in targets):
            raise ValueError("Frozen native validation endpoint targets must be finite.")
    panel_arrays = {
        "row_indices": np.asarray(row_indices, dtype=np.int64),
        "native_flat_indices": np.stack(flat_ids),
        "role_ids": np.stack(role_ids),
        "coordinates_D": np.stack(coordinates),
        "targets_mps": np.stack(targets),
    }
    panel_hash = _array_digest(*panel_arrays.values())
    panel_name = "dev24_q8192_native_endpoint_panel.npz" if board_scope == "development" else "full_valid90_q8192_native_endpoint_panel.npz"
    panel_path = output / panel_name
    np.savez_compressed(panel_path, **panel_arrays)
    panel_description = {
        "path": str(panel_path),
        "sha256": _sha256_file(panel_path),
        "panel_sha256": panel_hash,
        "board_scope": board_scope,
        "split": "validation",
        "profile_id": provider.profile_id,
        "row_count": len(endpoint_rows),
        "row_order": endpoint_rows,
        "query_count_per_row": 8192,
        "role_query_counts": endpoint_role_query_counts(
            panel_arrays["role_ids"],
            tuple(provider.role_query_counts),
            expected_query_count=8192,
        ),
        "role_query_counts_basis": "counts derived from each saved endpoint row's role_ids; per-row Q8192",
        "row_native_panel_sha256": row_hashes,
        "coordinate_frame": "wind_aligned_rotor_diameters_v1; native x downstream, y crosswind, z vertical; no second rotation",
        "target_units": "m/s",
        "native_ids": "flat C-order IDs in (z,y,x) cell storage",
    }
    all_models = {
        **{label: (model, path, payload, "native_content_hypergraph")
           for label, (model, path, payload) in new_models.items()},
        **{label: (model, path, payload, "retained_forward_model")
           for label, (model, path, payload) in reference_models.items()},
    }
    if len(all_models) != len(new_models) + len(reference_models):
        raise ValueError("Endpoint model labels must be unique across native and retained checkpoints.")
    results: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    for label, (model, path, payload, family) in all_models.items():
        records: list[dict[str, Any]] = []
        predictions_hash: list[str] = []
        for batch in batches:
            receivers = batch.receivers
            if family == "native_content_hypergraph":
                architecture = str(model.get_config()["architecture"])
                scene = provider.make_scene(batch.scene_inputs)
                with torch.no_grad():
                    prediction_object, _ = provider.predict_native(
                        model, scene, receivers, architecture, "joint", epoch=int(payload["epoch"])
                    )
                prediction = prediction_object.main_mps.detach().cpu().numpy()[...].astype(np.float32, copy=False)
            else:
                row = int(batch.case_keys[0])
                case = reference_view.run(row)
                initial = case_batch(case, case.module_centers)
                with torch.no_grad():
                    prepared = model.prepare_case(initial)
                prediction = decode_retained_query_tiles(
                    model,
                    prepared,
                    receivers.coordinates_D[0],
                    receivers.features[0],
                    tile_size=query_tile_size,
                )[None]
            predictions_hash.extend(
                _array_digest(np.asarray(prediction[index], dtype=np.float32))
                for index in range(len(batch.case_keys))
            )
            for index, row in enumerate(batch.case_keys):
                metadata = dict(batch.targets.case_metadata[index])
                records.append(_role_metrics_row(
                    provider,
                    metadata,
                    prediction[index],
                    batch.targets.velocity_mps[index],
                    receivers.role_ids[index],
                    receivers.coordinates_D[index],
                ))
        if [int(record["row_index"]) for record in records] != endpoint_rows:
            raise ValueError(f"Endpoint metrics for {label} missed or reordered a {board_scope} validation row.")
        model_metadata[label] = _checkpoint_metadata(label, path, payload, model, family)
        results[label] = {
            "checkpoint": model_metadata[label],
            "endpoint_panel_sha256": panel_hash,
            "prediction_row_sha256": dict(zip(map(str, endpoint_rows), predictions_hash)),
            "row_metrics": records,
            "field_score_mean": float(np.mean([record["field_score"] for record in records])),
            "height_profile_score_mean": float(np.mean([record["height_profile_score"] for record in records])),
            "role_summary": summarize_role_metrics(records),
        }
    return results, {"panel": panel_description, "models": model_metadata}


def run_evaluation(
    new_paths: Mapping[str, Path],
    reference_paths: Mapping[str, Path],
    output: Path,
    *,
    query_tile_size: int = 1024,
    endpoint_microbatch_cases: int = 1,
    board_scope: str = "development",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    if board_scope not in ("development", "full_train1000"):
        raise ValueError("Wind native reference board scope must be development or full_train1000.")
    if not new_paths:
        raise ValueError("At least one corrected content-hypergraph checkpoint is required as a native provider anchor.")
    if query_tile_size < 1 or endpoint_microbatch_cases < 1:
        raise ValueError("Query tile and endpoint case-microbatch sizes must be positive.")
    started = time.perf_counter()
    first_label = next(iter(new_paths))
    first_model, provider, first_payload = load_evaluation_components(new_paths[first_label], "cpu")
    if next(first_model.parameters()).device.type != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError("Wind native reference evaluation is restricted to CPU before separate authorization.")
    if board_scope == "development":
        _assert_dev24_provider(provider)
    else:
        if len(new_paths) != 1:
            raise ValueError("Full VALID90 comparison accepts only the matched A-H6 full_train1000 checkpoint.")
        full_binding = _assert_full_train1000_provider(provider)
        if (
            int(first_payload.get("epoch", first_payload.get("current_epoch", -1))) != 1000
            or first_model.get_config().get("architecture") != "A-H6"
        ):
            raise ValueError("Full VALID90 reference board requires the literal A-H6 epoch-1000 checkpoint.")
        expected_references = {
            "wind2102_e1000",
            "wind2103_e1000",
            "wind2102_mature_e2440",
            "wind2103_mature_e2475",
        }
        if set(reference_paths) != expected_references:
            raise ValueError("Full VALID90 board requires literal e1000 and mature 2102/2103 references.")
    new_models: dict[str, tuple[Any, Path, Mapping[str, Any]]] = {
        first_label: (first_model, new_paths[first_label], first_payload)
    }
    reference_binding = provider.identity_payload()
    expected_adapter_state = json.dumps(reference_binding["adapter_state"], sort_keys=True)
    for label, checkpoint in list(new_paths.items())[1:]:
        model, other_provider, payload = load_evaluation_components(checkpoint, "cpu")
        identity = other_provider.identity_payload()
        if (
            identity["subset_manifest_sha256"] != reference_binding["subset_manifest_sha256"]
            or identity["validation_rows"] != reference_binding["validation_rows"]
            or json.dumps(identity["adapter_state"], sort_keys=True) != expected_adapter_state
        ):
            raise ValueError(f"Matched Wind checkpoints do not share the exact {board_scope} and transform binding.")
        new_models[label] = (model, checkpoint, payload)
        del other_provider
    if board_scope == "full_train1000" and (
        reference_binding.get("test_row_indices_sha256") != full_binding.get("test_row_indices_sha256")
        or reference_binding.get("normalizer_sha256") != full_binding.get("normalizer_sha256")
    ):
        raise ValueError("Full VALID90 provider identity changed after scope validation.")
    output.mkdir(parents=True)
    rows = tuple(int(row) for row in provider.validation_rows)
    counts = {row: int(provider.view.metadata["n_turbines"][row]) for row in rows}
    plane_rows = select_low_high_development_rows(rows, counts)
    reference_models, reference_payloads, reference_view = _prepare_retained_models(
        reference_paths, provider, rows
    )
    plane_metadata: dict[str, Any] = {}
    for label, row in plane_rows.items():
        case = (
            provider.native_case_for_development_row(row)
            if board_scope == "development" else provider.view.run(row)
        )
        scene_inputs = provider.adapter.scene_inputs(case)
        scene = provider.adapter.make_scene_batch((scene_inputs,))
        plane = native_hub_plane(case)
        cut_bundle = native_downstream_crosswind_cuts(case, scene_inputs)
        plane_coordinates = plane["coordinates_D"].reshape(-1, 3)
        plane_flat = plane["flat_indices"].reshape(-1)
        cut_coordinates = np.concatenate([item["coordinates_D"] for item in cut_bundle["cuts"]], axis=0)
        cut_flat = np.concatenate([item["flat_indices"] for item in cut_bundle["cuts"]], axis=0)
        query_coordinates = np.concatenate((plane_coordinates, cut_coordinates), axis=0)
        query_flat = np.concatenate((plane_flat, cut_flat), axis=0)
        predictions: dict[str, np.ndarray] = {}
        for model_label, (model, _, _) in new_models.items():
            with torch.no_grad():
                organization = model.prepare(scene)
            values = decode_content_query_tiles(
                model,
                provider.adapter,
                scene_inputs,
                scene,
                organization,
                query_coordinates,
                query_flat,
                tile_size=query_tile_size,
                support=case.support,
            )
            predictions[model_label] = values
        plane_count = len(plane_flat)
        plane_predictions = {key: value[:plane_count].reshape(*plane["target_mps"].shape)
                             for key, value in predictions.items()}
        target_valid = np.isfinite(plane["target_mps"]).all(axis=-1)
        common_valid = target_valid.copy()
        for value in plane_predictions.values():
            common_valid &= np.isfinite(value).all(axis=-1)
        if not np.any(common_valid):
            raise FloatingPointError(f"No common finite native hub-plane support remains for row {row}.")
        arrays: dict[str, np.ndarray] = {
            "native_flat_indices": plane["flat_indices"],
            "coordinates_D": plane["coordinates_D"],
            "target_mps": plane["target_mps"],
            "target_valid": target_valid,
            "common_valid": common_valid,
        }
        for model_label, value in plane_predictions.items():
            arrays[f"prediction_{model_label}_mps"] = value
            arrays[f"error_{model_label}_mps"] = np.where(common_valid[..., None], value - plane["target_mps"], np.nan)
        cut_prediction_arrays: dict[str, np.ndarray] = {}
        cursor = plane_count
        for cut_index, item in enumerate(cut_bundle["cuts"]):
            query_count = len(item["flat_indices"])
            for model_label, value in predictions.items():
                key = f"prediction_{model_label}_mps"
                cut_prediction_arrays.setdefault(key, []).append(value[cursor:cursor + query_count])
            cursor += query_count
        cut_targets = np.stack([item["target_mps"] for item in cut_bundle["cuts"]])
        cut_coords = np.stack([item["coordinates_D"] for item in cut_bundle["cuts"]])
        cut_ids = np.stack([item["flat_indices"] for item in cut_bundle["cuts"]])
        cut_pred_arrays = {key: np.stack(values) for key, values in cut_prediction_arrays.items()}
        cut_valid = np.isfinite(cut_targets).all(axis=-1)
        for value in cut_pred_arrays.values():
            cut_valid &= np.isfinite(value).all(axis=-1)
        arrays.update({
            "crosswind_cut_flat_indices": cut_ids,
            "crosswind_cut_coordinates_D": cut_coords,
            "crosswind_cut_targets_mps": cut_targets,
            "crosswind_cut_common_valid": cut_valid,
        })
        for key, value in cut_pred_arrays.items():
            arrays[key.replace("prediction_", "crosswind_cut_prediction_")] = value
            model_label = key[len("prediction_"):-len("_mps")]
            arrays[f"crosswind_cut_error_{model_label}_mps"] = np.where(
                cut_valid[..., None], value - cut_targets, np.nan
            )
        archive_path = output / f"native_hub_{label}_row_{row:03d}.npz"
        np.savez_compressed(archive_path, **arrays)
        source = np.asarray(cut_bundle["source_center_D"], dtype=np.float64)
        plane_metadata[label] = {
            "row_index": int(row),
            "case": str(case.case),
            "layout_index": int(case.layout_index),
            "wind_direction_deg": float(case.wind_direction_deg),
            "n_turbines": int(case.n_turbines),
            "selection": (
                "minimum/maximum turbine count from input-only exposed validation rows; prefer row 69/426 when tied"
            ),
            "hub_plane": {
                "fixed_axis": "z",
                "requested_z_m": float(plane["requested_z_m"]),
                "actual_stored_z_m": float(plane["actual_z_m"]),
                "native_z_index": int(plane["z_index"]),
                "grid_shape_y_x": list(plane["target_mps"].shape[:2]),
                "native_ids_sha256": _array_digest(plane["flat_indices"]),
                "plane_values_sha256": _array_digest(plane["flat_indices"], plane["coordinates_D"], plane["target_mps"]),
                "stored_native_plane": True,
                "xy_projection_or_slab_average": False,
            },
            "upstream_source": {
                "selection": "minimum native downstream x/D among active geometry sources; literal source ID breaks ties",
                "source_index": int(cut_bundle["source_index"]),
                "source_id": int(cut_bundle["source_id"]),
                "center_D": source.astype(float).tolist(),
                "all_active_source_ids": np.asarray(scene_inputs.module_ids)[
                    np.asarray(scene_inputs.module_present) > 0.5
                ].astype(int).tolist(),
            },
            "crosswind_cuts": [
                {
                    key: value for key, value in item.items()
                    if key not in {"flat_indices", "coordinates_D", "target_mps"}
                } | {
                    "flat_ids_sha256": _array_digest(item["flat_indices"]),
                    "coordinates_and_target_sha256": _array_digest(
                        item["flat_indices"], item["coordinates_D"], item["target_mps"]
                    ),
                    "stored_native_x_and_hub_z": True,
                    "crosswind_axis": "native y",
                }
                for item in cut_bundle["cuts"]
            ],
            "common_valid_count": int(common_valid.sum()),
            "native_archive": str(archive_path),
            "native_archive_sha256": _sha256_file(archive_path),
            "field_components": ["Ux", "Uy", "Uz"],
            "field_units": "m/s",
            "coordinate_frame": "wind_aligned_rotor_diameters_v1",
            "raw_model_metrics": {
                model_label: {
                    "component_rmse_mps_on_common_support": np.sqrt(np.mean(
                        np.square(value[common_valid] - plane["target_mps"][common_valid]), axis=0
                    )).astype(float).tolist(),
                    "vector_rmse_mps_on_common_support": float(np.sqrt(np.mean(np.sum(
                        np.square(value[common_valid] - plane["target_mps"][common_valid]), axis=-1
                    )))),
                }
                for model_label, value in plane_predictions.items()
            },
        }
    endpoint_results, endpoint_metadata = _evaluate_endpoints(
        provider,
        new_models,
        {label: (model, reference_paths[label], reference_payloads[label])
         for label, model in reference_models.items()},
        reference_view,
        output,
        board_scope=board_scope,
        microbatch_cases=endpoint_microbatch_cases,
        query_tile_size=query_tile_size,
    )
    result = {
        "schema": (
            "wind_native_reference_evaluation_v1"
            if board_scope == "development" else "wind_native_full_valid90_reference_evaluation_v1"
        ),
        "status": "completed",
        "board_scope": board_scope,
        "partition_scope": (
            "wind_shared_fixed24_v1 exposed DEV24 only"
            if board_scope == "development" else
            "original seed-42 TRAIN420/VALID90 full_train1000; exposed VALID90 only; TEST targets locked"
        ),
        "dataset_binding": {
            "subset_id": provider.profile_id,
            "manifest_sha256": provider.training_fingerprint,
            "protocol": reference_binding["protocol"],
            "train_row_indices_sha256": reference_binding["train_row_indices_sha256"],
            "validation_row_indices": rows,
            "validation_row_indices_sha256": reference_binding["validation_row_indices_sha256"],
            "validation_layout_indices": reference_binding["validation_layout_indices"],
            "coordinate_frame": "wind_aligned_rotor_diameters_v1; native x downstream, y crosswind, z vertical; no second rotation",
            "channel_order": ["Ux", "Uy", "Uz"],
            "velocity_units": "m/s",
            "wind_test_target_values_read": False,
            **({
                "test_row_indices_sha256": reference_binding["test_row_indices_sha256"],
                "test_target_values_read": False,
            } if board_scope == "full_train1000" else {}),
        },
        "provider_identity": reference_binding,
        "plane_rows": plane_metadata,
        "endpoint_panel": endpoint_metadata["panel"],
        "endpoint_models": endpoint_results,
        "checkpoints": endpoint_metadata["models"],
        "execution": {
            "device": "cpu",
            "query_tile_size": query_tile_size,
            "endpoint_microbatch_cases": endpoint_microbatch_cases,
            "optimizer_updates": 0,
            "solver_attempts": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "training": False,
        },
    }
    destination = output / "result.json"
    destination.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-checkpoint", action="append", default=[], metavar="LABEL=PATH",
                        help="Corrected self-contained content-hypergraph checkpoint; repeat for A-D/A-H6.")
    parser.add_argument("--reference-checkpoint", action="append", default=[], metavar="LABEL=PATH",
                        help="Retained WindFarm forward checkpoint; repeat for classic/dense references.")
    parser.add_argument("--run2202-checkpoint", action="append", default=[], metavar="LABEL=PATH",
                        help="Literal Run2202 refined checkpoint; accepted only in CPU cost mode.")
    parser.add_argument("--cost-panel-result", type=Path,
                        help="Run matched low/high-M CPU timing on a saved fixed DEV24 Q8192 panel.")
    parser.add_argument("--board-scope", choices=("development", "full_train1000"), default="development",
                        help="Use the strict fixed DEV24 board or the separate full TRAIN420/VALID90 board.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-tile-size", type=int, default=1024)
    parser.add_argument("--endpoint-microbatch-cases", type=int, default=1)
    args = parser.parse_args(argv)
    new_paths = parse_named_paths(args.new_checkpoint, "--new-checkpoint")
    reference_paths = parse_named_paths(args.reference_checkpoint, "--reference-checkpoint")
    run2202_paths = parse_named_paths(args.run2202_checkpoint, "--run2202-checkpoint")
    if args.cost_panel_result is not None:
        if args.board_scope != "development":
            parser.error("CPU cost-panel mode is bound only to the saved fixed DEV24 receiver panel.")
        result = run_native_cpu_cost_measurements(
            new_paths,
            reference_paths,
            run2202_paths,
            args.cost_panel_result,
            args.output.expanduser().resolve(),
            query_tile_size=args.query_tile_size,
        )
        print(json.dumps({
            "status": result["status"],
            "output": str(args.output.expanduser().resolve()),
            "models": list(result["models"]),
            "rows": result["source_panel"]["timed_rows"],
            "warmups_per_model_row": result["timing_contract"]["warmups_per_model_row"],
            "timed_repeats_per_model_row": result["timing_contract"]["timed_repeats_per_model_row"],
        }, sort_keys=True), flush=True)
        return 0
    if run2202_paths:
        parser.error("--run2202-checkpoint requires --cost-panel-result; raw Run2202 states are cost-only here.")
    result = run_evaluation(
        new_paths,
        reference_paths,
        args.output.expanduser().resolve(),
        query_tile_size=args.query_tile_size,
        endpoint_microbatch_cases=args.endpoint_microbatch_cases,
        board_scope=args.board_scope,
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.expanduser().resolve()),
        "plane_rows": {key: value["row_index"] for key, value in result["plane_rows"].items()},
        "endpoint_models": list(result["endpoint_models"]),
        "endpoint_panel_sha256": result["endpoint_panel"]["panel_sha256"],
        "elapsed_seconds": result["execution"]["elapsed_seconds"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
