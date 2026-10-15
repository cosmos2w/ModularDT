#!/usr/bin/env python3
"""Measure Round3 Wind source-read panels on fixed native five-role queries.

This evaluator uses the checkpoint's own TRAIN-fitted adapter, a read-only
shared geometry catalogue, exposed VALID rows (or a small explicit TRAIN
fixture), and the native volume/hub-slab/downstream/near/background sampler.
It records physical per-role errors and bounded native hub planes/crosswind
cuts; it does not train, write caches, or read TEST targets.
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
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "tools", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from content_hypergraph_evaluate import load_evaluation_components
from honf_runtime.unified_training import SamplingKey
from wind_native_reference_evaluate import (
    _array_digest,
    _sha256_file,
    native_downstream_crosswind_cuts,
    native_hub_plane,
    select_low_high_development_rows,
)
from windfarm.content_hypergraph import WindContentReceivers
from windfarm.data import case_batch
from windfarm.geometry import support_features
from windfarm.training.content_hypergraph_task import (
    ROLE_NAMES,
    WindContentHypergraphTask,
    WindSourceReadTaskV1,
)


ROLE_METRIC_SCHEMA = "wind_source_read_native_role_metrics_v1"
POLICIES = (
    "full",
    "fitted",
    "attn95",
    "attn99",
    "nearest95",
    "nearest_fitted",
    "directional_fitted",
    "shuffle_fitted",
    "nearest99",
    "directional95",
    "directional99",
    "shuffle95",
    "shuffle99",
)
NATIVE_ROLE_QUERY_COUNT = 8192
MAX_TRAIN_FIXTURE_ROWS = 8
SOURCE_MODULES = {
    "evaluator": Path(__file__),
    "native_reference_helper": PROJECT / "tools" / "wind_native_reference_evaluate.py",
    "checkpoint_loader": PROJECT / "tools" / "content_hypergraph_evaluate.py",
    "retained_native_loader": PROJECT / "Case_WindFarm/src/windfarm/workflows/evaluate_forward.py",
    "wind_provider": PROJECT / "Case_WindFarm/src/windfarm/training/content_hypergraph_task.py",
    "wind_adapter": PROJECT / "Case_WindFarm/src/windfarm/content_hypergraph.py",
    "wind_normalization": PROJECT / "Case_WindFarm/src/windfarm/normalization.py",
    "wind_native_data_view": PROJECT / "Case_WindFarm/src/windfarm/data.py",
    "native_role_sampler": PROJECT / "Case_WindFarm/src/windfarm/workflows/joint_forward.py",
    "native_role_cache": PROJECT / "Case_WindFarm/src/windfarm/workflows/native_role_cache.py",
    "sampling_runtime": PROJECT / "src/honf_runtime/unified_training.py",
    "core_model": PROJECT / "src/honf_forward_core/content_hypergraph/model_v2.py",
    "direct_reader": PROJECT / "src/honf_forward_core/content_hypergraph/readers_v2.py",
    "source_reader_variants": PROJECT / "src/honf_forward_core/content_hypergraph/source_reader_variants.py",
    "core_decoding": PROJECT / "src/honf_forward_core/content_hypergraph/decoding.py",
    "source_read_decoder": PROJECT / "src/honf_forward_core/content_hypergraph/source_read_decoder_v1.py",
    "source_read_plans": PROJECT / "src/honf_forward_core/content_hypergraph/source_read_plans.py",
    "source_read_sparse": PROJECT / "src/honf_forward_core/content_hypergraph/source_read_sparse.py",
    "source_read_response": PROJECT / "src/honf_forward_core/content_hypergraph/source_read_response.py",
}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_module_hashes() -> dict[str, dict[str, str]]:
    hashes = {}
    for name, path in SOURCE_MODULES.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Required evaluation source module is missing: {resolved}")
        hashes[name] = {"path": str(resolved), "sha256": _sha256_file(resolved)}
    return hashes


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


def parse_case_keys(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    try:
        rows = tuple(int(piece.strip()) for piece in raw.split(","))
    except ValueError as exc:
        raise ValueError("--case-keys must be a comma-separated list of integer native row IDs.") from exc
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("--case-keys must be nonempty and contain unique native row IDs.")
    return rows


def choose_case_keys(provider: Any, split: str, requested: Sequence[int] | None) -> tuple[int, ...]:
    if split not in ("validation", "train-fixture"):
        raise ValueError("split must be validation or train-fixture.")
    available = tuple(
        int(value) for value in (provider.validation_rows if split == "validation" else provider.train_rows)
    )
    if requested is not None:
        selected = tuple(int(value) for value in requested)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(value not in set(available) for value in selected)
        ):
            raise ValueError(f"Requested {split} rows must be unique members of the sealed provider split.")
        if split == "train-fixture" and len(selected) > MAX_TRAIN_FIXTURE_ROWS:
            raise ValueError(f"TRAIN fixture requests are limited to {MAX_TRAIN_FIXTURE_ROWS} explicit rows.")
        return selected
    if split == "validation":
        return available
    # The default TRAIN fixture is input-only: minimum and maximum active M
    # among the sealed TRAIN rows, with stable native-row tie breaks.
    counts = {int(row): int(provider.view.metadata["n_turbines"][row]) for row in available}
    low_m = min(counts.values())
    high_m = max(counts.values())
    low = min(row for row in available if counts[row] == low_m)
    high_candidates = [row for row in available if counts[row] == high_m and row != low]
    if high_candidates:
        high = max(high_candidates)
    else:
        alternatives = [row for row in available if row != low]
        if not alternatives:
            return (low,)
        high = alternatives[-1]
    return (low, high)


def make_native_role_panel_batch(
    provider: WindSourceReadTaskV1,
    row: int,
    *,
    split: str,
    batch_index: int = 0,
) -> Any:
    """Use the original five-role sampler while bypassing the formal 2-region override."""
    if not isinstance(provider, WindSourceReadTaskV1):
        raise TypeError("Native source-read panels require a bound WindSourceReadTaskV1 provider.")
    if split not in ("validation", "train-fixture"):
        raise ValueError("Native five-role panels support validation or train-fixture only.")
    if split == "train-fixture" and len(provider.train_rows) < 1:
        raise ValueError("The sealed provider has no TRAIN rows for a bounded fixture.")
    if isinstance(batch_index, bool) or int(batch_index) < 0:
        raise ValueError("Native role panel batch_index must be a nonnegative integer.")
    stage = "native_endpoint" if split == "validation" else "source_read_train_fixture"
    key = SamplingKey(
        int(provider.sampling_seed),
        0,
        0,
        int(batch_index),
        stage,
        "shared",
        sampling_version=provider.sampling_version,
        dataset_id=provider.sampling_dataset_id,
    )
    # WindFormalA2DTaskV1 overrides this sampler to construct 6144+2048 and
    # VALID Q10240. Calling the original WindContentHypergraphTask method
    # explicitly retains the separate native diagnostic Q8192 five-role board.
    return WindContentHypergraphTask._make_batch(
        provider,
        (int(row),),
        key,
        validation=split == "validation",
        stream_validation=split == "validation",
        query_count=NATIVE_ROLE_QUERY_COUNT,
        track_samples=False,
    )


def _metric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Role metric summaries require nonempty finite row values.")
    worst_i = int(np.argmax(array))
    return {
        "mean": float(np.mean(array, dtype=np.float64)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "worst": float(array[worst_i]),
        "worst_row_position": worst_i,
        "count": int(array.size),
    }


def physical_role_metrics(
    metadata: Mapping[str, Any],
    prediction_mps: np.ndarray,
    target_mps: np.ndarray,
    role_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute raw physical RMSE by role and vector component; no role scales enter."""
    prediction = np.asarray(prediction_mps, dtype=np.float64)
    target = np.asarray(target_mps, dtype=np.float64)
    roles = np.asarray(role_ids, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("Wind native predictions and targets must align as [Q,3] physical m/s arrays.")
    if roles.shape != (prediction.shape[0],) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Wind native role IDs and finite prediction/target arrays must align as [Q].")
    result: dict[str, Any] = {**dict(metadata), "roles": {}}
    for role_index, role in enumerate(ROLE_NAMES):
        selected = roles == role_index
        if not selected.any():
            raise ValueError(f"Native fixed Q8192 panel has no queries for role {role!r}.")
        delta = prediction[selected] - target[selected]
        component_rmse = np.sqrt(np.mean(np.square(delta), axis=0, dtype=np.float64))
        vector_rmse = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=-1), dtype=np.float64)))
        result["roles"][role] = {
            "component_rmse_mps": component_rmse.astype(float).tolist(),
            "vector_rmse_mps": vector_rmse,
            "query_count": int(selected.sum()),
        }
    return result


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"row_count": len(rows)}
    if not rows:
        return result
    roles = tuple(rows[0]["roles"])
    result["roles"] = {}
    for role in roles:
        vector_values = np.asarray([row["roles"][role]["vector_rmse_mps"] for row in rows], dtype=np.float64)
        components = np.asarray([row["roles"][role]["component_rmse_mps"] for row in rows], dtype=np.float64)
        result["roles"][role] = {
            "vector_rmse_mps": _metric_summary(vector_values),
            "component_rmse_mps": {
                component: _metric_summary(components[:, index]) for index, component in enumerate(("Ux", "Uy", "Uz"))
            },
        }
    return result


def summarize_physical_role_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("A source-read native role summary requires at least one case row.")
    layout_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    module_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        layout_groups[str(int(row["layout_index"]))].append(row)
        module_groups[str(int(row["n_turbines"]))].append(row)
    return {
        "physical_metric_scope": "raw m/s RMSE; no five-role training scales or objective weights",
        "overall": _summarize_group(rows),
        "by_layout_index": {key: _summarize_group(value) for key, value in sorted(layout_groups.items())},
        "by_active_module_count_M": {
            key: _summarize_group(value) for key, value in sorted(module_groups.items(), key=lambda item: int(item[0]))
        },
    }


def _build_receivers(
    provider: Any,
    scene_inputs: Any,
    support: Any,
    coordinates: np.ndarray,
    flat_indices: np.ndarray,
    role_ids: np.ndarray,
) -> WindContentReceivers:
    coords = np.asarray(coordinates, dtype=np.float32)
    flat = np.asarray(flat_indices, dtype=np.int64)
    roles = np.asarray(role_ids, dtype=np.int8)
    features = provider.adapter.query_features(coords, support)
    local_query, local_module = provider.adapter.packed_local_pair_indices(scene_inputs, coords)
    return WindContentReceivers(
        coordinates_D=coords[None],
        features=np.asarray(features, dtype=np.float32)[None],
        flat_indices=flat[None],
        role_ids=roles[None],
        scene_keys=(scene_inputs.scene_key,),
        local_batch_indices=np.zeros(local_query.size, dtype=np.int64),
        local_query_indices=local_query,
        local_module_indices=local_module,
    )


def _decode_with_counts(
    model: Any,
    provider: Any,
    scene_inputs: Any,
    scene: Any,
    organization: Any,
    case: Any,
    coordinates: np.ndarray,
    flat_indices: np.ndarray,
    role_ids: np.ndarray,
    *,
    tile_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    receivers = _build_receivers(provider, scene_inputs, case.support, coordinates, flat_indices, role_ids)
    query = provider.adapter.make_query_batch(scene, receivers)
    model.validate_prepared_state(organization, query)
    from content_hypergraph_inspect import slice_query

    prediction = np.empty((len(coordinates), 3), dtype=np.float32)
    counts: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for start in range(0, len(coordinates), tile_size):
            stop = min(start + tile_size, len(coordinates))
            tile = slice_query(query, np.arange(start, stop, dtype=np.int64))
            output = model._decode_validated(organization, tile)
            physical, _profile, _residual = provider.adapter.physical_velocity(output.nonlinear_field, tile.coordinates)
            prediction[start:stop] = physical[0].detach().cpu().numpy().astype(np.float32, copy=False)
            for name, value in output.work_counts.items():
                counts[name] += int(value)
    return prediction, dict(counts)


def _validate_policy_model(model: Any, policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"Unknown source-read policy {policy!r}.")
    if policy == "fitted" or policy.endswith("_fitted"):
        controller = getattr(getattr(model, "direct_reader", None), "source_read", None)
        if controller is None:
            raise ValueError("The fitted policy requires a source-read checkpoint with a TRAIN-fitted controller.")
        if not bool(controller.trained) or not bool(controller.calibrated):
            raise ValueError("The fitted policy requires persisted TRAIN-trained/calibrated metadata.")


def _provider_data_binding(provider: Any) -> dict[str, Any]:
    identity = dict(provider.identity_payload())
    adapter_state = identity.get("adapter_state")
    transform = {
        "normalizer_and_background_source": identity.get("normalizer_and_background_source"),
        "normalizer_sha256": identity.get("normalizer_sha256"),
        "background_profile_sha256": identity.get("background_profile_sha256"),
        "adapter_state_sha256": _sha256_json(adapter_state),
    }
    return {
        "identity": identity,
        "identity_sha256": _sha256_json(identity),
        "transform_binding": transform,
    }


def _validate_source_provider(provider: Any) -> Mapping[str, Any]:
    if not isinstance(provider, WindSourceReadTaskV1):
        raise TypeError("This evaluator accepts only checkpoints bound to WindSourceReadTaskV1.")
    identity = provider.identity_payload()
    protocol = identity.get("protocol")
    if protocol not in ("source_read_development_v1", "source_read_full1000"):
        raise ValueError("Wind native source-read evaluation requires a declared source-read protocol scope.")
    expected_rows = (72, 24) if protocol == "source_read_development_v1" else (420, 90)
    if (len(provider.train_rows), len(provider.validation_rows)) != expected_rows:
        raise ValueError("Wind source-read evaluation requires the exact TRAIN72/DEV24 or TRAIN420/VALID90 scope.")
    if identity.get("wind_test_target_values_read") is not False:
        raise ValueError("Source-read evaluation requires the provider's explicit Wind TEST target lock.")
    if identity.get("source_reader_variant") not in ("D0", "D-scale", "S-Utility", "S-Complete"):
        raise ValueError("Source-read evaluation requires a declared supported reader variant.")
    if provider.role_component_scales is not None or provider.role_scale_calibration is not None:
        raise ValueError("Native five-role diagnostics must not load the historical role-scale cache dependency.")
    cache = provider.catalogue_cache
    if cache is None or not bool(cache.read_only_persistent):
        raise ValueError("Native source-read evaluation requires the shared catalogue in read-only mode.")
    if cache.persistent_dir is not None and not cache.read_only_persistent:
        raise ValueError("Native source-read evaluation requires a read-only persistent geometry catalogue.")
    if len(provider.validation_rows) not in (24, 90):
        raise ValueError("Source-read Wind validation is restricted to the exposed DEV24 or VALID90 panel.")
    return identity


def _prepare_native_role_panels(provider: Any, rows: Sequence[int], split: str) -> dict[int, dict[str, Any]]:
    panels = {}
    for batch_index, row in enumerate(rows):
        batch = make_native_role_panel_batch(provider, int(row), split=split, batch_index=batch_index)
        receivers, targets = batch.receivers, batch.targets
        coords = np.asarray(receivers.coordinates_D[0], dtype=np.float32).copy()
        flat = np.asarray(receivers.flat_indices[0], dtype=np.int64).copy()
        roles = np.asarray(receivers.role_ids[0], dtype=np.int8).copy()
        target = np.asarray(targets.velocity_mps[0], dtype=np.float32).copy()
        if coords.shape != (NATIVE_ROLE_QUERY_COUNT, 3) or target.shape != (NATIVE_ROLE_QUERY_COUNT, 3):
            raise ValueError("Native five-role diagnostic panel must contain exactly Q8192 physical samples.")
        counts = {role: int(np.count_nonzero(roles == index)) for index, role in enumerate(ROLE_NAMES)}
        if any(value <= 0 for value in counts.values()) or sum(counts.values()) != NATIVE_ROLE_QUERY_COUNT:
            raise ValueError("Native Q8192 diagnostic panel must retain every canonical Wind role.")
        panel_digest = _array_digest(flat, roles, coords)
        panels[int(row)] = {
            "coordinates_D": coords,
            "native_flat_indices": flat,
            "role_ids": roles,
            "targets_mps": target,
            "role_query_counts": counts,
            "query_sampling_sha256": str(batch.auxiliary["query_sampling_sha256"]),
            "panel_identity_sha256": panel_digest,
            "target_values_sha256": _array_digest(target),
            "native_endpoint_sha256": _array_digest(flat, roles, coords, target),
            "sampling_key": {
                "seed": int(provider.sampling_seed),
                "epoch": 0,
                "batch_index": int(batch_index),
                "stage": "native_endpoint" if split == "validation" else "source_read_train_fixture",
                "variant": "shared",
                "sampling_version": str(provider.sampling_version),
                "dataset_id": str(provider.sampling_dataset_id),
            },
        }
    return panels


def _case_metadata(provider: Any, row: int) -> dict[str, Any]:
    metadata = provider.view.metadata
    direction_values = metadata.get("wind_direction_deg", metadata.get("wd_deg"))
    if direction_values is None:
        raise ValueError("Native Wind metadata is missing its canonical wd_deg direction array.")
    return {
        "row_index": int(row),
        "case": str(
            metadata.get("case_name", metadata.get("case", [f"native_row_{row}"] * len(metadata["layout_index"]))[row])
        ),
        "layout_index": int(metadata["layout_index"][row]),
        "wind_direction_deg": float(direction_values[row]),
        "n_turbines": int(metadata["n_turbines"][row]),
    }


def _select_plane_rows(provider: Any, rows: Sequence[int]) -> dict[str, int]:
    counts = {int(row): int(provider.view.metadata["n_turbines"][row]) for row in rows}
    return select_low_high_development_rows(tuple(map(int, rows)), counts)


def _plane_export_rows(provider: Any, rows: Sequence[int], *, enabled: bool) -> dict[str, int]:
    return _select_plane_rows(provider, rows) if enabled else {}


def _native_plane_bundle(
    model: Any, provider: Any, case: Any, scene_inputs: Any, scene: Any, organization: Any, *, tile_size: int
) -> dict[str, np.ndarray]:
    plane = native_hub_plane(case)
    cuts = native_downstream_crosswind_cuts(case, scene_inputs)
    plane_coords = np.asarray(plane["coordinates_D"], dtype=np.float32).reshape(-1, 3)
    plane_ids = np.asarray(plane["flat_indices"], dtype=np.int64).reshape(-1)
    cut_coords = np.concatenate([np.asarray(item["coordinates_D"], dtype=np.float32) for item in cuts["cuts"]])
    cut_ids = np.concatenate([np.asarray(item["flat_indices"], dtype=np.int64) for item in cuts["cuts"]])
    coordinates = np.concatenate((plane_coords, cut_coords), axis=0)
    flat = np.concatenate((plane_ids, cut_ids))
    role_ids = np.zeros(flat.shape, dtype=np.int8)
    prediction, _counts = _decode_with_counts(
        model,
        provider,
        scene_inputs,
        scene,
        organization,
        case,
        coordinates,
        flat,
        role_ids,
        tile_size=tile_size,
    )
    plane_size = len(plane_ids)
    plane_prediction = prediction[:plane_size].reshape(*plane["target_mps"].shape)
    target_valid = np.isfinite(plane["target_mps"]).all(axis=-1)
    arrays: dict[str, np.ndarray] = {
        "native_flat_indices": np.asarray(plane["flat_indices"], dtype=np.int64),
        "coordinates_D": np.asarray(plane["coordinates_D"], dtype=np.float32),
        "target_mps": np.asarray(plane["target_mps"], dtype=np.float32),
        "target_valid": target_valid,
        "hub_plane_prediction_mps": plane_prediction,
        "hub_plane_error_mps": np.where(target_valid[..., None], plane_prediction - plane["target_mps"], np.nan),
        "crosswind_cut_flat_indices": np.stack([item["flat_indices"] for item in cuts["cuts"]]).astype(np.int64),
        "crosswind_cut_coordinates_D": np.stack([item["coordinates_D"] for item in cuts["cuts"]]).astype(np.float32),
        "crosswind_cut_targets_mps": np.stack([item["target_mps"] for item in cuts["cuts"]]).astype(np.float32),
        "crosswind_cut_prediction_mps": prediction[plane_size:].reshape(3, -1, 3),
        "crosswind_cut_requested_offsets_D": np.asarray(
            [item["requested_offset_D"] for item in cuts["cuts"]], dtype=np.float64
        ),
        "crosswind_cut_actual_offsets_D": np.asarray(
            [item["actual_offset_D"] for item in cuts["cuts"]], dtype=np.float64
        ),
    }
    cut_targets = arrays["crosswind_cut_targets_mps"]
    cut_valid = np.isfinite(cut_targets).all(axis=-1)
    arrays["crosswind_cut_common_valid"] = cut_valid
    arrays["crosswind_cut_error_mps"] = np.where(
        cut_valid[..., None], arrays["crosswind_cut_prediction_mps"] - cut_targets, np.nan
    )
    return arrays


def _cache_summary(provider: Any) -> dict[str, Any]:
    cache = provider.catalogue_cache
    if cache is None:
        return {"configured": False, "persistent_writes": 0}
    summary = cache.summary()
    return {
        "configured": True,
        "persistent_dir": None if cache.persistent_dir is None else str(cache.persistent_dir),
        "persistent_read_only": bool(cache.read_only_persistent),
        "persistent_writes": int(cache.disk_write_count),
        "persistent_disk_hits": int(cache.disk_hit_count),
        "memory_builds": int(cache.build_count),
        "cache_hits": int(cache.hit_count),
        "cache_misses": int(cache.miss_count),
        "cached_bytes": int(summary["cached_bytes"]),
    }


def _assert_exact_panel_match(
    reference: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if tuple(reference) != tuple(candidate):
        raise ValueError(f"{label} did not use the same ordered native physical panel rows.")
    for row in reference:
        for key in (
            "panel_identity_sha256",
            "target_values_sha256",
            "native_endpoint_sha256",
            "query_sampling_sha256",
        ):
            if reference[row][key] != candidate[row][key]:
                raise ValueError(f"{label} native five-role panel mismatch at row {row} ({key}).")


def _assert_saved_native_panel_match(
    expected_result_path: Path,
    panels: Mapping[int, Mapping[str, Any]],
    rows: Sequence[int],
) -> dict[str, Any]:
    """Require byte-identical endpoint queries before comparing retained predictions."""
    path = expected_result_path.expanduser().resolve()
    expected_result = json.loads(path.read_text(encoding="utf-8"))
    if (
        expected_result.get("schema") != "wind_source_read_native_evaluation_v1"
        or expected_result.get("status") != "completed"
        or expected_result.get("query_panel", {}).get("wind_test_target_values_read") is not False
    ):
        raise ValueError("Reference comparison requires a completed, TEST-locked source-read panel result.")
    expected_rows = tuple(int(value) for value in expected_result.get("row_order", ()))
    selected_rows = tuple(int(value) for value in rows)
    if expected_rows != selected_rows:
        raise ValueError("Reference comparison rows/order differ from the saved source-read panel.")
    expected_hashes = expected_result.get("query_panel", {}).get("row_native_endpoint_sha256", {})
    if set(expected_hashes) != {str(row) for row in selected_rows}:
        raise ValueError("Saved source-read result lacks exact per-row native endpoint hashes.")
    for row in selected_rows:
        observed = str(panels[row]["native_endpoint_sha256"])
        expected = str(expected_hashes[str(row)])
        if observed != expected:
            raise ValueError(
                f"Reference comparison native endpoint mismatch at row {row}: "
                f"expected {expected}, observed {observed}."
            )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": expected_result["schema"],
        "row_count": len(selected_rows),
        "row_order": list(selected_rows),
        "all_native_endpoint_hashes_match": True,
    }


def _evaluate_retained_references(
    reference_checkpoint_paths: Mapping[str, Path],
    provider: Any,
    panels: Mapping[int, Mapping[str, Any]],
    rows: Sequence[int],
    *,
    query_tile_size: int,
) -> dict[str, Any]:
    """Run trusted retained Wind checkpoints on the same target-free native queries."""
    if not reference_checkpoint_paths:
        return {}
    from wind_native_reference_evaluate import (
        _prepare_retained_models,
        decode_retained_query_tiles,
    )

    reference_models, payloads, reference_view = _prepare_retained_models(
        reference_checkpoint_paths, provider, rows
    )
    results: dict[str, Any] = {}
    for label, model in reference_models.items():
        payload = payloads[label]
        normalization = payload.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError(f"Retained reference {label!r} has no checkpoint-owned normalization.")
        if model.velocity_transform.to_dict() != dict(normalization):
            raise ValueError(f"Retained reference {label!r} did not preserve its checkpoint-owned transform.")
        row_metrics: list[dict[str, Any]] = []
        for row in rows:
            panel = panels[int(row)]
            case = reference_view.run(int(row))
            # This BatchData carries only prescribed inputs; it contains no target_field.
            materialization = case_batch(case, case.module_centers)
            if materialization.target_field is not None:
                raise RuntimeError("Retained reference materialization unexpectedly contains a target field.")
            with torch.no_grad():
                prepared = model.prepare_case(materialization)
                features = support_features(panel["coordinates_D"], case.support)
                prediction = decode_retained_query_tiles(
                    model,
                    prepared,
                    panel["coordinates_D"],
                    features,
                    tile_size=query_tile_size,
                )
            row_metric = physical_role_metrics(
                _case_metadata(provider, int(row)),
                prediction,
                panel["targets_mps"],
                panel["role_ids"],
            )
            row_metric.update(
                panel_identity_sha256=panel["panel_identity_sha256"],
                target_values_sha256=panel["target_values_sha256"],
                query_sampling_sha256=panel["query_sampling_sha256"],
                native_endpoint_sha256=panel["native_endpoint_sha256"],
                role_query_counts=dict(panel["role_query_counts"]),
            )
            row_metrics.append(row_metric)
        checkpoint = Path(reference_checkpoint_paths[label]).expanduser().resolve()
        results[label] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "completed_epoch": int(payload.get("epoch", payload.get("current_epoch", -1))),
            "family": "retained_classic_checkpoint",
            "normalizer_sha256": _sha256_json(normalization),
            "transform_source": "reference checkpoint-owned normalization; not copied from source-read model",
            "input_batch_policy": "WindFarmNativeView + target-free case_batch",
            "role_metrics": summarize_physical_role_metrics(row_metrics),
            "row_metrics": row_metrics,
            "execution_evidence": {
                "native_role_query_predictions_saved": False,
                "optimizer_updates": 0,
                "solver_attempts": 0,
                "wall_runtime_benchmark_measured": False,
                "query_tile_size": int(query_tile_size),
                "target_field_present_in_materialization_batch": False,
            },
        }
    return results


def run_reference_comparison(
    provider_checkpoint: Path,
    reference_checkpoint_paths: Mapping[str, Path],
    expected_panel_result: Path,
    output: Path,
    *,
    split: str = "validation",
    case_keys: Sequence[int] | None = None,
    query_tile_size: int = 8192,
) -> dict[str, Any]:
    """Evaluate retained checkpoints against a previously measured source-read panel."""
    if not reference_checkpoint_paths:
        raise ValueError("Reference-only evaluation requires at least one retained checkpoint.")
    if split not in ("validation", "train-fixture"):
        raise ValueError("Reference-only evaluation is restricted to sealed validation or TRAIN fixture rows.")
    if query_tile_size < 1:
        raise ValueError("Query tile size must be positive.")
    if torch.cuda.is_initialized():
        raise RuntimeError("Wind retained reference evaluation must not initialize CUDA.")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Reference evidence is immutable; choose a fresh output directory: {output}")
    started = time.perf_counter()
    source_checkpoint = provider_checkpoint.expanduser().resolve()
    source_model, provider, source_payload = load_evaluation_components(source_checkpoint, "cpu")
    source_identity = _validate_source_provider(provider)
    rows = choose_case_keys(provider, split, case_keys)
    split_name = "validation" if split == "validation" else "train-fixture"
    cache_before = _cache_summary(provider)
    panels = _prepare_native_role_panels(provider, rows, split_name)
    cache_after_panel = _cache_summary(provider)
    if cache_after_panel["persistent_writes"] != cache_before["persistent_writes"]:
        raise RuntimeError("Reference panel generation attempted a persistent geometry cache write.")
    panel_binding = _assert_saved_native_panel_match(expected_panel_result, panels, rows)
    expected_payload = json.loads(Path(panel_binding["path"]).read_text(encoding="utf-8"))
    if expected_payload.get("dataset_scope") != source_identity.get("protocol"):
        raise ValueError("Reference source checkpoint and saved native panel have different protocol scopes.")
    reference_results = _evaluate_retained_references(
        reference_checkpoint_paths,
        provider,
        panels,
        rows,
        query_tile_size=query_tile_size,
    )
    cache_after = _cache_summary(provider)
    if cache_after["persistent_writes"] != cache_before["persistent_writes"]:
        raise RuntimeError("Retained reference evaluation attempted a persistent geometry cache write.")
    saved_source_models = expected_payload.get("models", {})
    source_model_bindings = {}
    for label, item in saved_source_models.items():
        identity = item.get("identity", {})
        effective = identity.get("model_config", {})
        source_model_bindings[label] = {
            "checkpoint": item.get("checkpoint"),
            "checkpoint_sha256": item.get("checkpoint_sha256"),
            "declared_recipe_model": identity.get("recipe", {}).get("model", {}),
            "effective_core_config_from_get_config": effective,
            "effective_core_query_mode": (
                "whole_query" if effective.get("reader_query_chunk_size") is None else "internal_reader_chunking"
            ),
            "external_evaluator_query_tile_size": expected_payload.get("execution", {}).get("query_tile_size"),
        }
    d0_identity = source_identity
    effective_source_config = source_model.get_config()
    result = {
        "schema": "wind_source_read_retained_reference_evaluation_v1",
        "status": "completed",
        "dataset_scope": str(d0_identity.get("protocol")),
        "split": split_name,
        "row_order": list(rows),
        "source_provider": {
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": _sha256_file(source_checkpoint),
            "completed_epoch": int(source_payload.get("epoch", source_payload.get("current_epoch", -1))),
            "variant": d0_identity.get("source_reader_variant"),
            "dataset_identity_sha256": _sha256_json(d0_identity),
            "declared_recipe_model": dict(d0_identity.get("recipe", {}).get("model", {})),
            "effective_core_config_from_get_config": dict(effective_source_config),
            "effective_core_query_mode": (
                "whole_query"
                if effective_source_config.get("reader_query_chunk_size") is None
                else "internal_reader_chunking"
            ),
        },
        "source_module_hashes": _source_module_hashes(),
        "source_board_binding": {
            **panel_binding,
            "source_result_model_bindings": source_model_bindings,
            "query_tile_semantics": (
                "The source board's tile is external evaluation batching; it is distinct from each model's "
                "effective core reader_query_chunk_size."
            ),
        },
        "query_panel": {
            "sampler": "WindContentHypergraphTask._make_batch five-role native sampler",
            "query_count_per_row": NATIVE_ROLE_QUERY_COUNT,
            "role_order": list(ROLE_NAMES),
            "role_query_counts": panels[rows[0]]["role_query_counts"],
            "row_native_endpoint_sha256": {
                str(row): panels[row]["native_endpoint_sha256"] for row in rows
            },
            "row_panel_identity_sha256": {
                str(row): panels[row]["panel_identity_sha256"] for row in rows
            },
            "row_target_values_sha256": {
                str(row): panels[row]["target_values_sha256"] for row in rows
            },
            "sampling_dataset_id": str(provider.sampling_dataset_id),
            "wind_test_target_values_read": False,
        },
        "retained_classic_references": reference_results,
        "execution": {
            "device": "cpu",
            "query_tile_size": int(query_tile_size),
            "query_tile_semantics": "external reference inference batching; full Q8192 per call at the default",
            "optimizer_updates": 0,
            "solver_attempts": 0,
            "persistent_geometry_cache_writes": 0,
            "wall_runtime_benchmark_measured": False,
            "wind_test_target_values_read": False,
            "catalogue_cache_before": cache_before,
            "catalogue_cache_after_panel": cache_after_panel,
            "catalogue_cache_after_reference": cache_after,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def run_evaluation(
    checkpoint_paths: Mapping[str, Path],
    output: Path,
    *,
    policies: Sequence[str] | None = None,
    split: str = "validation",
    case_keys: Sequence[int] | None = None,
    reference_checkpoint_paths: Mapping[str, Path] | None = None,
    device: str = "cpu",
    executor: str = "packed",
    operating_point: int | None = None,
    query_tile_size: int = 8192,
    export_native_planes: bool = True,
) -> dict[str, Any]:
    if not checkpoint_paths:
        raise ValueError("At least one source-read checkpoint is required.")
    if device != "cpu":
        raise ValueError("Source-read Wind native accuracy evaluation is CPU-only; cost campaigns use a separate tool.")
    if torch.cuda.is_initialized():
        raise RuntimeError("Wind native source-read evaluation must not initialize CUDA.")
    if executor not in ("packed", "packet", "dense_masked"):
        raise ValueError("executor must be packed, packet, or dense_masked.")
    if query_tile_size < 1:
        raise ValueError("Query tile size must be positive.")
    policy_values = tuple(policies or ("full",))
    if (
        not policy_values
        or len(set(policy_values)) != len(policy_values)
        or any(value not in POLICIES for value in policy_values)
    ):
        raise ValueError("Policies must be a nonempty unique subset of the declared source-read modes.")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Source-read native evidence is immutable; choose a fresh output directory: {output}")

    started = time.perf_counter()
    loaded_models: dict[str, tuple[Any, Any, Mapping[str, Any], Path]] = {}
    providers: dict[str, Any] = {}
    identities: dict[str, dict[str, Any]] = {}
    first_provider = None
    first_rows: tuple[int, ...] | None = None
    split_name = "validation" if split == "validation" else "train-fixture"
    for label, checkpoint in checkpoint_paths.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, provider, payload = load_evaluation_components(checkpoint, device)
        identity = _validate_source_provider(provider)
        if model.get_config().get("architecture") != "A2-D":
            raise ValueError("Wind source-read native evaluation is bound to architecture A2-D.")
        rows = choose_case_keys(provider, split_name, case_keys)
        if split == "train-fixture" and len(rows) > MAX_TRAIN_FIXTURE_ROWS:
            raise ValueError("A TRAIN fixture cannot exceed the bounded eight-row scope.")
        for policy in policy_values:
            _validate_policy_model(model, policy)
        if first_provider is None:
            first_provider = provider
            first_rows = rows
        else:
            assert first_provider is not None and first_rows is not None
            first_identity = first_provider.identity_payload()
            if (
                rows != first_rows
                or identity.get("subset_manifest_sha256") != first_identity.get("subset_manifest_sha256")
                or identity.get("train_row_indices_sha256") != first_identity.get("train_row_indices_sha256")
                or identity.get("validation_row_indices_sha256") != first_identity.get("validation_row_indices_sha256")
                or identity.get("protocol") != first_identity.get("protocol")
            ):
                raise ValueError("Compared source-read checkpoints must share one protocol, cohort, and row order.")
        loaded_models[label] = (model, provider, payload, checkpoint)
        providers[label] = provider
        effective_core_config = model.get_config()
        declared_recipe_model = identity.get("recipe", {}).get("model", {})
        identities[label] = {
            **_provider_data_binding(provider),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "completed_epoch": int(payload.get("epoch", payload.get("current_epoch", -1))),
            "source_reader_variant": identity.get("source_reader_variant"),
            "source_read_model_policy": model.get_config().get("source_read_policy", "full"),
            "declared_recipe_model": dict(declared_recipe_model),
            "effective_core_config": dict(effective_core_config),
            "core_query_mode": (
                "whole_query"
                if effective_core_config.get("reader_query_chunk_size") is None
                else "internal_reader_chunking"
            ),
            "transform_binding": _provider_data_binding(provider)["transform_binding"],
        }

    assert first_provider is not None and first_rows is not None
    if split == "train-fixture" and case_keys is None:
        selection_rule = "minimum and maximum active module count M in sealed TRAIN rows; stable native-row tie breaks"
    elif split == "train-fixture":
        selection_rule = "explicit rows, validated against exact sealed TRAIN72/TRAIN420 membership"
    elif case_keys is None:
        selection_rule = "all exposed validation rows in sealed canonical order"
    else:
        selection_rule = "explicit rows, validated against exact sealed VALID24/VALID90 membership"

    # Bind every model to the same physical query and target bytes before
    # decoding. Each model later builds its scene with its own checkpoint-bound
    # TRAIN transform; transforms are never copied between variants.
    panel_provider = first_provider
    cache_before = _cache_summary(panel_provider)
    panels = _prepare_native_role_panels(panel_provider, first_rows, split_name)
    cache_after_panel = _cache_summary(panel_provider)
    if cache_after_panel["persistent_writes"] != cache_before["persistent_writes"]:
        raise RuntimeError("Native role panel sampling attempted a persistent geometry cache write.")
    per_provider_cache: dict[str, dict[str, Any]] = {
        next(iter(providers)): {"before": cache_before, "after_panel": cache_after_panel}
    }
    for label, provider in providers.items():
        if provider is panel_provider:
            continue
        before = _cache_summary(provider)
        other_panels = _prepare_native_role_panels(provider, first_rows, split_name)
        after = _cache_summary(provider)
        if after["persistent_writes"] != before["persistent_writes"]:
            raise RuntimeError(f"Native role panel sampling attempted a persistent geometry cache write for {label}.")
        _assert_exact_panel_match(panels, other_panels, label=label)
        per_provider_cache[label] = {"before": before, "after_panel": after}
    panel_hashes = {row: str(panel["panel_identity_sha256"]) for row, panel in panels.items()}
    selected_plane_rows = _plane_export_rows(panel_provider, first_rows, enabled=export_native_planes)
    output.mkdir(parents=True, exist_ok=False)

    model_results: dict[str, Any] = {}
    for label, (model, provider, payload, checkpoint) in loaded_models.items():
        row_metrics_by_policy: dict[str, dict[str, Any]] = {}
        work_by_policy: dict[str, dict[str, Any]] = {}
        plane_results: dict[int, dict[str, np.ndarray]] = {}
        for row in first_rows:
            panel = panels[int(row)]
            case = provider.view.run(int(row))
            scene_inputs = provider.adapter.scene_inputs(case)
            scene = provider.make_scene((scene_inputs,))
            row_metadata = _case_metadata(provider, int(row))
            for policy in policy_values:
                _validate_policy_model(model, policy)
                with model.source_read_override_v1(
                    policy,
                    executor=executor,
                    operating_point=operating_point,
                ):
                    setter = getattr(provider, "_set_source_read_training_epoch", None)
                    if callable(setter):
                        setter(model, int(payload.get("epoch", payload.get("current_epoch", 0))))
                    organization = model.prepare(scene)
                    prediction, work = _decode_with_counts(
                        model,
                        provider,
                        scene_inputs,
                        scene,
                        organization,
                        case,
                        panel["coordinates_D"],
                        panel["native_flat_indices"],
                        panel["role_ids"],
                        tile_size=query_tile_size,
                    )
                    metric_row = physical_role_metrics(
                        row_metadata,
                        prediction,
                        panel["targets_mps"],
                        panel["role_ids"],
                    )
                    metric_row["panel_identity_sha256"] = panel["panel_identity_sha256"]
                    metric_row["target_values_sha256"] = panel["target_values_sha256"]
                    metric_row["query_sampling_sha256"] = panel["query_sampling_sha256"]
                    metric_row["role_query_counts"] = dict(panel["role_query_counts"])
                    row_metrics_by_policy.setdefault(policy, {})[str(int(row))] = metric_row
                    row_work = work_by_policy.setdefault(policy, {})
                    for name, value in work.items():
                        row_work[name] = int(row_work.get(name, 0)) + int(value)

                    if int(row) in set(selected_plane_rows.values()):
                        arrays = plane_results.setdefault(int(row), {})
                        plane_bundle = _native_plane_bundle(
                            model,
                            provider,
                            case,
                            scene_inputs,
                            scene,
                            organization,
                            tile_size=query_tile_size,
                        )
                        for name, value in plane_bundle.items():
                            arrays[f"{policy}__{name}"] = value
                        arrays.setdefault("native_flat_indices", plane_bundle["native_flat_indices"])
                        arrays.setdefault("coordinates_D", plane_bundle["coordinates_D"])
                        arrays.setdefault("target_mps", plane_bundle["target_mps"])

        policy_results = {}
        for policy in policy_values:
            rows_for_policy = [row_metrics_by_policy[policy][str(int(row))] for row in first_rows]
            policy_results[policy] = {
                "role_metrics": summarize_physical_role_metrics(rows_for_policy),
                "row_metrics": rows_for_policy,
                "work_counts": work_by_policy[policy],
                "execution_evidence": {
                    "source_work_counters_reported": True,
                    "wall_runtime_benchmark_measured": False,
                    "executor_savings_claimed": False,
                    "full_access_fallback_is_separate_policy": policy == "full",
                },
            }

        for row, arrays in plane_results.items():
            path = output / f"native_hub_{label}_row_{row:03d}.npz"
            np.savez_compressed(path, **arrays)
            metadata = _case_metadata(provider, row)
            metadata.update(
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "native_flat_ids_sha256": _array_digest(arrays["native_flat_indices"]),
                    "coordinate_and_target_sha256": _array_digest(
                        arrays["native_flat_indices"], arrays["coordinates_D"], arrays["target_mps"]
                    ),
                    "permitted_export": "stored native hub plane and three stored native-y cuts at requested 2D/4D/8D offsets",
                }
            )
            plane_results[row] = metadata

        cache_after_evaluation = _cache_summary(provider)
        if cache_after_evaluation["persistent_writes"] != per_provider_cache[label]["before"]["persistent_writes"]:
            raise RuntimeError(
                f"Native source-read evaluation attempted a persistent geometry cache write for {label}."
            )
        model_results[label] = {
            "checkpoint": identities[label]["checkpoint"],
            "checkpoint_sha256": identities[label]["checkpoint_sha256"],
            "completed_epoch": identities[label]["completed_epoch"],
            "source_reader_variant": identities[label]["source_reader_variant"],
            "source_read_model_policy": identities[label]["source_read_model_policy"],
            "model_configuration": {
                "declared_recipe_model": identities[label]["declared_recipe_model"],
                "effective_core_config_from_get_config": identities[label]["effective_core_config"],
                "effective_core_query_mode": identities[label]["core_query_mode"],
                "external_evaluator_query_tile_size": int(query_tile_size),
                "external_tile_note": (
                    "The evaluator slices its requested native panel into this many queries per inference call; "
                    "this external tile is not the checkpoint's internal reader_query_chunk_size."
                ),
            },
            "dataset_identity_sha256": identities[label]["identity_sha256"],
            "transform_binding": identities[label]["transform_binding"],
            "identity": identities[label]["identity"],
            "policies": policy_results,
            "native_plane_exports": plane_results,
            "catalogue_cache": {
                "panel_sampling": per_provider_cache[label],
                "after_evaluation": cache_after_evaluation,
            },
        }

    result = {
        "schema": "wind_source_read_native_evaluation_v1",
        "metric_schema": ROLE_METRIC_SCHEMA,
        "status": "completed",
        "source_module_hashes": _source_module_hashes(),
        "dataset_scope": str(identities[next(iter(identities))]["identity"].get("protocol")),
        "split": split_name,
        "row_selection": selection_rule,
        "row_order": list(first_rows),
        "query_panel": {
            "sampler": "WindContentHypergraphTask._make_batch five-role native sampler",
            "query_count_per_row": NATIVE_ROLE_QUERY_COUNT,
            "role_order": list(ROLE_NAMES),
            "role_query_counts": panels[first_rows[0]]["role_query_counts"],
            "identity_basis": "exact native flat cell IDs, role IDs, wind-aligned coordinates, and targets; recomputed and verified for every model provider before decoding",
            "row_panel_identity_sha256": {str(row): digest for row, digest in panel_hashes.items()},
            "row_target_values_sha256": {str(row): panel["target_values_sha256"] for row, panel in panels.items()},
            "row_native_endpoint_sha256": {str(row): panel["native_endpoint_sha256"] for row, panel in panels.items()},
            "row_query_sampling_sha256": {str(row): panel["query_sampling_sha256"] for row, panel in panels.items()},
            "row_sampling_keys": {str(row): panel["sampling_key"] for row, panel in panels.items()},
            "round2_panel_hash_compatibility": "row_native_endpoint_sha256 uses the native flat-ID/role/coordinate/target digest format and can be compared to the saved Round2 per-row query_target_hashes; prediction reuse requires an exact hash match.",
            "selection_seed": int(first_provider.sampling_seed),
            "sampling_dataset_id": str(first_provider.sampling_dataset_id),
            "development_or_train_rows_only": True,
            "wind_test_target_values_read": False,
        },
        "plane_selection": {
            **selected_plane_rows,
            "native_hub_plane_exports_enabled": bool(export_native_planes),
            "basis": "input-only minimum/maximum active M within the selected exposed rows; row 69/426 tie preferences retained",
            "full_query_arrays_saved": False,
        },
        "models": model_results,
        "retained_classic_references": _evaluate_retained_references(
            reference_checkpoint_paths or {},
            panel_provider,
            panels,
            first_rows,
            query_tile_size=query_tile_size,
        ),
        "execution": {
            "device": "cpu",
            "executor": executor,
            "operating_point": operating_point,
            "query_tile_size": int(query_tile_size),
            "query_tile_semantics": (
                "external inference batching; effective core query mode is recorded per model and never inferred from this value"
            ),
            "optimizer_updates": 0,
            "solver_attempts": 0,
            "wall_runtime_benchmark_measured": False,
            "persistent_geometry_cache_writes": 0,
            "full_native_role_query_predictions_saved": False,
            "wind_test_target_values_read": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Self-contained source-read checkpoint; repeat to compare models on identical physical queries.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional retained classic Wind checkpoint evaluated on the exact same native query and target panel.",
    )
    parser.add_argument(
        "--reference-only-provider",
        type=Path,
        help="Bind a retained-reference-only run to one existing source-read provider checkpoint without reevaluating it.",
    )
    parser.add_argument(
        "--expected-panel-result",
        type=Path,
        help="Completed source-read result whose per-row native endpoint hashes must match before reference decoding.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        choices=POLICIES,
        help="Read-only source policy; defaults to full for an unambiguous dense anchor.",
    )
    parser.add_argument("--split", choices=("validation", "train-fixture"), default="validation")
    parser.add_argument("--case-keys", help="Optional comma-separated IDs from the checkpoint's sealed split.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--executor", choices=("packed", "packet", "dense_masked"), default="packed")
    parser.add_argument("--operating-point", type=int, choices=(0, 1))
    parser.add_argument("--query-tile-size", type=int, default=8192)
    parser.add_argument(
        "--no-plane-exports",
        action="store_true",
        help="Evaluate native role metrics without saving representative hub planes/cuts.",
    )
    args = parser.parse_args(argv)
    try:
        reference_checkpoints = parse_named_paths(args.reference_checkpoint, "--reference-checkpoint")
        if args.reference_only_provider is not None:
            if args.checkpoint:
                parser.error("--reference-only-provider cannot be combined with --checkpoint.")
            if args.expected_panel_result is None:
                parser.error("--reference-only-provider requires --expected-panel-result.")
            result = run_reference_comparison(
                args.reference_only_provider,
                reference_checkpoints,
                args.expected_panel_result,
                args.output,
                split=args.split,
                case_keys=parse_case_keys(args.case_keys),
                query_tile_size=args.query_tile_size,
            )
            print(json.dumps({"status": result["status"], "output": str(args.output),
                              "references": list(result["retained_classic_references"]),
                              "rows": result["row_order"]}))
            return 0
        if args.expected_panel_result is not None:
            parser.error("--expected-panel-result requires --reference-only-provider.")
        checkpoint_paths = parse_named_paths(args.checkpoint, "--checkpoint")
        selected_rows = parse_case_keys(args.case_keys)
        result = run_evaluation(
            checkpoint_paths,
            args.output,
            policies=args.policy,
            split=args.split,
            case_keys=selected_rows,
            reference_checkpoint_paths=reference_checkpoints,
            device=args.device,
            executor=args.executor,
            operating_point=args.operating_point,
            query_tile_size=args.query_tile_size,
            export_native_planes=not args.no_plane_exports,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.expanduser().resolve()),
                "models": list(result["models"]),
                "policies": list(result["models"][next(iter(result["models"]))]["policies"]),
                "rows": result["row_order"],
                "query_count_per_row": NATIVE_ROLE_QUERY_COUNT,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
