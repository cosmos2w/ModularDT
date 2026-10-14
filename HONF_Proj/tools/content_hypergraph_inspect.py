#!/usr/bin/env python3
"""Inspect same-weight A-H6 organization and receiver readout on a bound validation panel.

The command is CPU-only and read-only with respect to source data and model
weights. Generated arrays and the result manifest must live beneath the
authorized round root supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for directory in (
    PROJECT / "src",
    PROJECT / "tools",
    PROJECT / "Case_ThermalChannel/src",
    PROJECT / "Case_WindFarm/src",
):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from content_hypergraph_evaluate import canonical_sha256, json_safe, load_evaluation_components

from honf_forward_core.content_hypergraph import QueryBatch

CONDITIONS = (
    "original_full_h6",
    "uniform_receiver_access",
    "largest_contributing_edge_removed",
    "local_branch_removed",
)
CONDITIONS_V2 = (
    "original_full_h6", "uniform_receiver_access", "input_declared_edge_removed", "local_branch_removed",
)
THERMAL_REPRESENTATIVE_IDS = ("0291", "0687")
WIND_ROLE_INDEX = {
    "volume": 0,
    "hub_slab": 1,
    "downstream_envelope": 2,
    "near_turbine": 3,
    "background": 4,
}


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _put_array(arrays: dict[str, np.ndarray], name: str, value: Any) -> None:
    array = _numpy(value)
    if array.dtype.kind == "O":
        raise TypeError(f"Inspection array {name!r} cannot use object dtype.")
    arrays[name] = np.ascontiguousarray(array)


def _save_scene_and_organization(arrays: dict[str, np.ndarray], scene: Any, organization: Any) -> None:
    for name in (
        "module_features",
        "environment_features",
        "module_coordinates",
        "environment_coordinates",
        "module_mask",
        "environment_mask",
        "module_measure",
        "environment_measure",
        "module_ids",
        "environment_ids",
        "local_radius",
    ):
        _put_array(arrays, f"scene__{name}", getattr(scene, name))
    if hasattr(organization, "module_gather_weights"):
        for name in ("module_characteristic_lengths", "environment_characteristic_lengths",
                     "coordinate_origin", "coordinate_scale"):
            _put_array(arrays, f"scene__{name}", getattr(scene, name))
        for name in ("edges", "raw_module_states", "raw_environment_states", "module_states",
                     "environment_states", "scene_summary", "first_content", "mixed_content", "second_content",
                     "module_gather_weights", "environment_gather_weights", "edge_anchors", "edge_scales"):
            _put_array(arrays, f"organization__{name}", getattr(organization, name))
        for name, value in (organization.trace or {}).items():
            if torch.is_tensor(value):
                _put_array(arrays, f"organization_trace__{name}", value)
        if organization.source_coefficients is not None:
            _put_array(arrays, "organization__source_coefficients_C", organization.source_coefficients)
        return
    for name in (
        "edges",
        "module_incidence",
        "environment_incidence",
        "module_pooling_weights",
        "environment_pooling_weights",
        "module_edge_mass",
        "environment_edge_mass",
        "edge_anchors",
        "edge_scales",
    ):
        _put_array(arrays, f"organization__{name}", getattr(organization, name))
    if organization.source_coefficients is not None:
        _put_array(arrays, "organization__source_coefficients_C", organization.source_coefficients)


def _stable_argmin(values: np.ndarray, eligible: np.ndarray, tie_ids: np.ndarray) -> int:
    candidates = np.flatnonzero(eligible)
    if not candidates.size:
        raise ValueError("Input-only probe selection found no eligible native receiver.")
    order = np.lexsort((tie_ids[candidates], values[candidates]))
    return int(candidates[order[0]])


def _stable_argmax(values: np.ndarray, eligible: np.ndarray, tie_ids: np.ndarray) -> int:
    candidates = np.flatnonzero(eligible)
    if not candidates.size:
        raise ValueError("Input-only probe selection found no eligible native receiver.")
    order = np.lexsort((tie_ids[candidates], -values[candidates]))
    return int(candidates[order[0]])


def select_thermal_probes(
    coordinates: np.ndarray,
    native_ids: np.ndarray,
    fluid_mask: np.ndarray,
    module_centers: np.ndarray,
    module_radii: np.ndarray,
    local_radii: np.ndarray,
) -> dict[str, int]:
    """Choose near-body, support-overlap, and background points from geometry only."""
    points = np.asarray(coordinates, dtype=np.float64)
    ids = np.asarray(native_ids, dtype=np.int64)
    valid = np.asarray(fluid_mask, dtype=bool)
    centers = np.asarray(module_centers, dtype=np.float64)
    radii = np.asarray(module_radii, dtype=np.float64)
    support = np.asarray(local_radii, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or ids.shape != (points.shape[0],)
        or valid.shape != ids.shape
        or centers.ndim != 2
        or centers.shape[1] != 2
        or radii.shape != (centers.shape[0],)
        or support.shape != radii.shape
    ):
        raise ValueError("Thermal geometry probe arrays have incompatible native axes.")
    active = (radii > 0.0) & (support > 0.0)
    if not np.any(active) or not np.any(valid):
        raise ValueError("Thermal probe selection requires active modules and fluid receivers.")
    distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    surface_distance = np.abs(distance - radii[None, :])
    surface_distance[:, ~active] = np.inf
    near_score = np.min(surface_distance, axis=1)
    near = _stable_argmin(near_score, valid, ids)

    downstream = points[:, None, 0] > centers[None, :, 0]
    supported_downstream = (distance < support[None, :]) & downstream & active[None, :]
    overlap_count = supported_downstream.sum(axis=1)
    remaining = valid.copy()
    remaining[near] = False
    overlap_candidates = remaining & (overlap_count > 0)
    downstream_overlap = _stable_argmax(overlap_count, overlap_candidates, ids)

    remaining[downstream_overlap] = False
    outside_support = valid & (distance[:, active] >= support[None, active]).all(axis=1)
    outside_support &= remaining
    nearest_source_distance = np.min(distance[:, active], axis=1)
    background = _stable_argmax(nearest_source_distance, outside_support, ids)
    return {"near_body": near, "downstream_overlap": downstream_overlap, "background": background}


def select_wind_probes(
    coordinates_D: np.ndarray,
    native_ids: np.ndarray,
    role_ids: np.ndarray,
    module_centers_D: np.ndarray,
    module_radii_D: np.ndarray,
) -> dict[str, int]:
    """Choose three native Wind receivers from prescribed roles and geometry."""
    points = np.asarray(coordinates_D, dtype=np.float64)
    ids = np.asarray(native_ids, dtype=np.int64)
    roles = np.asarray(role_ids, dtype=np.int64)
    centers = np.asarray(module_centers_D, dtype=np.float64)
    radii = np.asarray(module_radii_D, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or ids.shape != (points.shape[0],)
        or roles.shape != ids.shape
        or centers.ndim != 2
        or centers.shape[1] != 3
        or radii.shape != (centers.shape[0],)
    ):
        raise ValueError("Wind geometry probe arrays have incompatible native axes.")
    active = radii > 0.0
    if not np.any(active):
        raise ValueError("Wind probe selection requires at least one active turbine.")
    distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    surface_distance = np.abs(distance - radii[None, :])
    surface_distance[:, ~active] = np.inf
    near = _stable_argmin(
        np.min(surface_distance, axis=1),
        roles == WIND_ROLE_INDEX["near_turbine"],
        ids,
    )

    relative = points[:, None, :] - centers[None, :, :]
    downstream_geometry = (
        (relative[..., 0] > 0.0)
        & (relative[..., 0] <= 10.0)
        & (np.square(relative[..., 1]) + np.square(relative[..., 2]) <= 1.5**2)
        & active[None, :]
    )
    downstream_count = downstream_geometry.sum(axis=1)
    downstream_role = roles == WIND_ROLE_INDEX["downstream_envelope"]
    downstream_candidates = downstream_role & (downstream_count > 0)
    if not np.any(downstream_candidates):
        downstream_candidates = downstream_role
    downstream = _stable_argmax(downstream_count, downstream_candidates, ids)

    background_role = roles == WIND_ROLE_INDEX["background"]
    nearest_distance = np.min(distance[:, active], axis=1)
    background = _stable_argmax(nearest_distance, background_role, ids)
    return {"near_body": near, "downstream_overlap": downstream, "background": background}


def rank_thermal_edge(signed_factor_contributions: Any) -> tuple[int, np.ndarray]:
    """Rank H6 edges by exact signed native-temperature factor contributions."""
    contribution = _numpy(signed_factor_contributions).astype(np.float64, copy=False)
    if contribution.ndim != 3 or contribution.shape[1] < 1:
        raise ValueError("Thermal edge contributions must have shape [probe,H,rank].")
    signed_edge = contribution.sum(axis=-1)
    scores = np.sqrt(np.mean(np.square(signed_edge), axis=0))
    return int(np.argmax(scores)), scores


def rank_wind_edge(access: Any, messages: Any) -> tuple[int, np.ndarray]:
    """Rank H6 edges by measured receiver-weighted latent messages."""
    alpha = _numpy(access).astype(np.float64, copy=False)
    message = _numpy(messages).astype(np.float64, copy=False)
    if alpha.ndim != 2 or message.ndim != 3 or message.shape[:2] != alpha.shape:
        raise ValueError("Wind edge use requires access [probe,H] and messages [probe,H,width].")
    weighted = alpha[..., None] * message
    scores = np.sqrt(np.mean(np.square(weighted).sum(axis=-1), axis=0))
    return int(np.argmax(scores)), scores


def removed_edge_gate(batch_size: int, edge_count: int, edge_index: int, *, device, dtype) -> torch.Tensor:
    if batch_size < 1 or edge_count < 1 or not 0 <= edge_index < edge_count:
        raise ValueError("An edge intervention needs a valid batch and edge index.")
    gates = torch.ones((batch_size, edge_count), device=device, dtype=dtype)
    gates[:, edge_index] = 0.0
    return gates


def _other_edge_access_equal(removed: torch.Tensor, reference: torch.Tensor, edge_index: int) -> bool:
    return all(
        torch.equal(removed[..., index], reference[..., index])
        for index in range(reference.shape[-1])
        if index != edge_index
    )


@contextmanager
def local_branch_zero_hook(core):
    """Zero local outputs after the reader while preserving its executed rows."""

    def zero_local_outputs(_module, _inputs, outputs):
        if not isinstance(outputs, tuple) or len(outputs) != 3:
            raise TypeError("Local detail reader output no longer matches the inspection hook contract.")
        context, pair_pack, temperature_coefficients = outputs
        return torch.zeros_like(context), pair_pack, torch.zeros_like(temperature_coefficients)

    handle = core.local_reader.register_forward_hook(zero_local_outputs)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_local_flow_output(core):
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        captured["flow_values"] = output.detach().clone()

    handle = core.local_reader.flow_output.register_forward_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


def slice_query(query: QueryBatch, indices: list[int] | np.ndarray | torch.Tensor) -> QueryBatch:
    """Build an exact query subset and remap its packed local pair indices."""
    selected = torch.as_tensor(indices, device=query.coordinates.device, dtype=torch.long)
    if selected.ndim != 1 or not selected.numel():
        raise ValueError("A query trace subset must contain one or more query indices.")
    old_to_new = torch.full(
        (query.num_queries,), -1, device=query.coordinates.device, dtype=torch.long
    )
    old_to_new[selected] = torch.arange(selected.numel(), device=selected.device)
    values: dict[str, Any] = {
        "coordinates": query.coordinates.index_select(1, selected),
        "features": None if query.features is None else query.features.index_select(1, selected),
        "query_mask": None if query.query_mask is None else query.query_mask.index_select(1, selected),
        "query_ids": None if query.query_ids is None else query.query_ids.index_select(1, selected),
        "receiver_key": (query.receiver_key, "inspection_subset", tuple(int(i) for i in selected.cpu())),
    }
    if query.local_batch_indices is not None:
        mapped = old_to_new[query.local_query_indices]
        keep = mapped >= 0
        values.update(
            local_batch_indices=query.local_batch_indices[keep],
            local_query_indices=mapped[keep],
            local_module_indices=query.local_module_indices[keep],
            local_weights=query.local_weights[keep],
        )
    return replace(query, **values)


def _save_local_pair_arrays(
    arrays: dict[str, np.ndarray],
    output_trace: dict[str, Any],
    flow_values: torch.Tensor | None,
    scene: Any,
    query: QueryBatch,
    *,
    controls: torch.Tensor | None,
) -> None:
    pair_query = output_trace["local_pair_query_indices"]
    pair_batch = output_trace["local_pair_batch_indices"]
    pair_module = output_trace["local_pair_module_indices"]
    support_weight = output_trace["local_pair_support_weights"]
    for name, value in (
        ("batch_indices", pair_batch),
        ("query_indices", pair_query),
        ("module_indices", pair_module),
        ("native_query_ids", query.query_ids[pair_batch, pair_query] if query.query_ids is not None else pair_query),
        ("support_weights", support_weight),
    ):
        _put_array(arrays, f"local_pairs__{name}", value)
    if flow_values is not None:
        if flow_values.shape[0] != pair_query.shape[0]:
            raise ValueError("Local pre-head flow values lost their exact packed pair ordering.")
        if scene.module_measure_reference is None:
            reference = torch.ones(scene.batch_size, device=flow_values.device, dtype=flow_values.dtype)
        else:
            reference = scene.module_measure_reference.to(device=flow_values.device, dtype=flow_values.dtype)
        measure_ratio = scene.module_measure[pair_batch, pair_module].to(flow_values.dtype) / reference[
            pair_batch
        ].clamp_min(torch.finfo(flow_values.dtype).tiny)
        flow_factor = support_weight.to(flow_values.dtype) * measure_ratio
        _put_array(arrays, "local_pairs__flow_output_pre_head", flow_values)
        _put_array(arrays, "local_pairs__flow_measure_ratio", measure_ratio)
        _put_array(arrays, "local_pairs__flow_context_contribution", flow_values * flow_factor[:, None])
    coefficients = output_trace.get("local_temperature_coefficients")
    if coefficients is not None:
        _put_array(arrays, "local_pairs__temperature_coefficients", coefficients)
        if controls is not None:
            contribution = support_weight * coefficients * controls[pair_batch, pair_module]
            _put_array(arrays, "local_pairs__temperature_contribution_normalized", contribution)


def _condition_trace_arrays(
    arrays: dict[str, np.ndarray],
    condition: str,
    output: Any,
    probe_indices: np.ndarray,
) -> None:
    if output.trace is None:
        raise RuntimeError("A-H6 inspection decode did not return its requested trace.")
    for name in (
        "edge_access",
        "edge_base_access",
        "edge_used_access",
        "edge_gates",
        "edge_messages",
        "edge_temperature_contributions",
    ):
        value = output.trace.get(name)
        if value is not None:
            selected = value if name == "edge_gates" else value[:, probe_indices]
            _put_array(arrays, f"{condition}__{name}", selected)


def _thermal_case_batch(provider: Any, selected: tuple[str, ...]):
    ids = set(provider.validation_case_ids)
    if any(case_id not in ids for case_id in selected):
        raise ValueError("Thermal inspection case panel is outside the bound validation membership.")
    return list(provider.native_evaluation_batches(split="validation", case_keys=selected, microbatch_size=1))


def _validate_thermal_development_binding(provider: Any) -> dict[str, Any]:
    from channelthermal.data.development_split import development_case_ids
    from channelthermal.training.unified_task import FIXED25_FINGERPRINT

    identity = provider.identity_payload()
    validation_ids = tuple(map(str, identity.get("validation_case_ids", ())))
    source_binding = identity.get("source_binding", {})
    manifest_validation_ids = tuple(map(str, development_case_ids(provider.manifest, "test")))
    if (
        identity.get("protocol") != "development"
        or identity.get("validation_scope") != "fixed25_v1_DEV22_exposed"
        or identity.get("dataset_profile") != "fixed25_v1"
        or len(validation_ids) != 22
        or validation_ids != tuple(map(str, provider.validation_case_ids))
        or validation_ids != manifest_validation_ids
        or identity.get("subset_manifest_sha256") != provider.manifest.get("manifest_sha256")
        or provider.manifest.get("manifest_sha256") != FIXED25_FINGERPRINT
        or identity.get("base_fixed25_manifest_sha256") != provider.base_manifest_sha256
        or provider.base_manifest_sha256 != FIXED25_FINGERPRINT
        or source_binding.get("base_manifest_sha256") != provider.base_manifest_sha256
        or source_binding.get("protocol") != "development"
        or source_binding.get("original_test_targets_outside_declared_panel_read") is not False
    ):
        raise ValueError("Thermal inspection provider is not bound to the exact exposed fixed25_v1 DEV22 panel.")
    return {
        "validation_case_ids": list(validation_ids),
        "validation_case_ids_sha256": identity["validation_case_ids_sha256"],
        "validation_scope": identity["validation_scope"],
        "subset_manifest_sha256": identity["subset_manifest_sha256"],
        "base_fixed25_manifest_sha256": identity["base_fixed25_manifest_sha256"],
        "source_binding": {
            key: source_binding[key]
            for key in (
                "dataset_id",
                "catalog_metadata_sha256",
                "base_manifest_sha256",
                "protocol",
                "normalization_fit_scope",
                "original_test_targets_outside_declared_panel_read",
            )
            if key in source_binding
        },
        "only_declared_exposed_validation_targets_read": True,
        "unexposed_original_test_targets_read": False,
    }


def _validate_thermal_full_binding(provider: Any) -> dict[str, Any]:
    from channelthermal.training.content_hypergraph_task import (
        THERMAL_CONTENT_FULL_TRAIN_DATASET,
        THERMAL_CONTENT_FULL_TRAIN_PROTOCOL,
    )
    from channelthermal.training.joint_task import validate_full_train_followup_membership
    from channelthermal.training.unified_task import FIXED25_FINGERPRINT

    identity = provider.identity_payload()
    source_binding = identity.get("source_binding", {})
    training_ids = tuple(map(str, identity.get("training_case_ids", ())))
    validation_ids = tuple(map(str, identity.get("validation_case_ids", ())))
    if training_ids != tuple(map(str, provider.training_case_ids)):
        raise ValueError("Thermal full-follow-up identity TRAIN IDs differ from the provider membership.")
    if validation_ids != tuple(map(str, provider.validation_case_ids)):
        raise ValueError("Thermal full-follow-up identity VALID IDs differ from the provider membership.")

    membership = validate_full_train_followup_membership(provider.training_case_ids, provider.validation_case_ids)
    manifest_followup = provider.manifest.get("full_train_followup1000", {})
    if (
        provider.protocol != "full_train1000"
        or identity.get("protocol") != "full_train1000"
        or identity.get("dataset_profile") != THERMAL_CONTENT_FULL_TRAIN_DATASET
        or identity.get("validation_scope") != "canonical89_exposed_original_test_excluding_0273"
        or identity.get("subset_manifest_sha256") != provider.manifest.get("manifest_sha256")
        or identity.get("training_case_ids_sha256") != membership["training_membership_sha256"]
        or identity.get("validation_case_ids_sha256") != membership["validation_membership_sha256"]
        or identity.get("base_fixed25_manifest_sha256") != FIXED25_FINGERPRINT
        or provider.base_manifest_sha256 != FIXED25_FINGERPRINT
        or source_binding.get("base_manifest_sha256") != FIXED25_FINGERPRINT
        or source_binding.get("protocol") != "full_train1000"
        or source_binding.get("normalization_fit_scope") != "exact original TRAIN600 only"
        or source_binding.get("original_test_targets_outside_declared_panel_read") is not False
        or source_binding.get("full_followup_membership") != membership
        or manifest_followup.get("execution_protocol") != THERMAL_CONTENT_FULL_TRAIN_PROTOCOL
        or manifest_followup.get("dataset_protocol") != THERMAL_CONTENT_FULL_TRAIN_DATASET
        or manifest_followup.get("base_manifest_sha256") != FIXED25_FINGERPRINT
        or manifest_followup.get("training_case_ids") != list(training_ids)
        or manifest_followup.get("validation_case_ids") != list(validation_ids)
        or manifest_followup.get("training_membership_sha256") != membership["training_membership_sha256"]
        or manifest_followup.get("validation_membership_sha256") != membership["validation_membership_sha256"]
        or manifest_followup.get("excluded_validation_id") != "0273"
    ):
        raise ValueError("Thermal full-follow-up provider is not bound to the exact TRAIN600/canonical89 contract.")
    return {
        "protocol": "full_train1000",
        "validation_scope": identity["validation_scope"],
        "validation_case_ids": list(validation_ids),
        "validation_case_ids_sha256": membership["validation_membership_sha256"],
        "training_membership_sha256": membership["training_membership_sha256"],
        "subset_manifest_sha256": identity["subset_manifest_sha256"],
        "base_fixed25_manifest_sha256": FIXED25_FINGERPRINT,
        "normalization_fit_scope": source_binding["normalization_fit_scope"],
        "full_followup_membership": membership,
        "unexposed_original_test_targets_read": membership["unexposed_original_test_targets_read"],
    }


def _select_thermal_cases(provider: Any, protocol: str | None = None) -> tuple[str, ...]:
    scope = protocol or provider.protocol
    if scope == "development":
        _validate_thermal_development_binding(provider)
    elif scope == "full_train1000":
        _validate_thermal_full_binding(provider)
    else:
        raise ValueError("Thermal inspection supports only development or full_train1000 providers.")
    validation = set(provider.validation_case_ids)
    if all(case_id in validation for case_id in THERMAL_REPRESENTATIVE_IDS):
        return THERMAL_REPRESENTATIVE_IDS
    if scope == "development":
        raise ValueError("Fixed Thermal inspection cases 0291 and 0687 are not both in the bound DEV22 panel.")
    selected = tuple(sorted(validation)[:2])
    if len(selected) != 2:
        raise ValueError("Thermal full-follow-up validation panel has fewer than two bound cases.")
    return selected


def _validate_wind_full_binding(provider: Any) -> dict[str, Any]:
    from wind_native_reference_evaluate import _assert_full_train1000_provider

    identity = _assert_full_train1000_provider(provider)
    validation_rows = np.asarray(provider.validation_rows, dtype=np.int64)
    if (
        provider.recipe.get("protocol") != "full_train1000"
        or identity.get("protocol") != "full_train1000"
        or identity.get("validation_rows") != validation_rows.astype(int).tolist()
        or identity.get("unexposed_test_target_values_read") is not False
        or identity.get("wind_test_target_values_read") is not False
    ):
        raise ValueError("Wind inspection provider changed after its strict full VALID90 metadata check.")
    return {
        "protocol": "full_train1000",
        "validation_scope": "seed42_full_train1000_VALID90_exposed",
        "validation_rows": validation_rows.astype(int).tolist(),
        "validation_row_indices_sha256": identity["validation_row_indices_sha256"],
        "subset_manifest_sha256": identity["subset_manifest_sha256"],
        "dataset_profile": identity["dataset_profile"],
        "test_target_values_read": False,
        "unexposed_test_target_values_read": False,
    }


def _validate_wind_development_binding(provider: Any) -> dict[str, Any]:
    from wind_native_reference_evaluate import _assert_dev24_provider

    if provider.recipe.get("protocol") != "development":
        raise ValueError("Wind development inspection requires its fixed DEV24 provider.")
    _assert_dev24_provider(provider)
    identity = provider.identity_payload()
    rows = np.asarray(provider.validation_rows, dtype=np.int64)
    if identity.get("protocol") != "development" or identity.get("validation_rows") != rows.astype(int).tolist():
        raise ValueError("Wind DEV24 identity differs from the provider validation rows.")
    return {
        "protocol": "development",
        "validation_scope": "wind_shared_fixed24_v1_DEV24_exposed",
        "validation_rows": rows.astype(int).tolist(),
        "validation_row_indices_sha256": identity["validation_row_indices_sha256"],
        "subset_manifest_sha256": identity["subset_manifest_sha256"],
        "dataset_profile": identity["dataset_profile"],
        "test_target_values_read": False,
        "unexposed_test_target_values_read": False,
    }


def _fp64_identity_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    actual64 = actual.to(torch.float64)
    expected64 = expected.to(device=actual.device, dtype=torch.float64)
    if actual64.shape != expected64.shape:
        raise ValueError("Thermal native edge identity tensors have different shapes.")
    error = (actual64 - expected64).abs()
    tolerance = 1.0e-9 + 1.0e-7 * expected64.abs()
    maximum_error = float(error.max().item()) if error.numel() else 0.0
    maximum_ratio = (
        float((error / tolerance.clamp_min(torch.finfo(torch.float64).tiny)).max().item())
        if error.numel()
        else 0.0
    )
    return {
        "max_abs_error": maximum_error,
        "max_abs_software_tolerance": float(tolerance.max().item()) if tolerance.numel() else 0.0,
        "max_error_to_tolerance_ratio": maximum_ratio,
        "passed": bool(torch.all(error <= tolerance)),
        "comparison_dtype": "float64",
        "software_tolerance": {"atol": 1.0e-9, "rtol": 1.0e-7},
    }


def _select_wind_rows(provider: Any, protocol: str | None = None) -> tuple[int, ...]:
    scope = protocol or provider.recipe["protocol"]
    if scope == "full_train1000":
        _validate_wind_full_binding(provider)
    elif scope == "development":
        _validate_wind_development_binding(provider)
    else:
        raise ValueError("Wind inspection supports only development or full_train1000 providers.")
    rows = np.asarray(provider.validation_rows, dtype=np.int64)
    counts = np.asarray(provider.view.metadata["n_turbines"], dtype=np.int64)
    low_count = int(counts[rows].min())
    high_count = int(counts[rows].max())
    if low_count == high_count:
        raise ValueError("Wind fixed DEV rows do not span low/high turbine counts.")
    preferred = (69, 426)
    if (
        all(row in set(rows.tolist()) for row in preferred)
        and counts[preferred[0]] == low_count
        and counts[preferred[1]] == high_count
    ):
        return preferred
    low = int(rows[counts[rows] == low_count][0])
    high = int(rows[counts[rows] == high_count][-1])
    return low, high


def _validate_checkpoint_scope(dataset: str, recipe: dict[str, Any], payload: dict[str, Any]) -> str:
    protocol = recipe.get("protocol")
    if protocol == "development":
        return "development"
    if protocol != "full_train1000":
        raise ValueError("Same-weight inspection supports only development or full_train1000 checkpoints.")
    epoch = payload.get("epoch")
    stop = recipe.get("approved_stop_after")
    if (
        type(epoch) is not int
        or epoch != 1000
        or type(stop) is not int
        or stop != 1000
        or str(recipe.get("architecture", "")).upper() not in {"A-H6", "A2-H6"}
        or dataset not in ("thermal", "wind")
    ):
        raise ValueError("Full-follow-up inspection requires an A-H6 checkpoint at literal approved epoch 1000.")
    return "full_train1000"


def _validation_panel_metadata(
    dataset: str,
    protocol: str,
    binding: dict[str, Any],
    selected: tuple[str, ...] | tuple[int, ...],
) -> dict[str, Any]:
    if dataset == "thermal":
        return {
            "scope": binding["validation_scope"],
            "source_partition": "original_h5_test_split_exposed_validation_panel",
            "bound_validation_count": len(binding["validation_case_ids"]),
            "bound_validation_membership_sha256": binding["validation_case_ids_sha256"],
            "inspected_case_ids": list(selected),
            "exposed_validation_target_values_read_for_metrics": True,
            "unexposed_original_test_targets_read": False,
        }
    return {
        "scope": binding["validation_scope"],
        "source_partition": (
            "original_full_train1000_VALID90_exposed"
            if protocol == "full_train1000" else "wind_shared_fixed24_v1_DEV24_exposed"
        ),
        "bound_validation_count": len(binding["validation_rows"]),
        "bound_validation_membership_sha256": binding["validation_row_indices_sha256"],
        "inspected_native_rows": list(selected),
        "exposed_validation_target_values_read_for_metrics": True,
        "wind_test_target_values_read": False,
        "unexposed_test_target_values_read": False,
    }


def _thermal_condition(
    *,
    model: Any,
    provider: Any,
    batch: Any,
    prepared: Any,
    condition: str,
    edge_index: int,
    probe_observation_indices: np.ndarray,
    probe_query_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    core = model.core
    gates = None
    uniform = condition == "uniform_receiver_access"
    zero_local = condition == "local_branch_removed"
    if condition == "largest_contributing_edge_removed":
        gates = removed_edge_gate(
            prepared.query_batch.batch_size,
            prepared.organization.num_hyperedges,
            edge_index,
            device=prepared.query_batch.coordinates.device,
            dtype=prepared.query_batch.coordinates.dtype,
        )
    started = time.perf_counter()
    capture = capture_local_flow_output(core) if condition == "original_full_h6" else None
    with (capture if capture is not None else _no_capture()) as captured:
        if zero_local:
            with local_branch_zero_hook(core):
                output = core.decode(
                    prepared.organization,
                    prepared.query_batch,
                    controls=batch.receivers.heat,
                    return_trace=True,
                    accumulation_dtype=torch.float64,
                    uniform_access=uniform,
                    edge_gates=gates,
                )
        else:
            output = core.decode(
                prepared.organization,
                prepared.query_batch,
                controls=batch.receivers.heat,
                return_trace=True,
                accumulation_dtype=torch.float64,
                uniform_access=uniform,
                edge_gates=gates,
            )
    elapsed = time.perf_counter() - started
    if output.thermal_temperature is None:
        raise RuntimeError("Thermal A-H6 omitted its affine temperature output.")
    native = model._extract_native_roles(
        prepared,
        output.thermal_temperature,
        output.nonlinear_field,
        accumulation_dtype=torch.float64,
    )
    from channelthermal.training.content_hypergraph_task import ThermalContentPredictions

    native_work = {
        "native_fluid_queries": int(batch.receivers.fluid_xy.shape[0] * batch.receivers.fluid_xy.shape[1]),
        "native_union_rows": int(prepared.union.unique_counts.sum().item()),
        "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        **dict(output.work_counts),
    }
    predictions = ThermalContentPredictions(prepared, native, work=native_work)
    metrics = provider.validation_metrics(predictions, batch.targets, {})
    _condition_trace_arrays(arrays, condition, output, probe_query_indices)
    _put_array(arrays, f"{condition}__normalized_field_probe", output.nonlinear_field[:, probe_query_indices])
    _put_array(
        arrays,
        f"{condition}__normalized_temperature_probe",
        output.thermal_temperature[:, probe_query_indices],
    )
    for name, value in native.items():
        _put_array(arrays, f"{condition}__native__{name}", value)
    for name, value in (
        ("field_targets", batch.targets.field_targets),
        ("point_weights", batch.targets.point_weights),
        ("interface_targets", batch.targets.interface_target),
        ("interface_valid_mask", batch.targets.interface_valid_mask),
        ("material_targets", batch.targets.material_targets),
    ):
        if f"targets__{name}" not in arrays:
            _put_array(arrays, f"targets__{name}", value)
    if condition == "original_full_h6":
        _save_local_pair_arrays(
            arrays,
            output.trace,
            captured.get("flow_values"),
            prepared.scene_batch,
            prepared.query_batch,
            controls=batch.receivers.heat,
        )
    probe_prediction = native["pred_field"][0, probe_observation_indices]
    probe_target = batch.targets.field_targets[0, probe_observation_indices]
    result = {
        "metrics": json_safe(metrics),
        "core_work_counts": dict(output.work_counts),
        "provider_work_counts": dict(provider.work_counts(batch, predictions, {"work": output.work_counts})),
        "elapsed_seconds_cpu": elapsed,
        "probe_prediction_physical": json_safe(probe_prediction),
        "probe_target_physical": json_safe(probe_target),
        "probe_delta_from_full_physical": None,
        "probe_error_physical": json_safe(probe_prediction - probe_target),
        "local_pair_rows_preserved": None,
        "edge_access_preserved": None,
        "uniform_access_exact": None,
        "full_native_prediction_recomputed": True,
    }
    return {"result": result, "output": output, "native": native}


@contextmanager
def _no_capture():
    yield {}


def _run_thermal_case(model: Any, provider: Any, batch: Any) -> dict[str, Any]:
    case_id = str(batch.case_keys[0])
    adapter = model
    core = model.core
    prepared = adapter.prepare_native(
        batch.scene_inputs.structure,
        batch.receivers.fluid_xy,
        local_query_points=batch.receivers.local_query_points,
        ntheta=batch.receivers.ntheta,
        scene_key=batch.scene_inputs.case_ids,
    )
    prepared.organization = core.prepare(prepared.scene_batch, return_trace=True)
    adapter.validate_prepared(prepared)
    native_ids = batch.receivers.fluid_xy[0]
    fluid_mask = ~prepared.native_solid_mask[0].reshape(-1).detach().cpu().numpy().astype(bool)
    active_modules = prepared.scene_batch.module_mask[0].detach().cpu().numpy().astype(bool)
    module_centers = prepared.scene_batch.module_coordinates[0].detach().cpu().numpy()[active_modules]
    module_radii = (prepared.scene_batch.local_radius[0] / 3.0).detach().cpu().numpy()[active_modules]
    local_radii = prepared.scene_batch.local_radius[0].detach().cpu().numpy()[active_modules]
    observation_ids = np.arange(native_ids.shape[0], dtype=np.int64)
    selection = select_thermal_probes(
        native_ids.detach().cpu().numpy(),
        observation_ids,
        fluid_mask,
        module_centers,
        module_radii,
        local_radii,
    )
    id_to_query = {
        int(value): index
        for index, value in enumerate(prepared.query_batch.query_ids[0].detach().cpu().tolist())
    }
    probe_observation_indices = np.asarray([selection[name] for name in ("near_body", "downstream_overlap", "background")])
    try:
        probe_query_indices = np.asarray([id_to_query[int(index)] for index in probe_observation_indices], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("Thermal fixed native fluid receiver is absent from the prepared grid union.") from exc
    if len(set(probe_query_indices.tolist())) != 3:
        raise RuntimeError("Thermal geometry probe selection returned duplicate native query nodes.")

    arrays: dict[str, np.ndarray] = {}
    _save_scene_and_organization(arrays, prepared.scene_batch, prepared.organization)
    _put_array(arrays, "query__coordinates", prepared.query_batch.coordinates)
    _put_array(arrays, "query__native_flat_ids", prepared.query_batch.query_ids)
    _put_array(arrays, "query__mask", prepared.query_batch.query_mask)
    _put_array(arrays, "probe__observation_indices", probe_observation_indices)
    _put_array(arrays, "probe__native_flat_ids", probe_observation_indices)
    _put_array(arrays, "probe__core_query_indices", probe_query_indices)
    _put_array(arrays, "probe__coordinates", native_ids[probe_observation_indices])

    baseline = _thermal_condition(
        model=model,
        provider=provider,
        batch=batch,
        prepared=prepared,
        condition="original_full_h6",
        edge_index=0,
        probe_observation_indices=probe_observation_indices,
        probe_query_indices=probe_query_indices,
        arrays=arrays,
    )
    edge_contribution = baseline["output"].trace["edge_temperature_contributions"][0, probe_query_indices]
    edge_index, edge_scores = rank_thermal_edge(edge_contribution)
    _put_array(arrays, "probe__thermal_edge_signed_contributions", edge_contribution.sum(dim=-1))
    _put_array(arrays, "probe__thermal_total_temperature_normalized", baseline["output"].thermal_temperature[0, probe_query_indices])
    _put_array(arrays, "edge_ranking__thermal_temperature_rms", edge_scores)
    _put_array(arrays, "edge_ranking__selected_edge", np.asarray(edge_index, dtype=np.int64))

    probe_query = slice_query(prepared.query_batch, probe_query_indices)
    source_kernels = core.export_source_kernels(
        prepared.organization, probe_query, accumulation_dtype=torch.float64
    )
    temperature_scale = model.field_std[4].detach().to(device=source_kernels.device, dtype=source_kernels.dtype)
    _put_array(arrays, "probe__source_kernel_K_per_physical_heat_normalized", source_kernels)
    _put_array(arrays, "probe__source_kernel_K_physical_temperature", source_kernels * temperature_scale)

    case_results: dict[str, Any] = {"original_full_h6": baseline["result"]}
    baseline_predictions = baseline["result"]["probe_prediction_physical"]
    removed_effect_check = None
    for condition in CONDITIONS[1:]:
        current = _thermal_condition(
            model=model,
            provider=provider,
            batch=batch,
            prepared=prepared,
            condition=condition,
            edge_index=edge_index,
            probe_observation_indices=probe_observation_indices,
            probe_query_indices=probe_query_indices,
            arrays=arrays,
        )
        current_prediction = np.asarray(current["result"]["probe_prediction_physical"], dtype=np.float64)
        current["result"]["probe_delta_from_full_physical"] = (current_prediction - np.asarray(baseline_predictions)).tolist()
        if condition == "local_branch_removed":
            current["result"]["local_pair_rows_preserved"] = bool(
                current["output"].work_counts["local_rows"] == baseline["output"].work_counts["local_rows"]
                and torch.equal(
                    current["output"].trace["local_pair_query_indices"],
                    baseline["output"].trace["local_pair_query_indices"],
                )
                and torch.equal(
                    current["output"].trace["local_pair_module_indices"],
                    baseline["output"].trace["local_pair_module_indices"],
                )
            )
            current["result"]["edge_access_preserved"] = bool(torch.equal(
                current["output"].trace["edge_used_access"], baseline["output"].trace["edge_used_access"]
            ))
            current["result"]["local_reader_rows_executed_before_zero"] = int(
                current["output"].work_counts["local_rows"]
            )
        if condition == "uniform_receiver_access":
            base = current["output"].trace["edge_base_access"]
            current["result"]["uniform_access_exact"] = bool(torch.allclose(
                base,
                torch.full_like(base, 1.0 / prepared.organization.num_hyperedges),
                rtol=0.0,
                atol=1.0e-7,
            ))
        case_results[condition] = current["result"]
        if condition == "largest_contributing_edge_removed":
            removed = current["output"]
            signed_removed = edge_contribution[:, edge_index].sum(dim=-1)
            factor_delta_normalized = baseline["output"].thermal_temperature[0, probe_query_indices] - removed.thermal_temperature[0, probe_query_indices]
            edge_grid_normalized = baseline["output"].trace[
                "edge_temperature_contributions"
            ][:, :, edge_index, :].sum(dim=-1)
            edge_native = model._extract_native_roles(
                prepared,
                edge_grid_normalized,
                None,
                accumulation_dtype=torch.float64,
            )
            native_checks = {}
            for role, key, channel in (
                ("fluid_temperature", "fluid_temperature", None),
                ("interface_temperature", "pred_interface", 0),
                ("interface_q_proxy", "pred_interface", 1),
                ("internal_temperature", "pred_internal_temperature", None),
            ):
                base_value = baseline["native"][key]
                removed_value = current["native"][key]
                expected = edge_native[key]
                if channel is not None:
                    base_value = base_value[..., channel : channel + 1]
                    removed_value = removed_value[..., channel : channel + 1]
                    expected = expected[..., channel : channel + 1]
                delta_value = base_value - removed_value
                comparison = _fp64_identity_error(delta_value, expected)
                native_checks[role] = comparison
                _put_array(arrays, f"edge_identity__native_error__{role}", delta_value - expected)
            factor_delta_error = factor_delta_normalized - signed_removed
            factor_identity = _fp64_identity_error(factor_delta_normalized, signed_removed)
            _put_array(arrays, "edge_identity__probe_temperature_error_normalized", factor_delta_error)
            removed_effect_check = {
                "probe_temperature_factor_identity": factor_identity,
                "native_temperature_extraction": native_checks,
                "original_base_access_preserved": bool(torch.equal(
                    current["output"].trace["edge_base_access"], baseline["output"].trace["edge_base_access"]
                )),
                "removed_edge_access_zero": bool(
                    torch.count_nonzero(current["output"].trace["edge_used_access"][:, :, edge_index]).item() == 0
                ),
                "other_edge_access_unchanged": _other_edge_access_equal(
                    current["output"].trace["edge_used_access"],
                    baseline["output"].trace["edge_used_access"],
                    edge_index,
                ),
                "nonlinear_flow_field_recomputed_under_gate": True,
            }
    case_results["largest_contributing_edge_removed"]["edge_removal_identity"] = removed_effect_check
    probe_records = []
    for name, observation_index, query_index in zip(
        ("near_body", "downstream_overlap", "background"),
        probe_observation_indices,
        probe_query_indices,
        strict=True,
    ):
        probe_records.append({
            "name": name,
            "selection_rule": {
                "near_body": "fluid native grid point nearest a prescribed module surface",
                "downstream_overlap": "downstream fluid point covered by the greatest count of smooth local supports",
                "background": "fluid point outside all smooth local supports, farthest from an active module",
            }[name],
            "native_observation_index": int(observation_index),
            "native_flat_id": int(observation_index),
            "core_query_index": int(query_index),
            "coordinates": json_safe(native_ids[int(observation_index)]),
        })
    return {
        "case_key": case_id,
        "case_results": case_results,
        "edge_ranking": {
            "rule": "RMS across three probes of the signed sum over Thermal response modes per edge",
            "scores": edge_scores.tolist(),
            "selected_edge": int(edge_index),
        },
        "probes": probe_records,
        "work_counts_are_full_execution": True,
        "edge_removal_identity": removed_effect_check,
        "arrays": arrays,
    }


def _wind_condition(
    *,
    model: Any,
    provider: Any,
    batch: Any,
    scene: Any,
    organization: Any,
    query: QueryBatch,
    condition: str,
    edge_index: int,
    probe_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    from windfarm.training.content_hypergraph_task import WindContentPredictions

    gates = None
    uniform = condition == "uniform_receiver_access"
    zero_local = condition == "local_branch_removed"
    if condition == "largest_contributing_edge_removed":
        gates = removed_edge_gate(
            query.batch_size,
            organization.num_hyperedges,
            edge_index,
            device=query.coordinates.device,
            dtype=query.coordinates.dtype,
        )
    started = time.perf_counter()
    capture = capture_local_flow_output(model) if condition == "original_full_h6" else None
    with (capture if capture is not None else _no_capture()) as captured:
        if zero_local:
            with local_branch_zero_hook(model):
                output = model.decode(
                    organization, query, return_trace=True, uniform_access=uniform, edge_gates=gates
                )
        else:
            output = model.decode(
                organization, query, return_trace=True, uniform_access=uniform, edge_gates=gates
            )
    elapsed = time.perf_counter() - started
    physical, profile, residual = provider.adapter.physical_velocity(output.nonlinear_field, query.coordinates)
    role_ids = torch.as_tensor(batch.receivers.role_ids.copy(), device=physical.device, dtype=torch.long)
    predictions = WindContentPredictions(physical, output.nonlinear_field, residual, profile, role_ids)
    auxiliary = {"organization": organization, "core_output": output}
    metrics = provider.validation_metrics(predictions, batch.targets, auxiliary)
    provider_work = provider.work_counts(batch, predictions, auxiliary)
    _condition_trace_arrays(arrays, condition, output, probe_indices)
    if condition == "original_full_h6":
        _save_local_pair_arrays(
            arrays,
            output.trace,
            captured.get("flow_values"),
            scene,
            query,
            controls=None,
        )
    _put_array(arrays, f"{condition}__normalized_residual", output.nonlinear_field)
    _put_array(arrays, f"{condition}__physical_velocity_mps", physical)
    if "targets__velocity_mps" not in arrays:
        _put_array(arrays, "targets__velocity_mps", batch.targets.velocity_mps)
    result = {
        "metrics": json_safe(metrics),
        "core_work_counts": dict(output.work_counts),
        "provider_work_counts": dict(provider_work),
        "elapsed_seconds_cpu": elapsed,
        "probe_prediction_physical_mps": json_safe(physical[0, probe_indices]),
        "probe_target_physical_mps": json_safe(batch.targets.velocity_mps[0, probe_indices]),
        "probe_delta_from_full_physical_mps": None,
        "probe_error_physical_mps": json_safe(physical[0, probe_indices] - batch.targets.velocity_mps[0, probe_indices]),
        "local_pair_rows_preserved": None,
        "edge_access_preserved": None,
        "uniform_access_exact": None,
        "full_native_prediction_recomputed": True,
    }
    return {"result": result, "output": output, "physical": physical}


def _run_wind_case(model: Any, provider: Any, batch: Any) -> dict[str, Any]:
    row = int(batch.case_keys[0])
    if row not in set(map(int, provider.validation_rows)):
        raise ValueError("Wind organizer inspection row is outside the sealed fixed DEV membership.")
    scene = provider.make_scene(batch.scene_inputs)
    query = provider.adapter.make_query_batch(scene, batch.receivers)
    organization = model.prepare(scene, return_trace=True)
    inputs = batch.scene_inputs[0]
    active_modules = np.asarray(inputs.module_present, dtype=np.float64) > 0.5
    radii = np.asarray(inputs.module_features[active_modules, 0], dtype=np.float64)
    probe_selection = select_wind_probes(
        batch.receivers.coordinates_D[0],
        batch.receivers.flat_indices[0],
        batch.receivers.role_ids[0],
        inputs.module_centers_D[active_modules],
        radii,
    )
    probe_indices = np.asarray([probe_selection[name] for name in ("near_body", "downstream_overlap", "background")], dtype=np.int64)
    if len(set(probe_indices.tolist())) != 3:
        raise RuntimeError("Wind input-only probe selection returned duplicate native receiver rows.")
    arrays: dict[str, np.ndarray] = {}
    _save_scene_and_organization(arrays, scene, organization)
    _put_array(arrays, "query__coordinates", query.coordinates)
    _put_array(arrays, "query__native_flat_ids", query.query_ids)
    _put_array(arrays, "query__role_ids", batch.receivers.role_ids)
    _put_array(arrays, "probe__query_indices", probe_indices)
    _put_array(arrays, "probe__native_flat_ids", batch.receivers.flat_indices[0, probe_indices])
    _put_array(arrays, "probe__coordinates_D", batch.receivers.coordinates_D[0, probe_indices])
    baseline = _wind_condition(
        model=model,
        provider=provider,
        batch=batch,
        scene=scene,
        organization=organization,
        query=query,
        condition="original_full_h6",
        edge_index=0,
        probe_indices=probe_indices,
        arrays=arrays,
    )
    used_access = baseline["output"].trace["edge_used_access"][0, probe_indices]
    messages = baseline["output"].trace["edge_messages"][0, probe_indices]
    edge_index, edge_scores = rank_wind_edge(used_access, messages)
    _put_array(arrays, "probe__weighted_edge_messages", used_access[..., None] * messages)
    _put_array(arrays, "edge_ranking__weighted_message_rms", edge_scores)
    _put_array(arrays, "edge_ranking__selected_edge", np.asarray(edge_index, dtype=np.int64))
    case_results = {"original_full_h6": baseline["result"]}
    baseline_probe = np.asarray(baseline["result"]["probe_prediction_physical_mps"], dtype=np.float64)
    for condition in CONDITIONS[1:]:
        current = _wind_condition(
            model=model,
            provider=provider,
            batch=batch,
            scene=scene,
            organization=organization,
            query=query,
            condition=condition,
            edge_index=edge_index,
            probe_indices=probe_indices,
            arrays=arrays,
        )
        probe_prediction = np.asarray(current["result"]["probe_prediction_physical_mps"], dtype=np.float64)
        current["result"]["probe_delta_from_full_physical_mps"] = (probe_prediction - baseline_probe).tolist()
        if condition == "local_branch_removed":
            current["result"]["local_pair_rows_preserved"] = bool(
                current["output"].work_counts["local_rows"] == baseline["output"].work_counts["local_rows"]
                and torch.equal(
                    current["output"].trace["local_pair_query_indices"],
                    baseline["output"].trace["local_pair_query_indices"],
                )
                and torch.equal(
                    current["output"].trace["local_pair_module_indices"],
                    baseline["output"].trace["local_pair_module_indices"],
                )
            )
            current["result"]["edge_access_preserved"] = bool(torch.equal(
                current["output"].trace["edge_used_access"], baseline["output"].trace["edge_used_access"]
            ))
            current["result"]["local_reader_rows_executed_before_zero"] = int(
                current["output"].work_counts["local_rows"]
            )
        if condition == "uniform_receiver_access":
            base = current["output"].trace["edge_base_access"]
            current["result"]["uniform_access_exact"] = bool(torch.allclose(
                base,
                torch.full_like(base, 1.0 / organization.num_hyperedges),
                rtol=0.0,
                atol=1.0e-7,
            ))
        if condition == "largest_contributing_edge_removed":
            base_access = baseline["output"].trace["edge_base_access"]
            removed_base_access = current["output"].trace["edge_base_access"]
            removed_access = current["output"].trace["edge_used_access"]
            expected_access = base_access.clone()
            expected_access[:, :, edge_index] = 0.0
            access_frozen = torch.equal(removed_base_access, base_access) and torch.equal(removed_access, expected_access)
            current["result"]["edge_gate_access_frozen_without_renormalization"] = bool(access_frozen)
            current["result"]["nonlinear_field_recomputed_under_gate"] = True
        case_results[condition] = current["result"]
    probe_records = []
    for name, query_index in zip(("near_body", "downstream_overlap", "background"), probe_indices, strict=True):
        probe_records.append({
            "name": name,
            "selection_rule": {
                "near_body": "near_turbine-role receiver closest to a prescribed module-radius surface",
                "downstream_overlap": "downstream-envelope receiver inside the greatest count of native 10D by 1.5D support boxes",
                "background": "background-role receiver farthest in 3D from an active module",
            }[name],
            "query_index": int(query_index),
            "native_flat_id": int(batch.receivers.flat_indices[0, query_index]),
            "coordinates_D": json_safe(batch.receivers.coordinates_D[0, query_index]),
            "role": tuple(WIND_ROLE_INDEX)[int(batch.receivers.role_ids[0, query_index])],
        })
    return {
        "case_key": row,
        "case_name": inputs.case_name,
        "layout_index": int(inputs.layout_index),
        "n_turbines": int(np.count_nonzero(inputs.module_present > 0.5)),
        "wind_direction_deg": float(inputs.wind_direction_deg),
        "case_results": case_results,
        "edge_ranking": {
            "rule": "RMS across three probes of the Euclidean norm of alpha_h times the measured latent edge message",
            "scores": edge_scores.tolist(),
            "selected_edge": int(edge_index),
        },
        "probes": probe_records,
        "work_counts_are_full_execution": True,
        "arrays": arrays,
    }


def _output_directory(round_root: Path, requested: Path | None, dataset: str, checkpoint_hash: str) -> Path:
    output = requested or round_root / "evaluation" / "content_hypergraph_inspection" / f"{dataset}_h6_{checkpoint_hash[:12]}"
    resolved = output.expanduser().resolve()
    if not resolved.is_relative_to(round_root.resolve()):
        raise ValueError("Inspection output must be inside the authorized round root.")
    if resolved.exists():
        raise FileExistsError("Inspection evidence is immutable; choose a fresh output directory.")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    round_root = args.round_root.expanduser().resolve()
    if not round_root.is_dir() or not (round_root / "authorization.json").is_file():
        raise ValueError("--round-root must be an existing authorized round with authorization.json.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    started = time.perf_counter()
    model, provider, payload = load_evaluation_components(checkpoint, device="cpu")
    recipe = payload["experiment_identity"]["recipe"]
    dataset = str(recipe["dataset"])
    protocol = _validate_checkpoint_scope(dataset, recipe, payload)
    core = model.core if dataset == "thermal" else model
    version2 = core.architecture == "A2-H6"
    if core.architecture not in {"A-H6", "A2-H6"}:
        raise ValueError("The four organization interventions require an A-H6 checkpoint.")
    if next(core.parameters()).device.type != "cpu":
        raise RuntimeError("Inspection model unexpectedly left CPU-only execution.")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = _output_directory(round_root, args.output, dataset, checkpoint_hash)
    output.mkdir(parents=True, exist_ok=False)

    if dataset == "thermal":
        if protocol == "development":
            binding = _validate_thermal_development_binding(provider)
        else:
            binding = _validate_thermal_full_binding(provider)
        selected = _select_thermal_cases(provider, protocol)
        batches = _thermal_case_batch(provider, selected)
        with torch.no_grad():
            if version2:
                from content_hypergraph_inspect_v2 import run_case
                cases = [run_case(model, provider, batch, dataset) for batch in batches]
            else:
                cases = [_run_thermal_case(model, provider, batch) for batch in batches]
    elif dataset == "wind":
        binding = (
            _validate_wind_development_binding(provider)
            if protocol == "development"
            else _validate_wind_full_binding(provider)
        )
        selected_rows = _select_wind_rows(provider, protocol)
        batches = list(provider.native_evaluation_batches(
            split="validation", case_keys=selected_rows, microbatch_size=1
        ))
        selected = selected_rows
        with torch.no_grad():
            if version2:
                from content_hypergraph_inspect_v2 import run_case
                cases = [run_case(model, provider, batch, dataset) for batch in batches]
            else:
                cases = [_run_wind_case(model, provider, batch) for batch in batches]
    else:
        raise ValueError(f"Unsupported content-hypergraph inspection dataset {dataset!r}.")

    array_files = []
    all_results = []
    for index, case in enumerate(cases):
        key = str(case["case_key"])
        array_path = output / f"inspection_{index:02d}_{key}.npz"
        np.savez_compressed(array_path, **case.pop("arrays"))
        with np.load(array_path) as stored:
            array_shapes = {name: list(stored[name].shape) for name in stored.files}
        array_files.append({
            "path": array_path.name,
            "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "array_shapes": array_shapes,
        })
        all_results.append(case)

    result = {
        "schema": "content_hypergraph_same_weight_inspection_v3" if version2 else "content_hypergraph_same_weight_inspection_v2",
        "status": "completed",
        "dataset": dataset,
        "architecture": core.architecture,
        "protocol": protocol,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "completed_epoch": int(payload["epoch"]),
        "provider_binding_sha256": canonical_sha256(payload["experiment_identity"]["provider_identity"]),
        "selection": (
            "fixed DEV membership and input geometry/counts only; no model-error selection"
            if protocol == "development"
            else "independently bound full_train1000 validation membership and input geometry/counts only; no model-error selection"
        ),
        "validation_panel": _validation_panel_metadata(dataset, protocol, binding, selected),
        "validation_binding": binding,
        "conditions": list(CONDITIONS_V2 if version2 else CONDITIONS),
        "cases": all_results,
        "array_files": array_files,
        "device": "cpu",
        "optimizer_updates": 0,
        "solver_attempts": 0,
        "target_access": (
            {
                "source_partition": "original_h5_test_split_exposed_validation_panel",
                "exposed_validation_target_values_read_for_metrics": True,
                "unexposed_original_test_targets_read": False,
            }
            if dataset == "thermal"
            else {
                "source_partition": binding["validation_scope"],
                "exposed_validation_target_values_read_for_metrics": True,
                "wind_test_target_values_read": False,
                "unexposed_test_target_values_read": False,
            }
        ),
        "execution_note": (
            "All interventions retain the dense full-H6 edge reader. Edge gates zero contributions after message computation; "
            "the local-zero hook preserves the packed rows and reports their original executed counts. No sparse-work or "
            "executor-savings claim is measured. Thermal affine factor contraction and native extraction accumulate in FP64 "
            "from the unchanged FP32 neural factors; no FP64 neural forward is used. Learned incidence and interventions "
            "describe conditional model use, not physical causality."
        ),
        "elapsed_seconds_cpu": time.perf_counter() - started,
    }
    if version2:
        result["execution_note"] = (
            "All six edge messages and all packed local rows execute in every condition. Fixed input-declared edge0 "
            "is zeroed after headwise softmax without renormalization; local gates zero the executed local branch. "
            "No sparse executor savings are measured. Thermal contractions and native extraction use FP64 with "
            "unchanged FP32 neural factors. Actual donor-normalized W and head access describe learned use, not causality."
        )
    (output / "result.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe({k: v for k, v in result.items() if k != "cases"}), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
