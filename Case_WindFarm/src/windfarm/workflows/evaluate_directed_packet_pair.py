"""Native Wind evaluation for the paired directional and summary MM scorers.

The evaluator stores action-aware input descriptors separately from measured
errors. It uses one shared native Q512 fit panel per training layout and two
Q2048 development panels whose native flat indices are disjoint by rejection
against the first panel's full query-index union.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.action_aware_frontier import (
    TypedActionSources,
    describe_frontier_action,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    MechanismPlan,
)
from honf_forward_core.interface_fields.budgeted_frontier import enumerate_frontier_cuts
from honf_forward_core.interface_fields.directional_packets import (
    DirectionalPacketOrganizer,
    DirectionalPacketScores,
    _transform_coordinates,
)
from honf_runtime.compat import load_trusted_checkpoint

from ..data import WindFarmNativeView, case_batch
from ..model import WindFarmForwardModel
from ..normalization import VelocityNormalizer
from . import directed_packet_pair as training
from .evaluate_forward import _compact_metadata, load_checkpoint
from .joint_forward import (
    DEFAULT_AUDIT_QUERY_COUNTS,
    NativeRoleCatalogue,
    NativeRoleCatalogueCache,
    NativeRoleSample,
    _batch_from_sample,
    _inverse_standardized_tensor,
    _native_rng,
    sample_native_role_queries,
)
from .native_cover_panel import TrainingLayout

OUTPUT_DIR = training.DEFAULT_OUTPUT / "native_evaluation"
EVALUATION_STOP_UTC = datetime(2026, 10, 3, 0, 14, tzinfo=timezone.utc)
EVALUATION_FINAL_UTC = EVALUATION_STOP_UTC + timedelta(hours=1)
FIT_QUERY_SEED = 20261003
DEV_QUERY_SEED = 20261004
ROLE_FEATURE_WIDTH = 8
MM_BUDGET = 0.9
EXPECTED_DEVICE_UUID = "GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e"


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _check_deadline(activity: str, sampling_stop_utc: datetime) -> None:
    now = datetime.now(timezone.utc)
    if now >= sampling_stop_utc:
        raise RuntimeError(
            f"Wind {activity} not started: authorized new training/sampling stop is {sampling_stop_utc.isoformat()}"
        )


def _check_final_deadline(activity: str, final_deadline_utc: datetime) -> None:
    now = datetime.now(timezone.utc)
    if now >= final_deadline_utc:
        raise RuntimeError(
            f"Wind {activity} not started: final evaluation deadline is {final_deadline_utc.isoformat()}"
        )


def _parse_utc_deadline(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    result = datetime.fromisoformat(normalized)
    if result.tzinfo is None:
        raise argparse.ArgumentTypeError("deadlines must include a UTC offset or Z")
    return result.astimezone(timezone.utc)


def _max_receiver_tree_depth(nodes: Sequence[Any]) -> int:
    """Compute actual topology depth from the indexed child links."""

    if not nodes:
        return 0
    maximum = 0
    pending = [(0, 0)]
    while pending:
        node_index, depth = pending.pop()
        if not 0 <= node_index < len(nodes):
            raise ValueError("receiver tree child index is outside its node table")
        maximum = max(maximum, depth)
        node = nodes[node_index]
        if node.left is None:
            if node.right is not None:
                raise ValueError("receiver tree node has a right child but no left child")
            continue
        if node.right is None:
            raise ValueError("receiver tree split node is missing its right child")
        pending.append((int(node.left), depth + 1))
        pending.append((int(node.right), depth + 1))
    return maximum


def _layout_from_record(record: Mapping[str, Any]) -> TrainingLayout:
    return TrainingLayout(
        layout_index=int(record["layout_index"]),
        rows=tuple(int(row) for row in record["rows_direction_order"]),
        turbine_count=int(record["turbine_count"]),
        feature_vector=tuple(float(value) for value in record["geometry_feature_vector"]),
    )


def _row_for_direction(view: WindFarmNativeView, layout: TrainingLayout, direction: float = 285.0) -> int:
    for row in layout.rows:
        if math.isclose(float(view.metadata["wd_deg"][int(row)]), direction, abs_tol=1e-8):
            return int(row)
    raise ValueError(f"layout {layout.layout_index} has no native {direction:g}-degree row")


def _role_indices(catalogue: NativeRoleCatalogue, role: str) -> np.ndarray:
    indices = catalogue.role_indices.get(role)
    if indices is None:
        return np.arange(catalogue.coordinates_D.shape[0], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _disjoint_panel(
    case: Any,
    rng: np.random.Generator,
    role_query_counts: Mapping[str, int],
    *,
    catalogue_cache: NativeRoleCatalogueCache,
    excluded_flat_indices: np.ndarray,
    sampling_stop_utc: datetime = EVALUATION_STOP_UTC,
) -> tuple[NativeRoleSample, dict[str, Any]]:
    """Sample the native role CDF conditional on not using panel-one cells."""

    catalogue = catalogue_cache.get(case)
    cell_count = int(catalogue.coordinates_D.shape[0])
    excluded = np.zeros(cell_count, dtype=bool)
    excluded_unique = np.unique(np.asarray(excluded_flat_indices, dtype=np.int64))
    excluded = excluded_unique[(excluded_unique >= 0) & (excluded_unique < cell_count)]
    excluded_mask = np.zeros(cell_count, dtype=bool)
    excluded_mask[excluded] = True
    pieces: list[np.ndarray] = []
    target_pieces: list[np.ndarray] = []
    flat_pieces: list[np.ndarray] = []
    role_slices: dict[str, slice] = {}
    role_support: dict[str, float] = {}
    role_metadata: dict[str, Any] = {}
    offset = 0
    for role in training.ROLE_NAMES:
        cdf = np.asarray(catalogue.role_cdf[role], dtype=np.float64)
        native_indices = _role_indices(catalogue, role)
        if cdf.shape != native_indices.shape:
            raise ValueError(f"native {role} CDF and flat-index axes do not align")
        remaining_measure = np.diff(np.concatenate(([0.0], cdf)))
        excluded_by_role = excluded_mask[native_indices]
        excluded_fraction = float(remaining_measure[excluded_by_role].sum(dtype=np.float64))
        remaining_fraction = max(0.0, 1.0 - excluded_fraction)
        if remaining_fraction <= 0.0:
            raise ValueError(f"panel one exhausted all native quadrature support for role {role}")
        count = int(role_query_counts[role])
        selected_parts: list[np.ndarray] = []
        selected_count = 0
        while selected_count < count:
            _check_deadline("disjoint development panel sampling", sampling_stop_utc)
            needed = count - selected_count
            draw_count = max(64, 2 * needed)
            positions = np.searchsorted(cdf, rng.random(draw_count), side="right")
            positions = np.minimum(positions, cdf.size - 1)
            candidate = native_indices[positions]
            accepted = candidate[~excluded_mask[candidate]]
            if accepted.size:
                take = accepted[:needed]
                selected_parts.append(take)
                selected_count += int(take.size)
        selected = np.concatenate(selected_parts)[:count].astype(np.int64, copy=False)
        pieces.append(catalogue.coordinates_D[selected])
        target_pieces.append(np.asarray(case.run.U[selected], dtype=np.float32).copy())
        flat_pieces.append(selected)
        role_slices[role] = slice(offset, offset + count)
        support = float(catalogue.role_support_volume_m3[role])
        role_support[role] = support
        role_metadata[role] = {
            "native_cell_count": int(native_indices.size),
            "excluded_unique_native_cells": int(excluded_by_role.sum()),
            "native_support_volume_m3": support,
            "excluded_support_volume_m3": float(support * excluded_fraction),
            "remaining_support_volume_m3": float(support * remaining_fraction),
            "remaining_support_fraction": remaining_fraction,
            "sampling_measure": "original native quadrature CDF conditioned by rejection of panel-one flat indices",
        }
        offset += count
    sample = NativeRoleSample(
        coordinates_D=np.concatenate(pieces, axis=0),
        target_mps=np.concatenate(target_pieces, axis=0),
        flat_indices=np.concatenate(flat_pieces, axis=0),
        role_slices=role_slices,
        role_support_volume_m3=role_support,
        role_sample_counts={role: int(role_query_counts[role]) for role in training.ROLE_NAMES},
        geometry_sha256=catalogue.geometry_sha256,
    )
    overlap_by_role = {
        role: int(
            np.intersect1d(
                np.unique(np.asarray(excluded_flat_indices, dtype=np.int64)),
                np.unique(sample.flat_indices[sample.role_slices[role]]),
                assume_unique=True,
            ).size
        )
        for role in training.ROLE_NAMES
    }
    intersection_count = int(
        np.intersect1d(
            np.unique(np.asarray(excluded_flat_indices, dtype=np.int64)),
            np.unique(sample.flat_indices),
            assume_unique=True,
        ).size
    )
    if intersection_count != 0 or any(overlap_by_role.values()):
        raise RuntimeError("disjoint Q2048 Wind panels share one or more native flat indices")
    metadata = {
        "cross_panel_unique_flat_index_intersection_count": intersection_count,
        "cross_panel_intersection_count_by_panel_two_role": overlap_by_role,
        "panel_one_unique_flat_index_count": int(np.unique(excluded_flat_indices).size),
        "panel_two_unique_flat_index_count": int(np.unique(sample.flat_indices).size),
        "finite_exclusion_note": (
            "panel two estimates each role's native quadrature measure conditional on excluding the "
            "finite union of panel-one native flat indices"
        ),
        "role_support": role_metadata,
        "panel_two_role_counts": {role: int(role_query_counts[role]) for role in training.ROLE_NAMES},
    }
    return sample, metadata


def _sample_metadata(sample: NativeRoleSample) -> dict[str, Any]:
    return {
        "flat_indices": np.asarray(sample.flat_indices, dtype=np.int64),
        "coordinates_D": np.asarray(sample.coordinates_D, dtype=np.float32),
        "reference_velocity_mps": np.asarray(sample.target_mps, dtype=np.float32),
        "role_slices": {key: [int(value.start), int(value.stop)] for key, value in sample.role_slices.items()},
        "role_query_counts": {key: int(value) for key, value in sample.role_sample_counts.items()},
        "role_support_volume_m3": {key: float(value) for key, value in sample.role_support_volume_m3.items()},
    }


def _restore_native_role_sample(
    case: Any,
    payload: Mapping[str, Any],
    catalogue_cache: NativeRoleCatalogueCache,
) -> NativeRoleSample:
    """Restore a persisted panel and verify its native coordinates and targets."""

    catalogue = catalogue_cache.get(case)
    flat_indices = np.asarray(payload["flat_indices"], dtype=np.int64)
    coordinates = np.asarray(payload["coordinates_D"], dtype=np.float32)
    target = np.asarray(payload["reference_velocity_mps"], dtype=np.float32)
    if (
        flat_indices.ndim != 1
        or coordinates.shape != (flat_indices.size, catalogue.coordinates_D.shape[1])
        or target.shape != (flat_indices.size, 3)
        or (flat_indices < 0).any()
        or (flat_indices >= catalogue.coordinates_D.shape[0]).any()
    ):
        raise ValueError("persisted Wind native sample has invalid shapes or flat indices")
    expected_coordinates = np.asarray(catalogue.coordinates_D[flat_indices], dtype=np.float32)
    expected_target = np.asarray(case.run.U[flat_indices], dtype=np.float32)
    if not np.allclose(coordinates, expected_coordinates, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("persisted Wind sample coordinates differ from the current native row")
    if not np.array_equal(target, expected_target):
        raise ValueError("persisted Wind sample targets differ from the current native row")
    role_slices = {str(role): slice(int(bounds[0]), int(bounds[1])) for role, bounds in payload["role_slices"].items()}
    role_counts = {str(role): int(count) for role, count in payload["role_query_counts"].items()}
    support_volume = {str(role): float(volume) for role, volume in payload["role_support_volume_m3"].items()}
    expected_roles = set(training.ROLE_NAMES)
    if (
        set(role_slices) != expected_roles
        or set(role_counts) != expected_roles
        or set(support_volume) != expected_roles
    ):
        raise ValueError("persisted Wind sample does not contain all five protected roles")
    if any(role_slices[role].stop - role_slices[role].start != role_counts[role] for role in training.ROLE_NAMES):
        raise ValueError("persisted Wind sample role slices and query counts disagree")
    return NativeRoleSample(
        coordinates_D=coordinates,
        target_mps=target,
        flat_indices=flat_indices,
        role_slices=role_slices,
        role_support_volume_m3=support_volume,
        role_sample_counts=role_counts,
        geometry_sha256=catalogue.geometry_sha256,
    )


def _role_rmse(
    prediction_mps: torch.Tensor, reference_mps: torch.Tensor, slices: Mapping[str, slice]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for role in training.ROLE_NAMES:
        error = prediction_mps[:, slices[role]] - reference_mps[:, slices[role]]
        values[role] = float(error.square().mean().sqrt().detach().cpu())
    return values


def _inverse_prediction(model: WindFarmForwardModel, prediction: torch.Tensor) -> torch.Tensor:
    return _inverse_standardized_tensor(prediction, model.velocity_transform)


def _predict_plan(
    model: WindFarmForwardModel,
    encoded: Any,
    plan: MechanismPlan,
    batch: Any,
    *,
    device: torch.device,
    final_deadline_utc: datetime | None = None,
) -> tuple[torch.Tensor, dict[str, int], float]:
    if final_deadline_utc is not None:
        _check_final_deadline("native prepare/decode inference", final_deadline_utc)
    model.core.backend.set_cover_executor("dense_masked")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prepared = model.core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(plan,))
    decoded = model.core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=batch.query_features,
        receiver_chunk_size=512,
        return_interaction_aux=True,
    )
    prediction = decoded["pred_field"]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    query_aux = decoded.get("_interaction_aux", {})
    query_ledger = {
        str(key): int(value.detach().cpu().item() if torch.is_tensor(value) else value)
        for key, value in query_aux.items()
        if str(key).startswith(("cover_qm_", "cover_qe_", "cover_executor_"))
        and (not torch.is_tensor(value) or value.numel() == 1)
    }
    ledger = {
        "preparation": training._ledger(prepared),
        "query_execution": query_ledger,
    }
    return prediction, ledger, float(elapsed)


def _role_features(catalogue: NativeRoleCatalogue) -> torch.Tensor:
    rows: list[np.ndarray] = []
    for role in training.ROLE_NAMES:
        indices = _role_indices(catalogue, role)
        cdf = np.asarray(catalogue.role_cdf[role], dtype=np.float64)
        probability = np.diff(np.concatenate(([0.0], cdf)))
        support = float(catalogue.role_support_volume_m3[role])
        weights = probability * support
        coords = np.asarray(catalogue.coordinates_D[indices], dtype=np.float64)
        total = float(weights.sum(dtype=np.float64))
        mean = (coords * weights[:, None]).sum(axis=0, dtype=np.float64) / max(total, 1e-30)
        variance = ((coords - mean) ** 2 * weights[:, None]).sum(axis=0, dtype=np.float64) / max(total, 1e-30)
        rows.append(
            np.concatenate(
                (mean, np.sqrt(np.maximum(variance, 0.0)), [math.log1p(indices.size), math.log1p(max(support, 0.0))])
            )
        )
    result = torch.as_tensor(np.stack(rows), dtype=torch.float32)
    if result.shape != (5, ROLE_FEATURE_WIDTH):
        raise RuntimeError("Wind receiver-role feature schema changed unexpectedly")
    return result


def _native_module_measures(encoded: Any, scores: DirectionalPacketScores) -> torch.Tensor:
    if scores.receiver_tree is None or scores.active_module_ids is None:
        raise RuntimeError("MM-local scores omitted native source IDs or anchor measures")
    measures = torch.zeros_like(encoded.module_present[0])
    ids = scores.active_module_ids.to(device=measures.device, dtype=torch.long)
    weights = scores.receiver_tree.universe.weights.to(device=measures.device, dtype=measures.dtype)
    if ids.shape != weights.shape:
        raise RuntimeError("native module source IDs and measures do not align")
    return measures.index_copy(0, ids, weights)


def _packet_action_features(
    organizer: DirectionalPacketOrganizer,
    scores: DirectionalPacketScores,
    encoded: Any,
    local_plan: MechanismPlan,
    cut: tuple[int, ...],
) -> torch.Tensor:
    if scores.receiver_tree is None or scores.node_embeddings is None or scores.module_embeddings is None:
        raise RuntimeError("directional action scores are missing their receiver/source embeddings")
    tree = scores.receiver_tree
    node_ids = torch.as_tensor(cut, device=scores.node_embeddings.device, dtype=torch.long)
    packet_embedding = scores.node_embeddings.index_select(0, node_ids)
    packet_centers: list[torch.Tensor] = []
    for node in cut:
        local_ids = torch.as_tensor(
            tree.nodes[node].anchor_indices, device=tree.universe.coordinates.device, dtype=torch.long
        )
        weights = tree.universe.weights.index_select(0, local_ids)
        coords = tree.universe.coordinates.index_select(0, local_ids)
        packet_centers.append((coords * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-8))
    centers = torch.stack(packet_centers)
    frame = None if organizer._physical_frame.numel() == 0 else organizer._physical_frame
    source_coordinates = _transform_coordinates(encoded.module_centers[0], frame)
    source_measures = _native_module_measures(encoded, scores)
    permissions = local_plan.permission_matrix("MM").index_select(0, node_ids).detach()
    source_scores = scores.mechanism_logits["MM"].index_select(0, node_ids).detach()
    described = describe_frontier_action(
        packet_embedding,
        centers,
        {
            "MM": TypedActionSources(
                source_features=scores.module_embeddings,
                source_coordinates=source_coordinates,
                source_measure=source_measures,
                permission=permissions,
                score=source_scores,
            )
        },
        mechanism_order=("MM",),
    )
    if scores.raw_pair_features is None:
        raise RuntimeError("directional packet rows omitted their signed geometry features")
    raw = scores.raw_pair_features.index_select(0, node_ids)
    measure = source_measures.to(device=raw.device, dtype=raw.dtype)
    geometry = (raw * measure[None, :, None]).sum(dim=1) / measure.sum().clamp_min(1e-8)
    return torch.cat((described, geometry), dim=-1).detach().to(device="cpu", dtype=torch.float32)


def _full_action_features(
    organizer: DirectionalPacketOrganizer,
    scores: DirectionalPacketScores,
    encoded: Any,
) -> torch.Tensor:
    local_plan = organizer._local_permission_plans(
        (scores,),
        encoded,
        hard=True,
        frontier_cuts=((0,),),
        budget_fractions={"MM": 1.0},
    )
    valid = encoded.module_present[0] > 0.5
    root_permissions = local_plan[0].permission_matrix("MM")[0]
    if not bool((root_permissions[valid] > 0).all()):
        raise RuntimeError("MM budget 1.0 did not retain every valid native full-access source")
    # The full descriptor describes one root packet with all valid MM sources.
    return _packet_action_features(organizer, scores, encoded, local_plan[0], (0,))


def _direct_mm_plan(
    base_plan: MechanismPlan,
    encoded: Any,
    pair_matrix: torch.Tensor,
) -> MechanismPlan:
    """Attach an exact native-coordinate MM matrix while leaving other routes full."""

    active = encoded.module_present[0] > 0.5
    weights = pair_matrix.to(device=active.device, dtype=encoded.module_present.dtype)
    weights = weights * active.to(weights.dtype)[:, None]
    weights = weights * active.to(weights.dtype)[None, :]
    weights = weights.clone()
    weights.fill_diagonal_(0.0)
    return base_plan.with_direct_pair_access(
        "MM",
        encoded.module_centers[0],
        weights,
        receiver_validity=active,
    )


def _source_union_collapse(pair_matrix: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    common_sources = pair_matrix.bool().any(dim=0) & active
    result = active[:, None] & common_sources[None, :]
    result = result.clone()
    result.fill_diagonal_(False)
    return result


def _direct_geometry_same_work(
    organizer: DirectionalPacketOrganizer,
    encoded: Any,
    parent_matrix: torch.Tensor,
) -> torch.Tensor:
    """Keep each receiver's closest sources at its exact parent MM degree."""

    eligible = (encoded.module_present[0][:, None] > 0.5) & (encoded.module_present[0][None, :] > 0.5)
    eligible = eligible.clone()
    eligible.fill_diagonal_(False)
    active = encoded.module_present[0] > 0.5
    frame = None if organizer._physical_frame.numel() == 0 else organizer._physical_frame
    coords = _transform_coordinates(encoded.module_centers[0], frame)
    parent_weights = parent_matrix.to(dtype=encoded.module_present.dtype)
    parent_weights = parent_weights * active.to(parent_weights.dtype)[:, None]
    parent_weights = parent_weights * active.to(parent_weights.dtype)[None, :]
    parent_weights = parent_weights.masked_fill(~eligible, 0.0)
    result = torch.zeros_like(parent_weights)
    for receiver_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
        candidates = torch.nonzero(eligible[receiver_id], as_tuple=False).flatten()
        row_values = parent_weights[receiver_id, candidates]
        selected_values = row_values[row_values > 0]
        keep_count = int(selected_values.numel())
        if keep_count > int(candidates.numel()):
            raise ValueError("MM receiver degree exceeds its eligible native source population")
        if keep_count == 0:
            continue
        delta = coords.index_select(0, candidates) - coords[receiver_id]
        scores = -delta.square().sum(dim=-1)
        order = torch.argsort(scores, descending=True, stable=True)[:keep_count]
        nearest = candidates.index_select(0, order)
        weighted_values = torch.sort(selected_values, descending=True).values
        result[receiver_id, nearest] = weighted_values
    if not torch.equal((result > 0).sum(dim=1), (parent_weights > 0).sum(dim=1)):
        raise RuntimeError("degree-matched geometry control changed a receiver's MM degree")
    if not torch.allclose(
        result.to(torch.float64).sum(dim=1),
        parent_weights.to(torch.float64).sum(dim=1),
        atol=1.0e-12,
        rtol=1.0e-12,
    ):
        raise RuntimeError("degree-matched geometry control changed receiver permission mass")
    for receiver_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
        if not torch.equal(
            torch.sort(result[receiver_id][result[receiver_id] > 0]).values,
            torch.sort(parent_weights[receiver_id][parent_weights[receiver_id] > 0]).values,
        ):
            raise RuntimeError("degree-matched geometry control changed receiver permission weights")
    return result


def _effective_same_work_rewire(
    parent_matrix: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Cyclically rewire eligible receiver/source pairs with exact action work."""

    eligible = active[:, None] & active[None, :]
    eligible = eligible.clone()
    eligible.fill_diagonal_(False)
    eligible_flat = torch.nonzero(eligible.flatten(), as_tuple=False).flatten()
    parent_weights = parent_matrix.to(dtype=torch.float32) * eligible.to(dtype=torch.float32)
    parent = parent_weights > 0
    work = int(parent.sum().item())
    if work <= 0 or work >= int(eligible_flat.numel()):
        return None, {
            "effective": False,
            "reason": "no nontrivial exact-work rewire exists for empty or complete support",
            "pair_work": work,
        }
    parent_flat = parent.flatten()
    sorted_weights = torch.sort(parent_weights[parent], descending=True).values
    # The first cyclic origin that changes the set is a deterministic uniform-
    # pair rewire. It preserves valid receiver/source IDs and excludes self.
    for offset in range(1, int(eligible_flat.numel()) + 1):
        selected = eligible_flat.roll(shifts=offset)[:work]
        candidate_support_flat = torch.zeros_like(parent_flat)
        candidate_support_flat[selected] = True
        if torch.equal(candidate_support_flat, parent_flat):
            continue
        candidate_flat = torch.zeros_like(parent_weights.flatten())
        candidate_flat[selected] = sorted_weights
        candidate = candidate_flat.reshape_as(parent_weights)
        parent_mass = float(parent_weights.to(torch.float64).sum().item())
        candidate_mass = float(candidate.to(torch.float64).sum().item())
        return candidate, {
            "effective": True,
            "method": "deterministic cyclic eligible-pair rewire",
            "preserves_exact_pair_work": True,
            "preserves_permission_mass": math.isclose(candidate_mass, parent_mass, rel_tol=1.0e-12, abs_tol=1.0e-12),
            "parent_permission_mass": parent_mass,
            "control_permission_mass": candidate_mass,
            "permission_mass_difference": candidate_mass - parent_mass,
            "preserves_permission_weight_multiset": True,
            "preserves_source_or_receiver_degrees": False,
            "cyclic_offset": int(offset),
            "pair_work": int(candidate_support_flat.sum().item()),
        }
    return None, {
        "effective": False,
        "reason": "cyclic eligible-pair scan did not find a changed mask",
        "pair_work": work,
    }


def _pair_mask_changes(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    before_weights = reference.to(dtype=torch.float32)
    after_weights = candidate.to(device=before_weights.device, dtype=torch.float32)
    before = before_weights > 0
    after = after_weights > 0
    added = torch.nonzero(after & ~before, as_tuple=False).detach().cpu()
    removed = torch.nonzero(before & ~after, as_tuple=False).detach().cpu()
    changed = torch.nonzero((after_weights - before_weights).abs() > 1.0e-7, as_tuple=False)
    changed_weights = (
        torch.stack(
            (
                before_weights[changed[:, 0], changed[:, 1]],
                after_weights[changed[:, 0], changed[:, 1]],
            ),
            dim=-1,
        )
        .detach()
        .cpu()
        if changed.numel()
        else torch.empty((0, 2), dtype=torch.float32)
    )
    return {
        "added_native_receiver_source_ids": added,
        "removed_native_receiver_source_ids": removed,
        "changed_permission_native_receiver_source_ids": changed.detach().cpu(),
        "changed_permission_before_after_weights": changed_weights,
        "added_pair_count": int(added.shape[0]),
        "removed_pair_count": int(removed.shape[0]),
        "changed_permission_count": int(changed.shape[0]),
    }


def _training_exposure_audit(output: Path) -> dict[str, Any]:
    """Summarize valid paired update logs, including per-layout cut exposure."""

    audit: dict[str, Any] = {}
    for arm in ("w_dir", "w_summary"):
        log_path = output / arm / "updates.jsonl"
        if not log_path.is_file():
            raise FileNotFoundError(f"completed paired update ledger is missing: {log_path}")
        phase_counts: Counter[str] = Counter()
        per_layout_cuts: dict[str, Counter[str]] = defaultdict(Counter)
        per_layout_k: dict[str, Counter[str]] = defaultdict(Counter)
        per_layout_candidate_actions: dict[str, list[int]] = defaultdict(list)
        optimizer_updates = 0
        read_only_full_checks = 0
        restricted_exposures = 0
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            phase = str(record.get("phase", "unknown"))
            phase_counts[phase] += 1
            optimizer_updates += int(
                bool(record.get("physical_optimizer_step")) or bool(record.get("scorer_optimizer_step"))
            )
            if "readonly_full" in phase:
                read_only_full_checks += 1
            if phase not in {"scorer_only_restricted", "coadapt_restricted"}:
                continue
            action = record.get("selected_action") or {}
            cut = action.get("selected_cut_nodes", action.get("requested_cut_nodes"))
            if cut is None:
                continue
            layout = str(int(record["layout_index"]))
            signature = ",".join(str(int(value)) for value in cut)
            per_layout_cuts[layout][signature] += 1
            k = action.get("realized_nonredundant_MM_packet_count", action.get("requested_cut_K"))
            if k is not None:
                per_layout_k[layout][str(int(k))] += 1
            exact_candidates = action.get("distinct_exact_action_count")
            if exact_candidates is not None:
                per_layout_candidate_actions[layout].append(int(exact_candidates))
            restricted_exposures += 1
        layout_ids = sorted(set(per_layout_cuts) | set(per_layout_k), key=int)
        audit[arm] = {
            "valid_update_ledger_path": str(log_path),
            "phase_row_counts": dict(sorted(phase_counts.items())),
            "logged_optimizer_step_count": optimizer_updates,
            "read_only_full_check_count": read_only_full_checks,
            "restricted_action_exposure_count": restricted_exposures,
            "per_layout_cut_node_tuple_exposure": {
                layout: dict(sorted(per_layout_cuts.get(layout, {}).items())) for layout in layout_ids
            },
            "per_layout_realized_nonredundant_k_exposure": {
                layout: dict(sorted(per_layout_k.get(layout, {}).items())) for layout in layout_ids
            },
            "per_layout_distinct_exact_action_candidate_count_range": {
                layout: {
                    "min": min(per_layout_candidate_actions[layout]),
                    "max": max(per_layout_candidate_actions[layout]),
                    "mean": float(np.mean(per_layout_candidate_actions[layout])),
                    "records": len(per_layout_candidate_actions[layout]),
                }
                for layout in layout_ids
                if per_layout_candidate_actions.get(layout)
            },
        }
    return audit


def _materialize_model(
    state_dict: Mapping[str, Any],
    base_checkpoint: Path,
    materialization_batch: Any,
    *,
    module_feature_dim: int,
    environment_feature_dim: int,
    device: torch.device,
    feature_mode: str | None = None,
) -> tuple[WindFarmForwardModel, DirectionalPacketOrganizer | None]:
    model, _payload = load_checkpoint(
        base_checkpoint,
        device=device,
        materialization_batch=materialization_batch,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    if feature_mode is None:
        return model, None
    organizer = DirectionalPacketOrganizer(
        state_dim=int(model.config.hidden_dim),
        module_feature_dim=int(module_feature_dim),
        environment_feature_dim=int(environment_feature_dim),
        hidden_dim=96,
        feature_mode=feature_mode,
        geometry_length_scale=1.0,
        scorer_lr=training.SCORER_LR,
    ).to(device)
    return model, organizer


def _load_arm_model(
    arm: str,
    checkpoint_path: Path,
    lock: Mapping[str, Any],
    materialization_batch: Any,
    *,
    device: torch.device,
) -> tuple[WindFarmForwardModel, DirectionalPacketOrganizer, dict[str, Any]]:
    state = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    mode = "directional" if arm == "w_dir" else "summary"
    model, organizer = _materialize_model(
        state["model_state_dict"],
        Path(lock["base_model_checkpoint"]),
        materialization_batch,
        module_feature_dim=int(materialization_batch.module_features.shape[-1]),
        environment_feature_dim=int(materialization_batch.env_features.shape[-1]),
        device=device,
        feature_mode=mode,
    )
    assert organizer is not None
    organizer.load_state_dict(state["organizer_state_dict"], strict=True)
    ramp_update = int(state.get("restricted_optimizer_updates", state.get("scorer_updates", 0)))
    organizer.set_update(ramp_update)
    if int(state.get("scorer_updates", 0)) >= 100 and float(organizer.residual_ramp) != 1.0:
        raise RuntimeError("trained directional scorer did not restore its completed residual ramp")
    return model, organizer, state


def _make_scores(
    organizer: DirectionalPacketOrganizer,
    encoded: Any,
    trees: tuple[Any, ...],
    *,
    geometry_prior: bool,
    restored_update: int,
) -> tuple[DirectionalPacketScores, ...]:
    if geometry_prior:
        organizer.set_update(0)
    try:
        scores = organizer.score_cases(
            encoded,
            {
                "module_states": encoded.module_tokens,
                "environment_states": encoded.env_tokens,
                "global_state": encoded.global_token,
            },
            trees,
        )
    finally:
        if geometry_prior:
            organizer.set_update(restored_update)
    return scores


def _action_rows_for_case(
    *,
    model: WindFarmForwardModel,
    organizer: DirectionalPacketOrganizer,
    encoded: Any,
    base_tree: Any,
    scores: tuple[DirectionalPacketScores, ...],
    batch: Any,
    sample: NativeRoleSample,
    case: Any,
    sample_key: str,
    prediction_payloads: dict[str, Any],
    final_deadline_utc: datetime,
    full_prediction_std: torch.Tensor,
    full_prediction_ledger: Mapping[str, int],
    full_prediction_seconds: float,
    teacher_prediction_std: torch.Tensor,
    initial_g_prediction_std: torch.Tensor | None,
    role_features: torch.Tensor,
    case_context: Mapping[str, Any],
    support_source: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    local_tree = scores[0].receiver_tree
    if local_tree is None:
        raise RuntimeError("MM-local score did not expose its receiver tree")
    all_cuts = enumerate_frontier_cuts(local_tree, max_depth=2)
    candidates: list[tuple[tuple[int, ...], MechanismPlan, dict[str, Any], torch.Tensor]] = []
    seen: set[tuple[float, ...]] = set()
    duplicate_cuts: list[list[int]] = []
    for raw_cut in all_cuts:
        _local, compiled, detail = training._action_descriptor(
            organizer, scores, encoded, base_tree, tuple(int(value) for value in raw_cut)
        )
        signature = tuple(float(value) for value in detail["exact_action_signature"])
        if signature in seen:
            duplicate_cuts.append([int(value) for value in raw_cut])
            continue
        seen.add(signature)
        candidates.append(
            (
                tuple(int(value) for value in raw_cut),
                compiled,
                detail,
                _packet_action_features(organizer, scores[0], encoded, _local, tuple(int(value) for value in raw_cut)),
            )
        )

    active = encoded.module_present[0] > 0.5
    eligible_pairs = int(active.sum().item()) * max(0, int(active.sum().item()) - 1)
    full_plan = MechanismPlan.full_access(base_tree, encoded.module_present[0], int(encoded.env_coords.shape[1]))
    full_matrix = active[:, None] & active[None, :]
    full_matrix = full_matrix.clone()
    full_matrix.fill_diagonal_(False)
    budget_full = torch.ones((5,), dtype=torch.float32)
    budget_sparse = torch.tensor((MM_BUDGET, 1.0, 1.0, 1.0, 1.0), dtype=torch.float32)
    input_rows: list[dict[str, Any]] = []
    reference = _inverse_prediction(model, batch.target_field)
    student_full = _inverse_prediction(model, full_prediction_std)
    teacher_full = _inverse_prediction(model, teacher_prediction_std)
    full_rmse = _role_rmse(student_full, reference, sample.role_slices)
    teacher_rmse = _role_rmse(teacher_full, reference, sample.role_slices)
    initial_g_full = None if initial_g_prediction_std is None else _inverse_prediction(model, initial_g_prediction_std)
    drift_rmse = None if initial_g_full is None else _role_rmse(student_full, initial_g_full, sample.role_slices)
    feature_full = _full_action_features(organizer, scores[0], encoded)
    native_source_measures = _native_module_measures(encoded, scores[0])
    reference_key = f"{sample_key}::native_reference"
    teacher_key = f"{sample_key}::retained_wfull"
    initial_g_key = f"{sample_key}::initial_g_u4910"
    full_key = f"{case_context['case_key']}::same_weight_full"
    prediction_payloads.setdefault(reference_key, reference.detach().cpu().to(torch.float32).squeeze(0))
    prediction_payloads.setdefault(teacher_key, teacher_full.detach().cpu().to(torch.float32).squeeze(0))
    if initial_g_full is not None:
        prediction_payloads.setdefault(initial_g_key, initial_g_full.detach().cpu().to(torch.float32).squeeze(0))
    prediction_payloads[full_key] = student_full.detach().cpu().to(torch.float32).squeeze(0)

    def append_row(
        action_key: str,
        plan: MechanismPlan,
        pair_matrix: torch.Tensor,
        packet_rows: torch.Tensor,
        budget: torch.Tensor,
        detail: Mapping[str, Any],
        *,
        full_access: bool,
        candidate_prediction_std: torch.Tensor,
        ledger: Mapping[str, int],
        elapsed: float,
        control_for_action_key: str | None = None,
        control_metadata: Mapping[str, Any] | None = None,
        mask_reference: torch.Tensor | None = None,
    ) -> None:
        candidate = _inverse_prediction(model, candidate_prediction_std)
        candidate_rmse = _role_rmse(candidate, reference, sample.role_slices)
        access_delta = _role_rmse(candidate, student_full, sample.role_slices)
        valid_weights = pair_matrix.to(device=active.device, dtype=encoded.module_present.dtype)
        valid_weights = valid_weights * active.to(valid_weights.dtype)[:, None]
        valid_weights = valid_weights * active.to(valid_weights.dtype)[None, :]
        valid_weights = valid_weights.clone()
        valid_weights.fill_diagonal_(0.0)
        valid_matrix = valid_weights > 0
        pair_count = int(valid_matrix.sum().detach().cpu())
        positive_weights = valid_weights[valid_matrix]
        fractional_weights = positive_weights[(positive_weights - 1.0).abs() > 1.0e-7]
        raw_k = detail.get("realized_nonredundant_MM_packet_count")
        nonredundant_k = 1 if full_access else (None if raw_k is None else int(raw_k))
        if full_access:
            prediction_key = full_key
        else:
            prediction_key = f"{case_context['case_key']}::{support_source}::{action_key}"
            prediction_payloads[prediction_key] = candidate.detach().cpu().to(torch.float32).squeeze(0)
        mask_changes = (
            _pair_mask_changes(valid_weights, valid_weights)
            if mask_reference is None
            else _pair_mask_changes(mask_reference, valid_weights)
        )
        row = {
            **dict(case_context),
            "support_source": support_source,
            "action_key": action_key,
            "control_for_action_key": control_for_action_key,
            "control_metadata": None if control_metadata is None else dict(control_metadata),
            "parent_requested_cut_nodes": (
                list(detail.get("requested_cut_nodes", []))
                if control_for_action_key is None
                else list((control_metadata or {}).get("parent_requested_cut_nodes", []))
            ),
            "mask_changes_vs_parent_action": mask_changes,
            "requested_cut_nodes": list(detail.get("requested_cut_nodes", [])),
            "requested_mm_cut_k": 1 if full_access else detail.get("requested_cut_K"),
            "mm_local_k": None if full_access else nonredundant_k,
            "nonredundant_k_for_action_model": nonredundant_k,
            "full_access": bool(full_access),
            "exposed_action": bool(control_for_action_key is None and not full_access),
            "trained_sparse": bool(
                control_for_action_key is None and not full_access and 0 < pair_count < eligible_pairs
            ),
            "exact_work": pair_count,
            "full_work": eligible_pairs,
            "work_fraction": float(pair_count / eligible_pairs) if eligible_pairs else 0.0,
            "MM_eligible_valid_nonself_pairs": eligible_pairs,
            "MM_pair_permission_unique_pairs": pair_count,
            "MM_permission_mass": float(valid_weights.to(torch.float64).sum().detach().cpu()),
            "MM_fractional_positive_permission_count": int(fractional_weights.numel()),
            "MM_positive_permission_weight_min": (
                float(positive_weights.min().detach().cpu()) if positive_weights.numel() else None
            ),
            "MM_positive_permission_weight_max": (
                float(positive_weights.max().detach().cpu()) if positive_weights.numel() else None
            ),
            "MM_source_union_count": int(valid_matrix.any(dim=0).sum().detach().cpu()),
            "MM_receiver_rows_with_sources": int(valid_matrix.any(dim=1).sum().detach().cpu()),
            "MM_source_counts_per_receiver": [int(value) for value in valid_matrix.sum(dim=1).cpu().tolist()],
            "candidate_role_rmse_mps": candidate_rmse,
            "same_weight_full_role_rmse_mps": full_rmse,
            "incumbent_role_rmse_mps": teacher_rmse,
            "access_error_vs_same_weight_full_role_rmse_mps": access_delta,
            "full_model_drift_from_g_u4910_role_rmse_mps": drift_rmse,
            "numerical_floor_role_mps": list(case_context["numerical_floor_role_mps"]),
            "physical_allowance_role_mps": [0.0] * len(training.ROLE_NAMES),
            "packet_rows": packet_rows.detach().cpu().to(torch.float32),
            "budget_vector": budget.detach().cpu().to(torch.float32),
            "receiver_role_features": role_features.detach().cpu().to(torch.float32),
            "receiver_coordinates": encoded.module_centers[0].detach().cpu().to(torch.float32),
            "receiver_ids": torch.arange(encoded.module_centers.shape[1], dtype=torch.long),
            "source_coordinates": encoded.module_centers[0].detach().cpu().to(torch.float32),
            "source_ids": torch.arange(encoded.module_centers.shape[1], dtype=torch.long),
            "source_measures": native_source_measures.detach().cpu().to(torch.float32),
            "valid_mm_permission_matrix": valid_matrix.detach().cpu(),
            "native_mm_permission_weights": valid_weights.detach().cpu().to(torch.float32),
            "native_hard_preparation_ledger": dict(ledger.get("preparation", {})),
            "native_query_execution_ledger": dict(ledger.get("query_execution", {})),
            "inference_seconds_prepare_decode_total": float(elapsed),
            "prediction_key": prediction_key,
            "same_weight_full_prediction_key": full_key,
            "initial_g_full_prediction_key": initial_g_key if initial_g_full is not None else None,
            "retained_wfull_prediction_key": teacher_key,
            "native_reference_prediction_key": reference_key,
            "direction_deg": float(case.wind_direction_deg),
            "query_count": int(batch.query_xy.shape[1]),
            "query_counts_by_role": dict(sample.role_sample_counts),
            "role_support_volume_m3": dict(sample.role_support_volume_m3),
        }
        input_rows.append(row)

    full_action_matrix = full_matrix.to(device=encoded.module_present.device)
    if support_source == "trained_support":
        append_row(
            "full_access",
            full_plan,
            full_action_matrix,
            feature_full,
            budget_full,
            {"realized_nonredundant_MM_packet_count": 1},
            full_access=True,
            candidate_prediction_std=full_prediction_std,
            ledger=full_prediction_ledger,
            elapsed=full_prediction_seconds,
        )
    for cut, plan, detail, packet_rows in candidates:
        action_key = "cut_" + "_".join(str(node) for node in cut)
        prediction, ledger, elapsed = _predict_plan(
            model, encoded, plan, batch, device=device, final_deadline_utc=final_deadline_utc
        )
        direct = plan.direct_pair_access_for("MM")
        if direct is None:
            raise RuntimeError("sparse candidate compiled without a direct MM pair matrix")
        append_row(
            action_key,
            plan,
            direct.weights,
            packet_rows,
            budget_sparse,
            detail,
            full_access=False,
            candidate_prediction_std=prediction,
            ledger=ledger,
            elapsed=elapsed,
        )
        parent_matrix = direct.weights.clone()
        active = encoded.module_present[0] > 0.5
        parent_matrix = parent_matrix * active.to(parent_matrix.dtype)[:, None]
        parent_matrix = parent_matrix * active.to(parent_matrix.dtype)[None, :]
        parent_matrix.fill_diagonal_(0.0)
        parent_work = int((parent_matrix > 0).sum().item())
        if 0 < parent_work < eligible_pairs:
            controls: list[tuple[str, torch.Tensor, dict[str, Any]]] = []
            union = _source_union_collapse(parent_matrix, active)
            controls.append(
                (
                    "source_union_collapse",
                    union,
                    {
                        "method": "one common MM source union exposed to every valid receiver, with self-exclusion",
                        "preserves_pair_work": int((union > 0).sum().item()) == parent_work,
                        "parent_pair_work": parent_work,
                        "control_pair_work": int((union > 0).sum().item()),
                        "parent_source_union_count": int((parent_matrix > 0).any(dim=0).sum().item()),
                        "parent_permission_mass": float(parent_matrix.to(torch.float64).sum().item()),
                        "control_permission_mass": float(union.to(torch.float64).sum().item()),
                        "extra_permission_mass": float(
                            union.to(torch.float64).sum().item() - parent_matrix.to(torch.float64).sum().item()
                        ),
                        "parent_requested_cut_nodes": list(detail["requested_cut_nodes"]),
                    },
                )
            )
            geometry = _direct_geometry_same_work(organizer, encoded, parent_matrix)
            controls.append(
                (
                    "direct_geometry_same_work",
                    geometry,
                    {
                        "method": "nearest eligible native sources per receiver by squared physical distance",
                        "preserves_receiver_degrees": torch.equal(
                            (geometry > 0).sum(dim=1), (parent_matrix > 0).sum(dim=1)
                        ),
                        "preserves_receiver_permission_mass": torch.allclose(
                            geometry.to(torch.float64).sum(dim=1),
                            parent_matrix.to(torch.float64).sum(dim=1),
                            atol=1.0e-12,
                            rtol=1.0e-12,
                        ),
                        "parent_permission_mass": float(parent_matrix.to(torch.float64).sum().item()),
                        "control_permission_mass": float(geometry.to(torch.float64).sum().item()),
                        "permission_mass_difference": float(
                            geometry.to(torch.float64).sum().item() - parent_matrix.to(torch.float64).sum().item()
                        ),
                        "preserves_exact_pair_work": int((geometry > 0).sum().item()) == parent_work,
                        "parent_pair_work": parent_work,
                        "control_pair_work": int((geometry > 0).sum().item()),
                        "parent_requested_cut_nodes": list(detail["requested_cut_nodes"]),
                    },
                )
            )
            rewired, rewire_metadata = _effective_same_work_rewire(parent_matrix, active)
            if rewired is not None:
                rewire_metadata["parent_requested_cut_nodes"] = list(detail["requested_cut_nodes"])
                controls.append(("effective_pair_rewire_same_work", rewired, rewire_metadata))
            else:
                # Preserve the failed availability audit without claiming a
                # measured control prediction.
                input_rows[-1]["same_work_rewire_availability"] = rewire_metadata
            for control_name, control_matrix, control_metadata in controls:
                control_plan = _direct_mm_plan(full_plan, encoded, control_matrix)
                control_prediction, control_ledger, control_seconds = _predict_plan(
                    model,
                    encoded,
                    control_plan,
                    batch,
                    device=device,
                    final_deadline_utc=final_deadline_utc,
                )
                append_row(
                    f"{action_key}::{control_name}",
                    control_plan,
                    control_matrix,
                    packet_rows,
                    budget_sparse,
                    {"realized_nonredundant_MM_packet_count": None},
                    full_access=False,
                    candidate_prediction_std=control_prediction,
                    ledger=control_ledger,
                    elapsed=control_seconds,
                    control_for_action_key=action_key,
                    control_metadata=control_metadata,
                    mask_reference=parent_matrix,
                )
    detail_summary = {
        "mm_local_tree_node_count": len(local_tree.nodes),
        "mm_local_tree_depth": _max_receiver_tree_depth(local_tree.nodes),
        "candidate_frontier_cut_count": len(all_cuts),
        "distinct_exact_action_count": len(candidates),
        "duplicate_cut_nodes": duplicate_cuts,
        "mm_eligible_valid_nonself_pairs": eligible_pairs,
        "scorer_residual_ramp": float(organizer.residual_ramp),
    }
    return input_rows, detail_summary


def _field_slice_payload(case: Any, z_index: int, diameter_m: float) -> dict[str, Any]:
    run = case.run
    if not 0 <= int(z_index) < int(run.nz):
        raise ValueError("predeclared Wind field slice z index is outside the native run")
    plane_size = int(run.nx) * int(run.ny)
    start = int(z_index) * plane_size
    flat = np.arange(start, start + plane_size, dtype=np.int64)
    coordinates = np.column_stack(
        (
            np.asarray(run.x_m, dtype=np.float64)[np.arange(plane_size) % int(run.nx)] / diameter_m,
            np.asarray(run.y_m, dtype=np.float64)[np.arange(plane_size) // int(run.nx)] / diameter_m,
            np.asarray(run.z_m, dtype=np.float64)[np.full(plane_size, int(z_index))] / diameter_m,
        )
    ).astype(np.float32)
    reference = np.asarray(run.U[flat], dtype=np.float32).copy()
    valid = np.isfinite(reference).all(axis=-1) & np.isfinite(coordinates).all(axis=-1)
    return {
        "flat_indices": flat,
        "coordinates_D": coordinates,
        "reference_velocity_mps": reference,
        "native_valid_mask": valid.astype(bool),
        "z_index": int(z_index),
        "z_D": float(np.asarray(run.z_m, dtype=np.float64)[int(z_index)] / diameter_m),
        "shape_yx": [int(run.ny), int(run.nx)],
    }


def _native_executor_comparison(
    model: WindFarmForwardModel,
    encoded: Any,
    sparse_plan: MechanismPlan,
    batch: Any,
    dense_prediction: torch.Tensor,
    dense_query_execution_ledger: Mapping[str, int],
    dense_prediction_seconds: float,
    *,
    device: torch.device,
    final_deadline_utc: datetime,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "comparison": "same fixed encoded case, plan, and Q2048 dev panel; prepare plus decode wall time with CUDA synchronization",
        "dense_masked": {"supported": True},
    }
    _check_final_deadline("dense-masked executor comparison", final_deadline_utc)
    dense_state = model.core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(sparse_plan,))
    result["dense_masked"].update(
        {
            "preparation_ledger": training._ledger(dense_state),
            "query_execution_ledger": dict(dense_query_execution_ledger),
            "inference_seconds_prepare_decode_total": float(dense_prediction_seconds),
            "prediction_parity_reference": "dense-masked prediction returned by candidate action evaluation",
        }
    )
    model.core.backend.set_cover_executor("packed")
    try:
        _check_final_deadline("packed executor comparison", final_deadline_utc)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        packed_state = model.core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(sparse_plan,))
        packed_output = model.core.decode_queries(
            packed_state,
            batch.query_xy,
            query_features=batch.query_features,
            receiver_chunk_size=512,
            return_interaction_aux=True,
        )
        packed_prediction = packed_output["pred_field"]
        packed_query_aux = packed_output.get("_interaction_aux", {})
        packed_query_ledger = {
            str(key): int(value.detach().cpu().item() if torch.is_tensor(value) else value)
            for key, value in packed_query_aux.items()
            if str(key).startswith(("cover_qm_", "cover_qe_", "cover_executor_"))
            and (not torch.is_tensor(value) or value.numel() == 1)
        }
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        result["packed"] = {
            "supported": True,
            "preparation_ledger": training._ledger(packed_state),
            "query_execution_ledger": packed_query_ledger,
            "inference_seconds_prepare_decode_total": float(elapsed),
            "prediction_max_abs_difference_from_dense": float(
                (packed_prediction - dense_prediction).abs().max().detach().cpu()
            ),
            "prediction_allclose_dense": bool(
                torch.allclose(packed_prediction, dense_prediction, atol=2e-5, rtol=2e-5)
            ),
        }
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            raise
        result["packed"] = {
            "supported": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "limitation": "existing packed executor did not accept this compiled direct-MM plan; no executor refactor was attempted",
        }
    finally:
        model.core.backend.set_cover_executor("dense_masked")
    return result


@torch.no_grad()
def run_evaluation(
    *,
    output: Path = training.DEFAULT_OUTPUT,
    evaluation_dir: Path = OUTPUT_DIR,
    device_name: str = "cuda:1",
    sampling_stop_utc: datetime = EVALUATION_STOP_UTC,
    final_deadline_utc: datetime = EVALUATION_FINAL_UTC,
) -> dict[str, Any]:
    if device_name != "cuda:1" or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Wind native evaluation requires the authorized explicit cuda:1")
    device = torch.device(device_name)
    device_properties = torch.cuda.get_device_properties(device)
    device_uuid = str(getattr(device_properties, "uuid", "unavailable"))
    normalized_device_uuid = device_uuid.removeprefix("GPU-")
    normalized_expected_uuid = EXPECTED_DEVICE_UUID.removeprefix("GPU-")
    if normalized_device_uuid != normalized_expected_uuid:
        raise RuntimeError(f"cuda:1 resolved to {device_uuid}, expected {EXPECTED_DEVICE_UUID}")
    lock_path = output / "cohort_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"metadata-only cohort lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    completed: dict[str, Any] = {}
    for arm in ("w_dir", "w_summary"):
        latest = load_trusted_checkpoint(output / arm / "latest.pt", map_location="cpu")
        if int(latest.get("scorer_updates", 0)) < 300 or int(latest.get("coadapt_updates", 0)) < 700:
            raise RuntimeError(f"{arm} has not completed the authorized 300+700 actual-update target")
        completed[arm] = latest

    compact_path = training.resolve_path("project://Case_WindFarm/Dataset/links/wind_farm/family_tensor.npz")
    volume_path = training.resolve_path("project://Case_WindFarm/Dataset/links/wind_farm/family_volume")
    compact = _compact_metadata(compact_path)
    if sorted(set(training.COMPACT_GEOMETRY_KEYS) - set(compact)):
        raise ValueError("Wind compact metadata lacks required geometry keys")
    view = WindFarmNativeView(volume_path, compact_metadata=compact)
    source_payload = load_trusted_checkpoint(Path(lock["base_model_checkpoint"]), map_location="cpu")
    normalizer = VelocityNormalizer.from_dict(dict(source_payload["normalization"]))
    role_cache = NativeRoleCatalogueCache(max_cached_bytes=3 * 1024**3)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    native_sample_path = evaluation_dir / "native_samples.pt"
    if native_sample_path.is_file():
        saved_sample_payload = load_trusted_checkpoint(native_sample_path, map_location="cpu")
        if not isinstance(saved_sample_payload, Mapping):
            raise ValueError("persisted Wind native sample artifact must be a mapping")
        sample_payloads: dict[str, Any] = {str(key): dict(value) for key, value in saved_sample_payload.items()}
    else:
        sample_payloads = {}

    first_train = _layout_from_record(lock["training_layouts"][0])
    first_row = _row_for_direction(view, first_train)
    first_case = view.run(first_row)
    first_sample_key = f"train_fit_layout{first_train.layout_index}_row{first_row}_q512_fit"
    first_sample_payload = sample_payloads.get(first_sample_key)
    if first_sample_payload is not None:
        material_sample = _restore_native_role_sample(first_case, first_sample_payload, role_cache)
    else:
        _check_deadline("Q512 train action-label sampling", sampling_stop_utc)
        material_sample = sample_native_role_queries(
            first_case,
            _native_rng(FIT_QUERY_SEED, first_train.layout_index, first_row, 701),
            training.ROLE_QUERY_COUNTS,
            catalogue_cache=role_cache,
        )
        sample_payloads[first_sample_key] = {
            **_sample_metadata(material_sample),
            "panel_metadata": {
                "support_kind": "native full support via maintained role CDF",
                "role_support_volume_m3": dict(material_sample.role_support_volume_m3),
            },
            "partition": "train_fit",
            "layout_index": int(first_train.layout_index),
            "row_index": int(first_row),
            "wind_direction_deg": float(view.metadata["wd_deg"][first_row]),
        }
        _save_torch(native_sample_path, sample_payloads)
    materialization_batch = _batch_from_sample(first_case, material_sample, normalizer, device)

    teacher, _teacher_payload = load_checkpoint(
        Path(lock["source_anchor_teacher_checkpoint"]),
        device=device,
        materialization_batch=materialization_batch,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.core.backend.set_cover_mode("external")
    teacher.core.backend.set_cover_executor("dense_masked")
    g_state = load_trusted_checkpoint(Path(lock["source_checkpoint"]), map_location="cpu")
    initial_g, _ = _materialize_model(
        g_state["model_state_dict"],
        Path(lock["base_model_checkpoint"]),
        materialization_batch,
        module_feature_dim=int(materialization_batch.module_features.shape[-1]),
        environment_feature_dim=int(materialization_batch.env_features.shape[-1]),
        device=device,
    )

    checkpoints = {
        "scorer_u300": {arm: output / arm / "scorer_update_0300.pt" for arm in ("w_dir", "w_summary")},
        "final_u1000": {arm: output / arm / "latest.pt" for arm in ("w_dir", "w_summary")},
    }
    arm_models: dict[tuple[str, str], tuple[WindFarmForwardModel, DirectionalPacketOrganizer, dict[str, Any]]] = {}
    for checkpoint_label, paths in checkpoints.items():
        for arm, checkpoint_path in paths.items():
            arm_models[(checkpoint_label, arm)] = _load_arm_model(
                arm,
                checkpoint_path,
                lock,
                materialization_batch,
                device=device,
            )

    rows: list[dict[str, Any]] = []
    prediction_payloads: dict[str, Any] = {}
    field_payloads: dict[str, Any] = {}
    packed_comparison: dict[str, Any] | None = None
    field_slices_by_row = {int(item["row_index"]): item for item in lock["predeclared_field_slices"]}
    evaluation_cases: list[tuple[str, TrainingLayout, int, Mapping[str, Any]]] = []
    for record in lock["training_layouts"]:
        layout = _layout_from_record(record)
        evaluation_cases.append(("train_fit", layout, _row_for_direction(view, layout), {}))
    for record in lock["development_layouts"]:
        layout = _layout_from_record(record)
        row = _row_for_direction(view, layout)
        evaluation_cases.append(("development", layout, row, field_slices_by_row.get(row, {})))

    sampled_cases: list[
        tuple[str, TrainingLayout, int, Mapping[str, Any], Any, str, NativeRoleSample, dict[str, Any]]
    ] = []

    def register_sample(
        partition: str,
        layout: TrainingLayout,
        row: int,
        field_selection: Mapping[str, Any],
        case: Any,
        panel_label: str,
        sample: NativeRoleSample,
        panel_meta: dict[str, Any],
    ) -> None:
        sample_key = f"{partition}_layout{layout.layout_index}_row{row}_{panel_label}"
        sample_payloads[sample_key] = {
            **_sample_metadata(sample),
            "panel_metadata": panel_meta,
            "partition": partition,
            "layout_index": int(layout.layout_index),
            "row_index": int(row),
            "wind_direction_deg": float(view.metadata["wd_deg"][row]),
        }
        _save_torch(evaluation_dir / "native_samples.pt", sample_payloads)
        sampled_cases.append((partition, layout, row, field_selection, case, panel_label, sample, panel_meta))

    def load_saved_sample(
        partition: str,
        layout: TrainingLayout,
        row: int,
        case: Any,
        panel_label: str,
    ) -> tuple[NativeRoleSample, dict[str, Any]] | None:
        key = f"{partition}_layout{layout.layout_index}_row{row}_{panel_label}"
        payload = sample_payloads.get(key)
        if payload is None:
            return None
        if (
            payload.get("partition") != partition
            or int(payload.get("layout_index", -1)) != int(layout.layout_index)
            or int(payload.get("row_index", -1)) != int(row)
        ):
            raise ValueError(f"persisted Wind sample metadata does not match {key}")
        return (
            _restore_native_role_sample(case, payload, role_cache),
            dict(payload.get("panel_metadata", {})),
        )

    for partition, layout, row, field_selection in evaluation_cases:
        case = view.run(row)
        if partition == "train_fit":
            query_panel = "q512_fit"
            saved = load_saved_sample(partition, layout, row, case, query_panel)
            if saved is None:
                _check_deadline("Q512 train action-label sampling", sampling_stop_utc)
                sample = sample_native_role_queries(
                    case,
                    _native_rng(FIT_QUERY_SEED, layout.layout_index, row, 701),
                    training.ROLE_QUERY_COUNTS,
                    catalogue_cache=role_cache,
                )
                sample_support: dict[str, Any] = {
                    "support_kind": "native full support via maintained role CDF",
                    "role_support_volume_m3": dict(sample.role_support_volume_m3),
                }
            else:
                sample, sample_support = saved
            register_sample(
                partition,
                layout,
                row,
                field_selection,
                case,
                query_panel,
                sample,
                sample_support,
            )
        else:
            panel_one_label = "q2048_panel1_full_native_support"
            saved_panel_one = load_saved_sample(partition, layout, row, case, panel_one_label)
            if saved_panel_one is None:
                _check_deadline("Q2048 development panel sampling", sampling_stop_utc)
                panel_one = sample_native_role_queries(
                    case,
                    _native_rng(DEV_QUERY_SEED, layout.layout_index, row, 801),
                    DEFAULT_AUDIT_QUERY_COUNTS,
                    catalogue_cache=role_cache,
                )
                panel_one_meta = {
                    "support_kind": "maintained native role CDF",
                    "role_support_volume_m3": dict(panel_one.role_support_volume_m3),
                }
            else:
                panel_one, panel_one_meta = saved_panel_one
            register_sample(
                partition,
                layout,
                row,
                field_selection,
                case,
                panel_one_label,
                panel_one,
                panel_one_meta,
            )
            panel_two_label = "q2048_panel2_disjoint_conditional_support"
            saved_panel_two = load_saved_sample(partition, layout, row, case, panel_two_label)
            if saved_panel_two is None:
                _check_deadline("Q2048 development panel-two rejection sampling", sampling_stop_utc)
                panel_two, disjoint_meta = _disjoint_panel(
                    case,
                    _native_rng(DEV_QUERY_SEED, layout.layout_index, row, 802),
                    DEFAULT_AUDIT_QUERY_COUNTS,
                    catalogue_cache=role_cache,
                    excluded_flat_indices=panel_one.flat_indices,
                    sampling_stop_utc=sampling_stop_utc,
                )
            else:
                panel_two, disjoint_meta = saved_panel_two
            intersection = int(np.intersect1d(panel_one.flat_indices, panel_two.flat_indices).size)
            if intersection != 0:
                raise ValueError(f"persisted Wind development panels overlap at {intersection} native cells")
            disjoint_meta["cross_panel_unique_flat_index_intersection_count"] = intersection
            register_sample(
                partition,
                layout,
                row,
                field_selection,
                case,
                panel_two_label,
                panel_two,
                disjoint_meta,
            )

    fit_teacher_records: list[dict[str, Any]] = []
    if sampled_cases:
        for partition, layout, row, field_selection, case, panel_label, sample, panel_meta in sampled_cases:
            _check_final_deadline("native action inference", final_deadline_utc)
            sample_key = f"{partition}_layout{layout.layout_index}_row{row}_{panel_label}"
            batch = _batch_from_sample(case, sample, normalizer, device)
            ref_teacher_encoded = teacher.core.encode_case(batch)
            teacher_tree = teacher.core.backend.build_case_trees(ref_teacher_encoded)[0]
            teacher_plan = MechanismPlan.full_access(
                teacher_tree, ref_teacher_encoded.module_present[0], int(ref_teacher_encoded.env_coords.shape[1])
            )
            teacher_pred, _, _ = _predict_plan(
                teacher,
                ref_teacher_encoded,
                teacher_plan,
                batch,
                device=device,
                final_deadline_utc=final_deadline_utc,
            )
            initial_g_encoded = initial_g.core.encode_case(batch)
            initial_g_tree = initial_g.core.backend.build_case_trees(initial_g_encoded)[0]
            initial_g_plan = MechanismPlan.full_access(
                initial_g_tree, initial_g_encoded.module_present[0], int(initial_g_encoded.env_coords.shape[1])
            )
            initial_g_pred, _, _ = _predict_plan(
                initial_g,
                initial_g_encoded,
                initial_g_plan,
                batch,
                device=device,
                final_deadline_utc=final_deadline_utc,
            )
            teacher_physical = _inverse_prediction(teacher, teacher_pred)
            target_physical = _inverse_prediction(teacher, batch.target_field)
            teacher_rmse = _role_rmse(teacher_physical, target_physical, sample.role_slices)
            if partition == "train_fit":
                fit_teacher_records.append(
                    {"family_key": str(layout.layout_index), "wfull_role_rmse_mps": teacher_rmse}
                )

            catalogue = role_cache.get(case)
            receiver_roles = _role_features(catalogue)
            for checkpoint_label in checkpoints:
                for arm in ("w_dir", "w_summary"):
                    model, organizer, state = arm_models[(checkpoint_label, arm)]
                    model_encoded = model.core.encode_case(batch)
                    model_trees = model.core.backend.build_case_trees(model_encoded)
                    full_plan = MechanismPlan.full_access(
                        model_trees[0], model_encoded.module_present[0], int(model_encoded.env_coords.shape[1])
                    )
                    full_pred, full_ledger, full_seconds = _predict_plan(
                        model,
                        model_encoded,
                        full_plan,
                        batch,
                        device=device,
                        final_deadline_utc=final_deadline_utc,
                    )
                    case_context = {
                        "family_key": str(layout.layout_index),
                        "case_key": f"{arm}_{checkpoint_label}_layout{layout.layout_index}_row{row}_{panel_label}",
                        "partition": partition,
                        "layout_index": int(layout.layout_index),
                        "row_index": int(row),
                        "wind_direction_deg": float(view.metadata["wd_deg"][row]),
                        "model_arm": arm,
                        "checkpoint_label": checkpoint_label,
                        "checkpoint_path": str(checkpoints[checkpoint_label][arm]),
                        "scorer_updates": int(state.get("scorer_updates", 0)),
                        "coadapt_updates": int(state.get("coadapt_updates", 0)),
                        "restricted_optimizer_updates": int(state.get("restricted_optimizer_updates", 0)),
                        "residual_ramp": float(organizer.residual_ramp),
                        "query_panel": panel_label,
                        "query_count": int(batch.query_xy.shape[1]),
                        "query_counts_by_role": dict(sample.role_sample_counts),
                        "numerical_floor_role_mps": [1.0e-8] * len(training.ROLE_NAMES),
                    }
                    for support_source in ("trained_support", "geometry_prior"):
                        scores = _make_scores(
                            organizer,
                            model_encoded,
                            model_trees,
                            geometry_prior=(support_source == "geometry_prior"),
                            restored_update=int(state.get("restricted_optimizer_updates", 0)),
                        )
                        action_rows, detail = _action_rows_for_case(
                            model=model,
                            organizer=organizer,
                            encoded=model_encoded,
                            base_tree=model_trees[0],
                            scores=scores,
                            batch=batch,
                            sample=sample,
                            case=case,
                            sample_key=sample_key,
                            prediction_payloads=prediction_payloads,
                            full_prediction_std=full_pred,
                            full_prediction_ledger=full_ledger,
                            full_prediction_seconds=full_seconds,
                            teacher_prediction_std=teacher_pred,
                            initial_g_prediction_std=initial_g_pred,
                            final_deadline_utc=final_deadline_utc,
                            role_features=receiver_roles,
                            case_context=case_context,
                            support_source=support_source,
                            device=device,
                        )
                        for action_row in action_rows:
                            action_row["frontier_summary"] = detail
                            rows.append(action_row)

                        # Make each case/checkpoint/arm durable before the
                        # next native action sweep can consume more GPU time.
                        checkpoint_fragment = f"{sample_key}_{checkpoint_label}_{arm}_{support_source}"
                        partial_dir = evaluation_dir / "partial_action_rows"
                        _save_torch(partial_dir / f"{checkpoint_fragment}.pt", action_rows)
                        _save_torch(partial_dir / "prediction_arrays.pt", prediction_payloads)

                        # Save full field-grid evidence for the two predeclared
                        # hub-height slices, using the same native query grid for
                        # both arms, checkpoint stages, and support controls.
                        if field_selection and panel_label == "q2048_panel1_full_native_support":
                            field_info = _field_slice_payload(
                                case, int(field_selection["z_index"]), float(view.diameter_m)
                            )
                            field_coords = field_info["coordinates_D"]
                            field_batch = case_batch(case, field_coords, include_receiver_anchors=True).to(device)
                            field_encoded = model.core.encode_case(field_batch)
                            field_trees = model.core.backend.build_case_trees(field_encoded)
                            field_scores = _make_scores(
                                organizer,
                                field_encoded,
                                field_trees,
                                geometry_prior=(support_source == "geometry_prior"),
                                restored_update=int(state.get("restricted_optimizer_updates", 0)),
                            )
                            field_source_measures = _native_module_measures(field_encoded, field_scores[0])
                            for cut in enumerate_frontier_cuts(field_scores[0].receiver_tree, max_depth=2):
                                _local, compiled, action_detail = training._action_descriptor(
                                    organizer, field_scores, field_encoded, field_trees[0], tuple(cut)
                                )
                                field_prediction, field_ledger, field_seconds = _predict_plan(
                                    model,
                                    field_encoded,
                                    compiled,
                                    field_batch,
                                    device=device,
                                    final_deadline_utc=final_deadline_utc,
                                )
                                field_key = (
                                    f"layout{layout.layout_index}_row{row}_{checkpoint_label}_{arm}_"
                                    f"{support_source}_cut{'_'.join(str(value) for value in cut)}"
                                )
                                direct = compiled.direct_pair_access_for("MM")
                                active_field = field_encoded.module_present[0] > 0.5
                                field_permission = (
                                    torch.zeros_like(field_encoded.module_present[0][:, None])
                                    .expand(-1, field_encoded.module_present.shape[1])
                                    .clone()
                                    if direct is None
                                    else direct.weights > 0
                                )
                                field_permission &= active_field[:, None] & active_field[None, :]
                                field_permission.fill_diagonal_(False)
                                field_payloads[field_key] = {
                                    "selection": dict(field_selection),
                                    **field_info,
                                    "direction_deg": float(case.wind_direction_deg),
                                    "model_arm": arm,
                                    "checkpoint_label": checkpoint_label,
                                    "support_source": support_source,
                                    "cut_nodes": [int(value) for value in cut],
                                    "mm_local_k": int(action_detail["realized_nonredundant_MM_packet_count"]),
                                    "action_work_pairs": int(action_detail["MM_pair_permission_unique_pairs"]),
                                    "full_work_pairs": int(action_detail["MM_eligible_nonself_pairs"]),
                                    "prediction_velocity_mps": _inverse_prediction(model, field_prediction)
                                    .detach()
                                    .cpu()
                                    .squeeze(0)
                                    .numpy(),
                                    "native_hard_preparation_ledger": field_ledger["preparation"],
                                    "native_query_execution_ledger": field_ledger["query_execution"],
                                    "inference_seconds_prepare_decode_total": field_seconds,
                                    "receiver_coordinates": field_encoded.module_centers[0].detach().cpu(),
                                    "source_coordinates": field_encoded.module_centers[0].detach().cpu(),
                                    "receiver_ids": torch.arange(field_encoded.module_centers.shape[1]),
                                    "source_ids": torch.arange(field_encoded.module_centers.shape[1]),
                                    "source_measures": field_source_measures.detach().cpu(),
                                    "valid_mm_permission_matrix": field_permission.detach().cpu(),
                                    "native_mm_permission_weights": (
                                        (direct.weights * field_permission.to(direct.weights.dtype)).detach().cpu()
                                        if direct is not None
                                        else field_permission.to(torch.float32).cpu()
                                    ),
                                }
                            if support_source == "trained_support":
                                field_full_plan = MechanismPlan.full_access(
                                    field_trees[0],
                                    field_encoded.module_present[0],
                                    int(field_encoded.env_coords.shape[1]),
                                )
                                field_full_pred, field_full_ledger, field_full_seconds = _predict_plan(
                                    model,
                                    field_encoded,
                                    field_full_plan,
                                    field_batch,
                                    device=device,
                                    final_deadline_utc=final_deadline_utc,
                                )
                                field_key = f"layout{layout.layout_index}_row{row}_{checkpoint_label}_{arm}_full_access"
                                field_full_permissions = (active_field[:, None] & active_field[None, :]).clone()
                                field_full_permissions.fill_diagonal_(False)
                                field_payloads[field_key] = {
                                    "selection": dict(field_selection),
                                    **field_info,
                                    "direction_deg": float(case.wind_direction_deg),
                                    "model_arm": arm,
                                    "checkpoint_label": checkpoint_label,
                                    "support_source": "full_access",
                                    "cut_nodes": [],
                                    "mm_local_k": None,
                                    "action_work_pairs": int(field_encoded.module_present[0].sum().item())
                                    * max(0, int(field_encoded.module_present[0].sum().item()) - 1),
                                    "full_work_pairs": int(field_encoded.module_present[0].sum().item())
                                    * max(0, int(field_encoded.module_present[0].sum().item()) - 1),
                                    "prediction_velocity_mps": _inverse_prediction(model, field_full_pred)
                                    .detach()
                                    .cpu()
                                    .squeeze(0)
                                    .numpy(),
                                    "native_hard_preparation_ledger": field_full_ledger["preparation"],
                                    "native_query_execution_ledger": field_full_ledger["query_execution"],
                                    "inference_seconds_prepare_decode_total": field_full_seconds,
                                    "receiver_coordinates": field_encoded.module_centers[0].detach().cpu(),
                                    "source_coordinates": field_encoded.module_centers[0].detach().cpu(),
                                    "receiver_ids": torch.arange(field_encoded.module_centers.shape[1]),
                                    "source_ids": torch.arange(field_encoded.module_centers.shape[1]),
                                    "source_measures": field_source_measures.detach().cpu(),
                                    "valid_mm_permission_matrix": field_full_permissions.detach().cpu(),
                                    "native_mm_permission_weights": field_full_permissions.to(torch.float32).cpu(),
                                }
                                if checkpoint_label == "scorer_u300" and arm == "w_dir":
                                    for baseline_name, baseline_model in (
                                        ("retained_wfull", teacher),
                                        ("initial_g_u4910", initial_g),
                                    ):
                                        baseline_encoded = baseline_model.core.encode_case(field_batch)
                                        baseline_tree = baseline_model.core.backend.build_case_trees(baseline_encoded)[
                                            0
                                        ]
                                        baseline_plan = MechanismPlan.full_access(
                                            baseline_tree,
                                            baseline_encoded.module_present[0],
                                            int(baseline_encoded.env_coords.shape[1]),
                                        )
                                        baseline_prediction, baseline_ledger, baseline_seconds = _predict_plan(
                                            baseline_model,
                                            baseline_encoded,
                                            baseline_plan,
                                            field_batch,
                                            device=device,
                                            final_deadline_utc=final_deadline_utc,
                                        )
                                        baseline_key = (
                                            f"layout{layout.layout_index}_row{row}_{baseline_name}_full_access"
                                        )
                                        field_payloads[baseline_key] = {
                                            "selection": dict(field_selection),
                                            **field_info,
                                            "direction_deg": float(case.wind_direction_deg),
                                            "model_arm": baseline_name,
                                            "checkpoint_label": "retained_reference",
                                            "support_source": "full_access",
                                            "cut_nodes": [],
                                            "mm_local_k": None,
                                            "action_work_pairs": int(active_field.sum().item())
                                            * max(0, int(active_field.sum().item()) - 1),
                                            "full_work_pairs": int(active_field.sum().item())
                                            * max(0, int(active_field.sum().item()) - 1),
                                            "prediction_velocity_mps": _inverse_prediction(
                                                baseline_model, baseline_prediction
                                            )
                                            .detach()
                                            .cpu()
                                            .squeeze(0)
                                            .numpy(),
                                            "native_hard_preparation_ledger": baseline_ledger["preparation"],
                                            "native_query_execution_ledger": baseline_ledger["query_execution"],
                                            "inference_seconds_prepare_decode_total": baseline_seconds,
                                            "receiver_coordinates": baseline_encoded.module_centers[0].detach().cpu(),
                                            "source_coordinates": baseline_encoded.module_centers[0].detach().cpu(),
                                            "receiver_ids": torch.arange(baseline_encoded.module_centers.shape[1]),
                                            "source_ids": torch.arange(baseline_encoded.module_centers.shape[1]),
                                            "source_measures": torch.where(
                                                baseline_encoded.module_present[0] > 0.5,
                                                torch.ones_like(baseline_encoded.module_present[0]),
                                                torch.zeros_like(baseline_encoded.module_present[0]),
                                            )
                                            .detach()
                                            .cpu(),
                                            "valid_mm_permission_matrix": field_full_permissions.detach().cpu(),
                                            "native_mm_permission_weights": field_full_permissions.to(
                                                torch.float32
                                            ).cpu(),
                                        }
                    if field_payloads:
                        _save_torch(evaluation_dir / "partial_field_slices.pt", field_payloads)
                    if (
                        partition == "development"
                        and panel_label == "q2048_panel1_full_native_support"
                        and checkpoint_label == "final_u1000"
                        and arm == "w_dir"
                        and packed_comparison is None
                    ):
                        scores = _make_scores(
                            organizer,
                            model_encoded,
                            model_trees,
                            geometry_prior=False,
                            restored_update=int(state.get("restricted_optimizer_updates", 0)),
                        )
                        candidate_cuts = enumerate_frontier_cuts(scores[0].receiver_tree, max_depth=2)
                        for candidate_cut in candidate_cuts:
                            _local, packed_plan, _details = training._action_descriptor(
                                organizer, scores, model_encoded, model_trees[0], tuple(candidate_cut)
                            )
                            if _details["MM_pair_permission_unique_pairs"] < _details["MM_eligible_nonself_pairs"]:
                                dense_prediction, dense_ledger, dense_seconds = _predict_plan(
                                    model,
                                    model_encoded,
                                    packed_plan,
                                    batch,
                                    device=device,
                                    final_deadline_utc=final_deadline_utc,
                                )
                                packed_comparison = _native_executor_comparison(
                                    model,
                                    model_encoded,
                                    packed_plan,
                                    batch,
                                    dense_prediction,
                                    dense_ledger["query_execution"],
                                    dense_seconds,
                                    device=device,
                                    final_deadline_utc=final_deadline_utc,
                                )
                                packed_comparison.update(
                                    {
                                        "layout_index": int(layout.layout_index),
                                        "row_index": int(row),
                                        "action_cut_nodes": [int(value) for value in candidate_cut],
                                        "MM_local_K": int(_details["realized_nonredundant_MM_packet_count"]),
                                    }
                                )
                                break

    if not fit_teacher_records:
        raise RuntimeError("training-label evaluation did not produce retained W-full calibration rows")
    family_role_errors = np.stack(
        [
            np.asarray([record["wfull_role_rmse_mps"][role] for role in training.ROLE_NAMES], dtype=np.float64)
            for record in fit_teacher_records
        ]
    )
    role_floor = np.maximum(1.0e-8, 1.0e-6 * np.median(family_role_errors, axis=0))
    for row in rows:
        row["numerical_floor_role_mps"] = role_floor.tolist()
    _check_final_deadline("evaluation artifact finalization", final_deadline_utc)
    training_exposure = _training_exposure_audit(output)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "run_id": "directed_stable_20261002_wind",
        "device": str(device),
        "device_uuid": device_uuid,
        "query_policy": {
            "fit_labels": "one maintained Q512 native role panel per metadata-selected training layout",
            "development_panel_1": "maintained Q2048 native role sampler over full role support",
            "development_panel_2": "Q2048 native role CDF rejection sample conditioned on excluding all panel-one native flat indices",
            "development_panel_2_support_is_identical_to_panel_1": False,
            "role_order": list(training.ROLE_NAMES),
            "role_query_counts_q512": dict(training.ROLE_QUERY_COUNTS),
            "role_query_counts_q2048": dict(DEFAULT_AUDIT_QUERY_COUNTS),
            "cross_panel_intersection_counts": {
                key: value["panel_metadata"]["cross_panel_unique_flat_index_intersection_count"]
                for key, value in sample_payloads.items()
                if "panel2" in key
            },
        },
        "cohort_lock_path": str(lock_path),
        "retained_wfull_checkpoint": str(lock["source_anchor_teacher_checkpoint"]),
        "initial_g_checkpoint": str(lock["source_checkpoint"]),
        "numerical_floor_role_mps": role_floor.tolist(),
        "numerical_floor_rule": "max(1e-8, 1e-6 * median of retained W-full per-role RMSE across metadata-selected training layouts); same maintained Wind rule",
        "physical_allowance_role_mps": [0.0] * len(training.ROLE_NAMES),
        "risk_target_note": "candidate_role_rmse_mps and incumbent_role_rmse_mps are physical RMSE; same_weight_full and G-u4910 drift are reported separately from MM access error",
        "deadlines_utc": {
            "new_training_and_sampling_stop": sampling_stop_utc.isoformat(),
            "final_evaluation_delivery": final_deadline_utc.isoformat(),
        },
        "control_design": {
            "source_union_collapse": "same physical model weights; all valid receivers receive the sparse action's common selected-source union, with native self exclusion; pair work may differ and is reported",
            "direct_geometry_same_work": "each receiver keeps its exact parent out-degree and selects its nearest eligible physical sources, preserving total pair work and receiver coverage",
            "effective_pair_rewire_same_work": "deterministic cyclic eligible-pair rewire at exact sparse pair count; source and receiver degrees are not preserved; skipped only if support is empty or complete",
            "exact_mask_changes": "each control row contains added and removed native receiver/source ID pairs versus its trained sparse parent action",
        },
        "action_descriptor": {
            "function": "honf_forward_core.interface_fields.action_aware_frontier.describe_frontier_action",
            "typed_mechanism_order": ["MM"],
            "packet_rows_width": int(rows[0]["packet_rows"].shape[1]) if rows else None,
            "receiver_role_features_shape": list(rows[0]["receiver_role_features"].shape) if rows else None,
            "components": "MM-local node embedding; included/excluded MM source embedding, geometry, native measure, degree, and scorer summaries; appended source-measure-weighted signed 13-feature packet geometry",
            "source_ids_and_measures": "serialized in native padded-module order beside the descriptor and exact effective pair mask",
        },
        "checkpoint_states": {
            label: {
                arm: {
                    "path": str(checkpoints[label][arm]),
                    "scorer_updates": int(arm_models[(label, arm)][2].get("scorer_updates", 0)),
                    "coadapt_updates": int(arm_models[(label, arm)][2].get("coadapt_updates", 0)),
                    "restricted_optimizer_updates": int(
                        arm_models[(label, arm)][2].get("restricted_optimizer_updates", 0)
                    ),
                    "restored_residual_ramp": float(arm_models[(label, arm)][1].residual_ramp),
                }
                for arm in ("w_dir", "w_summary")
            }
            for label in checkpoints
        },
        "row_count": len(rows),
        "sample_count": len(sample_payloads),
        "field_slice_count": len(field_payloads),
        "training_exposure_audit": training_exposure,
        "permission_weight_audit": {
            "row_count_with_fractional_positive_mm_weights": sum(
                int(row["MM_fractional_positive_permission_count"] > 0) for row in rows
            ),
            "fractional_positive_mm_weight_entry_count": sum(
                int(row["MM_fractional_positive_permission_count"]) for row in rows
            ),
            "control_weight_policy": "degree-matched geometry preserves each receiver's positive permission-weight multiset; same-work rewire preserves the global permission-weight multiset and total mass; source-union collapse uses unit weights and reports changed work/mass",
        },
        "packed_executor_comparison": packed_comparison,
        "artifacts": {
            "action_rows": str(evaluation_dir / "action_rows.pt"),
            "prediction_arrays": str(evaluation_dir / "prediction_arrays.pt"),
            "native_samples": str(evaluation_dir / "native_samples.pt"),
            "predeclared_field_slices": str(evaluation_dir / "predeclared_field_slices.pt"),
        },
    }
    _save_torch(evaluation_dir / "action_rows.pt", rows)
    _save_torch(evaluation_dir / "prediction_arrays.pt", prediction_payloads)
    _save_torch(evaluation_dir / "native_samples.pt", sample_payloads)
    _save_torch(evaluation_dir / "predeclared_field_slices.pt", field_payloads)
    _json_write(evaluation_dir / "evaluation_manifest.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--sampling-stop-utc",
        type=_parse_utc_deadline,
        default=EVALUATION_STOP_UTC,
        help="UTC deadline after which the evaluator will not sample new native queries",
    )
    parser.add_argument(
        "--final-deadline-utc",
        type=_parse_utc_deadline,
        default=EVALUATION_FINAL_UTC,
        help="UTC deadline for completing already-sampled native inference and finalization",
    )
    args = parser.parse_args()
    result = run_evaluation(
        output=args.output,
        evaluation_dir=args.evaluation_dir,
        device_name=args.device,
        sampling_stop_utc=args.sampling_stop_utc,
        final_deadline_utc=args.final_deadline_utc,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
