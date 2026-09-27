"""Bounded Run 2103 native-cover oracle and input-only organizer workflow.

The short ``--preflight-only`` mode loads the exact checkpoint and exercises
the native all-access and input-organizer policy paths on one train row. The
full train-only panel mode is deliberately kept in this module so its probe,
oracle, training, and evaluation boundaries can be reviewed together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.adaptive_cover_oracle import (
    MeasuredCostCoefficients,
    OracleObservation,
    OracleProposal,
    RoleError,
    TeacherDistortionLimit,
    search_training_cover,
    teacher_preservation_gate,
)
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    compile_cover_pairs,
)
from honf_forward_core.interface_fields.input_cover_organizer import (
    InputOnlyCoverOrganizer,
)
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.paths import resolve_path
from torch import nn

from ..data import WindFarmNativeView, collate_windfarm
from ..geometry import (
    ANCHOR_ROLE_ENVIRONMENT,
    ANCHOR_ROLE_ROTOR_EDGE,
    ANCHOR_ROLE_ROTOR_HUB,
    ANCHOR_ROLE_ROTOR_INTERIOR,
    support_weights,
)
from ..normalization import VelocityNormalizer
from ..splits import make_group_split
from ..study_spatial import WeightedVelocityErrors, downstream_envelope, native_coordinates
from .evaluate_forward import load_checkpoint
from .native_cover_panel import (
    NativeProbeSet,
    make_disjoint_native_probes,
    select_training_layouts,
)
from .train_forward import _as_device_batch, _split_from_checkpoint


class _AllAccessPolicy(nn.Module):
    """Return root-only full access while preserving the current tree object."""

    def plan_cases(self, encoded: Any, prepared_state: Any, trees: Any) -> tuple[AdaptiveCoverPlan, ...]:
        del prepared_state
        return tuple(
            AdaptiveCoverPlan.full_access(tree, encoded.module_present[case], int(encoded.env_coords.shape[1]))
            for case, tree in enumerate(trees)
        )


class _FixedPlanPolicy(nn.Module):
    """Replay plans on the fresh geometry tree supplied by native preparation."""

    def __init__(self, plans: tuple[AdaptiveCoverPlan, ...]) -> None:
        super().__init__()
        self.plans = plans

    def plan_cases(self, encoded: Any, prepared_state: Any, trees: Any) -> tuple[AdaptiveCoverPlan, ...]:
        del encoded, prepared_state
        if len(trees) != len(self.plans):
            raise ValueError("fixed cover replay requires one saved plan per prepared case")
        return tuple(replace(plan, tree=tree) for plan, tree in zip(self.plans, trees, strict=True))


@dataclass
class _PanelCase:
    case: Any
    search: NativeProbeSet
    verification: NativeProbeSet
    search_batch: Any
    verification_batch: Any
    encoded: Any
    trees: tuple[CaseLocalReceiverTree, ...]
    full_plan: AdaptiveCoverPlan
    full_prepared: Any
    full_search_prediction: torch.Tensor
    full_search_standardized: torch.Tensor
    full_verification_prediction: torch.Tensor
    full_search_observation: OracleObservation
    full_prepare_ms: float
    full_search_decode_ms: float
    full_verification_decode_ms: float
    anchor_measure_provenance: dict[str, Any]
    oracle_plan: AdaptiveCoverPlan | None = None
    search_teacher_error: dict[str, Any] | None = None


TEACHER_DISTORTION_LIMIT = 0.10
ORACLE_OBSERVATION_CAP_PER_PASS = 48
MAX_ORACLE_PASSES_PER_CASE = 2
FROZEN_RUN_DIRECTORY = "Run_2103_20260913_135849_windfarm_dense_pairwise_b16_q8192"
COLD_WARM_ABS_TOLERANCE_MPS = 1.0e-5
COLD_WARM_REL_TOLERANCE = 1.0e-6
ANCHOR_ROLE_NAMES = {
    ANCHOR_ROLE_ENVIRONMENT: "environment",
    ANCHOR_ROLE_ROTOR_HUB: "rotor_hub",
    ANCHOR_ROLE_ROTOR_INTERIOR: "rotor_interior",
    ANCHOR_ROLE_ROTOR_EDGE: "rotor_edge",
}


def _probe_batch(case: Any, probe: Any, normalizer: VelocityNormalizer) -> dict[str, Any]:
    """Build the ordinary WindFarm model input plus aligned native evidence."""

    geometry = case.geometry_for_queries(probe.coordinates_D)
    sample: dict[str, Any] = {
        "module_centers": case.module_centers.copy(),
        "module_present": case.module_present.copy(),
        "module_features": case.module_features.copy(),
        "global_context": case.global_context.copy(),
        "env_coords": case.env_coords.copy(),
        "env_features": case.env_features.copy(),
        "env_weights": case.env_weights.copy(),
        "query_xy": geometry["query_xy"],
        "query_features": geometry["query_features"],
        "query_time": None,
        "target_field": normalizer.normalize(probe.target_mps),
        "case_name": case.case,
        "metadata": {
            "source_index": int(case.index),
            "case": case.case,
            "layout_index": int(case.layout_index),
            "wind_direction_deg": float(case.wind_direction_deg),
        },
    }
    if case.receiver_anchor_coords is not None:
        sample["receiver_anchor_coords"] = case.receiver_anchor_coords.copy()
        sample["receiver_anchor_weights"] = case.receiver_anchor_weights.copy()
        sample["receiver_anchor_roles"] = case.receiver_anchor_roles.copy()
    return collate_windfarm([sample])


def _anchor_mass_summary(weights: Any, roles: Any) -> dict[str, Any]:
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    role_values = np.asarray(roles, dtype=np.int64).reshape(-1)
    if weight_values.shape != role_values.shape:
        raise ValueError("receiver-anchor roles and weights must align")
    valid = weight_values > 0.0
    role_mass = {
        name: {
            "count": int(np.sum(valid & (role_values == role_code))),
            "raw_weight_sum": float(weight_values[valid & (role_values == role_code)].sum()),
        }
        for role_code, name in ANCHOR_ROLE_NAMES.items()
    }
    environment_total = role_mass["environment"]["raw_weight_sum"]
    module_total = sum(role_mass[name]["raw_weight_sum"] for name in (
        "rotor_hub", "rotor_interior", "rotor_edge"
    ))
    return {
        "anchor_count": int(valid.sum()),
        "role_mass": role_mass,
        "environment_total_raw_mass": environment_total,
        "module_rotor_total_raw_mass": float(module_total),
        "module_rotor_to_environment_mass_ratio": (
            float(module_total / environment_total) if environment_total > 0.0 else None
        ),
    }


def _role_balanced_anchor_batch(batch: Any) -> tuple[Any, dict[str, Any]]:
    """Balance only receiver-tree index mass; preserve source quadrature exactly."""

    weights = batch.receiver_anchor_weights
    roles = batch.receiver_anchor_roles
    if weights is None or roles is None:
        raise ValueError("role-balanced tree construction requires explicit receiver anchors")
    if weights.ndim != 2 or roles.shape != weights.shape or int(weights.shape[0]) != 1:
        raise ValueError("role-balanced tree construction requires one aligned receiver-anchor case")
    environment = roles[0] == ANCHOR_ROLE_ENVIRONMENT
    module_rotor = (roles[0] != ANCHOR_ROLE_ENVIRONMENT) & (weights[0] > 0)
    environment_mass = weights[0, environment].sum()
    module_mass = weights[0, module_rotor].sum()
    if float(environment_mass.detach().cpu()) <= 0.0 or float(module_mass.detach().cpu()) <= 0.0:
        raise ValueError("role-balanced tree construction requires positive environment and module masses")
    balanced = torch.zeros_like(weights)
    balanced[0, environment] = weights[0, environment] / environment_mass
    balanced[0, module_rotor] = weights[0, module_rotor] / module_mass
    raw_source_weights = batch.env_weights
    balanced_batch = replace(batch, receiver_anchor_weights=balanced)
    if raw_source_weights is not None and not torch.equal(balanced_batch.env_weights, raw_source_weights):
        raise RuntimeError("role-balanced index variant changed native environment source quadrature")
    provenance = {
        "variant": "role_balanced_tree_index",
        "raw_mass": _anchor_mass_summary(weights[0].detach().cpu().numpy(), roles[0].detach().cpu().numpy()),
        "balanced_mass": _anchor_mass_summary(balanced[0].detach().cpu().numpy(), roles[0].detach().cpu().numpy()),
        "rule": "normalize role-0 environment receiver-anchor mass to 1 and all nonzero rotor/module receiver-anchor mass jointly to 1",
        "changed_fields": ["receiver_anchor_weights"],
        "preserved_fields": ["receiver_anchor_coords", "receiver_anchor_roles", "env_weights", "env_coords", "env_features"],
        "source_quadrature_changed": False,
    }
    return balanced_batch, provenance


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


def _assert_tensor_fields_on_device(batch: Any, device: torch.device) -> None:
    """Catch host-side case inputs before they reach a CUDA model."""

    fields = vars(batch) if hasattr(batch, "__dict__") else {}
    misplaced = [
        name for name, value in fields.items()
        if torch.is_tensor(value) and value.device != device
    ]
    if misplaced:
        raise RuntimeError(f"WindFarm case batch tensor fields are not on {device}: {misplaced}")


def _model_batch(raw: Any, device: torch.device) -> Any:
    batch = _as_device_batch(raw, device)
    _assert_tensor_fields_on_device(batch, device)
    return batch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(callable_: Any, device: torch.device) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter()
    value = callable_()
    _sync(device)
    return value, (time.perf_counter() - started) * 1000.0


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_state(prepared: Any) -> dict[str, torch.Tensor]:
    encoded = prepared.encoded
    return {
        "module_states": encoded.module_tokens,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
    }


def _native_cover_proposals(
    plan: AdaptiveCoverPlan,
    encoded: Any,
    *,
    train_evidence_id: str,
    environment_blocks: int = 16,
    axis: int = 0,
    include_module_prunes: bool = True,
    include_child_environment_blocks: bool = False,
) -> tuple[OracleProposal, ...]:
    """Generate bounded geometry-block/source probes and split-plus-prune alternatives."""

    if axis not in (0, 1):
        raise ValueError("environment block axis must be x=0 or y=1")

    proposals: list[OracleProposal] = []
    if include_module_prunes:
        valid_modules = torch.nonzero(encoded.module_present[0] > 0.5, as_tuple=False).flatten().tolist()
        for source in valid_modules:
            membership = plan.module_membership.clone()
            membership[0, source] = 0.0
            candidate = replace(plan, module_membership=membership)
            proposals.append(OracleProposal("module_prune", 0, "module", int(source), candidate, train_evidence_id))

    source_coordinates = encoded.env_coords[0, :, axis].detach().cpu().numpy()
    block_ids = np.array_split(np.argsort(source_coordinates, kind="stable"), environment_blocks)
    for block_index, indices in enumerate(block_ids):
        if indices.size == 0:
            continue
        membership = plan.environment_membership.clone()
        membership[0, torch.as_tensor(indices, device=membership.device)] = 0.0
        candidate = replace(plan, environment_membership=membership)
        proposals.append(
            OracleProposal("environment_block_prune", 0, "environment", block_index, candidate, train_evidence_id)
        )

    tree = plan.tree
    if len(tree.nodes) > 2 and not tree.nodes[0].is_leaf:
        left_id, right_id = tree.nodes[0].left, tree.nodes[0].right
        assert left_id is not None and right_id is not None
        universe = tree.universe
        proposal_device = encoded.module_centers.device
        scale = universe.coordinate_scale.to(device=proposal_device, dtype=encoded.module_centers.dtype)
        left_ids = torch.as_tensor(tree.nodes[left_id].anchor_indices, device=scale.device, dtype=torch.long)
        right_ids = torch.as_tensor(tree.nodes[right_id].anchor_indices, device=scale.device, dtype=torch.long)
        anchor_coordinates = universe.coordinates.to(device=proposal_device, dtype=scale.dtype)
        left_center = anchor_coordinates.index_select(0, left_ids).mean(dim=0) / scale
        right_center = anchor_coordinates.index_select(0, right_ids).mean(dim=0) / scale
        modules = encoded.module_centers[0].to(device=proposal_device, dtype=scale.dtype) / scale
        environments = encoded.env_coords[0].to(device=proposal_device, dtype=scale.dtype) / scale
        module_distances = torch.stack(
            (torch.linalg.vector_norm(modules - left_center, dim=-1),
             torch.linalg.vector_norm(modules - right_center, dim=-1)), dim=0
        )
        environment_distances = torch.stack(
            (torch.linalg.vector_norm(environments - left_center, dim=-1),
             torch.linalg.vector_norm(environments - right_center, dim=-1)), dim=0
        )
        module_left = module_distances[0] <= module_distances[1]
        module_left &= encoded.module_present[0] > 0.5
        environment_left = environment_distances[0] <= environment_distances[1]

        nearest_child_plan: AdaptiveCoverPlan | None = None
        for alternative in ("nearest_child", "boundary_overlap"):
            gates = plan.split_gates.new_zeros(plan.split_gates.shape)
            gates[0] = 1.0
            module_membership = plan.module_membership.new_zeros(plan.module_membership.shape)
            environment_membership = plan.environment_membership.new_zeros(plan.environment_membership.shape)
            module_membership[left_id] = module_left.to(module_membership.dtype)
            module_membership[right_id] = (
                (encoded.module_present[0] > 0.5) & ~module_left
            ).to(module_membership.dtype)
            environment_membership[left_id] = environment_left.to(environment_membership.dtype)
            environment_membership[right_id] = (~environment_left).to(environment_membership.dtype)
            if alternative == "boundary_overlap":
                module_overlap = (module_distances[0] - module_distances[1]).abs() <= 0.25
                module_overlap &= encoded.module_present[0] > 0.5
                environment_overlap = (environment_distances[0] - environment_distances[1]).abs() <= 0.25
                module_membership[left_id, module_overlap] = 1.0
                module_membership[right_id, module_overlap] = 1.0
                environment_membership[left_id, environment_overlap] = 1.0
                environment_membership[right_id, environment_overlap] = 1.0
            candidate = AdaptiveCoverPlan(tree, gates, module_membership, environment_membership)
            if alternative == "nearest_child":
                nearest_child_plan = candidate
            proposals.append(
                OracleProposal(f"split_child_prune_{alternative}", 0, None, None, candidate, train_evidence_id)
            )
        if include_child_environment_blocks and nearest_child_plan is not None:
            for block_index, indices in enumerate(block_ids):
                membership = nearest_child_plan.environment_membership.clone()
                for child_id in (left_id, right_id):
                    membership[child_id, torch.as_tensor(indices, device=membership.device)] = 0.0
                if torch.equal(membership, nearest_child_plan.environment_membership):
                    continue
                candidate = replace(nearest_child_plan, environment_membership=membership)
                proposals.append(
                    OracleProposal(
                        "split_child_environment_block_prune",
                        0,
                        "environment",
                        block_index,
                        candidate,
                        train_evidence_id,
                    )
                )
    if len(proposals) > ORACLE_OBSERVATION_CAP_PER_PASS:
        raise RuntimeError(
            f"native proposal generator exceeded {ORACLE_OBSERVATION_CAP_PER_PASS} observations: {len(proposals)}"
        )
    return tuple(proposals)


def _probe_observation(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    probe: NativeProbeSet,
    normalizer: VelocityNormalizer,
    *,
    teacher_checkpoint_id: str,
    training_evidence_id: str,
) -> tuple[OracleObservation, dict[str, Any]]:
    pred = prediction.detach().reshape(-1, 3)
    teacher_values = teacher.detach().reshape(-1, 3)
    target = torch.as_tensor(probe.target_mps, device=pred.device, dtype=pred.dtype)
    channel_scale_mps = pred.new_tensor(normalizer.safe_std * normalizer.u_ref_mps).clamp_min(1.0e-6)
    role_errors: dict[str, RoleError] = {}
    diagnostics: dict[str, Any] = {}
    for name, mask_value in probe.roles.items():
        mask = torch.as_tensor(mask_value, device=pred.device, dtype=torch.bool)
        if not bool(mask.any()):
            continue
        teacher_delta = pred[mask] - teacher_values[mask]
        reference_delta = pred[mask] - target[mask]
        normalized = torch.sqrt(torch.mean((teacher_delta / channel_scale_mps) ** 2))
        reference_rmse = torch.sqrt(torch.mean(reference_delta**2))
        # Teacher-preservation evidence never asserts the separate physical
        # sufficiency state. Exact native reference values are reported below.
        role_errors[name] = RoleError(float(reference_rmse.cpu()), float(normalized.cpu()), False)
        diagnostics[name] = {
            "probe_count": int(mask.sum().cpu()),
            "teacher_normalized_vector_rmse": float(normalized.cpu()),
            "reference_vector_rmse_mps": float(reference_rmse.cpu()),
            "reference_component_rmse_mps": torch.mean(reference_delta**2, dim=0).sqrt().cpu().tolist(),
            "reference_is_exact_native_cell_gather": True,
            "sampling_interpretation": "stratified probe diagnostic; not an unbiased native-volume integral",
        }
    observation = OracleObservation(
        roles=role_errors,
        reference_evidence_id=f"native_train_{probe.row_index}_{probe.split}_{len(probe.flat_indices)}",
        teacher_checkpoint_id=teacher_checkpoint_id,
        training_evidence_id=training_evidence_id,
    )
    return observation, diagnostics


def _batch_query_tensors(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    query = batch.query_xy.float()
    features = batch.query_features.float()
    return query, features


def _prediction_for_probe(
    model: Any,
    prepared: Any,
    probe: NativeProbeSet,
    case: Any,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    geometry = case.geometry_for_queries(probe.coordinates_D)
    query = torch.as_tensor(geometry["query_xy"], device=device).unsqueeze(0)
    features = torch.as_tensor(geometry["query_features"], device=device).unsqueeze(0)
    with torch.inference_mode():
        output = model.decode(prepared, query, query_features=features)
    normalizer: VelocityNormalizer = model.velocity_transform
    mean = query.new_tensor(normalizer.mean)
    scale = query.new_tensor(normalizer.safe_std)
    physical = (output["pred_field"] * scale + mean) * float(normalizer.u_ref_mps)
    return physical, output


def _replay_plan_on_search(
    model: Any,
    batch: Any,
    plan: AdaptiveCoverPlan,
    *,
    device: torch.device,
) -> tuple[Any, torch.Tensor, dict[str, Any], float, float]:
    model.set_native_interaction_policy(_FixedPlanPolicy((plan,)))
    with torch.inference_mode():
        prepared, prepare_ms = _timed(lambda: model.prepare_case(batch), device)
    query, features = _batch_query_tensors(batch)
    with torch.inference_mode():
        output, decode_ms = _timed(lambda: model.decode(prepared, query, query_features=features), device)
    normalizer: VelocityNormalizer = model.velocity_transform
    mean = query.new_tensor(normalizer.mean)
    scale = query.new_tensor(normalizer.safe_std)
    physical = (output["pred_field"] * scale + mean) * float(normalizer.u_ref_mps)
    return prepared, physical, output, prepare_ms, decode_ms


def _replay_dense_without_policy(
    model: Any,
    batch: Any,
    *,
    device: torch.device,
) -> tuple[Any, torch.Tensor, Any, float, float]:
    """Measure the production Dense path, which deliberately skips tree building."""

    model.set_native_interaction_policy(None)
    with torch.inference_mode():
        prepared, prepare_ms = _timed(lambda: model.prepare_case(batch), device)
    query, features = _batch_query_tensors(batch)
    with torch.inference_mode():
        output, decode_ms = _timed(
            lambda: model.decode(prepared, query, query_features=features), device
        )
    normalizer: VelocityNormalizer = model.velocity_transform
    mean = query.new_tensor(normalizer.mean)
    scale = query.new_tensor(normalizer.safe_std)
    physical = (output["pred_field"] * scale + mean) * float(normalizer.u_ref_mps)
    return prepared, physical, output, prepare_ms, decode_ms


def _preparation_cache_ledger(prepared: Any) -> dict[str, int]:
    state = prepared.dense_prepared.backend_state
    ledger = state.get("cover_preparation_ledger", {})
    return {str(key): int(value) for key, value in ledger.items()}


def _cold_warm_policy_timings(
    model: Any,
    record: _PanelCase,
    supervision_plan: AdaptiveCoverPlan,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Pair Dense, cold full-access, cold partial, and warm partial forwards."""

    clear = getattr(model.core, "clear_native_interaction_tree_cache", None)
    cache_info = getattr(model.core, "native_interaction_tree_cache_info", None)
    if not callable(clear) or not callable(cache_info):
        raise TypeError("native interaction geometry-tree cache controls are unavailable")

    clear()
    dense_prepared, dense_prediction, _dense_output, dense_prepare_ms, dense_decode_ms = (
        _replay_dense_without_policy(model, record.search_batch, device=device)
    )
    del dense_prepared, _dense_output
    if not torch.equal(dense_prediction, record.full_search_prediction):
        raise RuntimeError("no-policy Dense replay changed the frozen teacher output")
    dense_cache = dict(cache_info())

    clear()
    full_prepared, full_prediction, _full_output, full_prepare_ms, full_decode_ms = (
        _replay_plan_on_search(model, record.search_batch, record.full_plan, device=device)
    )
    full_cache = dict(cache_info())
    full_ledger = _preparation_cache_ledger(full_prepared)
    del full_prepared, _full_output
    if not torch.equal(full_prediction, dense_prediction):
        raise RuntimeError("cold full-access policy replay changed the Dense teacher output")

    clear()
    cold_prepared, cold_prediction, _cold_output, cold_prepare_ms, cold_decode_ms = (
        _replay_plan_on_search(model, record.search_batch, supervision_plan, device=device)
    )
    cold_cache = dict(cache_info())
    cold_ledger = _preparation_cache_ledger(cold_prepared)
    cold_plan = cold_prepared.dense_prepared.backend_state["cover_plans"][0]
    cold_support = _plan_probe_summary(cold_plan, record.search_batch, cold_prepared.encoded)
    cold_mask_tensors = (
        cold_plan.split_gates,
        cold_plan.module_membership,
        cold_plan.environment_membership,
    )
    del cold_prepared, _cold_output

    warm_prepared, warm_prediction, _warm_output, warm_prepare_ms, warm_decode_ms = (
        _replay_plan_on_search(model, record.search_batch, supervision_plan, device=device)
    )
    warm_cache = dict(cache_info())
    warm_ledger = _preparation_cache_ledger(warm_prepared)
    warm_plan = warm_prepared.dense_prepared.backend_state["cover_plans"][0]
    warm_support = _plan_probe_summary(warm_plan, record.search_batch, warm_prepared.encoded)
    warm_mask_tensors = (
        warm_plan.split_gates,
        warm_plan.module_membership,
        warm_plan.environment_membership,
    )
    del warm_prepared, _warm_output
    prediction_delta = cold_prediction - warm_prediction
    prediction_rmse_delta = torch.sqrt(torch.mean(prediction_delta.square()))
    reference_norm = torch.linalg.vector_norm(dense_prediction).clamp_min(torch.finfo(dense_prediction.dtype).tiny)
    max_abs_delta_mps = float(prediction_delta.abs().max().cpu())
    relative_l2_delta = float((torch.linalg.vector_norm(prediction_delta) / reference_norm).cpu())
    output_within_tolerance = bool(torch.allclose(
        cold_prediction,
        warm_prediction,
        atol=COLD_WARM_ABS_TOLERANCE_MPS,
        rtol=COLD_WARM_REL_TOLERANCE,
    ))
    plan_masks_equal = all(
        torch.equal(cold_value, warm_value)
        for cold_value, warm_value in zip(cold_mask_tensors, warm_mask_tensors, strict=True)
    )
    support_equal = cold_support == warm_support
    if not output_within_tolerance or not plan_masks_equal or not support_equal:
        failure_message = (
            "cold/warm cover replay failed: "
            f"max_abs_delta_mps={max_abs_delta_mps:.9g}, "
            f"relative_l2_delta={relative_l2_delta:.9g}, "
            f"plan_masks_equal={plan_masks_equal}, support_equal={support_equal}"
        )
    else:
        failure_message = None
    model.set_native_interaction_policy(None)

    def timing(prepare_ms: float, decode_ms: float) -> dict[str, float]:
        return {
            "prepare_ms": float(prepare_ms),
            "decode_ms": float(decode_ms),
            "complete_ms": float(prepare_ms + decode_ms),
        }

    return {
        "scope": "synchronized one-case Q1024; cold means cache explicitly cleared before prepare; warm means immediate same-geometry re-prepare",
        "model_eval": bool(not model.training),
        "policy_none_dense": {
            **timing(dense_prepare_ms, dense_decode_ms),
            "tree_cache": dense_cache,
            "prediction_exact_to_full_access": True,
        },
        "root_full_access_policy_cold": {
            **timing(full_prepare_ms, full_decode_ms),
            "tree_cache": full_cache,
            "preparation_ledger": full_ledger,
            "prediction_exact_to_policy_none_dense": True,
        },
        "primary_supervision_plan_cold": {
            **timing(cold_prepare_ms, cold_decode_ms),
            "tree_cache": cold_cache,
            "preparation_ledger": cold_ledger,
            "plan_is_partial": supervision_plan is not record.full_plan,
            "actual_hard_support": cold_support,
            "prediction_max_abs_vs_policy_none_dense_mps": _max_abs(cold_prediction, dense_prediction),
        },
        "primary_supervision_plan_warm_same_geometry": {
            **timing(warm_prepare_ms, warm_decode_ms),
            "tree_cache": warm_cache,
            "preparation_ledger": warm_ledger,
            "plan_is_partial": supervision_plan is not record.full_plan,
            "actual_hard_support": warm_support,
            "prediction_max_abs_vs_policy_none_dense_mps": _max_abs(warm_prediction, dense_prediction),
        },
        "cold_warm_consistency": {
            "absolute_tolerance_mps": COLD_WARM_ABS_TOLERANCE_MPS,
            "relative_tolerance": COLD_WARM_REL_TOLERANCE,
            "max_abs_physical_delta_mps": max_abs_delta_mps,
            "rmse_physical_delta_mps": float(prediction_rmse_delta.cpu()),
            "relative_l2_physical_delta": relative_l2_delta,
            "within_declared_float_tolerance": output_within_tolerance,
            "plan_masks_equal": plan_masks_equal,
            "hard_supports_equal": support_equal,
            "failure_message": failure_message,
        },
    }


def _measure_cost_coefficients(
    model: Any,
    records: Sequence[_PanelCase],
    organizer: InputOnlyCoverOrganizer,
    *,
    device: torch.device,
    query_count: int,
) -> tuple[MeasuredCostCoefficients, dict[str, Any]]:
    """Measure reader, routing and preparation terms on initial train rows."""

    if len(records) < 1:
        raise ValueError("at least one training row is required for measured cost calibration")
    components: dict[str, list[float]] = {
        name: [] for name in (
            "preparation_ms", "organization_ms_per_node", "route_ms_per_path",
            "qm_ms_per_row", "qe_ms_per_row", "wrapper_ms",
        )
    }
    calibration_rows: list[int] = []
    backend = model.core.backend
    for record in records[: min(3, len(records))]:
        encoded = record.encoded
        prepared = record.full_prepared
        state = prepared.dense_prepared.backend_state
        query, _query_features = _batch_query_tensors(record.search_batch)
        q = query[0]
        full_plan = record.full_plan
        with torch.inference_mode():
            projected_receiver_features = model.core._receiver_features(prepared.dense_prepared, query)
            _, route_ms = _timed(
                lambda plan=full_plan, queries=q, case_encoded=encoded: compile_cover_pairs(
                    plan,
                    queries,
                    module_present=case_encoded.module_present[0],
                    environment_weights=case_encoded.env_weights[0],
                ),
                device,
            )
            module_pairs, environment_pairs, ledger = compile_cover_pairs(
                full_plan,
                q,
                module_present=encoded.module_present[0],
                environment_weights=encoded.env_weights[0],
            )
            _, qm_ms = _timed(
                lambda backend_state=state, case_encoded=encoded, query_values=query, pairs=module_pairs:
                    backend.read_module_pairs(backend_state, case_encoded, query_values, pairs), device
            )
            _, qe_ms = _timed(
                lambda backend_state=state, case_encoded=encoded, query_values=query,
                       query_features=projected_receiver_features,
                       pairs=environment_pairs: backend.read_environment_pairs(
                    backend_state, case_encoded, query_values, query_features, pairs
                ), device
            )
            _, organize_ms = _timed(
                lambda case_encoded=encoded, prepared_case=prepared, case_trees=record.trees:
                    organizer.score_cases(case_encoded, _input_state(prepared_case), case_trees), device
            )
        total_paths = max(ledger.qm_raw_paths + ledger.qe_raw_paths, 1)
        components["preparation_ms"].append(max(record.full_prepare_ms, 0.0))
        components["organization_ms_per_node"].append(
            organize_ms / max(len(record.trees[0].nodes), 1)
        )
        components["route_ms_per_path"].append(route_ms / total_paths)
        components["qm_ms_per_row"].append(qm_ms / max(ledger.qm_unique_rows, 1))
        components["qe_ms_per_row"].append(qe_ms / max(ledger.qe_unique_rows, 1))
        # The per-row compiler measurement above already includes pair
        # packing and exact union construction; keep that work in the route
        # coefficient rather than charging it a second time.
        components["wrapper_ms"].append(
            max(record.full_search_decode_ms - qm_ms - qe_ms, 0.0)
        )
        calibration_rows.append(int(record.case.index))
    medians = {name: statistics.median(values) for name, values in components.items()}
    measurement_id = (
        f"run2103_train_rows_{'-'.join(map(str, calibration_rows))}_Q{query_count}_synchronized"
    )
    coefficients = MeasuredCostCoefficients(
        measurement_id=measurement_id,
        preparation_ms=medians["preparation_ms"],
        organization_ms_per_node=medians["organization_ms_per_node"],
        route_ms_per_path=medians["route_ms_per_path"],
        qm_ms_per_row=medians["qm_ms_per_row"],
        qe_ms_per_row=medians["qe_ms_per_row"],
        pack_ms_per_row=0.0,
        wrapper_ms=medians["wrapper_ms"],
    )
    report = {
        "measurement_id": measurement_id,
        "training_rows": calibration_rows,
        "query_count": query_count,
        "aggregation": "median of synchronized component timings on first up to three train rows",
        "pair_packing_accounting": "included in measured route_ms_per_path; pack_ms_per_row set to zero to avoid double counting",
        "coefficients": {
            name: float(getattr(coefficients, name))
            for name in (
                "preparation_ms", "organization_ms_per_node", "route_ms_per_path",
                "qm_ms_per_row", "qe_ms_per_row", "pack_ms_per_row", "wrapper_ms",
            )
        },
    }
    return coefficients, report


def _run_probe_prediction(
    model: Any,
    prepared: Any,
    case: Any,
    probe: NativeProbeSet,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], float]:
    geometry = case.geometry_for_queries(probe.coordinates_D)
    query = torch.as_tensor(geometry["query_xy"], device=device).unsqueeze(0)
    features = torch.as_tensor(geometry["query_features"], device=device).unsqueeze(0)
    with torch.inference_mode():
        output, elapsed_ms = _timed(
            lambda: model.decode(prepared, query, query_features=features), device
        )
    normalizer: VelocityNormalizer = model.velocity_transform
    mean = query.new_tensor(normalizer.mean)
    scale = query.new_tensor(normalizer.safe_std)
    physical = (output["pred_field"] * scale + mean) * float(normalizer.u_ref_mps)
    return physical.detach(), output, elapsed_ms


def _build_panel_case(
    model: Any,
    case: Any,
    *,
    query_count: int,
    seed: int,
    teacher_checkpoint_id: str,
    device: torch.device,
    anchor_measure_variant: str = "raw",
) -> tuple[_PanelCase, dict[str, Any]]:
    normalizer: VelocityNormalizer = model.velocity_transform
    search, verification = make_disjoint_native_probes(case, query_count=query_count, seed=seed)
    search_batch = _model_batch(_probe_batch(case, search, normalizer), device)
    verification_batch = _model_batch(_probe_batch(case, verification, normalizer), device)
    anchor_measure_provenance = {
        "variant": "raw_receiver_anchor_measure",
        "raw_mass": _anchor_mass_summary(
            search_batch.receiver_anchor_weights[0].detach().cpu().numpy(),
            search_batch.receiver_anchor_roles[0].detach().cpu().numpy(),
        ),
        "source_quadrature_changed": False,
    }
    if anchor_measure_variant == "role_balanced":
        search_batch, anchor_measure_provenance = _role_balanced_anchor_batch(search_batch)
        verification_batch, verification_anchor_provenance = _role_balanced_anchor_batch(
            verification_batch
        )
        if verification_anchor_provenance["variant"] != anchor_measure_provenance["variant"]:
            raise RuntimeError("search and verification batches use different anchor-measure variants")
    elif anchor_measure_variant != "raw":
        raise ValueError(f"unsupported anchor-measure variant {anchor_measure_variant!r}")

    model.set_native_interaction_policy(None)
    with torch.no_grad():
        dense_prepared, dense_prepare_ms = _timed(lambda: model.prepare_case(search_batch), device)
    dense_search, _, dense_search_decode_ms = _run_probe_prediction(
        model, dense_prepared, case, search, device=device
    )
    dense_verification, _, dense_verification_decode_ms = _run_probe_prediction(
        model, dense_prepared, case, verification, device=device
    )

    model.set_native_interaction_policy(_AllAccessPolicy())
    with torch.no_grad():
        full_prepared, full_prepare_ms = _timed(lambda: model.prepare_case(search_batch), device)
    full_search, full_search_output, full_search_decode_ms = _run_probe_prediction(
        model, full_prepared, case, search, device=device
    )
    full_verification, _full_verification_output, full_verification_decode_ms = _run_probe_prediction(
        model, full_prepared, case, verification, device=device
    )
    all_access_delta = _max_abs(dense_search, full_search)
    if not torch.equal(dense_search, full_search) or not torch.equal(dense_verification, full_verification):
        raise RuntimeError(f"native full-access policy changed Dense teacher output by {all_access_delta} m/s")
    backend_state = full_prepared.dense_prepared.backend_state
    trees = tuple(backend_state["cover_trees"])
    plan = backend_state["cover_plans"][0]
    evidence_id = f"run2103_train_row_{case.index}_search_{seed}_Q{query_count}"
    full_observation, full_diagnostics = _probe_observation(
        full_search[0], dense_search[0], search, normalizer,
        teacher_checkpoint_id=teacher_checkpoint_id,
        training_evidence_id=evidence_id,
    )
    record = _PanelCase(
        case=case,
        search=search,
        verification=verification,
        search_batch=search_batch,
        verification_batch=verification_batch,
        encoded=full_prepared.encoded,
        trees=trees,
        full_plan=plan,
        full_prepared=full_prepared,
        full_search_prediction=full_search,
        full_search_standardized=full_search_output["pred_field"].detach(),
        full_verification_prediction=full_verification,
        full_search_observation=full_observation,
        full_prepare_ms=full_prepare_ms,
        full_search_decode_ms=full_search_decode_ms,
        full_verification_decode_ms=full_verification_decode_ms,
        anchor_measure_provenance=anchor_measure_provenance,
        search_teacher_error=full_diagnostics,
    )
    timings = {
        "row_index": int(case.index),
        "dense_prepare_ms": dense_prepare_ms,
        "dense_search_decode_ms": dense_search_decode_ms,
        "dense_verification_decode_ms": dense_verification_decode_ms,
        "all_access_prepare_ms": full_prepare_ms,
        "all_access_search_decode_ms": full_search_decode_ms,
        "all_access_verification_decode_ms": full_verification_decode_ms,
        "all_access_dense_max_abs_mps": all_access_delta,
    }
    return record, timings


def _native_background_profile(case: Any) -> np.ndarray:
    run = case.run
    x_D = np.asarray(run.x_m, dtype=np.float64) / float(case.diameter_m)
    y_D = np.asarray(run.y_m, dtype=np.float64) / float(case.diameter_m)
    xx, yy = np.meshgrid(x_D, y_D, indexing="xy")
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    hubs = np.asarray(case.module_centers, dtype=np.float64)
    profile = np.zeros(int(run.nz), dtype=np.float64)
    nxny = int(run.nx) * int(run.ny)
    for iz in range(int(run.nz)):
        coords = np.column_stack((xy, np.full(len(xy), float(run.z_m[iz]) / case.diameter_m)))
        background = ~downstream_envelope(coords, hubs)
        values = np.asarray(run.U[iz * nxny : (iz + 1) * nxny, 0], dtype=np.float64)
        profile[iz] = float(values[background].mean() if background.any() else values.mean())
    return profile


def _native_grid_metrics(
    model: Any,
    prepared: Any,
    case: Any,
    normalizer: VelocityNormalizer,
    *,
    device: torch.device,
    chunk_size: int = 8192,
) -> tuple[dict[str, Any], float]:
    """Stream a complete native field with bounded receiver memory."""

    if chunk_size < 1:
        raise ValueError("native grid chunk size must be positive")
    run = case.run
    wx, wy, wz = support_weights(run.x_m, run.y_m, run.z_m)
    wx = np.asarray(wx, dtype=np.float64) / float(case.diameter_m)
    wy = np.asarray(wy, dtype=np.float64) / float(case.diameter_m)
    wz = np.asarray(wz, dtype=np.float64) / float(case.diameter_m)
    accumulators = {
        name: WeightedVelocityErrors()
        for name in ("volume", "hub_slab", "downstream_envelope", "background", "near_turbine")
    }
    background_profile = _native_background_profile(case)
    wake_residual_error = 0.0
    wake_residual_energy = 0.0
    hub_height_D = float(case.hub_height_m) / float(case.diameter_m)
    hubs = np.asarray(case.module_centers, dtype=np.float64)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(run.cell_count), chunk_size):
            stop = min(start + chunk_size, int(run.cell_count))
            flat = np.arange(start, stop, dtype=np.int64)
            coords_D = native_coordinates(run, flat, float(case.diameter_m))
            geometry = case.geometry_for_queries(coords_D)
            query = torch.as_tensor(geometry["query_xy"], device=device).unsqueeze(0)
            features = torch.as_tensor(geometry["query_features"], device=device).unsqueeze(0)
            output = model.decode(prepared, query, query_features=features)
            normalized = output["pred_field"][0]
            mean = normalized.new_tensor(normalizer.mean)
            scale = normalized.new_tensor(normalizer.safe_std)
            prediction = ((normalized * scale + mean) * float(normalizer.u_ref_mps)).cpu().numpy()
            target = np.asarray(run.U[start:stop], dtype=np.float64).copy()
            ix = flat % int(run.nx)
            iy = (flat // int(run.nx)) % int(run.ny)
            iz = flat // (int(run.nx) * int(run.ny))
            weights = wx[ix] * wy[iy] * wz[iz]
            hub_slab = np.abs(coords_D[:, 2] - hub_height_D) <= 0.5
            wake = downstream_envelope(coords_D, hubs)
            relative_xy = coords_D[:, None, :2] - hubs[None, :, :2]
            near_turbine = (np.sum(relative_xy**2, axis=-1).min(axis=1) <= 1.5**2) & hub_slab
            masks = {
                "volume": np.ones(len(flat), dtype=bool),
                "hub_slab": hub_slab,
                "downstream_envelope": wake,
                "background": ~wake & ~hub_slab,
                "near_turbine": near_turbine,
            }
            for name, mask in masks.items():
                if bool(mask.any()):
                    accumulators[name].add(prediction[mask], target[mask], weights[mask])
            if bool(wake.any()):
                residual_prediction = prediction[wake, 0] - background_profile[iz[wake]]
                residual_target = target[wake, 0] - background_profile[iz[wake]]
                wake_residual_error += float(np.sum(weights[wake] * (residual_prediction - residual_target) ** 2))
                wake_residual_energy += float(np.sum(weights[wake] * residual_target**2))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    physical_std = normalizer.safe_std * float(normalizer.u_ref_mps)
    result = {name: value.result(physical_std, float(normalizer.u_ref_mps)) for name, value in accumulators.items()}
    result["wake_ux_residual_vs_reference_background"] = {
        "relative_l2": float(np.sqrt(wake_residual_error / wake_residual_energy))
        if wake_residual_energy > 0.0 else None,
        "error_squared_integral": wake_residual_error,
        "target_residual_energy_integral": wake_residual_energy,
        "background_source": "native reference Ux mean outside geometric downstream envelope at each z level",
    }
    result["native_grid"] = {
        "full_cell_count": int(run.cell_count),
        "chunk_size": int(chunk_size),
        "support_measure": "centre-box tensor-product quadrature in D^3",
    }
    return result, elapsed_ms


def _root_mask_plan(
    template: AdaptiveCoverPlan,
    module_mask: torch.Tensor,
    environment_mask: torch.Tensor,
) -> AdaptiveCoverPlan:
    gates = template.split_gates.new_zeros(template.split_gates.shape)
    module = template.module_membership.new_zeros(template.module_membership.shape)
    environment = template.environment_membership.new_zeros(template.environment_membership.shape)
    module[0] = module_mask.to(device=module.device, dtype=module.dtype)
    environment[0] = environment_mask.to(device=environment.device, dtype=environment.dtype)
    return AdaptiveCoverPlan(template.tree, gates, module, environment)


def _per_case_pair_control(record: _PanelCase) -> AdaptiveCoverPlan:
    assert record.oracle_plan is not None
    module_support = (record.oracle_plan.module_membership > 0).any(dim=0)
    environment_support = (record.oracle_plan.environment_membership > 0).any(dim=0)
    return _root_mask_plan(record.full_plan, module_support, environment_support)


def _population_fixed_masks(records: Sequence[_PanelCase]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    max_modules = max(int(record.full_plan.module_membership.shape[1]) for record in records)
    module_selected = np.zeros(max_modules, dtype=np.float64)
    module_present = np.zeros(max_modules, dtype=np.float64)
    environment_selected = np.zeros(int(records[0].full_plan.environment_membership.shape[1]), dtype=np.float64)
    active_module_counts: list[int] = []
    active_environment_counts: list[int] = []
    for record in records:
        assert record.oracle_plan is not None
        present = record.encoded.module_present[0].detach().cpu().numpy() > 0.5
        support = (record.oracle_plan.module_membership > 0).any(dim=0).detach().cpu().numpy()
        module_selected[: len(present)] += support
        module_present[: len(present)] += present
        env_support = (record.oracle_plan.environment_membership > 0).any(dim=0).detach().cpu().numpy()
        environment_selected += env_support
        active_module_counts.append(int((support & present).sum()))
        active_environment_counts.append(int(env_support.sum()))
    module_frequency = module_selected / np.maximum(module_present, 1.0)
    target_module_count = round(float(np.mean(active_module_counts)))
    module_mask = np.zeros(max_modules, dtype=bool)
    candidate_modules = np.flatnonzero(module_present > 0)
    ranked_modules = candidate_modules[np.argsort(-module_frequency[candidate_modules], kind="stable")]
    module_mask[ranked_modules[:target_module_count]] = True
    environment_frequency = environment_selected / len(records)
    target_environment_count = round(float(np.mean(active_environment_counts)))
    environment_mask = np.zeros(len(environment_frequency), dtype=bool)
    ranked_environment = np.argsort(-environment_frequency, kind="stable")
    environment_mask[ranked_environment[:target_environment_count]] = True
    summary = {
        "source": "training-population oracle support frequency, ranked to the mean realized G-oracle source count",
        "target_module_count": target_module_count,
        "target_environment_count": target_environment_count,
        "module_support_frequency": module_frequency.tolist(),
        "environment_support_frequency": environment_frequency.tolist(),
    }
    return module_mask, environment_mask, summary


def _evaluate_probe_variant(
    model: Any,
    record: _PanelCase,
    *,
    name: str,
    probe: NativeProbeSet,
    batch: Any,
    policy: nn.Module | None,
    reference_prediction: torch.Tensor,
    teacher_checkpoint_id: str,
    device: torch.device,
) -> tuple[dict[str, Any], Any, AdaptiveCoverPlan]:
    model.set_native_interaction_policy(policy)
    with torch.inference_mode():
        prepared, prepare_ms = _timed(lambda: model.prepare_case(batch), device)
    query, features = _batch_query_tensors(batch)
    with torch.inference_mode():
        output, decode_ms = _timed(
            lambda: model.decode(prepared, query, query_features=features), device
        )
    normalizer: VelocityNormalizer = model.velocity_transform
    mean = query.new_tensor(normalizer.mean)
    scale = query.new_tensor(normalizer.safe_std)
    physical = (output["pred_field"] * scale + mean) * float(normalizer.u_ref_mps)
    if policy is None:
        plan = record.full_plan
        backend_state = prepared.dense_prepared.backend_state
    else:
        backend_state = prepared.dense_prepared.backend_state
        plan = backend_state["cover_plans"][0]
    observation, role_metrics = _probe_observation(
        physical[0], reference_prediction[0], probe, normalizer,
        teacher_checkpoint_id=teacher_checkpoint_id,
        training_evidence_id=f"run2103_train_row_{record.case.index}_verification_{probe.row_index}",
    )
    query_pairs = compile_cover_pairs(
        plan,
        query[0],
        module_present=prepared.encoded.module_present[0],
        environment_weights=prepared.encoded.env_weights[0],
    )
    qm, qe, query_ledger = query_pairs
    access = plan.access(query[0])
    source_bearing = (plan.module_membership > 0).any(dim=1) | (plan.environment_membership > 0).any(dim=1)
    active_groups = int(((access.receiver_group > 0).any(dim=0) & source_bearing).sum().cpu())
    preparation_ledger = backend_state.get("cover_preparation_ledger")
    if preparation_ledger is None:
        module_count = int(prepared.encoded.module_present.shape[1])
        environment_count = int(prepared.encoded.env_coords.shape[1])
        preparation_ledger = {
            "mm_executed": module_count * module_count,
            "me_executed": module_count * environment_count,
            "em_executed": module_count * environment_count,
            "dense_fallback": True,
        }
    report = {
        "variant": name,
        "row_index": int(record.case.index),
        "layout_index": int(record.case.layout_index),
        "direction_deg": float(record.case.wind_direction_deg),
        "evidence_mode": "disjoint_stratified_train_probe",
        "role_metrics": role_metrics,
        "teacher_preservation": {
            "gate_0p05": teacher_preservation_gate(
                observation, {role: TeacherDistortionLimit(0.05) for role in probe.roles}
            )[0],
            "gate_0p10": teacher_preservation_gate(
                observation, {role: TeacherDistortionLimit(0.10) for role in probe.roles}
            )[0],
            "gate_0p20": teacher_preservation_gate(
                observation, {role: TeacherDistortionLimit(0.20) for role in probe.roles}
            )[0],
            "role_normalized_distortion": {
                role: error.teacher for role, error in observation.roles.items()
            },
        },
        "reference_fidelity": {
            "role_vector_rmse_mps": {
                role: error.reference for role, error in observation.roles.items()
            },
            "physical_reference_gate_applied": False,
        },
        "actual_hard_support": {
            "active_groups_on_verification_queries": active_groups,
            "source_support_modules": int((access.module_source > 0).any(dim=0).sum().cpu()),
            "source_support_environment": int((access.environment_source > 0).any(dim=0).sum().cpu()),
            "cover_k_on_anchor_universe": plan.active_group_count(),
            "qm_unique_rows": int(qm.unique_pair_count),
            "qe_unique_rows": int(qe.unique_pair_count),
            "qm_raw_paths": int(query_ledger.qm_raw_paths),
            "qe_raw_paths": int(query_ledger.qe_raw_paths),
            "typed_preparation_rows": {
                key: int(value) for key, value in preparation_ledger.items()
            },
        },
        "measured_complete_ms": float(prepare_ms + decode_ms),
        "preparation_ms": float(prepare_ms),
        "decode_ms": float(decode_ms),
    }
    return report, prepared, plan


def _plan_probe_summary(plan: AdaptiveCoverPlan, batch: Any, encoded: Any) -> dict[str, Any]:
    query = batch.query_xy[0]
    qm, qe, ledger = compile_cover_pairs(
        plan,
        query,
        module_present=encoded.module_present[0],
        environment_weights=encoded.env_weights[0],
    )
    access = plan.access(query)
    return {
        "hard_cover_k": int(plan.active_group_count()),
        "module_source_support": int((access.module_source > 0).any(dim=0).sum().detach().cpu()),
        "environment_source_support": int((access.environment_source > 0).any(dim=0).sum().detach().cpu()),
        "qm_unique_rows": int(qm.unique_pair_count),
        "qe_unique_rows": int(qe.unique_pair_count),
        "qm_raw_paths": int(ledger.qm_raw_paths),
        "qe_raw_paths": int(ledger.qe_raw_paths),
    }


def _tree_node_metadata(tree: CaseLocalReceiverTree) -> list[dict[str, Any]]:
    parent: list[int | None] = [None] * len(tree.nodes)
    depth = [0] * len(tree.nodes)
    pending = [0]
    while pending:
        node_id = pending.pop()
        node = tree.nodes[node_id]
        for child in (node.left, node.right):
            if child is not None:
                parent[child] = node_id
                depth[child] = depth[node_id] + 1
                pending.append(child)
    return [
        {
            "node_id": index,
            "depth": depth[index],
            "parent": parent[index],
            "left": node.left,
            "right": node.right,
            "split_axis": node.split_axis,
            "is_leaf": bool(node.is_leaf),
            "anchor_indices": [int(value) for value in node.anchor_indices],
        }
        for index, node in enumerate(tree.nodes)
    ]


def _plan_supervision(plan: AdaptiveCoverPlan, module_present: torch.Tensor) -> dict[str, Any]:
    """Export one coherent, directly observed cover plan as input-only labels."""

    active_nodes = {0}
    root = plan.tree.nodes[0]
    if float(plan.split_gates[0].detach().cpu()) > 0.5:
        active_nodes.update(child for child in (root.left, root.right) if child is not None)

    module_targets = plan.module_membership.detach().cpu()
    environment_targets = plan.environment_membership.detach().cpu()
    split_targets = plan.split_gates.detach().cpu()
    module_present_cpu = (module_present.detach().cpu() > 0.5)
    module_observed = torch.zeros_like(module_targets, dtype=torch.bool)
    environment_observed = torch.zeros_like(environment_targets, dtype=torch.bool)
    for node_id in active_nodes:
        module_observed[node_id] = module_present_cpu
        environment_observed[node_id] = True
    split_observed = torch.zeros_like(split_targets, dtype=torch.bool)
    if not root.is_leaf:
        split_observed[0] = True
    return {
        "semantics": "one coherent plan selected by estimated cost from directly observed teacher-adequate candidates and checked on disjoint queries; independent alternative trials are not unioned",
        "split_targets": split_targets.tolist(),
        "split_observed": split_observed.tolist(),
        "module_targets": module_targets.tolist(),
        "module_observed": module_observed.tolist(),
        "environment_targets": environment_targets.tolist(),
        "environment_observed": environment_observed.tolist(),
    }


def _oracle_trial_record(
    trial: Any,
    *,
    timing: Mapping[str, Any],
    plan_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "proposal": {
            "kind": trial.proposal.kind,
            "node": trial.proposal.node,
            "source_type": trial.proposal.source_type,
            "source_index": trial.proposal.source_index,
            "training_evidence_id": trial.proposal.train_evidence_id,
        },
        "candidate_plan": dict(plan_summary),
        "teacher_preservation": {
            "passed": trial.gate_passed,
            "reason": trial.reason,
            "role_normalized_probe_distortion": {
                role: error.teacher for role, error in trial.observation.roles.items()
            },
        },
        "probe_reference_diagnostics_mps": {
            role: error.reference for role, error in trial.observation.roles.items()
        },
        "estimated_complete_ms": float(trial.estimated_ms),
        "actual_synchronized_complete_ms": float(timing["complete_ms"]),
        "actual_synchronized_prepare_ms": float(timing["prepare_ms"]),
        "actual_synchronized_decode_ms": float(timing["decode_ms"]),
        "actual_typed_preparation_rows": timing.get("typed_preparation_rows", {}),
        "accepted_by_oracle": bool(trial.accepted),
    }


def _proposal_cache_key(row: int, pass_name: str, proposal: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row),
        str(pass_name),
        str(proposal.get("kind")),
        int(proposal.get("node", 0)),
        proposal.get("source_type"),
        proposal.get("source_index"),
    )


def _freeze_training_panel(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    output_dir: str | Path,
    layout_count: int,
    active_layouts: int,
    anchor_measure_variant: str = "raw",
) -> tuple[Path, dict[str, Any], str, WindFarmNativeView, Any, tuple[Any, ...], Path, dict[str, Any]]:
    """Validate checkpoint-owned split and write the geometry-only manifest."""

    if not 1 <= active_layouts <= layout_count:
        raise ValueError("active layout count must be between one and the panel size")
    if layout_count != 12:
        raise ValueError("the native WindFarm training panel is frozen at 12 complete layouts")
    if anchor_measure_variant not in {"raw", "role_balanced"}:
        raise ValueError(f"unsupported anchor-measure variant {anchor_measure_variant!r}")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.name != "best_field.pt" or checkpoint.parent.name != "checkpoints":
        raise ValueError("oracle workflow requires the frozen Run 2103 best_field checkpoint")
    if checkpoint.parent.parent.name != FROZEN_RUN_DIRECTORY:
        raise ValueError(f"oracle checkpoint must come from run directory {FROZEN_RUN_DIRECTORY!r}")
    checkpoint_hash = _checkpoint_sha256(checkpoint)
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    validate_checkpoint_identity(payload, case_id="WindFarm", model_family="honf_forward", workflow="forward")
    if int(payload.get("best_epoch", payload.get("checkpoint_epoch", -1))) != 2475:
        raise ValueError("oracle workflow is locked to Run 2103 e2475")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping) or model_config.get("forward_architecture") != "dense_pairwise_field":
        raise ValueError("oracle workflow requires the intact Dense Run 2103 architecture")

    compact_file = resolve_path(str(compact_path))
    compact_names = (
        "case", "layout", "layout_index", "wd_deg", "n_turbines", "turbine_xy_D",
        "U_ref", "D_m", "hub_height_m",
    )
    with np.load(compact_file, allow_pickle=False) as archive:
        compact = {name: np.asarray(archive[name]).copy() for name in compact_names}
    view = WindFarmNativeView(
        resolve_path(str(volume_path)), compact_metadata=compact, include_receiver_anchors=True
    )
    split = _split_from_checkpoint(view, payload)
    layouts = select_training_layouts(view, split.train, count=layout_count)
    raw_anchor_mass_rows = []
    for layout in layouts:
        for row in layout.rows:
            case = view.run(int(row))
            if case.receiver_anchor_weights is None or case.receiver_anchor_roles is None:
                raise ValueError("frozen WindFarm panel requires receiver-anchor mass metadata")
            raw_anchor_mass_rows.append({
                "row_index": int(row),
                "layout_index": int(case.layout_index),
                "direction_deg": float(case.wind_direction_deg),
                **_anchor_mass_summary(case.receiver_anchor_weights, case.receiver_anchor_roles),
            })
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    panel_manifest = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "run_id": "2103",
        "best_epoch": 2475,
        "selection_source": "checkpoint-owned seed-42 training split and compact layout geometry only",
        "layout_count": layout_count,
        "active_layouts_for_this_run": active_layouts,
        "anchor_measure_variant": anchor_measure_variant,
        "raw_receiver_anchor_role_masses": raw_anchor_mass_rows,
        "tree_index_measure_rule": (
            "raw WindFarm receiver-anchor weights"
            if anchor_measure_variant == "raw"
            else "environment total mass=1 and combined rotor/module total mass=1; source quadrature is unchanged"
        ),
        "oracle_limits_frozen_before_field_search": {
            "evidence_mode": "teacher_preservation",
            "teacher_normalized_vector_rmse_max_per_role": TEACHER_DISTORTION_LIMIT,
            "role_errors_are_stratified_probe_diagnostics": True,
            "reference_fidelity_gate_applied": False,
        },
        "layouts": [
            {
                "layout_index": item.layout_index,
                "training_rows_direction_order": list(item.rows),
                "turbine_count": item.turbine_count,
                "geometry_feature_vector": list(item.feature_vector),
                "activated_in_this_run": index < active_layouts,
            }
            for index, item in enumerate(layouts)
        ],
    }
    (destination / "panel_manifest.json").write_text(
        json.dumps(panel_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint, payload, checkpoint_hash, view, split, layouts, destination, panel_manifest


def write_training_panel_manifest(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    output_dir: str | Path,
    layout_count: int = 12,
    active_layouts: int = 1,
    anchor_measure_variant: str = "raw",
) -> dict[str, Any]:
    """CPU-only checkpoint split and geometry-manifest preflight."""

    _checkpoint, _payload, checkpoint_hash, _view, split, layouts, destination, manifest = _freeze_training_panel(
        checkpoint_path=checkpoint_path,
        volume_path=volume_path,
        compact_path=compact_path,
        output_dir=output_dir,
        layout_count=layout_count,
        active_layouts=active_layouts,
        anchor_measure_variant=anchor_measure_variant,
    )
    return {
        "manifest_path": str(destination / "panel_manifest.json"),
        "checkpoint_sha256": checkpoint_hash,
        "training_row_count": int(split.train.size),
        "selected_layout_count": len(layouts),
        "active_layout_count": int(manifest["active_layouts_for_this_run"]),
        "active_direction_rows": [row for item in layouts[:active_layouts] for row in item.rows],
    }


def run_oracle_benchmark(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    device: str | torch.device,
    output_dir: str | Path,
    query_count: int = 1024,
    seed: int = 42,
    layout_count: int = 12,
    active_layouts: int = 1,
    resume_observations_path: str | Path | None = None,
    anchor_measure_variant: str = "raw",
) -> dict[str, Any]:
    """Run the bounded first-pass oracle over all directions of selected layouts.

    The layout manifest is frozen from checkpoint-owned training geometry
    before field observations. Each row has at most 48 x-block/source
    candidates, then a smaller y-axis/child-block pass if no split survives.
    """

    (
        checkpoint, payload, checkpoint_hash, view, _split, layouts, destination, _panel_manifest
    ) = _freeze_training_panel(
        checkpoint_path=checkpoint_path,
        volume_path=volume_path,
        compact_path=compact_path,
        output_dir=output_dir,
        layout_count=layout_count,
        active_layouts=active_layouts,
        anchor_measure_variant=anchor_measure_variant,
    )
    teacher_checkpoint_id = f"Run2103:e2475:{checkpoint_hash[:16]}"
    target_device = torch.device(device)
    normalizer = VelocityNormalizer.from_dict(dict(payload["normalization"]))
    active_rows = [row for item in layouts[:active_layouts] for row in item.rows]
    first_case = view.run(active_rows[0])
    first_search, _first_verification = make_disjoint_native_probes(
        first_case, query_count=query_count, seed=seed
    )
    materialization_batch = _model_batch(_probe_batch(first_case, first_search, normalizer), target_device)
    model, loaded = load_checkpoint(
        checkpoint, device=target_device, materialization_batch=materialization_batch
    )
    if model.architecture != "dense_pairwise_field":
        raise ValueError(f"expected intact Dense checkpoint, received {model.architecture!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    observations_path = destination / "candidate_observations.jsonl"
    observations_path.write_text("", encoding="utf-8")
    resumed_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    resume_source = None if resume_observations_path is None else Path(resume_observations_path).expanduser().resolve()
    if resume_source is not None:
        for line in resume_source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cached = json.loads(line)
            key = _proposal_cache_key(cached["row_index"], cached["proposal_pass"], cached["proposal"])
            if key in resumed_cache:
                raise ValueError(f"resume observation log contains a duplicate proposal key: {key}")
            resumed_cache[key] = cached
    records: list[_PanelCase] = []
    baseline_timings: list[dict[str, Any]] = []
    for row in active_rows:
        case = view.run(row)
        record, timing = _build_panel_case(
            model,
            case,
            query_count=query_count,
            seed=seed,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
            anchor_measure_variant=anchor_measure_variant,
        )
        records.append(record)
        baseline_timings.append(timing)

    first_encoded = records[0].encoded
    first_tree = records[0].trees[0]
    role_count = max(8, int(first_tree.universe.roles.max().item()) + 1)
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(first_encoded.module_tokens.shape[-1]),
        module_feature_dim=int(first_encoded.module_features.shape[-1]),
        environment_feature_dim=0 if first_encoded.env_features is None else int(first_encoded.env_features.shape[-1]),
        hidden_dim=32,
        role_count=role_count,
    ).to(target_device)
    cost_coefficients, cost_measurement = _measure_cost_coefficients(
        model, records, organizer, device=target_device, query_count=query_count
    )

    row_results: list[dict[str, Any]] = []
    total_observations = 0
    resumed_observation_count = 0
    fresh_observation_count = 0
    candidate_wall_times: list[float] = []
    for record in records:
        evidence_id = f"run2103_train_row_{record.case.index}_search_{seed}_Q{query_count}"
        role_limits = {
            name: TeacherDistortionLimit(TEACHER_DISTORTION_LIMIT)
            for name in record.search.roles
        }
        timing_by_plan: dict[int, dict[str, Any]] = {}
        summary_by_plan: dict[int, dict[str, Any]] = {
            id(record.full_plan): _plan_probe_summary(record.full_plan, record.search_batch, record.encoded)
        }
        proposal_metadata: dict[int, dict[str, Any]] = {}
        pass_name_current = {"name": "x_environment_blocks"}
        baseline_replay_prepared, baseline_replay_prediction, _baseline_output, paired_prepare_ms, paired_decode_ms = (
            _replay_plan_on_search(model, record.search_batch, record.full_plan, device=target_device)
        )
        del baseline_replay_prepared, _baseline_output
        if not torch.equal(baseline_replay_prediction, record.full_search_prediction):
            raise RuntimeError("timed full-access fixed-plan replay changed native Dense teacher output")
        paired_baseline_timing = {
            "prepare_ms": float(paired_prepare_ms),
            "decode_ms": float(paired_decode_ms),
            "complete_ms": float(paired_prepare_ms + paired_decode_ms),
        }

        def observe(
            plan: AdaptiveCoverPlan,
            _record: _PanelCase = record,
            _evidence_id: str = evidence_id,
            _timing_by_plan: dict[int, dict[str, Any]] = timing_by_plan,
            _summary_by_plan: dict[int, dict[str, Any]] = summary_by_plan,
            _proposal_metadata: dict[int, dict[str, Any]] = proposal_metadata,
            _pass_name: dict[str, str] = pass_name_current,
            _candidate_wall_times: list[float] = candidate_wall_times,
            _observations_path: Path = observations_path,
            _resume_cache: dict[tuple[Any, ...], dict[str, Any]] = resumed_cache,
            _resume_source: Path | None = resume_source,
        ) -> OracleObservation:
            nonlocal resumed_observation_count, fresh_observation_count
            if plan is _record.full_plan:
                return _record.full_search_observation
            candidate_proposal = _proposal_metadata.get(id(plan), {})
            cache_key = _proposal_cache_key(_record.case.index, _pass_name["name"], candidate_proposal)
            cached = _resume_cache.get(cache_key)
            if cached is not None:
                if cached["proposal"].get("training_evidence_id") != _evidence_id:
                    raise ValueError("resumed candidate belongs to different native search probes")
                observation = OracleObservation(
                    roles={
                        role: RoleError(
                            reference=value,
                            teacher=cached["role_teacher_normalized_probe_rmse"][role],
                            resolved=False,
                        )
                        for role, value in cached["role_reference_probe_rmse_mps"].items()
                    },
                    reference_evidence_id=None,
                    teacher_checkpoint_id=teacher_checkpoint_id,
                    training_evidence_id=_evidence_id,
                )
                _timing_by_plan[id(plan)] = {
                    "prepare_ms": float(cached["actual_synchronized_prepare_ms"]),
                    "decode_ms": float(cached["actual_synchronized_decode_ms"]),
                    "complete_ms": float(cached["actual_synchronized_complete_ms"]),
                    "typed_preparation_rows": dict(cached.get("typed_preparation_rows", {})),
                }
                _summary_by_plan[id(plan)] = _plan_probe_summary(plan, _record.search_batch, _record.encoded)
                _candidate_wall_times.append(_timing_by_plan[id(plan)]["complete_ms"])
                resumed = dict(cached)
                resumed["observation_source"] = "resumed_from_prior_measured_forward"
                resumed["resumed_from"] = str(_resume_source) if _resume_source is not None else None
                resumed["candidate_hard_support"] = _summary_by_plan[id(plan)]
                resumed["timing_scope"] = "synchronized prepare_case including tree build plus decode"
                with _observations_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(resumed, sort_keys=True) + "\n")
                resumed_observation_count += 1
                return observation
            prepared, prediction, _output, prepare_ms, decode_ms = _replay_plan_on_search(
                model, _record.search_batch, plan, device=target_device
            )
            observation, _role_diagnostics = _probe_observation(
                prediction[0],
                _record.full_search_prediction[0],
                _record.search,
                normalizer,
                teacher_checkpoint_id=teacher_checkpoint_id,
                training_evidence_id=_evidence_id,
            )
            backend_state = prepared.dense_prepared.backend_state
            typed_rows = backend_state.get("cover_preparation_ledger", {})
            _timing_by_plan[id(plan)] = {
                "prepare_ms": float(prepare_ms),
                "decode_ms": float(decode_ms),
                "complete_ms": float(prepare_ms + decode_ms),
                "typed_preparation_rows": {
                    str(key): int(value) if not torch.is_tensor(value) else int(value.detach().cpu())
                    for key, value in typed_rows.items()
                },
            }
            _candidate_wall_times.append(float(prepare_ms + decode_ms))
            _summary_by_plan[id(plan)] = _plan_probe_summary(plan, _record.search_batch, _record.encoded)
            proposal = _proposal_metadata.get(id(plan), {})
            fresh_observation_count += 1
            observed = {
                "row_index": int(_record.case.index),
                "proposal_pass": _pass_name["name"],
                "proposal": proposal,
                "candidate_hard_support": _summary_by_plan[id(plan)],
                "role_teacher_normalized_probe_rmse": {
                    role: error.teacher for role, error in observation.roles.items()
                },
                "role_reference_probe_rmse_mps": {
                    role: error.reference for role, error in observation.roles.items()
                },
                "actual_synchronized_prepare_ms": float(prepare_ms),
                "actual_synchronized_decode_ms": float(decode_ms),
                "actual_synchronized_complete_ms": float(prepare_ms + decode_ms),
                "typed_preparation_rows": _timing_by_plan[id(plan)]["typed_preparation_rows"],
                "observation_source": "new_synchronized_forward",
                "timing_scope": "synchronized prepare_case including tree build plus decode",
            }
            with _observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(observed, sort_keys=True) + "\n")
            return observation

        pass_results: list[tuple[Any, str]] = []
        for axis, pass_name in ((0, "x_environment_blocks"), (1, "y_environment_blocks")):
            if axis == 1 and pass_results:
                selected_first = pass_results[0][0].selected_plan
                if float(selected_first.split_gates[0].detach().cpu()) > 0.5:
                    break
            pass_name_current["name"] = pass_name
            called = False

            def propose(
                plan: AdaptiveCoverPlan,
                _axis: int = axis,
                _record: _PanelCase = record,
                _evidence_id: str = evidence_id,
                _proposal_metadata: dict[int, dict[str, Any]] = proposal_metadata,
            ) -> tuple[OracleProposal, ...]:
                nonlocal called
                if called:
                    return ()
                called = True
                proposals = _native_cover_proposals(
                    plan,
                    _record.encoded,
                    train_evidence_id=_evidence_id,
                    environment_blocks=16,
                    axis=_axis,
                    include_module_prunes=_axis == 0,
                    include_child_environment_blocks=_axis == 1,
                )
                for proposal in proposals:
                    _proposal_metadata[id(proposal.plan)] = {
                        "kind": proposal.kind,
                        "node": proposal.node,
                        "source_type": proposal.source_type,
                        "source_index": proposal.source_index,
                        "training_evidence_id": proposal.train_evidence_id,
                    }
                return proposals

            search_result = search_training_cover(
                record.full_plan,
                record.search_batch.query_xy[0],
                module_present=record.encoded.module_present[0],
                environment_weights=record.encoded.env_weights[0],
                observe=observe,
                propose=propose,
                costs=cost_coefficients,
                max_evaluations=ORACLE_OBSERVATION_CAP_PER_PASS,
                minimum_cost_improvement=0.01,
                evidence_mode="teacher_preservation",
                teacher_limits=role_limits,
                explore_alternatives=True,
            )
            pass_results.append((search_result, pass_name))

        selected, selected_pass = min(
            pass_results,
            key=lambda pair: pair[0].selected_estimated_ms,
        )
        oracle_selected_plan = selected.selected_plan
        oracle_selected_observation = selected.selected_observation
        oracle_selected_estimated_ms = float(selected.selected_estimated_ms)
        oracle_selected_proposal = next(
            (
                trial.proposal
                for result, _pass_name in pass_results
                for trial in result.trials
                if trial.proposal.plan is oracle_selected_plan
            ),
            None,
        )
        oracle_selected_timing = (
            paired_baseline_timing
            if oracle_selected_plan is record.full_plan
            else timing_by_plan[id(oracle_selected_plan)]
        )
        measured_faster_override = (
            oracle_selected_plan is not record.full_plan
            and oracle_selected_timing["complete_ms"] > paired_baseline_timing["complete_ms"]
        )
        if measured_faster_override:
            decision_plan = record.full_plan
            decision_observation = record.full_search_observation
            decision_timing = paired_baseline_timing
            decision_estimated_ms = float(selected.baseline_estimated_ms)
            decision_source = "full_access_retained_after_direct_paired_timing"
        else:
            decision_plan = oracle_selected_plan
            decision_observation = oracle_selected_observation
            decision_timing = oracle_selected_timing
            decision_estimated_ms = oracle_selected_estimated_ms
            decision_source = "teacher_oracle_estimated_cost_winner"

        decision_policy: nn.Module | None = None
        if decision_plan is not record.full_plan:
            decision_policy = _FixedPlanPolicy((decision_plan,))
        verification_report, _verification_prepared, _verification_plan = _evaluate_probe_variant(
            model,
            record,
            name="G-oracle-selected",
            probe=record.verification,
            batch=record.verification_batch,
            policy=decision_policy,
            reference_prediction=record.full_verification_prediction,
            teacher_checkpoint_id=teacher_checkpoint_id,
            device=target_device,
        )
        if oracle_selected_plan is decision_plan:
            oracle_label_verification_report = verification_report
        else:
            oracle_label_policy = (
                None
                if oracle_selected_plan is record.full_plan
                else _FixedPlanPolicy((oracle_selected_plan,))
            )
            oracle_label_verification_report, _label_prepared, _label_plan = _evaluate_probe_variant(
                model,
                record,
                name="G-oracle-teacher-label",
                probe=record.verification,
                batch=record.verification_batch,
                policy=oracle_label_policy,
                reference_prediction=record.full_verification_prediction,
                teacher_checkpoint_id=teacher_checkpoint_id,
                device=target_device,
            )
        oracle_search_gate_passed, oracle_search_gate_reasons = teacher_preservation_gate(
            oracle_selected_observation, role_limits
        )
        oracle_label_disjoint_gate_passed = bool(
            oracle_label_verification_report["teacher_preservation"]["gate_0p10"]
        )
        oracle_label_plan_verified = bool(
            oracle_search_gate_passed and oracle_label_disjoint_gate_passed
        )
        primary_supervision_plan = (
            oracle_selected_plan if oracle_label_plan_verified else record.full_plan
        )
        primary_supervision_source = (
            "estimated_cost_winner_verified_on_disjoint_queries"
            if oracle_label_plan_verified
            else "full_access_fallback_after_selected_candidate_failed_teacher_gate"
        )
        cache_timing_report = (
            _cold_warm_policy_timings(
                model,
                record,
                primary_supervision_plan,
                device=target_device,
            )
            if record is records[0]
            else None
        )
        if cache_timing_report is not None:
            cache_report_path = destination / "cold_warm_timing.json"
            cache_report_path.write_text(
                json.dumps(cache_timing_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            consistency = cache_timing_report["cold_warm_consistency"]
            if (
                not cache_timing_report["model_eval"]
                or not consistency["within_declared_float_tolerance"]
                or not consistency["plan_masks_equal"]
                or not consistency["hard_supports_equal"]
            ):
                raise RuntimeError(
                    "cold/warm cover consistency check failed; see "
                    f"{cache_report_path}"
                )
        trial_records: list[dict[str, Any]] = []
        for search_result, pass_name in pass_results:
            for trial in search_result.trials:
                key = id(trial.proposal.plan)
                timing = timing_by_plan.get(key)
                if timing is None:
                    raise RuntimeError("oracle trial has no synchronized forward observation timing")
                entry = _oracle_trial_record(
                    trial,
                    timing=timing,
                    plan_summary=summary_by_plan[key],
                )
                entry["proposal_pass"] = pass_name
                trial_records.append(entry)
        total_observations += len(trial_records)
        selected_summary = _plan_probe_summary(decision_plan, record.search_batch, record.encoded)
        oracle_selected_summary = _plan_probe_summary(
            oracle_selected_plan, record.search_batch, record.encoded
        )
        baseline_summary = _plan_probe_summary(record.full_plan, record.search_batch, record.encoded)
        if len(trial_records) > ORACLE_OBSERVATION_CAP_PER_PASS * MAX_ORACLE_PASSES_PER_CASE:
            raise RuntimeError("per-case bounded oracle observation ceiling was exceeded")
        row_results.append(
            {
                "row_index": int(record.case.index),
                "layout_index": int(record.case.layout_index),
                "direction_deg": float(record.case.wind_direction_deg),
                "anchor_measure": record.anchor_measure_provenance,
                "search_queries": int(query_count),
                "verification_queries": int(query_count),
                "search_verification_native_cells_disjoint": not bool(
                    np.intersect1d(record.search.flat_indices, record.verification.flat_indices).size
                ),
                "probe_sampling_label": "stratified; role errors are diagnostics, not unbiased full-volume estimates",
                "evidence_mode": "teacher_preservation",
                "teacher_distortion_limit_normalized_vector_rmse": TEACHER_DISTORTION_LIMIT,
                "reference_fidelity_gate_applied": False,
                "cost_measurement_id": cost_coefficients.measurement_id,
                "baseline_initial_all_access_prepare_plus_decode_ms": float(
                    record.full_prepare_ms + record.full_search_decode_ms
                ),
                "baseline_actual_synchronized_ms": paired_baseline_timing["complete_ms"],
                "baseline_actual_hard_support": baseline_summary,
                "baseline_teacher_probe_role_rmse": {
                    role: error.teacher for role, error in record.full_search_observation.roles.items()
                },
                "selected_pass": selected_pass,
                "selected_plan_source": decision_source,
                "oracle_estimated_selected_ms": oracle_selected_estimated_ms,
                "selected_estimated_ms": decision_estimated_ms,
                "selected_actual_synchronized_prepare_ms": decision_timing["prepare_ms"],
                "selected_actual_synchronized_decode_ms": decision_timing["decode_ms"],
                "selected_actual_synchronized_ms": decision_timing["complete_ms"],
                "estimated_winner_was_slower_than_full_access": measured_faster_override,
                "selected_actual_hard_support": selected_summary,
                "selected_teacher_probe_role_rmse": {
                    role: error.teacher for role, error in decision_observation.roles.items()
                },
                "teacher_oracle_search_winner": {
                    "proposal": (
                        {"kind": "root_full_access"}
                        if oracle_selected_proposal is None
                        else {
                            "kind": oracle_selected_proposal.kind,
                            "node": oracle_selected_proposal.node,
                            "source_type": oracle_selected_proposal.source_type,
                            "source_index": oracle_selected_proposal.source_index,
                            "training_evidence_id": oracle_selected_proposal.train_evidence_id,
                        }
                    ),
                    "teacher_search_gate_passed": bool(oracle_search_gate_passed),
                    "teacher_search_gate_reasons": list(oracle_search_gate_reasons),
                    "teacher_disjoint_verification_gate_passed": oracle_label_disjoint_gate_passed,
                    "teacher_adequate_for_primary_labels": oracle_label_plan_verified,
                    "estimated_complete_ms": oracle_selected_estimated_ms,
                    "actual_synchronized_prepare_ms": float(oracle_selected_timing["prepare_ms"]),
                    "actual_synchronized_decode_ms": float(oracle_selected_timing["decode_ms"]),
                    "actual_synchronized_complete_ms": float(oracle_selected_timing["complete_ms"]),
                    "actual_hard_support": oracle_selected_summary,
                    "search_teacher_role_distortion": {
                        role: error.teacher for role, error in oracle_selected_observation.roles.items()
                    },
                    "disjoint_verification": oracle_label_verification_report,
                    "split_gates": oracle_selected_plan.split_gates.detach().cpu().tolist(),
                    "module_membership": oracle_selected_plan.module_membership.detach().cpu().tolist(),
                    "environment_membership": oracle_selected_plan.environment_membership.detach().cpu().tolist(),
                },
                "cold_warm_policy_timings": cache_timing_report,
                "selected_reference_probe_role_rmse_mps": {
                    role: error.reference for role, error in decision_observation.roles.items()
                },
                "disjoint_verification": verification_report,
                "oracle_trial_count": len(trial_records),
                "candidate_observation_mean_synchronized_ms": (
                    float(np.mean([item["actual_synchronized_complete_ms"] for item in trial_records]))
                    if trial_records else None
                ),
                "candidate_observations": trial_records,
            }
        )
        partial_path = destination / "partial_oracle_benchmark.json"
        partial_path.write_text(
            json.dumps(
                {
                    "workflow": "native_cover_teacher_oracle_first_benchmark_partial",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "device": str(target_device),
                    "completed_rows": row_results,
                    "candidate_observation_count": total_observations,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        full_native_metrics, full_native_decode_ms = _native_grid_metrics(
            model,
            record.full_prepared,
            record.case,
            normalizer,
            device=target_device,
        )
        oracle_native_metrics, oracle_native_decode_ms = _native_grid_metrics(
            model,
            _verification_prepared,
            record.case,
            normalizer,
            device=target_device,
        )
        row_results[-1]["native_grid_validation"] = {
            "interpretation": "full native-grid physical-reference quadrature, separate from stratified teacher-oracle probe diagnostics",
            "G-full": {
                "prepare_ms": float(paired_baseline_timing["prepare_ms"]),
                "full_grid_decode_ms": float(full_native_decode_ms),
                "complete_ms": float(paired_baseline_timing["prepare_ms"] + full_native_decode_ms),
                "physical_reference_metrics": full_native_metrics,
            },
            "G-oracle-selected": {
                "prepare_ms": float(decision_timing["prepare_ms"]),
                "full_grid_decode_ms": float(oracle_native_decode_ms),
                "complete_ms": float(decision_timing["prepare_ms"] + oracle_native_decode_ms),
                "physical_reference_metrics": oracle_native_metrics,
            },
        }
        partial_path.write_text(
            json.dumps(
                {
                    "workflow": "native_cover_teacher_oracle_first_benchmark_partial",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "device": str(target_device),
                    "completed_rows": row_results,
                    "candidate_observation_count": total_observations,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        plan_json = {
            "row_index": int(record.case.index),
            "layout_index": int(record.case.layout_index),
            "direction_deg": float(record.case.wind_direction_deg),
            "anchor_measure": record.anchor_measure_provenance,
            "selection_pass": selected_pass,
            "selected_proposal": (
                "full_access_measured_faster_override"
                if measured_faster_override
                else {
                    "kind": "root_full_access"
                }
                if oracle_selected_proposal is None
                else {
                    "kind": oracle_selected_proposal.kind,
                    "node": oracle_selected_proposal.node,
                    "source_type": oracle_selected_proposal.source_type,
                    "source_index": oracle_selected_proposal.source_index,
                    "training_evidence_id": oracle_selected_proposal.train_evidence_id,
                }
            ),
            "teacher_gate": {
                "mode": "teacher_preservation",
                "limit_per_role": TEACHER_DISTORTION_LIMIT,
                "search_role_errors": {
                    role: error.teacher for role, error in decision_observation.roles.items()
                },
                "disjoint_verification_role_errors": verification_report["teacher_preservation"]["role_normalized_distortion"],
                "disjoint_verification_gate_passed": verification_report["teacher_preservation"]["gate_0p10"],
            },
            "node_order": _tree_node_metadata(decision_plan.tree),
            "module_source_order": [
                {"index": index, "center_D": record.case.module_centers[index].tolist(),
                 "present": bool(record.case.module_present[index] > 0.5)}
                for index in range(len(record.case.module_present))
            ],
            "environment_source_order": [
                {"index": index, "coordinate_D": record.case.env_coords[index].tolist()}
                for index in range(len(record.case.env_coords))
            ],
            "decision_plan": decision_source,
            "measured_execution_plan": {
                "source": decision_source,
                "full_access": decision_plan is record.full_plan,
                "direct_paired_timing_ms": {
                    "prepare": float(decision_timing["prepare_ms"]),
                    "decode": float(decision_timing["decode_ms"]),
                    "complete": float(decision_timing["complete_ms"]),
                },
                "actual_hard_support": selected_summary,
            },
            "split_gates": decision_plan.split_gates.detach().cpu().tolist(),
            "module_membership": decision_plan.module_membership.detach().cpu().tolist(),
            "environment_membership": decision_plan.environment_membership.detach().cpu().tolist(),
            "primary_supervision_source": primary_supervision_source,
            "primary_supervision": _plan_supervision(
                primary_supervision_plan, record.encoded.module_present[0]
            ),
            "cold_warm_policy_timings": cache_timing_report,
            "teacher_oracle_search_winner": {
                "proposal": (
                    {"kind": "root_full_access"}
                    if oracle_selected_proposal is None
                    else {
                        "kind": oracle_selected_proposal.kind,
                        "node": oracle_selected_proposal.node,
                        "source_type": oracle_selected_proposal.source_type,
                        "source_index": oracle_selected_proposal.source_index,
                        "training_evidence_id": oracle_selected_proposal.train_evidence_id,
                    }
                ),
                "teacher_search_gate_passed": bool(oracle_search_gate_passed),
                "teacher_disjoint_verification_gate_passed": oracle_label_disjoint_gate_passed,
                "teacher_adequate_for_primary_labels": oracle_label_plan_verified,
                "estimated_complete_ms": oracle_selected_estimated_ms,
                "actual_synchronized_complete_ms": float(oracle_selected_timing["complete_ms"]),
                "actual_hard_support": oracle_selected_summary,
                "disjoint_verification": oracle_label_verification_report,
                "split_gates": oracle_selected_plan.split_gates.detach().cpu().tolist(),
                "module_membership": oracle_selected_plan.module_membership.detach().cpu().tolist(),
                "environment_membership": oracle_selected_plan.environment_membership.detach().cpu().tolist(),
            },
            "typed_preparation_rows": decision_timing.get("typed_preparation_rows", {}),
            "alternative_candidate_observations": trial_records,
        }
        (destination / f"selected_plan_row_{record.case.index}.json").write_text(
            json.dumps(plan_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if total_observations > 4096:
        raise RuntimeError(f"oracle exceeded the 4096-observation panel ceiling: {total_observations}")
    report = {
        "workflow": "native_cover_teacher_oracle_first_benchmark",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "run_id": "2103",
        "best_epoch": int(loaded.get("best_epoch", loaded.get("checkpoint_epoch", -1))),
        "device": str(target_device),
        "anchor_measure_variant": anchor_measure_variant,
        "panel_manifest": str(destination / "panel_manifest.json"),
        "active_training_layouts": active_layouts,
        "active_direction_rows": len(active_rows),
        "query_count_per_disjoint_search_and_verification_probe": int(query_count),
        "search_proposal_ceiling_per_pass_per_case": ORACLE_OBSERVATION_CAP_PER_PASS,
        "max_passes_per_case": MAX_ORACLE_PASSES_PER_CASE,
        "maximum_observations_per_case": ORACLE_OBSERVATION_CAP_PER_PASS * MAX_ORACLE_PASSES_PER_CASE,
        "actual_candidate_observations": total_observations,
        "resumed_candidate_observations_counted_once": resumed_observation_count,
        "new_candidate_observations": fresh_observation_count,
        "resume_observation_source": None if resume_source is None else str(resume_source),
        "candidate_mean_synchronized_ms": float(np.mean(candidate_wall_times)) if candidate_wall_times else None,
        "candidate_timing_scope": "synchronized prepare_case including tree build plus decode; estimated row-work is not a speedup claim",
        "cost_coefficients_are_measured": True,
        "cost_measurement": cost_measurement,
        "baseline_timings": baseline_timings,
        "oracle_limits": {
            "mode": "teacher_preservation",
            "maximum_normalized_vector_rmse_by_role": TEACHER_DISTORTION_LIMIT,
            "fixed_before_candidate_search": True,
            "physical_reference_gate_applied": False,
        },
        "rows": row_results,
        "interpretation": "Train-only diagnostic of covers that preserve the frozen teacher on disjoint native probes; source support is not physical causality or reference sufficiency.",
    }
    (destination / "oracle_benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def run_preflight(
    *,
    checkpoint_path: str | Path,
    volume_path: str | Path,
    compact_path: str | Path,
    device: str | torch.device,
    output_dir: str | Path,
    query_count: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Check exact e2475, Dense/all-access parity, and GPU organizer planning."""

    if query_count < 16:
        raise ValueError("preflight query count must be at least 16 for both probe splits")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    if int(payload.get("best_epoch", payload.get("checkpoint_epoch", -1))) != 2475:
        raise ValueError("native-cover preflight is locked to the Run 2103 e2475 best-field checkpoint")
    if str(payload.get("run_id", payload.get("provenance", {}).get("run_id", "2103"))) != "2103":
        raise ValueError("native-cover preflight is locked to Run 2103")
    compact_file = resolve_path(str(compact_path))
    with np.load(compact_file, allow_pickle=False) as archive:
        compact = {key: np.asarray(archive[key]).copy() for key in (
            "case", "layout", "layout_index", "wd_deg", "n_turbines", "turbine_xy_D", "U_ref", "D_m", "hub_height_m"
        )}
    view = WindFarmNativeView(
        resolve_path(str(volume_path)), compact_metadata=compact, include_receiver_anchors=True
    )
    split = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    row = int(split.train[0])
    case = view.run(row)
    search, verification = make_disjoint_native_probes(case, query_count=query_count, seed=seed)
    normalizer = VelocityNormalizer.from_dict(dict(payload["normalization"]))
    target_device = torch.device(device)
    batch = _model_batch(_probe_batch(case, search, normalizer), target_device)
    started = time.perf_counter()
    model, loaded = load_checkpoint(
        checkpoint, device=target_device, materialization_batch=batch
    )
    if model.architecture != "dense_pairwise_field":
        raise ValueError(f"expected intact Dense checkpoint, received {model.architecture!r}")
    q = torch.as_tensor(search.coordinates_D, device=target_device).unsqueeze(0)
    q_features = torch.as_tensor(case.geometry_for_queries(search.coordinates_D)["query_features"], device=target_device).unsqueeze(0)
    target = torch.as_tensor(search.target_mps, device=target_device)
    with torch.inference_mode():
        prepared_dense = model.prepare_case(batch)
        dense_prediction = model.predict_physical(prepared_dense, q, q_features, receiver_chunk_size=32)

    model.set_native_interaction_policy(_AllAccessPolicy())
    with torch.inference_mode():
        prepared_all = model.prepare_case(batch)
    trees = prepared_all.dense_prepared.backend_state["cover_trees"]
    if not bool(prepared_all.dense_prepared.backend_state["cover_all_access"]):
        raise RuntimeError("root-only full-access preflight did not select Dense fallback")
    with torch.inference_mode():
        all_prediction = model.predict_physical(prepared_all, q, q_features, receiver_chunk_size=32)
    dense_all_delta = _max_abs(dense_prediction, all_prediction)

    encoded = prepared_all.encoded
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=0 if encoded.env_features is None else int(encoded.env_features.shape[-1]),
        hidden_dim=32,
        role_count=max(8, int(trees[0].universe.roles.max().item()) + 1),
    ).to(target_device).eval()
    model.set_native_interaction_policy(organizer)
    with torch.inference_mode():
        prepared_input = model.prepare_case(batch)
    organizer_tree = prepared_input.dense_prepared.backend_state["cover_trees"][0]
    organizer_plan = prepared_input.dense_prepared.backend_state["cover_plans"][0]
    if organizer_tree.universe.coordinates.device != target_device:
        raise RuntimeError("organizer geometry anchors were not moved to the model device")
    if organizer_plan.split_gates.device != target_device:
        raise RuntimeError("organizer plan was not produced on the model device")
    with torch.inference_mode():
        input_prediction = model.predict_physical(prepared_input, q, q_features, receiver_chunk_size=32)
    access = organizer_plan.access(q[0])
    source_bearing = (organizer_plan.module_membership > 0).any(dim=1) | (
        (organizer_plan.environment_membership > 0).any(dim=1)
    )
    active_groups_on_probe = int(
        ((access.receiver_group > 0).any(dim=0) & source_bearing).sum().cpu()
    )
    if not bool(torch.isfinite(input_prediction).all()):
        raise FloatingPointError("input-only organizer produced nonfinite native predictions")
    verification_is_disjoint = not np.intersect1d(search.flat_indices, verification.flat_indices).size
    if not verification_is_disjoint:
        raise RuntimeError("native search and verification cells overlap")
    report = {
        "workflow": "native_cover_organizer_preflight",
        "checkpoint": str(checkpoint),
        "run_id": "2103",
        "checkpoint_epoch": int(loaded.get("best_epoch", loaded.get("checkpoint_epoch", -1))),
        "row_index": row,
        "case": case.case,
        "layout_index": int(case.layout_index),
        "search_queries": int(query_count),
        "verification_queries": int(query_count),
        "disjoint_native_cells": verification_is_disjoint,
        "all_access_dense_max_abs_mps": dense_all_delta,
        "all_access_exact": bool(torch.equal(dense_prediction, all_prediction)),
        "input_policy_anchor_device": str(organizer_tree.universe.coordinates.device),
        "input_policy_prediction_finite": True,
        "input_policy_teacher_rmse_mps": float(torch.mean((input_prediction[0] - all_prediction[0]) ** 2).sqrt().cpu()),
        "input_policy_reference_rmse_mps": float(torch.mean((input_prediction[0] - target) ** 2).sqrt().cpu()),
        "input_policy_active_groups_on_search_queries": active_groups_on_probe,
        "input_policy_source_support_modules": int((organizer_plan.module_membership > 0).any(dim=0).sum().cpu()),
        "input_policy_source_support_environment": int((organizer_plan.environment_membership > 0).any(dim=0).sum().cpu()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--compact", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--oracle-dir", default=None)
    parser.add_argument("--fit-updates", type=int, default=100)
    parser.add_argument("--native-validation-layouts", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout-count", type=int, default=12)
    parser.add_argument("--active-layouts", type=int, default=12)
    parser.add_argument("--resume-observations", default=None)
    parser.add_argument("--anchor-measure", choices=("raw", "role_balanced"), default="raw")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    active_modes = sum((args.manifest_only, args.preflight_only, args.fit_only))
    if active_modes > 1:
        raise ValueError("--manifest-only, --preflight-only, and --fit-only are distinct modes")
    if args.fit_only and not args.oracle_dir:
        raise ValueError("--fit-only requires --oracle-dir")
    if args.preflight_only and args.anchor_measure != "raw":
        raise ValueError("role-balanced anchor indexing is a separate oracle mode, not preflight")
    query_count = args.query_count if args.query_count is not None else (32 if args.preflight_only else 1024)
    try:
        if args.manifest_only:
            report = write_training_panel_manifest(
                checkpoint_path=args.checkpoint,
                volume_path=args.volume,
                compact_path=args.compact,
                output_dir=args.output_dir,
                layout_count=args.layout_count,
                active_layouts=args.active_layouts,
                anchor_measure_variant=args.anchor_measure,
            )
        elif args.fit_only:
            from .native_cover_organizer_fit import run_organizer_fit

            report = run_organizer_fit(
                checkpoint_path=args.checkpoint,
                volume_path=args.volume,
                compact_path=args.compact,
                oracle_dir=args.oracle_dir,
                device=args.device,
                output_dir=args.output_dir,
                query_count=query_count,
                seed=args.seed,
                updates=args.fit_updates,
                native_validation_layouts=args.native_validation_layouts,
            )
        elif args.preflight_only:
            report = run_preflight(
                checkpoint_path=args.checkpoint,
                volume_path=args.volume,
                compact_path=args.compact,
                device=args.device,
                output_dir=args.output_dir,
                query_count=query_count,
                seed=args.seed,
            )
        else:
            report = run_oracle_benchmark(
                checkpoint_path=args.checkpoint,
                volume_path=args.volume,
                compact_path=args.compact,
                device=args.device,
                output_dir=args.output_dir,
                query_count=query_count,
                seed=args.seed,
                layout_count=args.layout_count,
                active_layouts=args.active_layouts,
                resume_observations_path=args.resume_observations,
                anchor_measure_variant=args.anchor_measure,
            )
    except Exception as exc:
        destination = Path(args.output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        candidate_log = destination / "candidate_observations.jsonl"
        partial_result = destination / "partial_oracle_benchmark.json"
        cold_warm_result = destination / "cold_warm_timing.json"
        failure = {
            "workflow": "native_cover_oracle_failure_manifest",
            "checkpoint_argument": str(args.checkpoint),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "candidate_observation_log": str(candidate_log),
            "candidate_observations_written": (
                sum(
                    bool(line.strip())
                    for line in candidate_log.read_text(encoding="utf-8").splitlines()
                )
                if candidate_log.is_file()
                else 0
            ),
            "partial_result": str(partial_result) if partial_result.is_file() else None,
            "cold_warm_diagnostic": str(cold_warm_result) if cold_warm_result.is_file() else None,
        }
        (destination / "failure_manifest.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
