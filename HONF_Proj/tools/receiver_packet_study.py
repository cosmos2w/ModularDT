#!/usr/bin/env python3
"""Maintained TRAIN teacher-table, packet comparison, and inverse-consumer tools.

The TRAIN table is sourced from exact coefficients of a frozen reconstruction
operator. DEV coefficients are accepted only by the verifier and never enter
the fitter or its residual calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXED25_MANIFEST_SHA256 = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from honf_forward_core.interface_fields.receiver_packets import (
    ActionDomain,
    InteractionBinding,
    ProposerFitReceipt,
    ReceiverPacket,
    ReceiverPairScorer,
    apply_affine_packet_increment,
    deterministic_cover,
    exact_teacher_sensitivities,
    fit_receiver_pair_scorer,
    geometry_equal_k_controls,
    prepared_context_fingerprints,
    response_rms_budget,
    scorer_parameter_sha256,
    union_receiver_packets,
    validate_bound_receiver_coordinates,
)


@dataclass(frozen=True)
class TrainTeacherTable:
    pair_features: torch.Tensor
    raw_sensitivities: torch.Tensor
    action_importance: torch.Tensor
    layout_ids: tuple[str, ...]
    patch_indices: torch.Tensor
    source_ids: tuple[str, ...]
    teacher_checkpoint_id: str
    manifest_sha256: str
    selected_layout_count: int


def validate_fixed25_train_authority(
    *,
    manifest_path: Path,
    selection_receipt_path: Path,
    selection_rule_path: Path,
    checkpoint_path: Path,
    layout_ids: Sequence[str],
    partition_labels: Sequence[str],
    teacher_checkpoint_id: str,
    expected_selection_rule_sha256: str,
    expected_manifest_sha256: str = FIXED25_MANIFEST_SHA256,
) -> dict[str, object]:
    """Bind a CLI TRAIN table to the authoritative fixed25 manifest/teacher receipt."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    selection_receipt_path = Path(selection_receipt_path).expanduser().resolve()
    selection_rule_path = Path(selection_rule_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("TRAIN table authority manifest bytes do not match fixed25_v1")
    manifest = json.loads(manifest_bytes)
    if manifest.get("scope") != "fixed_input_stratified_development_only":
        raise ValueError("Authority manifest is not the fixed input-stratified development profile")
    expected_layouts = tuple(str(value) for value in manifest["partitions"]["train"]["case_ids"])
    layouts = tuple(str(value) for value in layout_ids)
    partitions = tuple(str(value).upper() for value in partition_labels)
    if len(expected_layouts) != 150 or len(set(expected_layouts)) != 150:
        raise ValueError("fixed25_v1 authority manifest must define 150 unique TRAIN layouts")
    if layouts != expected_layouts:
        raise ValueError("Teacher table layout membership/order differs from authoritative fixed25_v1 TRAIN")
    if len(partitions) != len(layouts) or any(value != "TRAIN" for value in partitions):
        raise ValueError("Only authoritative primary TRAIN layouts may fit or calibrate packet scoring")

    rule_sha256 = hashlib.sha256(selection_rule_path.read_bytes()).hexdigest()
    if rule_sha256 != expected_selection_rule_sha256:
        raise ValueError("Supplied teacher selection rule is not the sealed fixed25_v1 rule")
    receipt = json.loads(selection_receipt_path.read_text())
    selected_arm = receipt.get("selected_arm")
    guards_pass = receipt.get("all_B4_promotion_guards_pass")
    if selected_arm not in {"R-geom", "R-flowctx"} or not isinstance(guards_pass, bool):
        raise ValueError("Teacher selection receipt must name an arm and aggregate B4 guard result")
    if (selected_arm == "R-flowctx") != guards_pass:
        raise ValueError("Selected teacher violates the sealed R-flowctx promotion rule")
    if receipt.get("selected_from_saved_100_cadence_checkpoints") is not True:
        raise ValueError("Selected teacher must come from common saved100-cadence checkpoints")
    selected_epoch = receipt.get("selected_epoch")
    if not isinstance(selected_epoch, int) or selected_epoch <= 0 or selected_epoch % 100:
        raise ValueError("Teacher receipt must select a positive saved100-cadence checkpoint")
    if receipt.get("selection_rule_sha256") != rule_sha256:
        raise ValueError("Teacher selection receipt does not bind the supplied sealed selection rule")
    if Path(receipt.get("checkpoint_path", "")).expanduser().resolve() != checkpoint_path:
        raise ValueError("Selected teacher checkpoint path differs from its authority receipt")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if receipt.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Selected teacher checkpoint bytes differ from the authority receipt")
    expected_teacher_id = f"thermal-response-checkpoint-sha256:{checkpoint_sha256}"
    if teacher_checkpoint_id != expected_teacher_id:
        raise ValueError("Teacher table identity does not include the selected checkpoint SHA-256")
    return {
        "profile": "fixed25_v1",
        "manifest_sha256": manifest_sha256,
        "train_layout_count": len(expected_layouts),
        "selection_receipt_sha256": hashlib.sha256(selection_receipt_path.read_bytes()).hexdigest(),
        "selection_rule_sha256": rule_sha256,
        "selected_arm": selected_arm,
        "selected_epoch": selected_epoch,
        "checkpoint_sha256": checkpoint_sha256,
    }


def build_train_teacher_table(
    *,
    pair_features: torch.Tensor,
    teacher_kernels: torch.Tensor,
    patch_weights: torch.Tensor,
    action_radii: torch.Tensor,
    layout_ids: Sequence[str],
    partition_labels: Sequence[str],
    source_ids_by_layout: Sequence[Sequence[str]],
    source_present: torch.Tensor,
    teacher_checkpoint_id: str,
    manifest_sha256: str,
    expected_layout_count: int = 150,
) -> TrainTeacherTable:
    """Flatten a primary-TRAIN-only exact-K table for patch/source fitting.

    Shapes are features ``[L,P,M,F]``, exact scalar-response kernels
    ``[L,P,Q,M]``, quadrature ``[L,P,Q]`` and radii ``[L,M]``. Invalid padded
    sources are excluded from the fit. The environment/context ancestry stays
    in each cheap feature vector and is not represented as an actuated source.
    """
    layouts = tuple(str(value) for value in layout_ids)
    partitions = tuple(str(value).upper() for value in partition_labels)
    if not teacher_checkpoint_id or not manifest_sha256:
        raise ValueError("Teacher checkpoint and fixed TRAIN manifest identities are required")
    if len(layouts) != len(set(layouts)) or len(layouts) != expected_layout_count:
        raise ValueError(f"Teacher table requires exactly {expected_layout_count} unique primary TRAIN layouts")
    if len(partitions) != len(layouts) or any(value != "TRAIN" for value in partitions):
        raise ValueError("Only primary TRAIN layouts may fit or calibrate the packet proposer")
    if pair_features.ndim != 4 or teacher_kernels.ndim != 4 or patch_weights.ndim != 3:
        raise ValueError("Teacher table tensors must use [L,P,M,F], [L,P,Q,M], [L,P,Q] shapes")
    layouts_n, patches, modules, _feature_width = pair_features.shape
    if layouts_n != len(layouts) or teacher_kernels.shape[:2] != (layouts_n, patches):
        raise ValueError("Feature and exact-kernel layout/patch dimensions disagree")
    if teacher_kernels.shape[2] != patch_weights.shape[2] or teacher_kernels.shape[3] != modules:
        raise ValueError("Exact K and patch quadrature dimensions disagree")
    if patch_weights.shape[:2] != (layouts_n, patches) or action_radii.shape != (layouts_n, modules):
        raise ValueError("Patch weights or action radii do not align with the teacher table")
    if source_present.shape != (layouts_n, modules) or len(source_ids_by_layout) != layouts_n:
        raise ValueError("Source presence and physical source identities must align with layouts")
    if not all(
        bool(torch.isfinite(value).all()) for value in (pair_features, teacher_kernels, patch_weights, action_radii)
    ):
        raise ValueError("TRAIN teacher-table inputs must be finite")
    if bool((action_radii < 0).any()) or bool((source_present < 0).any()) or bool((source_present > 1).any()):
        raise ValueError("Source radii must be nonnegative and source presence must lie in [0,1]")

    selected_features = []
    selected_raw = []
    selected_importance = []
    selected_layout_ids = []
    selected_patch_indices = []
    selected_source_ids = []
    for layout_index, layout_id in enumerate(layouts):
        ids = tuple(str(value) for value in source_ids_by_layout[layout_index])
        if len(ids) != modules:
            raise ValueError(f"Source identity count differs from padded source width for {layout_id}")
        active = source_present[layout_index] > 0.5
        active_ids = tuple(ids[index] for index in torch.nonzero(active, as_tuple=False).flatten().tolist())
        if not active_ids or any(not value for value in active_ids) or len(set(active_ids)) != len(active_ids):
            raise ValueError(f"Active physical source IDs must be unique and nonempty for {layout_id}")
        if bool((patch_weights[layout_index] < 0).any()) or not torch.allclose(
            patch_weights[layout_index].sum(-1),
            torch.ones(patches, device=patch_weights.device, dtype=patch_weights.dtype),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"Each TRAIN patch must have normalized nonnegative quadrature for {layout_id}")
        for patch_index in range(patches):
            raw, action_mass = exact_teacher_sensitivities(
                teacher_kernels[layout_index, patch_index],
                patch_weights[layout_index, patch_index],
                action_radii[layout_index],
            )
            for source_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
                selected_features.append(pair_features[layout_index, patch_index, source_index])
                selected_raw.append(raw[source_index])
                selected_importance.append(action_mass[source_index])
                selected_layout_ids.append(layout_id)
                selected_patch_indices.append(patch_index)
                selected_source_ids.append(ids[source_index])
    return TrainTeacherTable(
        pair_features=torch.stack(selected_features),
        raw_sensitivities=torch.stack(selected_raw),
        action_importance=torch.stack(selected_importance),
        layout_ids=tuple(selected_layout_ids),
        patch_indices=torch.tensor(selected_patch_indices, dtype=torch.long),
        source_ids=tuple(selected_source_ids),
        teacher_checkpoint_id=teacher_checkpoint_id,
        manifest_sha256=manifest_sha256,
        selected_layout_count=len(layouts),
    )


def save_train_teacher_table(table: TrainTeacherTable, path: Path) -> None:
    """Write portable, pickle-free TRAIN rows under ignored generated storage."""
    path = Path(path).resolve()
    generated_root = (ROOT / "diagnostics/generated").resolve()
    if not path.is_relative_to(generated_root):
        raise ValueError("Generated packet tables must stay under diagnostics/generated/")
    if path.suffix != ".npz":
        raise ValueError("Packet teacher-table output must use the `.npz` extension")
    if path.exists():
        raise FileExistsError("Preserve prior packet teacher tables; choose a new output identity")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pair_features=table.pair_features.detach().cpu().numpy(),
        raw_sensitivities=table.raw_sensitivities.detach().cpu().numpy(),
        action_importance=table.action_importance.detach().cpu().numpy(),
        layout_ids=np.asarray(table.layout_ids, dtype="U128"),
        patch_indices=table.patch_indices.cpu().numpy(),
        source_ids=np.asarray(table.source_ids, dtype="U256"),
        teacher_checkpoint_id=np.asarray(table.teacher_checkpoint_id),
        manifest_sha256=np.asarray(table.manifest_sha256),
        partition=np.asarray("TRAIN"),
        selected_layout_count=np.asarray(table.selected_layout_count, dtype=np.int64),
    )


def load_train_teacher_table(path: Path, *, expected_layout_count: int = 150) -> TrainTeacherTable:
    """Load and revalidate a pickle-free primary-TRAIN teacher table."""
    with np.load(path, allow_pickle=False) as arrays:
        if str(arrays["partition"].item()) != "TRAIN":
            raise ValueError("Packet scorer refuses non-TRAIN teacher tables")
        layout_ids = tuple(str(value) for value in arrays["layout_ids"].tolist())
        unique_layouts = tuple(sorted(set(layout_ids)))
        if len(unique_layouts) != expected_layout_count:
            raise ValueError(f"Packet fit requires {expected_layout_count} selected TRAIN layouts")
        return TrainTeacherTable(
            torch.from_numpy(arrays["pair_features"].copy()),
            torch.from_numpy(arrays["raw_sensitivities"].copy()),
            torch.from_numpy(arrays["action_importance"].copy()),
            layout_ids,
            torch.from_numpy(arrays["patch_indices"].copy()).long(),
            tuple(str(value) for value in arrays["source_ids"].tolist()),
            str(arrays["teacher_checkpoint_id"].item()),
            str(arrays["manifest_sha256"].item()),
            int(arrays["selected_layout_count"].item()),
        )


def load_packet_scorer(
    model_path: Path, *, device: str = "cpu"
) -> tuple[ReceiverPairScorer, ProposerFitReceipt, dict[str, object]]:
    """Load a generated scorer with safe weights-only deserialization."""
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    scorer = ReceiverPairScorer(int(payload["input_width"]), int(payload["hidden_width"]))
    scorer.load_state_dict(payload["state_dict"], strict=True)
    scorer.to(device).eval()
    receipt = ProposerFitReceipt(
        seed=int(metadata["seed"]),
        epochs=int(metadata["epochs"]),
        training_layout_ids=tuple(metadata["training_layout_ids"]),
        calibration_layout_ids=tuple(metadata["calibration_layout_ids"]),
        train_only_log_floor=float(metadata["train_only_log_floor"]),
        train_only_residual_margin=float(metadata["train_only_residual_margin"]),
        calibration_max_positive_residual=float(metadata["calibration_max_positive_residual"]),
        fit_losses=tuple(float(value) for value in metadata["fit_losses"]),
        optimizer_updates=int(metadata["optimizer_updates"]),
        parameter_count=int(metadata["parameter_count"]),
        parameter_sha256=str(metadata["parameter_sha256"]),
        calibration_scope=str(metadata["calibration_scope"]),
    )
    if scorer_parameter_sha256(scorer) != receipt.parameter_sha256:
        raise ValueError("Saved proposer weights differ from the TRAIN calibration receipt")
    if metadata.get("partition") != "TRAIN" or metadata.get("dev_used_for_fit") is not False:
        raise ValueError("Packet scorer metadata must certify TRAIN-only fitting and calibration")
    return scorer, receipt, metadata


def summarize_packet_catalog(
    packet_source_lists: Sequence[Sequence[str]],
    physical_source_ids: Sequence[str],
) -> dict[str, object]:
    """Report actual K_src per receiver packet and nonredundant K_packet."""
    physical_ids = tuple(str(value) for value in physical_source_ids)
    if not physical_ids or any(not value for value in physical_ids) or len(set(physical_ids)) != len(physical_ids):
        raise ValueError("Packet catalogue requires unique physical donor IDs")
    if not packet_source_lists:
        raise ValueError("Packet catalogue must contain at least one declared receiver packet")
    canonical_lists = []
    for row in packet_source_lists:
        values = tuple(str(value) for value in row)
        if len(values) != len(set(values)):
            raise ValueError("A receiver packet cannot repeat a physical donor")
        if any(value not in set(physical_ids) for value in values):
            raise ValueError("Packet catalogue contains a source outside the physical donor set")
        canonical_lists.append(tuple(sorted(values)))
    unique = sorted(set(canonical_lists))
    counts = [len(row) for row in canonical_lists]
    return {
        "K_packet": len(unique),
        "K_src_by_receiver_packet": counts,
        "K_src_min": min(counts),
        "K_src_median": float(np.median(counts)),
        "K_src_max": max(counts),
        "nonredundant_source_lists": [list(row) for row in unique],
        "receiver_packet_count": len(canonical_lists),
        "physical_donor_count": len(physical_ids),
    }


def evaluate_patch_selectors(
    *,
    kernel: torch.Tensor,
    patch_weights: torch.Tensor,
    action_radii: torch.Tensor,
    delta_control: torch.Tensor,
    source_ids: Sequence[str],
    learned_source_ids: Sequence[str],
    source_coordinates: torch.Tensor,
    receiver_center: torch.Tensor,
    flow_direction: torch.Tensor,
    budget_fraction: float,
    train_response_rms: float,
    reference_increment: torch.Tensor | None = None,
    protected_near_source_ids: Sequence[str] = (),
    measured_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Compare full, exact cover, learned proposal, and equal-K geometry controls.

    The exact kernel is used here only as an offline verifier. Proposal-time
    code accepts no kernel. A native reference is optional and remains a
    separate physical-error column from teacher distortion.
    """
    ids = tuple(str(value) for value in source_ids)
    if kernel.ndim != 2 or delta_control.shape != (len(ids),) or action_radii.shape != (len(ids),):
        raise ValueError("Patch verification expects K[Q,M] and source-aligned controls/radii")
    if reference_increment is not None and reference_increment.shape != (kernel.shape[0],):
        raise ValueError("Native reference increment must align with patch receivers")
    if not bool(torch.isfinite(delta_control).all()) or bool((delta_control.abs() > action_radii + 1e-7).any()):
        raise ValueError("Saved action falls outside this packet's declared radius domain")
    if patch_weights.shape != (kernel.shape[0],) or not torch.isclose(
        patch_weights.sum(),
        patch_weights.new_tensor(1.0),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Verification patch weights must be normalized")
    raw, importance = exact_teacher_sensitivities(kernel, patch_weights, action_radii)
    budget = response_rms_budget(budget_fraction, train_response_rms)
    exact_budget_ids, _ = deterministic_cover(
        importance,
        ids,
        budget,
        protected_source_ids=protected_near_source_ids,
    )
    learned_ids = tuple(str(value) for value in learned_source_ids)
    if not set(learned_ids).issubset(ids):
        raise ValueError("Learned packet contains unknown physical source IDs")
    near = {str(value) for value in protected_near_source_ids}
    learned_set = set(learned_ids) | near
    if not learned_set.issubset(ids):
        raise ValueError("Protected near-source IDs must belong to the teacher source catalogue")
    learned_ids = tuple(source_id for source_id in ids if source_id in learned_set)
    equal_k = len(learned_ids)
    equal_geometry = (
        geometry_equal_k_controls(
            source_coordinates,
            receiver_center,
            ids,
            flow_direction,
            equal_k,
            protected_source_ids=tuple(near),
        )
        if equal_k
        else {"nearest": (), "upstream": ()}
    )
    exact_equal_k_order = sorted(range(len(ids)), key=lambda index: (-float(importance[index]), ids[index]))
    exact_equal_k = set(near)
    for index in exact_equal_k_order:
        if len(exact_equal_k) >= equal_k:
            break
        exact_equal_k.add(ids[index])
    selector_ids = {
        "full": ids,
        "exact_cover": exact_budget_ids,
        "exact_equal_k": tuple(source_id for source_id in ids if source_id in exact_equal_k),
        "learned_proposal": learned_ids,
        "nearest_equal_k": equal_geometry["nearest"],
        "upstream_equal_k": equal_geometry["upstream"],
    }
    full_field = kernel @ delta_control
    physical = None if reference_increment is None else reference_increment
    rows = {}
    for name, selected in selector_ids.items():
        keep = torch.tensor([source_id in set(selected) for source_id in ids], device=kernel.device)
        approximate = kernel[:, keep] @ delta_control[keep] if bool(keep.any()) else torch.zeros_like(full_field)
        omission = full_field - approximate
        omitted_bound = float(importance[~keep].sum())
        distortion = torch.sqrt((patch_weights * omission.square()).sum())
        row = {
            "selected_source_ids": list(selected),
            "K_src": len(selected),
            "teacher_distortion_rms": float(distortion),
            "exact_omitted_triangle_bound": omitted_bound,
            "bound_violation": bool(float(distortion) > omitted_bound + 1e-6),
            "full_model_physical_response_rmse": None
            if physical is None
            else float(torch.sqrt((patch_weights * (full_field - physical).square()).sum())),
            "selected_physical_response_rmse": None
            if physical is None
            else float(torch.sqrt((patch_weights * (approximate - physical).square()).sum())),
            "actual_execution": None if measured_receipts is None else measured_receipts.get(name),
        }
        rows[name] = row
    return {
        "scope": "offline_exact_teacher_verifier; physical reference optional and separately labeled",
        "budget_fraction": budget_fraction,
        "budget_in_temperature_units": budget,
        "raw_sensitivity": {source_id: float(raw[index]) for index, source_id in enumerate(ids)},
        "radius_weighted_action_importance": {
            source_id: float(importance[index]) for index, source_id in enumerate(ids)
        },
        "K_packet_not_inferred": None,
        "selectors": rows,
    }


@dataclass(frozen=True)
class IncrementRead:
    receiver_ids: tuple[str, ...]
    receivers: torch.Tensor
    exact_baseline: torch.Tensor
    packets: tuple[ReceiverPacket, ...]
    receiver_features: torch.Tensor | None = None
    receiver_tensor_ids: torch.Tensor | None = None
    baseline_binding: InteractionBinding | None = None


@dataclass(frozen=True)
class FixedGeometryIncrementTask:
    task_id: str
    operator: object
    prepared_context: object
    delta_control: torch.Tensor
    binding: InteractionBinding
    action_domain: ActionDomain
    observed: IncrementRead
    held: IncrementRead


def _native_source_id(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


@torch.no_grad()
def _apply_padding_only_control(
    task: FixedGeometryIncrementTask,
    read: IncrementRead,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Measure the pre-MLP all-active gather control without activating packets."""
    operator = task.operator
    binding = task.binding
    context = task.prepared_context
    if getattr(operator, "mode", None) != "direct" or int(getattr(operator, "query_width", 0)) != 0:
        raise ValueError("Padding-only control requires the bound direct, query_width=0 operator")
    if binding.capability != "affine_scalar_increment":
        raise ValueError("Padding-only control requires an affine scalar-control binding")
    baseline_binding = read.baseline_binding or binding
    if baseline_binding.fingerprint != binding.fingerprint:
        raise ValueError("Padding-only control exact baseline belongs to a stale scene")
    receiver_validation_seconds = validate_bound_receiver_coordinates(binding, read.receivers, read.receiver_ids)
    context_geometry, context_state = prepared_context_fingerprints(context)
    if (context_geometry, context_state) != (binding.geometry_fingerprint, binding.context_fingerprint):
        raise ValueError("Padding-only control prepared context differs from its scene binding")
    present = context.present
    native_source_ids = context.source_ids
    if present.ndim != 2 or native_source_ids.shape != present.shape:
        raise ValueError("Padding-only control prepared source catalog is malformed")
    for batch_index in range(present.shape[0]):
        slots = tuple(torch.nonzero(present[batch_index] > 0.5, as_tuple=False).flatten().tolist())
        if slots != binding.source_slots:
            raise ValueError("Padding-only control source slots differ from the bound active physical catalog")
        native_ids = tuple(_native_source_id(native_source_ids[batch_index, slot]) for slot in slots)
        if native_ids != binding.prepared_source_ids:
            raise ValueError("Padding-only control native source identities differ from the bound physical sources")
    if not task.action_domain.accepts(task.delta_control, binding.source_ids):
        raise ValueError("Padding-only control action is outside its declared componentwise action box")
    if hasattr(operator, "assert_owned"):
        operator.assert_owned(context)
    context.assert_fresh()
    batch, queries, _dimension = read.receivers.shape
    full_source_count = int(context.centers.shape[1])
    source_keep = torch.zeros(batch, queries, full_source_count, dtype=torch.bool, device=read.receivers.device)
    source_keep[:, :, list(binding.source_slots)] = True
    if not hasattr(operator, "prepare_receivers_subset"):
        raise RuntimeError("Shared interaction core does not expose the pre-MLP subset gather API")
    response = operator.prepare_receivers_subset(
        context,
        read.receivers,
        source_keep=source_keep,
        receiver_features=read.receiver_features,
        receiver_ids=read.receiver_tensor_ids,
    )
    padded_delta = task.delta_control.new_zeros(task.delta_control.shape[0], full_source_count)
    padded_delta[:, list(binding.source_slots)] = task.delta_control
    increment = operator.apply_increment(response, padded_delta, accumulation_dtype=read.exact_baseline.dtype)
    if read.exact_baseline.shape != increment.shape:
        raise ValueError("Padding-only control increment differs from the exact baseline receiver/output shape")
    execution = getattr(response, "execution_receipt", {})
    receipt = {
        "mode": "padding_only_control_all_active_sources",
        "control_scope": "all active physical sources retained; only absent padded slots can be removed",
        "active_source_omissions": 0,
        "full_fallback": False,
        "exact_baseline_reused": True,
        "used_pre_mlp_subset_gather": True,
        "receiver_binding_validation_seconds": receiver_validation_seconds,
        "source_ids": list(binding.source_ids),
        "source_slots": list(binding.source_slots),
        "prepared_source_ids": list(binding.prepared_source_ids),
        "selected_active_source_pair_rows": batch * queries * len(binding.source_slots),
        "executed_source_rows": int(execution.get("fine_rows", 0)),
        "full_far_rows": int(execution.get("full_far_rows", batch * queries * full_source_count)),
        "executed_subset_receipt": execution,
        "proposal_K_calls": 0,
        "teacher_K_calls": 0,
    }
    return read.exact_baseline + increment, receipt


@torch.no_grad()
def run_fixed_geometry_inverse_consumer(
    tasks: Sequence[FixedGeometryIncrementTask],
) -> dict[str, object]:
    """Run exactly four saved TRAIN model-only increments on observed/held reads."""
    if len(tasks) != 4 or len({task.task_id for task in tasks}) != 4:
        raise ValueError("The bounded inverse software consumer requires exactly four unique TRAIN tasks")
    result = {
        "scope": "model-only fixed-geometry increments; no physical solve or design search",
        "optimizer_updates": 0,
        "physical_reference_attempts": 0,
        "receiver_binding_validation_seconds": 0.0,
        "full_mode_fine_rows": 0,
        "advisory_mode_fine_rows": 0,
        "experimental_mode_fine_rows": 0,
        "experimental_mode_near_rows": 0,
        "experimental_mode_full_far_rows": 0,
        "padding_only_control_fine_rows": 0,
        "padding_only_control_near_rows": 0,
        "padding_only_control_full_far_rows": 0,
        "padding_only_control_pre_mlp_subset_calls": 0,
        "full_mode_near_rows": 0,
        "full_mode_full_far_rows": 0,
        "advisory_mode_near_rows": 0,
        "advisory_mode_full_far_rows": 0,
        "experimental_pre_mlp_subset_calls": 0,
        "mode_receiver_apply_seconds": {mode: 0.0 for mode in ("full", "packet_advisory", "packet_experimental")},
        "padding_only_control_receiver_apply_seconds": 0.0,
        "mode_peak_extra_allocated_bytes_max": {mode: 0 for mode in ("full", "packet_advisory", "packet_experimental")},
        "mode_peak_extra_allocated_bytes_measured_calls": {mode: 0 for mode in ("full", "packet_advisory", "packet_experimental")},
        "padding_only_control_peak_extra_allocated_bytes_max": 0,
        "padding_only_control_peak_extra_allocated_bytes_measured_calls": 0,
        "receiver_read_calls_per_mode": {mode: 0 for mode in ("full", "packet_advisory", "packet_experimental")},
        "tasks": {},
    }
    for task in tasks:
        task_result = {}
        for read_name in ("observed", "held"):
            read: IncrementRead = getattr(task, read_name)
            packet_union = union_receiver_packets(
                task.binding,
                task.action_domain,
                read.packets,
                read.receiver_ids,
                device=read.receivers.device,
            )
            modes = {}
            for mode in ("full", "packet_advisory", "packet_experimental"):
                if read.receivers.is_cuda:
                    torch.cuda.synchronize(read.receivers.device)
                    allocated_before = torch.cuda.memory_allocated(read.receivers.device)
                    torch.cuda.reset_peak_memory_stats(read.receivers.device)
                else:
                    allocated_before = None
                started = perf_counter()
                prediction, receipt = apply_affine_packet_increment(
                    task.operator,
                    task.prepared_context,
                    read.receivers,
                    read.exact_baseline,
                    task.delta_control,
                    task.binding,
                    task.binding,
                    task.action_domain,
                    packet_union,
                    mode=mode,
                    requested_receiver_ids=read.receiver_ids,
                    receiver_features=read.receiver_features,
                    receiver_ids=read.receiver_tensor_ids,
                    baseline_binding=read.baseline_binding or task.binding,
                    accumulation_dtype=read.exact_baseline.dtype,
                )
                if read.receivers.is_cuda:
                    torch.cuda.synchronize(read.receivers.device)
                receipt["receiver_application_seconds"] = perf_counter() - started
                result["mode_receiver_apply_seconds"][mode] += receipt["receiver_application_seconds"]
                result["receiver_read_calls_per_mode"][mode] += 1
                if allocated_before is not None:
                    peak = int(torch.cuda.max_memory_allocated(read.receivers.device))
                    extra_peak = max(0, peak - allocated_before)
                    receipt["torch_peak_extra_allocated_bytes"] = extra_peak
                    result["mode_peak_extra_allocated_bytes_max"][mode] = max(
                        result["mode_peak_extra_allocated_bytes_max"][mode], extra_peak
                    )
                    result["mode_peak_extra_allocated_bytes_measured_calls"][mode] += 1
                else:
                    receipt["torch_peak_extra_allocated_bytes"] = None
                result["receiver_binding_validation_seconds"] += receipt["receiver_binding_validation_seconds"]
                execution = receipt["executed_subset_receipt"]
                if mode == "full":
                    result["full_mode_fine_rows"] += execution["fine_rows"]
                    result["full_mode_near_rows"] += execution["near_rows"]
                    result["full_mode_full_far_rows"] += execution["full_far_rows"]
                elif mode == "packet_advisory":
                    result["advisory_mode_fine_rows"] += execution["fine_rows"]
                    result["advisory_mode_near_rows"] += execution["near_rows"]
                    result["advisory_mode_full_far_rows"] += execution["full_far_rows"]
                else:
                    result["experimental_mode_fine_rows"] += execution["fine_rows"]
                    result["experimental_mode_near_rows"] += execution["near_rows"]
                    result["experimental_mode_full_far_rows"] += execution["full_far_rows"]
                    result["experimental_pre_mlp_subset_calls"] += int(receipt["used_pre_mlp_subset_gather"])
                modes[mode] = {"prediction": prediction, "receipt": receipt}
            if read.receivers.is_cuda:
                torch.cuda.synchronize(read.receivers.device)
                allocated_before = torch.cuda.memory_allocated(read.receivers.device)
                torch.cuda.reset_peak_memory_stats(read.receivers.device)
            else:
                allocated_before = None
            padding_started = perf_counter()
            padding_prediction, padding_receipt = _apply_padding_only_control(task, read)
            if read.receivers.is_cuda:
                torch.cuda.synchronize(read.receivers.device)
            padding_seconds = perf_counter() - padding_started
            padding_receipt["receiver_application_seconds"] = padding_seconds
            result["padding_only_control_receiver_apply_seconds"] += padding_seconds
            padding_execution = padding_receipt["executed_subset_receipt"]
            result["receiver_binding_validation_seconds"] += padding_receipt["receiver_binding_validation_seconds"]
            result["padding_only_control_fine_rows"] += padding_execution["fine_rows"]
            result["padding_only_control_near_rows"] += padding_execution["near_rows"]
            result["padding_only_control_full_far_rows"] += padding_execution["full_far_rows"]
            result["padding_only_control_pre_mlp_subset_calls"] += int(
                padding_receipt["used_pre_mlp_subset_gather"]
            )
            if allocated_before is not None:
                extra_peak = max(
                    0,
                    int(torch.cuda.max_memory_allocated(read.receivers.device)) - allocated_before,
                )
                padding_receipt["torch_peak_extra_allocated_bytes"] = extra_peak
                result["padding_only_control_peak_extra_allocated_bytes_max"] = max(
                    result["padding_only_control_peak_extra_allocated_bytes_max"], extra_peak
                )
                result["padding_only_control_peak_extra_allocated_bytes_measured_calls"] += 1
            else:
                padding_receipt["torch_peak_extra_allocated_bytes"] = None
            modes["padding_only_control"] = {"prediction": padding_prediction, "receipt": padding_receipt}
            full = modes["full"]["prediction"]
            torch.testing.assert_close(padding_prediction, full)
            task_result[read_name] = {
                "modes": modes,
                "experimental_model_distortion_rms": float(
                    torch.sqrt((modes["packet_experimental"]["prediction"] - full).square().mean())
                ),
                "same_exact_baseline": all(
                    mode_value["receipt"]["exact_baseline_reused"] for mode_value in modes.values()
                ),
                "padding_only_model_distortion_rms": float((padding_prediction - full).square().mean().sqrt()),
            }
        result["tasks"][task.task_id] = task_result
    return result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-train-table", help="derive exact sensitivity labels from primary TRAIN K")
    build.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ: pair_features, teacher_kernels, patch_weights, action_radii, layout_ids, partition_labels, source_ids_by_layout, source_present, teacher_checkpoint_id, manifest_sha256",
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--authority-profile", choices=("fixed25_v1",), required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--selection-receipt", type=Path, required=True)
    build.add_argument("--selection-rule", type=Path, required=True)
    build.add_argument("--expected-selection-rule-sha256", required=True)
    build.add_argument("--checkpoint", type=Path, required=True)
    fit = subparsers.add_parser("fit", help="fit one TRAIN-only packet pair scorer")
    fit.add_argument("--table", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True, help="Output model `.pt` under diagnostics/generated/")
    fit.add_argument("--authority-profile", choices=("fixed25_v1",), required=True)
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--selection-receipt", type=Path, required=True)
    fit.add_argument("--selection-rule", type=Path, required=True)
    fit.add_argument("--expected-selection-rule-sha256", required=True)
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=500)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--hidden-width", type=int, default=32)
    fit.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.command == "build-train-table":
        with np.load(args.input, allow_pickle=False) as arrays:
            layout_ids = arrays["layout_ids"].astype(str).tolist()
            partition_labels = arrays["partition_labels"].astype(str).tolist()
            teacher_checkpoint_id = str(arrays["teacher_checkpoint_id"].item())
            manifest_sha256 = str(arrays["manifest_sha256"].item())
            authority = validate_fixed25_train_authority(
                manifest_path=args.manifest,
                selection_receipt_path=args.selection_receipt,
                selection_rule_path=args.selection_rule,
                checkpoint_path=args.checkpoint,
                layout_ids=layout_ids,
                partition_labels=partition_labels,
                teacher_checkpoint_id=teacher_checkpoint_id,
                expected_selection_rule_sha256=args.expected_selection_rule_sha256,
            )
            if manifest_sha256 != authority["manifest_sha256"]:
                raise ValueError("Teacher NPZ manifest identity differs from authoritative fixed25_v1 bytes")
            table = build_train_teacher_table(
                pair_features=torch.from_numpy(arrays["pair_features"].copy()),
                teacher_kernels=torch.from_numpy(arrays["teacher_kernels"].copy()),
                patch_weights=torch.from_numpy(arrays["patch_weights"].copy()),
                action_radii=torch.from_numpy(arrays["action_radii"].copy()),
                layout_ids=layout_ids,
                partition_labels=partition_labels,
                source_ids_by_layout=arrays["source_ids_by_layout"].astype(str).tolist(),
                source_present=torch.from_numpy(arrays["source_present"].copy()),
                teacher_checkpoint_id=teacher_checkpoint_id,
                manifest_sha256=manifest_sha256,
            )
        save_train_teacher_table(table, args.output)
        print(
            json.dumps(
                {
                    "rows": int(table.raw_sensitivities.numel()),
                    "layouts": table.selected_layout_count,
                    "teacher_checkpoint_id": table.teacher_checkpoint_id,
                    "manifest_sha256": table.manifest_sha256,
                    "partition": "TRAIN",
                    "authority": authority,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "fit":
        if not 1 <= args.epochs <= 500:
            raise ValueError("The fixed receiver-packet development fit is limited to 1..500 epochs")
        table = load_train_teacher_table(args.table)
        authority = validate_fixed25_train_authority(
            manifest_path=args.manifest,
            selection_receipt_path=args.selection_receipt,
            selection_rule_path=args.selection_rule,
            checkpoint_path=args.checkpoint,
            layout_ids=tuple(dict.fromkeys(table.layout_ids)),
            partition_labels=("TRAIN",) * len(set(table.layout_ids)),
            teacher_checkpoint_id=table.teacher_checkpoint_id,
            expected_selection_rule_sha256=args.expected_selection_rule_sha256,
        )
        if table.manifest_sha256 != authority["manifest_sha256"]:
            raise ValueError("Fitter table manifest identity differs from authoritative fixed25_v1 bytes")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            scorer = ReceiverPairScorer(table.pair_features.shape[-1], args.hidden_width).to(args.device)
        receipt = fit_receiver_pair_scorer(
            scorer,
            table.pair_features,
            table.raw_sensitivities,
            table.layout_ids,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            review=lambda epoch, loss: print(json.dumps({"epoch": epoch, "train_loss": loss}), flush=True),
        )
        output = Path(args.output).resolve()
        generated_root = (ROOT / "diagnostics/generated").resolve()
        if not output.is_relative_to(generated_root):
            raise ValueError("Packet model output must stay under diagnostics/generated/")
        if output.exists() or output.with_suffix(".json").exists():
            raise FileExistsError("Preserve prior packet proposer outputs; choose a new output identity")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_width": scorer.input_width,
                "hidden_width": scorer.hidden_width,
                "state_dict": {key: value.detach().cpu() for key, value in scorer.state_dict().items()},
            },
            output,
        )
        summary = {
            "teacher_checkpoint_id": table.teacher_checkpoint_id,
            "train_manifest_sha256": table.manifest_sha256,
            "authority": authority,
            "selected_train_layout_count": table.selected_layout_count,
            "partition": "TRAIN",
            "dev_used_for_fit": False,
            "epochs": receipt.epochs,
            "seed": receipt.seed,
            "parameter_count": receipt.parameter_count,
            "optimizer_updates": receipt.optimizer_updates,
            "training_layout_ids": list(receipt.training_layout_ids),
            "calibration_layout_ids": list(receipt.calibration_layout_ids),
            "train_only_log_floor": receipt.train_only_log_floor,
            "train_only_residual_margin": receipt.train_only_residual_margin,
            "calibration_max_positive_residual": receipt.calibration_max_positive_residual,
            "fit_losses": list(receipt.fit_losses),
            "calibration_scope": receipt.calibration_scope,
        }
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": str(output),
                    **{
                        key: summary[key]
                        for key in (
                            "epochs",
                            "parameter_count",
                            "optimizer_updates",
                            "train_only_log_floor",
                            "train_only_residual_margin",
                            "selected_train_layout_count",
                        )
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
