"""Fresh paired learning run for the reusable Wind directed MM packet scorer.

The cohort lock is written from compact geometry before any development
field errors are gathered. Training begins from the exact Run2112 G u4910
checkpoint; W-dir and W-summary share physical weights, organizer seed,
layout/direction/query stream, and native role objective.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CaseLocalReceiverTree,
    MechanismPlan,
)
from honf_forward_core.interface_fields.budgeted_frontier import enumerate_frontier_cuts
from honf_forward_core.interface_fields.directional_packets import (
    DirectionalPacketOrganizer,
    DirectionalPacketScores,
    _transform_coordinates,
)
from honf_forward_core.interface_fields.native_joint_shadow import hard_value_soft_organizer_forward
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.paths import resolve_path

from ..data import COMPACT_GEOMETRY_KEYS, WindFarmNativeView
from ..model import WindFarmForwardModel
from ..splits import GroupSplit, make_group_split
from .evaluate_forward import _compact_metadata, load_checkpoint
from .joint_forward import (
    NativeRoleCatalogueCache,
    _batch_from_sample,
    _inverse_standardized_tensor,
    _native_rng,
    sample_native_role_queries,
)
from .native_cover_panel import (
    ORGANIZER_DEVELOPMENT_LAYOUT_INDICES,
    TrainingLayout,
    select_training_layouts,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CASE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "diagnostics/generated/directed_stable_20261002/wind"
SOURCE_RUN = (
    REPO_ROOT / "Trained_Results/WindFarm/HONF_Forward_Runs/Run_2112_controlled_maturation_20260929" / "arms/g_packet"
)
SOURCE_CHECKPOINT_SIDECAR = SOURCE_RUN / "latest_checkpoint.json"
ANCHOR_CALIBRATION_PATH = SOURCE_RUN.parent.parent / "anchor_calibration.json"
ANCHOR_TEACHER_CHECKPOINT = (
    REPO_ROOT
    / "Trained_Results/WindFarm/HONF_Forward_Runs/Run_2110_20260927_164943_windfarm_joint_forward_maturation"
    / "arms/w_full/checkpoints/updates_001500.pt"
)
ACTIVE_REUSE_CONFIG = CASE_ROOT / "configs/active_packet_organizer_reuse.json"
ROLE_NAMES = ("volume", "hub_slab", "downstream_envelope", "near_turbine", "background")
ROLE_QUERY_COUNTS = {
    "volume": 384,
    "hub_slab": 32,
    "downstream_envelope": 32,
    "near_turbine": 32,
    "background": 32,
}
TRAINING_SEED = 20261002
ORG_SEED = 821102
MAX_PHYSICAL_LR = 1.0e-5
SCORER_LR = 1.0e-3


def _parse_utc_deadline(value: str) -> datetime:
    """Parse an explicit ISO-8601 deadline and normalize it to UTC."""

    try:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("deadline must be an ISO-8601 timestamp") from error
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise argparse.ArgumentTypeError("deadline must include a timezone")
    return deadline.astimezone(timezone.utc)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _check_training_deadline(
    phase: str,
    update: int,
    stop_after_utc: datetime | None = None,
) -> None:
    if stop_after_utc is None:
        return
    now = datetime.now(timezone.utc)
    if stop_after_utc.tzinfo is None or stop_after_utc.utcoffset() is None:
        raise ValueError("training stop deadline must be timezone-aware")
    stop_after_utc = stop_after_utc.astimezone(timezone.utc)
    if now >= stop_after_utc:
        raise RuntimeError(
            f"Wind {phase} update {update} not started: authorized training/sampling stop "
            f"was {stop_after_utc.isoformat()}"
        )


def _trim_history_to_checkpoint(path: Path, scorer_updates: int, coadapt_updates: int) -> None:
    """Archive post-checkpoint attempt rows before rewriting the active history."""

    if not path.is_file():
        return
    original_lines = path.read_text(encoding="utf-8").splitlines()
    retained: list[str] = []
    discarded: list[str] = []
    for line in original_lines:
        record = json.loads(line)
        phase = str(record.get("phase", ""))
        if phase in {"scorer_only_restricted", "stage1_readonly_full_access_check"}:
            if int(record.get("scorer_update", 0)) <= int(scorer_updates):
                retained.append(line)
            else:
                discarded.append(line)
        elif phase.startswith("coadapt_"):
            if int(record.get("coadapt_update", 0)) <= int(coadapt_updates):
                retained.append(line)
            else:
                discarded.append(line)
        else:
            retained.append(line)
    if not discarded:
        return
    archive_path = path.with_name("discarded_resume_updates.jsonl")
    with archive_path.open("a", encoding="utf-8") as stream:
        stream.write("".join(f"{line}\n" for line in discarded))
        stream.flush()
        os.fsync(stream.fileno())
    _json_append(
        path.with_name("resume_attempts.jsonl"),
        {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_history": path.name,
            "checkpoint_scorer_updates": int(scorer_updates),
            "checkpoint_coadapt_updates": int(coadapt_updates),
            "discarded_row_count": len(discarded),
            "discarded_rows_path": archive_path.name,
        },
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(f"{line}\n" for line in retained), encoding="utf-8")
    os.replace(temporary, path)


def _cache_statistics(cache: NativeRoleCatalogueCache) -> dict[str, int]:
    stats = cache.summary()
    keys = (
        "catalogue_count",
        "catalogue_build_count",
        "cache_hit_count",
        "cache_miss_count",
        "cache_eviction_count",
        "cached_bytes",
        "peak_cached_bytes",
    )
    return {key: int(stats[key]) for key in keys}


def _load_role_scales() -> dict[str, float]:
    """Read the fixed scales declared by the maintained Wind role objective."""

    config = json.loads(ACTIVE_REUSE_CONFIG.read_text(encoding="utf-8"))
    raw = config["forward"]["stage_a"]["role_loss_scales_mps"]
    scales = {str(name): float(value) for name, value in raw.items()}
    if set(scales) != set(ROLE_NAMES) or any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError("maintained Wind stage-A role scales are incomplete or invalid")
    return scales


def _load_split(view: WindFarmNativeView) -> GroupSplit:
    groups = np.asarray(view.metadata["layout_index"], dtype=np.int64)
    split = make_group_split(groups, seed=42)
    stored_path = resolve_path("project://Case_WindFarm/Dataset/derived/forward_velocity_v1/split_indices.npz")
    if stored_path.is_file():
        with np.load(stored_path, allow_pickle=False) as archive:
            if not all(name in archive.files for name in ("train", "validation", "test")):
                raise ValueError("stored seed-42 split is missing a partition")
            for name in ("train", "validation", "test"):
                if not np.array_equal(archive[name], getattr(split, name)):
                    raise ValueError("stored native split differs from the documented seed-42 group split")
    return split


def _layout_record(layout: TrainingLayout) -> dict[str, Any]:
    return {
        "layout_index": int(layout.layout_index),
        "rows_direction_order": [int(row) for row in layout.rows],
        "turbine_count": int(layout.turbine_count),
        "geometry_feature_vector": [float(value) for value in layout.feature_vector],
    }


def _predeclared_field_slices(
    view: WindFarmNativeView,
    layouts: tuple[TrainingLayout, ...],
) -> list[dict[str, Any]]:
    features = np.asarray([item.feature_vector for item in layouts], dtype=np.float64)
    center = np.median(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    distances = np.sum(((features - center) / scale) ** 2, axis=1)
    representative = layouts[int(np.argmin(distances))]
    large_order = sorted(
        layouts,
        key=lambda item: (
            -int(item.turbine_count),
            -float(item.feature_vector[3]),
            -float(item.feature_vector[1] * item.feature_vector[2]),
            int(item.layout_index),
        ),
    )
    difficult = next(item for item in large_order if item.layout_index != representative.layout_index)
    selections = (("representative_median_geometry", representative), ("largest_m_metadata", difficult))
    result: list[dict[str, Any]] = []
    for label, layout in selections:
        row = next(int(row) for row in layout.rows if float(view.metadata["wd_deg"][int(row)]) == 285.0)
        run = view.volume.run(row)
        z_D = np.asarray(run.z_m, dtype=np.float64) / float(view.diameter_m)
        z_index = int(np.argmin(np.abs(z_D - 0.875)))
        result.append(
            {
                "selection_label": label,
                "layout_index": int(layout.layout_index),
                "row_index": row,
                "wind_direction_deg": 285.0,
                "turbine_count": int(layout.turbine_count),
                "geometry_feature_vector": [float(value) for value in layout.feature_vector],
                "z_index": z_index,
                "z_D": float(z_D[z_index]),
                "selection_basis": "development compact geometry only; no new field errors inspected",
            }
        )
    return result


def load_or_lock_cohort(output: Path) -> tuple[dict[str, Any], WindFarmNativeView, GroupSplit]:
    """Freeze whole-layout identities and two field slices before dev outcomes."""

    compact_path = resolve_path("project://Case_WindFarm/Dataset/links/wind_farm/family_tensor.npz")
    volume_path = resolve_path("project://Case_WindFarm/Dataset/links/wind_farm/family_volume")
    compact = _compact_metadata(compact_path)
    missing = sorted(set(COMPACT_GEOMETRY_KEYS) - set(compact))
    if missing:
        raise ValueError(f"WindFarm compact metadata lacks {missing}")
    view = WindFarmNativeView(volume_path, compact_metadata=compact)
    split = _load_split(view)
    train_rows = np.asarray(
        [
            row
            for row in split.train
            if int(view.metadata["layout_index"][row]) not in ORGANIZER_DEVELOPMENT_LAYOUT_INDICES
        ],
        dtype=np.int64,
    )
    training = select_training_layouts(view, train_rows, count=30)
    development = select_training_layouts(view, split.validation, count=12)
    established = set(ORGANIZER_DEVELOPMENT_LAYOUT_INDICES)
    if established.intersection(item.layout_index for item in training):
        raise RuntimeError("established organizer-development layouts entered the training cohort")
    cohort_path = output / "cohort_lock.json"
    role_scales = _load_role_scales()
    if cohort_path.is_file():
        locked = json.loads(cohort_path.read_text(encoding="utf-8"))
        if len(locked.get("training_layouts", [])) < 30 or len(locked.get("development_layouts", [])) < 12:
            raise ValueError("existing cohort lock does not meet the 30/12 layout contract")
        if locked.get("role_objective_scales_mps") not in (None, role_scales):
            raise ValueError("existing cohort lock uses different fixed role-objective scales")
        source_payload = load_trusted_checkpoint(Path(locked["source_checkpoint"]), map_location="cpu")
        base_model_checkpoint = Path(source_payload["source_checkpoint"]).resolve()
        if base_model_checkpoint != ANCHOR_TEACHER_CHECKPOINT.resolve():
            raise ValueError("locked Run2112 source does not point to the retained Run2110 W-full u1500 base")
        changed = False
        if locked.get("base_model_checkpoint") != str(base_model_checkpoint):
            locked["base_model_checkpoint"] = str(base_model_checkpoint)
            changed = True
        if "role_objective_scales_mps" not in locked:
            locked["role_objective"] = "equal mean of five role MSEs divided by declared fixed scale squared"
            locked["role_objective_scales_mps"] = role_scales
            locked["role_objective_config"] = str(ACTIVE_REUSE_CONFIG)
            changed = True
        if changed:
            _json_write(cohort_path, locked)
        return locked, view, split
    source_sidecar = json.loads(SOURCE_CHECKPOINT_SIDECAR.read_text(encoding="utf-8"))
    source_payload = load_trusted_checkpoint(Path(source_sidecar["checkpoint"]), map_location="cpu")
    base_model_checkpoint = Path(source_payload["source_checkpoint"]).resolve()
    anchor_calibration = json.loads(ANCHOR_CALIBRATION_PATH.read_text(encoding="utf-8"))
    if anchor_calibration.get("incumbent_identity") != "exact retained Run2110 W-full u1500":
        raise ValueError("stored full-anchor calibration is not the retained Run2110 W-full u1500 contract")
    if not ANCHOR_TEACHER_CHECKPOINT.is_file():
        raise FileNotFoundError(f"fixed Run2110 W-full u1500 teacher is missing: {ANCHOR_TEACHER_CHECKPOINT}")
    if base_model_checkpoint != ANCHOR_TEACHER_CHECKPOINT.resolve():
        raise ValueError("Run2112 G-u4910 was not initialized from the retained Run2110 W-full u1500 source")
    slices = _predeclared_field_slices(view, development)
    lock = {
        "schema_version": 1,
        "run_id": "directed_stable_20261002_wind",
        "selected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_basis": "seed-42 official grouped split and compact geometry only; frozen before new dev errors",
        "source_checkpoint": str(source_sidecar["checkpoint"]),
        "source_update": int(source_sidecar["update_count"]),
        "base_model_checkpoint": str(base_model_checkpoint),
        "source_anchor_teacher_checkpoint": str(ANCHOR_TEACHER_CHECKPOINT),
        "source_full_anchor_coefficient": float(anchor_calibration["anchor_weight"]),
        "source_full_anchor_coefficient_field": str(ANCHOR_CALIBRATION_PATH),
        "source_full_anchor_calibration": {
            "incumbent_identity": str(anchor_calibration["incumbent_identity"]),
            "training_only": bool(anchor_calibration["training_only"]),
            "calibration_rule": str(anchor_calibration["calibration_rule"]),
            "normalized_fixed_incumbent_anchor_loss": float(
                anchor_calibration["normalized_fixed_incumbent_anchor_loss"]
            ),
        },
        "train_layout_selection": "farthest-point coverage after robust geometry standardization over complete eligible training layouts",
        "development_layout_selection": "same metadata-only farthest-point coverage over official validation layouts",
        "split_seed": 42,
        "training_layouts": [_layout_record(item) for item in training],
        "development_layouts": [_layout_record(item) for item in development],
        "training_layout_indices": [int(item.layout_index) for item in training],
        "development_layout_indices": [int(item.layout_index) for item in development],
        "established_organizer_development_layouts_excluded_from_training": sorted(established),
        "paired_native_rows": {
            "training": [int(row) for item in training for row in item.rows],
            "development": [int(row) for item in development for row in item.rows],
        },
        "predeclared_field_slices": slices,
        "field_frame": "WindFarm stored coordinates are already wind aligned; no second rotation",
        "physical_length_scale": {"value": 1.0, "unit": "rotor diameters", "source": "D=80 m dataset coordinate frame"},
        "role_objective": "equal mean of five role MSEs divided by declared fixed scale squared",
        "role_objective_scales_mps": role_scales,
        "role_objective_config": str(ACTIVE_REUSE_CONFIG),
    }
    _json_write(cohort_path, lock)
    return lock, view, split


def _save_arm(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _physical_parameters(core: Any) -> tuple[list[torch.nn.Parameter], list[str]]:
    selected: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, parameter in core.named_parameters():
        if "mm_message" in name or "module_update" in name:
            selected.append(parameter)
            names.append(name)
    if not selected:
        raise RuntimeError("the source core has no named MM message/module-update parameters")
    return selected, names


def _role_loss(
    prediction_std: torch.Tensor,
    target_std: torch.Tensor,
    role_slices: dict[str, slice],
    normalizer: Any,
    role_scales: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = _inverse_standardized_tensor(prediction_std, normalizer)
    target = _inverse_standardized_tensor(target_std, normalizer)
    return _role_objective_from_physical(prediction, target, role_slices, role_scales)


def _role_objective_from_physical(
    prediction: torch.Tensor,
    target: torch.Tensor,
    role_slices: dict[str, slice],
    role_scales: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    errors: dict[str, torch.Tensor] = {}
    for role in ROLE_NAMES:
        error = prediction[:, role_slices[role]] - target[:, role_slices[role]]
        errors[role] = error.square().mean()
    loss = torch.stack([errors[role] / (role_scales[role] ** 2) for role in ROLE_NAMES]).mean()
    return loss, {role: float(errors[role].detach().cpu()) for role in ROLE_NAMES}


def _full_access_prediction(model: WindFarmForwardModel, encoded: Any, batch: Any) -> tuple[torch.Tensor, Any]:
    core = model.core
    trees = core.backend.build_case_trees(encoded)
    if len(trees) != 1:
        raise ValueError("directed Wind training expects one native case per update")
    plan = MechanismPlan.full_access(trees[0], encoded.module_present[0], int(encoded.env_coords.shape[1]))
    prepared = core.prepare(encoded, encoded.module_tokens, fixed_cover_plans=(plan,))
    prediction = core.decode_queries(
        prepared,
        batch.query_xy,
        query_features=batch.query_features,
        receiver_chunk_size=512,
    )["pred_field"]
    return prediction, prepared


def _ledger(prepared: Any) -> dict[str, int]:
    state = prepared.backend_state
    values = state.get("cover_preparation_ledger", {}) if isinstance(state, dict) else {}
    return {str(key): int(value) for key, value in values.items() if str(key).startswith("cover_prepare_")}


def _action_signature(matrix: torch.Tensor) -> tuple[float, ...]:
    values = matrix.detach().clone()
    if values.shape[0] == values.shape[1]:
        values.fill_diagonal_(0.0)
    return tuple(float(value) for value in values.flatten().cpu().tolist())


def _action_descriptor(
    organizer: DirectionalPacketOrganizer,
    scores: tuple[DirectionalPacketScores, ...],
    encoded: Any,
    base_tree: CaseLocalReceiverTree,
    cut: tuple[int, ...],
) -> tuple[MechanismPlan, MechanismPlan, dict[str, Any]]:
    local_tree = scores[0].receiver_tree
    if local_tree is None:
        raise RuntimeError("MM-local score omitted its receiver tree")
    local_plan = organizer._local_permission_plans(
        scores,
        encoded,
        hard=True,
        frontier_cuts=(cut,),
        budget_fractions={"MM": 0.9},
    )[0]
    compiled = organizer._compile_local_mm_plan(scores[0], encoded, base_tree, 0, local_plan)
    frame = None if organizer._physical_frame.numel() == 0 else organizer._physical_frame
    local_receiver_coords = _transform_coordinates(encoded.module_centers[0], frame)
    access = local_tree.access(local_receiver_coords, local_plan.split_gates)
    membership = local_plan.permission_matrix("MM")
    active = encoded.module_present[0] > 0.5
    pair_union = torch.zeros((int(active.numel()), int(active.numel())), device=active.device, dtype=torch.bool)
    nonredundant_packets = 0
    for node in cut:
        contribution = (
            (access[:, node] > 0)[:, None] & (membership[node] > 0)[None, :] & active[:, None] & active[None, :]
        )
        contribution.fill_diagonal_(False)
        novel = contribution & ~pair_union
        if bool(novel.any()):
            nonredundant_packets += 1
        pair_union |= contribution
    direct = compiled.direct_pair_access_for("MM")
    if direct is None:
        raise RuntimeError("compiled plan is missing exact MM direct-pair permissions")
    matrix = direct.weights * active.to(direct.weights.dtype)[:, None]
    matrix = matrix * active.to(direct.weights.dtype)[None, :]
    matrix = matrix.clone()
    matrix.fill_diagonal_(0.0)
    effective_counts = (matrix > 0).sum(dim=1)
    details = {
        "requested_cut_nodes": [int(index) for index in cut],
        "requested_cut_K": len(cut),
        "realized_nonredundant_MM_packet_count": int(nonredundant_packets),
        "MM_pair_permission_unique_pairs": int((matrix > 0).sum().detach().cpu()),
        "MM_source_union_count": int((matrix > 0).any(dim=0).sum().detach().cpu()),
        "MM_receiver_rows_with_sources": int((effective_counts > 0).sum().detach().cpu()),
        "MM_source_counts_per_receiver": [int(value) for value in effective_counts.cpu().tolist()],
        "MM_eligible_nonself_pairs": int(active.sum().item() * (active.sum().item() - 1)),
        "exact_action_signature": _action_signature(matrix),
    }
    return local_plan, compiled, details


def _select_distinct_action(
    organizer: DirectionalPacketOrganizer,
    encoded: Any,
    base_tree: CaseLocalReceiverTree,
    update: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    score_inputs = {
        "module_states": encoded.module_tokens,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
    }
    scores = organizer.score_cases(encoded, score_inputs, (base_tree,))
    local_tree = scores[0].receiver_tree
    if local_tree is None:
        raise RuntimeError("MM-local score omitted its receiver tree")
    candidates = enumerate_frontier_cuts(local_tree, max_depth=2)
    unique: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    signatures: set[tuple[float, ...]] = set()
    duplicate_cuts: list[dict[str, Any]] = []
    for cut in candidates:
        local_plan, _compiled, detail = _action_descriptor(organizer, scores, encoded, base_tree, tuple(cut))
        del local_plan
        signature = detail.pop("exact_action_signature")
        if signature in signatures:
            duplicate_cuts.append({"cut_nodes": [int(index) for index in cut], "duplicate": True})
        else:
            signatures.add(signature)
            detail["exact_action_signature"] = signature
            unique.append((tuple(cut), detail))
    if not unique:
        raise RuntimeError("all bounded MM cuts collapsed to no native action")
    requested_index = int(((update - 1) // 30) % max(len(candidates), 1))
    selected_index = requested_index % len(unique)
    selected_cut, selected_detail = unique[selected_index]
    return selected_cut, {
        **selected_detail,
        "requested_action_index": requested_index,
        "selected_distinct_action_index": selected_index,
        "candidate_cut_count": len(candidates),
        "distinct_exact_action_count": len(unique),
        "duplicate_cut_nodes": duplicate_cuts,
        "selected_cut_nodes": [int(index) for index in selected_cut],
    }


def _sample_for_update(
    view: WindFarmNativeView,
    layout: TrainingLayout,
    direction_position: int,
    normalizer: Any,
    device: torch.device,
    *,
    update: int,
    cache: NativeRoleCatalogueCache,
) -> tuple[Any, Any, int, Any]:
    row = int(layout.rows[int(direction_position) % len(layout.rows)])
    case = view.run(row)
    counts = ROLE_QUERY_COUNTS
    sample = sample_native_role_queries(
        case,
        _native_rng(TRAINING_SEED, int(update), row, 87),
        counts,
        catalogue_cache=cache,
    )
    batch = _batch_from_sample(case, sample, normalizer, device)
    return case, sample, row, batch


def _new_arm(
    arm: str,
    feature_mode: str,
    materialization_batch: Any,
    device: torch.device,
    checkpoint_path: Path,
    teacher_checkpoint_path: Path,
    latest_path: Path,
    history_path: Path,
) -> tuple[WindFarmForwardModel, DirectionalPacketOrganizer, Any, Any, Any, Any, list[str], dict[str, Any]]:
    torch.manual_seed(ORG_SEED)
    np.random.seed(ORG_SEED)
    source_state = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    if source_state.get("run_id") != "2112" or source_state.get("arm") != "g_packet":
        raise ValueError("Wind paired initialization is not the exact Run2112 G-u4910 state")
    if int(source_state.get("update_count", -1)) != 4910:
        raise ValueError("Wind paired initialization must use Run2112 G update 4910")
    if Path(source_state.get("source_checkpoint", "")).resolve() != Path(teacher_checkpoint_path).resolve():
        raise ValueError("Run2112 G checkpoint and fixed Run2110 anchor source no longer match")
    model, _base_payload = load_checkpoint(
        teacher_checkpoint_path,
        device=device,
        materialization_batch=materialization_batch,
    )
    model.load_state_dict(source_state["model_state_dict"], strict=True)
    if model.config.forward_architecture != "dense_pairwise_field":
        raise ValueError("directed-packet training requires the exact Run2112 Dense pairwise core")
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    teacher, _teacher_payload = load_checkpoint(
        teacher_checkpoint_path,
        device=device,
        materialization_batch=materialization_batch,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    organizer = DirectionalPacketOrganizer(
        state_dim=int(model.config.hidden_dim),
        module_feature_dim=int(materialization_batch.module_features.shape[-1]),
        environment_feature_dim=int(materialization_batch.env_features.shape[-1]),
        hidden_dim=96,
        feature_mode=feature_mode,
        geometry_length_scale=1.0,
        scorer_lr=SCORER_LR,
    ).to(device)
    score_parameters = [parameter for parameter in organizer.parameters() if parameter.requires_grad]
    score_optimizer = torch.optim.AdamW(score_parameters, lr=SCORER_LR, weight_decay=0.0)
    physical_parameters, physical_names = _physical_parameters(model.core)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state: dict[str, Any] | None = None
    if latest_path.is_file():
        state = load_trusted_checkpoint(latest_path, map_location="cpu")
        if state.get("arm") != arm:
            raise ValueError(f"saved arm file {latest_path} belongs to {state.get('arm')!r}")
        model.load_state_dict(state["model_state_dict"], strict=True)
        organizer.load_state_dict(state["organizer_state_dict"], strict=True)
        score_optimizer.load_state_dict(state["score_optimizer_state_dict"])
    completed_scorer = int(0 if state is None else state.get("scorer_updates", 0))
    completed_coadapt = int(0 if state is None else state.get("coadapt_updates", 0))
    physical_optimizer = None
    if completed_coadapt > 0:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in physical_parameters:
            parameter.requires_grad_(True)
        physical_optimizer = torch.optim.AdamW(physical_parameters, lr=MAX_PHYSICAL_LR, weight_decay=0.0)
        if state is not None and state.get("physical_optimizer_state_dict") is not None:
            physical_optimizer.load_state_dict(state["physical_optimizer_state_dict"])
    summary = {
        "arm": arm,
        "feature_mode": feature_mode,
        "source_checkpoint": str(checkpoint_path),
        "source_update": 4910,
        "scorer_trainable_parameter_count": int(sum(p.numel() for p in score_parameters)),
        "physical_trainable_parameter_count": int(sum(p.numel() for p in physical_parameters)),
        "physical_trainable_parameter_names": physical_names,
        "physical_scope_note": "mm_message plus shared module_update; module_update also consumes ME context; all other physical parameters frozen",
        "scorer_updates": completed_scorer,
        "coadapt_updates": completed_coadapt,
    }
    return model, organizer, teacher, score_optimizer, physical_optimizer, state, physical_names, summary


def _checkpoint_state(
    arm: str,
    model: WindFarmForwardModel,
    organizer: DirectionalPacketOrganizer,
    score_optimizer: Any,
    physical_optimizer: Any,
    *,
    scorer_updates: int,
    coadapt_updates: int,
    restricted_count: int,
    full_replay_count: int,
    readonly_full_check_count: int,
    action_exposure: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm": arm,
        "scorer_updates": int(scorer_updates),
        "coadapt_updates": int(coadapt_updates),
        "actual_optimizer_updates": int(scorer_updates + coadapt_updates),
        "restricted_optimizer_updates": int(restricted_count),
        "coadapt_full_access_optimizer_updates": int(full_replay_count),
        "stage1_readonly_full_checks": int(readonly_full_check_count),
        "model_state_dict": model.state_dict(),
        "organizer_state_dict": organizer.state_dict(),
        "score_optimizer_state_dict": score_optimizer.state_dict(),
        "physical_optimizer_state_dict": None if physical_optimizer is None else physical_optimizer.state_dict(),
        "action_exposure": action_exposure,
    }


def _run_arm(
    *,
    arm: str,
    feature_mode: str,
    lock: dict[str, Any],
    view: WindFarmNativeView,
    output: Path,
    scorer_target: int,
    coadapt_target: int,
    pilot: bool,
    device: torch.device,
    stop_after_utc: datetime | None,
) -> dict[str, Any]:
    checkpoint_path = Path(lock["source_checkpoint"])
    source = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    base_source = load_trusted_checkpoint(Path(source["source_checkpoint"]), map_location="cpu")
    from ..normalization import VelocityNormalizer

    normalizer = VelocityNormalizer.from_dict(dict(base_source["normalization"]))
    layouts = tuple(
        TrainingLayout(
            int(item["layout_index"]),
            tuple(int(row) for row in item["rows_direction_order"]),
            int(item["turbine_count"]),
            tuple(float(value) for value in item["geometry_feature_vector"]),
        )
        for item in lock["training_layouts"]
    )
    preflight_case = view.run(layouts[0].rows[0])
    preflight_sample = sample_native_role_queries(
        preflight_case,
        _native_rng(TRAINING_SEED, 0, preflight_case.index, 87),
        ROLE_QUERY_COUNTS,
        catalogue_cache=NativeRoleCatalogueCache(max_cached_bytes=2 * 1024**3),
    )
    preflight_batch = _batch_from_sample(preflight_case, preflight_sample, normalizer, device)
    arm_dir = output / arm
    latest_path = arm_dir / "latest.pt"
    history_path = arm_dir / "updates.jsonl"
    model, organizer, teacher, score_optimizer, physical_optimizer, state, physical_names, arm_summary = _new_arm(
        arm,
        feature_mode,
        preflight_batch,
        device,
        checkpoint_path,
        Path(lock["source_anchor_teacher_checkpoint"]),
        latest_path,
        history_path,
    )
    completed_scorer = 0 if state is None else int(state.get("scorer_updates", 0))
    completed_coadapt = 0 if state is None else int(state.get("coadapt_updates", 0))
    restricted_count = 0 if state is None else int(state.get("restricted_optimizer_updates", 0))
    full_replay_count = 0 if state is None else int(state.get("coadapt_full_access_optimizer_updates", 0))
    readonly_full_check_count = 0 if state is None else int(state.get("stage1_readonly_full_checks", 0))
    action_exposure = {} if state is None else dict(state.get("action_exposure", {}))
    _trim_history_to_checkpoint(history_path, completed_scorer, completed_coadapt)
    role_scales = _load_role_scales()
    anchor_coefficient = float(lock["source_full_anchor_coefficient"])
    cache = NativeRoleCatalogueCache(max_cached_bytes=4 * 1024**3)
    arm_dir.mkdir(parents=True, exist_ok=True)
    _json_write(
        arm_dir / "arm_manifest.json",
        {
            **arm_summary,
            "role_objective": "equal mean of five role MSEs divided by declared fixed scale squared",
            "role_scales_mps": role_scales,
            "role_objective_config": str(ACTIVE_REUSE_CONFIG),
            "full_anchor_teacher": str(lock["source_anchor_teacher_checkpoint"]),
            "full_anchor_teacher_coefficient": anchor_coefficient,
            "full_anchor_coefficient_source": lock["source_full_anchor_coefficient_field"],
            "query_counts": ROLE_QUERY_COUNTS,
            "query_total": int(sum(ROLE_QUERY_COUNTS.values())),
            "source_measure": "native role sampler; role means; no new quadrature reweighting",
            "scorer_lr": SCORER_LR,
            "physical_lr": MAX_PHYSICAL_LR,
            "coadapt_scope": physical_names,
        },
    )
    if pilot:
        scorer_target = min(int(scorer_target), completed_scorer + 1)
        coadapt_target = completed_coadapt
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    model.train()
    organizer.train()
    # Stage 1: actual scorer learning uses restricted native MM plans. Every
    # third update also records a full-access read-only check on the same Q512.
    for scorer_update in range(completed_scorer + 1, scorer_target + 1):
        _check_training_deadline("scorer", scorer_update, stop_after_utc)
        layout = layouts[(scorer_update - 1) % len(layouts)]
        direction = ((scorer_update - 1) // (len(layouts) * 5)) % 3
        case, sample, row, batch = _sample_for_update(
            view,
            layout,
            direction,
            normalizer,
            device,
            update=scorer_update,
            cache=cache,
        )
        _check_training_deadline("scorer optimizer", scorer_update, stop_after_utc)
        del case
        organizer.set_update(scorer_update)
        score_optimizer.zero_grad(set_to_none=True)
        core = model.core
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        encoded = core.encode_case(batch)
        base_trees = core.backend.build_case_trees(encoded)
        cut, action = _select_distinct_action(organizer, encoded, base_trees[0], restricted_count + 1)
        bridge = hard_value_soft_organizer_forward(
            core,
            encoded,
            encoded.module_tokens,
            organizer,
            batch.query_xy,
            batch.query_features,
            receiver_chunk_size=512,
            collect_hard_aux=False,
            frontier_cuts=(cut,),
            budget_fractions={"MM": 0.9},
        )
        loss, role_mse = _role_loss(
            bridge.prediction,
            batch.target_field,
            dict(sample.role_slices),
            normalizer,
            role_scales,
        )
        if not loss.requires_grad:
            raise RuntimeError("restricted stage-1 loss has no scorer gradient")
        loss.backward()
        score_grad = float(
            torch.nn.utils.clip_grad_norm_([p for p in organizer.parameters() if p.requires_grad], max_norm=5.0)
            .detach()
            .cpu()
        )
        score_optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        ledger = _ledger(bridge.hard_prepared)
        record = {
            "phase": "scorer_only_restricted",
            "arm": arm,
            "scorer_update": scorer_update,
            "coadapt_update": 0,
            "actual_optimizer_update": scorer_update,
            "row_index": row,
            "layout_index": int(view.metadata["layout_index"][row]),
            "wind_direction_deg": float(view.metadata["wd_deg"][row]),
            "query_count": int(batch.query_xy.shape[1]),
            "query_counts_by_role": ROLE_QUERY_COUNTS,
            "loss_normalized": float(loss.detach().cpu()),
            "role_mse_mps2": role_mse,
            "scorer_gradient_norm_preclip": score_grad,
            "scorer_optimizer_step": True,
            "physical_optimizer_step": False,
            "step_seconds_after_sampling": step_seconds,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "residual_ramp": organizer.residual_ramp,
            "selected_action": {key: value for key, value in action.items() if key != "exact_action_signature"},
            "native_hard_preparation_ledger": ledger,
            "MM_eligible_valid_nonself_pairs": int(action["MM_eligible_nonself_pairs"]),
            "source_union_count": int(action["MM_source_union_count"]),
            "role_sampler_cache": _cache_statistics(cache),
        }
        restricted_count += 1
        exposure_key = f"layout{record['layout_index']}_cut{','.join(map(str, cut))}_K{action['realized_nonredundant_MM_packet_count']}"
        action_exposure[exposure_key] = int(action_exposure.get(exposure_key, 0)) + 1
        _json_append(history_path, record)
        if scorer_update % 50 == 0 or scorer_update == scorer_target or scorer_update == 1:
            saved = _checkpoint_state(
                arm,
                model,
                organizer,
                score_optimizer,
                None,
                scorer_updates=scorer_update,
                coadapt_updates=0,
                restricted_count=restricted_count,
                full_replay_count=full_replay_count,
                readonly_full_check_count=readonly_full_check_count,
                action_exposure=action_exposure,
            )
            _save_arm(latest_path, saved)
            _save_arm(arm_dir / f"scorer_update_{scorer_update:04d}.pt", saved)
        if scorer_update % 3 == 0:
            # Exact full-access replay has no scorer dependency. It is logged
            # as a read-only check and deliberately has no backward/optimizer call.
            with torch.no_grad():
                full_prediction, full_prepared = _full_access_prediction(model, encoded, batch)
                full_loss, full_roles = _role_loss(
                    full_prediction, batch.target_field, dict(sample.role_slices), normalizer, role_scales
                )
            readonly_full_check_count += 1
            _json_append(
                history_path,
                {
                    "phase": "stage1_readonly_full_access_check",
                    "arm": arm,
                    "scorer_update": scorer_update,
                    "coadapt_update": 0,
                    "optimizer_step": False,
                    "loss_normalized": float(full_loss.detach().cpu()),
                    "role_mse_mps2": full_roles,
                    "native_full_preparation_ledger": _ledger(full_prepared),
                    "note": "exact full access has no scorer gradient; no fabricated backward pass",
                },
            )
    scorer_updates_done = max(completed_scorer, scorer_target)
    # Stage 2: 700 real MM co-adaptation updates; one of each four is an
    # exact full-access replay that trains physical MM message/update weights
    # and adds the recorded Run2110 W-full u1500 anchor-teacher term.
    if scorer_updates_done >= scorer_target:
        physical_parameters, physical_names = _physical_parameters(model.core)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in physical_parameters:
            parameter.requires_grad_(True)
        physical_optimizer = torch.optim.AdamW(physical_parameters, lr=MAX_PHYSICAL_LR, weight_decay=0.0)
        if state is not None and state.get("physical_optimizer_state_dict") is not None:
            physical_optimizer.load_state_dict(state["physical_optimizer_state_dict"])
        teacher_parameters = [p for p in teacher.parameters()]
        for parameter in teacher_parameters:
            parameter.requires_grad_(False)
        for coadapt_update in range(completed_coadapt + 1, coadapt_target + 1):
            _check_training_deadline("co-adaptation", coadapt_update, stop_after_utc)
            global_update = scorer_updates_done + coadapt_update
            layout = layouts[(global_update - 1) % len(layouts)]
            direction = ((global_update - 1) // (len(layouts) * 5)) % 3
            case, sample, row, batch = _sample_for_update(
                view,
                layout,
                direction,
                normalizer,
                device,
                update=global_update,
                cache=cache,
            )
            _check_training_deadline("co-adaptation optimizer", coadapt_update, stop_after_utc)
            del case
            full_replay = coadapt_update % 4 == 0
            model.train()
            physical_optimizer.zero_grad(set_to_none=True)
            score_optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            encoded = model.core.encode_case(batch)
            action: dict[str, Any] = {"action_type": "full_access_replay"}
            if full_replay:
                prediction, prepared = _full_access_prediction(model, encoded, batch)
                with torch.no_grad():
                    teacher_encoded = teacher.core.encode_case(batch)
                    teacher_prediction, _teacher_prepared = _full_access_prediction(teacher, teacher_encoded, batch)
                supervised_loss, role_mse = _role_loss(
                    prediction, batch.target_field, dict(sample.role_slices), normalizer, role_scales
                )
                student_physical = _inverse_standardized_tensor(prediction, model.velocity_transform)
                teacher_physical = _inverse_standardized_tensor(teacher_prediction, teacher.velocity_transform)
                anchor_loss, _anchor_role_mse = _role_objective_from_physical(
                    student_physical, teacher_physical, dict(sample.role_slices), role_scales
                )
                loss = supervised_loss + anchor_coefficient * anchor_loss
                physical_optimizer.zero_grad(set_to_none=True)
                if not loss.requires_grad:
                    raise RuntimeError("full-access co-adaptation loss has no physical gradient")
                loss.backward()
                score_grad = 0.0
                physical_grad = float(torch.nn.utils.clip_grad_norm_(physical_parameters, 5.0).detach().cpu())
                physical_optimizer.step()
                torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - step_started
                prepared_for_ledger = prepared
                action.update(
                    {
                        "full_anchor_loss_normalized": float(anchor_loss.detach().cpu()),
                        "full_anchor_coefficient": anchor_coefficient,
                        "full_anchor_teacher_checkpoint": str(lock["source_anchor_teacher_checkpoint"]),
                    }
                )
                full_replay_count += 1
            else:
                model.core.backend.set_cover_mode("external")
                model.core.backend.set_cover_executor("dense_masked")
                organizer.set_update(restricted_count + 1)
                base_trees = model.core.backend.build_case_trees(encoded)
                cut, action = _select_distinct_action(organizer, encoded, base_trees[0], restricted_count + 1)
                bridge = hard_value_soft_organizer_forward(
                    model.core,
                    encoded,
                    encoded.module_tokens,
                    organizer,
                    batch.query_xy,
                    batch.query_features,
                    receiver_chunk_size=512,
                    collect_hard_aux=False,
                    frontier_cuts=(cut,),
                    budget_fractions={"MM": 0.9},
                )
                loss, role_mse = _role_loss(
                    bridge.prediction,
                    batch.target_field,
                    dict(sample.role_slices),
                    normalizer,
                    role_scales,
                )
                if not loss.requires_grad:
                    raise RuntimeError("restricted co-adaptation loss has no scorer/physical gradient")
                loss.backward()
                score_parameters = [p for p in organizer.parameters() if p.requires_grad]
                score_grad = float(torch.nn.utils.clip_grad_norm_(score_parameters, 5.0).detach().cpu())
                physical_grad = float(torch.nn.utils.clip_grad_norm_(physical_parameters, 5.0).detach().cpu())
                score_optimizer.step()
                physical_optimizer.step()
                prepared_for_ledger = bridge.hard_prepared
                restricted_count += 1
                exposure_key = f"layout{int(view.metadata['layout_index'][row])}_cut{','.join(map(str, cut))}_K{action['realized_nonredundant_MM_packet_count']}"
                action_exposure[exposure_key] = int(action_exposure.get(exposure_key, 0)) + 1
                torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - step_started
            if full_replay:
                pass
            coadapt_record = {
                "phase": "coadapt_full_access_replay" if full_replay else "coadapt_restricted",
                "arm": arm,
                "scorer_update": restricted_count,
                "coadapt_update": coadapt_update,
                "actual_optimizer_update": scorer_updates_done + coadapt_update,
                "row_index": row,
                "layout_index": int(view.metadata["layout_index"][row]),
                "wind_direction_deg": float(view.metadata["wd_deg"][row]),
                "query_count": int(batch.query_xy.shape[1]),
                "query_counts_by_role": ROLE_QUERY_COUNTS,
                "loss_normalized": float(loss.detach().cpu()),
                "role_mse_mps2": role_mse,
                "scorer_gradient_norm_preclip": score_grad,
                "physical_gradient_norm_preclip": physical_grad,
                "scorer_optimizer_step": not full_replay,
                "physical_optimizer_step": True,
                "step_seconds_after_sampling": step_seconds,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "residual_ramp": organizer.residual_ramp,
                "selected_action": {
                    key: value for key, value in action.items() if key not in {"exact_action_signature"}
                },
                "native_hard_preparation_ledger": _ledger(prepared_for_ledger),
                "MM_eligible_valid_nonself_pairs": int(
                    action.get(
                        "MM_eligible_nonself_pairs",
                        int(batch.module_present.sum().item()) * (int(batch.module_present.sum().item()) - 1),
                    )
                ),
                "source_union_count": int(action.get("MM_source_union_count", int(batch.module_present.sum().item()))),
                "physical_trainable_parameter_names": physical_names,
                "role_sampler_cache": _cache_statistics(cache),
            }
            _json_append(history_path, coadapt_record)
            if coadapt_update % 50 == 0 or coadapt_update == coadapt_target or coadapt_update == 1:
                saved = _checkpoint_state(
                    arm,
                    model,
                    organizer,
                    score_optimizer,
                    physical_optimizer,
                    scorer_updates=scorer_updates_done,
                    coadapt_updates=coadapt_update,
                    restricted_count=restricted_count,
                    full_replay_count=full_replay_count,
                    readonly_full_check_count=readonly_full_check_count,
                    action_exposure=action_exposure,
                )
                _save_arm(latest_path, saved)
                _save_arm(arm_dir / f"coadapt_update_{coadapt_update:04d}.pt", saved)
    saved = _checkpoint_state(
        arm,
        model,
        organizer,
        score_optimizer,
        physical_optimizer,
        scorer_updates=scorer_updates_done,
        coadapt_updates=coadapt_target,
        restricted_count=restricted_count,
        full_replay_count=full_replay_count,
        readonly_full_check_count=readonly_full_check_count,
        action_exposure=action_exposure,
    )
    _save_arm(latest_path, saved)
    return {
        "arm": arm,
        "feature_mode": feature_mode,
        "scorer_updates": int(scorer_updates_done),
        "coadapt_updates": int(coadapt_target),
        "actual_optimizer_updates": int(scorer_updates_done + coadapt_target),
        "restricted_optimizer_updates": int(restricted_count),
        "coadapt_full_access_optimizer_updates": int(full_replay_count),
        "stage1_readonly_full_checks": int(readonly_full_check_count),
        "checkpoint": str(latest_path),
        "action_exposure_count": len(action_exposure),
        "role_sampler_cache": _cache_statistics(cache),
    }


def run_pair(
    *,
    output: Path = DEFAULT_OUTPUT,
    scorer_target: int = 300,
    coadapt_target: int = 700,
    pilot: bool = False,
    device_name: str = "cuda:1",
    stop_after_utc: datetime | None = None,
) -> dict[str, Any]:
    if stop_after_utc is not None:
        if stop_after_utc.tzinfo is None or stop_after_utc.utcoffset() is None:
            raise ValueError("training stop deadline must be timezone-aware")
        stop_after_utc = stop_after_utc.astimezone(timezone.utc)
    _check_training_deadline("run initialization", 0, stop_after_utc)
    if device_name != "cuda:1":
        raise ValueError("the authorized Wind physical GPU must be addressed explicitly as cuda:1")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("authorized explicit cuda:1 is not available in the ordinary device visibility")
    device = torch.device(device_name)
    output.mkdir(parents=True, exist_ok=True)
    lock, view, _split = load_or_lock_cohort(output)
    if len(lock["training_layouts"]) < 30 or len(lock["development_layouts"]) < 12:
        raise RuntimeError("metadata lock does not satisfy at least 30 training and 12 development layouts")
    # Preserve paired initialization and stream. Arms run sequentially under
    # the same explicit physical device; no visibility remapping is applied.
    torch.cuda.synchronize(device)
    device_properties = torch.cuda.get_device_properties(device)
    summaries: list[dict[str, Any]] = []
    for arm, feature_mode in (("w_dir", "directional"), ("w_summary", "summary")):
        result = _run_arm(
            arm=arm,
            feature_mode=feature_mode,
            lock=lock,
            view=view,
            output=output,
            scorer_target=scorer_target,
            coadapt_target=coadapt_target,
            pilot=pilot,
            device=device,
            stop_after_utc=stop_after_utc,
        )
        summaries.append(result)
        torch.cuda.empty_cache()
    run_manifest = {
        "schema_version": 1,
        "run_id": "directed_stable_20261002_wind",
        "device": str(device),
        "device_name": str(device_properties.name),
        "device_uuid": str(getattr(device_properties, "uuid", "unavailable")),
        "visibility_remapped": False,
        "source_checkpoint": lock["source_checkpoint"],
        "cohort_lock": str(output / "cohort_lock.json"),
        "query_counts": ROLE_QUERY_COUNTS,
        "per_arm_targets": {"scorer_actual_updates": scorer_target, "coadapt_actual_updates": coadapt_target},
        "pair_summaries": summaries,
        "sample_seed": TRAINING_SEED,
        "optimizer_initialization_seed": ORG_SEED,
        "stage1_full_access_checks_are_readonly": True,
        "coadapt_full_replay_fraction": 0.25,
        "new_training_sampling_stop_utc": (None if stop_after_utc is None else stop_after_utc.isoformat()),
        "development_errors_opened": False,
        "pilot_only": bool(pilot),
    }
    _json_write(output / "run_manifest.json", run_manifest)
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scorer-target", type=int, default=300)
    parser.add_argument("--coadapt-target", type=int, default=700)
    parser.add_argument("--pilot", action="store_true", help="run one actual restricted scorer update per arm only")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--stop-after-utc",
        type=_parse_utc_deadline,
        default=None,
        help="UTC cutoff for new sampling and training (ISO-8601 with timezone)",
    )
    arguments = parser.parse_args()
    result = run_pair(
        output=arguments.output,
        scorer_target=arguments.scorer_target,
        coadapt_target=arguments.coadapt_target,
        pilot=arguments.pilot,
        device_name=arguments.device,
        stop_after_utc=arguments.stop_after_utc,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
