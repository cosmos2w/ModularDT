"""Matched, exposure-controlled continuation of Thermal Run1509 G/P at u200.

This is a new experiment branch. It restores the exact saved physical and route
weights plus optimizer state, then applies the fixed 0.90 cut curriculum and an
independent 600-case historical clock. It never launches a reference solver.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from honf_runtime.compat import load_trusted_checkpoint

CASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(CASE_ROOT / "scripts"))

import run_active_packet_forward as forward  # noqa: E402
from channelthermal.data.datasets import GlobalChannelThermalDataset  # noqa: E402
from channelthermal.evaluation.loading import load_model  # noqa: E402
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil  # noqa: E402
from channelthermal.response_control.active_packet import ThermalCoverPlanBuilder, ThermalHardValueSoftOperator  # noqa: E402
from channelthermal.response_control.contracts import DesignInput, context_inputs, role_queries_from_stencil  # noqa: E402
from channelthermal.response_control.historical import HistoricalValueSource  # noqa: E402
from channelthermal.response_control.losses import compute_stencil_loss_terms, weighted_masked_mse  # noqa: E402
from channelthermal.response_control.maturation import (  # noqa: E402
    ThermalMaturationSchedule,
    available_frontier_for_paths,
    complementary_soft_shadow_variants,
    summarize_realized_cut_records,
)
from channelthermal.response_control.native import DifferentiableThermalOperator  # noqa: E402
from channelthermal.response_control.runner import (  # noqa: E402
    _configure_native_expanded_response_interface_scope,
    _make_input_template,
    _mixed_specs,
    _resolve_dataset_path,
    derive_training_scales,
)
from channelthermal.response_control.sampling import ReceiverSamplingConfig, sample_training_panel  # noqa: E402
from channelthermal.response_control.training import (  # noqa: E402
    StagedTrainingConfig,
    TrainingStage,
    TrainingStep,
    checkpoint_payload,
    run_staged_fit,
)
from honf_forward_core.interface_fields.action_aware_frontier import (  # noqa: E402
    TypedActionSources,
    describe_frontier_action,
    describe_realized_plan,
    receiver_role_descriptors,
)
from honf_forward_core.interface_fields.budgeted_frontier import frontier_paths  # noqa: E402
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer  # noqa: E402


BASE_RUN = (
    PROJECT_ROOT
    / "Trained_Results/ThermalChannel/HONF_Forward_Runs/"
    "Run_1509_20260928_active_packet_organization_attempt07"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/generated/thermal_maturation_20260929/controlled_run"
)
DEFAULT_ATLAS = forward.DEFAULT_ATLAS
DEFAULT_REFERENCE = forward.DEFAULT_CHECKPOINT
START_UPDATE = 200
ADDITIONAL_UPDATE_CAP = 2400
TOTAL_UPDATE_CAP = START_UPDATE + ADDITIONAL_UPDATE_CAP
EXPECTED_BASE_CHECKPOINT_SHA256 = {
    "G": "c55c623db398e10f16594cd271c711f1363b5a51102fcd964595aea56a6260c4",
    "P": "f4756623386caae71202843e7bd6e77f1df6ce9f7ce70a4bf49423e098bf0d5b",
}
EXPECTED_BASE_ATTEMPTED_STEPS = {"G": 210, "P": 200}
EXPECTED_REFERENCE_SHA256 = "71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066"
EXPECTED_ROUTE_DIAGNOSTIC_SHA256 = "fe2291d1bccfa7540aa290c7714010d86e880b05264ba98c674b880040d39121"
PHYSICAL_GRAD_CLIP = 10.0
ROUTE_GRAD_CLIP = 0.1
ANCHOR_LAMBDA_RANGE = (1.0e-4, 1.0)
DEFAULT_ANCHOR_LAMBDA = 1.0e-3
LANE_GPU_HOUR_CAP = 24.0
AUTHORIZED_GPU2_UUID = "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"
SHADOW_FULL_REPLAY_PATH = (
    PROJECT_ROOT
    / "diagnostics/generated/active_packet_reuse_20260929/shadow_full_budget_replay/thermal_replay.jsonl"
)
ROUTE_LOCAL_DIAGNOSTIC_PATH = (
    PROJECT_ROOT / "diagnostics/generated/thermal_maturation_20260929/route_local_diagnostic.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_gpu2_identity() -> tuple[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "2",
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in result.stdout.strip().split(",", maxsplit=1)]
    if len(fields) != 2 or not all(fields):
        raise RuntimeError(f"Could not parse physical GPU2 identity from nvidia-smi: {result.stdout!r}")
    return fields[0], fields[1]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _gpu2_compute_process_ids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "2",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip() and line.strip().lower() != "no running processes found"
    }


def _validated_loss_weight_map(value: Any, *, label: str) -> dict[str, float]:
    """Validate a complete positive loss-weight map for this runner's loss terms."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping of every configured loss term.")
    expected_terms = set(forward.LOSS_WEIGHTS)
    if set(value) != expected_terms:
        raise ValueError(f"{label} terms differ from the configured loss-term set.")
    weights: dict[str, float] = {}
    for name, raw_value in value.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{label}[{name!r}] must be a finite positive number.")
        weight = float(raw_value)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"{label}[{name!r}] must be a finite positive number.")
        weights[name] = weight
    return weights


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _loss_weight_transition_spec_sha256(transition: Mapping[str, Any]) -> str:
    """Hash the immutable fields that define one loss-weight transition."""

    matched_arms = transition["matched_arms"]
    sources = transition["source_checkpoints"]
    spec = {
        "transition_id": transition["transition_id"],
        "matched_arms": list(matched_arms),
        "first_new_update": int(transition["first_new_update"]),
        "source_checkpoints": {
            arm: {
                "update": int(sources[arm]["update"]),
                "sha256": str(sources[arm]["sha256"]),
            }
            for arm in matched_arms
        },
        "old_loss_weights": _validated_loss_weight_map(
            transition["old_loss_weights"], label="old_loss_weights"
        ),
        "new_loss_weights": _validated_loss_weight_map(
            transition["new_loss_weights"], label="new_loss_weights"
        ),
        "source_manifest_sha256": transition["source_manifest_sha256"],
        "source_manifest_snapshot_path": transition["source_manifest_snapshot_path"],
        "evidence_sha256": transition["evidence_sha256"],
    }
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _loss_weight_transition_checkpoint_marker(
    transition: Mapping[str, Any],
    *,
    arm: str,
    loss_weights: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the provenance marker shared by preflight and later checkpoints."""

    sources = transition.get("source_checkpoints")
    source = sources.get(arm) if isinstance(sources, Mapping) else None
    if not isinstance(source, Mapping):
        raise TypeError("The loss-weight transition has no source checkpoint for this arm.")
    new_weights = _validated_loss_weight_map(
        transition.get("new_loss_weights"), label="transition new_loss_weights"
    )
    active_weights = _validated_loss_weight_map(loss_weights, label="active loss_weights")
    if active_weights != new_weights:
        raise ValueError("The checkpoint loss weights differ from the declared transition weights.")
    return {
        "transition_id": str(transition["transition_id"]),
        "source_checkpoint_sha256": str(source["sha256"]),
        "source_manifest_sha256": str(transition["source_manifest_sha256"]),
        "source_manifest_snapshot_path": str(transition["source_manifest_snapshot_path"]),
        "transition_spec_sha256": str(transition["transition_spec_sha256"]),
        "evidence_sha256": str(transition["evidence_sha256"]),
        "first_new_update": int(transition["first_new_update"]),
        "loss_weights": new_weights,
    }


def _mark_loss_weight_transition_checkpoint(
    payload: Mapping[str, Any],
    transition: Mapping[str, Any],
    *,
    arm: str,
    loss_weights: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy a completed checkpoint payload and attach the active transition marker."""

    marked = dict(payload)
    marked["loss_weight_transition"] = _loss_weight_transition_checkpoint_marker(
        transition, arm=arm, loss_weights=loss_weights
    )
    return marked


def _prepare_loss_weight_transition_payload(
    payload: Mapping[str, Any],
    *,
    arm: str,
    checkpoint_sha256: str,
    transition: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Apply a manifest-bound loss-weight transition once, preserving optimizer and RNG state."""

    matched_arms = transition.get("matched_arms")
    sources = transition.get("source_checkpoints")
    source = sources.get(arm) if isinstance(sources, Mapping) else None
    old_weight_values = transition.get("old_loss_weights")
    new_weight_values = transition.get("new_loss_weights")
    transition_id = transition.get("transition_id")
    spec_sha = transition.get("transition_spec_sha256")
    evidence_sha = transition.get("evidence_sha256")
    source_manifest_sha = transition.get("source_manifest_sha256")
    first_update = transition.get("first_new_update")
    source_manifest_snapshot_path = transition.get("source_manifest_snapshot_path")
    if (
        not isinstance(matched_arms, list)
        or not matched_arms
        or not all(isinstance(name, str) and name for name in matched_arms)
        or len(set(matched_arms)) != len(matched_arms)
        or arm not in matched_arms
        or not isinstance(source, Mapping)
        or not isinstance(transition_id, str)
        or not transition_id
        or not all(character.isalnum() or character in "_-" for character in transition_id)
        or not _is_sha256_digest(spec_sha)
        or not _is_sha256_digest(evidence_sha)
        or not _is_sha256_digest(source_manifest_sha)
        or isinstance(first_update, bool)
        or not isinstance(first_update, int)
        or first_update <= 1
        or not isinstance(source_manifest_snapshot_path, str)
        or not Path(source_manifest_snapshot_path).is_absolute()
    ):
        raise ValueError("The manifest does not contain a complete loss-weight transition record.")
    snapshot_path = Path(source_manifest_snapshot_path).expanduser()
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise ValueError("The immutable source-manifest snapshot is missing or is not a regular file.")
    if _sha256(snapshot_path) != source_manifest_sha:
        raise ValueError("The immutable source-manifest snapshot SHA256 differs from the transition record.")
    if not isinstance(sources, Mapping) or set(sources) != set(matched_arms):
        raise ValueError("The matched-arm list and per-arm source checkpoint map differ.")
    old_weights = _validated_loss_weight_map(old_weight_values, label="old_loss_weights")
    new_weights = _validated_loss_weight_map(new_weight_values, label="new_loss_weights")
    for source_arm in matched_arms:
        source_record = sources[source_arm]
        source_update = source_record.get("update") if isinstance(source_record, Mapping) else None
        source_sha = source_record.get("sha256") if isinstance(source_record, Mapping) else None
        if (
            isinstance(source_update, bool)
            or not isinstance(source_update, int)
            or source_update != first_update - 1
            or not _is_sha256_digest(source_sha)
        ):
            raise ValueError("The loss-weight transition source or provenance binding is invalid.")
    source_sha = str(source["sha256"])
    source_update = int(source["update"])
    if _loss_weight_transition_spec_sha256(transition) != spec_sha:
        raise ValueError("The loss-weight transition specification SHA256 differs from its fields.")
    if not isinstance(payload.get("optimizer"), Mapping):
        raise TypeError("The transitioned checkpoint optimizer state must be a mapping.")
    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping):
        raise TypeError("The transitioned checkpoint training config must be a mapping.")
    if payload.get("arm") != training_config.get("arm"):
        raise ValueError("The transitioned checkpoint arm/config identity is inconsistent.")
    marker = _loss_weight_transition_checkpoint_marker(
        transition, arm=arm, loss_weights=new_weights
    )
    calibrated_weights = _validated_loss_weight_map(
        payload.get("calibrated_loss_weights"), label="checkpoint calibrated_loss_weights"
    )
    update = int(payload.get("actual_optimizer_updates", -1))
    if (
        update == source_update
        and checkpoint_sha256 == source_sha
        and calibrated_weights == old_weights
        and payload.get("loss_weight_transition") is None
    ):
        prepared = dict(payload)
        prepared["calibrated_loss_weights"] = new_weights
        prepared["loss_weight_transition"] = marker
        return prepared, "transitioned_from_exact_source_checkpoint"
    if (
        update >= source_update
        and calibrated_weights == new_weights
        and payload.get("loss_weight_transition") == marker
    ):
        return dict(payload), "already_transitioned_checkpoint"
    raise ValueError("The checkpoint is neither the exact old transition source nor a provenance-bound transitioned checkpoint.")


def _checkpoint_values_equal(left: Any, right: Any) -> bool:
    """Compare checkpoint payload values exactly, including nested tensors and NumPy RNG state."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_checkpoint_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_checkpoint_values_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _persist_resume_preflight_checkpoint(
    checkpoint_dir: Path,
    *,
    arm: str,
    prepared_payload: Mapping[str, Any],
    transition_id: str,
    source_update: int,
) -> tuple[Path, str, bool]:
    """Save the preflight payload or adopt an exact atomic-write orphan after a crash."""

    if not transition_id or not all(character.isalnum() or character in "_-" for character in transition_id):
        raise ValueError("The transition identifier is not safe for a checkpoint filename.")
    path = checkpoint_dir / f"{arm}_u{source_update:04d}_resume_preflight_{transition_id}.pt"
    if path.exists():
        existing = load_trusted_checkpoint(path, map_location="cpu")
        if not _checkpoint_values_equal(existing, prepared_payload):
            raise ValueError("Orphaned resume-preflight checkpoint differs from the exact transitioned source payload.")
        return path, _sha256(path), True
    _atomic_torch_save(path, prepared_payload)
    return path, _sha256(path), False


def _apply_u300_protocol_amendment(
    manifest_path: Path,
    parity_path: Path,
    *,
    active_gpu2_pids: set[int] | None = None,
) -> dict[str, Any]:
    """Apply the one-time parity-bound source transition from stopped u300."""

    manifest_path = manifest_path.expanduser().resolve()
    parity_path = parity_path.expanduser().resolve()
    if manifest_path.name != "run_manifest.json" or not manifest_path.is_file() or not parity_path.is_file():
        raise FileNotFoundError("The u300 amendment requires the existing manifest and parity result.")
    old_bytes = manifest_path.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    manifest = json.loads(old_bytes)
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity_sha = _sha256(parity_path)
    if manifest.get("u300_protocol_amendment") is not None:
        raise ValueError("The u300 protocol amendment is one-time only.")
    if manifest.get("status") != "passed_timing_gate_u300":
        raise ValueError("The source transition requires the stopped matched u300 gate.")
    latest_recorded_driver_sha = next(
        (
            row.get("new_driver_sha256")
            for row in reversed(manifest.get("driver_source_amendments", []))
            if row.get("kind") == "post_u300_execution_mode_guard_source_amendment"
        ),
        None,
    )
    if not latest_recorded_driver_sha or manifest.get("driver_sha256") != latest_recorded_driver_sha:
        raise ValueError("The old manifest SHA does not match its latest recorded driver amendment.")
    if parity.get("status") != "passed" or any(
        not parity.get("arms", {}).get(arm, {}).get("gate", {}).get("all_gates_pass")
        for arm in ("G", "P")
    ):
        raise ValueError("The exact G/P parity artifact did not pass all gates.")
    if (
        parity.get("controlled_manifest_sha256") != old_sha
        or int(parity.get("optimizer_calls", -1)) != 0
        or int(parity.get("reference_solver_calls", -1)) != 0
        or parity.get("changed_model_weights") is not False
        or parity.get("physical_gpu", {}).get("uuid") != AUTHORIZED_GPU2_UUID
    ):
        raise ValueError("Parity is not bound to this unchanged u300 manifest/checkpoints/device.")
    if manifest.get("physical_gpu_uuid") != AUTHORIZED_GPU2_UUID:
        raise ValueError("The stopped manifest is not bound to the authorized physical GPU2.")

    checkpoint_hashes: dict[str, str] = {}
    bindings = parity.get("train_only_input", {}).get("checkpoint_bindings", {})
    for arm in ("G", "P"):
        state = manifest.get("arms", {}).get(arm, {})
        checkpoint_path = Path(str(state.get("latest_checkpoint", ""))).expanduser().resolve()
        checkpoint_sha = state.get("latest_checkpoint_sha256")
        binding = bindings.get(arm, {})
        if (
            state.get("status") != "passed_timing_gate"
            or int(state.get("latest_update", -1)) != 300
            or int(state.get("final_update", -1)) != 300
            or not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint_sha
            or binding.get("sha256") != checkpoint_sha
        ):
            raise ValueError(f"The {arm} arm is not bound to its unchanged passed u300 checkpoint.")
        checkpoint_hashes[arm] = str(checkpoint_sha)
    if parity.get("checkpoint_sha256") != checkpoint_hashes:
        raise ValueError("Parity checkpoint hashes differ from the exact stopped G/P checkpoints.")
    parity_runner_sha = parity.get("source_sha256", {}).get(str(Path(__file__).resolve()))
    if not parity_runner_sha:
        raise ValueError("Parity result lacks the pre-amendment runner SHA.")

    attempts_path = manifest_path.parent / "optimizer_attempts.jsonl"
    if not attempts_path.is_file() or any(
        int(row.get("completed_updates_before_attempt", -1)) >= 300
        for row in (
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("arm") in {"G", "P"}
    ):
        raise ValueError("An optimizer attempt is open at or beyond the u300 boundary.")
    running_pids = _gpu2_compute_process_ids() if active_gpu2_pids is None else set(active_gpu2_pids)
    if running_pids - {os.getpid()}:
        raise RuntimeError(f"GPU2 is not idle: active compute PIDs {sorted(running_pids)}.")

    backup = manifest_path.with_name("run_manifest_before_u301_protocol_amendment.json")
    if backup.exists():
        if backup.read_bytes() != old_bytes:
            raise FileExistsError("The protocol-amendment backup does not match the original manifest.")
    else:
        temporary = backup.with_name(f".{backup.name}.tmp")
        temporary.write_bytes(old_bytes)
        os.replace(temporary, backup)

    final_driver_sha = _sha256(Path(__file__).resolve())
    lineage = {
        "old_manifest_sha256": old_sha,
        "old_driver_sha256": manifest["driver_sha256"],
        "parity_runner_sha256": parity_runner_sha,
        "final_driver_sha256": final_driver_sha,
        "parity_result_sha256": parity_sha,
        "checkpoint_sha256": checkpoint_hashes,
        "first_new_update": 301,
        "backup_manifest_path": str(backup),
        "backup_manifest_sha256": _sha256(backup),
    }
    manifest["u300_protocol_amendment"] = {
        **lineage,
        "old_phase_updates": [201, 300],
        "old_phase_preserved_without_relabeling": True,
        "new_phase": {
            "capacity_fraction": 0.90,
            "random_tail_fluid_queries": 128,
            "random_tail_solid_queries_per_module": 16,
            "soft_shadow": "complementary 5-of-10 variant masks in deterministic sparse-update pairs; baseline/history scale 1 and selected variants scale 2",
        },
        "optimizer_calls": 0,
        "reference_solver_calls": 0,
        "timestamp_utc": _utc_now(),
    }
    manifest["driver_sha256"] = final_driver_sha
    manifest.setdefault("driver_source_amendments", []).append(
        {"kind": "u300_complementary_shadow_protocol_amendment", **lineage}
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _scale_record(scales: Any) -> dict[str, Any]:
    return {
        "value": {key: list(value) for key, value in scales.value.items()},
        "finite": {key: list(value) for key, value in scales.finite.items()},
        "finite_peak": float(scales.solid_temperature),
        "pressure_value": float(scales.pressure_value),
        "pressure_response": float(scales.pressure_response),
    }


def _gradient_norm_from_values(values: tuple[torch.Tensor | None, ...]) -> float:
    squared = sum(
        float(value.detach().double().square().sum().cpu())
        for value in values
        if value is not None
    )
    return math.sqrt(squared)


def _restore_global_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    for device_name, state in (payload.get("cuda_rng_state_by_model_device") or {}).items():
        device = torch.device(device_name)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(state, device=device)


class _MaturationController:
    """Adapt the pure family/action schedule to the Thermal plan-builder API."""

    def __init__(self, family_ids: tuple[str, ...], extra_route: str) -> None:
        self.maturation = ThermalMaturationSchedule(
            family_ids,
            start_update=START_UPDATE,
            seed=7319,
            sparse_before_full=4,
        )
        self.family_ids = family_ids
        self.extra_route = str(extra_route).upper()
        self.current_update = START_UPDATE
        self.resolved_cut_paths: tuple[str, ...] = ()

    def plan(self, completed_updates_before: int | None = None):
        absolute = self.current_update if completed_updates_before is None else int(completed_updates_before)
        return self.maturation.plan(absolute)

    def budgets(self) -> dict[str, float]:
        fraction = self.plan().capacity_fraction
        return {"QE": fraction, self.extra_route: fraction}

    def cut(self, tree: Any) -> tuple[int, ...]:
        plan = self.plan()
        cut, paths = available_frontier_for_paths(tree, plan.requested_cut_paths)
        self.resolved_cut_paths = paths
        return cut


def _config(*, max_wall_seconds: float) -> StagedTrainingConfig:
    active_terms = (
        "value",
        "finite",
        "finite_peak",
        "pressure_value",
        "pressure_response",
    )
    reviews = {205, 210}
    reviews.update(range(220, TOTAL_UPDATE_CAP + 1, 20))
    reviews.update({1200, 1400, TOTAL_UPDATE_CAP})
    return StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=TOTAL_UPDATE_CAP,
        max_epochs=TOTAL_UPDATE_CAP,
        total_optimizer_update_ceiling=TOTAL_UPDATE_CAP,
        checkpoint_every_updates=10,
        max_wall_seconds=max_wall_seconds,
        review_updates=tuple(sorted(reviews)),
        random_seed=7319,
        deterministic_eval_mode=True,
        deterministic_algorithms=True,
        response_ramp_start_update=10,
        response_ramp_end_update=START_UPDATE,
        response_ramp_terms=("finite", "finite_peak", "pressure_value", "pressure_response"),
        required_response_terms=("finite", "finite_peak", "pressure_value", "pressure_response"),
        include_feasibility_bce=False,
        stages=(TrainingStage("controlled_primary_0p90", 0, TOTAL_UPDATE_CAP, active_terms),),
    )


def _review_updates() -> tuple[int, ...]:
    return tuple(sorted(_config(max_wall_seconds=7200.0).review_updates))


def _setup_bundle(
    arm: str,
    source_payload: Mapping[str, Any],
    *,
    reference_checkpoint: Path,
    encoded: Any,
    extra_route: str,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    forward_model, model_payload = load_model(reference_checkpoint, device)
    _configure_native_expanded_response_interface_scope(forward_model)
    forward_model.load_state_dict(
        {key.removeprefix("forward_model."): value for key, value in source_payload["model"].items() if key.startswith("forward_model.")},
        strict=True,
    )
    route_model = forward._route_module(arm, forward_model.core, encoded, device)
    route_state = {
        key.removeprefix("route_model."): value
        for key, value in source_payload["model"].items()
        if key.startswith("route_model.")
    }
    route_model.load_state_dict(route_state, strict=True)
    bundle = forward._TrainingBundle(forward_model, route_model)
    if not all(key.startswith(("forward_model.", "route_model.")) for key in source_payload["model"]):
        raise ValueError("Run1509 warm-start state has an unexpected model key layout.")
    bundle.eval()
    return forward_model, route_model, bundle


def _anchor_loss(
    student_prediction: Any,
    teacher_prediction: Any,
    stencil: Any,
    scales: Any,
) -> torch.Tensor:
    terms = []
    student_values = getattr(student_prediction, "values", {"baseline": student_prediction})
    teacher_values = getattr(teacher_prediction, "values", {"baseline": teacher_prediction})
    records = (("baseline", stencil.baseline), *tuple(stencil.variants.items()))
    if set(student_values) != set(teacher_values) or set(student_values) != {label for label, _ in records}:
        raise ValueError("Student, retained incumbent, and Thermal stencil states do not align for anchor loss.")
    for label, record in records:
        if record.output is None:
            raise ValueError("Retained-incumbent anchor requires solved train response roles.")
        for role_name, role in record.output.roles.items():
            value = weighted_masked_mse(
                student_values[label].role_values[role_name],
                teacher_values[label].role_values[role_name].detach(),
                valid_mask=role.valid_mask,
                quadrature_weights=role.quadrature_weights,
                scales=scales.value[role_name],
            )
            if value is not None:
                terms.append(value)
    if not terms:
        raise RuntimeError("The retained-incumbent anchor has no observed Thermal roles.")
    return torch.stack(terms).mean()


def _calibrate_anchor_lambda(
    *,
    arms: tuple[str, ...],
    base_payloads: Mapping[str, Mapping[str, Any]],
    reference_model: nn.Module,
    template: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    source_encoded: Any,
    extra_route: str,
    anchor_stencil: Any,
    scales: Any,
    device: torch.device,
) -> dict[str, Any]:
    teacher = DifferentiableThermalOperator(
        reference_model,
        template,
        dataset_config=checkpoint_payload["train_config"]["dataset"],
        normalization_stats=checkpoint_payload["global_normalization_stats"],
        query_batch_size=2048,
    )
    reference_model.eval()
    with torch.no_grad():
        teacher_prediction = forward.predict_stencil(teacher, anchor_stencil, device=device)
    measurements = {}
    for arm in arms:
        forward_model, route_model, bundle = _setup_bundle(
            arm,
            base_payloads[arm],
            reference_checkpoint=Path(str(checkpoint_payload["_checkpoint_path"])),
            encoded=source_encoded,
            extra_route=extra_route,
            device=device,
        )
        native = DifferentiableThermalOperator(
            forward_model,
            template,
            dataset_config=checkpoint_payload["train_config"]["dataset"],
            normalization_stats=checkpoint_payload["global_normalization_stats"],
            query_batch_size=2048,
        )
        predictions = forward.predict_stencil(native, anchor_stencil, device=device)
        terms = compute_stencil_loss_terms(
            predictions,
            anchor_stencil,
            scales=scales,
            mixed_specs=(),
            enabled_terms=("value",),
            include_feasibility_bce=False,
        )
        data_loss = terms.terms["value"]
        incumbent_loss = _anchor_loss(predictions, teacher_prediction, anchor_stencil, scales)
        parameters = tuple(value for value in forward_model.parameters() if value.requires_grad)
        data_gradient = torch.autograd.grad(data_loss, parameters, retain_graph=True, allow_unused=True)
        anchor_gradient = torch.autograd.grad(incumbent_loss, parameters, retain_graph=False, allow_unused=True)
        data_norm = _gradient_norm_from_values(data_gradient)
        anchor_norm = _gradient_norm_from_values(anchor_gradient)
        measurements[arm] = {
            "reference_value_loss": float(data_loss.detach().cpu()),
            "incumbent_distortion_loss": float(incumbent_loss.detach().cpu()),
            "reference_value_gradient_l2": data_norm,
            "incumbent_distortion_gradient_l2": anchor_norm,
        }
        del bundle, route_model, forward_model, native
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reference_rms = math.sqrt(sum(row["reference_value_gradient_l2"] ** 2 for row in measurements.values()) / len(measurements))
    anchor_rms = math.sqrt(sum(row["incumbent_distortion_gradient_l2"] ** 2 for row in measurements.values()) / len(measurements))
    raw_lambda = (
        0.1 * reference_rms / anchor_rms
        if anchor_rms > 1.0e-12
        else DEFAULT_ANCHOR_LAMBDA
    )
    selected_lambda = min(ANCHOR_LAMBDA_RANGE[1], max(ANCHOR_LAMBDA_RANGE[0], raw_lambda))
    for row in measurements.values():
        row["anchor_gradient_fraction_after_lambda"] = (
            selected_lambda * row["incumbent_distortion_gradient_l2"]
            / max(row["reference_value_gradient_l2"], 1.0e-12)
        )
    return {
        "calibration_split": "train",
        "calibration_family_id": anchor_stencil.physical_family_id,
        "calibration_rule": "0.1 * RMS_G_P(reference_value_grad_l2) / RMS_G_P(incumbent_distortion_grad_l2)",
        "gradient_norm_aggregation": "root_mean_square_over_matched_G_and_P",
        "raw_lambda": raw_lambda,
        "lambda_clip": list(ANCHOR_LAMBDA_RANGE),
        "selected_shared_lambda": selected_lambda,
        "clipped": selected_lambda != raw_lambda,
        "default_when_anchor_gradient_zero": DEFAULT_ANCHOR_LAMBDA,
        "per_arm": measurements,
    }


def _action_descriptor_smoke(
    *,
    checkpoint: Path,
    payload: Mapping[str, Any],
    source_encoded: Any,
    route_state: Mapping[str, Any],
    template: Mapping[str, Any],
    stencil: Any,
    extra_route: str,
    device: torch.device,
) -> dict[str, Any]:
    model, model_payload = load_model(checkpoint, device)
    _configure_native_expanded_response_interface_scope(model)
    physical_state = {
        key.removeprefix("forward_model."): value
        for key, value in payload["model"].items()
        if key.startswith("forward_model.")
    }
    if not physical_state:
        raise ValueError("The action smoke requires exact physical weights from the saved G-u200 checkpoint.")
    model.load_state_dict(physical_state, strict=True)
    route_model = forward._route_module("G", model.core, source_encoded, device)
    exact_route_state = {
        key.removeprefix("route_model."): value
        for key, value in payload["model"].items()
        if key.startswith("route_model.")
    }
    if route_state and (
        exact_route_state.keys() != route_state.keys()
        or any(not torch.equal(exact_route_state[key], route_state[key]) for key in exact_route_state)
    ):
        raise ValueError("The action smoke route state disagrees with the exact G-u200 payload.")
    route_model.load_state_dict(exact_route_state, strict=True)
    native = DifferentiableThermalOperator(
        model,
        template,
        dataset_config=model_payload["train_config"]["dataset"],
        normalization_stats=model_payload["global_normalization_stats"],
        query_batch_size=2048,
    )

    class ScaffoldSchedule:
        current_update = START_UPDATE

        def budgets(self):
            return {"QE": 0.90, extra_route: 0.90}

        def cut(self, tree):
            return available_frontier_for_paths(tree, ("L", "R"))[0]

    schedule = ScaffoldSchedule()
    hard, soft = forward._route_builder(
        "G",
        core=model.core,
        route_model=route_model,
        extra_route=extra_route,
        schedule=schedule,
    )
    paired = ThermalHardValueSoftOperator(native, hard, soft)
    baseline = stencil.baseline
    design = DesignInput.from_state(baseline.design, device=device)
    context = context_inputs(baseline.context)
    queries = role_queries_from_stencil(stencil, device=device)
    with torch.no_grad():
        paired(design, context, queries)
    if not hard.last_scores or not hard.last_plans or not hard.last_trees or not hard.last_encoded:
        raise RuntimeError("The Thermal G sparse plan did not retain realized input-only score evidence.")
    score = hard.last_scores[0]
    plan = hard.last_plans[0]
    encoded = hard.last_encoded
    cut = tuple(hard.last_records[0]["frontier"])
    packet_rows = describe_realized_plan(score, plan, encoded, cut)
    role_rows = receiver_role_descriptors(plan.tree, role_count=route_model.role_count)
    if packet_rows.ndim != 2 or packet_rows.shape[0] != len(cut):
        raise RuntimeError("Action descriptor packet shape disagrees with the realized K=2 cut.")
    if not torch.isfinite(packet_rows).all() or not torch.isfinite(role_rows).all():
        raise FloatingPointError("Action descriptor smoke produced nonfinite input-only features.")

    tree = plan.tree
    centers = []
    for node_index in cut:
        indices = list(tree.nodes[node_index].anchor_indices)
        weights = tree.universe.weights[indices]
        centers.append((tree.universe.coordinates[indices] * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-8))
    packet_centers = torch.stack(centers)
    typed: dict[str, TypedActionSources] = {}
    for mechanism, features, coordinates, measure in (
        ("MM", score.module_embeddings, encoded.module_centers[0], encoded.module_present[0]),
        ("QE", score.environment_embeddings, encoded.env_coords[0], encoded.env_weights[0]),
    ):
        assert features is not None
        typed[mechanism] = TypedActionSources(
            source_features=features,
            source_coordinates=coordinates,
            source_measure=measure,
            permission=plan.permission_matrix(mechanism)[list(cut)].detach(),
            score=score.mechanism_logits[mechanism][list(cut)].detach(),
        )
    original = describe_frontier_action(score.node_embeddings[list(cut)], packet_centers, typed)
    joint_support = []
    for mechanism in ("MM", "QE"):
        item = typed[mechanism]
        source_bearing = item.source_measure > 0
        joint_support.append((item.permission[:, source_bearing] > 0).to(torch.uint8))
    joint_support_rows = torch.cat(joint_support, dim=1)
    signatures = [tuple(int(value) for value in row.cpu().tolist()) for row in joint_support_rows]
    nonempty_signatures = {signature for signature in signatures if any(signature)}
    nonredundant_k = len(nonempty_signatures)
    if nonredundant_k < 1:
        raise RuntimeError("The Thermal sparse-action smoke has no source-bearing MM or QE packet support.")
    collapsed_rows = [
        {"frontier_index": int(cut[index]), "joint_support_signature": list(signature)}
        for index, signature in enumerate(signatures)
        if not any(signature) or signatures.index(signature) != index
    ]
    permuted: dict[str, TypedActionSources] = {}
    for mechanism, item in typed.items():
        order = torch.arange(item.source_measure.numel() - 1, -1, -1, device=item.source_measure.device)
        permuted[mechanism] = TypedActionSources(
            source_features=item.source_features[order],
            source_coordinates=item.source_coordinates[order],
            source_measure=item.source_measure[order],
            permission=item.permission[:, order],
            score=item.score[:, order],
        )
    shuffled = describe_frontier_action(score.node_embeddings[list(cut)], packet_centers, permuted)
    permutation_delta = float((original - shuffled).abs().max().detach().cpu())
    if not torch.allclose(original, shuffled, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("Thermal action descriptors changed under joint source permutation.")
    return {
        "status": "passed",
        "arm": "G",
        "action": "two_packet_scaffold",
        "split": stencil.split.value,
        "family_id": stencil.physical_family_id,
        "frontier_paths": list(frontier_paths(tree, cut, max_depth=3)),
        "raw_frontier_k": len(cut),
        "nonredundant_k": nonredundant_k,
        "nonredundant_k_rule": "unique nonempty joint MM+QE binary support signatures over source-bearing inputs",
        "collapsed_cut_rows": collapsed_rows,
        "packet_feature_shape": list(packet_rows.shape),
        "receiver_role_feature_shape": list(role_rows.shape),
        "all_input_features_finite": True,
        "source_permutation_max_abs_delta": permutation_delta,
        "source_permutation_invariant": True,
        "target_fields_used": False,
        "weight_source": "exact Run1509 G-u200 physical and route state",
        "physical_checkpoint_sha256": _sha256(Path(str(payload["_checkpoint_path"]))) if payload.get("_checkpoint_path") else None,
    }


def _load_replay_inputs(
    *,
    base_run: Path,
    atlas_dir: Path,
    reference_checkpoint: Path,
    device: torch.device,
    query_batch_size: int,
) -> dict[str, Any]:
    base_manifest_path = base_run / "run_manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError(base_manifest_path)
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("status") != "paused_after_P_u200":
        raise ValueError("Controlled maturation requires the saved paired Run1509 G/P u200 gate.")
    atlas_sampling = ReceiverSamplingConfig(
        max_fluid_queries=128,
        solid_queries_per_module=8,
        hot_solid_points_per_module=4,
        random_seed=2317,
    )
    raw_stencils, initial_sampled_stencils, sampling_summaries = forward._selected_train_stencils(
        atlas_dir, atlas_sampling
    )
    atlas_hashes = {
        (atlas_dir / f"train_{family}_responses.npz").name: _sha256(atlas_dir / f"train_{family}_responses.npz")
        for family in forward.TRAIN_ATLAS_IDS
    }
    if base_manifest.get("stage_a", {}).get("atlas_hashes") != atlas_hashes:
        raise ValueError("Current Thermal response atlases differ from the exact Run1509 u200 cohort.")
    ref_model, ref_payload = load_model(reference_checkpoint, device)
    if int(ref_payload.get("epoch", ref_payload.get("current_epoch", -1))) != 4738:
        raise ValueError("The retained Run1804 reference must be the exact e4738 checkpoint.")
    if str(ref_model.config.core_honf.forward_architecture) != "dense_pairwise_field":
        raise ValueError("The retained Run1804 checkpoint is not the native Dense architecture.")
    _configure_native_expanded_response_interface_scope(ref_model)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    dataset_path = _resolve_dataset_path(ref_payload, None)
    dataset_sha = _sha256(dataset_path)
    if dataset_sha != base_manifest.get("dataset_sha256"):
        raise ValueError("Packed Thermal train dataset changed after the Run1509 u200 base.")
    train_dataset = GlobalChannelThermalDataset(
        dataset_path,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(train_dataset)
    capture_operator = DifferentiableThermalOperator(
        ref_model,
        template,
        dataset_config=ref_payload["train_config"]["dataset"],
        normalization_stats=ref_payload["global_normalization_stats"],
        query_batch_size=query_batch_size,
        capture_packet_inputs=True,
    )
    setup_stencil = initial_sampled_stencils[0]
    with torch.no_grad():
        capture_operator(
            DesignInput.from_state(setup_stencil.baseline.design, device=device),
            context_inputs(setup_stencil.baseline.context),
            role_queries_from_stencil(setup_stencil, device=device),
        )
    packet_inputs = capture_operator.last_packet_inputs
    if not isinstance(packet_inputs, Mapping) or packet_inputs.get("encoded") is None:
        raise RuntimeError("Native Thermal setup did not expose input-only packet encodings.")
    teacher_operator = DifferentiableThermalOperator(
        ref_model,
        template,
        dataset_config=ref_payload["train_config"]["dataset"],
        normalization_stats=ref_payload["global_normalization_stats"],
        query_batch_size=query_batch_size,
    )
    historical_sampling = ReceiverSamplingConfig(
        max_fluid_queries=3072,
        solid_queries_per_module=128,
        hot_solid_points_per_module=16,
        random_seed=2317,
    )
    historical_source = HistoricalValueSource.from_dataset(train_dataset, sampling=historical_sampling)
    if len(historical_source.case_ids) != 600 or len(set(historical_source.case_ids)) != 600:
        raise ValueError("Run1509 Thermal historical replay requires 600 unique shuffled train cases.")
    scales = derive_training_scales(
        initial_sampled_stencils,
        smooth_peak_beta=1.0,
        historical_value_source=historical_source,
    )
    if _scale_record(scales) != base_manifest.get("stage_a", {}).get("loss_scales"):
        raise ValueError("Recomputed training-only Thermal scales do not match the u200 manifest.")
    reference_sha = _sha256(reference_checkpoint)
    if reference_sha != EXPECTED_REFERENCE_SHA256 or reference_sha != base_manifest.get("checkpoint_sha256"):
        raise ValueError("Run1509's frozen reference checkpoint hash differs from Run1804 e4738.")
    payloads = {}
    hashes = {}
    attempted = {}
    for arm in ("G", "P"):
        checkpoint = base_run / "checkpoints" / f"{arm}_u0200_training_checkpoint.pt"
        arm_record = base_manifest.get("arms", {}).get(arm, {})
        if arm_record.get("status") != "paused_at_u200" or Path(str(arm_record.get("checkpoint"))).resolve() != checkpoint.resolve():
            raise ValueError(f"Run1509 {arm} is not bound to its exact saved u200 checkpoint.")
        hashes[arm] = _sha256(checkpoint)
        if hashes[arm] != EXPECTED_BASE_CHECKPOINT_SHA256[arm]:
            raise ValueError(f"Run1509 {arm} u200 checkpoint hash differs from the reviewed source checkpoint.")
        payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
        if payload.get("arm") != "R_response" or payload.get("actual_optimizer_updates") != START_UPDATE:
            raise ValueError(f"Run1509 {arm} checkpoint did not complete exactly 200 response updates.")
        expected_attempts = int(arm_record.get("attempted_optimizer_steps", -1))
        if expected_attempts != EXPECTED_BASE_ATTEMPTED_STEPS[arm] or int(payload.get("attempted_optimizer_steps", -2)) != expected_attempts:
            raise ValueError(f"Run1509 {arm} payload attempt count disagrees with its manifest.")
        if tuple(payload.get("historical_case_order", ())) != historical_source.case_ids:
            raise ValueError(f"Run1509 {arm} historical order differs from the independent 600-case replay.")
        if int(payload.get("historical_next_index", -1)) != START_UPDATE:
            raise ValueError(f"Run1509 {arm} historical clock cursor is not at saved u200.")
        payloads[arm] = payload
        attempted[arm] = expected_attempts
    base_evidence_paths = (
        base_manifest_path,
        base_run / "optimizer_attempts.jsonl",
        base_run / "G" / "training_steps.jsonl",
        base_run / "P" / "training_steps.jsonl",
        base_run / "G_route_work.jsonl",
        base_run / "P_route_work.jsonl",
        base_run / "route_pilot.json",
    )
    base_evidence_hashes = {
        str(path.resolve()): _sha256(path)
        for path in base_evidence_paths
        if path.is_file()
    }
    family_ids = tuple(item.physical_family_id for item in raw_stencils)
    if family_ids != tuple(item.physical_family_id for item in initial_sampled_stencils):
        raise RuntimeError("Raw and scale-sampling Thermal family orders differ.")
    return {
        "base_manifest": base_manifest,
        "base_manifest_path": base_manifest_path.resolve(),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "base_evidence_sha256": base_evidence_hashes,
        "raw_stencils": raw_stencils,
        "initial_sampled_stencils": initial_sampled_stencils,
        "sampling_summaries": sampling_summaries,
        "family_ids": family_ids,
        "atlas_hashes": atlas_hashes,
        "reference_model": ref_model,
        "teacher_operator": teacher_operator,
        "reference_payload": ref_payload,
        "reference_sha256": reference_sha,
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha,
        "dataset": train_dataset,
        "template": template,
        "source_encoded": packet_inputs["encoded"],
        "historical_source": historical_source,
        "historical_order": historical_source.case_ids,
        "scales": scales,
        "base_payloads": payloads,
        "base_checkpoint_hashes": hashes,
        "base_attempted_optimizer_steps": attempted,
        "extra_route": str(base_manifest["stage_a"]["selected_sparse_extra_route"]),
        "reference_checkpoint_path": reference_checkpoint.resolve(),
        "base_g_checkpoint_path": (base_run / "checkpoints" / "G_u0200_training_checkpoint.pt").resolve(),
        "base_run_path": base_run.resolve(),
        "atlas_dir": atlas_dir.resolve(),
    }


def _preflight_record(inputs: Mapping[str, Any], *, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    raw_stencils = inputs["raw_stencils"]
    family_ids = inputs["family_ids"]
    schedule = ThermalMaturationSchedule(family_ids, start_update=START_UPDATE, seed=7319, sparse_before_full=4)
    first_100 = [schedule.plan(START_UPDATE + relative) for relative in range(100)]
    sparse_counts = Counter(row.family_id for row in first_100 if not row.full_access_replay)
    total_first_100_visits = Counter(row.family_id for row in first_100)
    action_family_passes = Counter(
        (row.action, row.primary_pass)
        for row in first_100
        if not row.full_access_replay
    )
    cumulative_family_visits: Counter[str] = Counter()
    minimum_family_exposure_absolute_update = None
    for relative in range(1200):
        plan = schedule.plan(START_UPDATE + relative)
        cumulative_family_visits[plan.family_id] += 1
        if minimum_family_exposure_absolute_update is None and min(
            (cumulative_family_visits[family] for family in family_ids), default=0
        ) >= 100:
            minimum_family_exposure_absolute_update = START_UPDATE + relative + 1
            break
    anchor_stencil = sample_training_panel(
        (raw_stencils[0],),
        config=ReceiverSamplingConfig(128, 8, 4, random_seed=7319),
    )[0].stencil
    calibration = _calibrate_anchor_lambda(
        arms=("G", "P"),
        base_payloads=inputs["base_payloads"],
        reference_model=inputs["reference_model"],
        template=inputs["template"],
        checkpoint_payload={
            **inputs["reference_payload"],
            "_checkpoint_path": str(inputs["reference_checkpoint_path"]),
        },
        source_encoded=inputs["source_encoded"],
        extra_route=inputs["extra_route"],
        anchor_stencil=anchor_stencil,
        scales=inputs["scales"],
        device=device,
    )
    g_payload = inputs["base_payloads"]["G"]
    g_route_state = {
        key.removeprefix("route_model."): value
        for key, value in g_payload["model"].items()
        if key.startswith("route_model.")
    }
    action_smoke = _action_descriptor_smoke(
        checkpoint=inputs["reference_checkpoint_path"],
        payload={**g_payload, "_checkpoint_path": str(inputs["base_g_checkpoint_path"])},
        source_encoded=inputs["source_encoded"],
        route_state=g_route_state,
        template=inputs["template"],
        stencil=anchor_stencil,
        extra_route=inputs["extra_route"],
        device=device,
    )
    gpu_uuid, gpu_name = _physical_gpu2_identity()
    if gpu_uuid != AUTHORIZED_GPU2_UUID:
        raise RuntimeError(
            f"Physical GPU2 UUID {gpu_uuid} differs from the authorized device {AUTHORIZED_GPU2_UUID}."
        )
    if _sha256(SHADOW_FULL_REPLAY_PATH) != "01eec97e7a12b8c86d0a8389a90b0e9c4c690031f64627f4e95398930a671717":
        raise ValueError("The cited saved Thermal native full-budget replay changed after review.")
    if not ROUTE_LOCAL_DIAGNOSTIC_PATH.is_file():
        raise FileNotFoundError(ROUTE_LOCAL_DIAGNOSTIC_PATH)
    route_diagnostic_sha = _sha256(ROUTE_LOCAL_DIAGNOSTIC_PATH)
    if route_diagnostic_sha != EXPECTED_ROUTE_DIAGNOSTIC_SHA256:
        raise ValueError("The reviewed Thermal u200 route-local diagnostic changed after review.")
    return {
        "status": "passed",
        "run_id": "Thermal_Controlled_Maturation_from_Run1509_u200_20260929",
        "created_at_utc": _utc_now(),
        "driver_path": str(Path(__file__).resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "physical_gpu_index": 2,
        "physical_gpu_uuid": gpu_uuid,
        "physical_gpu_name": gpu_name,
        "logical_gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "base_run": str(inputs["base_run_path"]),
        "base_run_manifest": str(inputs["base_manifest_path"]),
        "base_run_manifest_sha256": inputs["base_manifest_sha256"],
        "base_run_evidence_sha256": inputs["base_evidence_sha256"],
        "base_checkpoint_sha256": inputs["base_checkpoint_hashes"],
        "old_completed_updates_per_arm": START_UPDATE,
        "old_attempted_optimizer_steps": inputs["base_attempted_optimizer_steps"],
        "old_attempted_update_asymmetry": {
            "G_extra_attempts_vs_completed": inputs["base_attempted_optimizer_steps"]["G"] - START_UPDATE,
            "P_extra_attempts_vs_completed": inputs["base_attempted_optimizer_steps"]["P"] - START_UPDATE,
        },
        "new_update_stop": args.stop_after_update,
        "new_updates_per_arm_to_requested_stop": args.stop_after_update - START_UPDATE,
        "new_optimizer_calls_per_arm_if_complete": args.stop_after_update - START_UPDATE,
        "family_ids": list(family_ids),
        "family_ids_per_sparse_update_first_100": dict(sparse_counts),
        "total_family_update_visits_first_100_including_full_replays": dict(total_first_100_visits),
        "first_100_actions": [
            {
                "relative_update": row.relative_update,
                "family_id": row.family_id,
                "phase": row.phase,
                "action": row.action,
                "requested_cut_paths": list(row.requested_cut_paths) if not row.full_access_replay else [],
                "capacity_fraction": row.capacity_fraction,
                "full_access_replay": row.full_access_replay,
                "primary_pass": row.primary_pass,
                "query_seed": row.query_seed,
            }
            for row in first_100
        ],
        "historical_case_count": len(inputs["historical_order"]),
        "historical_case_order_sha256": hashlib.sha256(
            "\n".join(inputs["historical_order"]).encode("utf-8")
        ).hexdigest(),
        "historical_cursor_at_start": START_UPDATE,
        "historical_case_ids_for_requested_stop": [
            inputs["historical_order"][(START_UPDATE + relative) % len(inputs["historical_order"])]
            for relative in range(args.stop_after_update - START_UPDATE)
        ],
        "historical_new_visits_per_arm_to_requested_stop": args.stop_after_update - START_UPDATE,
        "minimum_new_historical_visits_per_arm": 1200,
        "historical_minimum_stop_absolute_update": START_UPDATE + 1200,
        "min_new_visits_per_response_family_at_minimum": 100,
        "min_family_exposure_stop_absolute_update": minimum_family_exposure_absolute_update,
        "family_visit_counts_at_minimum_family_exposure": dict(cumulative_family_visits),
        "atlas_sha256": inputs["atlas_hashes"],
        "dataset_path": str(inputs["dataset_path"]),
        "dataset_sha256": inputs["dataset_sha256"],
        "reference_checkpoint": str(inputs["reference_checkpoint_path"]),
        "reference_checkpoint_sha256": inputs["reference_sha256"],
        "reference_solver_attempts": 0,
        "pre_maturation_diagnostics": {
            "saved_full_budget_replay": {
                "path": str(SHADOW_FULL_REPLAY_PATH.resolve()),
                "sha256": _sha256(SHADOW_FULL_REPLAY_PATH),
                "result": "bitwise output parity; 290 physical gradient matches; 40 route gradients None; planner bypass confirmed",
            },
            "route_local_diagnostic": {
                "path": str(ROUTE_LOCAL_DIAGNOSTIC_PATH.resolve()),
                "sha256": _sha256(ROUTE_LOCAL_DIAGNOSTIC_PATH),
            },
        },
        "extra_route": inputs["extra_route"],
        "primary_capacity": {"QE": 0.90, inputs["extra_route"]: 0.90},
        "schedule": {
            "seed": 7319,
            "sparse_before_full": 4,
            "first_two_sparse_family_passes": "root-split K=2 scaffold",
            "then": "two complete shuffled family passes each for root, two-packet, four-packet; rotate in blocks",
            "first_100_sparse_action_passes": dict((f"{action}:{pass_index}", count) for (action, pass_index), count in sorted(action_family_passes.items())),
        },
        "anchor_calibration": calibration,
        "action_descriptor_smoke": action_smoke,
        "gradient_clipping": {
            "physical_max_norm": PHYSICAL_GRAD_CLIP,
            "route_max_norm": ROUTE_GRAD_CLIP,
            "separate_groups": True,
            "preclip_and_postclip_norms_logged": True,
            "actual_parameter_update_norms_logged": True,
        },
        "resource_caps": {
            "additional_updates_per_arm": ADDITIONAL_UPDATE_CAP,
            "lane_gpu_hours": LANE_GPU_HOUR_CAP,
            "aggregate_round_gpu_hours": 48,
            "aggregate_optimizer_calls": 24000,
        },
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }


def _record_job_event(
    ledger_path: Path,
    *,
    event: str,
    run_id: str,
    started_at: float,
    status: str,
    output_dir: Path,
    arm: str | None = None,
    optimizer_calls: int | None = None,
) -> None:
    now = time.time()
    try:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        device_name = None
    try:
        gpu_uuid, _ = _physical_gpu2_identity()
    except Exception:
        gpu_uuid = None
    _append_jsonl(
        ledger_path,
        {
            "event": event,
            "run_id": run_id,
            "pid": os.getpid(),
            "timestamp_unix": now,
            "timestamp_utc": _utc_now(),
            "started_at_unix": started_at,
            "gpu_associated_active_seconds": max(0.0, now - started_at) if event in {"job_stop", "arm_stop"} else None,
            "status": status,
            "arm": arm,
            "optimizer_calls": optimizer_calls,
            "logical_device": "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") == "2" else None,
            "physical_gpu_index": 2 if os.environ.get("CUDA_VISIBLE_DEVICES") == "2" else None,
            "physical_gpu_uuid": gpu_uuid,
            "gpu_name": device_name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver_path": str(Path(__file__).resolve()),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "command_argv": list(sys.argv),
            "output_dir": str(output_dir),
        },
    )


def _filter_jsonl(path: Path, *, keep, source_checkpoint_update: int, recovery_path: Path) -> int:
    if not path.is_file():
        return 0
    retained: list[str] = []
    removed = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if keep(row):
                retained.append(json.dumps(row, sort_keys=True, allow_nan=False))
            else:
                _append_jsonl(
                    recovery_path,
                    {
                        "event": "discarded_uncheckpointed_derived_row",
                        "source_file": str(path),
                        "source_line": line_number,
                        "source_checkpoint_update": source_checkpoint_update,
                        "row": row,
                        "timestamp_utc": _utc_now(),
                    },
                )
                removed += 1
    if removed:
        temporary = path.with_name(f".{path.name}.recovery.tmp")
        temporary.write_text("".join(f"{line}\n" for line in retained), encoding="utf-8")
        os.replace(temporary, path)
    return removed


def _resume_optimizer_attempt_total(path: Path, arm: str, checkpoint_attempts: int) -> tuple[int, int]:
    if not path.is_file():
        return checkpoint_attempts, 0
    logged_total = checkpoint_attempts
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("arm") == arm:
                logged_total = max(
                    logged_total,
                    int(row.get("global_attempted_step_including_old_branch", checkpoint_attempts)),
                )
    return logged_total, logged_total - checkpoint_attempts


def _logged_optimizer_calls(path: Path, arm: str | None = None, *, since_unix: float | None = None) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(
            1
            for line in stream
            if line.strip()
            and (arm is None or json.loads(line).get("arm") == arm)
            and (since_unix is None or float(json.loads(line).get("timestamp_unix", 0.0)) >= since_unix)
        )


def _recorded_lane_active_seconds(path: Path) -> float:
    if not path.is_file():
        return 0.0
    total = 0.0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "job_stop":
                total += max(0.0, float(row.get("gpu_associated_active_seconds") or 0.0))
    return total


def _validate_p_execution_mode(
    route_model: Any, source_payload: Mapping[str, Any], *, requested_factorized: bool
) -> bool:
    """Reject a resumed P checkpoint under a different direct-scorer execution path."""

    resolved_factorized = bool(getattr(route_model, "factorized_first_layer", False))
    if resolved_factorized != requested_factorized:
        raise RuntimeError("The constructed P scorer differs from the requested factorized execution mode.")
    checkpoint_mode = source_payload.get("execution_mode")
    if checkpoint_mode is not None:
        if not isinstance(checkpoint_mode, Mapping) or not isinstance(
            checkpoint_mode.get("thermal_factor_direct_scorer"), bool
        ):
            raise ValueError("The manifest-bound P checkpoint has an invalid execution-mode record.")
        if checkpoint_mode["thermal_factor_direct_scorer"] != resolved_factorized:
            raise RuntimeError("The P checkpoint scorer execution mode differs from this invocation.")
    return resolved_factorized


def _run_arm(
    *,
    arm: str,
    inputs: Mapping[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    lambda_anchor: float,
) -> dict[str, Any]:
    config = _config(max_wall_seconds=args.max_wall_seconds)
    weight_transition = manifest.get("calibrated_loss_weight_transition")
    if weight_transition is None:
        active_loss_weights = _validated_loss_weight_map(
            forward.LOSS_WEIGHTS, label="configured loss weights"
        )
    else:
        active_loss_weights = _validated_loss_weight_map(
            manifest.get("loss_weights"), label="manifest loss_weights"
        )
        transition_weights = _validated_loss_weight_map(
            weight_transition.get("new_loss_weights"), label="transition new_loss_weights"
        )
        if active_loss_weights != transition_weights:
            raise ValueError("The active loss weights differ from the provenance-bound transition.")
    base_payload = inputs["base_payloads"][arm]
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_payload = None
    initial_update = START_UPDATE
    recovery = {"discarded_uncheckpointed_rows": {}, "uncheckpointed_optimizer_calls_counted": 0}
    loaded_checkpoint_sha256: str | None = None
    if args.resume:
        arm_manifest = manifest.get("arms", {}).get(arm, {})
        resume_path_value = arm_manifest.get("latest_checkpoint")
        if resume_path_value:
            resume_path = Path(str(resume_path_value)).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Manifest-bound {arm} checkpoint is missing: {resume_path}")
            expected_hash = arm_manifest.get("latest_checkpoint_sha256")
            if expected_hash != _sha256(resume_path):
                raise ValueError(f"Manifest-bound {arm} checkpoint SHA256 mismatch: {resume_path}")
            loaded_checkpoint_sha256 = str(expected_hash)
        else:
            resume_path = None
        if resume_path is not None:
            resume_payload = load_trusted_checkpoint(resume_path, map_location="cpu")
            initial_update = int(resume_payload.get("actual_optimizer_updates", -1))
            if not START_UPDATE <= initial_update < TOTAL_UPDATE_CAP:
                raise ValueError(f"{arm} resumed checkpoint update is outside the maturation branch.")
            if resume_payload.get("historical_case_order") != list(inputs["historical_order"]):
                raise ValueError(f"{arm} maturation checkpoint historical order differs from preflight.")
            if int(resume_payload.get("historical_next_index", -1)) != initial_update % len(inputs["historical_order"]):
                raise ValueError(f"{arm} maturation checkpoint historical cursor is wrong.")
            recovery_path = output_dir / "resume_recovery_rows.jsonl"
            recovery = {
                "checkpoint_path": str(resume_path),
                "checkpoint_sha256": expected_hash,
                "checkpoint_update": initial_update,
                "discarded_uncheckpointed_rows": {},
                "uncheckpointed_optimizer_calls_counted": 0,
            }
            if initial_update > START_UPDATE:
                recovery["discarded_uncheckpointed_rows"] = {
                    "training_steps": _filter_jsonl(
                        arm_dir / "training_steps.jsonl",
                        keep=lambda row: int(row["completed_update"]) <= initial_update,
                        source_checkpoint_update=initial_update,
                        recovery_path=recovery_path,
                    ),
                    "maturation_exposure": _filter_jsonl(
                        arm_dir / "maturation_exposure.jsonl",
                        keep=lambda row: int(row["completed_update"]) <= initial_update,
                        source_checkpoint_update=initial_update,
                        recovery_path=recovery_path,
                    ),
                    "gradient_steps": _filter_jsonl(
                        arm_dir / "gradient_steps.jsonl",
                        keep=lambda row: int(row["completed_updates_before_attempt"]) < initial_update,
                        source_checkpoint_update=initial_update,
                        recovery_path=recovery_path,
                    ),
                    "reviews": _filter_jsonl(
                        arm_dir / "reviews.jsonl",
                        keep=lambda row: int(row["completed_update"]) <= initial_update,
                        source_checkpoint_update=initial_update,
                        recovery_path=recovery_path,
                    ),
                    "route_work": _filter_jsonl(
                        output_dir / f"{arm}_route_work.jsonl",
                        keep=lambda row: int(row["optimizer_update"]) <= initial_update,
                        source_checkpoint_update=initial_update,
                        recovery_path=recovery_path,
                    ),
                }
            checkpoint_attempts = int(resume_payload.get("attempted_optimizer_steps", 0))
            actual_attempts, abandoned_calls = _resume_optimizer_attempt_total(
                output_dir / "optimizer_attempts.jsonl", arm, checkpoint_attempts
            )
            if abandoned_calls:
                resume_payload["attempted_optimizer_steps"] = actual_attempts
                recovery["uncheckpointed_optimizer_calls_counted"] = abandoned_calls
            manifest.setdefault("arms", {}).setdefault(arm, {})["resume_recovery"] = recovery
            _atomic_json(output_dir / "run_manifest.json", manifest)

    transition_status: str | None = None
    if weight_transition is not None:
        if resume_payload is None or loaded_checkpoint_sha256 is None:
            raise ValueError("A loss-weight transition requires an exact manifest-bound resume checkpoint.")
        resume_payload, transition_status = _prepare_loss_weight_transition_payload(
            resume_payload,
            arm=arm,
            checkpoint_sha256=loaded_checkpoint_sha256,
            transition=weight_transition,
        )
        if transition_status == "transitioned_from_exact_source_checkpoint":
            source_update = int(weight_transition["source_checkpoints"][arm]["update"])
            preflight_path, preflight_sha, recovered_orphan = _persist_resume_preflight_checkpoint(
                checkpoint_dir,
                arm=arm,
                prepared_payload=resume_payload,
                transition_id=str(weight_transition["transition_id"]),
                source_update=source_update,
            )
            arm_state = manifest["arms"].setdefault(arm, {})
            arm_state["resume_preflight_checkpoint"] = {
                "path": str(preflight_path),
                "sha256": preflight_sha,
                "update": source_update,
                "optimizer_calls": 0,
                "reference_solver_calls": 0,
                "source_checkpoint_sha256": loaded_checkpoint_sha256,
                "loss_weight_transition": resume_payload["loss_weight_transition"],
                "recovered_exact_orphan_after_crash": recovered_orphan,
            }
            arm_state.update({
                "latest_checkpoint": str(preflight_path),
                "latest_checkpoint_sha256": preflight_sha,
                "latest_update": source_update,
            })
            _atomic_json(output_dir / "run_manifest.json", manifest)
            loaded_checkpoint_sha256 = preflight_sha
    source_payload = resume_payload if resume_payload is not None else base_payload
    _restore_global_rng(source_payload)
    forward_model, route_model, bundle = _setup_bundle(
        arm,
        source_payload,
        reference_checkpoint=inputs["reference_checkpoint_path"],
        encoded=inputs["source_encoded"],
        extra_route=inputs["extra_route"],
        device=device,
    )
    resolved_factorized_scorer = (
        _validate_p_execution_mode(
            route_model, source_payload,
            requested_factorized=os.environ.get("THERMAL_FACTOR_DIRECT_SCORER", "0") == "1",
        )
        if arm == "P" else False
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in bundle.parameters() if parameter.requires_grad),
        lr=1.0e-5,
        weight_decay=1.0e-5,
    )
    optimizer.load_state_dict(source_payload["optimizer"])
    attempted_base = int(source_payload.get("attempted_optimizer_steps", 0))
    schedule = _MaturationController(inputs["family_ids"], inputs["extra_route"])
    schedule.current_update = initial_update
    hard_builder, soft_builder = forward._route_builder(
        arm,
        core=forward_model.core,
        route_model=route_model,
        extra_route=inputs["extra_route"],
        schedule=schedule,
    )
    native = DifferentiableThermalOperator(
        forward_model,
        inputs["template"],
        dataset_config=inputs["reference_payload"]["train_config"]["dataset"],
        normalization_stats=inputs["reference_payload"]["global_normalization_stats"],
        query_batch_size=args.query_batch_size,
    )
    paired = ThermalHardValueSoftOperator(native, hard_builder, soft_builder)
    state_labels = ("baseline", *tuple(inputs["raw_stencils"][0].variants))
    expected_variant_order = state_labels[1:]
    for raw_stencil in inputs["raw_stencils"]:
        actual_variant_order = tuple(raw_stencil.variants)
        if actual_variant_order != expected_variant_order:
            raise RuntimeError(
                "Thermal raw train stencil variant order differs from the scheduled "
                f"native-call order for family {raw_stencil.physical_family_id}: "
                f"expected {expected_variant_order}, got {actual_variant_order}."
            )
    route_log = output_dir / f"{arm}_route_work.jsonl"
    scheduled = forward._ScheduledThermalOperator(
        paired=paired,
        schedule=schedule,
        hard_builder=hard_builder,
        soft_builder=soft_builder,
        state_labels=state_labels,
        arm=arm,
        route_log=route_log,
        sparse_route_verified={"QE": False, inputs["extra_route"]: False},
    )
    metric_path = arm_dir / "training_steps.jsonl"
    gradient_path = arm_dir / "gradient_steps.jsonl"
    exposure_path = arm_dir / "maturation_exposure.jsonl"
    reviews_path = arm_dir / "reviews.jsonl"
    gradients_for_update: dict[int, dict[str, Any]] = {}
    pre_step_parameters: dict[int, dict[str, tuple[torch.Tensor, ...]]] = {}
    exposure_counts: Counter[tuple[str, str, str]] = Counter()
    historical_case_counts: Counter[str] = Counter()
    global_attempted_for_update: dict[int, int] = {}
    realized_cut_records_for_update: dict[int, list[dict[str, Any]]] = {}
    if metric_path.is_file():
        with metric_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                metadata = row.get("training_metadata", {})
                exposure_counts[(str(row.get("training_family_id")), str(metadata.get("phase")), str(metadata.get("action")))] += 1
                historical_id = row.get("historical_case_id")
                if historical_id is not None:
                    historical_case_counts[str(historical_id)] += 1
    pending_reviews: dict[int, dict[str, Any]] = {}

    def save_checkpoint(payload: Mapping[str, Any], label: str) -> None:
        update = int(payload["actual_optimizer_updates"])
        path = checkpoint_dir / f"{arm}_u{update:04d}_{label}.pt"
        saved_payload = dict(payload)
        if weight_transition is not None:
            saved_payload = _mark_loss_weight_transition_checkpoint(
                saved_payload,
                weight_transition,
                arm=arm,
                loss_weights=active_loss_weights,
            )
        if arm == "P":
            saved_payload["execution_mode"] = {
                "thermal_factor_direct_scorer": resolved_factorized_scorer,
                "environment": {"THERMAL_FACTOR_DIRECT_SCORER": "1" if resolved_factorized_scorer else "0"},
                "driver_path": str(Path(__file__).resolve()),
                "driver_sha256": _sha256(Path(__file__).resolve()),
            }
        _atomic_torch_save(path, saved_payload)
        state = manifest["arms"].setdefault(arm, {})
        state.update({
            "latest_checkpoint": str(path),
            "latest_checkpoint_sha256": _sha256(path),
            "latest_update": update,
            "attempted_optimizer_steps": int(payload["attempted_optimizer_steps"]),
        })
        if weight_transition is not None:
            state["loss_weight_transition_id"] = str(weight_transition["transition_id"])
            state["loss_weight_transition_source_checkpoint_sha256"] = (
                weight_transition["source_checkpoints"][arm]["sha256"]
            )
            state["loss_weight_transition_source_manifest_sha256"] = str(
                weight_transition["source_manifest_sha256"]
            )
            state["loss_weight_transition_source_manifest_snapshot_path"] = str(
                weight_transition["source_manifest_snapshot_path"]
            )
        if update in pending_reviews:
            review = pending_reviews.pop(update)
            review["checkpoint"] = str(path)
            review["checkpoint_sha256"] = _sha256(path)
            _append_jsonl(reviews_path, review)
        _atomic_json(output_dir / "run_manifest.json", manifest)

    def on_attempt(
        completed: int, attempted_total: int, scheduled_operator: Any = scheduled
    ) -> None:
        physical_parameters = tuple(value for value in forward_model.parameters() if value.requires_grad)
        route_parameters = tuple(value for value in route_model.parameters() if value.requires_grad)
        physical_raw = forward._gradient_norm(forward_model)
        route_raw = forward._gradient_norm(route_model)
        torch.nn.utils.clip_grad_norm_(physical_parameters, max_norm=PHYSICAL_GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(route_parameters, max_norm=ROUTE_GRAD_CLIP)
        physical_post = forward._gradient_norm(forward_model)
        route_post = forward._gradient_norm(route_model)
        gradients_for_update[completed] = {
            "optimizer_calls": 1,
            "global_attempted_optimizer_step": int(attempted_total),
            "shared_optimizer_step_calls": 1,
            "physical_trainable_parameter_count": len(physical_parameters),
            "route_trainable_parameter_count": len(route_parameters),
            "physical_parameter_group_received_nonzero_gradient": physical_post > 0.0,
            "route_parameter_group_received_nonzero_gradient": route_post > 0.0,
            "physical_gradient_l2_preclip": physical_raw,
            "route_gradient_l2_preclip": route_raw,
            "physical_gradient_l2_postclip": physical_post,
            "route_gradient_l2_postclip": route_post,
            "physical_clip_ratio": physical_post / physical_raw if physical_raw > 0.0 else None,
            "route_clip_ratio": route_post / route_raw if route_raw > 0.0 else None,
            "physical_clipped": physical_raw > PHYSICAL_GRAD_CLIP,
            "route_clipped": route_raw > ROUTE_GRAD_CLIP,
            "physical_clip_max_norm": PHYSICAL_GRAD_CLIP,
            "route_clip_max_norm": ROUTE_GRAD_CLIP,
            "route_optimizer_call_is_shared": True,
        }
        global_attempted_for_update[completed] = int(attempted_total)
        realized_cut_records_for_update[completed] = [
            {
                **record,
                "completed_updates_before_attempt": int(completed),
                "attempted_optimizer_step_including_old_branch": int(attempted_total),
            }
            for record in scheduled_operator.realized_cut_records_this_update
        ]
        pre_step_parameters[completed] = {
            "physical": tuple(parameter.detach().clone() for parameter in physical_parameters),
            "route": tuple(parameter.detach().clone() for parameter in route_parameters),
        }
        _append_jsonl(
            gradient_path,
            {
                "arm": arm,
                "completed_updates_before_attempt": completed,
                "global_attempted_optimizer_step": attempted_total,
                "optimizer_calls": 1,
                **gradients_for_update[completed],
                "capacity_before_step": schedule.budgets(),
                "timestamp_unix": time.time(),
                "timestamp_utc": _utc_now(),
            },
        )
        _append_jsonl(
            output_dir / "optimizer_attempts.jsonl",
            {
                "arm": arm,
                "completed_updates_before_attempt": completed,
                "global_attempted_step_including_old_branch": attempted_total,
                "old_completed_updates": START_UPDATE,
                "old_attempted_steps": inputs["base_attempted_optimizer_steps"][arm],
                "new_attempted_steps": attempted_total - inputs["base_attempted_optimizer_steps"][arm],
                "optimizer_calls": 1,
                "recorded_before_optimizer_step": True,
                "timestamp_unix": time.time(),
                "timestamp_utc": _utc_now(),
            },
        )
        scheduled_operator.optimizer_attempt(completed, attempted_total)

    def transform(
        stencil: Any, completed: int, scheduled_operator: Any = scheduled
    ) -> tuple[Any, Mapping[str, Any]]:
        if scheduled_operator.soft_shadow_scale_by_state is not None:
            raise RuntimeError(
                "Soft-shadow scale mapping was not reset after the previous optimizer attempt."
            )
        plan = schedule.plan(completed)
        if stencil.physical_family_id != plan.family_id:
            raise RuntimeError("Maturation family selector and requested raw stencil disagree.")
        selected_soft_variants = (
            ()
            if plan.full_access_replay
            else complementary_soft_shadow_variants(
                state_labels,
                relative_sparse_update=plan.relative_sparse_update,
                seed=schedule.maturation.seed,
            )
        )
        soft_shadow_scale_by_state = {
            "baseline": 0.0 if plan.full_access_replay else 1.0,
            **{
                label: (
                    0.0
                    if plan.full_access_replay or label not in selected_soft_variants
                    else 2.0
                )
                for label in state_labels[1:]
            },
            "historical_value_replay": 0.0 if plan.full_access_replay else 1.0,
        }
        if set(soft_shadow_scale_by_state) != {
            *state_labels,
            "historical_value_replay",
        }:
            raise RuntimeError("Soft-shadow scale mapping does not cover the scheduled call order.")
        if (
            not plan.full_access_replay
            and soft_shadow_scale_by_state["historical_value_replay"] != 1.0
        ):
            raise RuntimeError("Sparse historical-value replay must retain unit soft-shadow scale.")
        scheduled_operator.soft_shadow_scale_by_state = soft_shadow_scale_by_state
        sampled = sample_training_panel(
            (stencil,),
            config=replace(
                ReceiverSamplingConfig(
                    128,
                    8,
                    4,
                    random_seed=plan.query_seed,
                    random_tail_fluid_queries=128,
                    random_tail_solid_queries_per_module=16,
                ),
                random_seed=plan.query_seed,
            ),
        )[0]
        previous = schedule.plan(completed - 1) if completed > START_UPDATE else None
        metadata = {
            "family_id": plan.family_id,
            "family_index": plan.family_index,
            "primary_pass": plan.primary_pass,
            "pass_position": plan.pass_position,
            "relative_update": plan.relative_update,
            "relative_sparse_update": plan.relative_sparse_update,
            "phase": plan.phase,
            "action": plan.action,
            "requested_cut_paths": [] if plan.full_access_replay else list(plan.requested_cut_paths),
            "resolved_cut_paths": [],
            "full_access_replay": plan.full_access_replay,
            "capacity_fraction": plan.capacity_fraction,
            "query_seed": plan.query_seed,
            "sampling_summary": {
                "original_counts": dict(sampled.summary.original_counts),
                "sampled_counts": dict(sampled.summary.sampled_counts),
                "protected_counts": dict(sampled.summary.protected_counts),
                "solid_peak_query_coverage": dict(sampled.summary.solid_peak_query_coverage),
                "random_tail_counts": dict(sampled.summary.random_tail_counts),
                "random_tail_query_id_sha256": dict(
                    sampled.summary.random_tail_query_id_sha256
                ),
            },
            "soft_shadow": {
                "selection_rule": "complementary_5_of_10_by_relative_sparse_update_pair",
                "relative_sparse_update": plan.relative_sparse_update,
                "pair_index": plan.relative_sparse_update // 2,
                "complement_half": plan.relative_sparse_update % 2,
                "variant_state_order": list(state_labels[1:]),
                "selected_variant_labels": list(selected_soft_variants),
                "gradient_scale_by_state": dict(soft_shadow_scale_by_state),
                "realized_soft_state_labels": (
                    []
                    if plan.full_access_replay
                    else ["baseline", *selected_soft_variants, "historical_value_replay"]
                ),
                "route_gradient_inclusion_probability": 0.5,
            },
            "replay_of_sparse_action": previous.action if plan.full_access_replay and previous is not None else None,
            "replay_of_sparse_family": previous.family_id if plan.full_access_replay and previous is not None else None,
        }
        return sampled.stencil, metadata

    def on_step(step: TrainingStep) -> None:
        source_update = step.completed_update - 1
        gradient = gradients_for_update.pop(source_update, {})
        before = pre_step_parameters.pop(source_update, {})
        attempted_identity = global_attempted_for_update.pop(source_update, None)
        cut_records = realized_cut_records_for_update.pop(source_update, None)
        if attempted_identity is None or cut_records is None:
            raise RuntimeError("Completed Thermal update lacks a matching optimizer-attempt snapshot.")
        update_norms = {}
        for name, parameters in (("physical", tuple(value for value in forward_model.parameters() if value.requires_grad)), ("route", tuple(value for value in route_model.parameters() if value.requires_grad))):
            previous = before.get(name, ())
            squared = sum(
                float((parameter.detach() - old).double().square().sum().cpu())
                for parameter, old in zip(parameters, previous, strict=True)
            )
            update_norms[f"{name}_parameter_update_l2"] = math.sqrt(squared)
        metadata = dict(step.training_metadata)
        metadata["calibrated_loss_weights"] = dict(active_loss_weights)
        if weight_transition is not None:
            metadata["loss_weight_transition_id"] = str(weight_transition["transition_id"])
            metadata["loss_weight_transition_source_checkpoint_sha256"] = (
                weight_transition["source_checkpoints"][arm]["sha256"]
            )
            metadata["loss_weight_transition_source_manifest_sha256"] = str(
                weight_transition["source_manifest_sha256"]
            )
            metadata["loss_weight_transition_source_manifest_snapshot_path"] = str(
                weight_transition["source_manifest_snapshot_path"]
            )
            metadata["loss_weight_transition_spec_sha256"] = str(
                weight_transition["transition_spec_sha256"]
            )
            metadata["loss_weight_transition_evidence_sha256"] = str(
                weight_transition["evidence_sha256"]
            )
        cut_evidence = summarize_realized_cut_records(
            cut_records,
            expected_states=(*state_labels, "historical_value_replay"),
            completed_updates_before_attempt=source_update,
            attempted_optimizer_step_including_old_branch=attempted_identity,
            full_access_replay=bool(metadata.get("full_access_replay")),
        )
        metadata["realized_cut_evidence"] = cut_evidence
        metadata["resolved_cut_paths"] = (
            list(cut_evidence["baseline"]["frontier_paths"])
            if cut_evidence["status"] == "realized_sparse_cut"
            else []
        )
        metadata["resolved_cut_paths_status"] = (
            "actual_baseline_hard_route_record"
            if cut_evidence["status"] == "realized_sparse_cut"
            else "full_access_no_cut"
        )
        exposure_counts[(step.training_family_id or "unknown", str(metadata.get("phase")), str(metadata.get("action")))] += 1
        if step.historical_case_id is not None:
            historical_case_counts[str(step.historical_case_id)] += 1
        row = {
            "arm": arm,
            "completed_update": step.completed_update,
            "new_branch_optimizer_update": step.completed_update - START_UPDATE,
            "attempted_optimizer_step_this_invocation": step.attempted_optimizer_step,
            "attempted_optimizer_steps_including_old_branch": attempted_identity,
            "training_stencil_index": step.training_stencil_index,
            "training_family_id": step.training_family_id,
            "training_metadata": metadata,
            "historical_case_id": step.historical_case_id,
            "stage": step.stage,
            "active_terms": list(step.active_terms),
            "total_loss": step.total_loss,
            "loss_terms": dict(step.term_losses),
            "active_term_weights": dict(step.active_term_weights),
            "auxiliary_loss_diagnostics": dict(step.auxiliary_loss_diagnostics),
            "update_wall_seconds": step.update_wall_seconds,
            "update_gpu_milliseconds": step.update_gpu_milliseconds,
            "optimizer_calls": gradient.get("optimizer_calls", 1),
            "gradient_diagnostics": gradient,
            **update_norms,
        }
        _append_jsonl(metric_path, row)
        _append_jsonl(
            exposure_path,
            {
                "arm": arm,
                "completed_update": step.completed_update,
                "new_branch_optimizer_update": step.completed_update - START_UPDATE,
                "family_id": step.training_family_id,
                "primary_pass": metadata.get("primary_pass"),
                "phase": metadata.get("phase"),
                "action": metadata.get("action"),
                "requested_cut_paths": metadata.get("requested_cut_paths"),
                "resolved_cut_paths": metadata.get("resolved_cut_paths"),
                "resolved_cut_paths_status": metadata.get("resolved_cut_paths_status"),
                "realized_cut_evidence": cut_evidence,
                "capacity_fraction": metadata.get("capacity_fraction"),
                "full_access_replay": metadata.get("full_access_replay"),
                "historical_case_id": step.historical_case_id,
                "query_seed": metadata.get("query_seed"),
                "random_tail_query_id_sha256": metadata.get("sampling_summary", {}).get(
                    "random_tail_query_id_sha256", {}
                ),
                "soft_shadow": metadata.get("soft_shadow", {}),
                "timestamp_unix": time.time(),
            },
        )
        if step.completed_update == START_UPDATE + 1 or step.completed_update % 20 == 0 or step.completed_update == args.stop_after_update:
            print(
                json.dumps({
                    "arm": arm,
                    "update": step.completed_update,
                    "new_updates": step.completed_update - START_UPDATE,
                    "family": step.training_family_id,
                    "action": metadata.get("action"),
                    "full_access_replay": metadata.get("full_access_replay"),
                    "loss": step.total_loss,
                    "update_wall_seconds": step.update_wall_seconds,
                    "update_gpu_milliseconds": step.update_gpu_milliseconds,
                    "physical_gradient_preclip": gradient.get("physical_gradient_l2_preclip"),
                    "route_gradient_preclip": gradient.get("route_gradient_l2_preclip"),
                }, sort_keys=True),
                flush=True,
            )

    def select_index(completed: int) -> int:
        return schedule.plan(completed).family_index

    def anchor_auxiliary(completed: int, stencil: Any, predictions: Any, _terms: Mapping[str, torch.Tensor]):
        plan = schedule.plan(completed)
        if not plan.full_access_replay:
            return None, {}
        with torch.no_grad():
            teacher_predictions = forward.predict_stencil(
                inputs["teacher_operator"], stencil, device=device
            )
        anchor = _anchor_loss(predictions, teacher_predictions, stencil, inputs["scales"])
        weighted = float(lambda_anchor) * anchor
        return weighted, {
            "retained_anchor_unweighted_loss": float(anchor.detach().cpu()),
            "retained_anchor_weighted_loss": float(weighted.detach().cpu()),
            "retained_anchor_lambda": float(lambda_anchor),
        }

    def review(step: TrainingStep) -> str:
        snapshot = {
            "arm": arm,
            "completed_update": step.completed_update,
            "new_branch_optimizer_updates": step.completed_update - START_UPDATE,
            "effective_sparse_family_passes": (step.completed_update - START_UPDATE) * 4.0 / 5.0 / len(inputs["family_ids"]),
            "latest_family": step.training_family_id,
            "latest_action": dict(step.training_metadata).get("action"),
            "latest_total_loss": step.total_loss,
            "latest_role_and_history_losses": dict(step.term_losses),
            "cumulative_family_phase_action_visits": {
                f"{family}|{phase}|{action}": count
                for (family, phase, action), count in sorted(exposure_counts.items())
            },
            "new_historical_case_visits": sum(historical_case_counts.values()),
            "unique_historical_cases_seen_in_branch": len(historical_case_counts),
            "update_wall_seconds_recent": [
                json.loads(line)["update_wall_seconds"]
                for line in metric_path.read_text(encoding="utf-8").splitlines()[-40:]
            ] if metric_path.is_file() else [],
            "review_scope": "training_exposure_and_loss_snapshot; fixed/refreshed audit panels are run at post-gate review",
            "timestamp_unix": time.time(),
            "timestamp_utc": _utc_now(),
        }
        pending_reviews[step.completed_update] = snapshot
        return "continue"

    if resume_payload is None:
        # Keep the exact u200 optimizer/model start while opening a new recipe identity.
        sampler = random.Random(config.random_seed)
        branched = checkpoint_payload(
            bundle,
            optimizer,
            arm=config.arm,
            completed_updates=START_UPDATE,
            config=config,
            calibrated_weights=active_loss_weights,
            sampler_rng_state=sampler.getstate(),
            remaining_order=(),
            attempted_optimizer_steps=attempted_base,
            historical_case_order=inputs["historical_order"],
            historical_next_index=START_UPDATE,
        )
        save_checkpoint(branched, "maturation_branch_preflight")

    stop_at_update = args.stop_after_update
    if stop_at_update is None or not initial_update < stop_at_update <= TOTAL_UPDATE_CAP:
        raise ValueError("--stop-after-update must be ahead of the current checkpoint and at most u2600.")
    mixed = tuple(spec for stencil in inputs["raw_stencils"] for spec in _mixed_specs(stencil))
    result = run_staged_fit(
        scheduled,
        bundle,
        optimizer,
        inputs["raw_stencils"],
        scales=inputs["scales"],
        loss_weights=active_loss_weights,
        mixed_specs=mixed,
        historical_value_source=inputs["historical_source"],
        config=config,
        initial_update=initial_update,
        initial_attempted_optimizer_steps=attempted_base,
        resume_payload=resume_payload,
        stop_at_update=stop_at_update,
        device=device,
        on_checkpoint=save_checkpoint,
        on_optimizer_attempt=on_attempt,
        on_review=review,
        on_step=on_step,
        training_stencil_index_for_update=select_index,
        training_stencil_transform=transform,
        auxiliary_loss_fn=anchor_auxiliary,
    )
    family_visits = {
        family: sum(count for (logged_family, _phase, _action), count in exposure_counts.items() if logged_family == family)
        for family in inputs["family_ids"]
    }
    new_historical_visits = sum(historical_case_counts.values())
    if result.stopped_for_wall_time:
        arm_status = "resource_censored"
    elif result.final_update == stop_at_update == START_UPDATE + 100:
        arm_status = "passed_timing_gate"
    elif new_historical_visits >= 1200 and min(family_visits.values(), default=0) >= 100:
        arm_status = "minimum_exposure_reached"
    elif result.final_update == TOTAL_UPDATE_CAP:
        arm_status = "completed_update_cap"
    else:
        arm_status = "stopped_at_requested_update"
    summary = {
        "status": arm_status,
        "arm": arm,
        "initial_update": result.initial_update,
        "final_update": result.final_update,
        "new_branch_updates": result.final_update - START_UPDATE,
        "attempted_optimizer_calls_including_old_branch": result.total_attempted_optimizer_steps,
        "new_attempted_optimizer_calls": result.total_attempted_optimizer_steps - inputs["base_attempted_optimizer_steps"][arm],
        "new_historical_case_visits": new_historical_visits,
        "unique_new_historical_cases_visited": len(historical_case_counts),
        "family_visit_counts": family_visits,
        "minimum_family_visit_count": min(family_visits.values(), default=0),
        "wall_seconds": result.wall_seconds,
        "stopped_for_wall_time": result.stopped_for_wall_time,
        "latest_checkpoint": manifest["arms"].get(arm, {}).get("latest_checkpoint"),
        "latest_checkpoint_sha256": manifest["arms"].get(arm, {}).get("latest_checkpoint_sha256"),
        "training_steps": str(metric_path),
        "exposure_ledger": str(exposure_path),
        "route_work_ledger": str(route_log),
        "gradient_ledger": str(gradient_path),
        "review_ledger": str(reviews_path),
    }
    manifest["arms"][arm].update(summary)
    _atomic_json(output_dir / "run_manifest.json", manifest)
    del bundle, route_model, forward_model, optimizer, paired, native, scheduled
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    amendment_path = getattr(args, "u300_protocol_amendment", None)
    if amendment_path is not None:
        if not args.resume or not args.preflight_only:
            raise ValueError(
                "--u300-protocol-amendment requires --resume --preflight-only and cannot train."
            )
        if tuple(args.arms) != ("G", "P"):
            raise ValueError("The u300 amendment requires the matched G/P run manifest.")
        manifest = _apply_u300_protocol_amendment(
            output_dir / "run_manifest.json",
            amendment_path,
        )
        return {
            "status": "u300_protocol_amended_no_training",
            "run_id": manifest.get("run_id"),
            "output_dir": str(output_dir),
            "source_lineage": manifest["u300_protocol_amendment"],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "lane_execution_ledger.jsonl"
    prior_payload_files = [
        path
        for path in output_dir.rglob("*")
        if (path.is_file() or path.is_symlink()) and path != ledger_path
    ]
    run_id = "Thermal_Controlled_Maturation_from_Run1509_u200_20260929"
    job_started = time.time()
    _record_job_event(
        ledger_path,
        event="job_start",
        run_id=run_id,
        started_at=job_started,
        status="started",
        output_dir=output_dir,
    )
    status = "failed"
    try:
        if tuple(args.arms) != ("G", "P"):
            raise ValueError("The controlled Thermal lane requires the matched G-then-P arm order.")
        if not START_UPDATE < args.stop_after_update <= TOTAL_UPDATE_CAP:
            raise ValueError("--stop-after-update must be ahead of u200 and no later than u2600.")
        output_manifest_path = output_dir / "run_manifest.json"
        retryable_failed_preflight = (
            not output_manifest_path.is_file()
            and all(path in {output_dir / "failures.jsonl"} for path in prior_payload_files)
        )
        if not args.resume and prior_payload_files and not retryable_failed_preflight:
            sample_paths = [str(path) for path in prior_payload_files[:5]]
            raise FileExistsError(
                f"New controlled run directory had preexisting files before this launch: {sample_paths}"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
            raise RuntimeError("Thermal controlled maturation must set CUDA_VISIBLE_DEVICES=2.")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching deterministic CUDA work.")
        requested_direct_scorer_mode = os.environ.get("THERMAL_FACTOR_DIRECT_SCORER", "0")
        if requested_direct_scorer_mode not in {"0", "1"}:
            raise ValueError("THERMAL_FACTOR_DIRECT_SCORER must be 0 or 1.")
        gpu_uuid, gpu_name = _physical_gpu2_identity()
        if gpu_uuid != AUTHORIZED_GPU2_UUID:
            raise RuntimeError(
                f"Physical GPU2 UUID {gpu_uuid} differs from authorized UUID {AUTHORIZED_GPU2_UUID}."
            )
        device = torch.device(args.device)
        if device.type != "cuda" or device.index != 0 or not torch.cuda.is_available():
            raise RuntimeError("Thermal controlled maturation requires physical GPU2 as logical cuda:0.")
        torch.cuda.set_device(device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(7319)
        np.random.seed(7319)
        random.seed(7319)
        inputs = _load_replay_inputs(
            base_run=args.base_run.expanduser().resolve(),
            atlas_dir=args.atlas_dir.expanduser().resolve(),
            reference_checkpoint=args.reference_checkpoint.expanduser().resolve(),
            device=device,
            query_batch_size=args.query_batch_size,
        )
        base_run_manifest = inputs["base_manifest"]
        if args.resume:
            if not output_manifest_path.is_file():
                raise FileNotFoundError("--resume requires an existing controlled-maturation run manifest.")
            manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_id") != run_id or manifest.get("base_checkpoint_sha256") != inputs["base_checkpoint_hashes"]:
                raise ValueError("Existing maturation run does not match exact G/P u200 source hashes.")
            if manifest.get("driver_sha256") != _sha256(Path(__file__).resolve()):
                raise ValueError("Resume driver source differs from the preflight-bound script SHA256.")
            if not manifest.get("preflight", {}).get("status") == "passed":
                raise ValueError("Resume requires a completed, passing controlled-maturation preflight.")
            manifest["status"] = "resuming"
        else:
            manifest = {
                "status": "preflighting",
                "run_id": run_id,
                "started_at_utc": _utc_now(),
                "base_run": str(inputs["base_run_path"]),
                "base_run_manifest": str(inputs["base_manifest_path"]),
                "base_run_manifest_sha256": inputs["base_manifest_sha256"],
                "base_run_evidence_sha256": inputs["base_evidence_sha256"],
                "base_run_status": base_run_manifest.get("status"),
                "base_checkpoint_sha256": inputs["base_checkpoint_hashes"],
                "base_attempted_optimizer_steps": inputs["base_attempted_optimizer_steps"],
                "old_completed_updates_per_arm": START_UPDATE,
                "new_branch_update_cap_per_arm": ADDITIONAL_UPDATE_CAP,
                "new_branch_total_update_cap_per_arm": TOTAL_UPDATE_CAP,
                "dataset_sha256": inputs["dataset_sha256"],
                "reference_checkpoint_sha256": inputs["reference_sha256"],
                "atlas_sha256": inputs["atlas_hashes"],
                "historical_case_count": len(inputs["historical_order"]),
                "historical_case_order_sha256": hashlib.sha256("\n".join(inputs["historical_order"]).encode()).hexdigest(),
                "historical_clock_cursor_at_start": START_UPDATE,
                "schedule_seed": 7319,
                "query_seed_rule": "7319 + relative_update * 104729; same G/P case and schedule stream",
                "capacity_fraction": 0.90,
                "sparse_then_full_cadence": "4 sparse updates : 1 full-access native replay",
                "cut_schedule": "2 complete root-split K=2 scaffold family passes, then two root/two/four passes each, cyclic by shuffled family pass",
                "extra_route": inputs["extra_route"],
                "loss_weights": forward.LOSS_WEIGHTS,
                "loss_scales": _scale_record(inputs["scales"]),
                "anchor_calibration": None,
                "action_descriptor_smoke": None,
                "reference_solver_attempts": 0,
                "resource_caps": {
                    "additional_updates_per_arm": ADDITIONAL_UPDATE_CAP,
                    "new_optimizer_calls_per_arm": ADDITIONAL_UPDATE_CAP,
                    "lane_gpu_hours": LANE_GPU_HOUR_CAP,
                    "aggregate_round_gpu_hours": 48,
                    "aggregate_optimizer_calls": 24000,
                },
                "training_config": asdict(_config(max_wall_seconds=args.max_wall_seconds)),
                "driver_path": str(Path(__file__).resolve()),
                "driver_sha256": _sha256(Path(__file__).resolve()),
                "driver_command": " ".join(sys.argv),
                "driver_command_argv": list(sys.argv),
                "python": sys.executable,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "device": str(device),
                "physical_gpu": 2,
                "physical_gpu_uuid": gpu_uuid,
                "physical_gpu_name": gpu_name,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "arms": {},
                "invocation_history": [],
            }
            preflight = _preflight_record(inputs, device=device, args=args)
            manifest["anchor_calibration"] = preflight["anchor_calibration"]
            manifest["action_descriptor_smoke"] = preflight["action_descriptor_smoke"]
            manifest["preflight"] = preflight
            manifest["invocation_history"].append({
                "kind": "preflight_and_optional_training",
                "timestamp_utc": _utc_now(),
                "command_argv": list(sys.argv),
                "driver_sha256": _sha256(Path(__file__).resolve()),
                "stop_after_update": args.stop_after_update,
                "preflight_only": bool(args.preflight_only),
                "thermal_factor_direct_scorer_requested": requested_direct_scorer_mode == "1",
            })
        if args.resume:
            manifest.setdefault("invocation_history", []).append({
                "kind": "preflight_only_resume" if args.preflight_only else "training_resume",
                "timestamp_utc": _utc_now(),
                "command_argv": list(sys.argv),
                "driver_sha256": _sha256(Path(__file__).resolve()),
                "stop_after_update": args.stop_after_update,
                "preflight_only": bool(args.preflight_only),
                "thermal_factor_direct_scorer_requested": requested_direct_scorer_mode == "1",
            })
        if args.preflight_only:
            manifest["status"] = "preflight_passed_no_training"
            status = manifest["status"]
            manifest["preflight_only_finished_at_utc"] = _utc_now()
            _atomic_json(output_manifest_path, manifest)
            return {
                "status": status,
                "run_id": run_id,
                "output_dir": str(output_dir),
                "preflight": manifest["preflight"],
            }
        _atomic_json(output_manifest_path, manifest)
        pending_arms = [
            arm
            for arm in args.arms
            if not (
                args.resume
                and int(manifest.get("arms", {}).get(arm, {}).get("latest_update", START_UPDATE))
                >= args.stop_after_update
            )
        ]
        prior_lane_seconds = _recorded_lane_active_seconds(ledger_path)
        pretraining_seconds = max(0.0, time.time() - job_started)
        reserved_arm_overhead_seconds = 900.0
        projected_seconds = len(pending_arms) * (args.max_wall_seconds + reserved_arm_overhead_seconds)
        remaining_lane_seconds = LANE_GPU_HOUR_CAP * 3600.0 - prior_lane_seconds - pretraining_seconds
        if projected_seconds > remaining_lane_seconds:
            raise RuntimeError(
                "The requested arm wall limits plus the fixed setup reserve exceed the remaining 24 GPU-hour Thermal lane envelope."
            )
        manifest["lane_budget_preflight"] = {
            "lane_cap_gpu_hours": LANE_GPU_HOUR_CAP,
            "prior_job_active_seconds": prior_lane_seconds,
            "current_pretraining_active_seconds": pretraining_seconds,
            "pending_arms": pending_arms,
            "max_wall_seconds_per_arm": args.max_wall_seconds,
            "reserved_setup_seconds_per_arm": reserved_arm_overhead_seconds,
            "projected_active_seconds": projected_seconds,
            "remaining_active_seconds_before_training": remaining_lane_seconds,
        }
        _atomic_json(output_manifest_path, manifest)
        last_summary: dict[str, Any] | None = None
        last_arm: str | None = None
        for arm in args.arms:
            arm_state = manifest.get("arms", {}).get(arm, {})
            if args.resume and int(arm_state.get("latest_update", START_UPDATE)) >= args.stop_after_update:
                continue
            arm_started = time.time()
            _record_job_event(
                ledger_path,
                event="arm_start",
                run_id=run_id,
                started_at=arm_started,
                status="started",
                output_dir=output_dir,
                arm=arm,
            )
            arm_status = "failed"
            summary = None
            try:
                summary = _run_arm(
                    arm=arm,
                    inputs=inputs,
                    manifest=manifest,
                    output_dir=output_dir,
                    device=device,
                    args=args,
                    lambda_anchor=float(manifest["anchor_calibration"]["selected_shared_lambda"]),
                )
                arm_status = str(summary["status"])
            finally:
                _record_job_event(
                    ledger_path,
                    event="arm_stop",
                    run_id=run_id,
                    started_at=arm_started,
                    status=arm_status,
                    output_dir=output_dir,
                    arm=arm,
                    optimizer_calls=_logged_optimizer_calls(
                        output_dir / "optimizer_attempts.jsonl", arm, since_unix=arm_started
                    ),
                )
            assert summary is not None
            status = summary["status"]
            last_summary = summary
            last_arm = arm
            if summary["final_update"] != args.stop_after_update:
                break
        if last_summary is None or last_arm is None:
            if not all(
                int(manifest.get("arms", {}).get(arm, {}).get("latest_update", START_UPDATE)) >= args.stop_after_update
                for arm in args.arms
            ):
                raise RuntimeError("No Thermal arm was run or found at the requested stop update.")
            last_arm = "P"
            last_summary = manifest["arms"]["P"]
        all_at_target = all(
            int(manifest.get("arms", {}).get(arm, {}).get("final_update", manifest.get("arms", {}).get(arm, {}).get("latest_update", START_UPDATE)))
            >= args.stop_after_update
            for arm in args.arms
        )
        any_wall_censored = any(
            bool(manifest.get("arms", {}).get(arm, {}).get("stopped_for_wall_time", False))
            for arm in args.arms
        )
        if all_at_target and args.stop_after_update == START_UPDATE + 100:
            manifest["status"] = "passed_timing_gate_u300"
        elif all_at_target and args.stop_after_update >= START_UPDATE + 1200:
            manifest["status"] = "minimum_historical_and_family_exposure_reached"
        elif all_at_target and args.stop_after_update == TOTAL_UPDATE_CAP:
            manifest["status"] = "completed_update_cap"
        elif any_wall_censored:
            manifest["status"] = f"resource_censored_after_{last_arm}_u{last_summary['final_update']}"
        elif all_at_target and last_arm != "P":
            manifest["status"] = f"matched_G_P_u{args.stop_after_update}_after_{last_arm}_catchup"
        else:
            manifest["status"] = f"paused_after_{last_arm}_u{last_summary['final_update']}"
        status = manifest["status"]
        manifest["last_invocation_stopped_at_update"] = last_summary["final_update"]
        manifest["last_invocation_last_arm"] = last_arm
        manifest["last_invocation_requested_stop"] = args.stop_after_update
        manifest["last_invocation_finished_at_utc"] = _utc_now()
        _atomic_json(output_manifest_path, manifest)
        return {"status": manifest["status"], "run_id": run_id, "output_dir": str(output_dir), "last_arm_summary": last_summary}
    except Exception as exc:
        _append_jsonl(
            output_dir / "failures.jsonl",
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "timestamp_unix": time.time(),
                "timestamp_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        _record_job_event(
            ledger_path,
            event="job_stop",
            run_id=run_id,
            started_at=job_started,
            status=status,
            output_dir=output_dir,
            optimizer_calls=_logged_optimizer_calls(
                output_dir / "optimizer_attempts.jsonl", since_unix=job_started
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=BASE_RUN)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--arms", nargs="+", choices=("G", "P"), default=("G", "P"))
    parser.add_argument("--stop-after-update", type=int, default=300)
    parser.add_argument("--max-wall-seconds", type=float, default=7200.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--u300-protocol-amendment",
        type=Path,
        default=None,
        help="apply the one-time parity-bound protocol transition at exact G/P u300 (requires --resume --preflight-only)",
    )
    parser.add_argument("--preflight-only", action="store_true", help="write GPU-bound preflight evidence without optimizer updates")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr, flush=True)
        raise
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
