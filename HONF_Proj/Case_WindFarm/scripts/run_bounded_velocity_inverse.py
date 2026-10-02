"""Train and evaluate the opt-in normalized-coordinate Wind velocity inverse.

This maintained entrypoint is intentionally separate from the historical
epsilon-prediction gate/runner. All outputs live in the ignored diagnostics
tree; no new physical solver calls are made.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "HONF_Proj"
CASE = PROJECT / "Case_WindFarm"
for _entry in (PROJECT / "src", CASE / "src", CASE / "scripts"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import evaluate_controlled_maturation_panel as panel
import run_active_packet_reuse as inverse_runner
from honf_inverse_core.models.bounded_velocity import (
    VELOCITY_CHECKPOINT_FORMAT,
    require_velocity_checkpoint,
)
from honf_inverse_core.models.frozen_packet_diffusion import SpatialConditionalPacketDenoiser
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.paths import resolve_path

from windfarm.data import case_batch
from windfarm.inverse.bounded_velocity_completion import (
    WindVelocityCandidatePacketProvider,
    evaluate_matched_wind_velocity_denoising,
    fit_wind_sensor_velocity_transform,
    train_matched_wind_velocity,
    train_wind_velocity_overfit_diagnostic,
    wind_velocity_condition_from_known,
)
from windfarm.inverse.packet_completion import (
    DESIGN_LOWER_D,
    DESIGN_UPPER_D,
    SENSOR_OBSERVED_INDICES,
    WindFixedActionCandidateInterfaceBuilder,
    apply_wind_candidate_context,
    build_compact_public_wind_candidate_context,
    hidden_set_error_D,
    known_from_wind_task,
    make_wind_completion_task,
    wind_geometry_validity,
    wind_surrogate_predictor,
)

FREEZE_PATH = PROJECT / "diagnostics/generated/selected_wind_inverse_20261002/freeze_receipt.json"
PREFLIGHT_PATH = (
    PROJECT
    / "diagnostics/generated/selected_wind_inverse_20261002/cpu_preflight/cpu_feasibility_compact_v6_device_aware_loader.json"
)
CONFIG_PATH = CASE / "configs/active_packet_organizer_reuse.json"
OUTPUT_ROOT = PROJECT / "diagnostics/generated/directed_stable_20261002/inverse"
EXPECTED_GPU2_UUID = "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"
EXPERIMENT_ID = "wind_stable_velocity_inverse"
SOURCE_FREEZE_ID = "Run2112-G-u4910-forced-two_packet"
ACTION_KEY = "two_packet"
CAPACITY = {"QE": 0.95, "MM": 0.90}
STEPS = 20
MODEL_WIDTH = 96
MODEL_LAYERS = 3
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-2
PURE_NOISE_FRACTION = 0.25
SEED = 20261002
TRAIN_LAYOUT_COUNT = 24
EXPECTED_DEV_ROWS = (50, 131)


def _atomic_bytes(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any], *, once: bool = False) -> None:
    path = Path(path)
    if once and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if dict(existing) != dict(value):
            raise FileExistsError(f"Existing stable-inverse protocol differs: {path}")
        return
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _atomic_bytes(path, lambda stream: stream.write(payload))


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    array = np.asarray(value)
    _atomic_bytes(path, lambda stream: np.save(stream, array, allow_pickle=False))


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    arrays = {str(key): np.asarray(value) for key, value in values.items()}
    _atomic_bytes(path, lambda stream: np.savez_compressed(stream, **arrays))


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse_stop_after_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--stop-after-utc must include a timezone, for example 2026-10-03T00:14:00Z")
    return parsed.astimezone(timezone.utc)


def _check_deadline(deadline_utc: datetime | None) -> None:
    if deadline_utc is None:
        return
    now = datetime.now(timezone.utc)
    if now >= deadline_utc:
        raise RuntimeError(f"New inverse training/sampling closed at {deadline_utc.isoformat()}")


def _load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if freeze.get("inverse_outcomes_consulted") is not False:
        raise ValueError("Forward-only freeze must state inverse_outcomes_consulted=false")
    if preflight.get("status") != "cpu_feasibility_pass" or preflight.get("device") != "cpu":
        raise ValueError("Stable velocity fitting needs the existing passing CPU preflight")
    if preflight.get("freeze_receipt_sha256") != inverse_runner._file_sha256(FREEZE_PATH):
        raise ValueError("Existing CPU preflight does not bind the frozen forward receipt")
    legacy_adapter = CASE / "src/windfarm/inverse/packet_completion.py"
    if preflight.get("inverse_adapter_source_sha256") != inverse_runner._file_sha256(legacy_adapter):
        raise ValueError("Historical guarded inverse adapter changed after preflight")
    if preflight.get("selected_g_sha256") != freeze["endpoints"]["g_packet"]["sha256"]:
        raise ValueError("Selected G checkpoint differs from the existing frozen receipt")
    if preflight.get("selected_p_sha256") != freeze["endpoints"]["direct_pair_p"]["sha256"]:
        raise ValueError("Selected P checkpoint differs from the existing frozen receipt")
    if preflight.get("retained_wfull_sha256") != freeze["endpoints"]["retained_wfull_control"]["sha256"]:
        raise ValueError("Retained W-full source differs from the existing frozen receipt")
    for binding in freeze["artifact_hashes"].values():
        path = Path(binding["path"]).resolve()
        if not path.is_file() or inverse_runner._file_sha256(path) != binding["sha256"]:
            raise ValueError(f"Existing frozen artifact binding changed: {path}")
    for name in ("g_packet", "direct_pair_p", "retained_wfull_control"):
        binding = freeze["endpoints"][name]
        path = Path(binding["path"]).resolve()
        if not path.is_file() or inverse_runner._file_sha256(path) != binding["sha256"]:
            raise ValueError(f"Existing frozen endpoint binding changed: {path}")
    if not set(EXPECTED_DEV_ROWS).issubset(set(map(int, preflight["development_task_rows"]))):
        raise ValueError("Preselected row 50/131 development tasks no longer match preflight")
    config = inverse_runner._load_config(CONFIG_PATH)
    return freeze, preflight, config


def _build_context_and_training_tasks(config: Mapping[str, Any], preflight: Mapping[str, Any]):
    view, _canonical, train_rows, split_record = inverse_runner._native_inputs(config)
    locations_path = resolve_path(str(config["dataset"]["locations"]))
    locations = json.loads(locations_path.read_text(encoding="utf-8"))
    compact_path = resolve_path(str(locations[config["dataset"]["compact_dataset_id"]]))
    with np.load(compact_path, allow_pickle=False) as archive:
        compact_axes = tuple(np.asarray(archive[name], dtype=np.float64).copy() for name in ("x_D", "y_D", "z_D"))
    context_binding = preflight["candidate_context"]
    context = build_compact_public_wind_candidate_context(
        [("train", view.run(int(row))) for row in train_rows],
        compact_support_axes_D=compact_axes,
        compact_axes_sha256=str(context_binding["context_source_sha256"]),
    )
    if context.as_dict() != context_binding:
        raise ValueError("Rebuilt train/public-axis candidate context differs from preflight")

    metadata = view.metadata
    layout_by_row = np.asarray(metadata["layout_index"], dtype=np.int64)
    turbines_by_row = np.asarray(metadata["n_turbines"], dtype=np.int64)
    directions_by_row = np.asarray(metadata["wd_deg"], dtype=np.float64)
    xy_by_row = np.asarray(metadata["turbine_xy_D"], dtype=np.float32)
    rows_by_layout: dict[int, list[int]] = {}
    for row in train_rows:
        rows_by_layout.setdefault(int(layout_by_row[int(row)]), []).append(int(row))

    domain_excluded_layouts: list[int] = []
    eligible: list[tuple[int, int, list[int]]] = []
    required_directions = {270.0, 285.0, 300.0}
    for layout, rows in sorted(rows_by_layout.items()):
        rows = sorted(rows, key=lambda row: float(directions_by_row[row]))
        counts = {int(turbines_by_row[row]) for row in rows}
        directions = {float(directions_by_row[row]) for row in rows}
        if len(counts) != 1 or not required_directions.issubset(directions):
            continue
        count = next(iter(counts))
        supported = all(
            np.all(xy_by_row[row, :count] >= DESIGN_LOWER_D[None])
            and np.all(xy_by_row[row, :count] <= DESIGN_UPPER_D[None])
            for row in rows
        )
        if not supported:
            domain_excluded_layouts.append(layout)
            continue
        eligible.append((layout, count, rows))
    if len(eligible) < TRAIN_LAYOUT_COUNT:
        raise ValueError(
            f"Only {len(eligible)} eligible layouts remain under the fixed [-15,15]^2 D box; "
            f"need at least {TRAIN_LAYOUT_COUNT}"
        )
    eligible.sort(key=lambda item: (item[1], item[0]))
    selection = np.rint(np.linspace(0, len(eligible) - 1, TRAIN_LAYOUT_COUNT)).astype(np.int64)
    if len(set(map(int, selection))) != TRAIN_LAYOUT_COUNT:
        raise RuntimeError("Input-selected Wind cohort quantiles did not yield unique layouts")
    selected = [eligible[int(index)] for index in selection]

    training_tasks = []
    cohort_layouts = []
    for layout, count, rows in selected:
        cohort_layouts.append(
            {
                "layout_index": int(layout),
                "turbine_count": int(count),
                "rows_by_direction": [
                    {"row_index": int(row), "wind_direction_deg": float(directions_by_row[row])} for row in rows
                ],
            }
        )
        for row in rows:
            first = apply_wind_candidate_context(
                make_wind_completion_task(view.run(row), partition="train", hidden_count=1, seed=SEED + 2 * row),
                context,
            )
            first_hidden = tuple(np.asarray(first.clean_centers_D[~first.visible_mask, :2]).reshape(-1))
            second = None
            for offset in range(1, 256):
                candidate = apply_wind_candidate_context(
                    make_wind_completion_task(
                        view.run(row),
                        partition="train",
                        hidden_count=1,
                        seed=SEED + 2 * row + offset,
                    ),
                    context,
                )
                candidate_hidden = tuple(np.asarray(candidate.clean_centers_D[~candidate.visible_mask, :2]).reshape(-1))
                if candidate_hidden != first_hidden:
                    second = candidate
                    break
            if second is None:
                raise RuntimeError(f"Could not choose two distinct hidden slots for train row {row}")
            training_tasks.extend((first, second))
    if len({int(task.template_case.layout_index) for task in training_tasks}) != TRAIN_LAYOUT_COUNT:
        raise RuntimeError("Stable inverse train tasks lost a selected layout")
    diagnostic_positions = np.rint(np.linspace(0, TRAIN_LAYOUT_COUNT - 1, 8)).astype(np.int64)
    diagnostic_tasks = [
        training_tasks[int(position) * 6 + 2 * (int(position) % 3)] for position in diagnostic_positions
    ]
    if len({int(task.template_case.layout_index) for task in diagnostic_tasks}) != 8:
        raise RuntimeError("Diagnostic panel must contain eight distinct input-selected train layouts")
    return (
        view,
        train_rows,
        split_record,
        context,
        training_tasks,
        diagnostic_tasks,
        cohort_layouts,
        domain_excluded_layouts,
    )


def _development_tasks(view: Any, preflight: Mapping[str, Any], context: Any) -> list[Any]:
    tasks = []
    for row in EXPECTED_DEV_ROWS:
        if row not in set(map(int, preflight["development_task_rows"])):
            raise ValueError(f"Development row {row} is not in the frozen preselected panel")
        task = make_wind_completion_task(
            view.run(row), partition="development", hidden_count=1, seed=SEED + 49000 + row
        )
        tasks.append(apply_wind_candidate_context(task, context))
    return tasks


def _mask_record(task: Any) -> dict[str, Any]:
    return {
        "row_index": int(task.row_index),
        "layout_index": int(task.template_case.layout_index),
        "wind_direction_deg": float(task.template_case.wind_direction_deg),
        "turbine_count": int(task.template_case.n_turbines),
        "visible_mask": np.asarray(task.visible_mask, dtype=np.uint8).tolist(),
    }


def _trial_plan() -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    for condition in ("original", "changed_sensor14_x"):
        plan.extend(
            (
                {"row": "50", "condition": condition, "trained_model": "I-v-G", "access": "graph"},
                {"row": "50", "condition": condition, "trained_model": "I-v-dense", "access": "full"},
                {"row": "50", "condition": condition, "trained_model": "I-v-G", "access": "forced_full_same_weights"},
            )
        )
    plan.extend(
        (
            {"row": "131", "condition": "original", "trained_model": "I-v-G", "access": "graph"},
            {"row": "131", "condition": "original", "trained_model": "I-v-dense", "access": "full"},
        )
    )
    return plan


def _build_protocol(
    freeze: Mapping[str, Any],
    preflight: Mapping[str, Any],
    split_record: Mapping[str, Any],
    cohort_layouts: Sequence[Mapping[str, Any]],
    domain_excluded_layouts: Sequence[int],
    train_tasks: Sequence[Any],
    diagnostic_tasks: Sequence[Any],
    development_review_tasks: Sequence[Any],
    dev_rows: Sequence[int],
    center: np.ndarray,
    scale: np.ndarray,
    stop_after_utc: datetime | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_id": SOURCE_FREEZE_ID,
        "preflight_path": str(PREFLIGHT_PATH.resolve()),
        "preflight_status": preflight["status"],
        "freeze_receipt_path": str(FREEZE_PATH.resolve()),
        "selected_g_checkpoint": freeze["endpoints"]["g_packet"]["path"],
        "selected_g_checkpoint_sha256": freeze["endpoints"]["g_packet"]["sha256"],
        "retained_wfull_source_checkpoint": freeze["endpoints"]["retained_wfull_control"]["path"],
        "retained_wfull_source_sha256": freeze["endpoints"]["retained_wfull_control"]["sha256"],
        "frozen_action": ACTION_KEY,
        "action_budget_fractions": dict(CAPACITY),
        "forward_limit": "selected Run2112 G u4910 forced two_packet research action; inherited forward adequacy is limited",
        "dataset_split": dict(split_record),
        "target_free_design_box_D": {
            "lower_xy": DESIGN_LOWER_D.tolist(),
            "upper_xy": DESIGN_UPPER_D.tolist(),
            "source": "single shared training-supported coordinate contract; no row-native bounds or per-task scaler",
        },
        "domain_rule": "exclude any layout whose stored training geometry falls outside the fixed shared box; apply once before selection to both arms",
        "domain_excluded_training_layouts": list(map(int, domain_excluded_layouts)),
        "training_cohort": {
            "input_selected_layout_count": len(cohort_layouts),
            "layouts": list(cohort_layouts),
            "rows": sorted({int(task.row_index) for task in train_tasks}),
            "supervised_tasks": len(train_tasks),
            "hidden_slots_per_row": 2,
            "task_masks": [_mask_record(task) for task in train_tasks],
            "directions": [270.0, 285.0, 300.0],
            "selection": "24 turbine-count/layout quantiles among eligible train-only layouts; all available selected directions",
        },
        "overfit_diagnostic_preselected_before_outcomes": {
            "task_masks": [_mask_record(task) for task in diagnostic_tasks],
            "task_count": len(diagnostic_tasks),
            "optimizer_updates": 100,
            "model_count": 1,
            "primary_pair_restarts_from_fresh_baseline_initialization": True,
            "extra_optimizer_updates_counted_against_shared_400_remedy_allowance": 100,
            "no_reverse_trajectory": True,
            "clean_estimate_audit_times": [0.25, 0.5, 0.75, 1.0],
        },
        "primary_denoising_reviews_preselected_before_outcomes": {
            "review_updates_per_arm": [1, 200, 500, 1000],
            "training_task_masks": [_mask_record(task) for task in diagnostic_tasks],
            "development_task_masks": [_mask_record(task) for task in development_review_tasks],
            "audit_times": [0.25, 0.5, 0.75, 1.0],
            "metrics": [
                "masked_v_mse_normalized",
                "masked_v_mae_normalized",
                "clean_coordinate_mse_normalized",
                "clean_coordinate_mae_normalized",
                "clean_coordinate_rmse_D",
                "clean_coordinate_mae_D",
            ],
            "fixed_noise_shared_between_arms": True,
            "candidate_provider_shared_between_arms": True,
            "development_targets_enter_optimizer_or_condition": False,
            "reverse_trajectories_or_physical_solves": 0,
        },
        "sensor_velocity_transform_train_only_mps": {
            "center": np.asarray(center, dtype=np.float32).tolist(),
            "scale": np.asarray(scale, dtype=np.float32).tolist(),
            "unique_training_rows": len({int(task.row_index) for task in train_tasks}),
        },
        "training": {
            "prediction_type": "v=a*epsilon-b*z0",
            "state_parameterization": "affine_normalized_xy[-1,1]",
            "noise_schedule": "a=cos(pi*t/2), b=sin(pi*t/2), exact endpoints",
            "provider_proxy": "clip generated normalized state to [-1,1]; denoiser sees unclipped state; visible states exact",
            "sampler_steps": STEPS,
            "denoiser": "SpatialConditionalPacketDenoiser",
            "hidden_width": MODEL_WIDTH,
            "layers": MODEL_LAYERS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": 4,
            "implementation": "four matched micro-examples with one optimizer step after accumulated gradients",
            "pure_noise_endpoint_fraction": PURE_NOISE_FRACTION,
            "seed": SEED,
            "default_cumulative_updates_per_arm": 1000,
            "review_updates_per_arm": [200, 500, 1000],
        },
        "development_panel_preselected_before_outcomes": {
            "rows": list(map(int, dev_rows)),
            "hidden_count": 1,
            "task_seed_rule": "SEED + 49000 + row_index",
            "native_reference_available_only_for_scoring": True,
        },
        "sample_panel": {
            "max_attempted_or_completed_trajectories": 8,
            "planned_trials": _trial_plan(),
            "condition_definitions": {
                "original": "original historical row-50 observations",
                "changed_sensor14_x": (
                    "row 50 with only Ux at observed-array slot 14 (raw grid sensor index 22; "
                    "intended coordinate [12.2, 5.0, 0.875]D) replaced by the farther endpoint "
                    "of the current matched train-task range; this panel gives 8.299531 to "
                    "5.537949 m/s; sensor coordinates and physical reference stay fixed"
                ),
            },
            "same_initial_noise_per_task_across_all_conditions_and_arms": True,
            "per_step_raw_clean_and_state_persistence": True,
            "graph_full_intervention": "I-v-G graph versus same trained I-v-G forced-full access",
            "no_draw_retry_after_sampling_or_scoring_failure": True,
        },
        "new_physical_solver_calls": 0,
        "test_partition_opened": False,
        "operational_stop_after_utc": (None if stop_after_utc is None else stop_after_utc.isoformat()),
    }


def _save_protocol(path: Path, protocol: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if dict(current) != dict(protocol):
            current_sample = current.get("sample_panel", {})
            current_definitions = current_sample.get("condition_definitions", {})
            protocol_sample = protocol.get("sample_panel", {})
            protocol_definitions = protocol_sample.get("condition_definitions", {})
            old_index_label = (
                "row 50 with only the observed Ux value at raw sensor index 14 changed "
                "from 8.299531 to 5.537949 m/s; sensor coordinates and physical reference stay fixed"
            )
            current_corrected = json.loads(json.dumps(current))
            if current_definitions.get("changed_sensor14_x") == old_index_label:
                current_corrected["sample_panel"]["condition_definitions"]["changed_sensor14_x"] = (
                    protocol_definitions.get("changed_sensor14_x")
                )
            if current_corrected == dict(protocol):
                # A single known pre-draw label correction: prior prose confused
                # the observed-array index with the raw 24-slot sensor index.
                _atomic_json(path, protocol)
                return
            extension_key = "primary_denoising_reviews_preselected_before_outcomes"
            condition_key = "condition_definitions"
            new_top_level = set(protocol).difference(current)
            expected_new_top_level = {extension_key} if extension_key not in current else set()
            new_sample_fields = set(protocol_sample).difference(current_sample)
            expected_new_sample_fields = {condition_key} if condition_key not in current_sample else set()
            existing_fields_match = all(
                current[key] == protocol[key] for key in current if key != "sample_panel"
            ) and all(current_sample[key] == protocol_sample[key] for key in current_sample)
            is_preoutcome_extension = (
                new_top_level == expected_new_top_level
                and new_sample_fields == expected_new_sample_fields
                and existing_fields_match
            )
            if not is_preoutcome_extension:
                raise FileExistsError("Existing stable-inverse protocol differs from the pre-outcome panel")
            _atomic_json(path, protocol)
    else:
        _atomic_json(path, protocol, once=True)


def _check_gpu2(device_text: str) -> torch.device:
    if device_text != "cuda:2":
        raise ValueError("The stable Wind inverse is allocated explicitly to physical cuda:2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("Set ordinary GPU visibility; CUDA_VISIBLE_DEVICES must be unset or empty")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    physical = {}
    for line in query.stdout.splitlines():
        index, uuid = (value.strip() for value in line.split(",", 1))
        physical[int(index)] = uuid
    if physical.get(2) != EXPECTED_GPU2_UUID:
        raise RuntimeError(f"Physical GPU2 UUID changed: expected {EXPECTED_GPU2_UUID}, got {physical.get(2)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 2:
        raise RuntimeError("Physical cuda:2 is unavailable")
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
    if torch.cuda.current_device() != 2:
        raise RuntimeError("PyTorch current device is not physical GPU index 2")
    return device


def _load_forward_provider(freeze: Mapping[str, Any], config: Mapping[str, Any], task: Any, device: torch.device):
    g_binding = freeze["endpoints"]["g_packet"]
    g_path = Path(g_binding["path"]).resolve()
    g_payload = load_trusted_checkpoint(g_path, map_location="cpu")
    if not isinstance(g_payload, Mapping):
        raise TypeError("Selected G u4910 checkpoint is not a trusted mapping")
    source_path, source_payload, normalizer, source_sha = inverse_runner._load_source(config)
    wfull_binding = freeze["endpoints"]["retained_wfull_control"]
    if source_path.resolve() != Path(wfull_binding["path"]).resolve():
        raise ValueError("G materialization source is not the frozen Run2110 W-full checkpoint")
    if source_sha != wfull_binding["sha256"]:
        raise ValueError("G materialization source differs from the frozen W-full SHA")
    known = known_from_wind_task(task)
    batch = case_batch(
        task.template_case,
        known.observed_coordinates_D,
        velocity_mps=None,
        include_receiver_anchors=True,
    ).to(device)
    model, organizer = panel._load_student(
        arm="g",
        payload=g_payload,
        source_payload=source_payload,
        normalizer=normalizer,
        batch=batch,
        forward_config=config["forward"],
        device=device,
    )
    model.eval().requires_grad_(False)
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    organizer.eval().requires_grad_(False)
    if getattr(organizer, "frontier_utility_head", None) is not None:
        raise ValueError("Frozen G u4910 must not acquire a new utility-head interpretation")
    return model, organizer


def _make_denoiser_template(
    task: Any,
    center: np.ndarray,
    scale: np.ndarray,
    embedding_dim: int,
) -> SpatialConditionalPacketDenoiser:
    known = known_from_wind_task(task)
    condition = wind_velocity_condition_from_known(
        known,
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
        device="cpu",
    )
    return SpatialConditionalPacketDenoiser(
        coordinate_dim=2,
        design_dim=2,
        module_dim=int(condition.module_features.shape[-1]),
        sensor_dim=int(condition.sensor_features.shape[-1]),
        embedding_dim=int(embedding_dim),
        hidden_dim=MODEL_WIDTH,
        layers=MODEL_LAYERS,
    )


def _provider_factory(model: torch.nn.Module, organizer: torch.nn.Module, context: Any, g_sha: str):
    def factory(known):
        builder = WindFixedActionCandidateInterfaceBuilder(
            model,
            organizer,
            context=context,
            action_key=ACTION_KEY,
            budget_fractions=CAPACITY,
            numerical_state_version=f"selected-G-u4910-velocity-{g_sha[:16]}",
        )
        return WindVelocityCandidatePacketProvider(known, builder)

    return factory


def _expected_checkpoint_contract(
    protocol: Mapping[str, Any],
    train_tasks: Sequence[Any],
    center: np.ndarray,
    scale: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    return {
        "checkpoint_format": VELOCITY_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_id": SOURCE_FREEZE_ID,
        "seed": int(seed),
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": 4,
        "pure_noise_fraction": PURE_NOISE_FRACTION,
        "train_task_ids": [
            {
                "row_index": int(task.row_index),
                "layout_index": int(task.template_case.layout_index),
                "wind_direction_deg": float(task.template_case.wind_direction_deg),
                "turbine_count": int(task.template_case.n_turbines),
                "visible_mask": np.asarray(task.visible_mask, dtype=np.uint8).tolist(),
            }
            for task in train_tasks
        ],
        "sensor_velocity_center_mps": np.asarray(center, dtype=np.float32).tolist(),
        "sensor_velocity_scale_mps": np.asarray(scale, dtype=np.float32).tolist(),
        "prediction_type": "v=a*epsilon-b*z0",
        "state_parameterization": "affine_normalized_xy[-1,1]",
    }


def _load_velocity_models(
    checkpoint_path: Path,
    denoiser_template: torch.nn.Module,
    expected_contract: Mapping[str, Any],
    device: torch.device,
):
    payload = load_trusted_checkpoint(Path(checkpoint_path).resolve(), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Stable velocity checkpoint is not a trusted mapping")
    require_velocity_checkpoint(payload)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("source_freeze_id") != SOURCE_FREEZE_ID:
        raise ValueError("Velocity checkpoint belongs to a different matched experiment or frozen source")
    for key, value in expected_contract.items():
        if payload.get(key) != value:
            raise ValueError(f"Stable velocity checkpoint contract mismatch for {key}")
    updates = int(payload.get("update_count", -1))
    if updates < 200:
        raise ValueError("Final Wind sample panel requires at least 200 matched updates per arm")
    from copy import deepcopy

    from honf_inverse_core.models.bounded_velocity import BoundedVelocityPacketDiffusion

    graph = BoundedVelocityPacketDiffusion(deepcopy(denoiser_template), steps=STEPS).to(device)
    dense = BoundedVelocityPacketDiffusion(deepcopy(denoiser_template), steps=STEPS).to(device)
    graph.load_state_dict(payload["graph_state_dict"], strict=True)
    dense.load_state_dict(payload["dense_state_dict"], strict=True)
    graph.eval()
    dense.eval()
    return graph, dense, updates, payload


def _links_as_arrays(links: Any) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name in (
        "module_source",
        "sensor_source",
        "module_embeddings",
        "module_environment",
        "environment_module",
        "sensor_environment",
        "environment_embeddings",
        "environment_valid",
        "module_coordinates",
        "sensor_coordinates",
    ):
        value = getattr(links, name)
        if value is not None:
            values[name] = value.detach().cpu().numpy()
    return values


def _condition_changed_sensor14(
    known: Any,
    training_tasks: Sequence[Any],
    task: Any,
) -> tuple[Any, dict[str, Any]]:
    # Preserve historical observed-array slot 14. It maps to raw grid index
    # 22; confusing those two index spaces caused the previous preflight stop.
    observed_position = 14
    raw_sensor_index = int(tuple(SENSOR_OBSERVED_INDICES)[observed_position])
    component = 0
    observed = np.asarray(known.observed_velocity_mps, dtype=np.float32).copy()
    original = float(observed[observed_position, component])
    if not np.isclose(original, 8.299531, rtol=0.0, atol=2.0e-5):
        raise ValueError(
            f"Preselected row-50 observed-array slot 14 Ux differs from the frozen historical panel: {original}"
        )
    train_values = np.stack(
        [known_from_wind_task(train_task).observed_velocity_mps for train_task in training_tasks], axis=0
    )
    training_min = float(train_values[:, observed_position, component].min())
    training_max = float(train_values[:, observed_position, component].max())
    changed_value = training_min if abs(original - training_min) >= abs(original - training_max) else training_max
    observed[observed_position, component] = np.float32(changed_value)
    changed = replace(known, observed_velocity_mps=observed)
    return changed, {
        "observed_array_slot": observed_position,
        "raw_sensor_index": raw_sensor_index,
        "channel": "Ux",
        "original_value_mps": original,
        "changed_value_mps": float(observed[observed_position, component]),
        "training_min_mps": training_min,
        "training_max_mps": training_max,
        "endpoint_rule": "absolute replacement by the farther training-only min/max endpoint at the same observed slot and component",
        "intended_coordinate_D": np.asarray(known.observed_coordinates_D[observed_position]).tolist(),
        "native_coordinate_D": np.asarray(task.sensors.native_coordinates_D[raw_sensor_index]).tolist(),
    }


def _rms_generated(value: torch.Tensor, condition: Any) -> float:
    mask = condition.design_mask.to(value.dtype).unsqueeze(-1)
    count = float(mask.sum().cpu()) * int(value.shape[-1])
    return float((((value.square() * mask).sum() / max(count, 1.0)).sqrt()).cpu())


def _sample_one(
    *,
    trial_index: int,
    plan: Mapping[str, str],
    task: Any,
    known: Any,
    condition: Any,
    initial_noise: torch.Tensor,
    model: Any,
    dense: bool,
    builder_factory: Any,
    predictor: Any,
    context: Any,
    output_root: Path,
    updates: int,
    stop_after_utc: datetime | None,
    changed_sensor: Mapping[str, Any] | None,
    row50_changed_observed_velocity_mps: np.ndarray | None,
) -> dict[str, Any]:
    attempt_name = (
        f"trail_{trial_index:02d}_row{int(task.row_index)}_{plan['condition']}_{plan['trained_model']}_{plan['access']}"
    )
    attempt_dir = output_root / "trails" / attempt_name
    start_path = attempt_dir / "attempt_started.json"
    if attempt_dir.exists():
        raise FileExistsError(f"Refusing to redraw or overwrite an already attempted trail: {attempt_dir}")
    attempt_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        start_path,
        {
            "status": "attempt_started",
            "trial_index": int(trial_index),
            "plan": dict(plan),
            "row_index": int(task.row_index),
            "layout_index": int(task.template_case.layout_index),
            "checkpoint_update_count": int(updates),
            "initial_noise_seed_is_task_shared": True,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "new_solver_calls": 0,
        },
    )
    sample_started = time.perf_counter()

    def save_access(step_index: int, current_time: float, links: Any) -> None:
        arrays = _links_as_arrays(links)
        if not arrays:
            raise RuntimeError("Frozen provider did not return auditable access tensors")
        arrays["current_time"] = np.asarray([current_time], dtype=np.float32)
        arrays["module_source_nonzero"] = np.asarray([np.count_nonzero(arrays["module_source"])], dtype=np.int64)
        arrays["sensor_source_nonzero"] = np.asarray([np.count_nonzero(arrays["sensor_source"])], dtype=np.int64)
        _atomic_npz(attempt_dir / f"links_{step_index + 1:02d}.npz", arrays)

    def save_progress(
        completed_step: int,
        current_time: float,
        next_time: float,
        state: torch.Tensor,
        raw_clean: torch.Tensor | None,
    ) -> None:
        _atomic_npy(attempt_dir / f"state_{completed_step:02d}.npy", state.detach().cpu().numpy())
        record: dict[str, Any] = {
            "completed_step": int(completed_step),
            "current_time": float(current_time),
            "next_time": float(next_time),
            "generated_state_rms": _rms_generated(state, condition),
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        if raw_clean is not None:
            _atomic_npy(
                attempt_dir / f"raw_clean_estimate_{completed_step:02d}.npy",
                raw_clean.detach().cpu().numpy(),
            )
            record["raw_clean_time"] = float(current_time)
            record["generated_raw_clean_rms"] = _rms_generated(raw_clean, condition)
        _atomic_json(attempt_dir / f"step_{completed_step:02d}.json", record)

    def check_sample_step(_step_index: int, _current_time: float) -> None:
        _check_deadline(stop_after_utc)

    try:
        provider = builder_factory(known)
        trail = model.sample(
            condition,
            provider,
            dense=dense,
            initial_noise=initial_noise,
            progress_callback=save_progress,
            access_callback=save_access,
            pre_step_check=check_sample_step,
        )
        raw = trail.final_state.detach().cpu()
        projected = trail.projected_final_state.detach().cpu()
        lower = condition.design_lower[0].detach().cpu()
        upper = condition.design_upper[0].detach().cpu()
        from windfarm.inverse.bounded_velocity_completion import decode_normalized_wind_design

        raw_xy = decode_normalized_wind_design(raw[0], lower, upper).numpy()
        projected_xy = decode_normalized_wind_design(projected[0], lower, upper).numpy()
        height = np.full(
            (raw_xy.shape[0], 1),
            float(task.template_case.hub_height_m / task.template_case.diameter_m),
            dtype=np.float32,
        )
        raw_centers = np.concatenate((raw_xy, height), axis=-1).astype(np.float32)
        projected_centers = np.concatenate((projected_xy, height), axis=-1).astype(np.float32)
        visible = np.asarray(known.visible_mask, dtype=bool)
        visible_centers = np.asarray(known.template_case.module_centers, dtype=np.float32)
        projected_centers[visible] = visible_centers[visible]
        raw_centers[visible] = visible_centers[visible]
        radius = float(task.template_case.module_features[0, 0])
        projected_geometry = wind_geometry_validity(
            projected_centers,
            radius,
            support_lower_D=context.support.lower_D,
            support_upper_D=context.support.upper_D,
            support_name="compact_public_support",
        )
        raw_geometry = wind_geometry_validity(raw_centers, radius, support_name="declared_design_only")
        _atomic_npy(attempt_dir / "terminal_raw_normalized.npy", raw.numpy())
        _atomic_npy(attempt_dir / "terminal_projected_normalized.npy", projected.numpy())
        _atomic_npy(attempt_dir / "terminal_projection_mask.npy", trail.projection_mask.detach().cpu().numpy())
        _atomic_npy(attempt_dir / "terminal_raw_centers_D.npy", raw_centers)
        _atomic_npy(attempt_dir / "terminal_projected_centers_D.npy", projected_centers)
        _atomic_json(
            attempt_dir / "terminal_projection.json",
            {
                "projection_coordinate_fraction": trail.projection_coordinate_fraction,
                "generated_coordinate_count": trail.generated_coordinate_count,
                "overshoot_coordinate_count": int(trail.projection_mask.sum().cpu()),
                "raw_terminal_generated_rms": _rms_generated(trail.final_state, condition),
                "projected_terminal_generated_rms": _rms_generated(trail.projected_final_state, condition),
                "raw_terminal_geometry": raw_geometry,
                "projected_terminal_geometry": projected_geometry,
                "projected_minimum_spacing_D": projected_geometry["min_spacing_D"],
                "rotor_radius_D": radius,
                "repair_applied": False,
            },
        )
        _atomic_json(
            attempt_dir / "trail_persisted.json",
            {
                "status": "all_native_reverse_states_and_terminal_projection_persisted",
                "states_saved": len(trail.states),
                "raw_clean_estimates_saved": len(trail.clean_estimates),
                "link_steps_saved": trail.organizer_calls,
                "elapsed_seconds": time.perf_counter() - sample_started,
                "scoring_started_after_persistence": True,
            },
        )

        native_geometry = wind_geometry_validity(
            projected_centers,
            radius,
            support_lower_D=np.asarray(task.native_support_lower_D, dtype=np.float32),
            support_upper_D=np.asarray(task.native_support_upper_D, dtype=np.float32),
            support_name="native_domain",
        )
        _atomic_json(
            attempt_dir / "native_support_evaluation.json",
            {
                "status": "evaluated_after_full_trajectory_persistence",
                "target_bounds_used_in_provider_or_condition": False,
                "projected_native_row_geometry": native_geometry,
                "native_row_support_bounds_D": {
                    "lower": np.asarray(task.native_support_lower_D, dtype=np.float32).tolist(),
                    "upper": np.asarray(task.native_support_upper_D, dtype=np.float32).tolist(),
                },
                "public_domain_geometry": projected_geometry,
            },
        )

        # Development targets are opened for scoring only after every reverse
        # state, raw estimate, link trace, and terminal projection is durable.
        observed_coordinates = np.asarray(known.observed_coordinates_D, dtype=np.float32)
        held_coordinates = np.asarray(known.held_coordinates_D, dtype=np.float32)
        observed_target = np.asarray(known.observed_velocity_mps, dtype=np.float32)
        held_target = np.asarray(known.held_velocity_mps, dtype=np.float32)
        coordinates = np.concatenate((observed_coordinates, held_coordinates), axis=0).astype(np.float32)
        target = np.concatenate((observed_target, held_target), axis=0).astype(np.float32)
        raw_prediction = predictor(
            replace(
                known.template_case,
                module_centers=raw_centers.copy(),
                receiver_anchor_coords=None,
                receiver_anchor_weights=None,
                receiver_anchor_roles=None,
            ),
            coordinates,
        )
        predicted = predictor(
            replace(
                known.template_case,
                module_centers=projected_centers.copy(),
                receiver_anchor_coords=None,
                receiver_anchor_weights=None,
                receiver_anchor_roles=None,
            ),
            coordinates,
        )
        clean_candidate = replace(
            known.template_case,
            module_centers=np.asarray(task.clean_centers_D, dtype=np.float32).copy(),
            receiver_anchor_coords=None,
            receiver_anchor_weights=None,
            receiver_anchor_roles=None,
        )
        clean_prediction = predictor(clean_candidate, coordinates)
        _atomic_npz(
            attempt_dir / "final_surrogate_audit_arrays.npz",
            {
                "observed_coordinates_D": observed_coordinates,
                "held_coordinates_D": held_coordinates,
                "observed_target_velocity_mps": observed_target,
                "held_target_velocity_mps": held_target,
                "clean_reference_module_centers_D": np.asarray(task.clean_centers_D, dtype=np.float32),
                "visible_module_mask": np.asarray(task.visible_mask, dtype=np.uint8),
                "raw_terminal_prediction_velocity_mps": np.asarray(raw_prediction, dtype=np.float32),
                "projected_terminal_prediction_velocity_mps": np.asarray(predicted, dtype=np.float32),
                "clean_layout_prediction_velocity_mps": np.asarray(clean_prediction, dtype=np.float32),
            },
        )
        _atomic_json(
            attempt_dir / "scoring_arrays_persisted.json",
            {
                "status": "raw_predictions_targets_and_coordinates_persisted_before_scoring",
                "observed_count": len(observed_coordinates),
                "held_count": len(held_coordinates),
                "prediction_rows": len(predicted),
                "raw_and_projected_predictions_saved": True,
            },
        )
        observed_count = len(known.observed_coordinates_D)
        observed_error = float(np.sqrt(np.mean(np.square(predicted[:observed_count] - target[:observed_count]))))
        held_error = float(np.sqrt(np.mean(np.square(predicted[observed_count:] - target[observed_count:]))))
        clean_observed_error = float(
            np.sqrt(np.mean(np.square(clean_prediction[:observed_count] - target[:observed_count])))
        )
        changed_input_error = None
        original_observed_cross_error = None
        changed_observed_cross_error = None
        if changed_sensor is not None:
            changed_input_error = observed_error
        if int(task.row_index) == 50:
            if row50_changed_observed_velocity_mps is None:
                raise ValueError("Row-50 paired cross-scores require the predeclared changed observation")
            original_target = np.asarray(task.sensors.reference_velocity_mps, dtype=np.float32)[
                np.asarray(SENSOR_OBSERVED_INDICES, dtype=np.int64)
            ]
            changed_target = np.asarray(row50_changed_observed_velocity_mps, dtype=np.float32)
            changed_observed_cross_error = float(
                np.sqrt(np.mean(np.square(predicted[:observed_count] - changed_target)))
            )
            original_observed_cross_error = float(
                np.sqrt(np.mean(np.square(predicted[:observed_count] - original_target)))
            )
        hidden_error = hidden_set_error_D(
            projected_centers, np.asarray(task.clean_centers_D), np.asarray(task.visible_mask)
        )
        geometry_valid = bool(
            projected_geometry["finite"]
            and projected_geometry["inside_design_box"]
            and projected_geometry["inside_support"]
            and projected_geometry["rotors_nonoverlap"]
            and native_geometry["inside_native_domain"] is True
        )
        acceptable_surrogate = bool(geometry_valid and observed_error <= clean_observed_error + 0.5)
        outcome = {
            "status": "completed_scored",
            "trial_index": int(trial_index),
            "trial_id": attempt_name,
            "plan": dict(plan),
            "row_index": int(task.row_index),
            "layout_index": int(task.template_case.layout_index),
            "wind_direction_deg": float(task.template_case.wind_direction_deg),
            "turbine_count": int(task.template_case.n_turbines),
            "checkpoint_update_count": int(updates),
            "training_example_exposure_per_arm": int(updates * 4),
            "organizer_calls": int(trail.organizer_calls),
            "provider_candidate_calls": int(provider.calls),
            "initial_noise_shared_per_task": True,
            "initial_noise_seed": int(SEED + 73001 + int(task.row_index)),
            "projection_coordinate_fraction": trail.projection_coordinate_fraction,
            "raw_projected_geometry": projected_geometry,
            "raw_unprojected_geometry": raw_geometry,
            "public_domain_geometry": projected_geometry,
            "native_row_geometry": native_geometry,
            "public_domain_valid": bool(
                projected_geometry["finite"]
                and projected_geometry["inside_design_box"]
                and projected_geometry["inside_support"]
                and projected_geometry["rotors_nonoverlap"]
            ),
            "native_row_support_valid": native_geometry["inside_native_domain"] is True,
            "hidden_set_error_D": hidden_error,
            "surrogate_observed_rmse_mps": observed_error,
            "surrogate_held_rmse_mps": held_error,
            "surrogate_clean_layout_observed_rmse_mps": clean_observed_error,
            "surrogate_changed_input_rmse_mps": changed_input_error,
            "row50_original_observation_cross_score_mps": original_observed_cross_error,
            "row50_changed_observation_cross_score_mps": changed_observed_cross_error,
            "changed_condition_comparison_target": (
                "same changed observations for original and changed draws" if int(task.row_index) == 50 else None
            ),
            "surrogate_candidate_acceptable_not_CFD": acceptable_surrogate,
            "new_CFD_reference": False,
            "new_solver_calls": 0,
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        _atomic_json(attempt_dir / "outcome.json", outcome)
        attempt_record = json.loads(start_path.read_text(encoding="utf-8"))
        attempt_record.update(
            {
                "status": "completed_scored",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(start_path, attempt_record)
        return outcome
    except Exception as error:  # noqa: BLE001 - preserve the consumed trail slot and partial artifacts
        attempt_record = json.loads(start_path.read_text(encoding="utf-8"))
        attempt_record.update(
            {
                "status": "failed_attempt_no_retry",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _atomic_json(start_path, attempt_record)
        _atomic_json(
            attempt_dir / "failure.json",
            {
                "status": "failed_attempt_no_retry",
                "trial_index": int(trial_index),
                "plan": dict(plan),
                "row_index": int(task.row_index),
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - sample_started,
                "raw_progress_retained": True,
            },
        )
        return {
            "status": "failed_attempt_no_retry",
            "trial_index": int(trial_index),
            "trial_id": attempt_name,
            "plan": dict(plan),
            "row_index": int(task.row_index),
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - sample_started,
        }


def _run_sample_panel(
    *,
    freeze: Mapping[str, Any],
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    protocol: Mapping[str, Any],
    view: Any,
    context: Any,
    train_tasks: Sequence[Any],
    center: np.ndarray,
    scale: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    output_root: Path,
    seed: int,
    stop_after_utc: datetime | None,
) -> list[dict[str, Any]]:
    _check_deadline(stop_after_utc)
    g_model, organizer = _load_forward_provider(freeze, config, train_tasks[0], device)
    g_sha = str(freeze["endpoints"]["g_packet"]["sha256"])
    builder_factory = _provider_factory(g_model, organizer, context, g_sha)
    template = _make_denoiser_template(train_tasks[0], center, scale, int(organizer.node_encoder[0].out_features))
    expected = _expected_checkpoint_contract(protocol, train_tasks, center, scale, seed)
    graph_model, dense_model, updates, payload = _load_velocity_models(checkpoint_path, template, expected, device)
    if payload.get("forward_and_organizer_frozen") is not True:
        raise ValueError("Matched velocity checkpoint lacks frozen-provider training evidence")
    dev_tasks = _development_tasks(view, preflight, context)
    task_by_row = {int(task.row_index): task for task in dev_tasks}
    if set(task_by_row) != set(EXPECTED_DEV_ROWS):
        raise ValueError("Stable inverse sample panel must contain exactly row 50 and row 131")
    _atomic_json(
        output_root / "development_task_selection.json",
        {
            "rows": [_mask_record(task_by_row[row]) for row in EXPECTED_DEV_ROWS],
            "selection_rule": "preflight rows 50 and 131; hidden slot chosen with predeclared row seed",
            "hidden_target_coordinates_included": False,
        },
        once=True,
    )

    known_by_row = {row: known_from_wind_task(task_by_row[row]) for row in EXPECTED_DEV_ROWS}
    changed_known, changed_description = _condition_changed_sensor14(known_by_row[50], train_tasks, task_by_row[50])
    changed_target = np.asarray(changed_known.observed_velocity_mps, dtype=np.float32).copy()
    _atomic_json(
        output_root / "sample_preflight_correction.json",
        {
            "status": "corrected_pre_draw_intervention_mapping",
            "condition_label": "changed_sensor14_x",
            "previous_label_interpretation": "raw sensor index 14 (incorrect index space)",
            "corrected_interpretation": "observed-array slot 14 maps to raw grid sensor index 22",
            "observed_array_slot": changed_description["observed_array_slot"],
            "raw_sensor_index": changed_description["raw_sensor_index"],
            "component": changed_description["channel"],
            "intended_coordinate_D": changed_description["intended_coordinate_D"],
            "current_native_coordinate_D": changed_description["native_coordinate_D"],
            "original_value_mps": changed_description["original_value_mps"],
            "changed_value_mps": changed_description["changed_value_mps"],
            "current_training_endpoint_mps": {
                "minimum": changed_description["training_min_mps"],
                "maximum": changed_description["training_max_mps"],
            },
            "training_task_count": len(train_tasks),
            "replacement_rule": changed_description["endpoint_rule"],
            "changed_position_and_component_fixed": True,
            "sensor_coordinates_and_stored_physical_reference_unchanged": True,
            "prior_preflight_process": {
                "status": "rejected_before_sample_loop",
                "native_reverse_steps": 0,
                "attempt_started_records": 0,
                "trajectory_slots_consumed": 0,
                "process_wall_seconds": None,
                "estimated_upper_bound_seconds": 60,
                "wall_time_measured": False,
                "wall_time_note": "Elapsed process wall was not persisted by the prior entrypoint; charge one conservative 60-second upper-bound estimate, not a measured duration.",
            },
        },
        once=True,
    )
    mean = np.asarray(center, dtype=np.float32)
    std = np.asarray(scale, dtype=np.float32)
    conditions: dict[tuple[int, str], tuple[Any, Any]] = {}
    for row in EXPECTED_DEV_ROWS:
        known = known_by_row[row]
        conditions[(row, "original")] = (
            known,
            wind_velocity_condition_from_known(
                known,
                sensor_velocity_center_mps=mean,
                sensor_velocity_scale_mps=std,
                device=device,
            ),
        )
    conditions[(50, "changed_sensor14_x")] = (
        changed_known,
        wind_velocity_condition_from_known(
            changed_known,
            sensor_velocity_center_mps=mean,
            sensor_velocity_scale_mps=std,
            device=device,
        ),
    )

    trial_plan = _trial_plan()
    if len(trial_plan) != 8:
        raise RuntimeError("The predeclared native trajectory panel must contain exactly eight trials")
    existing = list((output_root / "trails").glob("trail_*/attempt_started.json"))
    if existing:
        raise FileExistsError(f"Found {len(existing)} already attempted trajectories; no rerun or redraw is permitted")
    if len(trial_plan) > 8:
        raise RuntimeError("Native trajectory ceiling exceeded before sampling")

    noise_by_row: dict[int, torch.Tensor] = {}
    noise_root = output_root / "initial_noise"
    for row in EXPECTED_DEV_ROWS:
        condition = conditions[(row, "original")][1]
        noise_generator = torch.Generator(device=device).manual_seed(seed + 73001 + row)
        noise = torch.randn(
            condition.known_state.shape,
            dtype=condition.known_state.dtype,
            device=device,
            generator=noise_generator,
        )
        noise_by_row[row] = noise
        _atomic_npy(noise_root / f"row{row}_shared_initial_noise.npy", noise.detach().cpu().numpy())

    predictor = wind_surrogate_predictor(g_model)
    outcome_path = output_root / "sample_outcomes.jsonl"
    outcomes = []
    for index, trial in enumerate(trial_plan):
        if stop_after_utc is not None and datetime.now(timezone.utc) >= stop_after_utc:
            break
        row = int(trial["row"])
        condition_name = trial["condition"]
        known, condition = conditions[(row, condition_name)]
        trained_name = trial["trained_model"]
        use_dense = trial["access"] in {"full", "forced_full_same_weights"}
        model = graph_model if trained_name == "I-v-G" else dense_model
        current_trial = dict(trial)
        current_trial["changed_sensor_description"] = (
            dict(changed_description) if condition_name == "changed_sensor14_x" else "none"
        )
        outcome = _sample_one(
            trial_index=index,
            plan=current_trial,
            task=task_by_row[row],
            known=known,
            condition=condition,
            initial_noise=noise_by_row[row],
            model=model,
            dense=use_dense,
            builder_factory=builder_factory,
            predictor=predictor,
            context=context,
            output_root=output_root,
            updates=updates,
            stop_after_utc=stop_after_utc,
            changed_sensor=(changed_description if condition_name == "changed_sensor14_x" else None),
            row50_changed_observed_velocity_mps=(changed_target if row == 50 else None),
        )
        outcomes.append(outcome)
        _append_jsonl(outcome_path, outcome)
    _atomic_json(
        output_root / "sample_panel_summary.json",
        {
            "status": "sample_panel_complete" if len(outcomes) == 8 else "sample_panel_incomplete",
            "planned_trajectories": 8,
            "attempted_trajectories": len(outcomes),
            "completed_scored": sum(row["status"] == "completed_scored" for row in outcomes),
            "failed_no_retry": sum(row["status"] != "completed_scored" for row in outcomes),
            "updates_per_arm": updates,
            "same_initial_noise_per_task": True,
            "new_solver_calls": 0,
            "outcomes": outcomes,
        },
    )
    intervention_pairs = []
    for row, condition in ((50, "original"), (50, "changed_sensor14_x"), (131, "original")):
        graph_name = f"trail_{next(i for i, item in enumerate(trial_plan) if item['row'] == str(row) and item['condition'] == condition and item['access'] == 'graph'):02d}_row{row}_{condition}_I-v-G_graph"
        full_access = "forced_full_same_weights" if row == 50 else "full"
        full_name = f"trail_{next(i for i, item in enumerate(trial_plan) if item['row'] == str(row) and item['condition'] == condition and item['access'] == full_access):02d}_row{row}_{condition}_{'I-v-G' if row == 50 else 'I-v-dense'}_{full_access}"
        graph_path = output_root / "trails" / graph_name / "links_01.npz"
        full_path = output_root / "trails" / full_name / "links_01.npz"
        if not graph_path.is_file() or not full_path.is_file():
            intervention_pairs.append(
                {
                    "row_index": row,
                    "condition": condition,
                    "status": "incomplete_or_failed_before_first_access_persisted",
                }
            )
            continue
        with (
            np.load(graph_path, allow_pickle=False) as graph_links,
            np.load(full_path, allow_pickle=False) as full_links,
        ):
            route_names = (
                "module_source",
                "sensor_source",
                "module_environment",
                "environment_module",
                "sensor_environment",
            )
            differing_routes = {
                name: {
                    "graph_nonzero": int(np.count_nonzero(graph_links[name])),
                    "full_nonzero": int(np.count_nonzero(full_links[name])),
                    "values_differ": bool(not np.array_equal(graph_links[name], full_links[name])),
                }
                for name in route_names
                if name in graph_links.files and name in full_links.files
            }
            embeddings_identical = np.array_equal(graph_links["module_embeddings"], full_links["module_embeddings"])
        intervention_pairs.append(
            {
                "row_index": row,
                "condition": condition,
                "graph_trial": graph_name,
                "full_access_trial": full_name,
                "effective_access_changed": any(item["values_differ"] for item in differing_routes.values()),
                "module_embeddings_identical": embeddings_identical,
                "routes": differing_routes,
            }
        )
    _atomic_json(
        output_root / "link_intervention_check.json",
        {
            "same_initial_noise_and_initial_candidate": True,
            "same_frozen_G_provider": True,
            "dense_access_applied_after_provider": True,
            "pairs": intervention_pairs,
        },
    )
    changed_comparisons = []
    for access in ("graph", "full", "forced_full_same_weights"):
        arm = "I-v-dense" if access == "full" else "I-v-G"
        original = next(
            (
                row
                for row in outcomes
                if row.get("status") == "completed_scored"
                and row["row_index"] == 50
                and row["plan"]["condition"] == "original"
                and row["plan"]["trained_model"] == arm
                and row["plan"]["access"] == access
            ),
            None,
        )
        changed = next(
            (
                row
                for row in outcomes
                if row.get("status") == "completed_scored"
                and row["row_index"] == 50
                and row["plan"]["condition"] == "changed_sensor14_x"
                and row["plan"]["trained_model"] == arm
                and row["plan"]["access"] == access
            ),
            None,
        )
        if original is None or changed is None:
            changed_comparisons.append(
                {
                    "trained_model": arm,
                    "access": access,
                    "status": "paired_draw_missing_or_unscored",
                }
            )
            continue
        original_cross = original["row50_changed_observation_cross_score_mps"]
        changed_cross = changed["row50_changed_observation_cross_score_mps"]
        changed_comparisons.append(
            {
                "trained_model": arm,
                "access": access,
                "status": "paired_cross_scored",
                "original_draw_scored_against_changed_observations_mps": original_cross,
                "changed_draw_scored_against_same_changed_observations_mps": changed_cross,
                "cross_score_improvement_mps": float(original_cross - changed_cross),
                "same_initial_noise": True,
                "same_model_weights": True,
                "changed_observation_is_not_an_independent_physical_realization": True,
            }
        )
    _atomic_json(
        output_root / "changed_observation_cross_scores.json",
        {
            "row_index": 50,
            "changed_observation": dict(changed_description),
            "comparison_target": "same changed observations for original and changed draws",
            "pairs": changed_comparisons,
        },
    )
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--updates", type=int, choices=(1, 200, 500, 1000), default=1)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--sample-panel", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stop-after-utc", type=str)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    stop_after_utc = _parse_stop_after_utc(args.stop_after_utc)
    _check_deadline(stop_after_utc)
    output_root = args.output_root.resolve()
    freeze, preflight, config = _load_frozen_inputs()

    # Assemble the target-free cohort, hidden-slot panel, and protocol before
    # loading CUDA models or recording any learning/sample outcome.
    (
        view,
        _train_rows,
        split_record,
        context,
        train_tasks,
        diagnostic_tasks,
        cohort_layouts,
        excluded_layouts,
    ) = _build_context_and_training_tasks(config, preflight)
    center, scale = fit_wind_sensor_velocity_transform(train_tasks)
    development_review_tasks = _development_tasks(view, preflight, context)
    protocol = _build_protocol(
        freeze,
        preflight,
        split_record,
        cohort_layouts,
        excluded_layouts,
        train_tasks,
        diagnostic_tasks,
        development_review_tasks,
        EXPECTED_DEV_ROWS,
        center,
        scale,
        stop_after_utc,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _save_protocol(output_root / "protocol.json", protocol)

    if args.seed != SEED:
        raise ValueError("The matched stable inverse seed is predeclared in protocol.json")
    if args.sample_panel and args.checkpoint is None:
        raise ValueError("--sample-panel requires --checkpoint")
    if args.sample_panel and args.diagnostic_only:
        raise ValueError("Choose either the training diagnostic or the final sample panel")
    if args.diagnostic_only and args.resume_checkpoint is not None:
        raise ValueError("The bounded diagnostic starts from the fresh baseline initialization")
    device = _check_gpu2(args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    if args.sample_panel:
        outcomes = _run_sample_panel(
            freeze=freeze,
            config=config,
            preflight=preflight,
            protocol=protocol,
            view=view,
            context=context,
            train_tasks=train_tasks,
            center=center,
            scale=scale,
            checkpoint_path=args.checkpoint.resolve(),
            device=device,
            output_root=output_root,
            seed=int(args.seed),
            stop_after_utc=stop_after_utc,
        )
        print(json.dumps({"status": "sample_panel_complete", "attempted": len(outcomes)}, sort_keys=True))
        return

    g_model, organizer = _load_forward_provider(freeze, config, train_tasks[0], device)
    g_sha = str(freeze["endpoints"]["g_packet"]["sha256"])
    provider_factory = _provider_factory(g_model, organizer, context, g_sha)
    denoiser_template = _make_denoiser_template(
        train_tasks[0], center, scale, int(organizer.node_encoder[0].out_features)
    )

    review_tasks = {
        "train": diagnostic_tasks,
        "development": development_review_tasks,
    }

    def save_primary_review(update_count: int, graph_model: Any, dense_model: Any) -> None:
        review_path = output_root / f"primary_denoising_review_u{int(update_count):04d}.json"
        if review_path.exists():
            existing = json.loads(review_path.read_text(encoding="utf-8"))
            if existing.get("review_update_count") != int(update_count):
                raise FileExistsError("Existing primary denoising review does not match its update milestone")
            return
        report = evaluate_matched_wind_velocity_denoising(
            graph_model,
            dense_model,
            review_tasks,
            provider_factory=provider_factory,
            sensor_velocity_center_mps=center,
            sensor_velocity_scale_mps=scale,
            seed=int(args.seed) ^ 0xD31F,
            audit_times=(0.25, 0.5, 0.75, 1.0),
        )
        report["review_update_count"] = int(update_count)
        report["experiment_id"] = EXPERIMENT_ID
        report["source_freeze_id"] = SOURCE_FREEZE_ID
        report["optimizer_update_exposure_per_arm"] = int(update_count * 4)
        _atomic_json(review_path, report, once=True)

    if args.diagnostic_only:
        started = time.perf_counter()
        diagnostic = train_wind_velocity_overfit_diagnostic(
            diagnostic_tasks,
            denoiser_template=denoiser_template,
            provider_factory=provider_factory,
            frozen_modules={"selected_g_u4910": g_model, "selected_g_organizer_u4910": organizer},
            sensor_velocity_center_mps=center,
            sensor_velocity_scale_mps=scale,
            updates=100,
            batch_size=4,
            steps=STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            pure_noise_fraction=PURE_NOISE_FRACTION,
            seed=int(args.seed) ^ 0xD1A6,
            device=device,
            attempt_log_path=output_root / "overfit_diagnostic_attempts.jsonl",
            stop_after_utc=stop_after_utc,
        )
        torch.cuda.synchronize(device)
        summary = {
            "status": (
                "bounded_overfit_diagnostic_complete"
                if diagnostic.updates == 100
                else "bounded_overfit_diagnostic_stopped_before_100"
            ),
            "task_count": len(diagnostic_tasks),
            "distinct_layout_count": len({int(task.template_case.layout_index) for task in diagnostic_tasks}),
            "requested_optimizer_updates_one_graph_model": 100,
            "optimizer_updates_one_graph_model": diagnostic.updates,
            "examples_seen": diagnostic.examples_seen,
            "pure_noise_endpoint_examples": diagnostic.endpoint_examples,
            "candidate_provider_calls": diagnostic.provider_calls,
            "loss_first_last": ([diagnostic.losses[0], diagnostic.losses[-1]] if diagnostic.losses else None),
            "clean_coordinate_mse_before_by_time": dict(diagnostic.clean_mse_before_by_time),
            "clean_coordinate_mse_after_by_time": dict(diagnostic.clean_mse_after_by_time),
            "clean_coordinate_mae_before_by_time": dict(diagnostic.clean_mae_before_by_time),
            "clean_coordinate_mae_after_by_time": dict(diagnostic.clean_mae_after_by_time),
            "sensor_condition_response_before_by_time": dict(diagnostic.observation_response_before_by_time),
            "sensor_condition_response_after_by_time": dict(diagnostic.observation_response_after_by_time),
            "audit_times": list(diagnostic.audit_times),
            "diagnostic_weights_used_for_primary_pair": False,
            "primary_pair_initialization": "untouched same-seed template; restart from fresh baseline after diagnostic",
            "extra_optimizer_updates_counted_against_shared_400_remedy_allowance": diagnostic.updates,
            "elapsed_seconds": time.perf_counter() - started,
            "physical_gpu_index": torch.cuda.current_device(),
            "physical_gpu_uuid": EXPECTED_GPU2_UUID,
            "solver_calls": 0,
        }
        _atomic_json(output_root / "overfit_diagnostic_summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return

    started = time.perf_counter()
    checkpoint_dir = output_root / "checkpoints"
    attempt_log = output_root / "optimizer_attempts.jsonl"
    resume_update_count = 0
    resume_optimizer_elapsed_seconds = 0.0
    if args.resume_checkpoint is not None:
        resume_payload = load_trusted_checkpoint(args.resume_checkpoint.resolve(), map_location="cpu")
        resume_update_count = int(resume_payload.get("update_count", 0))
        resume_optimizer_elapsed_seconds = float(resume_payload.get("optimizer_elapsed_seconds_total", 0.0))
    matched = train_matched_wind_velocity(
        train_tasks,
        denoiser_template=denoiser_template,
        provider_factory=provider_factory,
        frozen_modules={"selected_g_u4910": g_model, "selected_g_organizer_u4910": organizer},
        sensor_velocity_center_mps=center,
        sensor_velocity_scale_mps=scale,
        source_freeze_id=SOURCE_FREEZE_ID,
        updates=args.updates,
        batch_size=4,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        pure_noise_fraction=PURE_NOISE_FRACTION,
        seed=int(args.seed),
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=50,
        resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint else None,
        attempt_log_path=attempt_log,
        stop_after_utc=stop_after_utc,
        review_callback=save_primary_review,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    new_updates = matched.updates_per_arm - resume_update_count
    new_optimizer_seconds = matched.optimizer_elapsed_seconds - resume_optimizer_elapsed_seconds
    checkpoint = checkpoint_dir / f"updates_{matched.updates_per_arm:06d}.pt"
    summary = {
        "status": (
            "matched_velocity_training_complete"
            if matched.updates_per_arm == args.updates
            else "matched_velocity_training_stopped_before_requested_updates"
        ),
        "experiment_id": EXPERIMENT_ID,
        "requested_updates_per_arm": args.updates,
        "new_optimizer_updates_per_arm": new_updates,
        "updates_per_arm": matched.updates_per_arm,
        "optimizer_steps_per_arm": matched.updates_per_arm,
        "batch_size": matched.batch_size,
        "training_examples_per_arm": matched.training_examples_per_arm,
        "endpoint_examples_per_arm": matched.endpoint_examples_per_arm,
        "shared_provider_calls": matched.shared_provider_calls,
        "graph_loss_first_last": (
            [matched.graph_losses[0], matched.graph_losses[-1]] if matched.graph_losses else None
        ),
        "dense_loss_first_last": (
            [matched.dense_losses[0], matched.dense_losses[-1]] if matched.dense_losses else None
        ),
        "elapsed_seconds_this_invocation": elapsed,
        "optimizer_seconds_total": matched.optimizer_elapsed_seconds,
        "review_seconds_total": matched.review_elapsed_seconds,
        "wall_seconds_total_across_resumes": matched.wall_elapsed_seconds,
        "optimizer_seconds_this_invocation": new_optimizer_seconds,
        "seconds_per_new_optimizer_update": new_optimizer_seconds / max(1, new_updates),
        "wall_seconds_per_new_optimizer_update": elapsed / max(1, new_updates),
        "checkpoint_path": str(checkpoint.resolve()) if checkpoint.is_file() else None,
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "checkpoint_update_count": matched.updates_per_arm,
        "physical_gpu_index": torch.cuda.current_device(),
        "physical_gpu_uuid": EXPECTED_GPU2_UUID,
        "source_freeze_id": SOURCE_FREEZE_ID,
        "forward_and_organizer_frozen": True,
        "solver_calls": 0,
        "test_partition_opened": False,
    }
    _atomic_json(output_root / f"training_summary_{matched.updates_per_arm:06d}.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
