#!/usr/bin/env python3
"""Matched Run2112 WindFarm G/P controlled maturation from audited u100 states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "HONF_Proj"
CASE_SRC = PROJECT / "Case_WindFarm" / "src"
CORE_SRC = PROJECT / "src"
for _path in (str(CORE_SRC), str(CASE_SRC), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_active_packet_reuse as runner
from honf_forward_core.interface_fields.budgeted_frontier import canonical_pair_catalog
from honf_forward_core.interface_fields.native_direct_pair import hard_value_soft_direct_forward
from honf_forward_core.interface_fields.native_joint_shadow import hard_value_soft_organizer_forward
from honf_forward_core.interface_fields.action_aware_frontier import describe_realized_plan
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.paths import resolve_path
from windfarm.normalization import VelocityNormalizer
from windfarm.workflows.maturation import (
    WindMaturationSchedule,
    available_frontier_for_paths,
)


RUN_ID = "2112"
SEED = 2112
EXPECTED_SOURCE_SHA256 = runner.EXPECTED_SOURCE_SHA256
WARM_START_RUN = (
    PROJECT
    / "Trained_Results/WindFarm/HONF_Forward_Runs/Run_2111_20260929_022257_active_budgeted_packet_reuse"
)
MATURATION_HELPER_PATH = CASE_SRC / "windfarm/workflows/maturation.py"
WARM_CHECKPOINTS = {
    "g_packet": WARM_START_RUN / "arms/g_packet/checkpoints/updates_000100.pt",
    "direct_pair": WARM_START_RUN / "arms/direct_pair/checkpoints/updates_000100.pt",
}
WARM_SHA256 = {
    "g_packet": "811e69c51f0c32aba1d98ec52fd58fe75d97bfd3a3172783eca6b6cdbdf5d0c4",
    "direct_pair": "4883d21a13d4fa65972797c7dcedd508f735e56e7b52889dd67449e2ea709251",
}
DEVICE_UUID = "GPU-233fcd85-5c6a-6212-3f44-655252afec70"
PRIMARY_CAPACITY = {"QE": 0.95, "MM": 0.90}
FROZEN_090_PROBE = {"QE": 0.90, "MM": 0.90}
MAX_ADDITIONAL_UPDATES = 6000
LANE_BUDGET_SECONDS = 14 * 60 * 60
REVIEW_SPARSE_COUNTS = {204, 408, 816, 1632, 3264, 4896}
DURABLE_CHECKPOINT_INTERVAL = 50


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _gpu_uuid() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def _rows_sha256(rows: Sequence[int]) -> str:
    import hashlib

    encoded = json.dumps([int(value) for value in rows], separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _norm(gradients: Sequence[torch.Tensor | None]) -> float:
    values = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt().cpu())


def _group_parameter_delta(parameters: Sequence[torch.nn.Parameter], before: Sequence[torch.Tensor]) -> float:
    values = [
        (parameter.detach() - prior).float().square().sum()
        for parameter, prior in zip(parameters, before, strict=True)
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt().cpu())


def _copy_parameters(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _load_warm_start(arm: str) -> tuple[dict[str, Any], str]:
    path = WARM_CHECKPOINTS[arm]
    digest = runner._file_sha256(path)
    if digest != WARM_SHA256[arm]:
        raise ValueError(f"{arm} u100 warm-start SHA256 changed: {digest}")
    payload = load_trusted_checkpoint(path, map_location="cpu")
    if (
        not isinstance(payload, Mapping)
        or payload.get("run_id") != "2111"
        or payload.get("arm") != arm
        or int(payload.get("update_count", -1)) != 100
        or payload.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or payload.get("secondary_route") != "MM"
        or int(payload.get("seed", -1)) != 2111
        or not isinstance(payload.get("optimizer_state_dict"), Mapping)
    ):
        raise ValueError(f"{arm} is not the matched Run2111 u100/MM warm checkpoint")
    return dict(payload), digest


def _case_batch(
    *,
    view: Any,
    row_id: int,
    query_seed: int,
    role_counts: Mapping[str, int],
    sampler: Any,
    normalizer: VelocityNormalizer,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    case = view.run(int(row_id))
    sample = runner.sample_native_role_queries(
        case,
        np.random.default_rng(int(query_seed)),
        role_counts,
        catalogue_cache=sampler,
    )
    batch = runner._batch_from_sample(case, sample, normalizer, device)
    return case, sample, batch


def _measure_warmed_case_batch_panel(
    *,
    run_dir: Path,
    attempt_id: str,
    view: Any,
    train_rows: Sequence[int],
    role_counts: Mapping[str, int],
    normalizer: VelocityNormalizer,
    device: torch.device,
    row_count: int = 16,
) -> dict[str, Any]:
    """Measure cold construction and warmed repeats of native query batches."""
    output = run_dir / "case_batch_warm_panel.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    by_layout: dict[int, list[int]] = {}
    for row in train_rows:
        layout = int(view.metadata["layout_index"][int(row)])
        by_layout.setdefault(layout, []).append(int(row))
    layouts = sorted(by_layout)
    rng = np.random.default_rng(SEED + 911)
    selected_layouts = sorted(map(int, rng.choice(layouts, size=min(row_count, len(layouts)), replace=False)))
    selected_rows = []
    for layout in selected_layouts:
        candidates = sorted(by_layout[layout])
        selected_rows.append(candidates[int(rng.integers(0, len(candidates)))])

    sampler = runner.NativeRoleCatalogueCache()
    observations: list[dict[str, Any]] = []
    for index, row_id in enumerate(selected_rows):
        for repeat_index in range(3):
            hit_before = sampler.hit_count
            miss_before = sampler.miss_count
            started = time.perf_counter()
            case, sample, _batch = _case_batch(
                view=view,
                row_id=row_id,
                query_seed=SEED + 911_000 + index * 3 + repeat_index,
                role_counts=role_counts,
                sampler=sampler,
                normalizer=normalizer,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            hit = sampler.hit_count - hit_before
            miss = sampler.miss_count - miss_before
            observations.append({
                "row": row_id,
                "layout_index": int(view.metadata["layout_index"][row_id]),
                "wind_direction_deg": float(case.wind_direction_deg),
                "repeat_index": repeat_index,
                "cache_status": "hit" if hit else "miss" if miss else "none",
                "cache_hit_delta": hit,
                "cache_miss_delta": miss,
                "case_batch_elapsed_seconds": elapsed,
                "query_seed": SEED + 911_000 + index * 3 + repeat_index,
                "role_query_counts": dict(role_counts),
                "sampled_query_count": int(sample.coordinates_D.shape[0]),
            })
    cold = [row["case_batch_elapsed_seconds"] for row in observations if row["repeat_index"] == 0]
    warm = [row["case_batch_elapsed_seconds"] for row in observations if row["repeat_index"] > 0]
    record = {
        "run_id": RUN_ID,
        "attempt_id": attempt_id,
        "training_rows_only": True,
        "target_fields_used_for_metrics": False,
        "selected_row_count": len(selected_rows),
        "repeats_per_row": 3,
        "cold_cache_p50_seconds": float(np.median(cold)) if cold else None,
        "cold_cache_p90_seconds": float(np.quantile(cold, 0.9)) if cold else None,
        "warm_cache_p50_seconds": float(np.median(warm)) if warm else None,
        "warm_cache_p90_seconds": float(np.quantile(warm, 0.9)) if warm else None,
        "cache_summary": sampler.summary(),
        "observations": observations,
    }
    _write_json(output, record)
    return record


def _new_student(
    *,
    arm: str,
    warm: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    batch: Any,
    forward_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, list[torch.nn.Parameter], Any, list[torch.nn.Parameter], torch.optim.Optimizer]:
    model = runner._new_model_from_source(source_payload, normalizer, batch, device)
    model.load_state_dict(warm["model_state_dict"], strict=True)
    model.core.backend.set_cover_mode("external")
    model.core.backend.set_cover_executor("dense_masked")
    model.train()
    physical_parameters = runner._enable_physical_trainable_scope(model)
    encoded = model.core.encode_case(batch)
    tree = model.core.backend.build_case_trees(encoded)[0]
    if arm == "g_packet":
        control = runner._build_organizer(model.core, encoded, tree, forward_config, device)
        control.load_state_dict(warm["organizer_state_dict"], strict=True)
        control.train()
    else:
        feature_width = int(runner._direct_feature_tables(runner._detach_inputs(encoded))["width"])
        control = runner.BudgetConditionedDirectPairScorer(
            receiver_feature_dim=feature_width,
            source_feature_dim=feature_width,
            hidden_dim=96,
            source_chunk_size=128,
        ).to(device)
        control.load_state_dict(warm["direct_scorer_state_dict"], strict=True)
        control.train()
    route_parameters = list(control.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": physical_parameters, "lr": 1.0e-5, "weight_decay": 1.0e-5},
            {"params": route_parameters, "lr": 2.0e-4, "weight_decay": 1.0e-5},
        ]
    )
    optimizer.load_state_dict(warm["optimizer_state_dict"])
    return model, physical_parameters, control, route_parameters, optimizer


def _anchor_loss(
    *,
    student_prediction: torch.Tensor,
    incumbent_prediction: torch.Tensor,
    role_slices: Mapping[str, slice],
    normalizer: VelocityNormalizer,
    role_scales: Mapping[str, float],
) -> torch.Tensor:
    loss, _mse, _rmse = runner._role_objective(
        student_prediction,
        incumbent_prediction,
        role_slices,
        normalizer,
        role_scales,
    )
    return loss


def _calibrate_anchor(
    *,
    run_dir: Path,
    g_warm: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    view: Any,
    train_rows: Sequence[int],
    normalizer: VelocityNormalizer,
    role_counts: Mapping[str, int],
    role_scales: Mapping[str, float],
    forward_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    path = run_dir / "anchor_calibration.json"
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256:
            raise ValueError("stored fixed-incumbent anchor calibration belongs to another source")
        return record
    schedule = WindMaturationSchedule(train_rows, start_update=100, seed=SEED)
    first = schedule.plan(100)
    sampler = runner.NativeRoleCatalogueCache()
    _case, sample, batch = _case_batch(
        view=view,
        row_id=first.row_id,
        query_seed=first.query_seed,
        role_counts=role_counts,
        sampler=sampler,
        normalizer=normalizer,
        device=device,
    )
    student, physical, _control, _route_parameters, _optimizer = _new_student(
        arm="g_packet",
        warm=g_warm,
        source_payload=source_payload,
        normalizer=normalizer,
        batch=batch,
        forward_config=forward_config,
        device=device,
    )
    student.eval()
    _student_encoded, prediction = runner._full_access_prediction(student, batch)
    target_loss, _mse, _rmse = runner._role_objective(
        prediction, batch.target_field, sample.role_slices, normalizer, role_scales
    )
    with torch.no_grad():
        incumbent = runner._new_model_from_source(source_payload, normalizer, batch, device)
        incumbent.eval()
        incumbent_prediction = runner._full_access_prediction(incumbent, batch)[1]
    anchor = _anchor_loss(
        student_prediction=prediction,
        incumbent_prediction=incumbent_prediction,
        role_slices=sample.role_slices,
        normalizer=normalizer,
        role_scales=role_scales,
    )
    reference_gradients = torch.autograd.grad(target_loss, physical, retain_graph=True, allow_unused=True)
    anchor_gradients = torch.autograd.grad(anchor, physical, allow_unused=True)
    reference_norm = _norm(reference_gradients)
    anchor_norm = _norm(anchor_gradients)
    weight = 0.1 if anchor_norm <= 1.0e-12 else min(10.0, max(0.01, 0.1 * reference_norm / anchor_norm))
    record = {
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "g_u100_checkpoint_sha256": WARM_SHA256["g_packet"],
        "training_only": True,
        "row": int(first.row_id),
        "layout_index": int(view.metadata["layout_index"][first.row_id]),
        "query_seed": int(first.query_seed),
        "normalized_reference_loss": float(target_loss.detach().cpu()),
        "normalized_fixed_incumbent_anchor_loss": float(anchor.detach().cpu()),
        "reference_physical_gradient_l2": reference_norm,
        "anchor_physical_gradient_l2": anchor_norm,
        "anchor_weight": weight,
        "calibration_rule": "clamp(0.10 * reference_gradient_l2 / anchor_gradient_l2, 0.01, 10.0); 0.10 fallback if anchor gradient is zero",
        "incumbent_identity": "exact retained Run2110 W-full u1500",
    }
    _write_json(path, record)
    return record


def _canonical_route_work(
    *,
    plan: Any,
    encoded: Any,
    route: str,
) -> dict[str, Any]:
    tree = plan.tree
    catalog = canonical_pair_catalog(encoded, tree, route)
    access = plan.access_for(
        route,
        catalog.receiver_coordinates,
        int(catalog.source_validity.shape[0]),
        module_present=encoded.module_present[0] if route in {"MM", "EM", "QM"} else None,
    )
    support = (access > 0.0) & catalog.pair_validity
    weighted = support.to(torch.float64) * catalog.receiver_weights.to(torch.float64)[:, None]
    full = catalog.pair_validity.to(torch.float64) * catalog.receiver_weights.to(torch.float64)[:, None]
    live = plan.frontier_summary(route, receivers=catalog.receiver_coordinates)
    full_work = float(full.sum().detach().cpu())
    actual_work = float(weighted.sum().detach().cpu())
    return {
        "actual_canonical_work": actual_work,
        "full_canonical_work": full_work,
        "canonical_work_fraction": actual_work / full_work if full_work > 0.0 else 0.0,
        "canonical_selected_unique_pairs": int(support.sum().detach().cpu()),
        "canonical_full_unique_pairs": int(catalog.pair_validity.sum().detach().cpu()),
        "nonredundant_packets": int(live.nonredundant_packet_count),
    }


def _action_support_union_k(plan: Any, encoded: Any, cut: Sequence[int]) -> int:
    """Count distinct packet rows after unioning typed MM and QE support."""
    cut_rows = torch.as_tensor(cut, device=plan.split_gates.device, dtype=torch.long)
    module = plan.permission_matrix("MM")[cut_rows] > 0.0
    module = module & (encoded.module_present[0] > 0.5)[None, :]
    environment = plan.permission_matrix("QE")[cut_rows] > 0.0
    environment = environment & (encoded.env_weights[0] > 0.0)[None, :]
    signatures = torch.cat((module, environment), dim=1)
    bearing = signatures.any(dim=1)
    return len({tuple(row) for row in signatures[bearing].detach().cpu().tolist()})


def _gradient_arrays(
    *,
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    return [parameter.grad for parameter in parameters]


def _optimizer_route_state(optimizer: torch.optim.Optimizer, parameters: Sequence[torch.nn.Parameter]) -> list[Any]:
    values = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        values.append(None if step is None else float(step.detach().cpu()))
    return values


def _train_one_arm(
    *,
    arm: str,
    target_additional_updates: int,
    run_dir: Path,
    config: Mapping[str, Any],
    view: Any,
    train_rows: Sequence[int],
    split_record: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    source_path: Path,
    source_sha: str,
    anchor_calibration: Mapping[str, Any],
    stop_after_seconds: int | None,
    attempt_id: str,
    active_invocation_started_monotonic: float,
) -> dict[str, Any]:
    arm_name = "g_packet" if arm == "g" else "direct_pair"
    warm, warm_sha = _load_warm_start(arm_name)
    arm_dir = run_dir / "arms" / arm_name
    checkpoint_dir = arm_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ledger = arm_dir / "updates.jsonl"
    latest_record: dict[str, Any] | None = None
    sampler = runner.NativeRoleCatalogueCache()
    schedule = WindMaturationSchedule(train_rows, start_update=100, seed=SEED)
    device = torch.device("cuda:0")
    role_counts = {str(key): int(value) for key, value in config["forward"]["stage_a"]["role_query_counts"].items()}
    role_scales = {str(key): float(value) for key, value in config["forward"]["stage_a"]["role_loss_scales_mps"].items()}

    previous_update = 100
    physical_calls = int(warm.get("physical_optimizer_calls", 100))
    route_calls = int(warm.get("organizer_optimizer_calls", 0) if arm == "g" else warm.get("direct_scorer_optimizer_calls", 0))
    checkpoint_state = None
    latest_checkpoint_record = arm_dir / "latest_checkpoint.json"
    if not latest_checkpoint_record.is_file():
        latest_checkpoint_record = arm_dir / "latest_review.json"
    if latest_checkpoint_record.is_file():
        checkpoint_summary = json.loads(latest_checkpoint_record.read_text(encoding="utf-8"))
        previous_update = int(checkpoint_summary["update_count"])
        checkpoint_file = checkpoint_dir / f"updates_{previous_update:06d}.pt"
        if checkpoint_summary.get("checkpoint_sha256") != runner._file_sha256(checkpoint_file):
            raise ValueError("Run2112 resume checkpoint differs from its saved SHA256")
        checkpoint_state = load_trusted_checkpoint(checkpoint_file, map_location="cpu")
        if not isinstance(checkpoint_state, Mapping) or checkpoint_state.get("run_id") != RUN_ID:
            raise ValueError("Run2112 checkpoint identity is invalid")
        if checkpoint_state.get("train_rows_sha256") != split_record["student_train_rows_sha256"]:
            raise ValueError("Run2112 checkpoint train-row identity changed")
        if not ledger.is_file():
            raise FileNotFoundError("Run2112 resume checkpoint has no update ledger")
        with ledger.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not any(int(row["update_count"]) == previous_update for row in rows):
            raise ValueError("Run2112 update ledger does not contain the latest durable checkpoint")
        orphaned_updates = [row for row in rows if int(row["update_count"]) > previous_update]
        attempt_ledger = arm_dir / "optimizer_attempts.jsonl"
        attempt_rows = []
        if attempt_ledger.is_file():
            with attempt_ledger.open(encoding="utf-8") as stream:
                attempt_rows = [json.loads(line) for line in stream if line.strip()]
        recovery_ledger = arm_dir / "recovery_events.jsonl"
        archived_keys: set[str] = set()
        if recovery_ledger.is_file():
            with recovery_ledger.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        event = json.loads(line)
                        archived_keys.update(str(key) for key in event.get("orphaned_attempt_keys", []))
        orphaned_attempts = [
            row for row in attempt_rows
            if int(row.get("update_count", -1)) > previous_update
            and str(row.get("optimizer_attempt_key", "")) not in archived_keys
        ]
        if orphaned_updates or orphaned_attempts:
            _append_jsonl(recovery_ledger, {
                "event": "resume_archived_uncheckpointed_tail",
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "attempt_id": attempt_id,
                "resume_checkpoint_update": previous_update,
                "orphaned_update_count": len(orphaned_updates),
                "orphaned_update_attempt_ids": [row.get("optimizer_attempt_key") for row in orphaned_updates],
                "orphaned_attempt_keys": [row.get("optimizer_attempt_key") for row in orphaned_attempts],
                "optimizer_state_disposition": "discarded_with_uncheckpointed_state; retained as historical GPU-call evidence",
            })
        physical_calls = int(checkpoint_state["physical_optimizer_calls"])
        route_calls = int(checkpoint_state["route_optimizer_calls"])
        warm = dict(checkpoint_state)

    target_absolute_update = 100 + int(target_additional_updates)
    if target_additional_updates <= previous_update - 100:
        raise ValueError("target additional update count must exceed the saved Run2112 endpoint")
    if target_additional_updates > MAX_ADDITIONAL_UPDATES:
        raise ValueError("Wind maturation is capped at 6,000 additional physical updates per arm")

    first_next = schedule.plan(previous_update)
    init_hit_before = sampler.hit_count
    init_miss_before = sampler.miss_count
    init_case_batch_start = time.perf_counter()
    _case, _sample, init_batch = _case_batch(
        view=view,
        row_id=first_next.row_id,
        query_seed=first_next.query_seed,
        role_counts=role_counts,
        sampler=sampler,
        normalizer=normalizer,
        device=device,
    )
    torch.cuda.synchronize(device)
    _append_jsonl(run_dir / "case_batch_timings.jsonl", {
        "event": "model_initialization_case_batch",
        "attempt_id": attempt_id,
        "arm": arm_name,
        "row": int(first_next.row_id),
        "query_seed": int(first_next.query_seed),
        "case_batch_elapsed_seconds": time.perf_counter() - init_case_batch_start,
        "role_catalogue_cache_status": "hit" if sampler.hit_count > init_hit_before else "miss",
        "role_catalogue_cache_hit_count": sampler.hit_count,
        "role_catalogue_cache_miss_count": sampler.miss_count,
    })
    model, physical_parameters, control, route_parameters, optimizer = _new_student(
        arm=arm_name,
        warm=warm,
        source_payload=source_payload,
        normalizer=normalizer,
        batch=init_batch,
        forward_config=config["forward"],
        device=device,
    )
    if checkpoint_state is not None:
        if arm == "g":
            control.load_state_dict(checkpoint_state["organizer_state_dict"], strict=True)
        else:
            control.load_state_dict(checkpoint_state["direct_scorer_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        model.load_state_dict(checkpoint_state["model_state_dict"], strict=True)
    teacher = runner._new_model_from_source(source_payload, normalizer, init_batch, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    start_time = active_invocation_started_monotonic
    lane_budget_before = float(
        json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["gpu_active_seconds"]
    )
    elapsed_prior = 0.0 if checkpoint_state is None else float(checkpoint_state.get("elapsed_seconds_total", 0.0))
    completed_update = previous_update
    updates_this_invocation = 0
    optimizer_calls_this_invocation = 0
    route_calls_this_invocation = 0

    def save_checkpoint(record: Mapping[str, Any], *, review: bool) -> str:
        elapsed_total = elapsed_prior + time.monotonic() - start_time
        state = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "arm": arm_name,
            "update_count": int(record["update_count"]),
            "additional_update_count": int(record["additional_update_count"]),
            "model_state_sha256": runner._model_state_sha256(model),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "organizer_state_dict": control.state_dict() if arm == "g" else None,
            "direct_scorer_state_dict": control.state_dict() if arm == "p" else None,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_sha,
            "warm_start_checkpoint": str(WARM_CHECKPOINTS[arm_name]),
            "warm_start_checkpoint_sha256": warm_sha,
            "train_rows_sha256": split_record["student_train_rows_sha256"],
            "seed": SEED,
            "primary_capacity": PRIMARY_CAPACITY,
            "fixed_incumbent_anchor": dict(anchor_calibration),
            "physical_optimizer_calls": physical_calls,
            "route_optimizer_calls": route_calls,
            "optimizer_calls_this_invocation": optimizer_calls_this_invocation,
            "route_calls_this_invocation": route_calls_this_invocation,
            "elapsed_seconds_total": elapsed_total,
            "resource_budget_seconds": LANE_BUDGET_SECONDS,
        }
        path = checkpoint_dir / f"updates_{int(record['update_count']):06d}.pt"
        runner._atomic_checkpoint(path, state)
        summary = {key: value for key, value in record.items() if key != "step_elapsed_seconds"}
        cache_state = sampler.summary()
        summary["role_catalogue_cache_state"] = {
            key: value for key, value in cache_state.items() if key != "identities"
        }
        summary["checkpoint"] = str(path)
        summary["checkpoint_sha256"] = runner._file_sha256(path)
        summary["checkpoint_kind"] = "scheduled_review" if review else "durability_only"
        _write_json(arm_dir / "latest_checkpoint.json", summary)
        if review:
            _write_json(arm_dir / "latest_review.json", summary)
            _append_jsonl(run_dir / "reviews.jsonl", summary)
        return str(path)

    resource_budget_exhausted = False
    for absolute_update in range(previous_update + 1, target_absolute_update + 1):
        if stop_after_seconds is not None and time.monotonic() - start_time >= stop_after_seconds:
            break
        if lane_budget_before + time.monotonic() - start_time >= LANE_BUDGET_SECONDS:
            resource_budget_exhausted = True
            break
        schedule_item = schedule.plan(absolute_update - 1)
        update_cycle_start = time.perf_counter()
        cache_hit_before = sampler.hit_count
        cache_miss_before = sampler.miss_count
        case_batch_start = time.perf_counter()
        case, sample, batch = _case_batch(
            view=view,
            row_id=schedule_item.row_id,
            query_seed=schedule_item.query_seed,
            role_counts=role_counts,
            sampler=sampler,
            normalizer=normalizer,
            device=device,
        )
        torch.cuda.synchronize(device)
        case_batch_elapsed = time.perf_counter() - case_batch_start
        cache_hit_delta = sampler.hit_count - cache_hit_before
        cache_miss_delta = sampler.miss_count - cache_miss_before
        runner._set_update_seed(SEED, absolute_update)
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        anchor_value = 0.0
        if schedule_item.full_access_replay:
            encoded, prediction = runner._full_access_prediction(model, batch)
            with torch.no_grad():
                _teacher_encoded, teacher_prediction = runner._full_access_prediction(teacher, batch)
            objective, role_mse, role_rmse = runner._role_objective(
                prediction, batch.target_field, sample.role_slices, normalizer, role_scales
            )
            anchor_loss = _anchor_loss(
                student_prediction=prediction,
                incumbent_prediction=teacher_prediction,
                role_slices=sample.role_slices,
                normalizer=normalizer,
                role_scales=role_scales,
            )
            anchor_value = float(anchor_loss.detach().cpu())
            loss = objective + float(anchor_calibration["anchor_weight"]) * anchor_loss
            tree = model.core.backend.build_case_trees(encoded)[0]
            hard_plan = None
            result = None
            actual_access = {}
            route_projection_records = {}
            actual_paths: tuple[str, ...] = ()
        else:
            capacity = dict(PRIMARY_CAPACITY)
            model.core.backend.set_cover_mode("external")
            encoded = model.core.encode_case(batch)
            tree = model.core.backend.build_case_trees(encoded)[0]
            cut: tuple[int, ...] = ()
            actual_paths: tuple[str, ...] = ()
            if arm == "g":
                cut, actual_paths = available_frontier_for_paths(tree, schedule_item.requested_cut_paths)
                result = hard_value_soft_organizer_forward(
                    model.core,
                    encoded,
                    encoded.module_tokens,
                    control,
                    batch.query_xy,
                    batch.query_features,
                    receiver_chunk_size=int(model.config.interface_model.receiver_chunk_size),
                    budgets=capacity,
                    frontier_cuts=(cut,),
                    budget_fractions=capacity,
                )
                hard_plan = result.hard_plans[0]
                prediction = result.prediction
                actual_access = {
                    route: _canonical_route_work(plan=hard_plan, encoded=encoded, route=route)
                    for route in ("QE", "MM")
                }
                route_projection_records = {}
                smoke_path = run_dir / "diagnostics" / "action_feature_smoke.json"
                smoke_feature_path = run_dir / "diagnostics" / "action_feature_smoke.npz"
                if not smoke_path.is_file():
                    score_inputs = {
                        "module_states": encoded.module_tokens.detach(),
                        "environment_states": encoded.env_tokens.detach(),
                        "global_state": encoded.global_token.detach(),
                    }
                    action_scores = control.score_cases(encoded, score_inputs, (tree,), budgets=capacity)[0]
                    packet_rows = describe_realized_plan(action_scores, hard_plan, encoded, cut)
                    action_k = _action_support_union_k(hard_plan, encoded, cut)
                    if (
                        packet_rows.ndim != 2
                        or not bool(torch.isfinite(packet_rows).all())
                        or not (1 <= action_k <= int(packet_rows.shape[0]))
                    ):
                        raise RuntimeError("realized Wind action feature adapter returned invalid rows or K")
                    cpu_rows = packet_rows.detach().cpu().numpy()
                    digest = hashlib.sha256(cpu_rows.tobytes()).hexdigest()
                    smoke_feature_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(smoke_feature_path, packet_rows=cpu_rows)
                    _write_json(smoke_path, {
                        "run_id": RUN_ID,
                        "warm_start_checkpoint_sha256": WARM_SHA256["g_packet"],
                        "input_row": int(schedule_item.row_id),
                        "layout_index": int(view.metadata["layout_index"][schedule_item.row_id]),
                        "query_seed": int(schedule_item.query_seed),
                        "action": schedule_item.action,
                        "requested_cut_paths": list(schedule_item.requested_cut_paths),
                        "realized_cut_paths": list(actual_paths),
                        "packet_feature_shape": list(packet_rows.shape),
                        "packet_feature_sha256": digest,
                        "packet_features_finite": True,
                        "explicit_action_union_nonredundant_k": action_k,
                        "collapsed_cut": action_k < len(actual_paths),
                        "feature_archive": str(smoke_feature_path),
                        "measured_labels_used": False,
                    })
            else:
                hard_plan, soft_plan, projections, actual_access = _direct_plan_pair_by_route(
                    model=model,
                    encoded=encoded,
                    batch=batch,
                    scorer=control,
                    route_fractions=capacity,
                )
                result = hard_value_soft_direct_forward(
                    model.core,
                    encoded,
                    encoded.module_tokens,
                    (hard_plan,),
                    (soft_plan,),
                    batch.query_xy,
                    batch.query_features,
                    receiver_chunk_size=int(batch.query_xy.shape[1]),
                )
                prediction = result.prediction
                route_projection_records = {
                    route: {
                        "requested_work": float(projected.requested_work),
                        "achieved_work": float(projected.achieved_work),
                        "full_access_work": float(projected.full_access_work),
                        "selected_unique_pairs": int(projected.selected_unique_pairs),
                        "full_unique_pairs": int(projected.full_unique_pairs),
                    }
                    for route, projected in projections.items()
                }
            objective, role_mse, role_rmse = runner._role_objective(
                prediction, batch.target_field, sample.role_slices, normalizer, role_scales
            )
            loss = objective

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Run2112 {arm_name} loss at {absolute_update}")
        loss.backward()
        physical_preclip = runner._gradient_group_l2(physical_parameters)
        route_preclip = runner._gradient_group_l2(route_parameters)
        route_none = all(parameter.grad is None for parameter in route_parameters)
        route_versions = tuple(parameter._version for parameter in route_parameters)
        route_steps = _optimizer_route_state(optimizer, route_parameters)
        physical_before = _copy_parameters(physical_parameters)
        route_before = [] if schedule_item.full_access_replay else _copy_parameters(route_parameters)
        torch.nn.utils.clip_grad_norm_(physical_parameters, max_norm=1.0)
        if not schedule_item.full_access_replay:
            torch.nn.utils.clip_grad_norm_(route_parameters, max_norm=1.0)
        optimizer_attempt_key = f"{attempt_id}:{absolute_update}"
        _append_jsonl(arm_dir / "optimizer_attempts.jsonl", {
            "event": "optimizer_attempt",
            "optimizer_attempt_key": optimizer_attempt_key,
            "attempt_id": attempt_id,
            "run_id": RUN_ID,
            "arm": arm_name,
            "update_count": absolute_update,
            "row": int(schedule_item.row_id),
            "layout_index": int(view.metadata["layout_index"][schedule_item.row_id]),
            "query_seed": int(schedule_item.query_seed),
            "phase": schedule_item.phase,
            "action": schedule_item.action,
            "full_access_replay": bool(schedule_item.full_access_replay),
            "physical_parameter_group": True,
            "route_parameter_group": not schedule_item.full_access_replay,
            "optimizer_call_entered": True,
            "recorded_before_optimizer_step": True,
            "case_batch_elapsed_seconds": case_batch_elapsed,
            "role_catalogue_cache_status": "hit" if cache_hit_delta else "miss" if cache_miss_delta else "none",
            "role_catalogue_cache_hit_count": sampler.hit_count,
            "role_catalogue_cache_miss_count": sampler.miss_count,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        optimizer_calls_this_invocation += 1
        optimizer.step()
        torch.cuda.synchronize()
        if schedule_item.full_access_replay:
            if not route_none or tuple(parameter._version for parameter in route_parameters) != route_versions:
                raise RuntimeError("full-access replay unexpectedly reached or updated route parameters")
            if _optimizer_route_state(optimizer, route_parameters) != route_steps:
                raise RuntimeError("full-access replay unexpectedly advanced route AdamW state")
            physical_delta = _group_parameter_delta(physical_parameters, physical_before)
            route_delta = 0.0
        else:
            physical_delta = _group_parameter_delta(physical_parameters, physical_before)
            route_delta = _group_parameter_delta(route_parameters, route_before)
        update_wall_elapsed = time.perf_counter() - update_cycle_start
        physical_calls += 1
        if not schedule_item.full_access_replay:
            route_calls += 1
            route_calls_this_invocation += 1
        completed_update = absolute_update
        updates_this_invocation += 1
        elapsed_now = elapsed_prior + time.monotonic() - start_time
        if arm == "g" and not schedule_item.full_access_replay:
            route_details = actual_access
            realized_nonredundant_k = _action_support_union_k(hard_plan, encoded, cut)
            realized_paths = list(actual_paths)
            permission_status = {route: hard_plan.permission_status(route) for route in ("QE", "MM")}
        elif arm == "p" and not schedule_item.full_access_replay:
            route_details = actual_access
            realized_nonredundant_k = None
            realized_paths = []
            permission_status = {route: hard_plan.permission_status(route) for route in ("QE", "MM")}
        else:
            route_details = {}
            realized_nonredundant_k = None
            realized_paths = []
            permission_status = {route: "native_full_access" for route in ("QE", "MM")}
        record = {
            "run_id": RUN_ID,
            "arm": arm_name,
            "attempt_id": attempt_id,
            "optimizer_attempt_key": optimizer_attempt_key,
            "update_count": absolute_update,
            "additional_update_count": absolute_update - 100,
            "row": int(schedule_item.row_id),
            "layout_index": int(view.metadata["layout_index"][schedule_item.row_id]),
            "wind_direction_deg": float(case.wind_direction_deg),
            "query_seed": int(schedule_item.query_seed),
            "primary_pass": int(schedule_item.primary_pass),
            "pass_position": int(schedule_item.pass_position),
            "primary_sparse_update": int(schedule_item.relative_sparse_update),
            "effective_sparse_passes": (schedule_item.relative_sparse_update + 1) / len(train_rows),
            "processed_case_passes": (absolute_update - 100) / len(train_rows),
            "phase": schedule_item.phase,
            "requested_action": schedule_item.action,
            "requested_cut_paths": list(schedule_item.requested_cut_paths),
            "realized_cut_paths": realized_paths,
            "realized_nonredundant_k": realized_nonredundant_k,
            "full_access_replay": bool(schedule_item.full_access_replay),
            "capacity_vector": dict(schedule_item.capacity_vector),
            "objective_normalized_role_mse": float(objective.detach().cpu()),
            "anchor_normalized_role_mse": anchor_value,
            "anchor_weight": float(anchor_calibration["anchor_weight"]) if schedule_item.full_access_replay else 0.0,
            "combined_optimizer_loss": float(loss.detach().cpu()),
            "role_mse_mps2": role_mse,
            "role_rmse_mps": role_rmse,
            "permission_status": permission_status,
            "route_work": route_details,
            "route_projection": route_projection_records,
            "physical_gradient_preclip_l2": physical_preclip,
            "route_gradient_preclip_l2": route_preclip,
            "physical_parameter_update_l2": physical_delta,
            "route_parameter_update_l2": route_delta,
            "physical_optimizer_calls_total": physical_calls,
            "route_optimizer_calls_total": route_calls,
            "route_optimizer_step_applied": not schedule_item.full_access_replay,
            "step_elapsed_seconds": time.perf_counter() - step_start,
            "case_batch_elapsed_seconds": case_batch_elapsed,
            "update_wall_elapsed_seconds": update_wall_elapsed,
            "role_catalogue_cache_status": "hit" if cache_hit_delta else "miss" if cache_miss_delta else "none",
            "role_catalogue_cache_hit_count": sampler.hit_count,
            "role_catalogue_cache_miss_count": sampler.miss_count,
            "elapsed_seconds_total": elapsed_now,
        }
        latest_record = record
        _append_jsonl(ledger, record)
        sparse_count = schedule_item.relative_sparse_update + int(not schedule_item.full_access_replay)
        stage_start = (
            not schedule_item.full_access_replay
            and schedule_item.pass_position == 0
            and schedule_item.primary_pass in {2, 3, 4, 5, 6, 7}
        )
        scheduled_review = (
            sparse_count in REVIEW_SPARSE_COUNTS or stage_start or absolute_update == target_absolute_update
        )
        if scheduled_review or absolute_update % DURABLE_CHECKPOINT_INTERVAL == 0:
            record["gpu_active_seconds_elapsed"] = elapsed_now
            save_checkpoint(record, review=scheduled_review)
        if absolute_update % 10 == 0 or absolute_update == previous_update + 1:
            torch.cuda.synchronize()
            print(
                f"Run2112 {arm_name} update={absolute_update} action={schedule_item.action} "
                f"full={schedule_item.full_access_replay} loss={float(objective.detach().cpu()):.5g} "
                f"dt={record['step_elapsed_seconds']:.3f}s",
                flush=True,
            )

    if completed_update > previous_update and latest_record is not None:
        if not (arm_dir / "latest_checkpoint.json").is_file() or int(
            json.loads((arm_dir / "latest_checkpoint.json").read_text(encoding="utf-8"))["update_count"]
        ) != completed_update:
            save_checkpoint(latest_record, review=True)
    invocation_seconds = time.monotonic() - start_time
    return {
        "arm": arm_name,
        "run_dir": str(run_dir),
        "warm_start_checkpoint_sha256": warm_sha,
        "start_update": previous_update,
        "end_update": completed_update,
        "additional_updates_this_invocation": updates_this_invocation,
        "optimizer_calls_this_invocation": optimizer_calls_this_invocation,
        "route_optimizer_calls_this_invocation": route_calls_this_invocation,
        "elapsed_seconds_this_invocation": invocation_seconds,
        "median_step_seconds_last_100": None,
        "primary_capacity": PRIMARY_CAPACITY,
        "resource_budget_exhausted": resource_budget_exhausted,
    }


def _direct_plan_pair_by_route(
    *,
    model: Any,
    encoded: Any,
    batch: Any,
    scorer: Any,
    route_fractions: Mapping[str, float],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Build the matched P plan with independently fixed QE and MM capacities."""
    from dataclasses import replace

    encoded_for_scores = runner._detach_inputs(encoded)
    trees = model.core.backend.build_case_trees(encoded_for_scores)
    if len(trees) != 1:
        raise ValueError("Wind direct action expects one native case")
    tree = trees[0]
    tables = runner._direct_feature_tables(encoded_for_scores)
    base = runner.MechanismPlan(
        tree,
        tree.universe.coordinates.new_zeros((len(tree.nodes),)),
        encoded_for_scores.module_present[0],
        int(encoded_for_scores.env_coords.shape[1]),
    )
    hard_plan = base
    soft_plan = base
    projections: dict[str, Any] = {}
    actual: dict[str, Any] = {}
    for mechanism in ("QE", "MM"):
        catalog = canonical_pair_catalog(encoded_for_scores, tree, mechanism)
        if mechanism == "QE":
            receivers = batch.query_xy[0].detach()
            validity = torch.ones(receivers.shape[0], device=receivers.device, dtype=torch.bool)
        else:
            receivers = catalog.receiver_coordinates
            validity = catalog.receiver_validity
        hard_access, soft_access, projection = runner._direct_projected_access(
            scorer=scorer,
            encoded=encoded_for_scores,
            tree=tree,
            mechanism=mechanism,
            fraction=float(route_fractions[mechanism]),
            tables=tables,
            receiver_coordinates=receivers,
            receiver_validity=validity,
        )
        hard_plan = hard_plan.with_direct_pair_access(
            mechanism,
            hard_access.receiver_coordinates,
            hard_access.weights,
            receiver_validity=hard_access.receiver_validity,
        )
        soft_plan = soft_plan.with_direct_pair_access(
            mechanism,
            soft_access.receiver_coordinates,
            soft_access.weights,
            receiver_validity=soft_access.receiver_validity,
        )
        projections[mechanism] = projection
        if mechanism == "QE":
            full_pairs = int(validity.sum().detach().cpu()) * int(catalog.source_validity.sum().detach().cpu())
        else:
            full_pairs = int(catalog.pair_validity.sum().detach().cpu())
        actual[mechanism] = {
            "receiver_rows": int(hard_access.weights.shape[0]),
            "source_rows": int(hard_access.weights.shape[1]),
            "actual_selected_pairs": int((hard_access.weights > 0).sum().detach().cpu()),
            "actual_full_pairs": full_pairs,
            "canonical_eligible_pairs": int(catalog.pair_validity.sum().detach().cpu()),
            "canonical_work_fraction": (
                float(projection.achieved_work / projection.full_access_work)
                if projection.full_access_work > 0.0 else 0.0
            ),
            "canonical_selected_unique_pairs": int(projection.selected_unique_pairs),
            "canonical_full_unique_pairs": int(projection.full_unique_pairs),
        }
    return hard_plan, soft_plan, projections, actual


def _create_or_validate_run_dir(
    path: Path,
    *,
    split_record: Mapping[str, Any],
    source_path: Path,
    source_sha: str,
    driver_record: Mapping[str, Any],
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "run_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_id") != RUN_ID:
            raise ValueError("Run directory already belongs to another run identity")
        if not previous.get("preflight_pending", False) and (
            previous.get("source_checkpoint_sha256") != source_sha
            or previous.get("train_rows_sha256") != split_record["student_train_rows_sha256"]
            or previous.get("primary_capacity") != PRIMARY_CAPACITY
            or previous.get("schedule_seed") != SEED
        ):
            raise ValueError("Run2112 manifest does not match requested source, data, capacity, or schedule")
    manifest = {
        "run_id": RUN_ID,
        "arm_identity": "matched G/P controlled WindFarm maturation from exact Run2111 u100 states",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "warm_start_checkpoints": {key: str(value) for key, value in WARM_CHECKPOINTS.items()},
        "warm_start_sha256": dict(WARM_SHA256),
        "train_rows_sha256": split_record["student_train_rows_sha256"],
        "eligible_training_rows": 408,
        "train_layouts": 136,
        "schedule_seed": SEED,
        "row_sampling": "one seeded shuffle without replacement per 408-row sparse primary pass; both arms share order",
        "query_sampling": "fresh deterministic native query panel at every physical update, including same-row full replay",
        "sparse_full_cadence": {"sparse_updates": 4, "full_access_replay_updates": 1},
        "primary_capacity": dict(PRIMARY_CAPACITY),
        "historical_primary_probe_capacity": dict(FROZEN_090_PROBE),
        "stress_0p75_training": False,
        "scaffold": {"passes": 2, "paths": ["L", "R"], "k": 2},
        "action_blocks": {"actions": ["root", "two_packet", "four_packet"], "passes_each": 2},
        "max_additional_updates_per_arm": MAX_ADDITIONAL_UPDATES,
        "lane_gpu_budget_seconds": LANE_BUDGET_SECONDS,
        "gpu_active_seconds": float(previous.get("gpu_active_seconds", 0.0)),
        "arm_active_seconds": dict(previous.get("arm_active_seconds", {})),
        "optimizer_attempt_counts_by_arm": dict(previous.get("optimizer_attempt_counts_by_arm", {})),
        "preflight_pending": False,
        "gpu_device": {"index": 0, "uuid": DEVICE_UUID, "name": "NVIDIA RTX 6000 Ada Generation"},
        "optimizer": "warm-started AdamW states; separate max-norm-1.0 clipping for physical and route groups; full replays have route grads=None",
        "fixed_incumbent": "Run2110 W-full u1500",
        "maturation_helper": {
            "path": str(MATURATION_HELPER_PATH),
            "source_sha256": runner._file_sha256(MATURATION_HELPER_PATH),
            "revision_note": (
                "The u101-u355 updates used scaffold L/R only, whose frontier order is unchanged; "
                "path ordering is corrected before Phase B action cuts."
            ),
        },
        "maturation_helper_history": list(previous.get("maturation_helper_history", [])) + [{
            "path": str(MATURATION_HELPER_PATH),
            "source_sha256": runner._file_sha256(MATURATION_HELPER_PATH),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "revision_note": (
                "The u101-u355 updates used scaffold L/R only, whose frontier order is unchanged; "
                "path ordering is corrected before Phase B action cuts."
            ),
        }],
        "driver": dict(driver_record),
        "driver_history": list(previous.get("driver_history", [])) + [dict(driver_record)],
        "resource_accounting": "run_manifest.gpu_active_seconds sums full GPU-associated invocation wall time, including preflight/calibration and failed starts; attempts.jsonl records starts/stops and each arm optimizer_attempts.jsonl records pre-step calls",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _count_optimizer_attempts(path: Path, *, attempt_id: str | None = None) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "optimizer_attempt" and (attempt_id is None or row.get("attempt_id") == attempt_id):
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("g", "p"), required=True)
    parser.add_argument("--target-additional-updates", type=int, required=True,
                        help="cumulative additional physical updates from u100; maximum 6000")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop-after-seconds", type=int,
                        help="stop at an update boundary after this many active seconds")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    pid = os.getpid()
    arm_name = "g_packet" if args.arm == "g" else "direct_pair"
    attempt_id = f"{started_at.replace(':', '').replace('-', '')}_{pid}"
    script_path = Path(__file__).resolve()
    script_sha = runner._file_sha256(script_path)
    command_argv = [sys.executable, str(script_path), *sys.argv[1:]]
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_id") != RUN_ID:
            raise ValueError("Run directory already belongs to another run identity")
    else:
        _write_json(manifest_path, {
            "run_id": RUN_ID,
            "created_at_utc": started_at,
            "preflight_pending": True,
            "gpu_active_seconds": 0.0,
            "arm_active_seconds": {},
            "optimizer_attempt_counts_by_arm": {},
            "lane_gpu_budget_seconds": LANE_BUDGET_SECONDS,
        })
    start_event = {
        "event": "attempt_started",
        "run_id": RUN_ID,
        "arm": arm_name,
        "attempt_id": attempt_id,
        "pid": pid,
        "interpreter": sys.executable,
        "driver_script": str(script_path),
        "driver_source_sha256": script_sha,
        "maturation_helper_path": str(MATURATION_HELPER_PATH),
        "maturation_helper_source_sha256": runner._file_sha256(MATURATION_HELPER_PATH),
        "maturation_helper_resume_note": (
            "u101-u355 exercised scaffold L/R only; their behavior is unchanged by the L/R path-order correction."
        ),
        "command_argv": command_argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": started_at,
        "physical_gpu_index": 0,
        "gpu_uuid_expected": DEVICE_UUID,
        "requested_target_additional_updates": args.target_additional_updates,
        "warm_start_checkpoint_sha256": WARM_SHA256[arm_name],
    }
    _append_jsonl(run_dir / "attempts.jsonl", start_event)
    result: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        expected_interpreter = Path("/home/wanglz/miniconda3/envs/ModularDT/bin/python").resolve()
        if Path(sys.executable).resolve() != expected_interpreter:
            raise RuntimeError(f"Wind maturation requires {expected_interpreter}; got {sys.executable}")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "0":
            raise RuntimeError("WindFarm maturation requires CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
            raise RuntimeError("physical GPU0 is unavailable as logical cuda:0")
        if torch.cuda.get_device_name(0) != "NVIDIA RTX 6000 Ada Generation" or _gpu_uuid() != DEVICE_UUID:
            raise RuntimeError("CUDA_VISIBLE_DEVICES=0 did not resolve to the authorized physical GPU")
        if args.target_additional_updates < 1 or args.target_additional_updates > MAX_ADDITIONAL_UPDATES:
            raise ValueError("target additional updates must be within the 1..6000 lane cap")

        config_path = Path(runner.DEFAULT_CONFIG).resolve()
        config_sha = runner._file_sha256(config_path)
        driver_record = {
            "driver_script": str(script_path),
            "driver_source_sha256": script_sha,
            "maturation_helper_path": str(MATURATION_HELPER_PATH),
            "maturation_helper_source_sha256": runner._file_sha256(MATURATION_HELPER_PATH),
            "interpreter": sys.executable,
            "command_argv": command_argv,
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "cuda_visible_devices": "0",
            "physical_gpu_index": 0,
            "physical_gpu_uuid": DEVICE_UUID,
        }
        config = runner._load_config(config_path)
        view, _canonical, train_rows, split_record = runner._native_inputs(config)
        source_path, source_payload, normalizer, source_sha = runner._load_source(config)
        if source_sha != EXPECTED_SOURCE_SHA256:
            raise ValueError("Run2112 fixed incumbent source identity changed")
        manifest = _create_or_validate_run_dir(
            run_dir,
            split_record=split_record,
            source_path=source_path,
            source_sha=source_sha,
            driver_record=driver_record,
        )
        if _gpu_uuid() != manifest["gpu_device"]["uuid"]:
            raise RuntimeError("physical GPU UUID changed after Run2112 manifest initialization")
        if float(manifest["gpu_active_seconds"]) + (time.monotonic() - started_monotonic) >= LANE_BUDGET_SECONDS:
            raise RuntimeError("Run2112 14 GPU-hour lane allocation is exhausted")
        g_warm, _ = _load_warm_start("g_packet")
        role_counts = {str(key): int(value) for key, value in config["forward"]["stage_a"]["role_query_counts"].items()}
        role_scales = {str(key): float(value) for key, value in config["forward"]["stage_a"]["role_loss_scales_mps"].items()}
        anchor_calibration = _calibrate_anchor(
            run_dir=run_dir,
            g_warm=g_warm,
            source_payload=source_payload,
            view=view,
            train_rows=train_rows,
            normalizer=normalizer,
            role_counts=role_counts,
            role_scales=role_scales,
            forward_config=config["forward"],
            device=torch.device("cuda:0"),
        )
        timing_panel = _measure_warmed_case_batch_panel(
            run_dir=run_dir,
            attempt_id=attempt_id,
            view=view,
            train_rows=train_rows,
            role_counts=role_counts,
            normalizer=normalizer,
            device=torch.device("cuda:0"),
        )
        _append_jsonl(run_dir / "attempts.jsonl", {
            "event": "case_batch_timing_panel_completed",
            "attempt_id": attempt_id,
            "arm": arm_name,
            "cold_cache_p50_seconds": timing_panel["cold_cache_p50_seconds"],
            "cold_cache_p90_seconds": timing_panel["cold_cache_p90_seconds"],
            "warm_cache_p50_seconds": timing_panel["warm_cache_p50_seconds"],
            "warm_cache_p90_seconds": timing_panel["warm_cache_p90_seconds"],
        })
        result = _train_one_arm(
            arm=args.arm,
            target_additional_updates=args.target_additional_updates,
            run_dir=run_dir,
            config=config,
            view=view,
            train_rows=train_rows,
            split_record=split_record,
            source_payload=source_payload,
            normalizer=normalizer,
            source_path=source_path,
            source_sha=source_sha,
            anchor_calibration=anchor_calibration,
            stop_after_seconds=args.stop_after_seconds,
            attempt_id=attempt_id,
            active_invocation_started_monotonic=started_monotonic,
        )
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        elapsed_seconds = time.monotonic() - started_monotonic
        stopped_at = datetime.now(timezone.utc).isoformat()
        optimizer_attempt_count = _count_optimizer_attempts(
            run_dir / "arms" / arm_name / "optimizer_attempts.jsonl", attempt_id=attempt_id
        )
        stop_event = {
            "event": "attempt_stopped",
            "run_id": RUN_ID,
            "arm": arm_name,
            "attempt_id": attempt_id,
            "pid": pid,
            "stopped_at_utc": stopped_at,
            "active_wall_seconds": elapsed_seconds,
            "optimizer_attempts": optimizer_attempt_count,
            "status": "failed" if failure is not None else "completed",
            "failure": failure,
            "start_event_command_argv": command_argv,
        }
        _append_jsonl(run_dir / "attempts.jsonl", stop_event)
        if manifest_path.is_file():
            final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            final_manifest["gpu_active_seconds"] = float(final_manifest.get("gpu_active_seconds", 0.0)) + elapsed_seconds
            per_arm = final_manifest.setdefault("arm_active_seconds", {})
            per_arm[arm_name] = float(per_arm.get(arm_name, 0.0)) + elapsed_seconds
            counts = final_manifest.setdefault("optimizer_attempt_counts_by_arm", {})
            counts[arm_name] = _count_optimizer_attempts(run_dir / "arms" / arm_name / "optimizer_attempts.jsonl")
            final_manifest["last_attempt"] = stop_event
            _write_json(manifest_path, final_manifest)

    assert result is not None
    result.update({
        "attempt_id": attempt_id,
        "started_at_utc": started_at,
        "stopped_at_utc": stopped_at,
        "pid": pid,
        "active_wall_seconds_including_preflight": elapsed_seconds,
        "optimizer_attempts_in_ledger": optimizer_attempt_count,
    })
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result["lane_gpu_seconds_total"] = final_manifest["gpu_active_seconds"]
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
