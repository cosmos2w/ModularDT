#!/usr/bin/env python3
"""Run a matched, resumable Thermal value-weight pair from G-u1300."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from honf_runtime.compat import load_trusted_checkpoint

ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "diagnostics/generated/thermal_maturation_20260929/controlled_run"
SOURCE_CHECKPOINT = RUN_ROOT / "checkpoints/G_u1300_training_checkpoint.pt"
SOURCE_MANIFEST = RUN_ROOT / "run_manifest_u1300_closed_snapshot.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/generated/directed_stable_20261002/thermal/value_recovery_pair"
DEFAULT_CONFIG = CASE_ROOT / "configs/value_recovery_pair_20261002.json"
GPU2_UUID = "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"

sys.path.insert(0, str(CASE_ROOT / "scripts"))
import run_controlled_maturation as driver
from channelthermal.evaluation.loading import load_model
from channelthermal.response_control.contracts import (
    DesignInput,
    context_inputs,
    role_queries_from_stencil,
)
from channelthermal.response_control.historical import HistoricalValueSource
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.runner import derive_training_scales
from channelthermal.response_control.sampling import ReceiverSamplingConfig, sample_training_panel
from channelthermal.response_control.training import run_staged_fit


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class DeviceCheckingOperator:
    """Assert every complete-wrapper input/output tensor stays on cuda:2."""

    def __init__(self, operator: Any, device: torch.device, counter: dict[str, int], label: str) -> None:
        self.operator = operator
        self.device = device
        self.counter = counter
        self.label = label

    def __call__(self, design: Any, context: Any, queries: Any) -> Any:
        self.counter[f"{self.label}_attempted"] = self.counter.get(f"{self.label}_attempted", 0) + 1
        tensors: list[torch.Tensor] = []
        if hasattr(design, "module_centers") and isinstance(design.module_centers, torch.Tensor):
            tensors.append(design.module_centers)
        context_values = context if isinstance(context, Mapping) else getattr(context, "values", {})
        if isinstance(context_values, Mapping):
            tensors.extend(value for value in context_values.values() if isinstance(value, torch.Tensor))
        if isinstance(queries, Mapping):
            for query in queries.values():
                for field in ("coordinates", "query_features", "quadrature_weights", "valid_mask"):
                    value = getattr(query, field, None)
                    if isinstance(value, torch.Tensor):
                        tensors.append(value)
        if not tensors or any(tensor.device != self.device for tensor in tensors):
            observed = sorted({str(tensor.device) for tensor in tensors})
            raise RuntimeError(f"{self.label} native wrapper inputs are not all on {self.device}: {observed}.")
        output = self.operator(design, context, queries)
        self.counter[f"{self.label}_succeeded"] = self.counter.get(f"{self.label}_succeeded", 0) + 1
        return output


def _remap_cuda_rng_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Restore the historical masked-GPU2 stream on physical cuda:2."""

    saved_cuda = payload.get("cuda_rng_state_by_model_device") or {}
    keys = set(saved_cuda)
    if keys == {"cuda:0"}:
        return {"cuda:2": saved_cuda["cuda:0"]}
    if keys == {"cuda:2"}:
        return {"cuda:2": saved_cuda["cuda:2"]}
    raise ValueError(f"Expected the single physical-GPU2 RNG stream under cuda:0 or cuda:2, got {sorted(keys)}.")


def _stream_pair_audit(rows4: list[dict[str, Any]], rows8: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the paired family/history/query/action stream and loss weights."""

    fields = (
        "completed_update",
        "training_family_id",
        "historical_case_id",
        "training_stencil_index",
    )
    metadata_fields = (
        "query_seed",
        "source_plan_full_access_replay",
        "historical_stencil_action_at_source_cursor",
        "effective_pilot_access",
        "organizer_frozen",
        "shadow_optimization",
    )
    mismatches: list[dict[str, Any]] = []
    if len(rows4) != len(rows8):
        mismatches.append({"row": "count", "value4": len(rows4), "value8": len(rows8)})
    for index, (left, right) in enumerate(zip(rows4, rows8, strict=False)):
        for field in fields:
            if left.get(field) != right.get(field):
                mismatches.append({"row": index, "field": field, "value4": left.get(field), "value8": right.get(field)})
        left_meta = left.get("training_metadata", {})
        right_meta = right.get("training_metadata", {})
        for field in metadata_fields:
            if left_meta.get(field) != right_meta.get(field):
                mismatches.append(
                    {
                        "row": index,
                        "field": f"training_metadata.{field}",
                        "value4": left_meta.get(field),
                        "value8": right_meta.get(field),
                    }
                )
        left_weights = left.get("active_term_weights", {})
        right_weights = right.get("active_term_weights", {})
        all_terms = set(left_weights) | set(right_weights)
        for term in sorted(all_terms - {"value"}):
            if left_weights.get(term) != right_weights.get(term):
                mismatches.append(
                    {
                        "row": index,
                        "field": f"active_term_weights.{term}",
                        "value4": left_weights.get(term),
                        "value8": right_weights.get(term),
                    }
                )
        if left_weights.get("value") != 4.0 or right_weights.get("value") != 8.0:
            mismatches.append(
                {
                    "row": index,
                    "field": "active_term_weights.value",
                    "value4": left_weights.get("value"),
                    "value8": right_weights.get("value"),
                }
            )
    return {
        "update_count_value4": len(rows4),
        "update_count_value8": len(rows8),
        "fields_checked": [
            *fields,
            *(f"training_metadata.{field}" for field in metadata_fields),
            "active_term_weights[except value]",
        ],
        "paired_stream_equal": not mismatches,
        "mismatches": mismatches,
    }


def input_bundle(device: torch.device, query_batch_size: int) -> dict[str, Any]:
    """Load existing train-only atlases and dataset without starting a solver."""

    atlas_sampling = ReceiverSamplingConfig(
        max_fluid_queries=128,
        solid_queries_per_module=8,
        hot_solid_points_per_module=4,
        random_seed=2317,
    )
    raw_stencils, initial_sampled_stencils, sampling_summaries = driver.forward._selected_train_stencils(
        driver.DEFAULT_ATLAS, atlas_sampling
    )
    ref_model, ref_payload = load_model(driver.DEFAULT_REFERENCE, device)
    if int(ref_payload.get("epoch", ref_payload.get("current_epoch", -1))) != 4738:
        raise ValueError("Pilot teacher must be retained Run1804 e4738.")
    if any(parameter.device != device for parameter in ref_model.parameters()):
        raise RuntimeError("Retained reference model did not load wholly on cuda:2.")
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    dataset_path = driver._resolve_dataset_path(ref_payload, None)
    dataset = driver.GlobalChannelThermalDataset(
        dataset_path,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = driver._make_input_template(dataset)
    setup_counter: dict[str, int] = {}
    capture_native = DifferentiableThermalOperator(
        ref_model,
        template,
        dataset_config=ref_payload["train_config"]["dataset"],
        normalization_stats=ref_payload["global_normalization_stats"],
        query_batch_size=query_batch_size,
        capture_packet_inputs=True,
    )
    capture = DeviceCheckingOperator(capture_native, device, setup_counter, "setup_capture")
    first = initial_sampled_stencils[0]
    with torch.no_grad():
        capture(
            DesignInput.from_state(first.baseline.design, device=device),
            context_inputs(first.baseline.context),
            role_queries_from_stencil(first, device=device),
        )
    if (
        not isinstance(capture_native.last_packet_inputs, Mapping)
        or capture_native.last_packet_inputs.get("encoded") is None
    ):
        raise RuntimeError("Reference wrapper did not expose the shared input-only route encoding.")
    historical_sampling = ReceiverSamplingConfig(
        max_fluid_queries=3072,
        solid_queries_per_module=128,
        hot_solid_points_per_module=16,
        random_seed=2317,
    )
    historical_source = HistoricalValueSource.from_dataset(dataset, sampling=historical_sampling)
    if len(historical_source.case_ids) != 600 or len(set(historical_source.case_ids)) != 600:
        raise ValueError("Pilot requires the exact 600-case historical train-value source.")
    scales = derive_training_scales(
        initial_sampled_stencils,
        smooth_peak_beta=1.0,
        historical_value_source=historical_source,
    )
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if driver._scale_record(scales) != manifest["loss_scales"]:
        raise ValueError("Train-only pilot loss scales differ from the exact u1300 lineage recipe.")
    return {
        "raw_stencils": raw_stencils,
        "initial_sampled_stencils": initial_sampled_stencils,
        "sampling_summaries": sampling_summaries,
        "family_ids": tuple(stencil.physical_family_id for stencil in raw_stencils),
        "ref_model": ref_model,
        "ref_payload": ref_payload,
        "teacher_operator": DifferentiableThermalOperator(
            ref_model,
            template,
            dataset_config=ref_payload["train_config"]["dataset"],
            normalization_stats=ref_payload["global_normalization_stats"],
            query_batch_size=query_batch_size,
        ),
        "template": template,
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "source_encoded": capture_native.last_packet_inputs["encoded"],
        "historical_source": historical_source,
        "historical_order": historical_source.case_ids,
        "scales": scales,
        "extra_route": str(manifest["extra_route"]),
        "manifest": manifest,
        "setup_capture_counter": setup_counter,
    }


def build_arm(
    arm_label: str,
    value_weight: float,
    *,
    device: torch.device,
    source_payload: dict[str, Any],
    inputs: dict[str, Any],
    counter: dict[str, int],
    query_batch_size: int,
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    payload = dict(source_payload)
    weights = {str(key): float(value) for key, value in payload["calibrated_loss_weights"].items()}
    weights["value"] = float(value_weight)
    payload["calibrated_loss_weights"] = weights
    # The historical run saved its sole CUDA RNG stream as logical cuda:0 while
    # visibility was masked to physical GPU2. Remap explicitly; never restore
    # that state into the current physical GPU0.
    payload["cuda_rng_state_by_model_device"] = _remap_cuda_rng_state(payload)
    forward_model, route_model, bundle = driver._setup_bundle(
        "G",
        payload,
        reference_checkpoint=driver.DEFAULT_REFERENCE,
        encoded=inputs["source_encoded"],
        extra_route=inputs["extra_route"],
        device=device,
    )
    physical_parameters = tuple(parameter for parameter in forward_model.parameters() if parameter.requires_grad)
    route_parameters = tuple(route_model.parameters())
    bundle_parameters = tuple(parameter for parameter in bundle.parameters() if parameter.requires_grad)
    if len(physical_parameters) != 80 or len(route_parameters) != 40 or len(bundle_parameters) != 120:
        raise RuntimeError(
            f"Pilot parameter scope differs from saved G-u1300: physical={len(physical_parameters)}, "
            f"route={len(route_parameters)}, bundle={len(bundle_parameters)}."
        )
    if any(parameter.device != device for parameter in bundle_parameters):
        raise RuntimeError("A pilot model parameter is not on explicit cuda:2.")
    if any(parameter.device != device for parameter in (*tuple(forward_model.parameters()), *route_parameters)):
        raise RuntimeError("A loaded model parameter is not on explicit cuda:2.")
    # Build the exact checkpoint-compatible one-group AdamW inventory before
    # freezing the organizer, preserving its state while its gradients remain None.
    optimizer = torch.optim.AdamW(
        bundle_parameters,
        lr=1.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-5,
    )
    for parameter in route_parameters:
        parameter.requires_grad_(False)
    if sum(parameter.requires_grad for parameter in route_model.parameters()) != 0:
        raise RuntimeError("Organizer/route parameters were not frozen.")
    native = DeviceCheckingOperator(
        DifferentiableThermalOperator(
            forward_model,
            inputs["template"],
            dataset_config=inputs["ref_payload"]["train_config"]["dataset"],
            normalization_stats=inputs["ref_payload"]["global_normalization_stats"],
            query_batch_size=query_batch_size,
        ),
        device,
        counter,
        "student",
    )
    inputs["teacher_operator"] = DeviceCheckingOperator(
        inputs.get("teacher_native_operator", inputs["teacher_operator"]), device, counter, "teacher"
    )
    return forward_model, route_model, bundle, optimizer, native, payload


def execute_arm(
    *,
    arm_name: str,
    value_weight: float,
    stop_at_update: int,
    device: torch.device,
    source_payload: dict[str, Any],
    initial_update: int,
    inputs: dict[str, Any],
    query_batch_size: int,
    run_started: float,
    arm_started: float,
    output_dir: Path,
) -> dict[str, Any]:
    arm_dir = output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    forward_model, route_model, bundle, optimizer, native, resume_payload = build_arm(
        arm_name,
        value_weight,
        device=device,
        source_payload=source_payload,
        inputs=inputs,
        counter=counters,
        query_batch_size=query_batch_size,
    )
    controller = driver._MaturationController(inputs["family_ids"], inputs["extra_route"])
    config = driver._config(max_wall_seconds=7200.0)
    current_weights = {str(k): float(v) for k, v in resume_payload["calibrated_loss_weights"].items()}
    if set(current_weights) != {"value", "finite", "finite_peak", "pressure_value", "pressure_response"}:
        raise ValueError("Pilot loss terms differ from the recorded current recipe.")
    mixed = tuple(spec for stencil in inputs["raw_stencils"] for spec in driver._mixed_specs(stencil))
    weights_config = replace(
        ReceiverSamplingConfig(
            128,
            8,
            4,
            random_seed=7319,
            random_tail_fluid_queries=128,
            random_tail_solid_queries_per_module=16,
        ),
        random_seed=7319,
    )

    def select_index(completed: int) -> int:
        return controller.plan(completed).family_index

    def transform(stencil: Any, completed: int) -> tuple[Any, dict[str, Any]]:
        plan = controller.plan(completed)
        if stencil.physical_family_id != plan.family_id:
            raise RuntimeError("Pilot maturation family order differs from the saved G stream.")
        sampled = sample_training_panel(
            (stencil,),
            config=replace(weights_config, random_seed=plan.query_seed),
        )[0]
        return sampled.stencil, {
            "family_id": plan.family_id,
            "family_index": plan.family_index,
            "historical_stencil_action_at_source_cursor": plan.action,
            "source_plan_full_access_replay": plan.full_access_replay,
            "effective_pilot_access": "full_access",
            "query_seed": plan.query_seed,
            "organizer_frozen": True,
            "shadow_optimization": "disabled",
        }

    anchor_lambda = float(inputs["manifest"]["anchor_calibration"]["selected_shared_lambda"])

    def anchor_auxiliary(completed: int, stencil: Any, predictions: Any, _terms: Any):
        # Preserve the historical anchor cadence: it was attached to scheduled
        # full-replay positions, while every pilot forward is physically full access.
        if not controller.plan(completed).full_access_replay:
            return None, {}
        with torch.no_grad():
            teacher = driver.forward.predict_stencil(inputs["teacher_operator"], stencil, device=device)
        anchor = driver._anchor_loss(predictions, teacher, stencil, inputs["scales"])
        weighted = anchor_lambda * anchor
        return weighted, {
            "retained_anchor_unweighted_loss": float(anchor.detach().cpu()),
            "retained_anchor_weighted_loss": float(weighted.detach().cpu()),
            "retained_anchor_lambda": anchor_lambda,
        }

    gradient_snapshot: dict[str, Any] = {}
    attempted_before = int(resume_payload["attempted_optimizer_steps"])

    def on_attempt(completed: int, attempted_total: int) -> None:
        if completed == initial_update:
            physical_grads = [
                p.grad.detach().double().square().sum()
                for p in forward_model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            route_grads = [
                p.grad.detach().double().square().sum() for p in route_model.parameters() if p.grad is not None
            ]
            gradient_snapshot.update(
                {
                    "physical_gradient_l2_before_step": float(torch.sqrt(torch.stack(physical_grads).sum()).cpu())
                    if physical_grads
                    else 0.0,
                    "route_gradient_l2_before_step": float(torch.sqrt(torch.stack(route_grads).sum()).cpu())
                    if route_grads
                    else 0.0,
                    "physical_gradient_tensor_count": len(physical_grads),
                    "route_gradient_tensor_count": len(route_grads),
                    "physical_trainable_parameters": 80,
                    "route_trainable_parameters": 0,
                    "optimizer_parameter_inventory": 120,
                }
            )
        append_jsonl(
            arm_dir / "optimizer_attempts.jsonl",
            {
                "arm": arm_name,
                "completed_updates_before_attempt": completed,
                "attempted_optimizer_steps_including_old_branch": attempted_total,
                "optimizer_calls": 1,
                "timestamp_utc": utc_now(),
            },
        )

    def on_step(step: Any) -> None:
        row = {
            "arm": arm_name,
            "value_weight": value_weight,
            "completed_update": step.completed_update,
            "new_updates_this_arm": step.completed_update - 1300,
            "attempted_optimizer_step_in_arm": step.completed_update - 1300,
            "attempted_optimizer_step_in_process": step.attempted_optimizer_step,
            "attempted_optimizer_steps_including_old_branch": attempted_before + step.attempted_optimizer_step,
            "training_family_id": step.training_family_id,
            "training_stencil_index": step.training_stencil_index,
            "historical_case_id": step.historical_case_id,
            "active_terms": list(step.active_terms),
            "active_term_weights": dict(step.active_term_weights),
            "loss_terms": dict(step.term_losses),
            "total_loss": step.total_loss,
            "update_wall_seconds": step.update_wall_seconds,
            "update_gpu_milliseconds": step.update_gpu_milliseconds,
            "training_metadata": dict(step.training_metadata),
            "timestamp_utc": utc_now(),
        }
        append_jsonl(arm_dir / "training_steps.jsonl", row)
        if step.completed_update == 1301 and initial_update == 1300:
            append_jsonl(
                output_dir / "first_unit.jsonl",
                {
                    "arm": arm_name,
                    "value_weight": value_weight,
                    "source_checkpoint_update": initial_update,
                    "completed_update": step.completed_update,
                    "family_id": step.training_family_id,
                    "history_case_id": step.historical_case_id,
                    "training_stencil_index": step.training_stencil_index,
                    "query_seed": step.training_metadata.get("query_seed"),
                    "update_wall_seconds": step.update_wall_seconds,
                    "update_gpu_milliseconds": step.update_gpu_milliseconds,
                    "gradient_snapshot": gradient_snapshot,
                    "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "native_wrapper_call_counters": dict(counters),
                    "setup_capture_wrapper_call_counters": dict(inputs["setup_capture_counter"]),
                    "elapsed_since_process_start_seconds": time.monotonic() - run_started,
                    "elapsed_arm_wall_seconds": time.monotonic() - arm_started,
                    "timestamp_utc": utc_now(),
                },
            )
        append_jsonl(
            output_dir / "progress.jsonl",
            {
                "arm": arm_name,
                "completed_update": step.completed_update,
                "value_weight": value_weight,
                "family_id": step.training_family_id,
                "total_loss": step.total_loss,
                "update_wall_seconds": step.update_wall_seconds,
                "timestamp_utc": utc_now(),
            },
        )

    def on_checkpoint(payload: Any, label: str) -> None:
        update = int(payload["actual_optimizer_updates"])
        if label not in {"resume_preflight", "zero_update_preflight"} and (
            update % 50 != 0 and update != stop_at_update
        ):
            return
        path = arm_dir / "checkpoints" / f"u{update:04d}_{label}.pt"
        atomic_torch_save(path, payload)
        latest = arm_dir / "checkpoints" / "latest.json"
        write_json(latest, {"update": update, "path": str(path), "label": label})
        append_jsonl(
            arm_dir / "checkpoint_receipts.jsonl",
            {
                "update": update,
                "label": label,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "timestamp_utc": utc_now(),
            },
        )

    def on_review(step: Any) -> str:
        append_jsonl(
            arm_dir / "review_snapshots.jsonl",
            {
                "arm": arm_name,
                "completed_update": step.completed_update,
                "review_scope": "training_only; no development outcomes inspected",
                "total_loss": step.total_loss,
                "loss_terms": dict(step.term_losses),
                "family_id": step.training_family_id,
                "historical_case_id": step.historical_case_id,
                "timestamp_utc": utc_now(),
                "decision": "stop at this requested stage boundary"
                if step.completed_update == stop_at_update
                else "continue; training-only review",
            },
        )
        return "stop" if step.completed_update == stop_at_update else "continue"

    if device.index != 2 or torch.cuda.current_device() != 2:
        raise RuntimeError("Pilot model or CUDA current device is not physical cuda:2.")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    fit = run_staged_fit(
        native,
        bundle,
        optimizer,
        inputs["raw_stencils"],
        scales=inputs["scales"],
        loss_weights=current_weights,
        mixed_specs=mixed,
        historical_value_source=inputs["historical_source"],
        config=config,
        initial_update=initial_update,
        initial_attempted_optimizer_steps=attempted_before,
        resume_payload=resume_payload,
        stop_at_update=stop_at_update,
        device=device,
        on_checkpoint=on_checkpoint,
        on_optimizer_attempt=on_attempt,
        on_step=on_step,
        on_review=on_review,
        training_stencil_index_for_update=select_index,
        training_stencil_transform=transform,
        auxiliary_loss_fn=anchor_auxiliary,
    )
    summary = {
        "arm": arm_name,
        "value_weight": value_weight,
        "source_checkpoint": str(source_payload.get("_local_checkpoint_path", SOURCE_CHECKPOINT)),
        "source_checkpoint_sha256_from_closed_manifest": inputs["manifest"]["arms"]["G"]["latest_checkpoint_sha256"]
        if initial_update == 1300
        else None,
        "initial_update": fit.initial_update,
        "final_update": fit.final_update,
        "completed_optimizer_updates": fit.actual_optimizer_updates,
        "new_optimizer_attempts": fit.attempted_optimizer_steps,
        "attempted_optimizer_steps_including_old_branch": fit.total_attempted_optimizer_steps,
        "elapsed_arm_wall_seconds_including_loaded_state": time.monotonic() - arm_started,
        "fit_wall_seconds": fit.wall_seconds,
        "total_updates_from_u1300": fit.final_update - 1300,
        "status": "completed_requested_stage" if fit.final_update == stop_at_update else "bounded_fit_result",
        "weights": dict(fit.calibrated_weights),
        "gradient_snapshot_first_unit": gradient_snapshot,
        "native_wrapper_call_counters": counters,
        "setup_capture_wrapper_call_counters": dict(inputs["setup_capture_counter"]),
        "legacy_raw_runner_counters": {
            "inference_only_complete_wrapper_calls": (
                counters.get("teacher_attempted", 0) + inputs["setup_capture_counter"].get("setup_capture_attempted", 0)
            ),
        },
        "wrapper_call_purpose_accounting": {
            "student_training_attempted_succeeded": [
                counters.get("student_attempted", 0),
                counters.get("student_succeeded", 0),
            ],
            "anchor_teacher_loss_training_attempted_succeeded": [
                counters.get("teacher_attempted", 0),
                counters.get("teacher_succeeded", 0),
            ],
            "setup_capture_attempted_succeeded": [
                inputs["setup_capture_counter"].get("setup_capture_attempted", 0),
                inputs["setup_capture_counter"].get("setup_capture_succeeded", 0),
            ],
            "standalone_evaluation_attempted_succeeded": [0, 0],
            "legacy_counter_semantics": "The raw runner counter includes no-grad anchor-teacher calls used by the training objective plus setup captures; it is not a standalone inference count.",
        },
        "solver_calls": 0,
        "optimizer_calls": fit.attempted_optimizer_steps,
        "checkpoint_directory": str(arm_dir / "checkpoints"),
    }
    write_json(arm_dir / "summary.json", summary)
    return summary


def _latest_checkpoint(output_dir: Path, arm_name: str) -> Path:
    manifest = output_dir / arm_name / "checkpoints" / "latest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"No resumable checkpoint receipt at {manifest}.")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    path = Path(payload["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("both", "value4", "value8"), default="both")
    parser.add_argument("--updates-per-arm", type=int, default=240)
    parser.add_argument("--resume", action="store_true", help="resume both arms from their latest local checkpoints")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    process_start = time.monotonic()
    process_started_at = utc_now()
    write_json(
        output_dir / f"process_start_{process_started_at.replace(':', '').replace('-', '')}.json",
        {
            "started_at_utc": process_started_at,
            "monotonic_start": process_start,
            "arm": args.arm,
            "updates_per_arm": args.updates_per_arm,
            "physical_gpu_uuid_required": GPU2_UUID,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_request": "cuda:2 ordinary physical visibility",
            "legacy_raw_runner_counters": {"inference_only_complete_wrapper_calls": 0},
            "wrapper_call_purpose_accounting": {
                "student_training_attempted_succeeded": [0, 0],
                "anchor_teacher_loss_training_attempted_succeeded": [0, 0],
                "setup_capture_attempted_succeeded": [0, 0],
                "standalone_evaluation_attempted_succeeded": [0, 0],
                "legacy_counter_semantics": "Runner-start snapshot only; purpose totals are reconciled in paired_training_audit.json.",
            },
            "solver_calls": 0,
            "optimizer_calls": 0,
        },
    )
    try:
        recipe = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
        if args.updates_per_arm <= 0 or args.updates_per_arm > int(recipe["training"]["max_updates_per_arm"]):
            raise ValueError("updates-per-arm must be positive and within the paired recipe ceiling.")
        arm_names = ("value4", "value8") if args.arm == "both" else (args.arm,)
        gpu_uuid, gpu_name = driver._physical_gpu2_identity()
        if gpu_uuid != GPU2_UUID:
            raise RuntimeError(f"Physical GPU2 UUID mismatch: {gpu_uuid}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            raise RuntimeError("Ordinary GPU visibility required; CUDA_VISIBLE_DEVICES must be unset.")
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 2:
            raise RuntimeError("Explicit cuda:2 is not available under ordinary visibility.")
        device = torch.device("cuda:2")
        torch.cuda.set_device(device)
        if torch.cuda.current_device() != 2 or torch.cuda.get_device_name(2) != gpu_name:
            raise RuntimeError("Current PyTorch device does not match the assigned physical GPU2.")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        inputs = input_bundle(device, args.query_batch_size)
        inputs["teacher_native_operator"] = inputs["teacher_operator"]
        payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
        for arm_name in arm_names:
            source_path = (
                _latest_checkpoint(output_dir, arm_name)
                if args.resume
                else args.source_checkpoint.expanduser().resolve()
            )
            source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
            initial_update = int(source_payload.get("actual_optimizer_updates", -1))
            source_payload["_local_checkpoint_path"] = str(source_path)
            payloads[arm_name] = (source_path, source_payload)
        updates = {int(payload.get("actual_optimizer_updates", -1)) for _, payload in payloads.values()}
        if len(updates) != 1:
            raise ValueError("Paired arms must resume from the same completed update.")
        initial_update = updates.pop()
        for arm_name, (source_path, source_payload) in payloads.items():
            # Charge checkpoint read/loader work and verify the already reviewed SHA
            # only through its recorded manifest binding; do not create any new hash.
            if not args.resume and initial_update != 1300:
                raise ValueError("A new value-weight pair must start at exact G-u1300.")
            if not args.resume and source_path != SOURCE_CHECKPOINT.resolve():
                raise ValueError("A new pair must load the exact retained G-u1300 checkpoint path.")
            if not args.resume and int(source_payload.get("attempted_optimizer_steps", -1)) != 1310:
                raise ValueError("The retained G-u1300 checkpoint must carry its exact 1,310 attempted-step count.")
            if args.resume:
                expected_weight = 4.0 if arm_name == "value4" else 8.0
                actual_weight = float(source_payload.get("calibrated_loss_weights", {}).get("value", -1.0))
                if actual_weight != expected_weight:
                    raise ValueError(
                        f"{arm_name} resume checkpoint has value weight {actual_weight}, expected {expected_weight}."
                    )
            # The original G-u1300 run used masked physical GPU2 as logical cuda:0.
            # Under ordinary visibility that exact RNG stream must be restored on cuda:2.
            expected = inputs["manifest"]["arms"]["G"]["latest_checkpoint_sha256"]
            if (
                not args.resume
                and source_path == SOURCE_CHECKPOINT.resolve()
                and expected != "68690cb46e4ec8797e4133296314f229f4bfd24173b154ab844e889fc9d6b748"
            ):
                raise ValueError("Closed manifest does not bind the approved exact G-u1300 checkpoint.")
        stop_at_update = initial_update + args.updates_per_arm
        if stop_at_update > int(recipe["training"]["global_update_ceiling"]):
            raise ValueError("Requested stage exceeds the configured global update ceiling.")
        results = []
        for arm_name, (source_path, source_payload) in payloads.items():
            value_weight = 4.0 if arm_name == "value4" else 8.0
            summary = execute_arm(
                arm_name=arm_name,
                value_weight=value_weight,
                stop_at_update=stop_at_update,
                device=device,
                source_payload=source_payload,
                initial_update=initial_update,
                inputs=inputs,
                query_batch_size=args.query_batch_size,
                run_started=process_start,
                arm_started=time.monotonic(),
                output_dir=output_dir,
            )
            results.append(summary)
            del summary
            torch.cuda.empty_cache()
        result = {
            "status": "completed_requested_stage"
            if all(row["final_update"] == stop_at_update for row in results)
            else "bounded_fit_result",
            "gpu_uuid": gpu_uuid,
            "gpu_name": gpu_name,
            "physical_torch_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "setup_and_load_elapsed_wall_seconds": time.monotonic()
            - process_start
            - sum(row["elapsed_arm_wall_seconds_including_loaded_state"] for row in results),
            "total_process_elapsed_wall_seconds": time.monotonic() - process_start,
            "summaries": results,
            "setup_capture_wrapper_call_counters": dict(inputs["setup_capture_counter"]),
            "start_time_utc": process_started_at,
            "stop_time_utc": utc_now(),
        }
        if set(arm_names) == {"value4", "value8"}:
            paired_rows: dict[str, list[dict[str, Any]]] = {}
            for arm_name in arm_names:
                steps_path = output_dir / arm_name / "training_steps.jsonl"
                paired_rows[arm_name] = [
                    json.loads(line)
                    for line in steps_path.read_text(encoding="utf-8").splitlines()
                    if int(json.loads(line)["completed_update"]) > 1300
                ]
            audit = _stream_pair_audit(paired_rows["value4"], paired_rows["value8"])
            audit.update(
                {
                    "source_update": 1300,
                    "stage_start_update": initial_update,
                    "stage_stop_update": stop_at_update,
                    "value4_source_checkpoint": str(payloads["value4"][0]),
                    "value8_source_checkpoint": str(payloads["value8"][0]),
                    "value4_first_u1301": next(
                        (
                            row
                            for row in [
                                json.loads(line)
                                for line in (output_dir / "first_unit.jsonl").read_text(encoding="utf-8").splitlines()
                            ]
                            if row.get("arm") == "value4"
                        ),
                        None,
                    ),
                    "value8_first_u1301": next(
                        (
                            row
                            for row in [
                                json.loads(line)
                                for line in (output_dir / "first_unit.jsonl").read_text(encoding="utf-8").splitlines()
                            ]
                            if row.get("arm") == "value8"
                        ),
                        None,
                    ),
                }
            )
            write_json(output_dir / "paired_stream_audit.json", audit)
            if not audit["paired_stream_equal"]:
                raise RuntimeError("Value4/value8 training streams diverged; inspect paired_stream_audit.json.")
        write_json(output_dir / f"stage_u{initial_update:04d}_to_u{stop_at_update:04d}_result.json", result)
    except Exception as exc:
        write_json(
            output_dir / f"failure_{utc_now().replace(':', '').replace('-', '')}.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "started_at_utc": process_started_at,
                "failed_at_utc": utc_now(),
                "elapsed_wall_seconds": time.monotonic() - process_start,
                "gpu_uuid_required": GPU2_UUID,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
        raise


if __name__ == "__main__":
    main()
