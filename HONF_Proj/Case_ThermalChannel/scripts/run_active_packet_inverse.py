"""Bounded Thermal heat-allocation reuse of a frozen forward organizer.

The forward-only Stage-C choice must exist before this program reads inverse
tasks. All outputs are written beneath the ignored Run1509 result directory.
The script never calls the local reference solver; selected proposals are
frozen for a separate, individually counted physical check.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from channelthermal.evaluation.loading import load_model
from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.inverse.heat_allocation import (
    ThermalHeatFeatureScaler,
    ThermalHeatTask,
    build_thermal_heat_tasks,
    collate_thermal_heat_tasks,
)
from channelthermal.inverse.packet_reuse import (
    ThermalCandidateInterfaceBuilder,
    ThermalCandidatePacketProvider,
    _module_state_hash,
    _sample_arm,
    provider_known_inputs,
    train_matched_heat_diffusion,
)
from channelthermal.interaction_evidence.reference_adapter import load_stored_reference_case
from channelthermal.response_control.contracts import DesignInput, context_inputs, role_queries_from_record
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.runner import _make_input_template
from channelthermal.response_control.runner import _configure_native_expanded_response_interface_scope
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer
from honf_inverse_core.models.frozen_packet_diffusion import ConditionalPacketDenoiser, FrozenPacketDiffusion
from honf_runtime.compat import load_trusted_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    PROJECT_ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs/"
    "Run_1509_20260928_active_packet_organization_attempt07"
)
DEFAULT_FREEZE = RUN_ROOT / "stage_b_c_u0200_retry04/forward_selection_frozen.json"
DEFAULT_OUTPUT = RUN_ROOT / "inverse_frozen_heat_u0200"
DEFAULT_SOURCE = (
    PROJECT_ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs/"
    "Run_1804_20260905_081349_dense_pairwise_field_adaptation/checkpoints/best_field.pt"
)
DEFAULT_DATASET = Path(
    "/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5"
)
FORWARD_FIT_IDS = frozenset({"0001", "0273", "0304", "0318", "0320", "0333", "0335", "0348", "0350", "0319", "0349"})
SEED = 74029
SAMPLE_SEED = 17021


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to alter frozen evidence: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_attempt(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_forward_freeze(freeze_path: Path, source_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    freeze_path = freeze_path.expanduser().resolve()
    fit_path = freeze_path.with_name("stage_c_fit_frozen.json")
    if not freeze_path.is_file() or not fit_path.is_file():
        raise FileNotFoundError("Thermal forward-only fit and held freezes are both required.")
    selected = json.loads(freeze_path.read_text(encoding="utf-8"))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("case_rankings_frozen") is not True or fit.get("inverse_outcomes_consulted") is not False:
        raise RuntimeError("Stage-C fit did not freeze input-only rankings before inverse work.")
    frozen_path = Path(fit["selected_forward_checkpoint"]).resolve()
    if _sha256(frozen_path) != fit["selected_forward_checkpoint_sha256"]:
        raise RuntimeError("Stage-C frozen forward checkpoint changed after selection.")
    if _sha256(source_path) != fit["source_checkpoint_sha256"]:
        raise RuntimeError("The intact Run1804 source checkpoint hash changed.")
    if selected.get("selected_forward_checkpoint_sha256") != _sha256(frozen_path):
        raise RuntimeError("Final forward-only held review points to another frozen checkpoint.")
    if Path(selected["selected_forward_checkpoint"]).resolve() != frozen_path:
        raise RuntimeError("Final forward-only held review points to another checkpoint path.")
    if selected.get("inverse_outcomes_consulted") is not False:
        raise RuntimeError("Final forward selection must precede inverse outcomes.")
    held_path = Path(selected["held_forward_evaluation"]).resolve()
    if _sha256(held_path) != selected["held_forward_evaluation_sha256"]:
        raise RuntimeError("Held forward measurements changed after the selection was frozen.")
    held = json.loads(held_path.read_text(encoding="utf-8"))
    if held.get("fit_freeze_sha256") != _sha256(fit_path):
        raise RuntimeError("The held forward review does not bind the input-only fit freeze.")
    return fit, selected, _sha256(freeze_path)


def _development_tasks(tasks: tuple[ThermalHeatTask, ...], count: int = 8) -> tuple[ThermalHeatTask, ...]:
    """Select held tasks from geometry/context metadata, before inverse metrics."""

    eligible = sorted(
        (task for task in tasks if task.module_valid.sum() >= 2 and task.case_id not in FORWARD_FIT_IDS),
        key=lambda task: (int(task.module_valid.sum()), float(task.physical_context[0]), task.case_id),
    )
    if len(eligible) < count:
        raise ValueError("Too few M>=2 test cases for bounded inverse development.")
    indices = np.linspace(0, len(eligible) - 1, count).round().astype(int)
    return tuple(eligible[int(index)] for index in indices)


def _observed_only(task: ThermalHeatTask) -> ThermalHeatTask:
    """Keep the last four geometry-selected sensors disjoint for evaluation."""

    return replace(
        task,
        sensor_xy=task.sensor_xy[:8].copy(),
        sensor_features=task.sensor_features[:8].copy(),
    )


def _load_tasks(dataset_path: Path, *, include_test: bool) -> tuple[
    tuple[ThermalHeatTask, ...], tuple[ThermalHeatTask, ...],
    tuple[ThermalHeatTask, ...], ThermalHeatFeatureScaler,
]:
    all_train = build_thermal_heat_tasks(dataset_path, split="train", sensor_count=12)
    train = tuple(_observed_only(task) for task in all_train if task.case_id not in FORWARD_FIT_IDS)
    if any(task.case_id in FORWARD_FIT_IDS for task in train):
        raise RuntimeError("Forward utility families leaked into inverse fitting.")
    test_full = (
        _development_tasks(build_thermal_heat_tasks(dataset_path, split="test", sensor_count=12))
        if include_test else ()
    )
    test = tuple(_observed_only(task) for task in test_full)
    if len(train) < 8 or (include_test and len(set(task.case_id for task in test)) != 8):
        raise ValueError("Inverse cohorts are too small or duplicate a test task.")
    return train, test, test_full, ThermalHeatFeatureScaler.fit(train)


def _load_frozen_modules(
    fit: dict[str, Any], source_path: Path, first_task: ThermalHeatTask,
    scaler: ThermalHeatFeatureScaler, device: torch.device,
) -> tuple[torch.nn.Module, InputOnlyCoverOrganizer, ThermalCandidateInterfaceBuilder, int]:
    model, source = load_model(source_path, device)
    if int(source.get("epoch", source.get("current_epoch", -1))) != 4738:
        raise RuntimeError("Inverse reuse requires the exact Run1804 epoch-4738 source.")
    _configure_native_expanded_response_interface_scope(model)
    frozen_path = Path(fit["selected_forward_checkpoint"]).resolve()
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
    if frozen.get("selected_forward_arm") != "G" or frozen.get("source_checkpoint_sha256") != _sha256(source_path):
        raise RuntimeError("Frozen forward checkpoint has the wrong arm or source.")
    model.load_state_dict(frozen["physical_model_state"], strict=True)
    model.eval().requires_grad_(False)
    dataset_config = source["train_config"]["dataset"]
    normalization = source["global_normalization_stats"]
    probe = collate_thermal_heat_tasks((first_task,), scaler=scaler, device=device)
    temporary = SimpleNamespace(model=model, dataset_config=dataset_config, normalization_stats=normalization)
    with torch.no_grad():
        uniform_heat = (
            probe.total_heat[:, None, :] * probe.condition.module_valid[..., None]
            / probe.condition.module_valid.sum(dim=1).clamp_min(1)[:, None, None]
        )
        encoded = ThermalCandidateInterfaceBuilder._encode_candidate(
            temporary, uniform_heat, provider_known_inputs(probe), probe.condition
        )
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=int(encoded.env_features.shape[-1]),
        hidden_dim=96,
        role_count=8,
        frontier_utility_enabled=True,
        frontier_role_count=len(frozen["role_schema"]),
    ).to(device=device, dtype=encoded.module_tokens.dtype)
    organizer.load_state_dict(frozen["organizer_state"], strict=True)
    organizer.eval().requires_grad_(False)
    if _module_state_hash(organizer) != fit["selected_organizer_state_sha256"]:
        raise RuntimeError("Frozen organizer state differs from the forward-only fit decision.")
    builder = ThermalCandidateInterfaceBuilder(
        model, organizer,
        budget_fractions=frozen["budget_fractions"],
        role_tolerance=frozen["inverse_role_tolerance"],
        numerical_state_version=_sha256(frozen_path),
        dataset_config=dataset_config,
        normalization_stats=normalization,
    )
    # The exported packet embeddings are the organizer's projected source
    # states. Their width can differ from the physical core's encoded tokens.
    with torch.no_grad():
        interface = builder(uniform_heat, provider_known_inputs(probe), probe.condition)[0]
    if interface.module_embeddings is None or interface.environment_embeddings is None:
        raise RuntimeError("Frozen packet interface omitted source embeddings.")
    embedding_dim = int(interface.module_embeddings.shape[-1])
    if int(interface.environment_embeddings.shape[-1]) != embedding_dim:
        raise RuntimeError("Frozen packet source embedding widths disagree.")
    return model, organizer, builder, embedding_dim


def _denoiser(embedding_dim: int) -> ConditionalPacketDenoiser:
    return ConditionalPacketDenoiser(
        design_dim=1, module_dim=13, sensor_dim=3,
        embedding_dim=embedding_dim, hidden_dim=96, layers=3,
    )


def _task_manifest(
    train: tuple[ThermalHeatTask, ...], test: tuple[ThermalHeatTask, ...],
    scaler: ThermalHeatFeatureScaler, freeze_hash: str, source: Path, dataset: Path,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "forward_freeze_sha256": freeze_hash,
        "source_checkpoint_sha256": _sha256(source),
        "packed_dataset_sha256": _sha256(dataset),
        "train_case_ids": [task.case_id for task in train],
        "excluded_forward_fit_family_ids": sorted(FORWARD_FIT_IDS),
        "development_test_case_ids": [task.case_id for task in test],
        "development_selection_rule": "exclude forward-fit family aliases, then select eight evenly spaced test M>=2 cases sorted by module count, Re, case ID",
        "sensor_rule": "12 fixed geometry-only fluid grid cells; first eight observations supplied, last four withheld for disjoint checks",
        "module_feature_mean_train_only": scaler.module_mean.tolist(),
        "module_feature_std_train_only": scaler.module_std.tolist(),
        "sensor_feature_mean_train_only": scaler.sensor_mean.tolist(),
        "sensor_feature_std_train_only": scaler.sensor_std.tolist(),
        "inverse_goal": "positive heat vector with exactly the supplied total at known fixed geometry",
        "inverse_reference_scope": "local analytic-wake/shared-grid thermal generator; separate checked proposals only",
    }


def _train(args: argparse.Namespace, train: tuple[ThermalHeatTask, ...], scaler: ThermalHeatFeatureScaler,
           model: torch.nn.Module, organizer: InputOnlyCoverOrganizer,
           builder: ThermalCandidateInterfaceBuilder, embedding_dim: int,
           freeze_hash: str, device: torch.device) -> None:
    output = args.output.resolve()
    checkpoint_path = output / "checkpoints/latest_matched.pt"
    attempts = output / "inverse_optimizer_attempts.jsonl"
    resume = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path.exists() else None
    if resume is not None and resume.get("forward_freeze_sha256") != freeze_hash:
        raise RuntimeError("Inverse resume checkpoint belongs to a different frozen forward version.")
    attempted = {"I-G": 0, "I-dense": 0}
    if attempts.exists():
        with attempts.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("forward_freeze_sha256") != freeze_hash or row.get("arm") not in attempted:
                    raise RuntimeError("Inverse attempt ledger contains another frozen model or unknown arm.")
                attempted[row["arm"]] += 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    def save_checkpoint(payload: dict[str, Any]) -> None:
        _atomic_torch(checkpoint_path, {**payload, "forward_freeze_sha256": freeze_hash})

    def record_attempt(row: dict[str, Any]) -> None:
        arm = str(row["arm"])
        if attempted[arm] >= 800:
            raise RuntimeError(f"{arm} reached the 800 attempted-update ceiling, including replays.")
        _append_attempt(attempts, {**row, "forward_freeze_sha256": freeze_hash})
        attempted[arm] += 1

    start = time.perf_counter()
    result = train_matched_heat_diffusion(
        train, scaler=scaler, denoiser_template=_denoiser(embedding_dim),
        provider_factory=lambda known: ThermalCandidatePacketProvider(known, builder),
        frozen_modules={"forward": model, "organizer": organizer},
        updates=args.updates, batch_size=args.batch_size,
        steps=args.steps, seed=SEED, device=device,
        resume_payload=resume, checkpoint_callback=save_checkpoint,
        attempt_callback=record_attempt, checkpoint_every_updates=10,
    )
    endpoint = output / f"checkpoints/matched_u{args.updates:04d}.pt"
    if endpoint.exists():
        if _sha256(endpoint) != _sha256(checkpoint_path):
            raise FileExistsError(f"Refusing to alter a prior inverse endpoint: {endpoint}")
    else:
        temporary = endpoint.with_name(f".{endpoint.name}.tmp")
        shutil.copyfile(checkpoint_path, temporary)
        os.replace(temporary, endpoint)
    summary = {
        "format_version": 1,
        "status": "matched_inverse_training_completed",
        "updates_per_arm": result.updates_per_arm,
        "attempted_optimizer_calls_per_arm_including_replays": attempted,
        "diffusion_steps": args.steps,
        "equal_initial_denoiser_sha256": result.initial_denoiser_hash,
        "denoiser_parameters_per_arm": sum(parameter.numel() for parameter in result.graph_model.denoiser.parameters()),
        "graph_final_loss": result.graph_losses[-1],
        "dense_final_loss": result.dense_losses[-1],
        "graph_tail_median_loss": float(np.median(result.graph_losses[-20:])),
        "dense_tail_median_loss": float(np.median(result.dense_losses[-20:])),
        "frozen_hashes_before": result.frozen_state_hashes_before,
        "frozen_hashes_after": result.frozen_state_hashes_after,
        "complete_wall_seconds": time.perf_counter() - start,
        "optimizer_attempt_ledger": str(attempts),
        "checkpoint_path": str(endpoint),
        "checkpoint_sha256": _sha256(endpoint),
        "forward_freeze_sha256": freeze_hash,
    }
    _atomic_json(output / f"inverse_train_u{args.updates:04d}.json", summary, immutable=True)
    print(json.dumps(summary, indent=2))


def _sample_condition(
    tasks: tuple[ThermalHeatTask, ...], scaler: ThermalHeatFeatureScaler,
    graph: FrozenPacketDiffusion, dense: FrozenPacketDiffusion,
    builder: ThermalCandidateInterfaceBuilder, device: torch.device,
    *, control: str,
) -> dict[str, Any]:
    if control == "changed_temperature_plus_5_dataset_units":
        tasks = tuple(replace(task, sensor_features=np.column_stack((task.sensor_features[:, :2], task.sensor_features[:, 2] + 5.0)).astype(np.float32)) for task in tasks)
    if control == "four_observations":
        tasks = tuple(replace(task, sensor_xy=task.sensor_xy[:4].copy(), sensor_features=task.sensor_features[:4].copy()) for task in tasks)
    if control == "alternate_six_observations":
        tasks = tuple(replace(task, sensor_xy=task.sensor_xy[1:7].copy(), sensor_features=task.sensor_features[1:7].copy()) for task in tasks)
    batch = collate_thermal_heat_tasks(tasks, scaler=scaler, device=device)
    condition = batch.condition
    if control == "no_observations":
        condition = replace(condition, sensor_valid=torch.zeros_like(condition.sensor_valid))
    counts = {"graph": 0, "dense": 0}
    frontier_counts: dict[str, dict[str, int]] = {"graph": {}, "dense": {}}

    def provider_for(arm: str):
        underlying = ThermalCandidatePacketProvider(provider_known_inputs(batch), builder)

        def counted(state, current):
            counts[arm] += 1
            links = underlying(state, current)
            if arm == "dense":
                key = "full_inverse_access_after_same_frozen_builder"
                frontier_counts[arm][key] = frontier_counts[arm].get(key, 0) + len(underlying.last_frontier_evidence)
            else:
                for evidence in underlying.last_frontier_evidence:
                    scope = "research_unsupported" if evidence.unsupported_at_budget else "predicted_adequate"
                    key = f"{scope}/K{len(evidence.predicted_frontier)}"
                    frontier_counts[arm][key] = frontier_counts[arm].get(key, 0) + 1
            return links

        return counted

    start = time.perf_counter()
    with torch.no_grad():
        g_heat, _, g_trail, timesteps = _sample_arm(
            graph, condition, provider_for("graph"), total_heat=batch.total_heat,
            samples=8, seed=SAMPLE_SEED, dense=False, save_every=4,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        g_seconds = time.perf_counter() - start
        d_heat, _, d_trail, dense_timesteps = _sample_arm(
            dense, condition, provider_for("dense"), total_heat=batch.total_heat,
            samples=8, seed=SAMPLE_SEED, dense=True, save_every=4,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        d_seconds = time.perf_counter() - start - g_seconds
    if timesteps != dense_timesteps:
        raise RuntimeError("Matched sampler arms recorded different reverse steps.")
    return {
        "control": control,
        "case_ids": batch.case_ids,
        "graph_heat": g_heat.detach().cpu().numpy(),
        "dense_heat": d_heat.detach().cpu().numpy(),
        "graph_heat_trails": g_trail.detach().cpu().numpy(),
        "dense_heat_trails": d_trail.detach().cpu().numpy(),
        "reference_heat": batch.clean_heat.detach().cpu().numpy(),
        "total_heat": batch.total_heat.detach().cpu().numpy(),
        "module_valid": batch.condition.module_valid.detach().cpu().numpy(),
        "sensor_xy": batch.sensor_xy.detach().cpu().numpy(),
        "sensor_temperature_dataset_units": np.stack([task.sensor_features[:, 2] for task in tasks]),
        "trail_timesteps": timesteps,
        "candidate_interface_calls": counts,
        "candidate_frontier_scope_counts": frontier_counts,
        "sample_wall_seconds": {"I-G": g_seconds, "I-dense": d_seconds},
    }


def _metrics(values: np.ndarray, reference: np.ndarray, valid: np.ndarray, total: np.ndarray) -> dict[str, Any]:
    errors: list[float] = []
    pair_spread: list[float] = []
    heat_valid = 0
    sample_count = 0
    for case in range(values.shape[1]):
        mask = valid[case].astype(bool)
        target = reference[case, mask] / total[case, 0]
        samples = values[:, case, mask] / total[case, 0]
        good_draws: list[int] = []
        for draw, sample in enumerate(samples):
            sample_count += 1
            good = bool(np.isfinite(sample).all() and np.all(sample >= -1.0e-7) and abs(float(sample.sum()) - 1.0) < 1.0e-5)
            heat_valid += good
            if good:
                good_draws.append(draw)
                errors.append(float(np.abs(sample - target).sum()))
        for first_index, first in enumerate(good_draws):
            for second in good_draws[first_index + 1:]:
                pair_spread.append(float(np.abs(samples[first] - samples[second]).sum()))
    return {
        "valid_nonnegative_supplied_total_fraction": heat_valid / sample_count,
        "valid_heat_samples": heat_valid,
        "attempted_samples": sample_count,
        "median_L1_fraction_distance_to_one_known_design": (
            float(np.median(errors)) if errors else None
        ),
        "distance_is_conditional_only_on_heat_validity": True,
        "mean_pairwise_L1_fraction_diversity_among_valid_heat_samples": (
            float(np.mean(pair_spread)) if pair_spread else None
        ),
        "valid_heat_sample_pairs": len(pair_spread),
        "diversity_is_conditional_only_on_heat_validity": True,
        "reference_solver_acceptance_unmeasured_here": True,
    }


def _paired_fraction_shift(
    values: np.ndarray, baseline: np.ndarray, valid: np.ndarray, total: np.ndarray,
) -> float:
    """Compare matched samples using the collated, padded module validity."""
    if values.shape != baseline.shape or values.ndim != 3:
        raise ValueError("Paired samples must have the same [draw, case, padded module] shape.")
    if valid.shape != values.shape[1:] or total.shape != (values.shape[1], 1):
        raise ValueError("Paired sample masks and totals must match the padded case batch.")
    shifts = []
    for case in range(values.shape[1]):
        mask = valid[case].astype(bool)
        shifts.extend(
            np.abs(values[:, case, mask] - baseline[:, case, mask]).sum(axis=-1)
            / float(total[case, 0])
        )
    return float(np.mean(shifts))


def _sample_fixed_weight_rewire(
    tasks: tuple[ThermalHeatTask, ...], scaler: ThermalHeatFeatureScaler,
    graph: FrozenPacketDiffusion, builder: ThermalCandidateInterfaceBuilder,
    device: torch.device,
) -> dict[str, Any]:
    """Rewire packet permissions, preserving each packet and source degree."""

    from run_active_packet_stage_c import _rewire_membership

    swap_counts = {route: 0 for route in builder.budget_fractions}
    attempted_plans = 0

    def intervene(plan, case):
        nonlocal attempted_plans
        attempted_plans += 1
        updated = plan
        records = {}
        for route in sorted(builder.budget_fractions):
            updated, row = _rewire_membership(updated, route, SEED + case * 101 + sum(map(ord, route)))
            records[route] = row
            swap_counts[route] += int(row["accepted_switches"])
        return updated, records

    rewired = ThermalCandidateInterfaceBuilder(
        builder.model, builder.organizer,
        budget_fractions=builder.budget_fractions,
        role_tolerance=builder.role_tolerance.detach().cpu().tolist(),
        numerical_state_version=builder.numerical_state_version,
        evidence_scope="frozen-forward-fixed-weight-degree-and-packet-size-rewire",
        max_frontier_depth=builder.max_frontier_depth,
        dataset_config=builder.dataset_config,
        normalization_stats=builder.normalization_stats,
        plan_intervention=intervene,
    )
    batch = collate_thermal_heat_tasks(tasks, scaler=scaler, device=device)
    provider = ThermalCandidatePacketProvider(provider_known_inputs(batch), rewired)
    calls = 0

    def counted(state, condition):
        nonlocal calls
        calls += 1
        return provider(state, condition)

    start = time.perf_counter()
    with torch.no_grad():
        heat, _, trails, timesteps = _sample_arm(
            graph, batch.condition, counted,
            total_heat=batch.total_heat, samples=8, seed=SAMPLE_SEED,
            dense=False, save_every=4,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {
        "graph_heat": heat.detach().cpu().numpy(),
        "graph_heat_trails": trails.detach().cpu().numpy(),
        "timesteps": timesteps,
        "candidate_interface_calls": calls,
        "candidate_case_plans": attempted_plans,
        "accepted_packet_degree_preserving_switches": swap_counts,
        "complete_generation_seconds": time.perf_counter() - start,
    }


def _evaluate(args: argparse.Namespace, test: tuple[ThermalHeatTask, ...],
              test_full: tuple[ThermalHeatTask, ...], scaler: ThermalHeatFeatureScaler,
              builder: ThermalCandidateInterfaceBuilder, embedding_dim: int,
              freeze_hash: str, device: torch.device) -> None:
    output = args.output.resolve()
    checkpoint_path = output / f"checkpoints/matched_u{args.updates:04d}.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("forward_freeze_sha256") != freeze_hash or saved.get("completed_updates") != args.updates:
        raise RuntimeError("Inverse evaluation requires the exact requested paired checkpoint and frozen forward choice.")
    graph = FrozenPacketDiffusion(_denoiser(embedding_dim), task="heat", steps=args.steps).to(device)
    dense = FrozenPacketDiffusion(_denoiser(embedding_dim), task="heat", steps=args.steps).to(device)
    graph.load_state_dict(saved["graph_model_state"], strict=True)
    dense.load_state_dict(saved["dense_model_state"], strict=True)
    graph.eval().requires_grad_(False)
    dense.eval().requires_grad_(False)
    controls = ("as_observed", "changed_temperature_plus_5_dataset_units", "no_observations", "four_observations", "alternate_six_observations")
    summary: dict[str, Any] = {
        "format_version": 1, "forward_freeze_sha256": freeze_hash,
        "matched_inverse_checkpoint_sha256": _sha256(checkpoint_path),
        "updates_per_arm": args.updates, "reverse_steps": args.steps,
        "samples_per_task_per_arm_per_control": 8,
        "tasks": [task.case_id for task in test], "controls": {},
        "disjoint_held_sensors_per_task": 4,
    }
    all_arrays: dict[str, np.ndarray] = {}
    base: dict[str, Any] | None = None
    for control in controls:
        item = _sample_condition(test, scaler, graph, dense, builder, device, control=control)
        if control == "as_observed":
            base = item
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                all_arrays[f"{control}__{key}"] = value
        summary["controls"][control] = {
            "I-G": _metrics(item["graph_heat"], item["reference_heat"], item["module_valid"], item["total_heat"]),
            "I-dense": _metrics(item["dense_heat"], item["reference_heat"], item["module_valid"], item["total_heat"]),
            "candidate_interface_calls": item["candidate_interface_calls"],
            "candidate_frontier_scope_counts": item["candidate_frontier_scope_counts"],
            "complete_generation_seconds": item["sample_wall_seconds"],
            "trail_timesteps": list(item["trail_timesteps"]),
        }
        if control != "as_observed":
            assert base is not None
            for arm, key in (("I-G", "graph_heat"), ("I-dense", "dense_heat")):
                summary["controls"][control][arm]["paired_L1_fraction_shift_from_as_observed_mean"] = (
                    _paired_fraction_shift(item[key], base[key], base["module_valid"], base["total_heat"])
                )
        print(f"completed {control}: {item['candidate_interface_calls']}", flush=True)
    assert base is not None
    rewired = _sample_fixed_weight_rewire(test, scaler, graph, builder, device)
    all_arrays["fixed_weight_rewire__graph_heat"] = rewired["graph_heat"]
    all_arrays["fixed_weight_rewire__graph_heat_trails"] = rewired["graph_heat_trails"]
    rewire_metrics = _metrics(rewired["graph_heat"], base["reference_heat"], base["module_valid"], base["total_heat"])
    summary["fixed_weight_packet_rewire"] = {
        "I-G": rewire_metrics,
        "same_denoiser_weights_and_sample_noise_as_I_G": True,
        "paired_L1_fraction_shift_from_I_G_mean": _paired_fraction_shift(
            rewired["graph_heat"], base["graph_heat"], base["module_valid"], base["total_heat"]
        ),
        "packet_and_source_degrees_preserved": True,
        "accepted_switches": rewired["accepted_packet_degree_preserving_switches"],
        "candidate_interface_calls": rewired["candidate_interface_calls"],
        "candidate_case_plans": rewired["candidate_case_plans"],
        "complete_generation_seconds": rewired["complete_generation_seconds"],
        "trail_timesteps": list(rewired["timesteps"]),
    }
    all_arrays["disjoint_held_sensor_xy_m"] = np.stack([task.sensor_xy[8:] for task in test_full])
    all_arrays["disjoint_held_sensor_temperature_dataset_units"] = np.stack([task.sensor_features[8:, 2] for task in test_full])
    # The eight checker proposals are selected by a fixed index before any
    # new local-reference outcomes are opened. This is a deliberately neutral
    # preselection, not an optimization ranking or a surrogate acceptance.
    selected = []
    for case in (0, 2, 4, 6):
        task = test[case]
        for arm, key in (("I-G", "graph_heat"), ("I-dense", "dense_heat")):
            selected.append({
                "case_id": task.case_id, "arm": arm, "sample_index": 0,
                "module_centers_xy_m": task.module_centers.tolist(),
                "candidate_heat_dataset_units": base[key][0, case, :task.module_valid.size].tolist(),
                "supplied_total_heat_dataset_units": task.total_heat,
                "reference_solve_status": "not_opened_at_preselection",
            })
    proposal_manifest = {
        "format_version": 1,
        "preselection_rule": "sample index zero under shared seed 17021 for each arm on development task positions 0,2,4,6",
        "selection_uses_hidden_heat_or_reference_outcomes": False,
        "selected_before_local_reference_calls": True,
        "reference_attempt_ceiling_for_selected_proposals": 8,
        "generator": "local analytic-wake/shared-grid thermal model, not CFD",
        "forward_freeze_sha256": freeze_hash,
        "matched_inverse_checkpoint_sha256": _sha256(checkpoint_path),
        "proposals": selected,
    }
    array_path = output / "inverse_samples_and_trails.npz"
    if array_path.exists():
        raise FileExistsError(f"Refusing to replace saved inverse samples: {array_path}")
    temporary_array = array_path.with_name(f".{array_path.stem}.tmp.npz")
    np.savez_compressed(temporary_array, **all_arrays)
    os.replace(temporary_array, array_path)
    summary["sample_array_sha256"] = _sha256(array_path)
    proposal_manifest["inverse_samples_and_trails_sha256"] = summary["sample_array_sha256"]
    _atomic_json(output / "preselected_local_reference_proposals.json", proposal_manifest, immutable=True)
    summary["preselected_proposals_sha256"] = _sha256(output / "preselected_local_reference_proposals.json")
    _atomic_json(output / "inverse_sample_summary.json", summary, immutable=True)
    print(json.dumps(summary, indent=2))


def _score_candidates(args: argparse.Namespace, test_full: tuple[ThermalHeatTask, ...],
                      model: torch.nn.Module, device: torch.device,
                      freeze_hash: str) -> None:
    """Measure candidate sensor consistency with the same frozen full student.

    This is a surrogate check on new heat vectors, not a local-reference solve.
    The physical generator is invoked only by a separately authorized, counted
    workflow after the neutral proposal manifest has been frozen.
    """

    output = args.output.resolve()
    sample_path = output / "inverse_samples_and_trails.npz"
    sample_summary = json.loads((output / "inverse_sample_summary.json").read_text(encoding="utf-8"))
    if sample_summary["forward_freeze_sha256"] != freeze_hash or sample_summary["sample_array_sha256"] != _sha256(sample_path):
        raise RuntimeError("Inverse arrays differ from the frozen paired sample manifest.")
    target_json = output / "inverse_frozen_surrogate_sensor_checks.json"
    target_npz = output / "inverse_frozen_surrogate_sensor_checks.npz"
    if target_json.exists() or target_npz.exists():
        raise FileExistsError("Frozen surrogate sensor checks already exist; keep the measured output immutable.")
    source_path = args.source_checkpoint.expanduser().resolve()
    source = load_trusted_checkpoint(source_path, map_location="cpu")
    dataset = GlobalChannelThermalDataset(
        args.dataset.expanduser().resolve(), split="train", points_per_case=1,
        normalize_inputs=False, normalize_targets=False,
        random_point_sampling=False, include_grid=False, include_structure_targets=False,
    )
    template = _make_input_template(dataset)
    del dataset
    operator = DifferentiableThermalOperator(
        model, template,
        dataset_config=source["train_config"]["dataset"],
        normalization_stats=source["global_normalization_stats"],
        query_batch_size=512,
    )
    with np.load(sample_path) as data:
        graph_heat = np.asarray(data["as_observed__graph_heat"], dtype=np.float32)
        dense_heat = np.asarray(data["as_observed__dense_heat"], dtype=np.float32)
    if graph_heat.shape != dense_heat.shape or graph_heat.shape[:2] != (8, len(test_full)):
        raise RuntimeError("The two inverse arms do not have matched saved sample axes.")
    predictions = np.zeros((2, 8, len(test_full), 12), dtype=np.float32)
    clean_prediction = np.zeros((len(test_full), 12), dtype=np.float32)
    truth = np.stack([task.sensor_features[:, 2] for task in test_full]).astype(np.float32)
    case_times: list[float] = []
    start_all = time.perf_counter()
    for case, task in enumerate(test_full):
        start_case = time.perf_counter()
        record = load_stored_reference_case(args.dataset.expanduser().resolve(), task.case_id)
        original = DesignInput.from_state(record.design, device=device)
        active = torch.nonzero(original.module_present, as_tuple=False).reshape(-1)
        if int(active.numel()) != task.module_valid.size:
            raise RuntimeError(f"Active module count changed for Thermal test case {task.case_id}.")
        np.testing.assert_allclose(
            original.module_positions[active].detach().cpu().numpy(),
            task.module_centers, rtol=1.0e-6, atol=1.0e-6,
        )
        roles = dict(role_queries_from_record(record, device=device))
        roles["fluid_fields"] = replace(
            roles["fluid_fields"],
            query_features=torch.as_tensor(task.sensor_xy, dtype=torch.float32, device=device),
        )
        context = context_inputs(record.context)

        def predict(heat: np.ndarray | None) -> np.ndarray:
            if heat is None:
                design = original
            else:
                heating = torch.zeros_like(original.module_heating)
                heating[active] = torch.as_tensor(heat[:int(active.numel())], dtype=heating.dtype, device=device)
                design = DesignInput(original.module_positions, heating, original.module_present)
            with torch.no_grad():
                values = operator(design, context, roles).role_values["fluid_fields"]
            result = values[:, operator.field_names.index("temperature")].detach().cpu().numpy()
            if result.shape != (12,) or not np.isfinite(result).all():
                raise FloatingPointError("Frozen forward candidate sensor predictions are nonfinite or misaligned.")
            return result

        clean_prediction[case] = predict(None)
        for arm, array in enumerate((graph_heat, dense_heat)):
            for draw in range(8):
                predictions[arm, draw, case] = predict(array[draw, case])
        case_times.append(time.perf_counter() - start_case)
        print(f"scored Thermal inverse case {case+1}/{len(test_full)}: {task.case_id}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    residual = predictions - truth[None, None, :, :]
    clean_residual = clean_prediction - truth
    summary: dict[str, Any] = {
        "format_version": 1,
        "forward_freeze_sha256": freeze_hash,
        "inverse_samples_sha256": _sha256(sample_path),
        "reference_source": "stored clean-layout temperatures only; generated candidate heats have no stored reference",
        "candidate_prediction_scope": "same frozen G physical student under policy-free full access; not independent physical truth",
        "observed_sensor_count": 8,
        "disjoint_held_sensor_count": 4,
        "clean_design_full_student_rmse_observed_temperature_units": float(np.sqrt(np.mean(clean_residual[:, :8] ** 2))),
        "clean_design_full_student_rmse_held_temperature_units": float(np.sqrt(np.mean(clean_residual[:, 8:] ** 2))),
        "candidate_full_forward_calls": int(8 + predictions.shape[0] * predictions.shape[1] * predictions.shape[2]),
        "case_wall_seconds": case_times,
        "total_complete_wall_seconds": time.perf_counter() - start_all,
        "arms": {},
    }
    for arm, label in enumerate(("I-G", "I-dense")):
        observed = np.sqrt(np.mean(residual[arm, :, :, :8] ** 2, axis=-1))
        held = np.sqrt(np.mean(residual[arm, :, :, 8:] ** 2, axis=-1))
        summary["arms"][label] = {
            "candidate_count": int(observed.size),
            "observed_sensor_rmse_temperature_units_mean": float(np.mean(observed)),
            "observed_sensor_rmse_temperature_units_median": float(np.median(observed)),
            "disjoint_held_sensor_rmse_temperature_units_mean": float(np.mean(held)),
            "disjoint_held_sensor_rmse_temperature_units_median": float(np.median(held)),
            "per_task_observed_rmse_temperature_units_mean": np.mean(observed, axis=0).tolist(),
            "per_task_disjoint_held_rmse_temperature_units_mean": np.mean(held, axis=0).tolist(),
        }
    temporary_npz = target_npz.with_name(f".{target_npz.stem}.tmp.npz")
    np.savez_compressed(temporary_npz, predictions_temperature_units=predictions,
                        clean_predictions_temperature_units=clean_prediction,
                        stored_clean_temperatures_dataset_units=truth,
                        residual_temperature_units=residual)
    os.replace(temporary_npz, target_npz)
    summary["numerical_arrays_sha256"] = _sha256(target_npz)
    _atomic_json(target_json, summary, immutable=True)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "train", "evaluate", "score"), required=True)
    parser.add_argument("--forward-freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.updates <= 800:
        raise ValueError("Thermal inverse denoiser cap is 800 updates per arm.")
    if not 2 <= args.steps <= 20:
        raise ValueError("Bounded Thermal inverse sampling uses at most 20 reverse steps.")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    source_path = args.source_checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    fit, _, freeze_hash = _load_forward_freeze(args.forward_freeze, source_path)
    include_test = args.phase in {"evaluate", "score"}
    train, test, test_full, scaler = _load_tasks(dataset_path, include_test=include_test)
    output = args.output.expanduser().resolve()
    _atomic_json(
        output / "inverse_train_task_manifest.json",
        _task_manifest(train, (), scaler, freeze_hash, source_path, dataset_path),
        immutable=True,
    )
    if include_test:
        _atomic_json(
            output / "inverse_task_manifest.json",
            _task_manifest(train, test, scaler, freeze_hash, source_path, dataset_path),
            immutable=True,
        )
    model, organizer, builder, embedding_dim = _load_frozen_modules(fit, source_path, train[0], scaler, device)
    if args.phase == "preflight":
        batch = collate_thermal_heat_tasks((train[0],), scaler=scaler, device=device)
        with torch.no_grad():
            links = ThermalCandidatePacketProvider(provider_known_inputs(batch), builder)(
                torch.zeros_like(batch.clean_state), batch.condition
            )
        print(json.dumps({
            "status": "frozen_candidate_provider_preflight_passed",
            "case_id": train[0].case_id,
            "module_links": list(links.module_source.shape),
            "sensor_links": list(links.sensor_source.shape),
            "embedding_dim": embedding_dim,
            "forward_freeze_sha256": freeze_hash,
            "selected_frontier_evidence": [value.__dict__ for value in builder.last_frontier_evidence],
        }, indent=2))
    elif args.phase == "train":
        _train(args, train, scaler, model, organizer, builder, embedding_dim, freeze_hash, device)
    elif args.phase == "evaluate":
        _evaluate(args, test, test_full, scaler, builder, embedding_dim, freeze_hash, device)
    else:
        _score_candidates(args, test_full, model, device, freeze_hash)


if __name__ == "__main__":
    main()
