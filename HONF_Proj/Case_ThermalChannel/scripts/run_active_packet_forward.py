"""Train matched Thermal grouped-packet (G) and direct-pair (P) students.

The first bounded gate is 100 updates per arm: 50 exact full-access updates,
then 50 updates at the primary 0.90 QE plus selected MM/ME capacity over
sampled complete frontiers. An optional exact optimizer-state continuation to
u300 completes u51-200 at 0.90 and uses seeded shuffled 20/60/20 full/0.90/0.75
five-update blocks thereafter. All evidence comes from the eight named train
response families and packed train historical-value replay. Outputs remain
under ignored Run1509 directories; this script never launches a reference
solver.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation.loading import load_model
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.response_control.active_packet import (
    ThermalCoverPlanBuilder,
    ThermalHardValueSoftOperator,
    make_direct_scorer,
)
from channelthermal.response_control.contracts import (
    DesignInput,
    context_inputs,
    role_queries_from_stencil,
)
from channelthermal.response_control.algebra import predict_stencil
from channelthermal.response_control.historical import HistoricalValueSource
from channelthermal.response_control.losses import compute_stencil_loss_terms
from channelthermal.response_control.losses import weighted_masked_mse
from channelthermal.response_control.losses import historical_absolute_value_loss
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.runner import (
    _configure_native_expanded_response_interface_scope,
    _make_input_template,
    _mixed_specs,
    _resolve_dataset_path,
    derive_training_scales,
)
from channelthermal.response_control.sampling import (
    ReceiverSamplingConfig,
    sample_training_panel,
)
from channelthermal.response_control.training import (
    StagedTrainingConfig,
    TrainingStage,
    TrainingStep,
    restore_checkpoint_payload,
    run_staged_fit,
)
from honf_forward_core.interface_fields.budgeted_frontier import enumerate_frontier_cuts
from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs/"
    "Run_1804_20260905_081349_dense_pairwise_field_adaptation/checkpoints/best_field.pt"
)
DEFAULT_ATLAS = PROJECT_ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
DEFAULT_RUN = (
    PROJECT_ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs/"
    "Run_1509_20260928_active_packet_organization_attempt03"
)
TRAIN_ATLAS_IDS = ("0001", "0304", "0318", "0320", "0333", "0335", "0348", "0350")
LOSS_WEIGHTS = {
    "value": 1.0,
    "finite": 1.0,
    "finite_peak": 1.0,
    "pressure_value": 1.0,
    "pressure_response": 1.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class _TrainingBundle(nn.Module):
    """Keep the forward student and its independent route learner together."""

    def __init__(self, forward_model: nn.Module, route_model: nn.Module) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.route_model = route_model


class _FrontierSchedule:
    def __init__(
        self,
        *,
        warm_updates: int,
        seed: int,
        extra_route: str,
        extended_curriculum: bool = False,
    ) -> None:
        self.warm_updates = int(warm_updates)
        self.seed = int(seed)
        self.extra_route = str(extra_route).upper()
        self.extended_curriculum = bool(extended_curriculum)
        if self.extra_route not in {"MM", "ME"}:
            raise ValueError("Thermal Stage A must select exactly one of MM or ME.")
        if self.extended_curriculum and self.warm_updates != 50:
            raise ValueError("The planned Thermal curriculum uses exactly 50 initial full-access updates.")
        self.current_update = 0

    def budgets(self) -> dict[str, float]:
        if self.current_update < self.warm_updates:
            fraction = 1.0
        elif self.extended_curriculum and self.current_update >= 200:
            # The plan's post-primary mix is one full, three primary, and one
            # stress update per five-update block. Shuffle the order by a
            # deterministic block seed so both arms replay the same budget
            # stream without imposing any frontier-K histogram.
            slot = self.current_update - 200
            choices = [1.00, 0.90, 0.90, 0.90, 0.75]
            random.Random(self.seed + (slot // 5) * 1_000_003).shuffle(choices)
            fraction = choices[slot % 5]
        else:
            fraction = 0.90
        return {"QE": fraction, self.extra_route: fraction}

    def cut(self, tree: Any) -> tuple[int, ...]:
        cuts = enumerate_frontier_cuts(tree, max_depth=3)
        if not cuts:
            raise RuntimeError("Thermal receiver tree produced no valid complete frontier.")
        if self.current_update < self.warm_updates:
            return cuts[-1]
        # Stochastically expose the shallow root and the larger refinements,
        # with no prescribed K sequence or histogram. The root receives a
        # modest per-cut probability boost so uniform sampling over all cuts
        # does not starve it; realized K is reported from the route ledger.
        weights = [4.0 if len(cut) == 1 else 1.0 for cut in cuts]
        rng = random.Random(self.seed + self.current_update * 1_000_003)
        return tuple(rng.choices(cuts, weights=weights, k=1)[0])


def _budgeted_route_status(work: Any) -> str:
    """Describe absent route savings without relabeling them as sparse success."""

    if float(work.requested_fraction) >= 1.0:
        return "full_access_budget"
    full_pairs = int(work.full_unique_pairs)
    selected_pairs = int(work.selected_unique_pairs)
    if full_pairs <= 0:
        return "no_eligible_pairs"
    if selected_pairs <= 0:
        return "empty_projection_no_sparse_success"
    if selected_pairs >= full_pairs or not bool(work.sparse_success):
        return "no_savings_at_budget"
    return "sparse_success"


def _route_status_counts(path: Path) -> dict[str, dict[str, int]]:
    """Count actual budgeted route outcomes; unbudgeted routes remain bypasses."""

    result: dict[str, Counter[str]] = {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for mechanism, work in row.get("routes", {}).items():
                if float(work.get("requested_fraction", 1.0)) >= 1.0:
                    continue
                status = work.get("route_status")
                if status is None:
                    status = _budgeted_route_status(type("RouteStatus", (), work)())
                result.setdefault(str(mechanism), Counter())[str(status)] += 1
    return {
        mechanism: dict(sorted(counts.items()))
        for mechanism, counts in sorted(result.items())
    }


class _ScheduledThermalOperator:
    """Apply one fixed budget/cut across each whole absolute response stencil."""

    def __init__(
        self,
        *,
        paired: ThermalHardValueSoftOperator,
        schedule: _FrontierSchedule,
        hard_builder: ThermalCoverPlanBuilder,
        soft_builder: ThermalCoverPlanBuilder,
        state_labels: tuple[str, ...],
        arm: str,
        route_log: Path,
        sparse_route_verified: dict[str, bool],
    ) -> None:
        self.paired = paired
        self.schedule = schedule
        self.hard_builder = hard_builder
        self.soft_builder = soft_builder
        self.state_labels = state_labels
        self.arm = arm
        self.route_log = route_log
        self.sparse_route_verified = sparse_route_verified
        self.calls_this_update = 0
        self._apply_stage()

    def _apply_stage(self) -> None:
        budgets = self.schedule.budgets()
        self.hard_builder.budget_fractions = dict(budgets)
        self.soft_builder.budget_fractions = dict(budgets)

    def __call__(self, design: DesignInput, context: Mapping[str, Any], role_queries: Mapping[str, Any]):
        call_index = self.calls_this_update
        if call_index < len(self.state_labels):
            state_label = self.state_labels[call_index]
        elif call_index == len(self.state_labels):
            state_label = "historical_value_replay"
        else:
            raise RuntimeError("Unexpected extra native calls inside one Thermal optimizer update.")
        prediction = self.paired(design, context, role_queries)
        budgets = self.schedule.budgets()
        if all(value < 1.0 for value in budgets.values()):
            records = self.hard_builder.last_records
            if not records:
                raise RuntimeError("Sparse Thermal hard forward did not expose route-work records.")
            for case_record in records:
                route_rows = {}
                for mechanism, work in case_record["routes"].items():
                    route_rows[mechanism] = asdict(work)
                    if work.requested_fraction < 1.0:
                        route_status = _budgeted_route_status(work)
                        route_rows[mechanism]["route_status"] = route_status
                        if route_status == "sparse_success":
                            self.sparse_route_verified[mechanism] = True
                _append_jsonl(
                    self.route_log,
                    {
                        "arm": self.arm,
                        "optimizer_update": self.schedule.current_update + 1,
                        "state": state_label,
                        "frontier": case_record["frontier"],
                        "budgets": budgets,
                        "routes": route_rows,
                        "full_access_bypass_routes": case_record["full_access_bypass_routes"],
                    },
                )
        self.calls_this_update += 1
        return prediction

    def optimizer_attempt(self, completed: int, attempted_total: int) -> None:
        self.schedule.current_update = int(completed) + 1
        self.calls_this_update = 0
        self._apply_stage()


def _route_module(arm: str, core: Any, encoded: Any, device: torch.device) -> nn.Module:
    if arm == "G":
        return InputOnlyCoverOrganizer(
            state_dim=int(encoded.module_tokens.shape[-1]),
            module_feature_dim=int(encoded.module_features.shape[-1]),
            environment_feature_dim=int(encoded.env_features.shape[-1]),
            hidden_dim=96,
            role_count=8,
        ).to(device=device, dtype=encoded.module_tokens.dtype)
    if arm == "P":
        return make_direct_scorer(core, device=device, dtype=encoded.module_tokens.dtype, hidden_dim=96)
    raise ValueError(f"Unknown Thermal arm {arm!r}.")


def _route_builder(
    arm: str,
    *,
    core: Any,
    route_model: nn.Module,
    extra_route: str,
    schedule: _FrontierSchedule,
) -> tuple[ThermalCoverPlanBuilder, ThermalCoverPlanBuilder]:
    selector = lambda case, encoded, module_state, tree: schedule.cut(tree)
    common = {
        "core": core,
        "budget_fractions": schedule.budgets(),
        "extra_route": extra_route,
        "frontier_selector": selector,
    }
    if arm == "G":
        hard = ThermalCoverPlanBuilder(mode="G", organizer=route_model, **common)
        soft = ThermalCoverPlanBuilder(mode="G", organizer=route_model, **common)
    else:
        hard = ThermalCoverPlanBuilder(mode="P", direct_scorer=route_model, **common)
        soft = ThermalCoverPlanBuilder(mode="P", direct_scorer=route_model, **common)
    return hard, soft


def _selected_train_stencils(atlas_dir: Path, sampling: ReceiverSamplingConfig):
    raw = []
    for family_id in TRAIN_ATLAS_IDS:
        path = atlas_dir / f"train_{family_id}_responses.npz"
        stencil, _ = load_response_atlas_stencil(path)
        if stencil.split.value != "train":
            raise ValueError(f"Thermal response atlas {path} is not a train partition.")
        raw.append(stencil)
    if len({item.physical_family_id for item in raw}) != len(TRAIN_ATLAS_IDS):
        raise ValueError("Thermal Run1509 needs eight distinct declared train families.")
    sampled = sample_training_panel(raw, config=sampling)
    variants = [tuple(item.stencil.variants) for item in sampled]
    if any(row != variants[0] for row in variants[1:]):
        raise ValueError("All eight Thermal train families must have the same response-state layout.")
    return tuple(raw), tuple(item.stencil for item in sampled), tuple(item.summary for item in sampled)


def _config(*, cap: int, warm_updates: int, max_wall_seconds: float) -> StagedTrainingConfig:
    active_response = ("value", "finite", "finite_peak", "pressure_value", "pressure_response")
    stages = (
        TrainingStage("full_access_value_warmup", 0, min(10, warm_updates), ("value",)),
        TrainingStage("full_access_response_ramp", min(10, warm_updates), warm_updates, active_response),
        TrainingStage("sparse_primary_capacity", warm_updates, cap, active_response),
    )
    # Normalize zero-width warm intervals when callers request a shorter gate.
    normalized = tuple(stage for stage in stages if stage.stop_update > stage.start_update)
    if not normalized:
        normalized = (TrainingStage("sparse_primary_capacity", 0, cap, active_response),)
    elif normalized[0].start_update != 0:
        normalized = (TrainingStage("sparse_primary_capacity", 0, cap, active_response),)
    elif normalized[-1].stop_update < cap:
        normalized = (*normalized, TrainingStage("sparse_primary_capacity", normalized[-1].stop_update, cap, active_response))
    ramp_start = min(10, warm_updates)
    ramp_end = max(ramp_start + 1, warm_updates)
    ramp_terms = ("finite", "finite_peak", "pressure_value", "pressure_response") if ramp_end > ramp_start else ()
    return StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=cap,
        max_epochs=max(cap, 1),
        total_optimizer_update_ceiling=cap,
        checkpoint_every_updates=10,
        max_wall_seconds=max_wall_seconds,
        review_updates=(cap,),
        random_seed=2317,
        deterministic_eval_mode=True,
        deterministic_algorithms=True,
        response_ramp_start_update=(ramp_start if ramp_terms else None),
        response_ramp_end_update=(ramp_end if ramp_terms else None),
        response_ramp_terms=ramp_terms,
        required_response_terms=("finite", "finite_peak", "pressure_value", "pressure_response"),
        include_feasibility_bce=False,
        stages=normalized,
    )


def _gradient_norm(module: nn.Module) -> float:
    terms = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(terms).sum()).detach().cpu()) if terms else 0.0


def _step_row(step: TrainingStep) -> dict[str, Any]:
    return {
        "completed_update": step.completed_update,
        "attempted_optimizer_step": step.attempted_optimizer_step,
        "training_stencil_index": step.training_stencil_index,
        "historical_case_id": step.historical_case_id,
        "stage": step.stage,
        "active_terms": list(step.active_terms),
        "total_loss": step.total_loss,
        "loss_terms": dict(step.term_losses),
        "active_term_weights": dict(step.active_term_weights),
    }


def _normalized_value_loss(prediction: Any, record: Any, scales: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    if record.output is None:
        raise ValueError("Thermal route pilot requires a solved train response record.")
    losses: list[torch.Tensor] = []
    per_role: dict[str, float] = {}
    for role_name, role in record.output.roles.items():
        value = weighted_masked_mse(
            prediction.role_values[role_name],
            role.values,
            valid_mask=role.valid_mask,
            quadrature_weights=role.quadrature_weights,
            scales=scales[role_name],
        )
        if value is not None:
            losses.append(value)
            per_role[role_name] = float(value.detach().cpu())
    if not losses:
        raise RuntimeError("Thermal route pilot found no supported absolute role loss.")
    return torch.stack(losses).mean(), per_role


def _physical_rms(prediction: torch.Tensor, target: Any, role: Any) -> float:
    value = weighted_masked_mse(
        prediction,
        target,
        valid_mask=role.valid_mask,
        quadrature_weights=role.quadrature_weights,
        scales=1.0,
    )
    return 0.0 if value is None else float(torch.sqrt(value.detach()).cpu())


def _pilot_plan_builder(route: str, permission: torch.Tensor):
    def build(encoded: Any, base_module_state: torch.Tensor, trees: Any):
        del base_module_state
        plans = []
        for case, tree in enumerate(trees):
            module_present = encoded.module_present[case]
            plan = MechanismPlan.full_access(
                tree,
                module_present,
                int(encoded.env_coords.shape[1]),
            )
            if route == "MM":
                pair_valid = (
                    (module_present > 0.5)[:, None]
                    & (module_present > 0.5)[None, :]
                    & ~torch.eye(module_present.numel(), device=module_present.device, dtype=torch.bool)
                )
            elif route == "ME":
                pair_valid = (module_present > 0.5)[:, None] & (encoded.env_weights[case] > 0.0)[None, :]
            else:
                raise ValueError(f"Unsupported Thermal omission-pilot route {route!r}.")
            weights = permission * pair_valid.to(permission.dtype)
            plan = plan.with_direct_pair_access(
                route,
                encoded.module_centers[case],
                weights,
                receiver_validity=module_present > 0.5,
            )
            plans.append(plan)
        return tuple(plans)
    return build


def _train_only_route_pilot(
    operator: DifferentiableThermalOperator,
    stencils: tuple[Any, ...],
    scales: Any,
    *,
    device: torch.device,
    output_path: Path,
    environment_count: int,
    environment_valid_count: int,
) -> dict[str, Any]:
    """Choose MM versus ME from train-only omission effect and restoration gradient."""

    indexed = sorted(
        range(len(stencils)),
        key=lambda index: sum(bool(module.active) for module in stencils[index].baseline.design.modules),
    )
    chosen_indices = tuple(dict.fromkeys((indexed[0], indexed[-1])))
    evidence: dict[str, Any] = {
        "split": "train",
        "case_families": [stencils[index].physical_family_id for index in chosen_indices],
        "route_results": {},
        "selection_rule": (
            "maximize mean one-sided standardized train-loss increase from route omission, "
            "then add positive per-eligible-pair negative restoration-gradient direction"
        ),
        "reference_solver_attempts": 0,
    }
    aggregate: dict[str, list[float]] = {"MM": [], "ME": []}
    for route in ("MM", "ME"):
        route_rows: list[dict[str, Any]] = []
        for index in chosen_indices:
            stencil = stencils[index]
            if stencil.split.value != "train":
                raise ValueError("MM/ME route pilot cannot use non-train families.")
            variant_label = "i_plus" if "i_plus" in stencil.variants else next(iter(stencil.variants))
            base_record = stencil.baseline
            trial_record = stencil.variants[variant_label]
            queries = role_queries_from_stencil(stencil, device=device)
            with torch.no_grad():
                full_base = operator(
                    DesignInput.from_state(base_record.design, device=device),
                    context_inputs(base_record.context),
                    queries,
                )
                full_trial = operator(
                    DesignInput.from_state(trial_record.design, device=device),
                    context_inputs(trial_record.context),
                    queries,
                )

            parameter_shape = (
                int(operator.max_modules),
                int(operator.max_modules) if route == "MM" else int(environment_count),
            )
            # Evaluate the local addition direction at the omitted endpoint.
            # This is a true restoration derivative, unlike a deletion
            # sensitivity measured at permission=1.
            permission = nn.Parameter(torch.zeros(parameter_shape, device=device, dtype=next(operator.model.parameters()).dtype))
            if route == "MM":
                with torch.no_grad():
                    permission.fill_diagonal_(0.0)
            omitted = torch.zeros_like(permission)
            if route == "MM":
                omitted.fill_diagonal_(0.0)
            zero_builder = _pilot_plan_builder(route, omitted)
            soft_builder = _pilot_plan_builder(route, permission)
            with torch.no_grad():
                omitted_base = operator(
                    DesignInput.from_state(base_record.design, device=device),
                    context_inputs(base_record.context),
                    queries,
                    cover_plan_builder=zero_builder,
                )
                omitted_trial = operator(
                    DesignInput.from_state(trial_record.design, device=device),
                    context_inputs(trial_record.context),
                    queries,
                    cover_plan_builder=zero_builder,
                )
            soft_base = operator(
                DesignInput.from_state(base_record.design, device=device),
                context_inputs(base_record.context),
                queries,
                cover_plan_builder=soft_builder,
                detach_model_parameters=True,
            )
            soft_trial = operator(
                DesignInput.from_state(trial_record.design, device=device),
                context_inputs(trial_record.context),
                queries,
                cover_plan_builder=soft_builder,
                detach_model_parameters=True,
            )
            hard_value, hard_by_role = _normalized_value_loss(omitted_base, base_record, scales.value)
            full_value, full_by_role = _normalized_value_loss(full_base, base_record, scales.value)
            value_straight = type(omitted_base)(
                role_values={
                    name: omitted_base.role_values[name] + soft_base.role_values[name] - soft_base.role_values[name].detach()
                    for name in omitted_base.role_values
                },
                receiver_world_xy=omitted_base.receiver_world_xy,
            )
            finite_hard_terms: list[torch.Tensor] = []
            finite_soft_terms: list[torch.Tensor] = []
            finite_full_terms: list[torch.Tensor] = []
            physical_omission_by_role: dict[str, float] = {}
            physical_full_error_by_role: dict[str, float] = {}
            physical_omission_error_by_role: dict[str, float] = {}
            assert base_record.output is not None and trial_record.output is not None
            for role_name, role in base_record.output.roles.items():
                block = stencil.finite_change(variant_label, role_name)
                hard_delta = omitted_trial.role_values[role_name] - omitted_base.role_values[role_name]
                full_delta = full_trial.role_values[role_name] - full_base.role_values[role_name]
                soft_delta = (
                    soft_trial.role_values[role_name] - soft_base.role_values[role_name]
                )
                straight_delta = hard_delta + (soft_delta - soft_delta.detach())
                hard_mse = weighted_masked_mse(
                    hard_delta,
                    block.delta,
                    valid_mask=block.valid_mask,
                    quadrature_weights=block.quadrature_weights,
                    scales=scales.finite[role_name],
                )
                full_mse = weighted_masked_mse(
                    full_delta,
                    block.delta,
                    valid_mask=block.valid_mask,
                    quadrature_weights=block.quadrature_weights,
                    scales=scales.finite[role_name],
                )
                soft_mse = weighted_masked_mse(
                    straight_delta,
                    block.delta,
                    valid_mask=block.valid_mask,
                    quadrature_weights=block.quadrature_weights,
                    scales=scales.finite[role_name],
                )
                if hard_mse is not None and soft_mse is not None and full_mse is not None:
                    finite_hard_terms.append(hard_mse)
                    finite_soft_terms.append(soft_mse)
                    finite_full_terms.append(full_mse)
                physical_full_error_by_role[role_name] = _physical_rms(
                    full_base.role_values[role_name], role.values, role
                )
                physical_omission_error_by_role[role_name] = _physical_rms(
                    omitted_base.role_values[role_name], role.values, role
                )
                physical_omission_by_role[role_name] = _physical_rms(
                    omitted_base.role_values[role_name], full_base.role_values[role_name], role
                )
            if not finite_hard_terms:
                raise RuntimeError("Route pilot had no finite-response loss support.")
            hard_loss = hard_value + torch.stack(finite_hard_terms).mean()
            full_loss = full_value + torch.stack(finite_full_terms).mean()
            soft_loss = _normalized_value_loss(value_straight, base_record, scales.value)[0] + torch.stack(finite_soft_terms).mean()
            gradient = torch.autograd.grad(soft_loss, permission, allow_unused=False)[0]
            active_modules = int(sum(bool(module.active) for module in base_record.design.modules))
            if route == "MM":
                valid_count = active_modules * max(active_modules - 1, 0)
            else:
                valid_count = active_modules * int(environment_valid_count)
            restoration_benefit_per_eligible_pair = -float(gradient.detach().sum().cpu()) / max(valid_count, 1)
            loss_excess = max(float((hard_loss - full_loss).detach().cpu()), 0.0)
            route_score = loss_excess + max(restoration_benefit_per_eligible_pair, 0.0)
            aggregate[route].append(route_score)
            route_rows.append({
                "family_id": stencil.physical_family_id,
                "module_count": int(sum(bool(module.active) for module in base_record.design.modules)),
                "variant": variant_label,
                "normalized_full_value_loss": float(full_value.detach().cpu()),
                "normalized_omission_value_loss": float(hard_value.detach().cpu()),
                "normalized_full_finite_loss": float(torch.stack(finite_full_terms).mean().detach().cpu()),
                "normalized_omission_finite_loss": float(torch.stack(finite_hard_terms).mean().detach().cpu()),
                "one_sided_loss_excess": loss_excess,
                "restoration_benefit_per_eligible_pair_at_omitted_endpoint": restoration_benefit_per_eligible_pair,
                "restoration_gradient_l2": float(torch.linalg.vector_norm(gradient.detach()).cpu()),
                "eligible_permission_count": valid_count,
                "permission_evaluation_point": "all_eligible_weights_zero",
                "full_reference_rmse_by_role_physical_units": physical_full_error_by_role,
                "omission_reference_rmse_by_role_physical_units": physical_omission_error_by_role,
                "full_to_omission_change_rmse_by_role_physical_units": physical_omission_by_role,
                "role_value_normalized_losses_full": full_by_role,
                "role_value_normalized_losses_omission": hard_by_role,
            })
            permission.grad = None
            del full_base, full_trial, omitted_base, omitted_trial, soft_base, soft_trial
        evidence["route_results"][route] = {
            "per_case": route_rows,
            "mean_selection_score": float(np.mean(aggregate[route])) if aggregate[route] else 0.0,
        }
    selected = max(("MM", "ME"), key=lambda route: evidence["route_results"][route]["mean_selection_score"])
    if not any(evidence["route_results"][route]["mean_selection_score"] > 0.0 for route in ("MM", "ME")):
        evidence["selection_status"] = "neither_route_had_positive_measured_restoration_evidence"
    else:
        evidence["selection_status"] = "training_only_omission_restoration_pilot_selected"
    evidence["selected_extra_route"] = selected
    _atomic_json(output_path, evidence)
    return evidence


def run(args: argparse.Namespace) -> dict[str, Any]:
    invocation_started = time.monotonic()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    atlas_dir = args.atlas_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    manifest_path = output_dir / "run_manifest.json"
    existing_manifest = None
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("--resume requires an existing Run1509 run_manifest.json.")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif output_dir.exists() and any(path.is_file() or path.is_symlink() for path in output_dir.rglob("*")):
        raise FileExistsError(f"Run directory is not empty; preserving existing work: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or device.index != 0:
            raise RuntimeError("Thermal GPU execution requires CUDA_VISIBLE_DEVICES=2 and logical cuda:0.")
        torch.cuda.set_device(device)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(2317)
    np.random.seed(2317)
    random.seed(2317)
    startup_payload = {
            "status": "started",
            "run_id": "Run_1509_20260928_active_packet_organization",
            "pid": os.getpid(),
            "started_at_unix": time.time(),
            "device": str(device),
            "physical_gpu": 2 if device.type == "cuda" else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint": str(checkpoint_path),
            "atlas_dir": str(atlas_dir),
            "output_dir": str(output_dir),
            "optimizer_update_cap_per_arm": args.cap,
            "arms": list(args.arms),
            "reference_solver_attempt_cap": 20,
            "query_batch_size": int(args.query_batch_size),
            "stop_after_update": args.stop_after_update,
        }
    if not (output_dir / "startup.json").is_file():
        _atomic_json(output_dir / "startup.json", startup_payload)
    _append_jsonl(output_dir / "run_invocations.jsonl", startup_payload)

    sampling = ReceiverSamplingConfig(
        max_fluid_queries=128,
        solid_queries_per_module=8,
        hot_solid_points_per_module=4,
        random_seed=2317,
    )
    raw_stencils, training_stencils, sampling_summaries = _selected_train_stencils(atlas_dir, sampling)
    source_model, checkpoint = load_model(checkpoint_path, device)
    if int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))) != 4738:
        raise ValueError("Run1509 requires the exact Run1804 e4738 initialization.")
    if str(source_model.config.core_honf.forward_architecture) != "dense_pairwise_field":
        raise ValueError("Run1509 requires the intact checkpoint-native Dense architecture.")
    source_model.eval()
    dataset_root = _resolve_dataset_path(checkpoint, None)
    dataset_sha256 = _sha256(dataset_root)
    train_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(train_dataset)
    setup_operator = DifferentiableThermalOperator(
        source_model,
        template,
        dataset_config=checkpoint["train_config"]["dataset"],
        normalization_stats=checkpoint["global_normalization_stats"],
        query_batch_size=args.query_batch_size,
        capture_packet_inputs=True,
    )
    setup_stencil = training_stencils[0]
    setup_queries = role_queries_from_stencil(setup_stencil, device=device)
    with torch.no_grad():
        setup_operator(
            DesignInput.from_state(setup_stencil.baseline.design, device=device),
            context_inputs(setup_stencil.baseline.context),
            setup_queries,
        )
    packet_inputs = setup_operator.last_packet_inputs
    if not isinstance(packet_inputs, Mapping) or packet_inputs.get("encoded") is None:
        raise RuntimeError("Native Dense setup did not expose its input-only encoded packet panel.")
    encoded = packet_inputs["encoded"]
    if int(encoded.module_present.shape[0]) != 1:
        raise ValueError("The Thermal runner expects one physical case per native forward call.")
    historical_sampling = ReceiverSamplingConfig(
        max_fluid_queries=3072,
        solid_queries_per_module=128,
        hot_solid_points_per_module=16,
        random_seed=2317,
    )
    historical_source = HistoricalValueSource.from_dataset(train_dataset, sampling=historical_sampling)
    scales = derive_training_scales(
        training_stencils,
        smooth_peak_beta=args.smooth_peak_beta,
        historical_value_source=historical_source,
    )
    atlas_hashes = {
        path.name: _sha256(path)
        for path in sorted(atlas_dir.glob("train_*_responses.npz"))
        if path.stem.split("_")[1] in TRAIN_ATLAS_IDS
    }
    checkpoint_sha = _sha256(checkpoint_path)
    route_pilot_path = output_dir / "route_pilot.json"
    if args.resume:
        if not route_pilot_path.is_file():
            raise FileNotFoundError("Cannot resume Run1509 without its frozen training-only MM/ME pilot.")
        route_pilot = json.loads(route_pilot_path.read_text(encoding="utf-8"))
        if route_pilot.get("source_checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Saved route pilot belongs to a different source checkpoint.")
        if route_pilot.get("train_atlas_hashes") != atlas_hashes:
            raise ValueError("Saved route pilot belongs to different training atlases.")
        if route_pilot.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Saved route pilot belongs to a different packed dataset.")
    else:
        route_pilot = _train_only_route_pilot(
            setup_operator,
            training_stencils,
            scales,
            device=device,
            output_path=route_pilot_path,
            environment_count=int(encoded.env_coords.shape[1]),
            environment_valid_count=int(torch.count_nonzero(encoded.env_weights[0] > 0.0)),
        )
        route_pilot["source_checkpoint_sha256"] = checkpoint_sha
        route_pilot["train_atlas_hashes"] = atlas_hashes
        route_pilot["dataset_sha256"] = dataset_sha256
        _atomic_json(route_pilot_path, route_pilot)
    selected_extra_route = str(route_pilot["selected_extra_route"])
    if args.pilot_only:
        result = {
            "status": "pilot_only_complete",
            "run_id": "Run_1509_20260928_active_packet_organization",
            "wall_seconds": time.monotonic() - invocation_started,
            "route_pilot": route_pilot,
        }
        _atomic_json(output_dir / "pilot_only_complete.json", result)
        return result
    response_config = _config(
        cap=args.cap,
        warm_updates=args.warm_updates,
        max_wall_seconds=args.max_wall_seconds,
    )
    fresh_manifest: dict[str, Any] = {
        "status": "running",
        "run_id": "Run_1509_20260928_active_packet_organization",
        "started_at_unix": time.time(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "selected_epoch": int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))),
        "dataset_root": str(dataset_root),
        "dataset_sha256": dataset_sha256,
        "dataset_size_bytes": int(dataset_root.stat().st_size),
        "dataset_source": "local analytic_wake flow plus shared-grid thermal generator",
        "device": str(device),
        "physical_gpu": 2 if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "query_batch_size": int(args.query_batch_size),
        "deterministic_algorithms": bool(args.deterministic),
        "reference_solver_attempts": 0,
        "resource_caps": {
            "updates_per_arm": args.cap,
            "attempted_optimizer_steps_per_arm": 600,
            "physical_solver_attempts": 20,
        },
        "stage_a": {
            "warm_full_updates": args.warm_updates,
            "sparse_budget": {"QE": 0.90, selected_extra_route: 0.90},
            "sparse_frontier_sampling": {
                "method": "seeded weighted random selection over valid depth-three cuts",
                "root_per_cut_weight": 4.0,
                "other_per_cut_weight": 1.0,
                "forced_K_histogram": False,
            },
            "selected_sparse_extra_route": selected_extra_route,
            "route_pilot": route_pilot,
            "train_families": [item.physical_family_id for item in raw_stencils],
            "atlas_hashes": atlas_hashes,
            "loss_weights": LOSS_WEIGHTS,
            "loss_scales": {
                "value": dict(scales.value),
                "finite": dict(scales.finite),
                "finite_peak": scales.solid_temperature,
                "pressure_value": scales.pressure_value,
                "pressure_response": scales.pressure_response,
            },
            "sampling": [
                {
                    "physical_family_id": item.physical_family_id,
                    "split": item.split,
                    "original_counts": dict(item.original_counts),
                    "sampled_counts": dict(item.sampled_counts),
                    "protected_counts": dict(item.protected_counts),
                    "solid_peak_query_coverage": dict(item.solid_peak_query_coverage),
                    "inverse_probability_weighting": item.inverse_probability_weighting,
                }
                for item in sampling_summaries
            ],
        },
        "arms": {},
    }
    if args.resume:
        assert existing_manifest is not None
        if (
            existing_manifest.get("checkpoint_sha256") != checkpoint_sha
            or existing_manifest.get("dataset_sha256") != dataset_sha256
            or existing_manifest.get("stage_a", {}).get("atlas_hashes") != atlas_hashes
            or existing_manifest.get("stage_a", {}).get("selected_sparse_extra_route") != selected_extra_route
        ):
            raise ValueError("Existing Run1509 manifest does not match checkpoint, train atlases, or frozen route pilot.")
        manifest = existing_manifest
        manifest["status"] = "resuming"
        manifest.setdefault("resume_invocations", []).append(startup_payload)
        manifest.setdefault("query_batch_size_history", []).append(
            {
                "query_batch_size": int(args.query_batch_size),
                    "timestamp_unix": startup_payload["started_at_unix"],
            }
        )
        manifest.setdefault("resource_caps", {})["updates_per_arm"] = args.cap
        if args.cap >= 300:
            manifest["stage_a"]["continuation_curriculum"] = {
                "updates_1_to_50": "exact full-access warmup",
                "updates_51_to_200": "primary QE plus selected route at 0.90",
                "updates_201_to_300": "seeded shuffled five-update blocks: one full, three 0.90, one 0.75",
                "post_200_budget_fraction_counts": {"1.00": 20, "0.90": 60, "0.75": 20},
                "same_stream_for_G_and_P": True,
                "frontier_K_histogram_forced": False,
                "optimizer_rng_and_model_state_resumed_from_u100": True,
            }
    else:
        manifest = fresh_manifest
    _atomic_json(output_dir / "run_manifest.json", manifest)

    for arm in args.arms:
        arm_summary_existing = manifest.get("arms", {}).get(arm, {})
        if (
            args.resume
            and arm_summary_existing.get("status") == f"passed_u{args.cap}"
            and int(arm_summary_existing.get("updates", -1)) == args.cap
            and Path(str(arm_summary_existing.get("checkpoint", ""))).is_file()
        ):
            continue
        torch.manual_seed(2317)
        np.random.seed(2317)
        random.seed(2317)
        forward_model, arm_checkpoint = load_model(checkpoint_path, device)
        if _sha256(checkpoint_path) != manifest["checkpoint_sha256"] or int(
            arm_checkpoint.get("epoch", -1)
        ) != 4738:
            raise RuntimeError("Thermal arm did not reload the exact shared e4738 source checkpoint.")
        forward_model.eval()
        scope = _configure_native_expanded_response_interface_scope(forward_model)
        route_model = _route_module(arm, forward_model.core, encoded, device)
        schedule = _FrontierSchedule(
            warm_updates=args.warm_updates,
            seed=2317,
            extra_route=selected_extra_route,
            extended_curriculum=args.cap >= 300,
        )
        hard_builder, soft_builder = _route_builder(
            arm,
            core=forward_model.core,
            route_model=route_model,
            extra_route=selected_extra_route,
            schedule=schedule,
        )
        dtt = DifferentiableThermalOperator(
            forward_model,
            template,
            dataset_config=arm_checkpoint["train_config"]["dataset"],
            normalization_stats=arm_checkpoint["global_normalization_stats"],
            query_batch_size=args.query_batch_size,
        )
        paired = ThermalHardValueSoftOperator(dtt, hard_builder, soft_builder)
        resume_update = 0
        arm_dir = output_dir / arm
        arm_dir.mkdir(exist_ok=True)
        schedule.current_update = args.warm_updates
        hard_builder.budget_fractions = schedule.budgets()
        soft_builder.budget_fractions = schedule.budgets()
        smoke_routes = {mechanism: False for mechanism in schedule.budgets()}
        smoke_operator = _ScheduledThermalOperator(
            paired=paired,
            schedule=schedule,
            hard_builder=hard_builder,
            soft_builder=soft_builder,
            state_labels=("baseline", *tuple(training_stencils[0].variants)),
            arm=arm,
            route_log=arm_dir / "gpu_native_smoke_route_work.jsonl",
            sparse_route_verified=smoke_routes,
        )
        smoke_predictions = predict_stencil(smoke_operator, training_stencils[0], device=device)
        smoke_losses = compute_stencil_loss_terms(
            smoke_predictions,
            training_stencils[0],
            scales=scales,
            mixed_specs=(),
            enabled_terms=("value", "finite", "finite_peak", "pressure_value", "pressure_response"),
            include_feasibility_bce=False,
        )
        smoke_total = smoke_losses.total(LOSS_WEIGHTS)
        smoke_total.backward()
        smoke_physical_gradient = _gradient_norm(forward_model)
        smoke_route_gradient = _gradient_norm(route_model)
        if not all(smoke_routes.values()) or smoke_route_gradient <= 0.0 or smoke_physical_gradient <= 0.0:
            raise RuntimeError(
                f"GPU {arm} native smoke did not show active sparse permissions and both gradient paths: "
                f"routes={smoke_routes}, physical_grad={smoke_physical_gradient}, "
                f"route_grad={smoke_route_gradient}."
            )
        smoke_record = {
            "status": "passed",
            "arm": arm,
            "physical_gpu": 2 if device.type == "cuda" else None,
            "logical_device": str(device),
            "optimizer_updates": 0,
            "reference_solver_attempts": 0,
            "case_family": training_stencils[0].physical_family_id,
            "budget": schedule.budgets(),
            "route_support_verified": smoke_routes,
            "standardized_loss_terms": {name: float(value.detach().cpu()) for name, value in smoke_losses.terms.items()},
            "total_standardized_loss": float(smoke_total.detach().cpu()),
            "physical_gradient_l2": smoke_physical_gradient,
            "route_gradient_l2": smoke_route_gradient,
            "role_shapes": {name: list(value.shape) for name, value in paired.last_hard.role_values.items()},
        }
        _atomic_json(arm_dir / "gpu_native_smoke.json", smoke_record)
        if args.smoke_only:
            manifest["arms"][arm] = {
                "status": "native_smoke_passed",
                "gpu_native_smoke": smoke_record,
            }
            _atomic_json(output_dir / "run_manifest.json", manifest)
            continue
        bundle = _TrainingBundle(forward_model, route_model)
        bundle.zero_grad(set_to_none=True)
        schedule.current_update = resume_update
        hard_builder.budget_fractions = schedule.budgets()
        soft_builder.budget_fractions = schedule.budgets()
        sparse_verified = {mechanism: False for mechanism in schedule.budgets()}
        route_log = output_dir / f"{arm}_route_work.jsonl"
        scheduled = _ScheduledThermalOperator(
            paired=paired,
            schedule=schedule,
            hard_builder=hard_builder,
            soft_builder=soft_builder,
            state_labels=("baseline", *tuple(training_stencils[0].variants)),
            arm=arm,
            route_log=route_log,
            sparse_route_verified=sparse_verified,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in bundle.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        resume_payload = None
        resume_update = 0
        if args.resume:
            checkpoint_candidates = sorted(
                (output_dir / "checkpoints").glob(f"{arm}_u*_training_checkpoint.pt"),
                key=lambda path: int(path.stem.split("_u", 1)[1].split("_", 1)[0]),
            )
            if checkpoint_candidates:
                resume_payload = torch.load(
                    checkpoint_candidates[-1], map_location="cpu", weights_only=False
                )
                resume_update = int(resume_payload.get("actual_optimizer_updates", -1))
                if not 0 <= resume_update <= args.cap:
                    raise ValueError(f"{arm} resume checkpoint update is outside the requested cap.")
                attempt_ledger = output_dir / "optimizer_attempts.jsonl"
                arm_attempts = []
                if attempt_ledger.is_file():
                    with attempt_ledger.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            row = json.loads(line)
                            if row.get("arm") == arm:
                                arm_attempts.append(int(row["global_attempted_step"]))
                if arm_attempts:
                    resume_payload["attempted_optimizer_steps"] = max(
                        int(resume_payload.get("attempted_optimizer_steps", 0)),
                        max(arm_attempts),
                    )
                if int(resume_payload.get("attempted_optimizer_steps", -1)) < resume_update:
                    raise ValueError("Resume attempt ledger is behind completed optimizer updates.")
            if resume_update == args.cap:
                raise RuntimeError(
                    f"{arm} reached the update cap in a resumable checkpoint but did not record a complete arm summary."
                )
        attempt_base = int(resume_payload.get("attempted_optimizer_steps", 0)) if resume_payload else 0
        schedule.current_update = resume_update
        hard_builder.budget_fractions = schedule.budgets()
        soft_builder.budget_fractions = schedule.budgets()
        metric_path = arm_dir / "training_steps.jsonl"
        gradient_path = arm_dir / "gradient_steps.jsonl"
        checkpoint_dir = output_dir / "checkpoints"
        arm_started = time.monotonic()
        first_sparse_grad = {"recorded": False}

        def save_checkpoint(payload: Mapping[str, Any], label: str, *, current_arm: str = arm) -> None:
            update = int(payload["actual_optimizer_updates"])
            path = checkpoint_dir / f"{current_arm}_u{update:04d}_{label}.pt"
            _atomic_torch_save(path, dict(payload))
            manifest["arms"].setdefault(current_arm, {})["latest_checkpoint"] = str(path)
            manifest["arms"][current_arm]["latest_update"] = update
            _atomic_json(output_dir / "run_manifest.json", manifest)

        def on_attempt(completed: int, attempted_total: int, *, current_arm: str = arm) -> None:
            physical_grad = _gradient_norm(bundle.forward_model)
            route_grad = _gradient_norm(bundle.route_model)
            if schedule.current_update >= args.warm_updates and not first_sparse_grad["recorded"]:
                if route_grad <= 0.0:
                    raise RuntimeError(f"{current_arm} route learner received zero gradient at first sparse update.")
                first_sparse_grad["recorded"] = True
            _append_jsonl(
                gradient_path,
                {
                    "arm": current_arm,
                    "optimizer_update": int(completed) + 1,
                    "attempted_total": int(attempted_total),
                    "current_budget": schedule.budgets(),
                    "physical_gradient_l2": physical_grad,
                    "route_gradient_l2": route_grad,
                    "route_grad_nonzero": route_grad > 0.0,
                },
            )
            _append_jsonl(
                output_dir / "optimizer_attempts.jsonl",
                {
                    "arm": current_arm,
                    "completed_updates_before_attempt": int(completed),
                    "global_attempted_step": int(attempted_total),
                    "next_budget": schedule.budgets(),
                    "recorded_before_optimizer_step": True,
                    "timestamp_unix": time.time(),
                },
            )
            scheduled.optimizer_attempt(completed, attempted_total)

        def on_step(step: TrainingStep, *, current_arm: str = arm) -> None:
            row = {
                "arm": current_arm,
                **_step_row(step),
                "cumulative_attempted_optimizer_step": attempt_base + step.attempted_optimizer_step,
            }
            _append_jsonl(metric_path, row)
            if step.completed_update in {1, args.warm_updates, args.warm_updates + 1, args.cap}:
                print(json.dumps({
                    "arm": current_arm,
                    "update": step.completed_update,
                    "loss": step.total_loss,
                    "terms": dict(step.term_losses),
                    "seconds": time.monotonic() - arm_started,
                }, sort_keys=True), flush=True)

        mixed = tuple(
            spec
            for stencil in training_stencils
            for spec in _mixed_specs(stencil)
        )
        stop_at_update = args.stop_after_update if args.stop_after_update is not None else args.cap
        if stop_at_update > args.cap:
            raise ValueError("--stop-after-update cannot exceed the requested per-arm cap.")
        if stop_at_update <= resume_update:
            raise ValueError("--stop-after-update must be later than the current resumable checkpoint.")
        fit = run_staged_fit(
            scheduled,
            bundle,
            optimizer,
            training_stencils,
            historical_value_source=historical_source,
            scales=scales,
            loss_weights=LOSS_WEIGHTS,
            mixed_specs=mixed,
            config=response_config,
            initial_update=resume_update,
            resume_payload=resume_payload,
            stop_at_update=stop_at_update,
            device=device,
            on_checkpoint=save_checkpoint,
            on_optimizer_attempt=on_attempt,
            on_step=on_step,
        )
        if fit.final_update != args.cap and args.stop_after_update is not None:
            if fit.final_update != args.stop_after_update:
                raise TimeoutError(
                    f"{arm} stopped at completed update {fit.final_update}, before requested gate "
                    f"{args.stop_after_update}."
                )
            paused_checkpoint = manifest["arms"].get(arm, {}).get("latest_checkpoint")
            if not paused_checkpoint or not Path(paused_checkpoint).is_file():
                raise RuntimeError(f"{arm} update {fit.final_update} has no durable training checkpoint.")
            manifest["arms"][arm] = {
                "status": f"paused_at_u{fit.final_update}",
                "checkpoint": paused_checkpoint,
                "updates": fit.final_update,
                "attempted_optimizer_steps": fit.total_attempted_optimizer_steps,
                "attempted_updates_this_invocation": fit.attempted_optimizer_steps,
                "wall_seconds": time.monotonic() - arm_started,
                "first_sparse_route_gradient_nonzero": first_sparse_grad["recorded"],
                "sparse_support_verified": sparse_verified,
                "budgeted_route_status_counts": _route_status_counts(route_log),
                "gpu_native_smoke": smoke_record,
                "last_losses": _step_row(fit.history[-1]) if fit.history else None,
                "route_work_ledger": str(route_log),
                "training_loss_ledger": str(metric_path),
                "gradient_ledger": str(gradient_path),
            }
            manifest["status"] = f"paused_after_{arm}_u{fit.final_update}"
            manifest["last_pause_unix"] = time.time()
            _atomic_json(output_dir / "run_manifest.json", manifest)
            return {
                "status": manifest["status"],
                "run_id": manifest["run_id"],
                "wall_seconds": time.monotonic() - invocation_started,
                "route_pilot": route_pilot,
            }
        if fit.final_update != args.cap:
            raise TimeoutError(
                f"{arm} stopped at completed update {fit.final_update} (cap {args.cap})."
            )
        if fit.total_attempted_optimizer_steps > 600:
            raise RuntimeError(
                f"{arm} has {fit.total_attempted_optimizer_steps} cumulative attempts, above the 600-arm limit."
            )
        if args.warm_updates < args.cap and not all(sparse_verified.values()):
            raise RuntimeError(f"{arm} did not verify both nonzero sparse routes: {sparse_verified}.")
        state_path = output_dir / "checkpoints" / f"{arm}_u{args.cap:04d}_model.pt"
        _atomic_torch_save(state_path, {
            "run_id": manifest["run_id"],
            "arm": arm,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": manifest["checkpoint_sha256"],
            "actual_optimizer_updates": fit.final_update,
            "attempted_optimizer_steps": fit.total_attempted_optimizer_steps,
            "attempted_updates_this_invocation": fit.attempted_optimizer_steps,
            "model_state": bundle.state_dict(),
            "physical_model_state": bundle.forward_model.state_dict(),
            "route_model_state": bundle.route_model.state_dict(),
            "trainable_scope": scope,
            "budget_schedule": {
                "warm_full_updates": args.warm_updates,
                "primary_updates_51_to_200": 0.90,
                "post_200_mix": {"full": 0.20, "primary_0.90": 0.60, "stress_0.75": 0.20}
                if args.cap >= 300
                else None,
                "frontier_K_histogram_forced": False,
            },
            "train_family_ids": manifest["stage_a"]["train_families"],
        })
        arm_summary = {
            "status": f"passed_u{args.cap}",
            "checkpoint": str(state_path),
            "updates": fit.final_update,
            "attempted_optimizer_steps": fit.total_attempted_optimizer_steps,
            "attempted_updates_this_invocation": fit.attempted_optimizer_steps,
            "wall_seconds": time.monotonic() - arm_started,
            "trainable_scope": scope,
            "first_sparse_route_gradient_nonzero": first_sparse_grad["recorded"],
            "sparse_support_verified": sparse_verified,
            "budgeted_route_status_counts": _route_status_counts(route_log),
            "gpu_native_smoke": smoke_record,
            "last_losses": _step_row(fit.history[-1]) if fit.history else None,
            "route_work_ledger": str(route_log),
            "training_loss_ledger": str(metric_path),
            "gradient_ledger": str(gradient_path),
        }
        manifest["arms"][arm] = arm_summary
        _atomic_json(output_dir / "run_manifest.json", manifest)

    if args.smoke_only:
        manifest["status"] = "native_smoke_passed_both_arms"
        manifest["finished_at_unix"] = time.time()
        manifest["wall_seconds"] = time.time() - manifest["started_at_unix"]
        _atomic_json(output_dir / "run_manifest.json", manifest)
        return manifest
    all_arms_complete = all(
        manifest.get("arms", {}).get(arm, {}).get("status") == f"passed_u{args.cap}"
        and int(manifest.get("arms", {}).get(arm, {}).get("updates", -1)) == args.cap
        and Path(str(manifest.get("arms", {}).get(arm, {}).get("checkpoint", ""))).is_file()
        for arm in ("G", "P")
    )
    if not all_arms_complete:
        completed_now = [
            arm for arm in args.arms
            if manifest.get("arms", {}).get(arm, {}).get("status") == f"passed_u{args.cap}"
        ]
        manifest["status"] = f"partial_u{args.cap}_completed_{'_'.join(completed_now) or 'none'}"
        manifest["last_partial_unix"] = time.time()
        _atomic_json(output_dir / "run_manifest.json", manifest)
        return manifest

    frontier_histograms: dict[str, dict[str, int]] = {}
    route_status_histograms: dict[str, dict[str, dict[str, int]]] = {}
    for arm in ("G", "P"):
        route_path = output_dir / f"{arm}_route_work.jsonl"
        counts: Counter[str] = Counter()
        if route_path.is_file():
            with route_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("frontier") is not None:
                        counts[str(len(row["frontier"]))] += 1
        frontier_histograms[arm] = dict(sorted(counts.items(), key=lambda item: int(item[0])))
        route_status_histograms[arm] = _route_status_counts(route_path)
    manifest["stage_a"]["realized_frontier_K_observations_by_arm"] = frontier_histograms
    manifest["stage_a"]["budgeted_route_status_counts_by_arm"] = route_status_histograms
    manifest["status"] = f"passed_u{args.cap}_both_arms"
    manifest["finished_at_unix"] = time.time()
    manifest["wall_seconds"] = time.time() - manifest["started_at_unix"]
    _atomic_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cap", type=int, choices=(100, 300), default=100)
    parser.add_argument("--arms", nargs="+", choices=("G", "P"), default=("G", "P"),
                        help="Arms to advance in this invocation; exact checkpoints allow staged G/P continuation.")
    parser.add_argument("--warm-updates", type=int, default=50)
    parser.add_argument("--max-wall-seconds", type=float, default=5400.0)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument(
        "--stop-after-update",
        type=int,
        default=None,
        help="Pause after a specified absolute optimizer update and save an exact resumable checkpoint.",
    )
    parser.add_argument("--smooth-peak-beta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--pilot-only", action="store_true", help="Run the train-only MM/ME route pilot and stop before optimizer updates.")
    parser.add_argument("--smoke-only", action="store_true", help="Run the native sparse paired G/P path checks and stop before optimizer updates.")
    parser.add_argument("--resume", action="store_true", help="Resume the same Run1509 arm from its latest atomic staged-fit checkpoint.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output = run(args)
    except Exception as exc:
        failure_path = args.output_dir.expanduser().resolve() / "failure.json"
        _atomic_json(
            failure_path,
            {
                "status": "failed",
                "pid": os.getpid(),
                "failed_at_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "optimizer_attempt_ledger_exists": (
                    args.output_dir.expanduser().resolve() / "optimizer_attempts.jsonl"
                ).is_file(),
            },
        )
        raise
    print(json.dumps({
        "status": output["status"],
        "run_id": output["run_id"],
        "wall_seconds": output.get("wall_seconds"),
        "selected_extra_route": output.get("route_pilot", {}).get("selected_extra_route"),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the bounded launch
    raise SystemExit(main())
