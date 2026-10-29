"""Maintained native Thermal task for paired R13-5 full/joint95 readers."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from honf_runtime.unified_training import LossTerm, OptimizerGroupSpec, SamplingKey, ScheduleSpec
from ..data.datasets import fit_global_normalizer
from ..data.development_split import read_case_catalog
from .content_hypergraph_task import THERMAL_CONTENT_LOSS_WEIGHTS, ThermalContentPredictions
from .joint_task import _load_data_binding, _read_selected_cases
from .member_routing_task import (
    ThermalInteractionReusedMemberRoutingTask,
    _training_membership,
)


WORKFLOW = "interaction_reused_joint_member_routing_joint_training_v1"
PROVIDER_SCHEMA = "thermal_joint_member_routing_provider_v1"
CONFIG_SECTION_SCHEMA = "thermal_member_routing_r13_5_v1"
TRAINING_STATE_SCHEMA = "thermal_joint_member_routing_native_training_state_v1"
ARMS = ("R13-5-M-Joint95", "R13-5-ME-Joint95")
SOURCE_GUIDE_ARM = "R13-4-Guide"
JOINT_BLOCKS = ("H_MM_applied", "H_EM_applied", "H_EE_applied", "H_ME_applied")


@dataclass
class ThermalJointPairPredictions:
    """Same-parameter native predictions from all-eligible and joint95 reads."""

    full: ThermalContentPredictions
    restricted: ThermalContentPredictions
    work: Mapping[str, int | float]


def _finite_scalar(value: Any) -> float:
    result = float(value.detach().double().cpu()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError("R13-5 native health observed a nonfinite value.")
    return result


class JointMemberRoutingHealth:
    """Fixed TRAIN observation of paired fields, four applied blocks, and joint95."""

    STATE_SCHEMA = "thermal_joint_member_routing_health_state_v1"

    def __init__(self, provider, config: Mapping[str, Any]):
        if not isinstance(config, Mapping) or config.get("enabled") is not True:
            raise ValueError("Enabled R13-5 health requires enabled=true.")
        self.provider = provider
        self.observation_epochs = int(config.get("observation_epochs", 12))
        self.later_every = int(config.get("later_every", 10))
        self.pause_on_collapse = bool(config.get("pause_on_collapse", True))
        self.panel_module_counts = tuple(int(value) for value in config.get("panel_module_counts", (1, 2, 5, 12)))
        if (self.observation_epochs < 1 or self.later_every < 1 or not self.panel_module_counts
                or len(set(self.panel_module_counts)) != len(self.panel_module_counts)):
            raise ValueError("R13-5 health cadence and module-count panel are invalid.")
        from ..data.development_split import development_case_ids

        fixed_train = set(map(str, development_case_ids(provider.manifest, "train")))
        admitted = {str(case["case_id"]): case for case in provider.training_cases}
        by_count: dict[int, list[str]] = {count: [] for count in self.panel_module_counts}
        for case_id in fixed_train & set(admitted):
            present = np.asarray(admitted[case_id]["structure"]["module_present"]) > 0.5
            count = int(present.sum())
            if count in by_count:
                by_count[count].append(case_id)
        missing = [count for count, case_ids in by_count.items() if not case_ids]
        if missing:
            raise ValueError(f"R13-5 health panel lacks fixed TRAIN cases at module counts {missing}.")
        self.panel_case_ids = tuple(min(by_count[count]) for count in self.panel_module_counts)
        self.state = {
            "schema": self.STATE_SCHEMA,
            "panel_case_ids": list(self.panel_case_ids),
            "initial_observation": None,
            "last_observation": None,
            "last_observed_epoch": 0,
        }

    @property
    def initial_observation(self):
        return self.state.get("initial_observation")

    def scheduled(self, epoch: int) -> bool:
        epoch = int(epoch)
        initial = self.initial_observation
        origin = int(initial.get("training_epoch", 0)) if isinstance(initial, Mapping) else 0
        relative_epoch = epoch - origin
        if relative_epoch <= 0:
            return False
        if relative_epoch <= self.observation_epochs:
            return True
        return (relative_epoch - self.observation_epochs) % self.later_every == 0

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        if state is None:
            return
        if (not isinstance(state, Mapping) or state.get("schema") != self.STATE_SCHEMA
                or list(state.get("panel_case_ids", ())) != list(self.panel_case_ids)):
            raise ValueError("Saved R13-5 health state differs from its fixed TRAIN panel.")
        self.state = copy.deepcopy(dict(state))

    def record_initial(self, observation):
        self.state["initial_observation"] = copy.deepcopy(observation)
        self.state["last_observation"] = copy.deepcopy(observation)
        self.state["last_observed_epoch"] = int(observation["training_epoch"])

    def record_epoch(self, observation):
        self.state["last_observation"] = copy.deepcopy(observation)
        self.state["last_observed_epoch"] = int(observation["training_epoch"])
        return self.pause_reason(self.initial_observation, observation)

    @staticmethod
    def pause_reason(initial, current):
        """Stop only after previously resolved native heat dependence vanishes panel-wide."""
        if not isinstance(initial, Mapping):
            return None
        before = {int(case["module_count"]): case for case in initial.get("cases", ())}
        now = {int(case["module_count"]): case for case in current.get("cases", ())}
        if not before or not now:
            return None
        counts = (min(before), max(before))
        if any(count not in before or count not in now for count in counts):
            return None
        temperature_roles = ("temperature/fluid", "temperature/surface", "temperature/material")

        def resolved(case, mode):
            sensitivity = case["modes"][mode]["heat_plus10_fieldwise_native_delta"]
            repeatability = case["modes"][mode]["repeatability_fieldwise_native_delta"]
            return all(role in sensitivity and role in repeatability
                       and float(sensitivity[role]) > float(repeatability[role])
                       for role in temperature_roles)

        def unresolved(case, mode):
            sensitivity = case["modes"][mode]["heat_plus10_fieldwise_native_delta"]
            repeatability = case["modes"][mode]["repeatability_fieldwise_native_delta"]
            return all(role in sensitivity and role in repeatability
                       and float(sensitivity[role]) <= float(repeatability[role])
                       for role in temperature_roles)

        if all(resolved(before[count], mode) and unresolved(now[count], mode)
               for count in counts for mode in ("full", "restricted")):
            return {
                "kind": "native_temperature_heat_response_became_unresolved",
                "panel_module_counts": list(counts),
                "comparison": "initially resolved native temperatures versus each case and mode's same-input repeatability",
            }
        return None

    @staticmethod
    def _block_payload(prepared) -> dict[str, Any]:
        organization = prepared.organization
        provenance = getattr(organization, "H_route_provenance", {})
        stage_rows = {}
        if isinstance(provenance, Mapping):
            for block, row in provenance.items():
                if isinstance(row, Mapping):
                    stage_rows[str(block)] = {
                        key: str(row.get(key)) for key in
                        ("stage_identity", "route", "receiver_role", "donor_role")
                        if row.get(key) is not None
                    }
        return {
            name: getattr(organization, name, None) for name in JOINT_BLOCKS
        } | {
            "H_exchange": getattr(organization, "H_exchange", None),
            "H_self_M": getattr(organization, "H_self_M", None),
            "H_self": getattr(organization, "H_self", None),
            "H": getattr(organization, "H", None),
            "provenance": stage_rows,
        }

    @staticmethod
    def _native_with_first_eight_trace(model, prepared):
        """Decode the complete native query union while tracing its first eight rows."""
        from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1

        query = prepared.query_batch
        query_count = int(query.num_queries)
        if query_count < 1:
            raise ValueError("R13-5 health requires a nonempty prepared query union.")
        tile_size = int(model.core.query_tile_size)
        if tile_size < 1:
            raise ValueError("R13-5 health requires a positive core query tile size.")
        trace_rows = min(8, query_count)
        fields, counts, first_trace = [], {}, None
        start = 0
        while start < query_count:
            stop = min(start + (trace_rows if start == 0 else tile_size), query_count)
            tile = slice_query_v1(query, start, stop)
            output = model.core.decode(prepared.organization, tile, return_trace=(start == 0))
            fields.append(output.joint_fields)
            for name, value in output.work_counts.items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
            if start == 0:
                first_trace = output.trace
            start = stop
        if not isinstance(first_trace, Mapping):
            raise RuntimeError("R13-5 native health did not capture its bounded actual decode trace.")
        joint_fields = torch.cat(fields, dim=1)
        native = model._extract_native_roles(
            prepared, joint_fields[..., 3:4], joint_fields[..., :3],
        )
        return native, counts, first_trace

    @staticmethod
    def _route_health_inputs(model, prepared, block_values):
        route_by_block = {
            "H_MM_applied": "mm", "H_EM_applied": "em",
            "H_EE_applied": "ee", "H_ME_applied": "me",
        }
        stages = getattr(prepared.organization, "final_applied_context_stages", {})
        incidence = getattr(model.core, "applied_incidence", None)
        route_modules = getattr(incidence, "routes", {})
        allowed_by_block = {}
        route_parameter_refs = {}
        for block, route in route_by_block.items():
            stage = stages.get(route) if isinstance(stages, Mapping) else None
            snapshot = stage.get("snapshot") if isinstance(stage, Mapping) else None
            allowed = getattr(snapshot, "allowed", None)
            value = block_values.get(block)
            if not torch.is_tensor(value) or not torch.is_tensor(allowed):
                raise RuntimeError(f"R13-5 health lacks the {route.upper()} applied gate or eligibility mask.")
            try:
                allowed = allowed.to(device=value.device, dtype=torch.bool).expand_as(value)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"R13-5 health {route.upper()} eligibility axes differ from the applied gate."
                ) from exc
            allowed_by_block[block] = allowed
            module = route_modules[route] if route in route_modules else None
            if module is None:
                raise RuntimeError(f"R13-5 health lacks the live {route.upper()} route calibration.")
            route_parameter_refs[block] = tuple(
                (name, parameter) for name, parameter in module.named_parameters()
                if parameter.requires_grad
            )
        return route_by_block, allowed_by_block, route_parameter_refs

    @staticmethod
    def _gradient_norm(gradient):
        return 0.0 if gradient is None else _finite_scalar(gradient.detach().double().norm())

    def _observe_case(self, model, case_id: str) -> dict[str, Any]:
        provider = self.provider
        if int(provider.primary_queries) != 1024:
            raise ValueError("R13-5 health uses the maintained complete provider-sampled Q=1024 TRAIN objective.")
        key = SamplingKey(
            seed=int(provider.seed), epoch=1, update_index=0, microbatch_index=0,
            stage="joint", arm=str(provider.training_sampling_arm),
            sampling_version=provider.sampling_version,
            dataset_id=provider.sampling_dataset_id,
        )
        batch = provider.make_batch((case_id,), key)
        if int(batch.receivers.fluid_xy.shape[1]) != 1024:
            raise ValueError("R13-5 health batch does not contain the complete fixed Q=1024 TRAIN sample.")
        prepared = provider._prepare_for_probe(model, batch)
        block_values = self._block_payload(prepared)
        _route_by_block, allowed_by_block, route_parameter_refs = self._route_health_inputs(
            model, prepared, block_values,
        )
        value_tensors = {
            name: getattr(prepared.organization, name, None)
            for name in ("member_values_M", "member_values_E")
        }
        value_tensors = {name: value for name, value in value_tensors.items()
                         if torch.is_tensor(value) and value.requires_grad}
        block_tensors = {name: block_values.get(name) for name in JOINT_BLOCKS}
        block_tensors = {name: value for name, value in block_tensors.items()
                         if torch.is_tensor(value) and value.requires_grad}
        mode_summaries = {}
        native_by_mode = {}
        trace_by_mode = {}
        for mode in ("full", "restricted"):
            with model.core.read_execution(mode):
                native, work, trace = self._native_with_first_eight_trace(model, prepared)
            prediction = ThermalContentPredictions(prepared, native, work=work)
            roles = provider._role_losses(prediction, batch.targets)
            native_by_mode[mode] = native
            trace_by_mode[mode] = trace
            objective = sum(float(provider.native_weights[name]) * values.mean()
                            for name, values in roles.items())
            address_scores = trace.get("proposal_scores_joint")
            gradient_targets = []
            target_indices = {}
            for name, value in block_tensors.items():
                target_indices[name] = len(gradient_targets)
                gradient_targets.append(value)
            for name, value in value_tensors.items():
                target_indices[name] = len(gradient_targets)
                gradient_targets.append(value)
            address_gradient_index = None
            if torch.is_tensor(address_scores) and address_scores.requires_grad:
                address_gradient_index = len(gradient_targets)
                gradient_targets.append(address_scores)
            read_path_target_indices = {}
            for role in ("M", "E"):
                for kind, trace_key in (
                    ("read_strength", f"member_read_strength_{role}"),
                    ("context", f"context_{role}"),
                ):
                    value = trace.get(trace_key)
                    if torch.is_tensor(value) and value.requires_grad:
                        read_path_target_indices[(kind, role)] = len(gradient_targets)
                        gradient_targets.append(value)
            route_parameter_indices = {}
            for block, parameters in route_parameter_refs.items():
                indices = []
                for name, parameter in parameters:
                    indices.append((name, len(gradient_targets)))
                    gradient_targets.append(parameter)
                route_parameter_indices[block] = indices
            if not gradient_targets or not objective.requires_grad:
                raise RuntimeError("R13-5 complete native health objective is disconnected from live route parameters.")
            gradients = torch.autograd.grad(
                objective, tuple(gradient_targets), allow_unused=True,
                retain_graph=(mode == "full"),
            )
            dense_norms = {}
            allowed_norms = {}
            positive_norms = {}
            edge_activity = {}
            for block in JOINT_BLOCKS:
                gradient = gradients[target_indices[block]] if block in target_indices else None
                value = block_values.get(block)
                allowed = allowed_by_block[block]
                positive = allowed & (value.detach() > 0)
                dense_norms[block] = self._gradient_norm(gradient)
                allowed_norms[block] = self._gradient_norm(
                    None if gradient is None else torch.where(allowed, gradient, torch.zeros_like(gradient))
                )
                positive_norms[block] = self._gradient_norm(
                    None if gradient is None else torch.where(positive, gradient, torch.zeros_like(gradient))
                )
                positive_values = value.detach()[positive]
                edge_activity[block] = {
                    "allowed_edges": int(allowed.sum().item()),
                    "actual_positive_edges": int(positive.sum().item()),
                    "positive_gate_mean": (_finite_scalar(positive_values.mean())
                                           if positive_values.numel() else 0.0),
                }
            route_parameter_norms = {}
            for block, indices in route_parameter_indices.items():
                parameter_norms = [self._gradient_norm(gradients[index]) for _name, index in indices]
                route_parameter_norms[block] = math.sqrt(sum(value * value for value in parameter_norms))
            value_gradient_norms = {
                role: self._gradient_norm(gradients[target_indices[name]])
                if name in target_indices else 0.0
                for role, name in (("M", "member_values_M"), ("E", "member_values_E"))
            }
            score_gradient = (None if address_gradient_index is None else
                              gradients[address_gradient_index])
            if torch.is_tensor(score_gradient):
                module_count = int(prepared.organization.scene.num_modules)
                score_gradient = score_gradient[:, :min(8, int(score_gradient.shape[1]))]
                address_score_gradient_norms = {
                    "M": _finite_scalar(score_gradient[..., :module_count].detach().double().norm()),
                    "E": _finite_scalar(score_gradient[..., module_count:].detach().double().norm()),
                }
            else:
                address_score_gradient_norms = {"M": 0.0, "E": 0.0}
            read_path_adjoint_norms = {
                kind: {
                    role: (self._gradient_norm(gradients[read_path_target_indices[(kind, role)]])
                           if (kind, role) in read_path_target_indices else 0.0)
                    for role in ("M", "E")
                }
                for kind in ("read_strength", "context")
            }
            role_losses = {name: _finite_scalar(value.detach().mean()) for name, value in roles.items()}
            conditional_preferences = {}
            read_trace_means = {}
            for role in ("M", "E"):
                expectation = trace.get(f"member_conditional_expectation_{role}")
                entropy = trace.get(f"member_conditional_entropy_{role}")
                # The maintained core trace already normalizes entropy by the full eligible scene count.
                normalized_entropy = entropy if torch.is_tensor(entropy) else None
                context_norm = trace.get(f"member_context_norm_{role}")
                if not torch.is_tensor(context_norm):
                    context = trace.get(f"context_{role}")
                    context_norm = context.norm(dim=-1) if torch.is_tensor(context) else None
                candidate = trace.get(f"candidate_mask_{role}")
                read_trace_means[f"member_conditional_expectation_{role}"] = (
                    _finite_scalar(expectation.detach().mean()) if torch.is_tensor(expectation) else None
                )
                read_trace_means[f"member_conditional_entropy_normalized_{role}"] = (
                    _finite_scalar(normalized_entropy.detach().mean())
                    if torch.is_tensor(normalized_entropy) else None
                )
                for suffix, trace_key in (
                    ("read_contrast", f"member_read_contrast_{role}"),
                    ("read_strength", f"member_read_strength_{role}"),
                    ("gate_derivative", f"member_gate_derivative_{role}"),
                ):
                    value = trace.get(trace_key)
                    read_trace_means[f"member_{suffix}_{role}"] = (
                        _finite_scalar(value.detach().mean()) if torch.is_tensor(value) else None
                    )
                read_trace_means[f"member_context_norm_{role}"] = (
                    _finite_scalar(context_norm.detach().mean()) if torch.is_tensor(context_norm) else None
                )
                read_trace_means[f"nonempty_candidate_query_fraction_{role}"] = (
                    _finite_scalar(candidate.any(-1).float().mean()) if torch.is_tensor(candidate) else None
                )
            for role in ("M", "E"):
                value = trace.get(f"conditional_preference_{role}")
                allowed = trace.get(f"full_eligible_mask_{role}")
                if torch.is_tensor(value):
                    selected = value.detach()[allowed] if torch.is_tensor(allowed) else value.detach().reshape(-1)
                    conditional_preferences[role] = {
                        "mean": _finite_scalar(selected.mean()) if selected.numel() else 0.0,
                        "max": _finite_scalar(selected.max()) if selected.numel() else 0.0,
                        "positive_entries": int((selected > 0).sum().item()),
                    }
            retained_rows = min(8, int(trace["full_eligible_mask"].shape[1]))
            mode_summaries[mode] = {
                "native_fieldwise_normalized_mse": role_losses,
                "native_objective_weighted_normalized_mse": sum(
                    float(provider.native_weights[name]) * value for name, value in role_losses.items()
                ),
                "native_objective_dense_adjoint_norm_by_applied_block": dense_norms,
                "native_objective_allowed_edge_adjoint_norm_by_applied_block": allowed_norms,
                "native_objective_actual_positive_edge_adjoint_norm_by_applied_block": positive_norms,
                "native_objective_applied_route_parameter_gradient_norm_by_block": route_parameter_norms,
                "native_objective_member_value_gradient_norm_by_role": value_gradient_norms,
                "native_objective_member_read_strength_adjoint_l2_by_role": read_path_adjoint_norms["read_strength"],
                "native_objective_member_context_adjoint_l2_by_role": read_path_adjoint_norms["context"],
                "member_read_trace_means": read_trace_means,
                "applied_block_edge_activity": edge_activity,
                "native_objective_proposal_address_score_gradient_norm_by_role": address_score_gradient_norms,
                "applied_stage_provenance": block_values["provenance"],
                "full_eligible_count_mean": _finite_scalar(trace["full_eligible_mask"].sum(-1).float().mean()),
                "joint_candidate_count_mean": _finite_scalar(trace["joint_candidate_mask"].sum(-1).float().mean()),
                "candidate_role_counts_mean": {
                    role: _finite_scalar(trace[f"candidate_mask_{role}"].sum(-1).float().mean())
                    for role in ("M", "E")
                },
                "prior_role_mass_mean": {
                    role: _finite_scalar(trace[f"prior_role_mass_{role}"].mean())
                    for role in ("M", "E") if f"prior_role_mass_{role}" in trace
                },
                "retained_role_mass_mean": {
                    role: _finite_scalar(trace[f"retained_joint_mass_{role}"].mean())
                    for role in ("M", "E") if f"retained_joint_mass_{role}" in trace
                },
                "conditional_preference_summary": conditional_preferences,
                "address_block_mass_mean": {
                    block: _finite_scalar(trace[f"address_block_mass_{block}"].mean())
                    for block in ("MM", "EM", "EE", "ME", "self_M")
                    if f"address_block_mass_{block}" in trace
                },
                "candidate_original_id_rows": {
                    role: (trace[f"candidate_ids_{role}"].detach().cpu().tolist()
                           if torch.is_tensor(trace.get(f"candidate_ids_{role}")) else [])
                    for role in ("M", "E")
                },
                "prepared_union_query_count": int(prepared.query_batch.num_queries),
                "retained_trace_query_count": retained_rows,
                "native_primary_fluid_query_count": int(batch.receivers.fluid_xy.shape[1]),
                "native_loss_query_scope": "complete provider-sampled Q=1024 TRAIN batch with all native surface and material targets",
                "trace_query_scope": "first eight rows of the actual prepared query union; native losses and block gradients use the complete union",
                "decode_work_counts": {str(name): int(value) for name, value in work.items()},
            }
        for role in ("M", "E"):
            full_pref = trace_by_mode["full"].get(f"conditional_preference_{role}")
            retained = trace_by_mode["restricted"].get(f"candidate_mask_{role}")
            mode_summaries["restricted"].setdefault("full_reader_preference_capture", {})[role] = (
                _finite_scalar((full_pref * retained.to(full_pref)).sum(-1).mean())
                if torch.is_tensor(full_pref) and torch.is_tensor(retained) else None
            )
        repeatability, heat_sensitivity = {}, {}
        with torch.no_grad():
            repeat_prepared = provider._prepare_for_probe(model, batch)
            heat_batch = replace(batch, receivers=replace(
                batch.receivers, heat=batch.receivers.heat * 1.1,
            ))
            heat_prepared = provider._prepare_for_probe(model, heat_batch)
            for mode in ("full", "restricted"):
                with model.core.read_execution(mode):
                    repeat_native, _repeat_counts, _repeat_output = model.apply_native_with_work_counts(
                        repeat_prepared, batch.receivers.heat,
                    )
                    heat_native, _heat_counts, _heat_output = model.apply_native_with_work_counts(
                        heat_prepared, heat_batch.receivers.heat,
                    )
                from .member_routing_health import _native_field_delta
                repeatability[mode] = _native_field_delta(
                    native_by_mode[mode], repeat_native, model, prepared,
                )
                heat_sensitivity[mode] = _native_field_delta(
                    native_by_mode[mode], heat_native, model, prepared,
                )
                mode_summaries[mode]["repeatability_fieldwise_native_delta"] = repeatability[mode]
                mode_summaries[mode]["heat_plus10_fieldwise_native_delta"] = heat_sensitivity[mode]
        expected = set(THERMAL_CONTENT_LOSS_WEIGHTS) - {"response"}
        if any(set(summary["native_fieldwise_normalized_mse"]) != expected
               for summary in mode_summaries.values()):
            raise RuntimeError("R13-5 health did not observe every maintained native receiver role.")
        return {
            "case_id": str(case_id),
            "module_count": int(prepared.organization.scene.module_mask[0].sum().item()),
            "modes": mode_summaries,
            "prescribed_heat_input": batch.receivers.heat.detach().cpu().tolist(),
        }

    def observe(self, model, *, training_epoch: int) -> dict[str, Any]:
        from honf_runtime.unified_training import capture_rng_state, restore_rng_state

        rng = capture_rng_state()
        provider_state = copy.deepcopy(self.provider.training_state_dict())
        training_mode = bool(model.training)
        core = model.core
        core_clock = (getattr(core, "_joint_epoch", None), getattr(core, "_joint_update_index", None))
        device = torch.device(self.provider.device)
        model.eval()
        try:
            with torch.enable_grad():
                cases = [self._observe_case(model, case_id) for case_id in self.panel_case_ids]
        finally:
            model.train(training_mode)
            if core_clock[0] is not None:
                core._joint_epoch = core_clock[0]
            if core_clock[1] is not None:
                core._joint_update_index = core_clock[1]
            self.provider.load_training_state_dict(provider_state)
            restore_rng_state(rng)
        return {
            "schema": "thermal_joint_member_routing_health_v1",
            "training_epoch": int(training_epoch),
            "scope": "fixed TRAIN cases; complete Q=1024 native objective with first-eight prepared-union routing trace",
            "case_ids": list(self.panel_case_ids),
            "cases": cases,
            "all_four_applied_blocks_observed": all(
                all(name in case["modes"][mode]["native_objective_actual_positive_edge_adjoint_norm_by_applied_block"]
                    for name in JOINT_BLOCKS)
                for case in cases for mode in ("full", "restricted")
            ),
            "mode_restored": bool(model.training) == training_mode,
            "device": str(device),
            "interpretation": "Per-role losses, candidate support and block gradients are descriptive; unused M or E value reads are not failures when prescribed context carries their controls.",
        }

class _PairedMemberRoutingTask(ThermalInteractionReusedMemberRoutingTask):
    """Keep Thermal data, normalization, objective and sampling on the maintained provider."""

    training_state_schema = TRAINING_STATE_SCHEMA

    def identity_payload(self):
        payload = super().identity_payload()
        payload.update({
            "schema": self.provider_schema,
            "workflow": self.workflow,
            "architecture": self.model.architecture,
            "member_routing_schema": self.model.CORE_SCHEMA,
            "adapter_state": self.model.state_payload(),
            "training_sampling_arm": self.training_sampling_arm,
            "loss_lineage": {
                "source_kind": self.source_kind,
                "parent_completed_epoch": self.source_parent_age,
                "new_reader_epochs_start_at": self.source_parent_age + 1,
                "native_loss_roles": [role for role in THERMAL_CONTENT_LOSS_WEIGHTS
                                      if role != "response" and role in self.native_weights],
                "paired_paths": ["full_all_eligible", "restricted_joint95"],
            },
            "training_reader_objective": (
                "one half full all-eligible native loss plus one half restricted joint95 native loss; "
                "same original prepared context and prescribed inputs"
            ),
            "restricted_evaluation_execution": "joint95",
            "joint_member_coverage": 0.95,
            "objective": {
                **copy.deepcopy(payload.get("objective", {})),
                "paired_full_restricted": True,
                "guide_kl_weight": 0.0,
                "organizer_objective": False,
            },
        })
        return payload

    def optimizer_groups(self, model, arm, stage):
        del stage
        if model is not self.model or arm != model.architecture:
            raise ValueError("Wrong R13-5 member-routing optimizer owner.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names:
            raise ValueError("R13-5 model has no trainable parameters.")
        optimizer = self.recipe["optimizer"]
        schedule_horizon = int(self.recipe["training"]["optimizer_schedule_horizon"])
        lr = float(optimizer["lr"])
        schedule = ScheduleSpec(
            peak_lr=lr, warmup_start_lr=lr, warmup_epochs=0,
            hold_through_epoch=schedule_horizon - 1,
            total_epochs=schedule_horizon, final_lr=lr,
        )
        return (OptimizerGroupSpec(
            "joint", names, schedule,
            weight_decay=float(optimizer["weight_decay"]),
            betas=tuple(map(float, optimizer["betas"])),
            eps=float(optimizer["eps"]),
        ),)

    def _paired_predictions(self, prepared, physical_heat):
        counts_by_mode = {}
        natives = {}
        for mode in ("full", "restricted"):
            with self.model.core.read_execution(mode):
                native, counts, _output = self.model.apply_native_with_work_counts(prepared, physical_heat)
            natives[mode] = native
            counts_by_mode[mode] = counts
        work = {
            "native_fluid_queries": int(prepared.fluid_xy.shape[0] * prepared.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        for mode, counts in counts_by_mode.items():
            work.update({f"core_decode_joint_{mode}_{name}": int(value) for name, value in counts.items()})
        return ThermalJointPairPredictions(
            ThermalContentPredictions(prepared, natives["full"], work=counts_by_mode["full"]),
            ThermalContentPredictions(prepared, natives["restricted"], work=counts_by_mode["restricted"]),
            work,
        )

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0,
                       temperature=None, **kwargs):
        del phase, epoch, temperature
        if kwargs:
            raise TypeError(f"Unsupported R13-5 native prediction options: {sorted(kwargs)}")
        if model is not self.model or execution_mode != model.architecture:
            raise ValueError("Wrong R13-5 member-routing model/arm.")
        prepared = model.prepare_native(
            scene.structure, receivers.fluid_xy, physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points, ntheta=receivers.ntheta,
            scene_key=scene.case_ids,
        )
        if model.training:
            paired = self._paired_predictions(prepared, receivers.heat)
            return paired, {"work": dict(paired.work)}
        read_mode = getattr(model.core, "_read_execution_override", None) or "restricted"
        if read_mode not in ("full", "restricted"):
            raise ValueError(f"Unsupported R13-5 evaluation read mode {read_mode!r}.")
        with model.core.read_execution(read_mode):
            native, counts, _output = model.apply_native_with_work_counts(prepared, receivers.heat)
        work = {
            "native_fluid_queries": int(receivers.fluid_xy.shape[0] * receivers.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
            f"member_routing_read_mode_{read_mode}": 1,
        }
        work.update({f"core_prepare_primary_{name}": int(value)
                     for name, value in prepared.organization.work_counts.items()})
        work.update({f"core_decode_primary_{name}": int(value) for name, value in counts.items()})
        return ThermalContentPredictions(prepared, native, work=work), {"work": work}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        if not isinstance(predictions, ThermalJointPairPredictions):
            return super().loss_terms(predictions, targets, phase, auxiliary_state)
        if phase != "joint":
            raise ValueError("R13-5 paired native losses are defined only for the joint phase.")
        full_roles = self._role_losses(predictions.full, targets)
        restricted_roles = self._role_losses(predictions.restricted, targets)
        if set(full_roles) != set(restricted_roles) or set(full_roles) != set(self.native_weights):
            raise ValueError("R13-5 full and joint95 native loss roles differ.")
        if isinstance(auxiliary_state, dict):
            auxiliary_state["joint95_native_path_losses"] = {
                path: {name: float(values.detach().mean()) for name, values in roles.items()}
                for path, roles in (("full", full_roles), ("restricted", restricted_roles))
            }
        metric_work = dict(predictions.work)
        metric_prefix = "joint_member_native_loss_metric__"
        metric_work[f"{metric_prefix}case_count"] = float(len(targets.case_ids))
        for path, roles in (("full", full_roles), ("restricted", restricted_roles)):
            for name, values in roles.items():
                metric_work[f"{metric_prefix}{path}__{name}__sum"] = float(
                    values.detach().double().sum().cpu()
                )
        predictions.work = metric_work
        return {
            name: LossTerm(
                (full_roles[name].sum() + restricted_roles[name].sum()) * 0.5,
                len(targets.case_ids), float(self.native_weights[name]),
            ) for name in full_roles
        }

    def work_counts(self, batch, predictions, auxiliary_state):
        del batch, auxiliary_state
        if isinstance(predictions, ThermalJointPairPredictions):
            return {str(name): value for name, value in predictions.work.items()}
        if predictions.work is not None:
            return {str(name): value for name, value in predictions.work.items()}
        return {}

    def source_training_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state, Mapping) or state.get("schema") != "thermal_member_routing_native_training_state_v3":
            raise ValueError("R13-5 transfer requires an R13-4 Guide provider training state.")
        converted = copy.deepcopy(dict(state))
        converted["schema"] = self.training_state_schema
        converted.pop("member_routing_health", None)
        return converted

    def prepare_transfer_health(self, model, source_epoch: int | None = None):
        health = self._member_routing_health
        if health is None:
            return None
        if model is not self.model:
            raise ValueError("R13-5 health baseline must observe its owned transferred model.")
        origin = self.source_parent_age if source_epoch is None else int(source_epoch)
        if origin != self.source_parent_age:
            raise ValueError("R13-5 health baseline must match the declared source epoch.")
        if health.initial_observation is not None:
            return None
        observation = health.observe(model, training_epoch=origin)
        health.record_initial(observation)
        return observation

    def on_phase_start(self, *, model, arm, epoch, phase, temperature):
        return super().on_phase_start(
            model=model, arm=arm, epoch=epoch, phase=phase, temperature=temperature,
        )

    def completed_training_epoch(self, model, engine, epoch, output, row):
        delegate = getattr(super(), "completed_training_epoch", None)
        if callable(delegate):
            delegate(model, engine, epoch, output, row)
        row["reader_epoch_context"] = {
            "schema": "thermal_joint_member_routing_epoch_context_v1",
            "source_kind": self.source_kind,
            "source_parent_completed_epoch": self.source_parent_age,
            "absolute_training_epoch": int(epoch),
            "additional_epochs_after_parent": max(0, int(epoch) - self.source_parent_age),
        }
        work = row.get("work_counts", {})
        prefix = "joint_member_native_loss_metric__"
        case_count = float(work.get(f"{prefix}case_count", 0.0)) if isinstance(work, Mapping) else 0.0
        if case_count > 0:
            paths = {path: {} for path in ("full", "restricted")}
            for key, numerator in work.items():
                if not key.startswith(prefix) or not key.endswith("__sum"):
                    continue
                suffix = key[len(prefix):-len("__sum")]
                path, role = suffix.split("__", 1)
                if path in paths:
                    paths[path][role] = float(numerator) / case_count
            row["paired_reader_native_losses"] = {
                "schema": "thermal_joint_member_paired_native_loss_summary_v1",
                "epoch_context": copy.deepcopy(row["reader_epoch_context"]),
                "per_case_mean_normalized_mse": paths,
                "objective": "arithmetic mean of matching all-eligible and joint95 native role losses",
            }
