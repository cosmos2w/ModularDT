"""One native fine reader for the three shared-core organizing strategies.

Physical kernel names match B-fine. Plans are built before simultaneous
MM/ME/EM mixing. The initial reference executor evaluates dense rectangles;
positive permissions are logical work, not a claim of GPU row savings.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn

from honf_forward_core.training.hypergraph_shadow import (
    bridge_local_context,
    detach_tree,
    local_attention_context,
    local_modulated_sum,
)

from .dense_pairwise import DensePairwiseField
from .hypergraph_interventions import fixed_structure_intervention, membership_intervention
from .typed_hypergraph_state import (
    SOURCE_TYPE,
    ProjectedSourceAccess,
    TypedSourceAccess,
    prepare_projected_action,
    prepare_structural_cost,
    source_moments,
    structural_cost,
)
from .types import EncodedInterfaceCase


class TypedHypergraphField(DensePairwiseField):
    phase_diagnostic_prefix = "hypergraph_"
    supports_local_context_shadow = True

    def __init__(self, hidden_dim: int, message_hidden_dim: int, num_heads: int,
                 fourier_frequencies: int, *, architecture: str, spatial_dim: int,
                 module_characteristic_length: float, control_dim: int = 16,
                 activation_checkpointing: bool = False, options: dict | None = None):
        super().__init__(hidden_dim, message_hidden_dim, num_heads, fourier_frequencies,
                         activation_checkpointing=activation_checkpointing)
        settings = dict(options or {})
        self.live_task_controls = architecture == "native_context_global_control_honf"
        self.uses_structural_objective = not self.live_task_controls
        if architecture in {"adaptive_receiver_hypergraph_honf", "faithful_receiver_hypergraph_honf",
                            "native_context_tree_honf"}:
            from .adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph
            if architecture in {"faithful_receiver_hypergraph_honf", "native_context_tree_honf"}:
                settings.update(faithful_controls=True, measure_consistent=True)
            self.organizer = AdaptiveReceiverHypergraph(hidden_dim, spatial_dim=spatial_dim,
                                                       control_dim=control_dim, **settings)
        elif self.live_task_controls:
            from .global_control_hypergraph import GlobalControlHypergraph
            self.organizer = GlobalControlHypergraph(hidden_dim, spatial_dim=spatial_dim,
                                                     control_dim=control_dim, **settings)
        elif architecture in {"overlap_control_hypergraph_honf", "local_overlap_hypergraph_honf"}:
            from .overlap_control_hypergraph import OverlapControlHypergraph
            self.organizer = OverlapControlHypergraph(
                hidden_dim, spatial_dim, control_dim=control_dim,
                local_access=architecture == "local_overlap_hypergraph_honf",
                module_characteristic_length=module_characteristic_length, **settings,
            )
        else:
            raise ValueError(f"Unknown typed hypergraph architecture: {architecture}")
        self.control_gain = nn.ModuleDict({name: nn.Linear(control_dim, 1)
                                          for name in ("MM", "ME", "EM", "QM")})
        self.control_gain["QE"] = nn.Linear(control_dim, num_heads)
        self.control_score = nn.Linear(control_dim, num_heads)
        for module in (*self.control_gain.values(), self.control_score):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.permission_mode = "hard"
        self.control_execution = "projected"
        self.training_gradient_mode = "whole_wrapper_shadow_v1"
        if architecture == "native_context_tree_honf":
            self.training_gradient_mode = "local_context_shadow_v1"
        elif self.live_task_controls:
            self.training_gradient_mode = "ordinary_task_v1"
            self.supports_local_context_shadow = False
            # One source-independent QE score cancels in source softmax.
            self.control_score.requires_grad_(False)
        self.execution_mode = "dense_masked_reference"
        self.execution_receiver_chunk = 128
        self.plan_intervention = "normal"
        self.epoch = 0
        self.total_epochs = 1000
        self.last_structural_cost: torch.Tensor | None = None
        self._phase_costs: list[torch.Tensor] = []

    def set_training_progress(self, *, epoch: int, total_epochs: int | None = None):
        self.epoch = int(epoch)
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)
        setter = getattr(self.organizer, "set_epoch", None)
        if callable(setter):
            setter(self.epoch)

    def selection_state(self):
        return {"epoch": self.epoch, "total_epochs": self.total_epochs}

    def set_execution_mode(self, mode: str, *, receiver_chunk_size: int = 128) -> None:
        """Choose a tested same-operator executor for frozen inference."""
        if mode not in {"dense_masked_reference", "rectangular_subset"} or receiver_chunk_size < 1:
            raise ValueError("Unknown fine executor or nonpositive receiver tile size")
        if self.training and mode != "dense_masked_reference":
            raise ValueError("The campaign trains with the dense reference executor")
        self.execution_mode, self.execution_receiver_chunk = mode, int(receiver_chunk_size)

    def _source_tiles(self, access, *, receiver_chunk_size=None):
        receiver_chunk_size = self.execution_receiver_chunk if receiver_chunk_size is None else receiver_chunk_size
        for case in range(access.support.shape[0]):
            for start in range(0, access.support.shape[1], receiver_chunk_size):
                stop = min(start + receiver_chunk_size, access.support.shape[1])
                sources = torch.nonzero(access.support[case, start:stop].any(0), as_tuple=False).flatten()
                if sources.numel():
                    yield case, start, stop, sources

    def _fine_mlp(self, module, parts, access, output_dim, *, execution_mode=None, receiver_chunk_size=None):
        """Evaluate physical kernels on rectangular source unions per tile.

        Policy moments and control projections remain dense. The measured rows
        below refer to fine message/geometry kernels, separately from policy
        work and attention cells. Positive permissions inside a rectangle can
        still be sparse; they retain the same exact masks and normalization.
        """
        execution_mode = self.execution_mode if execution_mode is None else execution_mode
        allocated = access.support.numel()
        if execution_mode == "dense_masked_reference":
            value = self._mlp(module, torch.cat(parts, -1))
            return value, {"executed_rows": allocated, "allocated_rows": allocated,
                           "executed_eligible_pairs": access.diagnostics["eligible_pairs"], "fine_calls": 1}
        if self.training:
            raise ValueError("Rectangular execution is frozen-inference only")
        value = parts[0].new_zeros((*access.support.shape, output_dim))
        rows = calls = eligible = 0
        valid = access.diagnostics["pair_valid"]
        for case, start, stop, sources in self._source_tiles(access, receiver_chunk_size=receiver_chunk_size):
            features = torch.cat([part[case, start:stop].index_select(1, sources) for part in parts], -1)
            messages = self._mlp(module, features)
            value[case, start:stop].index_copy_(1, sources, messages)
            rows += (stop - start) * sources.numel()
            calls += 1
            eligible += int(valid[case, start:stop].index_select(1, sources).sum())
        return value, {"executed_rows": rows, "allocated_rows": allocated,
                       "executed_eligible_pairs": eligible, "fine_calls": calls}

    def set_plan_intervention(self, mode: str) -> None:
        if mode not in {"normal", "full_access", "root_union", "control_identity",
                        "geometry", "rewire", "exchange", "fixed_structure", "fixed_summary"}:
            raise ValueError(f"Unknown typed plan intervention: {mode}")
        if self.training and mode != "normal":
            raise ValueError("Same-weight plan interventions are evaluation-only.")
        if mode == "fixed_summary" and getattr(self, "training_population_summary", None) is None:
            raise ValueError("Fixed summary requires train-input calibration before evaluation")
        self.plan_intervention = mode

    def _access(self, plan, receivers, mechanism, receiver_tokens=None, pair_valid=None, *, mode=None, soft=None):
        mode = self.plan_intervention if mode is None else mode
        soft = self.permission_mode == "soft" if soft is None else soft
        access = self.organizer.access(plan, receivers, mechanism, receiver_tokens,
                                       soft=soft, pair_valid=pair_valid)
        if mode in {"full_access", "root_union"}:
            kind = SOURCE_TYPE[mechanism]
            if mode == "full_access":
                member = plan.source_valid[kind][:, None].to(receivers.dtype)
            else:
                member = (plan.memberships[mechanism].sum(1, keepdim=True) > 0).to(receivers.dtype)
            controls = plan.controls[mechanism].mean(1, keepdim=True)
            original_control = access.control
            effective_validity = access.diagnostics["pair_valid"]
            access = source_moments(receivers.new_ones((*receivers.shape[:2], 1)), member,
                                    controls, plan.source_measures[kind], plan.source_valid[kind],
                                    pair_valid=effective_validity, near=access.near)
            if mode == "full_access":
                # Change access alone. Previously omitted pairs keep their
                # normal zero control; existing source controls stay intact.
                access = replace(access, control=original_control)
        if mode == "control_identity":
            access = replace(access, control=torch.zeros_like(access.control))
        if self.training and not soft and not self.live_task_controls:
            access = detach_tree(access)
        return access

    def _prepare_actions(self, plan, *, detach_projection=False):
        actions = {}
        for mechanism in plan.memberships:
            gain = self.control_gain[mechanism]
            if mechanism == "QE":
                weight = torch.cat((self.control_score.weight, gain.weight), 0)
                bias = torch.cat((self.control_score.bias, gain.bias), 0)
            else:
                weight, bias = gain.weight, gain.bias
            actions[mechanism] = prepare_projected_action(
                plan.memberships[mechanism], plan.controls[mechanism], weight, bias,
                detach_projection=detach_projection)
        return actions

    def _numerical_access(self, plan, receivers, mechanism, actions, receiver_tokens=None,
                          pair_valid=None, *, mode=None, soft=None, diagnostics=False):
        """Small-channel action path; full vectors remain public export only."""
        mode = self.plan_intervention if mode is None else mode
        soft = bool(plan.strategy_data["soft"]) if soft is None else soft
        if actions is None:
            return self._access(plan, receivers, mechanism, receiver_tokens, pair_valid,
                                mode=mode, soft=soft)
        access = self.organizer.access(plan, receivers, mechanism, receiver_tokens,
            soft=soft, pair_valid=pair_valid, prepared_action=actions[mechanism],
            include_diagnostics=diagnostics,
            detach_permissions=self.training and not soft and not self.live_task_controls)
        if mode in {"full_access", "root_union"}:
            kind = SOURCE_TYPE[mechanism]
            member = (plan.source_valid[kind][:, None].to(receivers.dtype) if mode == "full_access"
                      else (plan.memberships[mechanism].sum(1, keepdim=True) > 0).to(receivers.dtype))
            controls = plan.controls[mechanism].mean(1, keepdim=True)
            gain = self.control_gain[mechanism]
            if mechanism == "QE":
                weight = torch.cat((self.control_score.weight, gain.weight), 0)
                bias = torch.cat((self.control_score.bias, gain.bias), 0)
            else:
                weight, bias = gain.weight, gain.bias
            action = prepare_projected_action(member, controls, weight, bias)
            original = access.projected
            original_diagnostics = access.diagnostics
            access = source_moments(receivers.new_ones((*receivers.shape[:2], 1)), member,
                controls, plan.source_measures[kind], plan.source_valid[kind],
                pair_valid=original_diagnostics["pair_valid"], near=access.near,
                prepared_action=action, include_diagnostics=diagnostics)
            if "receiver_measures" in original_diagnostics:
                access.diagnostics["receiver_measures"] = original_diagnostics["receiver_measures"]
            if mode == "full_access":
                access = replace(access, projected=original)
        if mode == "control_identity":
            access = replace(access, projected=torch.zeros_like(access.projected))
        # Hard organizer paths stay detached, while physical projections remain
        # live. Detach the plan in prepare, not its projected physical action.
        return access

    def _modulate(self, messages, access: TypedSourceAccess, mechanism: str, *, mode=None, soft=None):
        mode = self.plan_intervention if mode is None else mode
        soft = self.permission_mode == "soft" if soft is None else soft
        weight = access.weight.to(messages.dtype) if soft else access.weight
        if isinstance(access, ProjectedSourceAccess):
            gain = access.projected
        else:
            control = (access.control.to(self.control_gain[mechanism].weight.dtype)
                       if soft else access.control)
            gain = self.control_gain[mechanism](control)
        if mode == "control_identity":
            gain = torch.zeros_like(gain)
        return messages * (1.0 + torch.tanh(gain)) * weight[..., None]

    def _reduce_messages(self, messages, access, mechanism, *, soft_access=None,
                         source_measure=None, mode=None, soft=None):
        hard = self._modulate(messages, access, mechanism, mode=mode, soft=soft)
        if source_measure is not None:
            hard = hard * source_measure[..., None]
        hard = hard.sum(2)
        if soft_access is None:
            return hard
        soft = local_modulated_sum(messages.detach(), soft_access.weight,
                                   soft_access.projected,
                                   None if source_measure is None else source_measure.detach())
        return bridge_local_context(hard, soft)

    @staticmethod
    def _ledger(mechanism, access, work):
        if isinstance(work, int):
            work = {"executed_rows": work, "allocated_rows": work,
                    "executed_eligible_pairs": access.diagnostics["eligible_pairs"], "fine_calls": 1}
        counts = {key: torch.as_tensor(value, device=access.weight.device, dtype=torch.long)
                  for key, value in work.items()}
        counts["padded_rows"] = counts["executed_rows"] - counts["executed_eligible_pairs"]
        counts["skipped_eligible_pairs"] = access.diagnostics["eligible_pairs"] - counts["executed_eligible_pairs"]
        return {f"hypergraph_{mechanism}_{name}": value for name, value in {
            **{key: value for key, value in access.diagnostics.items() if value.ndim == 0},
            **counts,
        }.items()}

    def prepare(self, encoded: EncodedInterfaceCase, module_states: torch.Tensor, *,
                return_routing_maps=False, interaction_context=None):
        phase = int(str(getattr(interaction_context, "phase", "P0"))[-1])
        soft = self.permission_mode == "soft"
        local_shadow = (self.training and torch.is_grad_enabled() and not soft
                        and self.training_gradient_mode == "local_context_shadow_v1")
        if local_shadow and (self.plan_intervention != "normal" or self.execution_mode != "dense_masked_reference"):
            raise ValueError("Local-context shadow requires the ordinary hard dense training operator")
        if phase == 0:
            self._phase_costs = []
        plan = self.organizer.prepare(encoded, module_states, phase=phase, soft=soft)
        plan = membership_intervention(plan, self.plan_intervention)
        if self.plan_intervention == "fixed_structure":
            plan = fixed_structure_intervention(plan)
        if self.plan_intervention == "fixed_summary":
            plan = self.training_population_summary.apply(plan)
        if self.training and not soft and not self.live_task_controls:
            plan = detach_tree(plan)
        _batch, modules, _ = module_states.shape
        environments = encoded.env_tokens.shape[1]
        present = encoded.module_present
        count = present.sum(1, keepdim=True)
        centres, env = encoded.module_centers, encoded.env_coords
        mm_valid = ((present[:, :, None] > .5) & (present[:, None] > .5)
                    & ~torch.eye(modules, device=centres.device, dtype=torch.bool)[None])
        me_valid = (present[:, :, None] > .5).expand(-1, -1, environments)
        em_valid = (present[:, None] > .5).expand(-1, environments, -1)
        actions = self._prepare_actions(plan) if self.control_execution == "projected" else None
        soft_plan = (self.organizer.prepare(detach_tree(encoded), module_states.detach(), phase=phase, soft=True)
                     if local_shadow else None)
        soft_actions = self._prepare_actions(soft_plan, detach_projection=True) if local_shadow else None
        soft_accesses = {}
        if local_shadow:
            for tau, points, tokens, valid in (("MM", centres, module_states, mm_valid),
                    ("ME", centres, module_states, me_valid), ("EM", env, encoded.env_tokens, em_valid)):
                soft_accesses[tau] = self._numerical_access(soft_plan, points.detach(), tau,
                    soft_actions, tokens.detach(), valid, mode="normal", soft=True)
        mm_access = self._numerical_access(plan, centres, "MM", actions, module_states, mm_valid,
                                           diagnostics=return_routing_maps)
        me_access = self._numerical_access(plan, centres, "ME", actions, module_states, me_valid,
                                           diagnostics=return_routing_maps)
        em_access = self._numerical_access(plan, env, "EM", actions, encoded.env_tokens, em_valid,
                                           diagnostics=return_routing_maps)
        relative = (centres[:, :, None] - centres[:, None]) / encoded.coordinate_scale
        mm, mm_work = self._fine_mlp(self.mm_message, (
            module_states[:, :, None].expand(-1, -1, modules, -1),
            module_states[:, None].expand(-1, modules, -1, -1), self.relative_fourier(relative)),
            mm_access, self.hidden_dim)
        a_mm = self._reduce_messages(mm, mm_access, "MM", soft_access=soft_accesses.get("MM")) / (1.0 + count[..., None])
        relative = (centres[:, :, None] - env[:, None]) / encoded.coordinate_scale
        me, me_work = self._fine_mlp(self.me_message, (
            module_states[:, :, None].expand(-1, -1, environments, -1),
            encoded.env_tokens[:, None].expand(-1, modules, -1, -1), self.relative_fourier(relative)),
            me_access, self.hidden_dim)
        a_me = self._reduce_messages(me, me_access, "ME", soft_access=soft_accesses.get("ME"),
                                     source_measure=encoded.env_weights[:, None])
        a_me = a_me / encoded.env_weights.sum(1)[:, None, None].clamp_min(1.e-12)
        em, em_work = self._fine_mlp(self.em_message, (
            encoded.env_tokens[:, :, None].expand(-1, -1, modules, -1),
            module_states[:, None].expand(-1, environments, -1, -1),
            self.relative_fourier(-relative.transpose(1, 2))), em_access, self.hidden_dim)
        a_em = self._reduce_messages(em, em_access, "EM", soft_access=soft_accesses.get("EM")) / (1.0 + count[..., None])
        contextual_modules = (module_states + self.module_update(torch.cat((
            module_states, a_mm, a_me, encoded.global_token[:, None].expand(-1, modules, -1)), -1))) * present[..., None]
        contextual_env = encoded.env_tokens + self.env_update(torch.cat((
            encoded.env_tokens, a_em, encoded.global_token[:, None].expand(-1, environments, -1)), -1))
        key, value = self.project_environment_sources(contextual_env)
        ledger = {}
        for mechanism, access, work in (("MM", mm_access, mm_work),
                                       ("ME", me_access, me_work), ("EM", em_access, em_work)):
            ledger.update(self._ledger(mechanism, access, work))
        preparation_accesses = {"MM": mm_access, "ME": me_access, "EM": em_access}
        structural_plan = soft_plan if local_shadow else plan
        structural_accesses = soft_accesses if local_shadow else preparation_accesses
        structural = (prepare_structural_cost(structural_accesses, structural_plan)
                      if self.training and self.uses_structural_objective else None)
        return {"module_tokens": contextual_modules, "env_tokens": contextual_env,
                "hypergraph_actions": actions, "hypergraph_action_plan": plan, "hypergraph_structural_preparation": structural,
                "hypergraph_soft_plan": soft_plan, "hypergraph_soft_actions": soft_actions,
                "hypergraph_soft_accesses": soft_accesses,
                "hypergraph_training_gradient_mode": self.training_gradient_mode,
                "projected_key": key, "projected_value": value, "hypergraph_plan": plan,
                "hypergraph_accesses": preparation_accesses,
                "hypergraph_ledger": ledger, "hypergraph_phase": phase,
                "hypergraph_executor": self.execution_mode,
                "hypergraph_receiver_chunk": self.execution_receiver_chunk,
                "hypergraph_intervention": self.plan_intervention,
                "return_routing_maps": return_routing_maps}

    def preparation_aux(self, state, *, include_diagnostics=False):
        result = dict(state["hypergraph_ledger"])
        plan = state["hypergraph_plan"]
        result["hypergraph_admitted_groups"] = (plan.admission > 0).sum(-1).detach()
        result["hypergraph_allocated_groups"] = plan.admission.new_full(
            (plan.admission.shape[0],), plan.admission.shape[-1]).detach()
        result["hypergraph_phase"] = plan.admission.new_tensor(state["hypergraph_phase"])
        result.update({f"hypergraph_{key}": value.detach() for key, value in plan.diagnostics.items()
                       if torch.is_tensor(value)})
        if include_diagnostics:
            for mechanism in plan.memberships:
                result[f"hypergraph_{mechanism}_membership"] = plan.memberships[mechanism].detach()
                result[f"hypergraph_{mechanism}_controls"] = plan.controls[mechanism].detach()
            result["hypergraph_group_centres"] = plan.centres.detach()
        return result

    def export_typed_state(self, state):
        plan = state["hypergraph_plan"]
        mode = state["hypergraph_intervention"]
        soft = bool(plan.strategy_data["soft"])
        return {**plan.export(), "receiver_access": lambda receivers, mechanism, receiver_tokens=None:
                self._access(plan, receivers, mechanism, receiver_tokens, mode=mode, soft=soft),
                "work": state["hypergraph_ledger"], "executor": state["hypergraph_executor"]}

    def read(self, state, encoded, receivers, receiver_features, *, return_routing_maps=False):
        plan = state["hypergraph_plan"]
        mode = state["hypergraph_intervention"]
        soft = bool(plan.strategy_data["soft"])
        execution_mode = state["hypergraph_executor"]
        receiver_chunk = state["hypergraph_receiver_chunk"]
        actions = state.get("hypergraph_actions")
        if actions is not None and state.get("hypergraph_action_plan", plan) is not plan:
            # Explicit diagnostic source refinement/rebinding must rebuild its
            # current action rather than consume an old phase projection.
            actions = self._prepare_actions(plan)
            state["hypergraph_actions"] = actions
            state["hypergraph_action_plan"] = plan
            state["hypergraph_structural_preparation"] = None
        qm = self._numerical_access(plan, receivers, "QM", actions, mode=mode, soft=soft,
                                    diagnostics=return_routing_maps)
        qe = self._numerical_access(plan, receivers, "QE", actions, mode=mode, soft=soft,
                                    diagnostics=return_routing_maps)
        soft_plan = state.get("hypergraph_soft_plan")
        soft_qm = soft_qe = None
        if soft_plan is not None:
            soft_qm = self._numerical_access(soft_plan, receivers.detach(), "QM", state["hypergraph_soft_actions"],
                                            mode="normal", soft=True)
            soft_qe = self._numerical_access(soft_plan, receivers.detach(), "QE", state["hypergraph_soft_actions"],
                                            mode="normal", soft=True)
        batch, queries, _ = receivers.shape
        modules = state["module_tokens"].shape[1]
        environments = state["env_tokens"].shape[1]
        relative = (receivers[:, :, None] - encoded.module_centers[:, None]) / encoded.coordinate_scale
        messages, qm_work = self._fine_mlp(self.query_module_message, (
            state["module_tokens"][:, None].expand(-1, queries, -1, -1),
            self.relative_fourier(relative), encoded.global_token[:, None, None].expand(-1, queries, modules, -1)),
            qm, self.hidden_dim, execution_mode=execution_mode, receiver_chunk_size=receiver_chunk)
        module_context = self.query_module_output(self._reduce_messages(messages, qm, "QM", mode=mode, soft_access=soft_qm, soft=soft)
            / (1.0 + encoded.module_present.sum(1)[:, None, None]))
        if environments:
            relative = (receivers[:, :, None] - encoded.env_coords[:, None]) / encoded.coordinate_scale
            bias, qe_work = self._fine_mlp(self.env_geometry_bias, (self.relative_fourier(relative),),
                                         qe, self.env_attention.num_heads, execution_mode=execution_mode,
                                         receiver_chunk_size=receiver_chunk)
            bias = bias.permute(0, 3, 1, 2)
            query = self.env_attention.project_query(self.env_query(receiver_features))
            safe_weight = torch.where(qe.support, encoded.env_weights[:, None] * qe.weight,
                                      torch.ones_like(qe.weight))
            if isinstance(qe, ProjectedSourceAccess):
                modulation, gain = qe.projected.split(self.env_attention.num_heads, -1)
                modulation, gain = modulation.permute(0, 3, 1, 2), gain.permute(0, 3, 1, 2)
            else:
                control = qe.control.to(self.control_score.weight.dtype)
                modulation = self.control_score(control).permute(0, 3, 1, 2)
                gain = self.control_gain["QE"](control).permute(0, 3, 1, 2)
            if mode == "control_identity":
                modulation, gain = torch.zeros_like(modulation), torch.zeros_like(gain)
            # Keep the logarithm inside the wide permission chain. Casting a
            # tiny positive weight first would recreate FP32 reciprocal overflow.
            log_weight = safe_weight.log()[:, None]
            if soft:
                log_weight = log_weight.to(query.dtype)
            if execution_mode == "dense_masked_reference":
                score = torch.matmul(query, state["projected_key"].transpose(-1, -2))
                score = score / math.sqrt(self.env_attention.head_dim)
                score = score + bias + log_weight
                score = score + modulation
                attention = self._masked_attention(score, qe.support)
                context = torch.matmul(attention * (1.0 + torch.tanh(gain)), state["projected_value"])
                qe_work["attention_cells"] = batch * queries * environments * self.env_attention.num_heads
            else:
                attention = query.new_zeros((batch, self.env_attention.num_heads, queries, environments))
                context = query.new_zeros(query.shape)
                cells = 0
                for case, start, stop, sources in self._source_tiles(qe, receiver_chunk_size=receiver_chunk):
                    key = state["projected_key"][case].index_select(1, sources)
                    value = state["projected_value"][case].index_select(1, sources)
                    score = torch.matmul(query[case, :, start:stop], key.transpose(-1, -2))
                    score = score / math.sqrt(self.env_attention.head_dim)
                    score = score + bias[case, :, start:stop].index_select(2, sources)
                    score = score + log_weight[case, :, start:stop].index_select(2, sources)
                    score = score + modulation[case, :, start:stop].index_select(2, sources)
                    active = qe.support[case, start:stop].index_select(1, sources)
                    block_attention = self._masked_attention(score[None], active[None])[0]
                    attention[case, :, start:stop].index_copy_(2, sources, block_attention)
                    weighted = block_attention * (1.0 + torch.tanh(gain[case, :, start:stop].index_select(2, sources)))
                    context[case, :, start:stop] = torch.matmul(weighted, value)
                    cells += (stop - start) * sources.numel() * self.env_attention.num_heads
                qe_work["attention_cells"] = cells
            if soft_qe is not None:
                soft_score, soft_gain = soft_qe.projected.split(self.env_attention.num_heads, -1)
                soft_context = local_attention_context(query.detach(), state["projected_key"].detach(),
                    state["projected_value"].detach(), bias.detach(), soft_qe.weight, soft_qe.support,
                    soft_score, soft_gain, encoded.env_weights.detach())
                context = bridge_local_context(context, soft_context)
            context = context.transpose(1, 2).reshape(batch, queries, self.hidden_dim)
            environment_context = self.env_attention.output(context)
            # An empty eligible row supplies no fabricated source/output bias.
            if soft_qe is None:
                environment_context = environment_context * qe.support.any(-1)[..., None]
            else:
                # Hard empty rows have zero context and no output bias. Keep
                # the soft context derivative open for eligible omitted donors
                # instead of masking it again after the completed-context bridge.
                environment_context = environment_context - (self.env_attention.output.bias
                    * (~qe.support.any(-1))[..., None])
        else:
            attention = None
            environment_context = module_context.new_zeros(module_context.shape)
            qe_work = {"executed_rows": 0, "allocated_rows": 0, "executed_eligible_pairs": 0,
                       "fine_calls": 0, "attention_cells": 0}
        accesses = {**state["hypergraph_accesses"], "QM": qm, "QE": qe}
        metrics = {}
        cost = None
        if self.uses_structural_objective and (self.training or return_routing_maps):
            cost_accesses = ({**state["hypergraph_soft_accesses"], "QM": soft_qm, "QE": soft_qe}
                             if soft_plan is not None else accesses)
            cost, metrics = structural_cost(cost_accesses, soft_plan if soft_plan is not None else plan,
                preparation=state.get("hypergraph_structural_preparation"))
        self.last_structural_cost = cost
        aux = {**self._ledger("QM", qm, qm_work), **self._ledger("QE", qe, qe_work)}
        # Live numerator/denominator scalars combine across uneven query tiles.
        if cost is not None:
            aux["hypergraph_structural_numerator"] = cost * queries
            aux["hypergraph_structural_denominator"] = cost.new_tensor(queries)
        if return_routing_maps:
            # Preparation pairs were executed once and already belong to
            # preparation_aux. Returning them in each query tile duplicates
            # their counts when InterfaceFieldCore sums read work.
            preparation_counts = {f"{tau}_unique_pairs" for tau in ("MM", "ME", "EM")}
            aux.update({f"hypergraph_{key}": value.detach() for key, value in metrics.items()
                        if key not in preparation_counts})
            for mechanism, access in (("QM", qm), ("QE", qe)):
                aux[f"hypergraph_{mechanism}_weight"] = access.weight.detach()
                aux[f"hypergraph_{mechanism}_edge_access"] = access.edge_access.detach()
                if access.near is not None:
                    aux[f"hypergraph_{mechanism}_near"] = access.near.detach()
            if attention is not None:
                aux["hypergraph_environment_attention"] = attention.detach()
        return module_context + environment_context, aux

    @staticmethod
    def _masked_attention(score, support):
        score = score.masked_fill(~support[:, None], torch.finfo(score.dtype).min)
        attention = torch.softmax(score, -1) * support[:, None]
        return attention / attention.sum(-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).tiny)


__all__ = ["TypedHypergraphField"]
