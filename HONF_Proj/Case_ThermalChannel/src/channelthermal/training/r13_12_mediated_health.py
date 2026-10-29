"""TRAIN-only observations of the live module-mediated native learning path."""
from __future__ import annotations

import copy
from dataclasses import replace

import torch

from honf_runtime.unified_training import SamplingKey, capture_rng_state, restore_rng_state
from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1
from .content_hypergraph_task import ThermalContentPredictions
from .joint_member_routing_task import _finite_scalar
from .member_routing_health import _native_field_delta
from .r13_8_anchored_task import AnchoredMemberRoutingHealth


class MediatedMemberRoutingHealth(AnchoredMemberRoutingHealth):
    """Reuse fixed membership/cadence while observing the actual composed read."""

    STATE_SCHEMA = "thermal_r13_12_mediated_health_state_v1"

    def _observe_case(self, model, case_id):
        provider = self.provider
        key = SamplingKey(
            seed=int(provider.seed), epoch=1, update_index=0, microbatch_index=0,
            stage="joint", arm=provider.training_sampling_arm,
            sampling_version=provider.sampling_version, dataset_id=provider.sampling_dataset_id,
        )
        batch = provider.make_batch((case_id,), key)
        prepared = provider._prepare_for_probe(model, batch)
        paired = model.core.decode_paired(prepared.organization, prepared.query_batch)
        outputs = dict(zip(("full", "restricted"), paired[:2]))
        native, losses = {}, {}
        objective = None
        for mode, output in outputs.items():
            fields = output.joint_fields
            native[mode] = model._extract_native_roles(prepared, fields[..., 3:4], fields[..., :3])
            prediction = ThermalContentPredictions(prepared, native[mode], work=output.work_counts)
            terms = provider._role_losses(prediction, batch.targets)
            losses[mode] = {name: _finite_scalar(value.detach().mean()) for name, value in terms.items()}
            term = sum(float(provider.native_weights[name]) * value.mean() for name, value in terms.items())
            objective = 0.5 * term if objective is None else objective + 0.5 * term

        organization = prepared.organization
        targets = {
            "applied_EM": organization.anchored_applied_EM,
            "applied_EE": organization.anchored_applied_EE,
            "applied_ME": organization.mediated_applied_ME,
            "member_values_M": organization.member_values_M,
            "member_values_E": organization.member_values_E,
        }
        if model.core.mediated_mode == "G":
            targets["geometry_conditioning"] = organization.mediated_ee_geometry
        modules = {
            "module_source_scorer": model.core.member_router.member_M,
            "environment_scorer": model.core.member_router.member_E,
            "module_value_strength": model.core.typed_read_gates["M"],
            "environment_value_strength": model.core.typed_read_gates["E"],
        }
        named = [(name, tensor) for name, tensor in targets.items() if tensor.requires_grad]
        named += [(f"{name}.{index}", tensor) for name, module in modules.items()
                  for index, tensor in enumerate(module.parameters()) if tensor.requires_grad]
        gradients = torch.autograd.grad(objective, [tensor for _, tensor in named], allow_unused=True)
        paths = {
            name: {"available": gradient is not None,
                   "l2": 0.0 if gradient is None else _finite_scalar(gradient.detach().double().norm())}
            for (name, _), gradient in zip(named, gradients)
        }
        # Empty or normalized singleton routes can have exactly zero adjoints;
        # record that fact without imposing a nonzero-gradient admission rule.
        trace_queries = select_query_rows_v1(
            prepared.query_batch, torch.arange(min(8, prepared.query_batch.num_queries), device=provider.device),
        )
        summaries, modes = {}, {}
        with torch.no_grad():
            repeat_prepared = provider._prepare_for_probe(model, batch)
            heat_batch = replace(batch, receivers=replace(batch.receivers, heat=batch.receivers.heat * 1.1))
            heat_prepared = provider._prepare_for_probe(model, heat_batch)
            for mode in outputs:
                trace_output = model.core.decode(organization, trace_queries,
                                                 intervention={"read_mode": mode}, return_trace=True)
                trace = trace_output.trace
                summaries[mode] = {}
                for name in ("mediated_source_alpha", "mediated_environment_prior",
                             "anchor_read_strength_M", "anchor_read_strength_E"):
                    value = trace.get(name)
                    if torch.is_tensor(value) and value.numel():
                        summaries[mode][name] = {"shape": list(value.shape),
                                                "mean": _finite_scalar(value.mean()),
                                                "min": _finite_scalar(value.min()),
                                                "max": _finite_scalar(value.max())}
                alpha = trace["mediated_source_alpha"]
                selected = trace["mediated_source_selected"]
                reads = trace["anchor_read_indices"]
                from honf_forward_core.interaction_reused_member_routing.anchored import build_e_access_measure
                scene = prepared.scene_batch
                m = scene.num_modules
                mu_e = build_e_access_measure(
                    scene.module_mask, scene.environment_mask, scene.environment_measure,
                    dtype=alpha.dtype, access_measure=model.core.access_measure,
                )[:, m:]
                row_mass = (organization.mediated_applied_ME * mu_e[:, None, :]).sum(-1)
                empty_mass = (alpha[:, 1:] * (row_mass[reads[:, 0]] == 0)).sum(-1)
                member = trace["anchor_read_member_index"]
                rid = trace["anchor_read_member_read_index"]
                own = member == m + trace["anchor_read_E_slots"][rid]
                own_beta = alpha.new_zeros(alpha.shape[0]).index_add(
                    0, rid[own], trace["anchor_read_probabilities"][own]
                    * trace["anchor_read_strength"][rid[own], 1],
                )
                summaries[mode].update(
                    source_alpha0_mean=_finite_scalar(alpha[:, 0].mean()),
                    final_own_E_beta_mean=_finite_scalar(own_beta.mean()),
                    source_alpha_mass_retained_mean=_finite_scalar((alpha * selected).sum(-1).mean()),
                    zero_selected_M_fraction=_finite_scalar((~selected[:, 1:].any(-1)).float().mean()),
                    source_mass_on_empty_ME_rows_mean=_finite_scalar(empty_mass.mean()),
                    raw_ME_positive_row_mass_min=(
                        _finite_scalar(row_mass[row_mass > 0].min()) if bool((row_mass > 0).any()) else 0.0),
                    raw_ME_positive_row_mass_mean=(
                        _finite_scalar(row_mass[row_mass > 0].mean()) if bool((row_mass > 0).any()) else 0.0),
                )
                repeated = model.apply_native(repeat_prepared, intervention={"read_mode": mode})
                heated = model.apply_native(heat_prepared, intervention={"read_mode": mode})
                modes[mode] = {
                    "heat_plus10_fieldwise_native_delta": _native_field_delta(native[mode], heated, model, prepared),
                    "repeatability_fieldwise_native_delta": _native_field_delta(native[mode], repeated, model, prepared),
                }
        return {"case_id": str(case_id), "module_count": int(prepared.scene_batch.module_mask.sum()),
                "native_fieldwise_normalized_mse": losses, "native_objective_gradient_paths": paths,
                "trace": summaries, "modes": modes,
                "paired_shared_work_counts": dict(paired[2])}

    def observe(self, model, *, training_epoch):
        rng = capture_rng_state()
        provider_state = copy.deepcopy(self.provider.training_state_dict())
        training_mode = bool(model.training)
        clock = (model.core._joint_epoch, model.core._joint_update_index)
        model.eval()
        try:
            with torch.enable_grad():
                cases = [self._observe_case(model, case_id) for case_id in self.panel_case_ids]
        finally:
            model.train(training_mode)
            model.core._joint_epoch, model.core._joint_update_index = clock
            self.provider.load_training_state_dict(provider_state)
            restore_rng_state(rng)
        return {"schema": "thermal_r13_12_mediated_health_v1", "training_epoch": int(training_epoch),
                "scope": "Fixed TRAIN panel; complete native paired objective; eight traced queries per case",
                "case_ids": list(self.panel_case_ids), "cases": cases,
                "mode_restored": model.training == training_mode,
                "interpretation": "Live native adjoints, source/member statistics and prescribed heat sensitivity; zero adjoints on empty or normalized singleton paths are reported separately."}
