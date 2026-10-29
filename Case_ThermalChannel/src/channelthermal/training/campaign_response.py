"""One train-family finite-response pair per complete native epoch after e100."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import EvidenceSplit
from channelthermal.response_control.contracts import DesignInput, role_queries_from_stencil
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.sampling import ReceiverSamplingConfig, sample_training_stencil
from channelthermal.response_control.thermal import pressure_drop_from_field
from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward
from honf_runtime.paths import resolve_path

from .campaign import HYPERGRAPH_ARCHITECTURES, heat_null_training_enabled
from .campaign_work import CampaignForwardWork, measured_wrapper_calls


class _ResponseCall(nn.Module):
    def __init__(self, model, template, dataset_config, stats):
        super().__init__()
        self.model = model
        self.template, self.dataset_config, self.stats = template, dataset_config, stats

    @property
    def core(self):
        return self.model.core

    def forward(self, design, context, queries):
        operator = DifferentiableThermalOperator(self.model, self.template, dataset_config=self.dataset_config,
                                                normalization_stats=self.stats, query_batch_size=1024)
        return dict(operator(design, context, queries).role_values)


class NativeCampaignResponse:
    """Cycle families and perturbations with a fixed, train-only scale basis."""

    def __init__(self, model, dataset, dataset_config, settings: dict[str, Any]):
        paths = [resolve_path(path) for path in settings.get("response_stencils", [])]
        if not paths:
            raise ValueError("Campaign response exposure requires established train-family stencils.")
        loaded = [load_response_atlas_stencil(Path(path)) for path in paths]
        stencils = [stencil for stencil, _metadata in loaded]
        if any(stencil.split is not EvidenceSplit.TRAIN for stencil in stencils):
            raise ValueError("Campaign auxiliary response cannot consume development/final-review references.")
        if len({stencil.physical_family_id for stencil in stencils}) != len(stencils):
            raise ValueError("Campaign response families must be unique.")
        development = dataset_config.get("development_subset")
        if development is not None:
            from channelthermal.data.development_split import development_case_ids

            selected = set(development_case_ids(development, "train"))
            if len(stencils) > 4 or any(metadata.get("anchor_id") not in selected for _stencil, metadata in loaded):
                raise ValueError("Development response exposure requires at most four selected training anchors.")
        self.model, self.stencils = model, stencils
        self.stats = dataset.normalizer.stats
        self.wrapper = _ResponseCall(model, dataset[0], dataset_config, self.stats)
        self.sampling = ReceiverSamplingConfig(max_fluid_queries=1024, solid_queries_per_module=128,
                                               hot_solid_points_per_module=16, random_seed=0)
        self.settings = settings
        self.heat_null = None
        if heat_null_training_enabled(settings):
            from .campaign_null_response import NativeHeatNullResponse

            self.heat_null = NativeHeatNullResponse(model, dataset, dataset_config, settings["heat_null_response"],
                organizer_gradient_policy=settings.get("organizer_gradient_policy", "whole_wrapper_shadow_v1"))

    def __call__(self, epoch: int, native_loss: torch.Tensor, accumulation_weight: float) -> tuple[torch.Tensor, dict]:
        started = time.perf_counter()
        index = epoch - 101
        stencil = self.stencils[index % len(self.stencils)]
        names = sorted(stencil.variants)
        label = names[(index // len(self.stencils)) % len(names)]
        pair = ResponseStencil(stencil.baseline, {label: stencil.variants[label]})
        sampled = sample_training_stencil(pair, config=self.sampling)
        pair = sampled.stencil
        device = native_loss.device
        queries = role_queries_from_stencil(pair, device=device)
        if self.model.config.core_honf.forward_architecture in HYPERGRAPH_ARCHITECTURES:
            def call(model, *args):
                return hard_value_soft_hypergraph_forward(model, *args,
                    gradient_policy=self.settings.get("organizer_gradient_policy", "whole_wrapper_shadow_v1"))
        else:
            call = lambda model, *args: model(*args)
        context = dict(pair.baseline.context.values)
        base_design = DesignInput.from_state(pair.baseline.design, device=device)
        trial_design = DesignInput.from_state(pair.variants[label].design, device=device)
        with CampaignForwardWork(self.model.core) as measured_work:
            base = call(self.wrapper, base_design, context, queries)
            trial = call(self.wrapper, trial_design, context, queries)
        role_scales = {
            "fluid_fields": np.asarray(self.stats["field_std_by_channel"]),
            "interface": np.asarray(self.stats.get("interface_targets_std", self.stats.get("interface_target_std"))),
            "solid_temperature": np.asarray(self.stats["internal_temperature_std"]),
        }
        terms = []
        for role in queries:
            base_role, trial_role = pair.baseline.output.roles[role], pair.variants[label].output.roles[role]
            target = torch.as_tensor(np.array(trial_role.values - base_role.values, copy=True), device=device, dtype=native_loss.dtype)
            valid = torch.as_tensor(np.array(base_role.valid_mask & trial_role.valid_mask, copy=True), device=device)
            scale = torch.as_tensor(role_scales[role], device=device, dtype=native_loss.dtype).reshape(1, -1).clamp_min(1e-8)
            weight = torch.as_tensor(np.array(base_role.quadrature_weights, copy=True), device=device, dtype=native_loss.dtype)[:, None] * valid
            error = ((trial[role] - base[role] - target) / scale).square()
            terms.append((error * weight).sum() / weight.sum().clamp_min(1))
        pindex = queries["fluid_fields"].channel_names.index("p")
        pressure = pressure_drop_from_field(trial["fluid_fields"], queries["fluid_fields"], trial_design, context) - pressure_drop_from_field(base["fluid_fields"], queries["fluid_fields"], base_design, context)
        pressure_target = pair.variants[label].output.quantities["pressure_drop"].value - pair.baseline.output.quantities["pressure_drop"].value
        terms.append(((pressure - pressure_target) / float(role_scales["fluid_fields"].reshape(-1)[pindex])) ** 2)
        for slot, module_id in enumerate(pair.baseline.output.active_module_ids):
            rows = torch.tensor([i for i, value in enumerate(queries["solid_temperature"].receiver_slots) if value == slot], device=device)
            delta = trial["solid_temperature"][rows].max() - base["solid_temperature"][rows].max()
            reference = pair.variants[label].output.module_peak_temperature[module_id] - pair.baseline.output.module_peak_temperature[module_id]
            terms.append(((delta - reference) / float(role_scales["solid_temperature"].reshape(-1)[0])) ** 2)
        response_loss = torch.stack(terms).mean()
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        state = getattr(self.model, "campaign_training_state", {})
        if len(state.get("response_scale_samples", [])) < 5:
            native_gradients = torch.autograd.grad(native_loss * accumulation_weight, parameters, retain_graph=True, allow_unused=True)
            response_gradients = torch.autograd.grad(response_loss, parameters, retain_graph=True, allow_unused=True)
            native_norm = sum(float(((torch.zeros_like(parameter) if value is None else value.detach()) + (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach())).square().sum())
                              for parameter, value in zip(parameters, native_gradients)) ** .5
            response_norm = sum(float(value.detach().square().sum()) for value in response_gradients if value is not None) ** .5
            coefficient = min(.05 * native_norm / response_norm, .1) if response_norm > 1e-10 and native_norm > 1e-10 else 0.
            state.setdefault("response_scale_samples", []).append(coefficient)
            state["response_scale"] = float(statistics.median(state["response_scale_samples"]))
            state["response_native_gradient_norm"] = native_norm
            state["response_gradient_norm"] = response_norm
            self.model.campaign_training_state = state
        ramp = min(max((epoch - 100) / 100, 0), 1)
        coefficient = state.get("response_scale", 0.) * ramp
        total_loss = response_loss * coefficient / accumulation_weight
        wrapper_calls = measured_wrapper_calls(measured_work.records)
        whole_shadow = (self.model.config.core_honf.forward_architecture in HYPERGRAPH_ARCHITECTURES
                        and self.settings.get("organizer_gradient_policy", "whole_wrapper_shadow_v1") == "whole_wrapper_shadow_v1")
        metrics = {
            "response_family": pair.physical_family_id, "response_variant": label,
            "response_loss": float(response_loss.detach()), "response_coefficient": coefficient,
            "response_examples": 2, "response_wrapper_calls": wrapper_calls if wrapper_calls is not None else 2 * (2 if whole_shadow else 1),
            "response_wrapper_count_basis": "measured native P0 preparations" if wrapper_calls is not None else "named organizer gradient policy; unlabelled backend",
            "response_queries": 2 * sum(int(query.query_features.shape[0]) for query in queries.values()),
            "response_seconds": time.perf_counter() - started,
            "response_forward_work": measured_work.records,
            "response_work_scope": "actual forward calls including prepared query chunks; backward/recompute excluded",
            "response_reference": "stored analytic/shared-grid generator; q_normal proxy; absolute training scales",
        }
        if self.heat_null is not None:
            null_loss, null_metrics = self.heat_null(epoch, native_loss, accumulation_weight)
            total_loss = total_loss + null_loss
            metrics.update(null_metrics)
        return total_loss, metrics
