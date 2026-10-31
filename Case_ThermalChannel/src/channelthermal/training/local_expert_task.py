"""Case-owned organized conditioning of the frozen Thermal local operator."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import LossTerm
from ..local_ports import FrozenThermalLocal, fit_port_statistics
from ..native_observations import interpolate_native, _column
from .content_hypergraph_task import ThermalContentPredictions
from .joint_member_routing_task import ThermalJointPairPredictions
from .local_ports_task import ThermalLocalPortsTask
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA,
                                  DETAIL_SOURCE_KIND, _VARIANTS, ThermalR138AnchoredMemberRoutingTask)


class OrganizedTemperatureHead(nn.Module):
    """Predict a temperature correction from restricted organized reads and inputs."""
    def __init__(self, context_dim, temperature_scale, hidden_dim=64):
        super().__init__()
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))
        # 4 fields, 2 center, 2 domain, 6 material, heat, 3 angular features.
        self.net = nn.Sequential(nn.Linear(2 * context_dim + 18, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, contexts, fields, structure, heat):
        centers = structure["module_centers"]
        n = fields.shape[-2]
        theta = torch.arange(n, dtype=fields.dtype, device=fields.device) * (2 * torch.pi / n)
        angular = torch.stack((theta, theta.cos(), theta.sin()), -1)
        angular = angular.view(1, 1, n, 3).expand(*fields.shape[:-1], 3)
        lengths = torch.cat((_column(structure, "domain_length_x", centers, 12.0),
                             _column(structure, "domain_length_y", centers, 6.0)), -1)
        prescribed = torch.cat((centers, lengths[:, None].expand_as(centers),
                                structure["material_params"][:, None].expand(-1, centers.shape[1], -1),
                                heat[..., None]), -1)
        prescribed = torch.sign(prescribed) * torch.log1p(prescribed.abs())
        prescribed = prescribed[..., None, :].expand(*fields.shape[:-1], prescribed.shape[-1])
        x = torch.cat((*contexts, fields, prescribed, angular), -1)
        temperature = fields[..., 3] + self.temperature_scale.to(fields) * self.net(x)[..., 0]
        return temperature, angular


def prescribed_h(structure, nx, ny):
    centers = structure["module_centers"]
    lengths = torch.cat((_column(structure, "domain_length_x", centers, 12.0),
                         _column(structure, "domain_length_y", centers, 6.0)), -1)
    material = structure["material_params"]
    radius = material[:, 5:6]
    delta = torch.minimum((lengths / lengths.new_tensor([nx, ny])).amin(-1, keepdim=True), .15 * radius)
    solid, fluid = material[:, 3:4], material[:, 4:5]
    if bool(((solid <= 0) | (fluid <= 0) | (delta <= 0)).any()):
        raise ValueError("Prescribed conductivity and outside offset must be positive.")
    return (2 * solid * fluid / (solid + fluid)) / delta


def organized_features(adapter, prepared, *, frozen):
    """Interpolate live restricted reads using the unchanged native outside union."""
    adapter.validate_prepared(prepared)
    query, union, stencils, *rest = adapter._query_union(prepared.scene_batch, prepared.fluid_xy[:, :1], None, 64)
    pieces, counts = [], {}
    for start in range(0, query.num_queries, adapter.core.query_tile_size):
        tile = slice_query_v1(query, start, min(start + adapter.core.query_tile_size, query.num_queries))
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            out = adapter.core.decode(prepared.organization, tile, intervention={"read_mode": "restricted"},
                                      return_trace=True)
            raw = out.joint_fields
            indices = [0, 1, 2, 4]
            physical = raw * adapter.field_std[indices].to(raw) + adapter.field_mean[indices].to(raw)
            pieces.append(torch.cat((out.trace["context_M"], out.trace["context_E"], physical), -1))
            for name, value in out.work_counts.items():
                counts[name] = counts.get(name, 0) + int(value)
    values = torch.cat(pieces, 1)
    observed = interpolate_native(values, stencils["outside"])
    shape = rest[-1]["outside"]
    observed = observed.reshape(prepared.source_present.shape[0], *shape, observed.shape[-1])
    valid = stencils["outside"].valid.reshape(observed.shape[:-1])
    d = adapter.core.context_dim
    return (observed[..., :d], observed[..., d:2*d]), observed[..., 2*d:], valid, int(union.unique_counts.sum()), counts


class ThermalLocalExpertTask(ThermalR138AnchoredMemberRoutingTask):
    _with_port_targets = ThermalLocalPortsTask._with_port_targets
    def make_batch(self, case_keys, key):
        return self._with_port_targets(super().make_batch(case_keys, key))

    def validation_batches(self):
        return iter(self._with_port_targets(batch) for batch in super().validation_batches())

    def _native_case_batch(self, cases, indices, *, split, ntheta=64):
        return self._with_port_targets(super()._native_case_batch(cases, indices, split=split, ntheta=ntheta))

    _port_case_counts = staticmethod(ThermalLocalPortsTask._port_case_counts)

    def configure_expert(self, recipe):
        self.recipe = copy.deepcopy(recipe)
        spec = recipe["local_expert"]
        self.phase = spec["phase"]
        if self.phase not in ("frozen", "joint"):
            raise ValueError("Unknown local expert phase.")
        self.local_weight = float(spec["local_weight"])
        self.outside_weight = float(spec["outside_weight"])
        if any(not math.isfinite(v) or v < 0 for v in (self.local_weight, self.outside_weight)):
            raise ValueError("Loss coefficients must be finite and nonnegative.")
        self.port_statistics = fit_port_statistics(self.training_cases)
        self._port_cases = {str(case["case_id"]): case for case in (*self.training_cases, *self.validation_cases)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(recipe["initialization"]["seed"]) + 1)
            self.model.local_expert_head = OrganizedTemperatureHead(self.model.core.context_dim,
                                                                    self.port_statistics["std"][0]).to(self.device)
        self.local_operator = FrozenThermalLocal(spec["local_checkpoint"]).to(self.device).eval()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self.phase == "joint" or name.startswith("local_expert_head."))

    def identity_payload(self):
        result = super().identity_payload()
        result["objective"] = {"paired_native_enabled": self.phase == "joint",
                               "native_weights": copy.deepcopy(self.native_weights) if self.phase == "joint" else {},
                               "local_weight": self.local_weight, "outside_temperature_weight": self.outside_weight}
        result["local_expert"] = {**copy.deepcopy(self.recipe["local_expert"]),
                                  "statistics": self.port_statistics,
                                  "h_interpretation": "prescribed harmonic conductivity / native offset; packed effective h has epsilon, clipping and validity effects",
                                  "feature_cache": "none; no persistent or RAM feature cache",
                                  "validation_field_score": "assembled direct fluid and local material/interface"}
        return result

    def phase_reporting(self, *, arm, epoch, phase):
        result = dict(super().phase_reporting(arm=arm, epoch=epoch, phase=phase))
        result["all_joint_parameters_active"] = self.phase == "joint"
        result["local_expert_phase"] = self.phase
        result["flow_and_temperature_supervised_together"] = self.phase == "joint"
        return result

    def loss_metadata(self):
        result = super().loss_metadata() if self.phase == "joint" else {}
        for role in ("temperature/material", "temperature/surface", "q_proxy"):
            result[f"expert/{role}"] = {
                "panel": "native_prediction", "label": f"Local {role}",
                "unit": "dimensionless scaled squared error",
                "formula": "Native physical mask and scale; equal active-module means then equal case means.",
                "weight": str(self.local_weight / 3),
            }
        result["expert/outside_temperature"] = {
            "panel": "native_prediction", "label": "Local condition outside temperature",
            "unit": "dimensionless scaled squared error",
            "formula": "Present modules, valid outside receivers, finite target; equal nonempty case means.",
            "weight": str(self.outside_weight),
        }
        return result

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None, **kwargs):
        prediction, aux = super().predict_native(model, scene, receivers, execution_mode, phase, epoch, temperature, **kwargs)
        native = prediction.restricted if isinstance(prediction, ThermalJointPairPredictions) else prediction
        prepared = native.native_prepared
        contexts, fields, valid, union_count, counts = organized_features(model, prepared, frozen=self.phase == "frozen")
        t, angular = model.local_expert_head(contexts, fields, prepared.structure, receivers.heat)
        h = prescribed_h(prepared.structure, model.nx, model.ny)[:, None, :].expand_as(t)
        ports = torch.cat((angular, t[..., None], h[..., None]), -1)
        ports = ports * prepared.source_present[..., None, None].to(ports)
        local = self.local_operator(prepared.structure, receivers.heat, ports, receivers.local_query_points)
        prediction.expert = local
        prediction.expert_ports = ports
        prediction.expert_valid = valid
        prediction.expert_present = prepared.source_present
        prediction.work = dict(prediction.work or {})
        prediction.work["expert_outside_union_rows"] = union_count
        prediction.work.update({f"core_decode_expert_{name}": value for name, value in counts.items()})
        prediction.work["expert_local_calls"] = 1
        prediction.work["expert_local_active_modules"] = int(prepared.source_present.sum())
        aux["work"] = prediction.work
        return prediction, aux

    def _assembled_prediction(self, prediction, targets):
        native = prediction.restricted if isinstance(prediction, ThermalJointPairPredictions) else prediction
        output = dict(native.native_main)
        output["pred_internal_temperature"] = prediction.expert["internal_temperature"]
        n = targets.interface_target.shape[-2]
        if 64 % n:
            raise ValueError("Local interface angles must match native sampled angles.")
        output["pred_interface"] = prediction.expert["interface_pred"][..., ::64 // n, :]
        return ThermalContentPredictions(native.native_prepared, output)

    def validation_metrics(self, predictions, targets, auxiliary_state):
        return super().validation_metrics(self._assembled_prediction(predictions, targets), targets, auxiliary_state)

    def _expert_terms(self, prediction, targets):
        roles = self._role_losses(self._assembled_prediction(prediction, targets), targets)
        count = len(targets.case_ids)
        result = {f"expert/{name}": LossTerm(roles[name].sum(), count, self.local_weight / 3)
                  for name in ("temperature/material", "temperature/surface", "q_proxy")}
        if not torch.equal(prediction.expert_valid, targets.outside_valid):
            raise ValueError("Expert outside geometry differs from its target-only denominator.")
        target = targets.teacher_ports[..., 3].to(prediction.expert_ports)
        mask = prediction.expert_present[..., None].bool() & prediction.expert_valid & torch.isfinite(target)
        err = (prediction.expert_ports[..., 3] - torch.nan_to_num(target)) / self.port_statistics["std"][0]
        per_case = (err.square() * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
        result["expert/outside_temperature"] = LossTerm(per_case.sum(), int(mask.any((1, 2)).sum()), self.outside_weight)
        return result

    def loss_denominators(self, batches, phase="joint", arm=ARM_DETAIL):
        result = dict(super().loss_denominators(batches, phase=phase, arm=arm)) if self.phase == "joint" else {}
        count = sum(len(batch.targets.case_ids) for batch in batches)
        for role in ("temperature/material", "temperature/surface", "q_proxy"):
            result[f"expert/{role}"] = count
        outside = sum(self._port_case_counts(batch.targets, batch.scene_inputs.structure["module_present"])[0] for batch in batches)
        if outside:
            result["expert/outside_temperature"] = outside
        return result

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        result = super().loss_terms(predictions, targets, phase, auxiliary_state) if self.phase == "joint" else {}
        result.update({k: v for k, v in self._expert_terms(predictions, targets).items() if v.denominator})
        return result

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch=None, arm=None):
        result = super().validation_loss_terms(predictions, targets, auxiliary_state, batch=batch, arm=arm) if self.phase == "joint" else {}
        result.update({k: v for k, v in self._expert_terms(predictions, targets).items() if v.denominator})
        return result


def build_thermal_local_expert_task(recipe, *, arm=ARM_DETAIL, device="cpu"):
    spec = recipe["local_expert"]
    if spec.get("schema") != "thermal_organized_local_expert_v1" or recipe["training"]["membership"] != "fixed25_train150_dev22":
        raise ValueError("Local expert requires its named schema and TRAIN150 membership.")
    base = copy.deepcopy(recipe)
    base["initialization"].update(source_kind=DETAIL_SOURCE_KIND, source_checkpoint=None,
                                  pretrained_weights=False, source_arm=f"scratch_common_seed{base['initialization']['seed']}",
                                  source_parent_age=0)
    base["member_routing"]["health"] = {"enabled": False}
    model, provider = _build_thermal_member_routing_task(base, arm=arm, device=device,
        task_type=ThermalLocalExpertTask, variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], "arm": arm}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=DETAIL_SOURCE_KIND, lineage="local expert", required_address_mode=None)
    source = load_trusted_checkpoint(spec["source_checkpoint"], map_location="cpu")
    old_recipe = source["experiment_identity"]["recipe"]
    if (old_recipe["dataset"] != recipe["dataset"] or old_recipe["training"]["membership"] != "fixed25_train150_dev22"
            or old_recipe["member_routing"]["model"] != recipe["member_routing"]["model"]):
        raise ValueError("Transfer source must use the exact TRAIN150 model and membership.")
    expected = 100 if spec["phase"] == "frozen" else 50
    if int(source["epoch"]) != expected:
        raise ValueError("Transfer source has the wrong completed epoch.")
    for key in ("field_mean", "field_std", "interface_target_std", "internal_temperature_std"):
        if not torch.equal(model.state_dict()[key].cpu(), source["model_state_dict"][key].cpu()):
            raise ValueError("Source TRAIN150 normalization differs from the current data binding.")
    if spec["phase"] == "frozen":
        if (old_recipe.get("name") != "thermal_r15_2_joint_train150_v1"
                or old_recipe.get("local_expert") or old_recipe.get("local_ports", {}).get("enabled")):
            raise ValueError("Frozen transfer requires the direct segmented T0 source.")
        model.load_state_dict(source["model_state_dict"], strict=True)
        provider.configure_expert(recipe)
    else:
        if (old_recipe.get("name") != "thermal_r15_2_h_frozen_train150_v1"
                or old_recipe.get("local_expert", {}).get("phase") != "frozen"
                or old_recipe["local_expert"]["local_checkpoint"] != spec["local_checkpoint"]):
            raise ValueError("Joint transfer requires H-Frozen endpoint.")
        provider.configure_expert(recipe)
        model.load_state_dict(source["model_state_dict"], strict=True)
    return model, provider
