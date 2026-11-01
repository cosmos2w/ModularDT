"""R15-3 paired native training through a finite local-response update."""
from __future__ import annotations

import copy

import torch

from honf_forward_core.interaction_reused_member_routing import anchored_core_from_config
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import LossTerm
from ..local_ports import FrozenThermalLocal
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from ..thermal_coupling import local_response, outside_geometry, prepare_response, response_scene, transplant_widened_model
from .content_hypergraph_task import ThermalContentPredictions
from .joint_member_routing_task import ThermalJointPairPredictions
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA,
                                 DETAIL_SOURCE_KIND, _VARIANTS, ThermalR138AnchoredMemberRoutingTask)


class ThermalFeedbackAdapter(ThermalDetailAnchoredMemberRoutingAdapter):
    SCHEMA = "thermal_response_feedback_adapter_v1"

    def scene_batch(self, structure, *, scene_key=None):
        return response_scene(super().scene_batch(structure, scene_key=scene_key), None, None, None)


class ThermalResponseFeedbackTask(ThermalR138AnchoredMemberRoutingTask):
    def configure_feedback(self, recipe):
        self.recipe = copy.deepcopy(recipe)
        self.feedback = bool(recipe["thermal_feedback"]["return_response"])
        self.source_kind = recipe["initialization"]["source_kind"]
        self.source_parent_age = int(recipe["thermal_feedback"]["source_epoch"])
        self.local_operator = FrozenThermalLocal(recipe["thermal_feedback"]["local_checkpoint"]).to(self.device).eval()
        self.response_mean = self.model.normalization_stats["interface_targets_mean"]
        self.response_scale = self.model.normalization_stats["interface_targets_std"]

    def identity_payload(self):
        result = super().identity_payload()
        result["training_reader_objective"] = ("paired branch-consistent final native + 0.25 initial native + "
            "0.1 final local roles + 0.05 direct/local interface consistency")
        result["thermal_feedback"] = {**self.recipe["thermal_feedback"], "response_mean": self.response_mean,
            "response_scale": self.response_scale, "response_fit_scope": "declared TRAIN membership",
            "angles": 64, "ordering": "surface T[64], signed native q-proxy[64], present",
            "local_asset_exposure": "TRAIN600/DEV22/VAL89", "preparations": "two finite stages; response 1 is not fed back"}
        return result

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None, **kwargs):
        del phase, epoch, temperature
        if kwargs or model is not self.model or execution_mode != model.architecture:
            raise ValueError("Unexpected Thermal response prediction request.")
        first = model.prepare_native(scene.structure, receivers.fluid_xy, physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points, ntheta=receivers.ntheta, scene_key=scene.case_ids)
        geometry = outside_geometry(model, first)
        modes = ("full", "restricted") if model.training else ("restricted",)
        initial, final, local0, local1, work = {}, {}, {}, {}, {"preparations": 1, "local_calls": 0, "port_decodes": 0, "port_geometry_builds": 1}
        work.update({f"stage0_prepare_{k}": int(v) for k, v in first.organization.work_counts.items()})
        for mode in modes:
            with model.core.read_execution(mode):
                native, counts, _ = model.apply_native_with_work_counts(first, receivers.heat)
            initial[mode] = ThermalContentPredictions(first, native, work=counts)
            work.update({f"stage0_{mode}_decode_{k}": int(v) for k, v in counts.items()})
            local0[mode], counts = local_response(model, self.local_operator, first, receivers.heat, receivers.local_query_points, mode, geometry)
            work["local_calls"] += 1
            work["port_decodes"] += 1
            work.update({f"stage0_{mode}_{k}": int(v) for k, v in counts.items()})
            if self.feedback:
                second = prepare_response(model, first, local0[mode]["interface_pred"], self.response_mean, self.response_scale)
                work["preparations"] += 1
                work.update({f"stage1_{mode}_prepare_{k}": int(v) for k, v in second.organization.work_counts.items()})
            else:
                second = first
            if self.feedback:
                with model.core.read_execution(mode):
                    native, counts, _ = model.apply_native_with_work_counts(second, receivers.heat)
                final[mode] = ThermalContentPredictions(second, native, work=counts)
                local1[mode], port_counts = local_response(model, self.local_operator, second, receivers.heat, receivers.local_query_points, mode, geometry)
                work["local_calls"] += 1
                work["port_decodes"] += 1
                work.update({f"stage1_{mode}_{k}": int(v) for k, v in port_counts.items()})
            else:
                final[mode], local1[mode] = initial[mode], local0[mode]
            if self.feedback:
                work.update({f"stage1_{mode}_decode_{k}": int(v) for k, v in final[mode].work.items()})
            else:
                work[f"stage1_{mode}_reused_stage0"] = 1
        def paired(items):
            return ThermalJointPairPredictions(items["full"], items["restricted"], work) if model.training else items["restricted"]
        prediction = paired(final)
        prediction.initial = paired(initial)
        prediction.local0 = local0
        prediction.local1 = local1
        prediction.work = work
        return prediction, {"work": work}

    def validation_metrics(self, predictions, targets, auxiliary_state):
        # Report the assembled restricted field: joint fluid with final local
        # material/interface values. Native direct losses remain separate.
        native = predictions.restricted if isinstance(predictions, ThermalJointPairPredictions) else predictions
        output = dict(native.native_main)
        local = predictions.local1["restricted"]
        count = targets.interface_target.shape[-2]
        if 64 % count:
            raise ValueError("Native interface angles are not aligned with 64 ports.")
        output["pred_interface"] = local["interface_pred"][..., ::64 // count, :]
        output["pred_internal_temperature"] = local["internal_temperature"]
        return super().validation_metrics(ThermalContentPredictions(native.native_prepared, output), targets, auxiliary_state)

    @staticmethod
    def _check_native_finite(prediction, targets, *, label):
        prepared, output = prediction.native_prepared, prediction.native_main

        def check(value, mask, role, module_axis=False):
            bad = mask & ~torch.isfinite(value)
            if bad.any():
                index = torch.nonzero(bad)[0].tolist()
                case_id = targets.case_ids[index[0]]
                module = (f", module {int(prepared.scene_batch.module_ids[index[0], index[1]])}"
                          if module_axis else "")
                raise FloatingPointError(f"{label} {role} is nonfinite for case {case_id}{module}; index {index}.")

        reference = targets.field_targets.to(output["pred_field"])
        valid = (targets.point_weights > 0) & prepared.stencils["fluid"].valid
        for channel, role in enumerate(("u", "v", "p", "omega", "fluid T")):
            check(output["pred_field"][..., channel], valid & torch.isfinite(reference[..., channel]), role)
        present = prepared.source_present[..., None]
        reference = targets.interface_target.to(output["pred_interface"])
        surface = prepared.stencils["surface"].valid.reshape(reference.shape[:-1])
        outside = prepared.stencils["outside"].valid.reshape(reference.shape[:-1])
        observed = targets.interface_valid_mask > .5
        for channel, role in enumerate(("surface T", "q proxy")):
            mask = present & surface & observed & torch.isfinite(reference[..., channel])
            if channel == 1:
                mask = mask & outside
            check(output["pred_interface"][..., channel], mask, role, True)
        target = targets.material_targets.to(output["pred_internal_temperature"])
        valid = present & prepared.stencils["material"].valid.reshape(target.shape) & torch.isfinite(target)
        check(output["pred_internal_temperature"][..., 0], valid, "material T", True)

    def _terms(self, prediction, targets):
        paired = isinstance(prediction, ThermalJointPairPredictions)
        modes = ("full", "restricted") if paired else ("restricted",)
        per_mode = []
        for mode in modes:
            final = getattr(prediction, mode) if paired else prediction
            initial = getattr(prediction.initial, mode) if paired else prediction.initial
            self._check_native_finite(final, targets, label=f"{mode} final direct")
            self._check_native_finite(initial, targets, label=f"{mode} initial direct")
            native, first = self._role_losses(final, targets), self._role_losses(initial, targets)
            local = prediction.local1[mode]
            n = targets.interface_target.shape[-2]
            if 64 % n:
                raise ValueError("Native interface angles are not aligned with 64 ports.")
            local_interface = local["interface_pred"][..., ::64 // n, :]
            assembled = dict(final.native_main)
            assembled["pred_interface"] = local_interface
            assembled["pred_internal_temperature"] = local["internal_temperature"]
            assembled_prediction = ThermalContentPredictions(final.native_prepared, assembled)
            self._check_native_finite(assembled_prediction, targets, label=f"{mode} final local")
            local_roles = self._role_losses(assembled_prediction, targets)
            prepared = final.native_prepared
            shape = local_interface.shape[:-1]
            surface = prepared.stencils["surface"].valid.reshape(shape)
            outside = prepared.stencils["outside"].valid.reshape(shape)
            reference = targets.interface_target.to(local_interface)
            masks = prepared.source_present[..., None] & surface & (targets.interface_valid_mask > .5)
            consistency = []
            for channel in range(2):
                mask = masks & torch.isfinite(reference[..., channel])
                if channel == 1:
                    mask = mask & outside
                diff = (final.native_main["pred_interface"][..., channel] - local_interface[..., channel]) / self.model.interface_target_std[channel]
                if not torch.isfinite(diff[mask]).all():
                    raise ValueError("Nonfinite interface prediction on native reference mask.")
                count = mask.sum(-1)
                module = (torch.where(mask, diff, torch.zeros_like(diff)).square().sum(-1) / count.clamp_min(1))
                active = count > 0
                consistency.append((module * active).sum(-1) / active.sum(-1).clamp_min(1))
            per_mode.append((native, first, local_roles, sum(consistency) / 2))
        result = {}
        cases = len(targets.case_ids)
        for name, weight in self.native_weights.items():
            result[f"final/{name}"] = LossTerm(sum(row[0][name].sum() for row in per_mode) / len(modes), cases, float(weight))
            result[f"initial/{name}"] = LossTerm(sum(row[1][name].sum() for row in per_mode) / len(modes), cases, .25 * float(weight))
        for name in ("temperature/material", "temperature/surface", "q_proxy"):
            result[f"local/{name}"] = LossTerm(sum(row[2][name].sum() for row in per_mode) / len(modes), cases, .1 / 3)
        result["interface/consistency"] = LossTerm(sum(row[3].sum() for row in per_mode) / len(modes), cases, .05)
        return result

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        return self._terms(predictions, targets)

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch=None, arm=None):
        return self._terms(predictions, targets)

    def loss_denominators(self, batches, phase="joint", arm=ARM_DETAIL):
        count = sum(len(batch.targets.case_ids) for batch in batches)
        return {**{f"{stage}/{role}": count for stage in ("final", "initial") for role in self.native_weights},
                **{f"local/{role}": count for role in ("temperature/material", "temperature/surface", "q_proxy")},
                "interface/consistency": count}

    def loss_metadata(self):
        terms = {}
        for stage, factor in (("final", 1), ("initial", .25)):
            for role, weight in self.native_weights.items():
                terms[f"{stage}/{role}"] = {"panel": "native_prediction", "label": f"{stage} {role}", "unit": "scaled squared error", "formula": "Paired full/restricted TRAIN; restricted validation, native reference masks and case reductions.", "weight": str(factor * weight)}
        for role in ("temperature/material", "temperature/surface", "q_proxy"):
            terms[f"local/{role}"] = {"panel": "native_prediction", "label": f"local {role}", "unit": "scaled squared error", "formula": "Frozen local output, native physical masks and equal module/case reductions.", "weight": str(.1 / 3)}
        terms["interface/consistency"] = {"panel": "native_prediction", "label": "Direct/local interface consistency", "unit": "scaled squared error", "formula": "Equal surface T and q-proxy on aligned native reference masks.", "weight": "0.05"}
        return terms


def build_thermal_response_feedback_task(recipe, *, arm=ARM_DETAIL, device="cpu"):
    spec = recipe["thermal_feedback"]
    if spec.get("schema") != "thermal_response_feedback_v1" or spec.get("enabled") is not True or type(spec.get("return_response")) is not bool:
        raise ValueError("Expected the R15-3 Thermal feedback recipe.")
    source = load_trusted_checkpoint(spec["source_checkpoint"], map_location="cpu")
    if int(source["epoch"]) != int(spec["source_epoch"]):
        raise ValueError("Thermal source epoch does not match the declared transfer.")
    base = copy.deepcopy(recipe)
    base["initialization"].update(source_kind=DETAIL_SOURCE_KIND, source_checkpoint=None, pretrained_weights=False, source_parent_age=0, source_arm=f"scratch_common_seed{base['initialization']['seed']}")
    base["member_routing"]["health"] = {"enabled": False}
    variants = {DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], "adapter": ThermalFeedbackAdapter, "arm": arm}}
    model, provider = _build_thermal_member_routing_task(base, arm=arm, device=device,
        task_type=ThermalResponseFeedbackTask, variants=variants, section_schema=DETAIL_CONFIG_SCHEMA,
        source_kind=DETAIL_SOURCE_KIND, lineage="Thermal response feedback", required_address_mode=None)
    source_recipe = source["experiment_identity"]["recipe"]
    if source_recipe["training"]["membership"] != recipe["training"]["membership"] or any(source_recipe["dataset"][key] != recipe["dataset"][key] for key in ("name", "data_path", "manifest_path")):
        raise ValueError("Transfer source dataset or TRAIN membership differs from the declared phase.")
    source_config = copy.deepcopy(source_recipe["member_routing"]["model"])
    requested_config = copy.deepcopy(recipe["member_routing"]["model"])
    # Query tiling changes execution only and does not change saved tensors.
    source_config.pop("query_tile_size", None)
    requested_config.pop("query_tile_size", None)
    if source_config != requested_config:
        raise ValueError("Transfer source core configuration differs from the requested core.")
    source_adapter = source["experiment_identity"]["provider_identity"]["adapter_state"]
    if source_adapter["normalization_stats"] != model.normalization_stats or float(source_adapter["heat_scale"]) != model.heat_scale:
        raise ValueError("Source TRAIN transforms differ from the phase data binding.")
    original = source["model_state_dict"]
    for key in ("field_mean", "field_std", "interface_target_std", "internal_temperature_std"):
        if not torch.equal(model.state_dict()[key].cpu(), original[key].cpu()):
            raise ValueError(f"Source and declared TRAIN transform differ: {key}.")
    config = copy.deepcopy(model.core.get_config())
    if config["base_model_config"]["module_input_dim"] != 9 or config["base_model_config"]["coordinate_dim"] != 2:
        raise ValueError("Source must declare the 9-feature two-dimensional core.")
    config["base_model_config"]["module_input_dim"] = 138
    model.core = anchored_core_from_config(config).to(device)
    transplant_widened_model(model, original)
    provider.configure_feedback(recipe)
    return model, provider
