"""Shared native and frozen-local loss mechanics for Thermal port fitting."""
import torch
from honf_runtime.unified_training import LossTerm
from .content_hypergraph_task import ThermalContentPredictions
from .joint_member_routing_task import ThermalJointPairPredictions
from .r13_8_anchored_task import ThermalR138AnchoredMemberRoutingTask


class ThermalPortBaseTask(ThermalR138AnchoredMemberRoutingTask):
    def validation_metrics(self, predictions, targets, auxiliary_state):
        # Assemble joint fluid with final local material/interface values.
        # Paired training predictions use the restricted member.
        native = predictions.restricted if isinstance(predictions, ThermalJointPairPredictions) else predictions
        output = dict(native.native_main)
        mode = "restricted" if isinstance(predictions, ThermalJointPairPredictions) else getattr(predictions, "read_mode", "restricted")
        local = predictions.local1[mode]
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
        modes = ("full", "restricted") if paired else (getattr(prediction, "read_mode", "restricted"),)
        per_mode = []
        for mode in modes:
            final = getattr(prediction, mode) if paired else prediction
            self._check_native_finite(final, targets, label=f"{mode} final direct")
            native = self._role_losses(final, targets)
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
            per_mode.append((native, local_roles, sum(consistency) / 2))
        result = {}
        cases = len(targets.case_ids)
        for name, weight in self.native_weights.items():
            result[f"final/{name}"] = LossTerm(sum(row[0][name].sum() for row in per_mode) / len(modes), cases, float(weight))
        for name in ("temperature/material", "temperature/surface", "q_proxy"):
            result[f"local/{name}"] = LossTerm(sum(row[1][name].sum() for row in per_mode) / len(modes), cases, .1 / 3)
        result["interface/consistency"] = LossTerm(sum(row[2].sum() for row in per_mode) / len(modes), cases, .05)
        return result

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        return self._terms(predictions, targets)

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch=None, arm=None):
        return self._terms(predictions, targets)

    def loss_metadata(self):
        terms = {}
        for stage, factor in (("final", 1),):
            for role, weight in self.native_weights.items():
                terms[f"{stage}/{role}"] = {"panel": "native_prediction", "label": f"{stage} {role}", "unit": "scaled squared error", "formula": "Paired full/restricted TRAIN; restricted validation, native reference masks and case reductions.", "weight": str(factor * weight)}
        for role in ("temperature/material", "temperature/surface", "q_proxy"):
            terms[f"local/{role}"] = {"panel": "native_prediction", "label": f"local {role}", "unit": "scaled squared error", "formula": "Frozen local output, native physical masks and equal module/case reductions.", "weight": str(.1 / 3)}
        terms["interface/consistency"] = {"panel": "native_prediction", "label": "Direct/local interface consistency", "unit": "scaled squared error", "formula": "Equal surface T and q-proxy on aligned native reference masks.", "weight": "0.05"}
        return terms
