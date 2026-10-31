"""R15-2 Thermal paired native training with supervised organized ports."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from honf_runtime.unified_training import LossTerm
from ..local_coupling import build_local_module_params_from_global
from ..local_ports import FrozenThermalLocal, ThermalPortHead, fit_port_statistics, organized_outside64, port_loss
from ..native_observations import _column, native_cell_center_stencil
from .content_hypergraph_task import ThermalContentTargets
from .joint_member_routing_task import ThermalJointPairPredictions
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA,
                                 DETAIL_SOURCE_KIND, _VARIANTS,
                                 ThermalR138AnchoredMemberRoutingTask)


@dataclass(frozen=True)
class ThermalPortTargets(ThermalContentTargets):
    teacher_ports: torch.Tensor | None = None
    teacher_h_valid: torch.Tensor | None = None
    outside_valid: torch.Tensor | None = None


class ThermalLocalPortsTask(ThermalR138AnchoredMemberRoutingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if float(self.recipe.get("local_ports", {}).get("lambda_port", 1.0)) != 1.0:
            raise ValueError("R15-2 uses lambda_port=1 for its initial comparison.")
        statistics = fit_port_statistics(self.training_cases)
        # Core construction already consumed its own forked seed. Head creation
        # has a separate stream and cannot alter either core or sampler streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.recipe["initialization"].get("seed", 0)) + 1)
            self.model.port_head = ThermalPortHead(statistics).to(self.device)
        self.port_statistics = statistics
        self._port_cases = {str(case["case_id"]): case
                            for case in (*self.training_cases, *self.validation_cases)}

    def _with_port_targets(self, batch):
        width = batch.scene_inputs.structure["module_present"].shape[1]
        cases = [self._port_cases[case_id] for case_id in batch.targets.case_ids]
        ports = torch.as_tensor(np.stack([case["teacher_port_tokens"][:width] for case in cases]),
                                device=self.device)
        valid = torch.as_tensor(np.stack([case["interface_condition_valid_mask"][:width] for case in cases]),
                                device=self.device)
        structure = batch.scene_inputs.structure
        centers = structure["module_centers"]
        lengths = torch.cat((_column(structure, "domain_length_x", centers, 12.0),
                             _column(structure, "domain_length_y", centers, 6.0)), -1)
        radius = structure["material_params"][:, 5:6]
        delta = torch.minimum((lengths / lengths.new_tensor([self.model.nx, self.model.ny])).amin(-1, keepdim=True),
                              .15 * radius)
        theta = torch.arange(64, device=centers.device, dtype=centers.dtype) * (2 * torch.pi / 64)
        normals = torch.stack((theta.cos(), theta.sin()), -1)
        points = centers[:, :, None] + (radius + delta)[:, None, :, None] * normals[None, None]
        outside_valid = native_cell_center_stencil(points, lengths, nx=self.model.nx,
                                                   ny=self.model.ny).valid.reshape(ports.shape[:-1])
        base = {field.name: getattr(batch.targets, field.name) for field in fields(ThermalContentTargets)}
        return replace(batch, targets=ThermalPortTargets(**base, teacher_ports=ports,
                       teacher_h_valid=valid, outside_valid=outside_valid))

    def make_batch(self, case_keys, key):
        return self._with_port_targets(super().make_batch(case_keys, key))

    def validation_batches(self):
        return iter(self._with_port_targets(batch) for batch in super().validation_batches())

    def _native_case_batch(self, cases, indices, *, split, ntheta=64):
        return self._with_port_targets(super()._native_case_batch(cases, indices, split=split, ntheta=ntheta))

    def identity_payload(self):
        payload = super().identity_payload()
        payload["local_ports"] = {"schema": "thermal_local_ports_v1", "statistics": self.port_statistics,
                                  "fit_scope": "selected TRAIN only", "lambda_port": 1.0,
                                  "outside_angles": 64, "read_mode": "restricted"}
        return payload

    def loss_metadata(self):
        result = {}
        for name, weight in self.native_weights.items():
            result[name] = {"panel": "native_prediction", "label": name,
                            "unit": "dimensionless scaled squared error",
                            "formula": "TRAIN averages full and restricted native losses; validation uses restricted native loss. Native masks and per-case role reductions are unchanged.",
                            "weight": str(weight)}
        for name, label in (("port/outside_temperature", "Port outside temperature"),
                            ("port/h", "Port effective coefficient")):
            result[name] = {"panel": "native_prediction", "label": label,
                            "unit": "dimensionless scaled squared error",
                            "formula": "Restricted organized 64-angle read; per-case mean over present modules and valid outside receivers; h additionally uses the stored h-valid mask. Empty cases are excluded from the channel case denominator. Normalize using selected-TRAIN standard deviation with safe-scale convention. Stored h is an epsilon-regularized, clipped target; the nonnegative head adds no upper clip.",
                            "weight": "0.5 (lambda_port=1, equal T and h shares)"}
        return result

    def build_local_readout(self):
        """Load the declared frozen asset for a terminal inference readout."""
        return FrozenThermalLocal(self.recipe["local_ports"]["checkpoint_path"]).to(self.device).eval()

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0,
                       temperature=None, **kwargs):
        prediction, aux = super().predict_native(model, scene, receivers, execution_mode, phase,
                                               epoch, temperature, **kwargs)
        native = prediction.restricted if isinstance(prediction, ThermalJointPairPredictions) else prediction
        prepared = native.native_prepared
        outside, valid, counts = organized_outside64(model, prepared)
        structure = prepared.structure
        prescribed = build_local_module_params_from_global(
            receivers.heat, None, structure["material_params"], prepared.source_present)[..., :3]
        prediction.port_values = model.port_head(outside, prescribed, prepared.source_present)
        prediction.port_outside = outside
        prediction.port_valid = valid
        prediction.port_present = prepared.source_present
        prediction.work = dict(prediction.work or {})
        prediction.work.update({f"core_decode_ports_{key}": value for key, value in counts.items()})
        aux["work"] = dict(prediction.work)
        return prediction, aux

    def _port_terms(self, prediction, targets):
        teacher, valid = targets.teacher_ports, targets.teacher_h_valid
        if not torch.equal(prediction.port_valid, targets.outside_valid):
            raise ValueError("Port loss geometry differs from its target-only denominator.")
        t, h = port_loss(prediction.port_values, teacher, prediction.port_present,
                         prediction.port_valid, valid, self.model.port_head.std)
        present = prediction.port_present[..., None].bool() & prediction.port_valid.bool()
        prediction.work = dict(prediction.work or {})
        prediction.work["port_temperature_valid_count"] = int(
            (present & torch.isfinite(teacher[..., 3])).sum())
        prediction.work["port_h_valid_count"] = int(
            (present & valid.bool() & torch.isfinite(teacher[..., 4])).sum())
        counts = self._port_case_counts(targets, prediction.port_present)
        return {"port/outside_temperature": LossTerm(t.sum(), counts[0], 0.5),
                "port/h": LossTerm(h.sum(), counts[1], 0.5)}

    @staticmethod
    def _port_case_counts(targets, present):
        mask = present[..., None].bool() & targets.outside_valid.bool()
        t = mask & torch.isfinite(targets.teacher_ports[..., 3])
        h = mask & targets.teacher_h_valid.bool() & torch.isfinite(targets.teacher_ports[..., 4])
        return int(t.any((1, 2)).sum()), int(h.any((1, 2)).sum())

    def loss_denominators(self, batches, phase="joint", arm="R14-Detail-LocalECover"):
        result = dict(super().loss_denominators(batches, phase=phase, arm=arm))
        counts = [self._port_case_counts(batch.targets, batch.scene_inputs.structure["module_present"])
                  for batch in batches]
        for name, index in (("port/outside_temperature", 0), ("port/h", 1)):
            count = sum(row[index] for row in counts)
            if count:
                result[name] = count
        return result

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        result = super().loss_terms(predictions, targets, phase, auxiliary_state)
        result.update({name: term for name, term in self._port_terms(predictions, targets).items()
                       if term.denominator})
        return result

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch=None, arm=None):
        result = super().validation_loss_terms(predictions, targets, auxiliary_state, batch=batch, arm=arm)
        result.update(self._port_terms(predictions, targets))
        return result


def build_thermal_local_ports_task(config, *, arm=None, device="cpu"):
    arm = arm or ARM_DETAIL
    return _build_thermal_member_routing_task(
        config, arm=arm, device=device, task_type=ThermalLocalPortsTask,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], "arm": arm}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=DETAIL_SOURCE_KIND,
        lineage="R15-2 local ports", required_address_mode=None,
    )
