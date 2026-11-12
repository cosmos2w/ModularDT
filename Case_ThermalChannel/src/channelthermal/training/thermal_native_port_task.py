"""Fresh joint native Thermal fields and physical port supervision."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from honf_forward_core.read_execution import resolve_read_mode
from honf_runtime.unified_training import LossTerm, OptimizerGroupSpec, ScheduleSpec

from ..native_observations import interpolate_native
from .content_hypergraph_task import ThermalContentPredictions, ThermalContentTargets
from .joint_member_routing_task import ThermalJointPairPredictions
from .r13_8_anchored_task import ARM_DETAIL, ThermalR138AnchoredMemberRoutingTask


@dataclass(frozen=True)
class NativePortTargets(ThermalContentTargets):
    outside_temperature: torch.Tensor | None = None
    outside_valid: torch.Tensor | None = None


def _module_mean(error, mask):
    if not torch.isfinite(error[mask]).all():
        raise ValueError('Nonfinite prediction on valid physical ports.')
    count = mask.sum(-1)
    active = count > 0
    values = torch.where(mask, error, torch.zeros_like(error)).square().sum(-1) / count.clamp_min(1)
    return (values * active).sum(-1) / active.sum(-1).clamp_min(1)


class ThermalNativePortTask(ThermalR138AnchoredMemberRoutingTask):
    """R14 joint predictor; no frozen model, edited data, or local composition."""

    def _validate_initialization(self, initialization):
        if self.recipe.get('training_profile') == 'state_response':
            self.seed = int(initialization.get('seed', 0))
            super()._validate_initialization(initialization)
            self.seed = int(self.recipe.get('training', {}).get('seed', self.seed))
        else:
            super()._validate_initialization(initialization)

    def configure_native(self):
        options = self.recipe.get('thermal_joint_response', {}).get('native_ports', {})
        count = options.get('angle_count', 64)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or 64 % count:
            raise ValueError('Port angle_count must be a positive divisor of the saved 64-angle catalogue.')
        weight = float(options.get('outside_temperature_weight', .05))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Outside-temperature port weight must be finite and nonnegative.')
        self.port_angle_count, self.outside_temperature_weight = count, weight
        self._port_cases = {str(c['case_id']): c for c in (*self.training_cases, *self.validation_cases)}

    def optimizer_groups(self, model, arm, stage):
        del stage
        if model is not self.model or arm != model.architecture:
            raise ValueError('Wrong R16-10 model or arm.')
        opt = self.recipe['optimizer']
        return (OptimizerGroupSpec('joint', tuple(n for n, p in model.named_parameters() if p.requires_grad),
                ScheduleSpec(**self._schedule_payload()), weight_decay=float(opt['weight_decay']),
                betas=tuple(map(float, opt['betas'])), eps=float(opt['eps'])),)

    def _attach_ports(self, batch, targets, valid, outside, outside_valid):
        device = batch.targets.field_targets.device
        base = {f.name: getattr(batch.targets, f.name) for f in fields(ThermalContentTargets)}
        base.update(interface_target=torch.as_tensor(targets, device=device),
                    interface_valid_mask=torch.as_tensor(valid, device=device))
        ports = NativePortTargets(**base, outside_temperature=torch.as_tensor(outside, device=device),
                                  outside_valid=torch.as_tensor(outside_valid, device=device))
        angles = ports.interface_target.shape[-2]
        if (ports.interface_target.shape[-1] != 2 or angles not in (64, self.port_angle_count)
                or ports.outside_temperature.shape != ports.interface_target.shape[:-1]
                or ports.interface_valid_mask.shape != ports.interface_target.shape[:-1]
                or ports.outside_valid.shape != ports.interface_target.shape[:-1]):
            raise ValueError('Expected aligned native physical port observations.')
        stride = angles // self.port_angle_count
        ports = replace(ports, interface_target=ports.interface_target[..., ::stride, :],
                        interface_valid_mask=ports.interface_valid_mask[..., ::stride],
                        outside_temperature=ports.outside_temperature[..., ::stride],
                        outside_valid=ports.outside_valid[..., ::stride])
        return replace(batch, targets=ports, receivers=replace(batch.receivers, ntheta=self.port_angle_count))

    def with_raw_port_targets(self, batch, raw):
        """Attach the given physical endpoint's own observations, never baseline labels."""
        width = batch.scene_inputs.structure['module_present'].shape[1]
        interface = batch.targets.interface_target
        count = raw['interface_target'].shape[0]
        angles = interface.shape[-2]
        if count > width or interface.shape[-1] != 2 or angles not in (64, self.port_angle_count):
            raise ValueError('Raw endpoint port catalogue does not match scene.')
        outside = torch.full(interface.shape[:-1], float('nan'), device=interface.device)
        valid = torch.zeros_like(outside, dtype=torch.bool)
        stride = 64 // angles
        outside[:, :count] = torch.as_tensor(raw['teacher_ports'][..., ::stride, 3], device=outside.device)
        valid[:, :count] = torch.as_tensor(raw['outside_valid'][..., ::stride], device=outside.device)
        return self._attach_ports(batch, interface, batch.targets.interface_valid_mask, outside, valid)

    def _with_native_ports(self, batch, cases=None):
        cases = self._port_cases if cases is None else {str(c['case_id']): c for c in cases}
        rows = [cases[str(i)] for i in batch.targets.case_ids]
        width = batch.scene_inputs.structure['module_present'].shape[1]
        target = np.stack([c['interface_target'][:width] for c in rows])
        valid = np.stack([c['interface_condition_valid_mask'][:width] for c in rows])
        ports = np.stack([c['teacher_port_tokens'][:width, :, 3] for c in rows])
        # Geometric validity is checked again from the actual prepared outside stencil.
        return self._attach_ports(batch, target, valid, ports, np.isfinite(ports))

    def make_batch(self, case_keys, key):
        return self._with_native_ports(super().make_batch(case_keys, key))

    def validation_batches(self):
        return iter(self._with_native_ports(b) for b in super().validation_batches())

    def _native_case_batch(self, cases, indices, *, split, ntheta=64):
        batch = super()._native_case_batch(cases, indices, split=split, ntheta=64)
        return self._with_native_ports(batch, cases)

    def _load_native_cases(self, case_ids):
        if self.recipe.get('training', {}).get('membership') != 'catalog_train':
            return super()._load_native_cases(case_ids)
        from ..data.datasets import H5Normalizer
        from .joint_task import _read_selected_cases

        requested = tuple(map(str, case_ids))
        allowed = set(self.training_case_ids) | set(self.validation_case_ids)
        if not requested or len(set(requested)) != len(requested) or not set(requested).issubset(allowed):
            raise ValueError('Native case IDs must be unique members of the selected TRAIN/validation panel.')
        normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32)
                                   for key, value in self.adapter.normalization_stats.items()})
        groups = {}
        for case_id in requested:
            groups.setdefault(self.case_split_by_id[case_id], []).append(case_id)
        cases = []
        for split, ids in groups.items():
            cases.extend(_read_selected_cases(self.data_path, ids, split=split, normalizer=normalizer,
                include_grid=True, include_raster_geometry=getattr(self.adapter.core, 'module_input_dim', None) == 10,
                include_prescribed_patch=getattr(self.adapter, 'prescribed_patch_mode', 'none') != 'none'))
        index = {case_id: i for i, case_id in enumerate(requested)}
        return sorted(cases, key=lambda case: index[str(case['case_id'])])

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None, **kwargs):
        del phase, epoch, temperature
        if kwargs or model is not self.model or execution_mode != model.architecture:
            raise ValueError('Wrong R16-10 prediction request.')
        old_tile = model.core.query_tile_size
        if not model.training:
            tile = self.recipe.get('evaluation', {}).get('inference_query_tile_size', 1024)
            if type(tile) is not int or tile < 1:
                raise ValueError('Inference query tile size must be a positive integer.')
            model.core.query_tile_size = tile
        try:
            return self._predict_prepared(model, scene, receivers)
        finally:
            model.core.query_tile_size = old_tile

    def _predict_prepared(self, model, scene, receivers):
        prepared = model.prepare_native(scene.structure, receivers.fluid_xy, physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points, ntheta=receivers.ntheta, scene_key=scene.case_ids)
        def convert(output):
            native = model._extract_native_roles(prepared, output.joint_fields[..., 3:4], output.joint_fields[..., :3])
            native = dict(native)
            temperature = model._physical_temperature(output.joint_fields[..., 3])
            native['pred_outside_temperature'] = interpolate_native(
                temperature[..., None], prepared.stencils['outside']).reshape(
                    prepared.source_present.shape[0], *prepared.role_shapes['outside'])
            return ThermalContentPredictions(prepared, native, work=output.work_counts)
        if model.training:
            full, restricted, shared = model.core.decode_paired(prepared.organization, prepared.query_batch)
            work = {f'paired_{k}': int(v) for k, v in shared.items()}
            pred = ThermalJointPairPredictions(convert(full), convert(restricted), work)
        else:
            mode = resolve_read_mode(model.core)
            with model.core.read_execution(mode):
                _, counts, output = model.apply_native_with_work_counts(prepared, receivers.heat)
            pred = convert(output)
            work = {f'member_routing_read_mode_{mode}': 1, **{str(k): int(v) for k, v in counts.items()}}
            pred.work = work
        return pred, {'work': work}

    def _native_terms(self, predictions, targets):
        modes = (predictions.full, predictions.restricted) if isinstance(predictions, ThermalJointPairPredictions) else (predictions,)
        all_roles, outside_losses = [], []
        for prediction in modes:
            all_roles.append(self._role_losses(prediction, targets))
            prepared = prediction.native_prepared
            value = prediction.native_main['pred_outside_temperature']
            target = targets.outside_temperature.to(value)
            mask = (prepared.source_present[..., None].bool() & targets.outside_valid.to(value.device).bool()
                    & prepared.stencils['outside'].valid.reshape_as(value) & torch.isfinite(target))
            outside_losses.append(_module_mean((value - torch.nan_to_num(target)) / self.model.field_std[4], mask))
        count = len(targets.case_ids)
        terms = {name: LossTerm(sum(r[name].sum() for r in all_roles) / len(modes), count, float(weight))
                 for name, weight in self.native_weights.items()}
        terms['port/outside_temperature'] = LossTerm(sum(v.sum() for v in outside_losses) / len(modes), count, self.outside_temperature_weight)
        return terms

    def endpoint_state_loss(self, prediction, batch):
        terms = self._native_terms(prediction, batch.targets)
        return sum(t.numerator / t.denominator * t.weight for t in terms.values())

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        if phase != 'joint':
            raise ValueError('R16-10 uses the joint phase.')
        return self._native_terms(predictions, targets)

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch=None, arm=None):
        return self._native_terms(predictions, targets)

    def loss_denominators(self, batches, phase='joint', arm=ARM_DETAIL):
        result = super().loss_denominators(batches, phase, arm)
        result['port/outside_temperature'] = sum(len(b.targets.case_ids) for b in batches)
        return result

    def loss_metadata(self):
        result = {name: {'panel': 'native_prediction', 'label': name, 'unit': 'scaled squared error',
                         'formula': 'Direct joint physical fields with valid native masks.', 'weight': str(weight)}
                  for name, weight in self.native_weights.items()}
        result['port/outside_temperature'] = {'panel': 'native_prediction', 'label': 'outside temperature',
            'unit': 'scaled squared error', 'formula': f'Physical {self.port_angle_count}-angle outside temperature, equal valid module and case reductions.', 'weight': str(self.outside_temperature_weight)}
        return result
