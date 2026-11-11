"""Fresh R16-10 joint-field provider with physical 64-angle supervision."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from honf_runtime.unified_training import LossTerm, OptimizerGroupSpec, ScheduleSpec
from ..data.datasets import H5Normalizer
from ..native_observations import interpolate_native
from .content_hypergraph_task import ThermalContentPredictions, ThermalContentTargets
from .joint_member_routing_task import ThermalJointPairPredictions
from .joint_task import _read_selected_cases
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (
    ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA, DETAIL_SOURCE_KIND,
    ThermalR138AnchoredMemberRoutingTask, _VARIANTS,
)


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


def stratified_validation_ids(cases, seed=20261108):
    """Select nine source-ordered examples across native module-count strata."""
    if len(cases) != 89:
        raise ValueError('Expected canonical VAL89 before selecting the validation panel.')
    rng = np.random.default_rng(seed)
    groups = {}
    for i, case in enumerate(cases):
        count = int((np.asarray(case['structure']['module_present']) > .5).sum())
        groups.setdefault(count, []).append(i)
    queues = {k: list(rng.permutation(v)) for k, v in sorted(groups.items())}
    chosen = []
    while len(chosen) < 9:
        for key in queues:
            if queues[key] and len(chosen) < 9:
                chosen.append(queues[key].pop())
    return tuple(str(cases[i]['case_id']) for i in sorted(chosen))


class ThermalR1610NativeTask(ThermalR138AnchoredMemberRoutingTask):
    """R14 joint predictor; no frozen model, edited data, or local composition."""

    def configure_native(self):
        self._port_cases = {str(c['case_id']): c for c in (*self.training_cases, *self.validation_cases)}

    def _schedule_payload(self):
        return dict(peak_lr=3e-4, warmup_start_lr=3e-4, warmup_epochs=0,
                    hold_through_epoch=1000, total_epochs=2000, final_lr=3e-5)

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
        if ports.interface_target.shape[-2:] != (64, 2) or ports.outside_temperature.shape != ports.interface_target.shape[:-1]:
            raise ValueError('Expected aligned 64-angle physical port observations.')
        return replace(batch, targets=ports, receivers=replace(batch.receivers, ntheta=64))

    def with_raw_port_targets(self, batch, raw):
        """Attach the given physical endpoint's own observations, never baseline labels."""
        width = batch.scene_inputs.structure['module_present'].shape[1]
        interface = batch.targets.interface_target
        count = raw['interface_target'].shape[0]
        if count > width or interface.shape[-2:] != (64, 2):
            raise ValueError('Raw endpoint port catalogue does not match scene.')
        outside = torch.full(interface.shape[:-1], float('nan'), device=interface.device)
        valid = torch.zeros_like(outside, dtype=torch.bool)
        outside[:, :count] = torch.as_tensor(raw['teacher_ports'][..., 3], device=outside.device)
        valid[:, :count] = torch.as_tensor(raw['outside_valid'], device=outside.device)
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

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None, **kwargs):
        del phase, epoch, temperature
        if kwargs or model is not self.model or execution_mode != model.architecture:
            raise ValueError('Wrong R16-10 prediction request.')
        old_tile = model.core.query_tile_size
        if not model.training:
            model.core.query_tile_size = 1024
        try:
            return self._predict_prepared(model, scene, receivers)
        finally:
            model.core.query_tile_size = old_tile

    def _predict_prepared(self, model, scene, receivers):
        prepared = model.prepare_native(scene.structure, receivers.fluid_xy, physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points, ntheta=64, scene_key=scene.case_ids)
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
            mode = getattr(model.core, '_read_execution_override', None) or 'restricted'
            if mode not in ('full', 'restricted'):
                raise ValueError('Unknown R16-10 read mode.')
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
        terms['port/outside_temperature'] = LossTerm(sum(v.sum() for v in outside_losses) / len(modes), count, .05)
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
            'unit': 'scaled squared error', 'formula': 'Physical 64-angle outside temperature, equal valid module and case reductions.', 'weight': '0.05'}
        return result


def build_thermal_r16_10_native_task(recipe, *, arm=ARM_DETAIL, device='cpu', task_type=ThermalR1610NativeTask,
                                     saved_adapter_state=None, saved_normalization_source=None,
                                     source_kind=DETAIL_SOURCE_KIND, lineage='R16-10 scratch'):
    training = recipe['training']
    if (int(training['effective_batch_size']) != 48 or not 1 <= int(training['microbatch_cases']) <= 24
            or int(recipe['member_routing']['model']['query_tile_size']) != 128):
        raise ValueError('R16-10 requires effective batch 48, microbatch at most 24 and training tile 128.')
    model, provider = _build_thermal_member_routing_task(recipe, arm=arm, device=device, task_type=task_type,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'arm': arm}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=source_kind, lineage=lineage, required_address_mode=None,
        saved_adapter_state=saved_adapter_state, saved_normalization_source=saved_normalization_source)
    val_ids = [case_id for case_id in provider.catalog_case_ids
               if provider.case_split_by_id[case_id] == 'test' and case_id != '0273']
    normalizer = H5Normalizer({k: np.asarray(v, dtype=np.float32) for k, v in model.normalization_stats.items()})
    cases = _read_selected_cases(provider.data_path, val_ids, split='test', normalizer=normalizer, include_grid=False)
    selected = set(stratified_validation_ids(cases))
    provider.validation_cases = tuple(c for c in cases if str(c['case_id']) in selected)
    provider.validation_case_ids = tuple(str(c['case_id']) for c in provider.validation_cases)
    if tuple(recipe['evaluation']['validation_case_ids']) != provider.validation_case_ids:
        raise ValueError('R16-10 configured validation IDs disagree with the sampled canonical panel.')
    provider._validation_index_by_id = {v: i for i, v in enumerate(provider.validation_case_ids)}
    provider.validation_scope = 'input_stratified_seed20261108_VAL9_of_89'
    provider.configure_native()
    return model, provider
