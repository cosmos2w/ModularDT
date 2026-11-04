"""One-preparation Thermal ports and frozen-local composition."""
from __future__ import annotations
import copy
from dataclasses import dataclass, fields, replace
import numpy as np
import torch
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import LossTerm
from ..local_ports import FrozenThermalLocal
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from ..thermal_coupling import local_response_from_outside, native_angle_view, outside_from_native_output
from .content_hypergraph_task import ThermalContentPredictions, ThermalContentTargets
from .joint_member_routing_task import ThermalJointPairPredictions
from .thermal_port_targets import with_port_targets
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA, DETAIL_SOURCE_KIND, _VARIANTS
from .thermal_port_base import ThermalPortBaseTask

@dataclass(frozen=True)
class ThermalPortResponseTargets(ThermalContentTargets):
    teacher_ports: torch.Tensor | None = None
    teacher_h_valid: torch.Tensor | None = None
    outside_valid: torch.Tensor | None = None
    interface_target64: torch.Tensor | None = None
    interface_valid64: torch.Tensor | None = None

def _module_mean(error, mask):
    """Equal valid-angle, active-module and case reductions."""
    if not torch.isfinite(error[mask]).all():
        raise ValueError('Nonfinite prediction on a valid Thermal port receiver.')
    count = mask.sum(-1)
    per_module = torch.where(mask, error, torch.zeros_like(error)).square().sum(-1) / count.clamp_min(1)
    active = count > 0
    return (per_module * active).sum(-1) / active.sum(-1).clamp_min(1)

def _anchored_port_residual(direct, reference, local, teacher, scale):
    """Reference-conditioned response is a fixed target, not a trainable path."""
    return (direct - torch.nan_to_num(reference) - (local - teacher.detach())) / scale

class ThermalPortResponseTask(ThermalPortBaseTask):

    def configure_port_response(self, recipe):
        self.recipe = copy.deepcopy(recipe)
        self.port_arm = recipe['thermal_port_response']['arm']
        self.source_kind = recipe['initialization']['source_kind']
        self.source_parent_age = int(recipe['thermal_port_response']['source_epoch'])
        self.local_operator = FrozenThermalLocal(recipe['thermal_port_response']['local_checkpoint']).to(self.device).eval()
        self._port_cases = {str(c['case_id']): c for c in (*self.training_cases, *self.validation_cases)}

    def _with_port_targets(self, batch):
        batch = with_port_targets(self, batch)
        width = batch.scene_inputs.structure['module_present'].shape[1]
        cases = [self._port_cases[i] for i in batch.targets.case_ids]
        target = torch.as_tensor(np.stack([c['interface_target'][:width] for c in cases]), device=self.device)
        valid = torch.as_tensor(np.stack([c['interface_condition_valid_mask'][:width] for c in cases]), device=self.device)
        if target.shape[-2:] != (64, 2) or valid.shape[-1] != 64:
            raise ValueError('Thermal P64 requires all 64 native T/q observations.')
        base = {f.name: getattr(batch.targets, f.name) for f in fields(ThermalContentTargets)}
        return replace(batch, targets=ThermalPortResponseTargets(**base, teacher_ports=batch.targets.teacher_ports, teacher_h_valid=batch.targets.teacher_h_valid, outside_valid=batch.targets.outside_valid, interface_target64=target, interface_valid64=valid))

    def make_batch(self, case_keys, key):
        batch = super().make_batch(case_keys, key)
        self.model.core.set_training_clock(self.source_parent_age + key.epoch, key.update_index)
        batch = self._with_port_targets(batch)
        return batch

    def validation_batches(self):
        return iter((self._with_port_targets(b) for b in super().validation_batches()))

    def _native_case_batch(self, cases, indices, *, split, ntheta=64):
        return self._with_port_targets(super()._native_case_batch(cases, indices, split=split, ntheta=ntheta))

    def identity_payload(self):
        result = super().identity_payload()
        result['thermal_port_response'] = {**self.recipe['thermal_port_response'], 'preparations_per_design': 1, 'teacher_use': 'supervised loss only', 'local_asset_exposure': 'TRAIN600/DEV22/VAL89'}
        return result

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None, **kwargs):
        del phase, temperature
        if kwargs or model is not self.model or execution_mode != model.architecture:
            raise ValueError('Unexpected Thermal port response prediction request.')
        model.core.set_training_clock(self.source_parent_age + int(epoch), int(getattr(model.core, '_joint_update_index', 0)))
        p64 = model.prepare_native(scene.structure, receivers.fluid_xy, physical_heat=receivers.heat, local_query_points=receivers.local_query_points, ntheta=64, scene_key=scene.case_ids)
        p = native_angle_view(p64, receivers.ntheta)
        modes = ('full', 'restricted') if model.training else ('restricted',)
        (native, local, full64, outside) = ({}, {}, {}, {})
        work = {'preparations': 1, 'local_calls': 0}
        work.update({f'prepare_{k}': int(v) for (k, v) in p.organization.work_counts.items()})
        for mode in modes:
            with model.core.read_execution(mode):
                (value, counts, raw) = model.apply_native_with_work_counts(p, receivers.heat)
            native[mode] = ThermalContentPredictions(p, value, work=counts)
            full64[mode] = model._extract_native_roles(p64, raw.joint_fields[..., 3:4], raw.joint_fields[..., :3])
            outside[mode] = outside_from_native_output(model, p64, raw)
            local[mode] = local_response_from_outside(model, self.local_operator, p, receivers.heat, receivers.local_query_points, outside[mode])
            work['local_calls'] += 1
            work.update({f'{mode}_decode_{k}': int(v) for (k, v) in counts.items()})
        prediction = ThermalJointPairPredictions(native['full'], native['restricted'], work) if model.training else native['restricted']
        prediction.initial = prediction
        prediction.local1 = local
        prediction.outside64 = outside
        prediction.native64 = full64
        prediction.prepared64 = p64
        prediction.local_points = receivers.local_query_points
        prediction.initial_heat = receivers.heat
        prediction.work = work
        return (prediction, {'work': work})

    def _terms(self, prediction, targets):
        result = super()._terms(prediction, targets)
        result = {k: v for (k, v) in result.items() if not k.startswith('initial/')}
        if self.port_arm == 'C0':
            return result
        paired = isinstance(prediction, ThermalJointPairPredictions)
        modes = ('full', 'restricted') if paired else ('restricted',)
        p = prediction.prepared64
        teacher_outside = targets.teacher_ports[..., 3].to(prediction.outside64[modes[0]])
        present = p.source_present[..., None].bool()
        complete = present & (~targets.outside_valid.bool() | ~torch.isfinite(teacher_outside))
        if complete.any():
            raise ValueError('Incomplete teacher outside64 sequence for an active module.')
        with torch.no_grad():
            teacher = local_response_from_outside(self.model, self.local_operator, p, prediction.initial_heat, prediction.local_points, teacher_outside)['interface_pred']
        prediction.work['teacher_local_calls'] = 1
        reference = targets.interface_target64.to(teacher)
        surface = p.stencils['surface'].valid.reshape_as(targets.interface_valid64)
        outside_valid = p.stencils['outside'].valid.reshape_as(targets.interface_valid64)
        (total_compat, total_outside) = ([], [])
        for mode in modes:
            direct = prediction.native64[mode]['pred_interface']
            local = prediction.local1[mode]['interface_pred']
            role = []
            for channel in range(2):
                mask = present & surface & torch.isfinite(reference[..., channel])
                if channel == 1:
                    mask = mask & outside_valid
                residual = _anchored_port_residual(direct[..., channel], reference[..., channel], local[..., channel], teacher[..., channel], self.model.interface_target_std[channel])
                role.append(_module_mean(residual, mask))
            total_compat.append(sum(role) / 2)
            valid = present & outside_valid & torch.isfinite(teacher_outside)
            error = (prediction.outside64[mode] - torch.nan_to_num(teacher_outside)) / self.model.field_std[4]
            total_outside.append(_module_mean(error, valid))
        cases = len(targets.case_ids)
        result['interface/consistency'] = LossTerm(sum((x.sum() for x in total_compat)) / len(modes), cases, 0.05)
        result['port/outside_temperature'] = LossTerm(sum((x.sum() for x in total_outside)) / len(modes), cases, 0.05)
        return result

    def loss_denominators(self, batches, phase='joint', arm=ARM_DETAIL):
        count = sum((len(batch.targets.case_ids) for batch in batches))
        result = {**{f'final/{role}': count for role in self.native_weights}, **{f'local/{role}': count for role in ('temperature/material', 'temperature/surface', 'q_proxy')}, 'interface/consistency': count}
        if self.port_arm != 'C0':
            result['port/outside_temperature'] = count
        return result

    def loss_metadata(self):
        result = {k: v for (k, v) in super().loss_metadata().items() if not k.startswith('initial/')}
        if self.port_arm != 'C0':
            result['interface/consistency']['formula'] = '64-angle direct target error minus predicted/local teacher response; teacher stopped.'
            result['port/outside_temperature'] = {'panel': 'native_prediction', 'label': 'outside temperature', 'unit': 'scaled squared error', 'formula': '64-angle physical outside temperature; equal active module and case reductions.', 'weight': '0.05'}
        return result

    def completed_training_epoch(self, model, engine, epoch, output, row):
        super().completed_training_epoch(model, engine, epoch, output, row)
        if 'reader_epoch_context' in row:
            row['reader_epoch_context']['absolute_training_epoch'] = self.source_parent_age + int(epoch)
            row['reader_epoch_context']['additional_epochs_after_parent'] = int(epoch)

def build_thermal_port_response_task(recipe, *, arm=ARM_DETAIL, device='cpu'):
    spec = recipe['thermal_port_response']
    if spec.get('schema') != 'thermal_port_response_v1' or spec.get('enabled') is not True or spec.get('arm') not in ('C0', 'P64'):
        raise ValueError('Expected a C0 or P64 Thermal port recipe.')
    initialization = recipe.get('initialization', {})
    families = []
    objective = recipe.get('objective', {})
    if initialization.get('source_kind') != 'thermal_port_response_explicit_transfer_v1' or initialization.get('source_checkpoint') != spec.get('source_checkpoint') or initialization.get('source_parent_age') != spec.get('source_epoch') or (initialization.get('pretrained_weights') is not True) or (initialization.get('source_arm') != arm) or (objective.get('response_training_families') != families) or (objective.get('response_fit') is not bool(families)):
        raise ValueError('Thermal port transfer or FIT response declaration is inconsistent.')
    source = load_trusted_checkpoint(spec['source_checkpoint'], map_location='cpu')
    if int(source['epoch']) != int(spec['source_epoch']):
        raise ValueError('Thermal source epoch differs from the declared transfer.')
    base = copy.deepcopy(recipe)
    base['initialization'].update(source_kind=DETAIL_SOURCE_KIND, source_checkpoint=None, pretrained_weights=False, source_parent_age=0, source_arm=f"scratch_common_seed{base['initialization']['seed']}")
    base['member_routing']['health'] = {'enabled': False}
    grid = spec.get('environment_grid_nx_ny', [24, 8])
    if grid not in ([24, 8], [12, 8], [12, 6]):
        raise ValueError('Unsupported R15-6 environment grid.')

    class GridAdapter(ThermalDetailAnchoredMemberRoutingAdapter):

        def __init__(self, *args, **kwargs):
            kwargs.update(environment_nx=grid[0], environment_ny=grid[1])
            super().__init__(*args, **kwargs)
    variants = {DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'adapter': GridAdapter, 'arm': arm}}
    (model, provider) = _build_thermal_member_routing_task(base, arm=arm, device=device, task_type=ThermalPortResponseTask, variants=variants, section_schema=DETAIL_CONFIG_SCHEMA, source_kind=DETAIL_SOURCE_KIND, lineage='Thermal one-pass port response', required_address_mode=None)
    source_recipe = source['experiment_identity']['recipe']
    if source_recipe['training']['membership'] != recipe['training']['membership'] or any((source_recipe['dataset'][k] != recipe['dataset'][k] for k in ('name', 'data_path', 'manifest_path'))):
        raise ValueError('Transfer source dataset or TRAIN membership differs.')
    source_config = copy.deepcopy(source_recipe['member_routing']['model'])
    requested_config = copy.deepcopy(recipe['member_routing']['model'])
    source_config.pop('query_tile_size', None)
    requested_config.pop('query_tile_size', None)
    if source_config != requested_config:
        raise ValueError('Transfer source core configuration differs.')
    adapter = source['experiment_identity']['provider_identity']['adapter_state']
    if adapter['normalization_stats'] != model.normalization_stats or float(adapter['heat_scale']) != model.heat_scale:
        raise ValueError('Transfer source TRAIN transforms differ.')
    model.load_state_dict(source['model_state_dict'], strict=True)
    provider.configure_port_response(recipe)
    return (model, provider)

def load_thermal_port_response_checkpoint(path, *, device='cpu', local_checkpoint=None):
    """Reconstruct the native model from checkpoint metadata without loading data.

    The frozen local operator remains the explicitly declared external asset.
    The returned clock identifies the original source age plus this run's age.
    An explicit local_checkpoint overrides the saved path for relocated assets.
    """
    saved = load_trusted_checkpoint(path, map_location='cpu')
    identity = saved.get('experiment_identity', {})
    recipe = identity.get('recipe', {})
    spec = recipe.get('thermal_port_response', {})
    adapter = identity.get('provider_identity', {}).get('adapter_state', {})
    if (identity.get('workflow') != 'relation_anchored_case_training_v1'
            or spec.get('schema') != 'thermal_port_response_v1' or spec.get('enabled') is not True
            or spec.get('arm') not in ('C0', 'P64')
            or recipe.get('member_routing', {}).get('arm') != saved.get('arm')
            or adapter.get('schema') != ThermalDetailAnchoredMemberRoutingAdapter.SCHEMA
            or adapter.get('environment_grid_nx_ny') != spec.get('environment_grid_nx_ny', [24, 8])):
        raise ValueError('Checkpoint is not a self-consistent Thermal port response run.')
    model = ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(adapter, device=device, state_dict=saved['model_state_dict'])
    model.core.set_training_clock(int(spec['source_epoch']) + int(saved['epoch']), 0)
    model.eval()
    local_path = spec['local_checkpoint'] if local_checkpoint is None else local_checkpoint
    local = FrozenThermalLocal(local_path).to(device).eval()
    return (model, local, saved)
