"""One-preparation Thermal port composition for R15-6 C0 and P64."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import LossTerm
from ..local_ports import FrozenThermalLocal
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from ..thermal_coupling import local_response_from_outside, native_angle_view, outside_from_native_output
from .content_hypergraph_task import (ThermalContentPredictions, ThermalContentTargets,
                                     ThermalContentSceneInputs, ThermalContentReceivers)
from .joint_member_routing_task import ThermalJointPairPredictions
from .local_ports_task import ThermalLocalPortsTask
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import (ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA,
                                 DETAIL_SOURCE_KIND, _VARIANTS)
from .thermal_response_feedback_task import ThermalResponseFeedbackTask


@dataclass(frozen=True)
class ThermalPortResponseTargets(ThermalContentTargets):
    teacher_ports: torch.Tensor | None = None
    teacher_h_valid: torch.Tensor | None = None
    outside_valid: torch.Tensor | None = None
    interface_target64: torch.Tensor | None = None
    interface_valid64: torch.Tensor | None = None
    response_pack: dict | None = None


@dataclass(frozen=True)
class ThermalPortResponseScene(ThermalContentSceneInputs):
    response_scenes: tuple = ()


def _module_mean(error, mask):
    """Equal valid-angle, active-module and case reductions."""
    if not torch.isfinite(error[mask]).all():
        raise ValueError("Nonfinite prediction on a valid Thermal port receiver.")
    count = mask.sum(-1)
    per_module = torch.where(mask, error, torch.zeros_like(error)).square().sum(-1) / count.clamp_min(1)
    active = count > 0
    return (per_module * active).sum(-1) / active.sum(-1).clamp_min(1)


def _anchored_port_residual(direct, reference, local, teacher, scale):
    """Reference-conditioned response is a fixed target, not a trainable path."""
    return ((direct - torch.nan_to_num(reference)) - (local - teacher.detach())) / scale


class ThermalPortResponseTask(ThermalResponseFeedbackTask):
    def configure_port_response(self, recipe):
        self.recipe = copy.deepcopy(recipe)
        self.port_arm = recipe['thermal_port_response']['arm']
        self.source_kind = recipe['initialization']['source_kind']
        self.source_parent_age = int(recipe['thermal_port_response']['source_epoch'])
        self.local_operator = FrozenThermalLocal(recipe['thermal_port_response']['local_checkpoint']).to(self.device).eval()
        self._port_cases = {str(c['case_id']): c for c in (*self.training_cases, *self.validation_cases)}
        if self.port_arm == 'R64':
            from .thermal_response_packs import load_fit_packs
            self.response_packs = load_fit_packs(recipe['thermal_port_response']['atlas_directory'], {
                'fluid_temperature': float(self.model.field_std[4]),
                'surface_temperature': float(self.model.interface_target_std[0]),
                'material_temperature': float(self.model.internal_temperature_std[0]),
                'q': float(self.model.interface_target_std[1]),
                'flow': self.model.field_std[:4].detach().cpu().numpy()})

    def _with_port_targets(self, batch):
        batch = ThermalLocalPortsTask._with_port_targets(self, batch)
        width = batch.scene_inputs.structure['module_present'].shape[1]
        cases = [self._port_cases[i] for i in batch.targets.case_ids]
        target = torch.as_tensor(np.stack([c['interface_target'][:width] for c in cases]), device=self.device)
        valid = torch.as_tensor(np.stack([c['interface_condition_valid_mask'][:width] for c in cases]), device=self.device)
        if target.shape[-2:] != (64, 2) or valid.shape[-1] != 64:
            raise ValueError('Thermal P64 requires all 64 native T/q observations.')
        base = {f.name: getattr(batch.targets, f.name) for f in fields(ThermalContentTargets)}
        return replace(batch, targets=ThermalPortResponseTargets(**base, teacher_ports=batch.targets.teacher_ports,
            teacher_h_valid=batch.targets.teacher_h_valid, outside_valid=batch.targets.outside_valid,
            interface_target64=target, interface_valid64=valid))

    def make_batch(self, case_keys, key):
        batch = super().make_batch(case_keys, key)
        self.model.core.set_training_clock(self.source_parent_age + key.epoch, key.update_index)
        batch = self._with_port_targets(batch)
        if self.port_arm == 'R64' and key.microbatch_index == 0:
            from .thermal_response_packs import FIT_FAMILIES, sample_fit_pack
            updates = math.ceil(len(self.training_cases) / self.effective_batch_size)
            positions = [i * updates // 4 for i in range(4)]
            if key.update_index in positions:
                family = FIT_FAMILIES[positions.index(key.update_index)]
                pack = sample_fit_pack(self.response_packs, family, key.epoch, seed=self.seed)
                scenes = self._response_scenes(pack)
                scene = ThermalPortResponseScene(batch.scene_inputs.structure,
                    batch.scene_inputs.response_structure, batch.scene_inputs.case_ids, scenes)
                batch = replace(batch, scene_inputs=scene,
                                targets=replace(batch.targets, response_pack=pack))
        return batch

    def validation_batches(self):
        return iter(self._with_port_targets(b) for b in super().validation_batches())

    def _native_case_batch(self, cases, indices, *, split, ntheta=64):
        return self._with_port_targets(super()._native_case_batch(cases, indices, split=split, ntheta=ntheta))

    def _response_scenes(self, pack):
        """Construct target-free native inputs from the saved physical designs."""
        scenes = []
        for name in ('baseline', 'geometry', 'heat'):
            record = pack[name]
            modules = record.design.active_modules
            count = len(modules)
            tensor = lambda a: torch.as_tensor(np.asarray(a).copy(), device=self.device, dtype=torch.float32)
            structure = {key: (torch.as_tensor(np.asarray(value).copy(), device=self.device, dtype=torch.long)
                               if key == 'module_source_ids' else tensor(value))[None]
                         for key, value in pack['structures'][name].items()}
            roles = pack['roles'][name]
            local = np.asarray(roles['solid_temperature']['query_features'])[:, :2].reshape(count, 32, 2)
            if not all(np.array_equal(local[0], local[i]) for i in range(1, count)):
                raise ValueError('Response local receivers must share aligned coordinates across original modules.')
            scene = ThermalContentSceneInputs(structure, case_ids=(pack['family_id'] + ':' + name,))
            receivers = ThermalContentReceivers(tensor(roles['fluid_fields']['query_features'][:, :2])[None],
                tensor(local[0])[None], structure['heat_powers'], ntheta=64)
            scenes.append((name, scene, receivers))
        return tuple(scenes)

    def identity_payload(self):
        result = super(ThermalResponseFeedbackTask, self).identity_payload()
        result['thermal_port_response'] = {**self.recipe['thermal_port_response'], 'preparations_per_design': 1,
            'teacher_use': 'supervised loss only', 'local_asset_exposure': 'TRAIN600/DEV22/VAL89'}
        if self.port_arm == 'R64':
            result['thermal_port_response']['response_scales'] = copy.deepcopy(self.response_packs.response_scales)
            result['thermal_port_response']['null_flow_scales'] = self.response_packs.null_flow_scales.tolist()
            result['thermal_port_response']['response_fit_families'] = ['0001', '0318', '0333', '0348']
            result['response_training_family_ids'] = ['0001', '0318', '0333', '0348']
            result['response_supervision'] = 'saved single geometry and balanced heat pairs; FIT-only scales'
            result['objective']['response_fit'] = True
            result['objective']['response_training_families'] = ['0001', '0318', '0333', '0348']
        return result

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch=0, temperature=None, **kwargs):
        del phase, temperature
        if kwargs or model is not self.model or execution_mode != model.architecture:
            raise ValueError('Unexpected Thermal port response prediction request.')
        model.core.set_training_clock(self.source_parent_age + int(epoch),
                                      int(getattr(model.core, '_joint_update_index', 0)))
        p64 = model.prepare_native(scene.structure, receivers.fluid_xy, physical_heat=receivers.heat,
            local_query_points=receivers.local_query_points, ntheta=64, scene_key=scene.case_ids)
        p = native_angle_view(p64, receivers.ntheta)
        modes = ('full', 'restricted') if model.training else ('restricted',)
        native, local, full64, outside = {}, {}, {}, {}
        work = {'preparations': 1, 'local_calls': 0}
        work.update({f'prepare_{k}': int(v) for k, v in p.organization.work_counts.items()})
        for mode in modes:
            with model.core.read_execution(mode):
                value, counts, raw = model.apply_native_with_work_counts(p, receivers.heat)
            native[mode] = ThermalContentPredictions(p, value, work=counts)
            full64[mode] = model._extract_native_roles(p64, raw.joint_fields[..., 3:4], raw.joint_fields[..., :3])
            outside[mode] = outside_from_native_output(model, p64, raw)
            local[mode] = local_response_from_outside(model, self.local_operator, p, receivers.heat,
                                                       receivers.local_query_points, outside[mode])
            work['local_calls'] += 1
            work.update({f'{mode}_decode_{k}': int(v) for k, v in counts.items()})
        prediction = ThermalJointPairPredictions(native['full'], native['restricted'], work) if model.training else native['restricted']
        prediction.initial = prediction
        prediction.local1 = local
        prediction.outside64 = outside
        prediction.native64 = full64
        prediction.prepared64 = p64
        prediction.local_points = receivers.local_query_points
        prediction.initial_heat = receivers.heat
        prediction.work = work
        if isinstance(scene, ThermalPortResponseScene):
            prediction.response_designs = {}
            for name, extra_scene, extra_receivers in scene.response_scenes:
                extra, _ = self.predict_native(model, extra_scene, extra_receivers, execution_mode,
                                               'joint', epoch=epoch)
                prediction.response_designs[name] = extra
                work['preparations'] += int(extra.work['preparations'])
                work['local_calls'] += int(extra.work['local_calls'])
                for key, value in extra.work.items():
                    work[f'response_{name}_{key}'] = int(value)
        return prediction, {'work': work}

    def _terms(self, prediction, targets):
        result = super()._terms(prediction, targets)
        result = {k: v for k, v in result.items() if not k.startswith('initial/')}
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
            teacher = local_response_from_outside(self.model, self.local_operator, p, prediction.initial_heat,
                                                  prediction.local_points, teacher_outside)['interface_pred']
        prediction.work['teacher_local_calls'] = 1
        reference = targets.interface_target64.to(teacher)
        # The stored condition mask describes packed h validity, not the
        # surface/q observations or outside temperature.
        surface = p.stencils['surface'].valid.reshape_as(targets.interface_valid64)
        outside_valid = p.stencils['outside'].valid.reshape_as(targets.interface_valid64)
        total_compat, total_outside = [], []
        for mode in modes:
            direct = prediction.native64[mode]['pred_interface']
            local = prediction.local1[mode]['interface_pred']
            role = []
            for channel in range(2):
                mask = present & surface & torch.isfinite(reference[..., channel])
                if channel == 1:
                    mask = mask & outside_valid
                residual = _anchored_port_residual(direct[..., channel], reference[..., channel],
                    local[..., channel], teacher[..., channel], self.model.interface_target_std[channel])
                role.append(_module_mean(residual, mask))
            total_compat.append(sum(role) / 2)
            valid = present & outside_valid & torch.isfinite(teacher_outside)
            error = (prediction.outside64[mode] - torch.nan_to_num(teacher_outside)) / self.model.field_std[4]
            total_outside.append(_module_mean(error, valid))
        cases = len(targets.case_ids)
        result['interface/consistency'] = LossTerm(sum(x.sum() for x in total_compat) / len(modes), cases, .05)
        result['port/outside_temperature'] = LossTerm(sum(x.sum() for x in total_outside) / len(modes), cases, .05)
        if targets.response_pack is not None:
            result.update(self._response_terms(prediction, targets.response_pack, modes))
        return result

    def _response_terms(self, prediction, pack, modes):
        designs = prediction.response_designs
        if set(designs) != {'baseline', 'geometry', 'heat'}:
            raise ValueError('Response pack must execute its three physical designs.')
        def branch(pred, mode):
            return getattr(pred, mode) if isinstance(pred, ThermalJointPairPredictions) else pred
        def masked_mean(diff, mask):
            if not torch.isfinite(diff[mask]).all():
                raise ValueError('Nonfinite prediction on a valid physical response receiver.')
            return torch.where(mask, diff, torch.zeros_like(diff)).square().sum() / mask.sum().clamp_min(1)
        thermal, null = [], []
        for mode in modes:
            base = branch(designs['baseline'], mode)
            for kind in ('geometry', 'heat'):
                trial = branch(designs[kind], mode)
                raw = pack['deltas'][kind]
                device = base.native_main['pred_field'].device
                def data(role, channel, shape):
                    values = torch.as_tensor(np.asarray(raw[role]['values'][:, channel]).copy(),
                                             device=device, dtype=base.native_main['pred_field'].dtype).reshape(shape)
                    mask = torch.as_tensor(np.asarray(raw[role]['valid_mask'][:, channel]).copy(), device=device).reshape(shape)
                    return values, mask
                valid_fluid = base.native_prepared.stencils['fluid'].valid & trial.native_prepared.stencils['fluid'].valid
                target, mask = data('fluid_fields', 4, (1, -1))
                diff = trial.native_main['pred_field'][..., 4] - base.native_main['pred_field'][..., 4] - target
                roles = [masked_mean(diff / self.response_packs.response_scales[kind]['fluid_temperature'], mask & valid_fluid)]
                count = base.native_prepared.source_present.shape[1]
                for channel, role in ((0, 'surface_temperature'), (1, 'q')):
                    target, mask = data('interface', channel, (1, count, 64))
                    mask = mask & base.native_prepared.stencils['surface'].valid.reshape_as(mask) & trial.native_prepared.stencils['surface'].valid.reshape_as(mask)
                    if channel == 1:
                        mask = mask & base.native_prepared.stencils['outside'].valid.reshape_as(mask) & trial.native_prepared.stencils['outside'].valid.reshape_as(mask)
                    errors = []
                    for source in ('direct', 'local'):
                        if source == 'direct':
                            lhs, rhs = trial.native_main['pred_interface'], base.native_main['pred_interface']
                        else:
                            lhs, rhs = designs[kind].local1[mode]['interface_pred'], designs['baseline'].local1[mode]['interface_pred']
                        errors.append(_module_mean((lhs[..., channel] - rhs[..., channel] - target) /
                            self.response_packs.response_scales[kind][role], mask).mean())
                    roles.append(sum(errors) / 2)
                target, mask = data('solid_temperature', 0, (1, count, 32))
                mask = mask & base.native_prepared.stencils['material'].valid.reshape_as(mask) & trial.native_prepared.stencils['material'].valid.reshape_as(mask)
                errors = []
                for source in ('direct', 'local'):
                    if source == 'direct':
                        lhs, rhs = trial.native_main['pred_internal_temperature'], base.native_main['pred_internal_temperature']
                    else:
                        lhs, rhs = designs[kind].local1[mode]['internal_temperature'], designs['baseline'].local1[mode]['internal_temperature']
                    errors.append(_module_mean((lhs[..., 0] - rhs[..., 0] - target) /
                        self.response_packs.response_scales[kind]['material_temperature'], mask).mean())
                roles.append(sum(errors) / 2)
                thermal.append(sum(roles) / len(roles))
                if kind == 'heat':
                    for channel in range(4):
                        _, mask = data('fluid_fields', channel, (1, -1))
                        diff = trial.native_main['pred_field'][..., channel] - base.native_main['pred_field'][..., channel]
                        null.append(masked_mean(diff / float(self.response_packs.null_flow_scales[channel]), mask & valid_fluid))
        return {'response/thermal': LossTerm(sum(thermal) / len(thermal), 1, .1),
                'response/heat_flow_null': LossTerm(sum(null) / len(null), 1, .05)}

    def loss_denominators(self, batches, phase='joint', arm=ARM_DETAIL):
        count = sum(len(batch.targets.case_ids) for batch in batches)
        result = {**{f'final/{role}': count for role in self.native_weights},
                  **{f'local/{role}': count for role in ('temperature/material', 'temperature/surface', 'q_proxy')},
                  'interface/consistency': count}
        if self.port_arm != 'C0':
            result['port/outside_temperature'] = count
        packs = sum(batch.targets.response_pack is not None for batch in batches)
        if packs:
            result.update({'response/thermal': packs, 'response/heat_flow_null': packs})
        return result

    def loss_metadata(self):
        result = {k: v for k, v in super().loss_metadata().items() if not k.startswith('initial/')}
        if self.port_arm != 'C0':
            result['interface/consistency']['formula'] = '64-angle direct target error minus predicted/local teacher response; teacher stopped.'
            result['port/outside_temperature'] = {'panel': 'native_prediction', 'label': 'outside temperature',
                'unit': 'scaled squared error', 'formula': '64-angle physical outside temperature; equal active module and case reductions.', 'weight': '0.05'}
        if self.port_arm == 'R64':
            for name, weight in (('response/thermal', '.1'), ('response/heat_flow_null', '.05')):
                result[name] = {'panel': 'response', 'label': name, 'unit': 'scaled squared error',
                    'formula': 'Four FIT families; saved physical pair masks; equal physical roles and paired read modes.', 'weight': weight}
        return result

    def completed_training_epoch(self, model, engine, epoch, output, row):
        super().completed_training_epoch(model, engine, epoch, output, row)
        if 'reader_epoch_context' in row:
            row['reader_epoch_context']['absolute_training_epoch'] = self.source_parent_age + int(epoch)
            row['reader_epoch_context']['additional_epochs_after_parent'] = int(epoch)
        if self.port_arm == 'R64':
            # The engine averages each named term over only updates where it
            # exists. Report its actual contribution averaged over all updates
            # so the recorded total is the real mean optimizer objective.
            count = int(row['optimizer_updates'])
            if count < 4:
                raise ValueError('R64 requires four distinct epoch-quarter response updates.')
            names = ('response/thermal', 'response/heat_flow_null')
            row['response_active_pack_mean'] = {name: row['train_losses'][name] for name in names}
            for name in names:
                row['train_losses'][name] *= 4.0 / count
            row['response_epoch_aggregation'] = {'active_packs': 4, 'optimizer_updates': count,
                'train_losses': 'weighted contribution averaged over all optimizer updates',
                'response_active_pack_mean': 'weighted mean over the four response updates'}


def build_thermal_port_response_task(recipe, *, arm=ARM_DETAIL, device='cpu'):
    spec = recipe['thermal_port_response']
    if spec.get('schema') != 'thermal_port_response_v1' or spec.get('enabled') is not True or spec.get('arm') not in ('C0', 'P64', 'R64'):
        raise ValueError('Expected C0, P64 or R64 Thermal port response recipe.')
    initialization = recipe.get('initialization', {})
    families = ['0001', '0318', '0333', '0348'] if spec['arm'] == 'R64' else []
    objective = recipe.get('objective', {})
    if (initialization.get('source_kind') != 'thermal_port_response_explicit_transfer_v1'
            or initialization.get('source_checkpoint') != spec.get('source_checkpoint')
            or initialization.get('source_parent_age') != spec.get('source_epoch')
            or initialization.get('pretrained_weights') is not True
            or initialization.get('source_arm') != arm
            or objective.get('response_training_families') != families
            or objective.get('response_fit') is not bool(families)):
        raise ValueError('Thermal port transfer or FIT response declaration is inconsistent.')
    source = load_trusted_checkpoint(spec['source_checkpoint'], map_location='cpu')
    if int(source['epoch']) != int(spec['source_epoch']):
        raise ValueError('Thermal source epoch differs from the declared transfer.')
    base = copy.deepcopy(recipe)
    base['initialization'].update(source_kind=DETAIL_SOURCE_KIND, source_checkpoint=None, pretrained_weights=False,
        source_parent_age=0, source_arm=f"scratch_common_seed{base['initialization']['seed']}")
    base['member_routing']['health'] = {'enabled': False}
    grid = spec.get('environment_grid_nx_ny', [24, 8])
    if grid not in ([24, 8], [12, 8], [12, 6]):
        raise ValueError('Unsupported R15-6 environment grid.')
    class GridAdapter(ThermalDetailAnchoredMemberRoutingAdapter):
        def __init__(self, *args, **kwargs):
            kwargs.update(environment_nx=grid[0], environment_ny=grid[1])
            super().__init__(*args, **kwargs)
    variants = {DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'adapter': GridAdapter, 'arm': arm}}
    model, provider = _build_thermal_member_routing_task(base, arm=arm, device=device,
        task_type=ThermalPortResponseTask, variants=variants, section_schema=DETAIL_CONFIG_SCHEMA,
        source_kind=DETAIL_SOURCE_KIND, lineage='Thermal one-pass port response', required_address_mode=None)
    source_recipe = source['experiment_identity']['recipe']
    if source_recipe['training']['membership'] != recipe['training']['membership'] or any(
        source_recipe['dataset'][k] != recipe['dataset'][k] for k in ('name', 'data_path', 'manifest_path')):
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
    return model, provider


def load_thermal_port_response_checkpoint(path, *, device='cpu'):
    """Reconstruct the native model from checkpoint metadata without loading data.

    The frozen local operator remains the explicitly declared external asset.
    The returned clock identifies the original source age plus this run's age.
    """
    saved = load_trusted_checkpoint(path, map_location='cpu')
    identity = saved.get('experiment_identity', {})
    recipe = identity.get('recipe', {})
    spec = recipe.get('thermal_port_response', {})
    adapter = identity.get('provider_identity', {}).get('adapter_state', {})
    if (identity.get('workflow') != 'relation_anchored_case_training_v1'
            or spec.get('schema') != 'thermal_port_response_v1' or spec.get('enabled') is not True
            or spec.get('arm') not in ('C0', 'P64', 'R64')
            or recipe.get('member_routing', {}).get('arm') != saved.get('arm')
            or adapter.get('schema') != ThermalDetailAnchoredMemberRoutingAdapter.SCHEMA
            or adapter.get('environment_grid_nx_ny') != spec.get('environment_grid_nx_ny', [24, 8])):
        raise ValueError('Checkpoint is not a self-consistent Thermal port response run.')
    model = ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(
        adapter, device=device, state_dict=saved['model_state_dict'])
    model.core.set_training_clock(int(spec['source_epoch']) + int(saved['epoch']), 0)
    model.eval()
    local = FrozenThermalLocal(spec['local_checkpoint']).to(device).eval()
    return model, local, saved
