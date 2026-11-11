"""Independent B0-to-R post-training with inherited native state and local history."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import MethodType

import torch

from honf_runtime.unified_training import TrainingEngine, _sampling_dataset_id
from .r16_10_family_task import build_thermal_r16_10_task
from .r16_10_native_task import ThermalR1610NativeTask

SOURCE_KIND = 'thermal_r16_10_b0_to_r_posttrain_v1'
SOURCE_EPOCH = 3000
POSTTRAIN_EPOCHS = 1000


def validate_parent(source):
    """Require the requested completed S B0 parent and its full native state."""
    identity = source['experiment_identity']
    recipe = identity['recipe']
    if (source.get('epoch') != SOURCE_EPOCH
            or recipe.get('r16_10_family', {}).get('arm') != 'B0'
            or recipe['r16_10_family'].get('population') != 600
            or recipe['training'].get('membership') != 'full_train600'
            or recipe['member_routing']['model'].get('access_measure') != 'inventory_unit_total'
            or len(identity['provider_identity'].get('training_case_ids', ())) != 600
            or [row['epoch'] for row in source.get('history', ())] != list(range(1, SOURCE_EPOCH + 1))):
        raise ValueError('Post-training requires the complete S B0 TRAIN600 epoch-3000 parent.')
    for name in ('model_state_dict', 'optimizer_state_by_name', 'rng_state', 'provider_training_state'):
        if not source.get(name):
            raise ValueError(f'Parent lacks its resumable {name}.')
    return identity


def make_recipe(source, source_checkpoint, *, name, output_root):
    """Retain the R recipe content while declaring a separate post-training lineage."""
    identity = validate_parent(source)
    recipe = copy.deepcopy(identity['recipe'])
    recipe['name'] = name
    recipe['initialization'].update(
        source_kind=SOURCE_KIND, source_checkpoint=str(Path(source_checkpoint).resolve()),
        pretrained_weights=True, source_arm='B0', source_parent_age=SOURCE_EPOCH,
        normalization_mode='inherited_train600_stats')
    recipe['optimizer'].update(kind='inherited_named_adamw_from_parent',
                               schedule='posttrain_constant_terminal_lr', lr=3e-5)
    recipe['storage']['output_root'] = str(Path(output_root).resolve())
    recipe['training'].update(local_horizon=POSTTRAIN_EPOCHS, optimizer_schedule_horizon=POSTTRAIN_EPOCHS,
        first_segment_epochs=100, checkpoint_epochs=list(range(100, POSTTRAIN_EPOCHS + 1, 100)),
        wall_clock_budget_seconds=180000 + math.ceil(
            float(source['provider_training_state'].get('training_elapsed_seconds', 0.))))
    recipe['r16_10_family']['arm'] = 'R'
    recipe['r16_10_posttrain'] = {
        'schema': SOURCE_KIND, 'source_epoch': SOURCE_EPOCH, 'additional_epochs': POSTTRAIN_EPOCHS,
        'native_sampler_epoch': '3000 + local_epoch', 'family_sampler_epoch': '10 + local_epoch',
        'auxiliary_ramp': 'min(local_epoch / 40, 1)', 'optimizer_state': 'inherited_named_adamw',
        'rng_state': 'inherited_parent', 'auxiliary_sampler': 'new_seed0',
        'coefficients': 'unchanged_original_S_B0_epoch10_calibration',
        'history': 'local_posttraining_only; parent history remains at source checkpoint',
    }
    return recipe


class PosttrainNativeTask(ThermalR1610NativeTask):
    def _validate_initialization(self, initialization):
        if (self.source_kind != SOURCE_KIND or self.source_parent_age != SOURCE_EPOCH
                or initialization.get('source_arm') != 'B0'
                or initialization.get('pretrained_weights') is not True
                or not initialization.get('source_checkpoint')
                or initialization.get('normalization_mode') != 'inherited_train600_stats'):
            raise ValueError('Post-training requires an explicit inherited B0 epoch-3000 initialization.')

    def _schedule_payload(self):
        return dict(peak_lr=3e-5, warmup_start_lr=3e-5, warmup_epochs=0,
                    hold_through_epoch=0, total_epochs=POSTTRAIN_EPOCHS, final_lr=3e-5)


class PosttrainEngine(TrainingEngine):
    def sampler_epoch(self, fit_epoch):
        return SOURCE_EPOCH + super().sampler_epoch(fit_epoch)

    def sampler_epoch_contract(self):
        return {'schema': SOURCE_KIND, 'source_parent_age': SOURCE_EPOCH,
                'fit_epoch_to_sampler_epoch': 'source_parent_age + local_epoch'}


def attach_posttrain_clock(provider, spec):
    """Use the R auxiliary ramp immediately without changing native sampling age."""
    original_auxiliary = provider.auxiliary_update_terms
    def auxiliary(self, model, *, epoch, update_index, update_count):
        yield from original_auxiliary(model, epoch=epoch + 10,
                                     update_index=update_index, update_count=update_count)
    provider.auxiliary_update_terms = MethodType(auxiliary, provider)
    original_identity = provider.identity_payload
    def identity(self):
        return {**original_identity(), 'r16_10_posttrain': copy.deepcopy(spec)}
    provider.identity_payload = MethodType(identity, provider)
    return provider


def build_posttrain_task(recipe, *, device='cpu', source=None, resume_payload=None):
    """Load the parent once for attachment; exact child resumes use their own state."""
    if (source is None) == (resume_payload is None):
        raise ValueError('Supply either the parent state or this child\'s own resume state.')
    if source is not None:
        parent = validate_parent(source)
        expected = make_recipe(source, recipe['initialization']['source_checkpoint'],
                               name=recipe['name'], output_root=recipe['storage']['output_root'])
        if recipe != expected:
            raise ValueError('Post-training changed an undeclared parent data, model or objective setting.')
    else:
        if resume_payload['experiment_identity']['recipe'] != recipe:
            raise ValueError('Post-training resume must retain its exact saved recipe.')
        parent = None
    model, provider = build_thermal_r16_10_task(
        recipe, arm=recipe['member_routing']['arm'], device=device, resume_payload=resume_payload,
        native_task_options={'task_type': PosttrainNativeTask, 'source_kind': SOURCE_KIND,
                             'lineage': 'R16-10 B0-to-R post-training'})
    if source is not None:
        current = provider.identity_payload()
        previous = parent['provider_identity']
        for key in ('training_case_ids', 'validation_case_ids', 'adapter_state', 'source_binding'):
            if current.get(key) != previous.get(key):
                raise ValueError(f'Post-training failed to retain parent {key}.')
        if _sampling_dataset_id(current) != parent['sampling_dataset_id']:
            raise ValueError('Post-training changed the parent native sampling namespace.')
        model.load_state_dict(source['model_state_dict'], strict=True)
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()
               if torch.is_tensor(v) and v.is_floating_point()):
            raise ValueError('Nonfinite parent model state.')
        provider.load_training_state_dict({
            'native': source['provider_training_state'],
            'r16_10_family': provider.family_objective.schedule.state_dict()})
    # The existing R implementation starts its auxiliary clock at epoch 11.
    # Advancing only that local clock makes the first post-training epoch use
    # a 1/40 ramp, while native case/query streams continue at parent age 3001.
    attach_posttrain_clock(provider, recipe['r16_10_posttrain'])
    return model, provider
