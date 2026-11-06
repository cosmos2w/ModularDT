"""P64-initialized, local-epoch R16-5 Thermal decision-learning forks."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import SamplingKey, TrainingEngine, _sampling_dataset_id

from ..data.datasets import H5Normalizer
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from .joint_task import _read_selected_cases
from .r13_6_member_routing_task import _build_thermal_member_routing_task
from .r13_8_anchored_task import _VARIANTS, ARM_DETAIL, DETAIL_CONFIG_SCHEMA, DETAIL_CORE_SCHEMA
from .thermal_state_response_task import _restore_response_scales

P64_RUN = Path('/data2/wanglz/ModularDT/r15_thermal_wind/r15_6/thermal/train600/'
               'thermal_r15_6_p64_grid192_train600_r14e5000_v1')
SOURCE_CHECKPOINT = str(P64_RUN / 'checkpoints/epoch_0100_model.pt')
P64_IDENTITY = str(P64_RUN / 'configs/experiment_identity.json')
H0_IDENTITY = ('/data2/wanglz/ModularDT/r16_1_thermal_20261104/train600/'
               'thermal_state_response_h0_train600_s0/configs/experiment_identity.json')
SOURCE_AGE = 5100
SOURCE_KIND = 'thermal_r16_5_p64_weight_only_v1'
FORK_SCHEMA = 'thermal_r16_5_p64_warm_fork_v1'
DECISION_SCHEMA = 'thermal_r16_5_decision_v1'


def _source_initialization():
    return {'source_kind': SOURCE_KIND, 'source_checkpoint': SOURCE_CHECKPOINT,
            'pretrained_weights': True, 'source_arm': ARM_DETAIL,
            'source_parent_age': SOURCE_AGE, 'normalization_mode': 'inherited_train600_stats', 'seed': 0}


def _read_identity(path):
    with Path(path).open() as stream:
        return json.load(stream)


def validate_source_identities(p64, h0):
    """Check the two saved providers before combining their distinct objectives."""
    p, h = p64['provider_identity'], h0['provider_identity']
    p_recipe, h_recipe = p64['recipe'], h0['recipe']
    port = p_recipe.get('thermal_port_response', {})
    response = h_recipe.get('thermal_state_response', {})
    optimizer = p_recipe.get('optimizer', {})
    expected_optimizer = {'kind': 'fresh_adamw_from_config', 'schedule': 'constant_config_lr',
                          'lr': 1e-4, 'weight_decay': 1e-5, 'betas': [0.9, 0.999],
                          'eps': 1e-8, 'gradient_clip': 1.0}
    if (port.get('schema') != 'thermal_port_response_v1' or port.get('enabled') is not True
            or port.get('arm') != 'P64' or port.get('source_epoch') != 5000
            or p_recipe.get('initialization', {}).get('source_parent_age') != 5000
            or p_recipe['training'].get('membership') != 'full_train600'
            or p_recipe['member_routing'].get('arm') != ARM_DETAIL
            or optimizer != expected_optimizer
            or response.get('schema') != 'thermal_state_response_v1'
            or response.get('enabled') is not True or response.get('families_per_epoch') != 4
            or response.get('pilot') is not None or response.get('finite_response_coefficient', .1) != .1):
        raise ValueError('P64 or original H0 saved identity has unexpected training semantics.')
    for key in ('training_case_ids', 'validation_case_ids'):
        if p.get(key) != h.get(key):
            raise ValueError(f'P64 and H0 disagree on {key}.')
    if (len(p['training_case_ids']) != 600 or len(p['validation_case_ids']) != 89
            or len(h.get('response_training_family_ids', [])) != 48
            or set(h_recipe['objective']['response_training_families']) != set(h['response_training_family_ids'])
            or p_recipe['objective']['native_weights'] != h_recipe['objective']['native_weights']):
        raise ValueError('Source TRAIN600, VAL89, native weights or original FIT48 differ.')
    for key in ('normalization_stats', 'heat_scale', 'native_grid_nx_ny', 'environment_grid_nx_ny'):
        if p['adapter_state'].get(key) != h['adapter_state'].get(key):
            raise ValueError(f'P64 and H0 disagree on adapter {key}.')
    for key in ('dataset_path', 'dataset_size_bytes', 'dataset_mtime_ns',
                'catalog_metadata_sha256', 'manifest_sha256'):
        if p['source_binding'].get(key) != h['source_binding'].get(key):
            raise ValueError(f'P64 and H0 disagree on data binding {key}.')
    if (_sampling_dataset_id(p) != p64.get('sampling_dataset_id')
            or p64['sampling_dataset_id'] != h0.get('sampling_dataset_id')):
        raise ValueError('P64 and H0 disagree on the native sampler namespace.')
    for identity in (p64, h0):
        engine = identity['engine_config']
        if (engine.get('sampling_version') != SamplingKey.CASE_EPOCH_VERSION
                or engine.get('deterministic_case_order') is not True
                or engine.get('seed') != 0 or engine.get('training_mode') != 'joint'):
            raise ValueError('Saved source sampler is not the declared deterministic joint stream.')
    scales = _restore_response_scales(h['response_scales'])
    sensitivity = dict(response['functional_floors'])
    if set(sensitivity) != {'J', 'P'} or any(sensitivity[k] != scales[k]['floor'] for k in sensitivity):
        raise ValueError('H0 functional sensitivity differs from its original FIT provider scales.')
    return {'p64_provider': p, 'h0_provider': h, 'response_scales': scales,
            'response_spec': copy.deepcopy(response), 'sensitivity': sensitivity}


def make_research_metadata(p64, h0):
    """Compute the shared research bands once from the original FIT labels."""
    from .r16_5_decision_task import decision_research_metadata
    from .thermal_state_response_task import ThermalResponseCorpus, load_saved_functional_responses

    source = validate_source_identities(p64, h0)
    spec = source['response_spec']
    corpus = ThermalResponseCorpus(spec['recipe_path'], spec['ledger_path'])
    if list(corpus.family_ids('fit')) != source['h0_provider']['response_training_family_ids']:
        raise ValueError('Saved H0 and original FIT family membership differ.')
    fit_file = Path(spec['recipe_path']).parent.parent / 'evidence/matched200/epoch200/fit/response_evaluation.json'
    labels = load_saved_functional_responses(fit_file, corpus, role='fit')
    metadata = decision_research_metadata(labels, source['sensitivity'], source['response_scales'])
    return {**metadata, 'fit_labels': str(fit_file), 'fit_families': list(corpus.family_ids('fit')),
            'fit_action_count': len(labels), 'selection': 'Q25 of abs(delta) greater than 4 * sensitivity'}


def make_warm_fork_recipe(p64, h0, research_metadata, *, arm, name, output_root, membership,
                          local_horizon, first_segment_epochs, wall_clock_budget_seconds=86400,
                          clean_save_reserve_seconds=600):
    """Derive a fork recipe; neither source checkpoint nor training output is written."""
    source = validate_source_identities(p64, h0)
    bands = research_metadata.get('research_scales', {})
    functional_scales = {name: source['response_scales'][name]['scale'] for name in ('J', 'P')}
    if (research_metadata.get('sensitivity') != source['sensitivity']
            or research_metadata.get('response_scales') != functional_scales
            or set(bands) != {'J', 'P'}
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in bands.values())):
        raise ValueError('The shared FIT research receipt differs from original H0 scales or sensitivity.')
    if arm not in ('PDelta', 'PDeltaD'):
        raise ValueError('R16-5 arm must be PDelta or PDeltaD.')
    if (membership not in ('fixed25_train150_dev22', 'full_train600')
            or type(local_horizon) is not int or type(first_segment_epochs) is not int
            or not 1 <= first_segment_epochs <= local_horizon <= 500
            or (membership == 'full_train600' and (local_horizon, first_segment_epochs) != (100, 100))):
        raise ValueError('Declare bounded qualification or the initial full-TRAIN600 100-epoch segment.')
    if (type(wall_clock_budget_seconds) is not int or type(clean_save_reserve_seconds) is not int
            or not 0 < clean_save_reserve_seconds < wall_clock_budget_seconds <= 86400):
        raise ValueError('Declare a positive clean-save reserve and an at-most-24-hour run budget.')
    recipe = copy.deepcopy(p64['recipe'])
    recipe['name'] = str(name)
    recipe['storage']['output_root'] = str(output_root)
    recipe['storage']['formal_forward_run_alias'] = membership == 'full_train600'
    recipe['initialization'] = _source_initialization()
    recipe['training'].update(membership=membership, local_horizon=local_horizon,
                              first_segment_epochs=first_segment_epochs, checkpoint_epochs=[local_horizon],
                              wall_clock_budget_seconds=wall_clock_budget_seconds,
                              clean_save_reserve_seconds=clean_save_reserve_seconds)
    recipe['objective']['response_fit'] = True
    recipe['objective']['response_training_families'] = list(source['h0_provider']['response_training_family_ids'])
    recipe['thermal_state_response'] = source['response_spec']
    recipe['r16_5_warm_fork'] = {'schema': FORK_SCHEMA, 'p64_identity': P64_IDENTITY,
                                 'h0_identity': H0_IDENTITY}
    recipe['r16_5_decision'] = {'schema': DECISION_SCHEMA, 'arm': arm,
                                'sensitivity': source['sensitivity'], 'research_scales': dict(bands)}
    return recipe


def validate_source_and_recipe(source, recipe, p64, h0):
    """Reject implicit source substitution and any unintended scientific recipe change."""
    training = recipe['training']
    # The bands are computed once before recipe construction and retained in each recipe.
    metadata = {'sensitivity': recipe['r16_5_decision']['sensitivity'],
                'response_scales': {name: h0['provider_identity']['response_scales'][name]['scale']
                                    for name in ('J', 'P')},
                'research_scales': recipe['r16_5_decision']['research_scales']}
    expected = make_warm_fork_recipe(p64, h0, metadata, arm=recipe['r16_5_decision']['arm'],
        name=recipe['name'], output_root=recipe['storage']['output_root'],
        membership=training['membership'], local_horizon=training['local_horizon'],
        first_segment_epochs=training['first_segment_epochs'],
        wall_clock_budget_seconds=training['wall_clock_budget_seconds'],
        clean_save_reserve_seconds=training['clean_save_reserve_seconds'])
    if recipe != expected:
        raise ValueError('R16-5 recipe differs from the derived P64 plus original FIT recipe.')
    if source is not None and (source.get('epoch') != 100 or source.get('current_epoch') != 100
            or source.get('arm') != ARM_DETAIL
            or [row.get('epoch') for row in source.get('history', [])] != list(range(1, 101))
            or json.dumps(source.get('experiment_identity'), sort_keys=True) != json.dumps(p64, sort_keys=True)):
        raise ValueError('Source must be the complete literal P64 epoch100 checkpoint.')
    return validate_source_identities(p64, h0)


def initialize_shared_weights(model, source_state):
    """Check every parameter and buffer name, shape and dtype before strict load."""
    target = model.state_dict()
    if set(target) != set(source_state):
        raise ValueError('P64 and fork tensor inventories differ.')
    for name, value in source_state.items():
        if not torch.is_tensor(value) or value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f'P64 and fork tensor shape or dtype differs for {name}.')
    model.load_state_dict(source_state, strict=True)
    return {'tensor_names': sorted(target), 'tensors': {
        name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
        for name, value in sorted(source_state.items())}, 'trainable_names': sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad)}


class ThermalR165WarmForkEngine(TrainingEngine):
    """Retain new local histories while continuing the source native sampler age."""
    def sampler_epoch(self, fit_epoch):
        return SOURCE_AGE + super().sampler_epoch(fit_epoch)

    def sampler_epoch_contract(self):
        return {'schema': 'thermal_r16_5_source_age_sampler_v1', 'source_parent_age': SOURCE_AGE,
                'fit_epoch_to_sampler_epoch': 'source_parent_age + local_epoch'}


def build_thermal_r16_5_warm_fork_task(recipe, *, arm=None, device='cpu', resume_payload=None,
                                       response_scales=None):
    """Build a literal P64 fork, or reconstruct an existing fork from its own state."""
    from .r16_5_decision_task import ThermalR165DecisionTask

    p64, h0 = _read_identity(P64_IDENTITY), _read_identity(H0_IDENTITY)
    source = None if resume_payload is not None else load_trusted_checkpoint(SOURCE_CHECKPOINT, map_location='cpu')
    references = validate_source_and_recipe(source, recipe, p64, h0)
    if arm is not None and arm != ARM_DETAIL:
        raise ValueError('R16-5 must retain the literal P64 model arm.')
    if response_scales is not None and _restore_response_scales(response_scales) != references['response_scales']:
        raise ValueError('Saved run response scales differ from original H0 FIT scales.')
    if resume_payload is not None:
        saved_identity = resume_payload.get('experiment_identity', {})
        if (saved_identity.get('recipe') != recipe or resume_payload.get('arm') != ARM_DETAIL
                or resume_payload.get('epoch', 0) < 1):
            raise ValueError('Resume requires this fork\'s own saved recipe, arm and local checkpoint.')
        saved_provider = saved_identity.get('provider_identity', {})
        if (saved_provider.get('response_scales') != references['response_scales']
                or saved_provider.get('adapter_state') != references['p64_provider']['adapter_state']):
            raise ValueError('Resume checkpoint changed the inherited P64 adapter or original FIT scales.')
    source_adapter = references['p64_provider']['adapter_state']
    grid = source_adapter['environment_grid_nx_ny']

    class GridAdapter(ThermalDetailAnchoredMemberRoutingAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, 'environment_nx': grid[0], 'environment_ny': grid[1]})

    model, provider = _build_thermal_member_routing_task(
        recipe, arm=ARM_DETAIL, device=device, task_type=ThermalR165DecisionTask,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'adapter': GridAdapter, 'arm': ARM_DETAIL}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=SOURCE_KIND, lineage='R16-5 P64 weight-only fork',
        required_address_mode=None, saved_adapter_state=source_adapter)
    adapter = model.state_payload()
    for key in ('normalization_stats', 'heat_scale', 'native_grid_nx_ny', 'environment_grid_nx_ny'):
        if adapter[key] != source_adapter[key]:
            raise ValueError(f'Fork failed to preserve P64 adapter {key}.')
    for key in ('dataset_path', 'dataset_size_bytes', 'dataset_mtime_ns', 'catalog_metadata_sha256', 'manifest_sha256'):
        if provider.source_binding[key] != references['p64_provider']['source_binding'][key]:
            raise ValueError(f'Fork data binding differs from P64 {key}.')
    source_ids = references['p64_provider']['training_case_ids']
    if (recipe['training']['membership'] == 'full_train600' and list(provider.training_case_ids) != source_ids
            or recipe['training']['membership'] == 'fixed25_train150_dev22'
            and (len(provider.training_case_ids) != 150 or not set(provider.training_case_ids) <= set(source_ids))):
        raise ValueError('Fork TRAIN membership differs from its declared source population.')
    val_ids = tuple(case_id for case_id in provider.catalog_case_ids
                    if provider.case_split_by_id[case_id] == 'test' and case_id != '0273')
    if list(val_ids) != references['p64_provider']['validation_case_ids']:
        raise ValueError('Fork validation membership differs from P64 VAL89.')
    normalizer = H5Normalizer({key: np.asarray(value, dtype=np.float32)
                               for key, value in source_adapter['normalization_stats'].items()})
    provider.validation_cases = _read_selected_cases(provider.data_path, val_ids, split='test', normalizer=normalizer,
                                                      include_grid=False, include_raster_geometry=False)
    provider.validation_case_ids = val_ids
    provider._validation_index_by_id = {value: index for index, value in enumerate(val_ids)}
    provider.validation_scope = 'canonical89_exposed_original_test_excluding_0273'
    provider._validation_batches_cache = None
    provider.configure_port_response(recipe)
    provider.source_parent_age = SOURCE_AGE
    provider.configure_state_response(recipe['thermal_state_response'], response_scales=references['response_scales'])
    provider.configure_decision(recipe['r16_5_decision'])
    provider.r16_5_source_tensor_inventory = initialize_shared_weights(
        model, (source if resume_payload is None else resume_payload)['model_state_dict'])
    if (list(provider.response_corpus.family_ids('fit')) != references['h0_provider']['response_training_family_ids']
            or _sampling_dataset_id(provider.identity_payload()) != p64['sampling_dataset_id']):
        raise ValueError('Fork changed original FIT48 or native sampling namespace.')
    return model, provider
