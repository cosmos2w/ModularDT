"""P64-initialized, matched 48-action R16-8 training task."""
from __future__ import annotations
import copy
import json
import os
import numpy as np
import torch
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from honf_runtime.compat import load_trusted_checkpoint
from .r16_5_warm_fork import *
from .r16_5_warm_fork import (_read_identity, _build_thermal_member_routing_task, _VARIANTS,
    _restore_response_scales, _sampling_dataset_id)
from .r16_5_decision_task import _DirectResponseProvider, _accumulate_gradients
from .r16_6_balanced_task import ThermalR166BalancedTask, make_balanced_recipe, balanced_snapshot_groups
from .thermal_state_response_task import (ACTIONS, ThermalStateResponseTask, response_pair,
    predict_response_pair, response_loss_terms)

COEFFICIENT = 0.0014978225391285025


def action_slots(n):
    if type(n) is not int or n < 1:
        raise ValueError('Expected a positive native batch count.')
    return tuple(a * n // 48 for a in range(48))


def selected_families(families, epoch, seed, count):
    families = tuple(sorted(map(str, families)))
    if len(set(families)) != len(families) or len(families) % count or type(epoch) is not int or epoch < 1:
        raise ValueError('Expected unique families, an integral cycle and a positive epoch.')
    cycle_length = len(families) // count
    cycle, offset = divmod(epoch - 1, cycle_length)
    order = np.random.default_rng(np.random.SeedSequence([int(seed), cycle])).permutation(len(families))
    return tuple(families[int(i)] for i in order[offset * count:(offset + 1) * count])


def make_content_recipe(p64, h0, *, arm, name, output_root, membership, local_horizon,
                        first_segment_epochs, corpus_manifest=None, wall_clock_budget_seconds=86400,
                        clean_save_reserve_seconds=600):
    if arm not in ('C8', 'S8'):
        raise ValueError('Expected C8 or S8.')
    if membership == 'full_train600':
        if (local_horizon, first_segment_epochs) != (500, 100) or not corpus_manifest:
            raise ValueError('Formal training requires a physical corpus and 500/100 epochs.')
        base_horizon, base_first = 100, 100
    elif membership == 'fixed25_train150_dev22':
        if type(local_horizon) is not int or type(first_segment_epochs) is not int or not 1 <= first_segment_epochs <= local_horizon <= 40:
            raise ValueError('Qualification horizon must be 1 to 40 epochs.')
        if corpus_manifest is not None:
            raise ValueError('Qualification reuses existing references before new corpus generation.')
        base_horizon, base_first = local_horizon, first_segment_epochs
    else:
        raise ValueError('Unknown membership.')
    recipe = make_balanced_recipe(p64, h0, arm='C', coefficient=COEFFICIENT, name=name,
        output_root=output_root, membership=membership, local_horizon=base_horizon,
        first_segment_epochs=base_first, wall_clock_budget_seconds=wall_clock_budget_seconds,
        clean_save_reserve_seconds=clean_save_reserve_seconds)
    recipe.pop('r16_6_balanced')
    if membership == 'full_train600':
        recipe['storage']['max_output_bytes'] = 4 * 1024 ** 3
    recipe['r16_8_content'] = {'arm': arm, 'coefficient': COEFFICIENT, 'source_age': SOURCE_AGE,
        'action_placement': 'floor(a*N/48)', 'action_weight': 'N/48', 'corpus_manifest': corpus_manifest}
    recipe['training'].update(local_horizon=local_horizon, first_segment_epochs=first_segment_epochs,
        checkpoint_epochs=list(range(100, 501, 100)) if membership == 'full_train600' else [local_horizon],
        monitor_every=100)
    return recipe


def validate_content_recipe(source, recipe, p64, h0):
    spec, t = recipe['r16_8_content'], recipe['training']
    expected = make_content_recipe(p64, h0, arm=spec['arm'], name=recipe['name'],
        output_root=recipe['storage']['output_root'], membership=t['membership'],
        local_horizon=t['local_horizon'], first_segment_epochs=t['first_segment_epochs'],
        corpus_manifest=spec['corpus_manifest'], wall_clock_budget_seconds=t['wall_clock_budget_seconds'],
        clean_save_reserve_seconds=t['clean_save_reserve_seconds'])
    if recipe != expected:
        raise ValueError('R16-8 recipe differs from the declared content recipe.')
    if source is not None and (source.get('epoch') != 100 or source.get('current_epoch') != 100
        or source.get('arm') != ARM_DETAIL or [r.get('epoch') for r in source.get('history', [])] != list(range(1, 101))
        or json.dumps(source.get('experiment_identity'), sort_keys=True) != json.dumps(p64, sort_keys=True)):
        raise ValueError('Expected literal completed P64 source.')
    return validate_source_identities(p64, h0)


def enable_content(model, arm):
    from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
    before = set(model.state_dict())
    routes = [m for m in model.modules() if isinstance(m, SpatialRouteAttentionV2)]
    if not routes:
        raise ValueError('P64 has no contextual attention routes.')
    for route in routes:
        route.enable_content_replacement(arm.lower())
    state = model.state_dict()
    new = set(state) - before
    if not new or any('.content_replacement.' not in name for name in new):
        raise ValueError('Unexpected content tensor inventory.')
    if any(torch.count_nonzero(state[name]).item() != 0 for name in new):
        raise ValueError('Content tensors must initialize to zero.')
    return {'inherited_names': sorted(before), 'new_names': sorted(new)}


class ThermalR168ContentTask(ThermalR166BalancedTask):
    def validation_metrics(self, predictions, targets, auxiliary_state):
        # Preserve the ordinary native masks/reductions but select checkpoints
        # using direct joint fields, including material and interface outputs.
        from .joint_member_routing_task import ThermalJointPairPredictions
        from .r13_8_anchored_task import ThermalR138AnchoredMemberRoutingTask
        native = predictions.restricted if isinstance(predictions, ThermalJointPairPredictions) else predictions
        return ThermalR138AnchoredMemberRoutingTask.validation_metrics(self, native, targets, auxiliary_state)

    def _prepare_snapshot(self, model, epoch):
        if self._snapshot_epoch != epoch:
            self._snapshot_epoch = epoch
            groups = balanced_snapshot_groups(model)
            content = [(n, p) for n, p in model.named_parameters()
                       if p.requires_grad and '.content_replacement.' in n]
            names = {n for n, _ in content}
            self._snapshot_groups = {k: [(n, p) for n, p in v if n not in names]
                                     for k, v in groups.items()}
            self._snapshot_groups['content_parameters'] = content
            self._snapshot_buffers = {n: {} for n in ('endpoint', 'delta', 'decision', 'native', 'port')}

    def configure_content(self, spec, *, qualification):
        self.configure_balanced(spec)
        self.qualification = qualification
        if qualification:
            self.new_corpus = None
        else:
            from .r16_8_corpus import ThermalR168Corpus
            self.new_corpus = ThermalR168Corpus(spec['corpus_manifest'])
            if len(self.new_corpus.family_ids('fit')) != 96:
                raise ValueError('Expected 96 physical FIT families.')

    def _action_schedule(self, epoch):
        old = [(self.response_corpus, f, a, 'original_fit') for f in
               self.response_corpus.fit_families_for_epoch(self.fit_sampler_epoch(epoch), self.seed) for a in ACTIONS]
        if self.qualification:
            remaining = sorted(set(self.response_corpus.family_ids('fit')) - {f for _, f, _, _ in old})
            order = np.random.default_rng(np.random.SeedSequence([int(self.seed), int(epoch), 168])).permutation(len(remaining))
            families = tuple(remaining[int(i)] for i in order[:8])
            new = [(self.response_corpus, f, a, 'qualification_reuse') for f in families for a in ACTIONS]
        else:
            families = selected_families(self.new_corpus.family_ids('fit'), epoch, self.seed, 8)
            new = [(self.new_corpus, f, a, 'new_fit') for f in families
                   for a in self.new_corpus.actions_for_family(f)]
        if len(old) != 16 or len(new) != 32:
            raise ValueError('Expected 16 original and 32 new actions.')
        return old + new

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if self._placement_epoch != epoch:
            self._placement_epoch, self._placements = epoch, []
        for index, (corpus, family, action, role) in enumerate(self._action_schedule(epoch)):
            if action_slots(update_count)[index] != update_index:
                continue
            if corpus is self.response_corpus:
                pair = response_pair(self, corpus, family, action, role='fit')
                labels = self.response_labels[(family, action)]
            else:
                pair = corpus.pair(self, family, action, role='fit')
                labels = corpus.physical_delta(family, action, role='fit')
            losses = predict_response_pair(_DirectResponseProvider(self, epoch), model, pair,
                self.response_scales, labels, self.response_spec['tau'])
            weight = update_count / 48
            terms = response_loss_terms(losses['endpoint'], losses['field_delta'],
                {'J': losses['J_delta'], 'P': losses['P_delta']}, multiplier=weight,
                finite_response_coefficient=COEFFICIENT)
            if self._want_snapshot(epoch, update_index, update_count):
                self._prepare_snapshot(model, epoch)
                _accumulate_gradients(self._snapshot_buffers['endpoint'], .25 * losses['endpoint'] * weight, self._snapshot_groups)
                _accumulate_gradients(self._snapshot_buffers['delta'], COEFFICIENT * weight *
                    (.8 * losses['field_delta'] + .1 * losses['J_delta'] + .1 * losses['P_delta']), self._snapshot_groups)
            self._record_response_action(epoch, family, action, role)
            if family not in self._response_epoch_families:
                self._response_epoch_families.append(family)
            self._placements.append({'index': index, 'batch': update_index, 'family_id': str(family), 'action': action, 'role': role})
            yield terms, {**losses['work'], 'response_actions': 1}

    def auxiliary_diagnostic_values(self, terms, *, update_count):
        return {name: float(term.numerator.detach()) * (update_count / 48)
                for name, term in terms.items() if not name.endswith('/endpoint')}

    def loss_metadata(self):
        result = super().loss_metadata()
        for name in self.auxiliary_loss_names:
            result[name]['formula'] = 'equal mean over 16 original and 32 additional actions; paired full and restricted native reads'
        return result

    def completed_training_epoch(self, model, engine, epoch, output, row):
        super().completed_training_epoch(model, engine, epoch, output, row)
        for suffix in ('placements', 'clipping', 'query_ids', 'learning_snapshot'):
            old = 'r16_6_' + suffix
            if old in row:
                row['r16_8_' + suffix] = row.pop(old)

    def identity_payload(self):
        result = super().identity_payload()
        result['r16_8_content'] = copy.deepcopy(self.balanced_spec)
        result['r16_8_native_validation'] = 'direct joint fields with native masks; restricted read for checkpoint selection'
        result['r16_8_action_population'] = '16 original plus 32 qualification-reused' if self.qualification else '16 original plus 32 new FIT'
        return result

def validate_validation_membership(provider, membership, p64_provider):
    """Retain the builder's fixed DEV22 or verify the literal formal VAL89."""
    if membership == 'fixed25_train150_dev22':
        from ..data.development_split import development_case_ids
        expected = tuple(map(str, development_case_ids(provider.manifest, 'test')))
        if len(expected) != 22 or tuple(provider.validation_case_ids) != expected:
            raise ValueError('Qualification validation must retain the fixed DEV22 membership.')
    elif membership == 'full_train600':
        expected = tuple(case_id for case_id in provider.catalog_case_ids
                         if provider.case_split_by_id[case_id] == 'test' and case_id != '0273')
        if (list(expected) != p64_provider['validation_case_ids']
                or tuple(provider.validation_case_ids) != expected):
            raise ValueError('Fork validation membership differs from P64 VAL89.')
    else:
        raise ValueError('Unknown R16-8 validation membership.')


def build_thermal_r16_8_task(recipe, *, arm=None, device='cpu', resume_payload=None,
                                       response_scales=None):
    """Build a literal P64 fork, or reconstruct an existing fork from its own state."""
    configure_r16_5_execution(recipe, device=device)

    p64, h0 = _read_identity(P64_IDENTITY), _read_identity(H0_IDENTITY)
    source = None if resume_payload is not None else load_trusted_checkpoint(SOURCE_CHECKPOINT, map_location='cpu')
    references = validate_content_recipe(source, recipe, p64, h0)
    if arm is not None and arm != ARM_DETAIL:
        raise ValueError('R16-8 must retain the literal P64 model arm.')
    if response_scales is not None and _restore_response_scales(response_scales) != references['response_scales']:
        raise ValueError('Saved run response scales differ from original H0 FIT scales.')
    if resume_payload is not None:
        saved_identity = resume_payload.get('experiment_identity', {})
        if (saved_identity.get('recipe') != recipe or resume_payload.get('arm') != ARM_DETAIL
                or resume_payload.get('epoch', 0) < 1):
            raise ValueError('Resume requires this fork\'s own saved recipe, arm and local checkpoint.')
        saved_provider = saved_identity.get('provider_identity', {})
        if (saved_provider.get('response_scales') != references['response_scales']
                or any(saved_provider.get('adapter_state', {}).get(k) != references['p64_provider']['adapter_state'].get(k)
                       for k in ('normalization_stats', 'heat_scale', 'native_grid_nx_ny', 'environment_grid_nx_ny'))):
            raise ValueError('Resume checkpoint changed the inherited P64 adapter or original FIT scales.')
    source_adapter = references['p64_provider']['adapter_state']
    grid = source_adapter['environment_grid_nx_ny']

    class GridAdapter(ThermalDetailAnchoredMemberRoutingAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, 'environment_nx': grid[0], 'environment_ny': grid[1]})

    model, provider = _build_thermal_member_routing_task(
        recipe, arm=ARM_DETAIL, device=device, task_type=ThermalR168ContentTask,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'adapter': GridAdapter, 'arm': ARM_DETAIL}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=SOURCE_KIND, lineage='R16-8 P64 weight-only fork',
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
    validate_validation_membership(provider, recipe['training']['membership'], references['p64_provider'])
    provider.configure_port_response(recipe)
    provider.source_parent_age = SOURCE_AGE
    provider.configure_state_response(recipe['thermal_state_response'], response_scales=references['response_scales'])
    provider.configure_content(recipe['r16_8_content'], qualification=recipe['training']['membership'] != 'full_train600')
    if recipe['execution'].get('deterministic_algorithms') is True:
        provider.r16_5_execution = {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if resume_payload is None:
        model.load_state_dict(source['model_state_dict'], strict=True)
    inventory = enable_content(model, recipe['r16_8_content']['arm'])
    if resume_payload is None:
        provider.r16_5_source_tensor_inventory = inventory
    else:
        model.load_state_dict(resume_payload['model_state_dict'], strict=True)
        provider.r16_5_source_tensor_inventory = copy.deepcopy(saved_provider['r16_6_source_tensor_inventory'])
        if provider.r16_5_source_tensor_inventory != inventory:
            raise ValueError('Resume tensor inventory differs from this content architecture.')
    if (list(provider.response_corpus.family_ids('fit')) != references['h0_provider']['response_training_family_ids']
            or _sampling_dataset_id(provider.identity_payload()) != p64['sampling_dataset_id']):
        raise ValueError('Fork changed original FIT48 or native sampling namespace.')
    return model, provider
