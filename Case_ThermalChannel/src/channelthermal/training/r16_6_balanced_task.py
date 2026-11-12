"""Balanced, source-initialized R16-6 joint field training."""
from __future__ import annotations
import copy
import math
import json
import torch
import numpy as np
from ..data.datasets import H5Normalizer
from ..member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.unified_training import capture_rng_state, restore_rng_state
from .r16_5_warm_fork import *
from .r16_5_warm_fork import (_read_identity, _build_thermal_member_routing_task, _VARIANTS,
    _read_selected_cases, _restore_response_scales, _sampling_dataset_id)
from .r16_5_decision_task import (ThermalR165DecisionTask, _DirectResponseProvider,
    _accumulate_gradients, _parameter_groups)
from .thermal_state_response_task import (ThermalStateResponseTask, ACTIONS, response_pair,
    predict_response_pair, response_loss_terms)


def action_slots(n):
    if type(n) is not int or n < 1:
        raise ValueError('Expected a positive number of native batches.')
    return tuple(a * n // 16 for a in range(16))


def calibration_slots():
    slots = action_slots(13)
    counts = [slots.count(i) for i in range(13)]
    singles = [i for i, n in enumerate(counts) if n == 1]
    omitted = singles[-2]  # final 24-case batch is retained
    return tuple(i for i in range(13) if i != omitted)


def make_balanced_recipe(p64, h0, *, arm, coefficient, name, output_root, membership,
                         local_horizon, first_segment_epochs, wall_clock_budget_seconds=86400,
                         clean_save_reserve_seconds=600):
    if arm not in ('C', 'M') or not math.isfinite(coefficient) or not 0 < coefficient <= .1:
        raise ValueError('Expected C/M and the positive calibrated coefficient at most 0.1.')
    source = validate_source_identities(p64, h0)
    meta = {'sensitivity': source['sensitivity'],
            'response_scales': {k: source['response_scales'][k]['scale'] for k in ('J', 'P')},
            'research_scales': {'J': 1., 'P': 1.}}
    recipe = make_warm_fork_recipe(p64, h0, meta, arm='PDelta', name=name, output_root=output_root,
        membership=membership, local_horizon=local_horizon, first_segment_epochs=first_segment_epochs,
        wall_clock_budget_seconds=wall_clock_budget_seconds, clean_save_reserve_seconds=clean_save_reserve_seconds)
    recipe.pop('r16_5_decision')
    recipe['r16_6_balanced'] = {'arm': arm, 'coefficient': coefficient,
        'source_age': SOURCE_AGE, 'action_placement': 'floor(a*N/16)', 'action_weight': 'N/16'}
    recipe['optimizer']['lr'] = 3e-5
    if arm == 'M':
        recipe['member_routing']['model']['base_model_config']['contextual_message'] = True
    return recipe


def validate_balanced_recipe(source, recipe, p64, h0):
    spec = recipe['r16_6_balanced']; t = recipe['training']
    expected = make_balanced_recipe(p64, h0, arm=spec['arm'], coefficient=spec['coefficient'],
        name=recipe['name'], output_root=recipe['storage']['output_root'], membership=t['membership'],
        local_horizon=t['local_horizon'], first_segment_epochs=t['first_segment_epochs'],
        wall_clock_budget_seconds=t['wall_clock_budget_seconds'], clean_save_reserve_seconds=t['clean_save_reserve_seconds'])
    if recipe != expected:
        raise ValueError('R16-6 recipe differs from the declared balanced source recipe.')
    if source is not None and (source.get('epoch') != 100 or source.get('current_epoch') != 100
        or source.get('arm') != ARM_DETAIL or [r.get('epoch') for r in source.get('history', [])] != list(range(1, 101))
        or json.dumps(source.get('experiment_identity'), sort_keys=True) != json.dumps(p64, sort_keys=True)):
        raise ValueError('Expected literal completed P64 source.')
    return validate_source_identities(p64, h0)


def initialize_balanced_weights(arm, model, source_state):
    target = model.state_dict()
    new = set(target) - set(source_state)
    def message_name(name):
        parts = name.split('.')
        return ('context_blocks' in parts and 'pair_message' in parts and
                parts[parts.index('pair_message') + 1] in ('receiver', 'donor', 'geometry', 'bias', 'output'))
    if set(source_state) - set(target) or (arm == 'C' and new) or (arm == 'M' and (not new or not all(message_name(x) for x in new))):
        raise ValueError('Unexpected source or contextual-message tensor inventory.')
    for name, value in source_state.items():
        if value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f'Source tensor mismatch: {name}')
    model.load_state_dict(source_state, strict=False)
    return {'inherited_names': sorted(source_state), 'new_names': sorted(new)}


def balanced_snapshot_groups(model):
    """Partition all trainable tensors, reserving message parameters as their own group."""
    groups = _parameter_groups(model)
    for name in groups:
        groups[name] = [(key, value) for key, value in groups[name] if '.pair_message.' not in key]
    assigned = {key for pairs in groups.values() for key, _ in pairs}
    groups['message_parameters'] = [(key, value) for key, value in model.named_parameters()
                                    if value.requires_grad and '.pair_message.' in key]
    assigned.update(key for key, _ in groups['message_parameters'])
    groups['other_inherited'] = [(key, value) for key, value in model.named_parameters()
                                  if value.requires_grad and key not in assigned]
    return groups


class ThermalR166BalancedTask(ThermalR165DecisionTask):
    auxiliary_loss_names = ThermalStateResponseTask.auxiliary_loss_names

    def _prepare_snapshot(self, model, epoch):
        if self._snapshot_epoch != epoch:
            self._snapshot_epoch = epoch
            self._snapshot_groups = balanced_snapshot_groups(model)
            self._snapshot_buffers = {name: {} for name in ('endpoint', 'delta', 'decision', 'native', 'port')}

    def configure_balanced(self, spec):
        self.balanced_spec = copy.deepcopy(spec)
        self.response_finite_response_coefficient = float(spec['coefficient'])
        self._clock_epoch = None
        self._snapshot_epoch = None
        self._snapshot_buffers = None
        self._placement_epoch = None

    def make_batch(self, case_keys, key):
        batch = super().make_batch(case_keys, key)
        epoch = int(key.epoch) - self.source_parent_age
        if getattr(self, '_query_record_epoch', None) != epoch:
            self._query_record_epoch, self._query_records = epoch, []
        for case_id in case_keys:
            original = self.training_cases[self._train_index_by_id[str(case_id)]]
            fluid = key.native_indices(len(original['query_xy']), self.primary_queries,
                str(case_id), 'fluid', 'primary_native_queries', replace=False)
            material = key.native_indices(len(original['module_internal_query_points']), 32,
                str(case_id), 'material', 'primary_native_queries', replace=False)
            self._query_records.append({'case_id': str(case_id), 'batch': key.update_index,
                'microbatch': key.microbatch_index, 'fluid_ids': fluid.tolist(), 'material_ids': material.tolist()})
        return batch

    def auxiliary_update_terms(self, model, *, epoch, update_index, update_count):
        if self._placement_epoch != epoch:
            self._placement_epoch, self._placements = epoch, []
        families = self.response_corpus.fit_families_for_epoch(self.fit_sampler_epoch(epoch), self.seed)
        actions = [(family, action) for family in families for action in ACTIONS]
        if len(actions) != 16:
            raise ValueError('Expected four families and sixteen actions.')
        for index, (family, action) in enumerate(actions):
            if action_slots(update_count)[index] != update_index:
                continue
            pair = response_pair(self, self.response_corpus, family, action, role='fit')
            losses = predict_response_pair(_DirectResponseProvider(self, epoch), model, pair, self.response_scales,
                                           self.response_labels[(family, action)], self.response_spec['tau'])
            terms = response_loss_terms(losses['endpoint'], losses['field_delta'],
                {'J': losses['J_delta'], 'P': losses['P_delta']}, multiplier=update_count / 16,
                finite_response_coefficient=self.balanced_spec['coefficient'])
            if self._want_snapshot(epoch, update_index, update_count):
                self._prepare_snapshot(model, epoch)
                self._snapshot_family = str(family)
                _accumulate_gradients(self._snapshot_buffers['endpoint'], .25 * losses['endpoint'] * update_count / 16, self._snapshot_groups)
                _accumulate_gradients(self._snapshot_buffers['delta'], self.balanced_spec['coefficient'] * (update_count / 16) * (.8 * losses['field_delta'] + .1 * losses['J_delta'] + .1 * losses['P_delta']), self._snapshot_groups)
            self._record_response_action(epoch, family, action, 'original_fit')
            if family not in self._response_epoch_families:
                self._response_epoch_families.append(family)
            self._placements.append({'index': index, 'batch': update_index, 'family_id': str(family), 'action': action})
            yield terms, {**losses['work'], 'response_actions': 1, 'response_families': int(action == ACTIONS[0])}

    auxiliary_diagnostic_values = ThermalStateResponseTask.auxiliary_diagnostic_values

    def loss_metadata(self):
        from .thermal_port_response_task import ThermalPortResponseTask
        result = ThermalPortResponseTask.loss_metadata(self)
        old = self.response_finite_response_coefficient
        self.response_finite_response_coefficient = self.balanced_spec['coefficient']
        try: result.update(ThermalStateResponseTask.loss_metadata(self))
        finally: self.response_finite_response_coefficient = old
        return result

    def identity_payload(self):
        from .thermal_port_response_task import ThermalPortResponseTask
        result = ThermalPortResponseTask.identity_payload(self)
        result['r16_6_balanced'] = copy.deepcopy(self.balanced_spec)
        result['r16_6_source_tensor_inventory'] = copy.deepcopy(getattr(self, 'r16_5_source_tensor_inventory', {}))
        return result

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        if epoch not in (1, 10) or update_index != 0:
            return
        self._before_step = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self._before_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    def after_optimizer_step(self, model, optimizer, norm, *, epoch, update_index):
        if getattr(self, '_clip_epoch', None) != epoch:
            self._clip_epoch, self._clip_count, self._clip_total = epoch, 0, 0
        self._clip_total += 1
        self._clip_count += int(float(norm) > 1.)
        if epoch not in (1, 10) or update_index != 0:
            return
        from .r16_5_decision_task import _gradient_report
        displacement = {key: {'actual': 0., 'decay': 0., 'preconditioned': 0.}
                        for key in ('inherited', 'new')}
        groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
        for name, p in model.named_parameters():
            if name not in self._before_step or name not in self._before_grad:
                continue
            before = self._before_step[name]
            group = groups[id(p)]
            change = p.detach() - before
            decay = -group['lr'] * group['weight_decay'] * before
            subset = 'new' if name in self.r16_5_source_tensor_inventory['new_names'] else 'inherited'
            displacement[subset]['actual'] += float(change.double().square().sum())
            displacement[subset]['decay'] += float(decay.double().square().sum())
            displacement[subset]['preconditioned'] += float((change - decay).double().square().sum())
        self._learning_snapshot = {'batch': update_index, 'global_preclip_norm': float(norm),
            'clipped': float(norm) > 1., 'clip_factor': min(1., 1. / (float(norm) + 1e-6)),
            'displacement_norms': {subset: {key: math.sqrt(value) for key, value in norms.items()}
                                   for subset, norms in displacement.items()},
            'groups': _gradient_report(self._snapshot_buffers, {**self._snapshot_groups,
                'all_trainable': [pair for group in self._snapshot_groups.values() for pair in group]})}
        del self._before_step, self._before_grad

    def completed_training_epoch(self, model, engine, epoch, output, row):
        # Skip R16-5's decision-only telemetry; retain shared response bookkeeping.
        super(ThermalR165DecisionTask, self).completed_training_epoch(model, engine, epoch, output, row)
        row['r16_6_placements'] = copy.deepcopy(self._placements)
        row['r16_6_clipping'] = {'batches': self._clip_total, 'clipped': self._clip_count,
                                  'fraction': self._clip_count / self._clip_total}
        query_path = Path(output) / 'history' / f'r16_6_query_ids_epoch_{epoch:04d}.json'
        query_path.parent.mkdir(parents=True, exist_ok=True)
        query_path.write_text(json.dumps(self._query_records, separators=(',', ':')) + '\n')
        row['r16_6_query_ids'] = str(query_path.relative_to(output))
        if epoch in (1, 10):
            self._learning_snapshot['case_ids'] = [r['case_id'] for r in self._query_records if r['batch'] == 0]
            self._learning_snapshot['actions'] = [r for r in self._placements if r['batch'] == 0]
            row['r16_6_learning_snapshot'] = self._learning_snapshot

def build_thermal_r16_6_task(recipe, *, arm=None, device='cpu', resume_payload=None,
                                       response_scales=None):
    """Build a literal P64 fork, or reconstruct an existing fork from its own state."""
    configure_r16_5_execution(recipe, device=device)

    p64, h0 = _read_identity(P64_IDENTITY), _read_identity(H0_IDENTITY)
    source = None if resume_payload is not None else load_trusted_checkpoint(SOURCE_CHECKPOINT, map_location='cpu')
    references = validate_balanced_recipe(source, recipe, p64, h0)
    if arm is not None and arm != ARM_DETAIL:
        raise ValueError('R16-6 must retain the literal P64 model arm.')
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
        recipe, arm=ARM_DETAIL, device=device, task_type=ThermalR166BalancedTask,
        variants={DETAIL_CORE_SCHEMA: {**_VARIANTS[DETAIL_CORE_SCHEMA], 'adapter': GridAdapter, 'arm': ARM_DETAIL}},
        section_schema=DETAIL_CONFIG_SCHEMA, source_kind=SOURCE_KIND, lineage='R16-6 P64 weight-only fork',
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
    provider.configure_balanced(recipe['r16_6_balanced'])
    if recipe['execution'].get('deterministic_algorithms') is True:
        provider.r16_5_execution = {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if resume_payload is None:
        provider.r16_5_source_tensor_inventory = initialize_balanced_weights(
            recipe['r16_6_balanced']['arm'], model, source['model_state_dict'])
    else:
        model.load_state_dict(resume_payload['model_state_dict'], strict=True)
        provider.r16_5_source_tensor_inventory = copy.deepcopy(saved_provider['r16_6_source_tensor_inventory'])
    if (list(provider.response_corpus.family_ids('fit')) != references['h0_provider']['response_training_family_ids']
            or _sampling_dataset_id(provider.identity_payload()) != p64['sampling_dataset_id']):
        raise ValueError('Fork changed original FIT48 or native sampling namespace.')
    return model, provider


def calibrate_source(model, provider, engine):
    """Measure twelve exact first formal epoch placements without optimizer or persistent draws."""
    import statistics
    if provider.balanced_spec['arm'] != 'C' or len(provider.training_case_ids) != 600:
        raise ValueError('Calibrate once on the C source and the full TRAIN600 membership.')
    rng = capture_rng_state()
    mode = model.training
    clock = (model.core._joint_epoch, model.core._joint_update_index)
    saved = {key: copy.deepcopy(value) for key, value in provider.__dict__.items()
             if key.startswith(('_clock', '_response_epoch', '_placement', '_snapshot', '_training_samples', '_validation_samples', 'training_samples', '_query_record', '_clip_'))}
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rows = []
    try:
        model.train(True)
        cases, dataset = engine.epoch_case_order(provider, 1)
        slices = engine.epoch_update_slices(len(cases))
        if len(slices) != 13 or len(cases[slices[-1]]) != 24:
            raise ValueError('Calibration requires the actual 13 TRAIN600 batches, final batch 24.')
        families = provider.response_corpus.fit_families_for_epoch(provider.fit_sampler_epoch(1), provider.seed)
        actions = [(family, action) for family in families for action in ACTIONS]
        for index in calibration_slots():
            batches = engine.prepare_update_microbatches(provider, arm=model.architecture, phase='joint', epoch=1,
                update_index=index, update_cases=cases[slices[index]], sampling_dataset_id=dataset)
            native = engine.diagnose_macro_update(model, provider, batches, phase='joint', arm=model.architecture,
                epoch=1, include_gradient_tensors=True, restore_random_state=False)
            abs_g = native['combined_gradient_tensors']
            delta_g = {}
            action_rows = []
            for a, (family, action) in enumerate(actions):
                if action_slots(13)[a] != index:
                    continue
                pair = response_pair(provider, provider.response_corpus, family, action, role='fit')
                losses = predict_response_pair(_DirectResponseProvider(provider, 1), model, pair,
                    provider.response_scales, provider.response_labels[(family, action)], provider.response_spec['tau'])
                absolute = .25 * 13 / 16 * losses['endpoint']
                delta = 13 / 16 * (.8 * losses['field_delta'] + .1 * losses['J_delta'] + .1 * losses['P_delta'])
                ga = torch.autograd.grad(absolute, [p for _, p in params], retain_graph=True, allow_unused=True)
                gd = torch.autograd.grad(delta, [p for _, p in params], allow_unused=True)
                for (name, _), x, y in zip(params, ga, gd):
                    if x is not None: abs_g[name] = abs_g.get(name, torch.zeros_like(x)) + x.detach()
                    if y is not None: delta_g[name] = delta_g.get(name, torch.zeros_like(y)) + y.detach()
                masks = []
                for view in losses['observations']:
                    masks.append({name: {'shape': list(base[2].shape),
                        'baseline_count': int(base[2].count_nonzero()),
                        'edited_count': int(view['edited'][name][2].count_nonzero()),
                        'common_count': int((base[2] & view['edited'][name][2]).count_nonzero())}
                        for name, base in view['baseline'].items()})
                action_rows.append({'family_id': str(family), 'action': action, 'physical_masks_by_view': masks,
                    'losses': {k: float(losses[k].detach()) for k in ('endpoint', 'field_delta', 'J_delta', 'P_delta')}})
            aa = dd = dot = 0.
            for name, _ in params:
                x, y = abs_g.get(name), delta_g.get(name)
                if x is not None: aa += float(x.double().square().sum())
                if y is not None: dd += float(y.double().square().sum())
                if x is not None and y is not None: dot += float((x.double() * y.double()).sum())
            an, dn = math.sqrt(aa), math.sqrt(dd)
            if not math.isfinite(an + dn) or an == 0 or dn == 0:
                raise FloatingPointError('Disconnected or nonfinite calibration gradient.')
            rows.append({'batch': index, 'case_ids': list(cases[slices[index]]), 'actions': action_rows,
                'query_receipts': native['batches'], 'native_losses': native['loss_terms'],
                'native_masks': [{name: {'shape': list(value.shape), 'nonzero': int(torch.count_nonzero(value))}
                    for name, value in vars(batch.targets).items()
                    if torch.is_tensor(value) and ('valid' in name or 'mask' in name or name == 'point_weights')}
                    for batch in batches],
                'absolute_zero_norm': an == 0., 'delta_zero_norm': dn == 0.,
                'absolute_norm': an, 'delta_norm': dn, 'cosine': dot / (an * dn),
                'ratio': an / max(dn, 1e-12)})
        coefficient = min(.1, .25 * statistics.median(r['ratio'] for r in rows))
        return {'source_age': SOURCE_AGE, 'local_epoch': 1, 'source_clock': SOURCE_AGE + 1,
                'batch_count': 13, 'omitted_batch': next(i for i in range(13) if i not in calibration_slots()),
                'examples': rows, 'coefficient': coefficient}
    finally:
        for key in list(provider.__dict__):
            if key.startswith(('_clock', '_response_epoch', '_placement', '_snapshot', '_training_samples', '_validation_samples', 'training_samples', '_query_record', '_clip_')):
                provider.__dict__.pop(key)
        provider.__dict__.update(saved)
        model.core.set_training_clock(*clock)
        model.train(mode)
        restore_rng_state(rng)
