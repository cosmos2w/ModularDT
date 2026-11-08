"""In-process, bounded actual-optimizer diagnostics for R16-10 training."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np

from .r16_10_diagnostics import counterfactual_updates, preserved_state
from .r16_10_family_task import R1610FamilyObjective, auxiliary_ramp


def _sampled_ids(ids):
    ids = tuple(map(str, ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Diagnostic input population must contain unique IDs.')
    count = max(1, int(np.floor(.1 * len(ids) + .5)))
    chosen = set(np.random.default_rng(20261108).choice(len(ids), count, replace=False).tolist())
    return tuple(value for index, value in enumerate(ids) if index in chosen)


def _build_b0_objective(provider, spec):
    from .r16_8_corpus import ThermalR168Corpus
    from .thermal_state_response_task import ThermalResponseCorpus, ACTIONS

    inputs_path = Path(spec['coefficients_path']).parent / f"training_inputs_train{spec['population']}.json"
    inputs = json.loads(inputs_path.read_text())
    original = ThermalResponseCorpus(spec['original_input_recipe'], spec['original_ledger'])
    new = ThermalR168Corpus(spec['new_fit_manifest'])
    labels = {(str(row['family']), row['action']): row['delta'] for row in inputs['original_labels']}
    objective = R1610FamilyObjective(provider, original, new, provider.training_case_ids, labels,
        response_scales=inputs['response_scales'], sensitivity=inputs['sensitivity'],
        research_scales=inputs['research_scales'])
    expected = {(family, action) for family in objective.schedule.families['original'] for action in ACTIONS}
    if set(labels) != expected:
        raise ValueError('Diagnostic labels must contain exactly eligible original FIT actions.')
    return objective


def _diagnose(provider, recipe, model, optimizer, epoch, clip_limit):
    from .r16_5_decision_task import _parameter_groups

    spec = recipe['r16_10_family']
    population, arm = int(spec['population']), spec['arm']
    started = time.perf_counter()
    with preserved_state(model, optimizer):
        model.train()
        objective = (_build_b0_objective(provider, spec) if arm == 'B0'
                     else provider.family_objective)
        schedule_state = copy.deepcopy(objective.schedule.state_dict())
        try:
            family_population = tuple((source, family) for source in ('original', 'new')
                                      for family in objective.schedule.families[source])
            sampled_family_keys = set(_sampled_ids(tuple(f'{source}:{family}' for source, family in family_population)))
            panel = tuple(item for item in family_population if f'{item[0]}:{item[1]}' in sampled_family_keys)
            native_panel = _sampled_ids(provider.training_case_ids)
            if (len(native_panel), len(panel)) != ({150: (15, 4), 600: (60, 14)}[population]):
                raise ValueError('Diagnostic calibration panels do not match the declared populations.')
            native_id, (source, family) = native_panel[0], panel[0]

            def native_loss():
                for batch in provider.native_evaluation_batches(split='train', case_keys=[native_id], microbatch_size=1):
                    pred, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers,
                                                      model.architecture, 'joint', epoch=epoch)
                    yield provider.endpoint_state_loss(pred, batch)

            def family_loss(name):
                def loss():
                    values, _ = objective.family_components(model, source, family)
                    return values[name]
                return loss

            losses = {'native': native_loss, **{key: family_loss(key) for key in ('endpoint', 'response', 'decision')}}
            weights = {'native': 1.0}
            if arm != 'B0':
                coefficients = json.loads(Path(spec['coefficients_path']).read_text())['coefficients']
                active = ('endpoint',) if arm == 'E' else (('endpoint', 'response', 'decision') if arm == 'D' else ('endpoint', 'response'))
                weights.update({key: auxiliary_ramp(epoch) * float(coefficients[key]) for key in active})
            groups = {name: [key for key, _ in pairs] for name, pairs in _parameter_groups(model).items() if pairs}
            result = counterfactual_updates(model, optimizer, losses, weights, clip_grad_norm=clip_limit,
                                            parameter_groups=groups)
            result.update(native_id=native_id, family_id=family, family_source=source,
                          native_calibration_panel=list(native_panel),
                          family_calibration_panel=[list(v) for v in panel],
                          combined_weights=weights, epoch=int(epoch),
                          sampling='first source-ordered member of seed20261108 calibration panel')
            result['complete_diagnostic_seconds'] = time.perf_counter() - started
            return result
        finally:
            objective.schedule.load_state_dict(schedule_state)


def attach_update_diagnostics(provider, recipe):
    """Compose existing callbacks; run only inside the owned training process."""
    spec = recipe['r16_10_family']
    population = int(spec['population'])
    if population not in (150, 600) or spec['arm'] not in ('B0', 'E', 'R', 'D', 'V'):
        raise ValueError('Invalid R16-10 update diagnostic recipe.')
    epochs = {50, 1000} if population == 150 else {50, 100}
    previous_step = getattr(provider, 'after_optimizer_step', None)
    previous_epoch = getattr(provider, 'completed_training_epoch', None)
    state = {'epoch': None, 'optimizer': None, 'updates': 0, 'clipped': 0}

    def after_step(self, model, optimizer, norm, *, epoch, update_index):
        if callable(previous_step):
            previous_step(model, optimizer, norm, epoch=epoch, update_index=update_index)
        if epoch not in epochs:
            return
        if state['epoch'] != epoch:
            state.update(epoch=epoch, optimizer=optimizer, updates=0, clipped=0)
        elif state['optimizer'] is not optimizer:
            raise ValueError('Optimizer changed during an R16-10 diagnostic epoch.')
        state['updates'] += 1
        state['clipped'] += int(float(norm) > float(recipe['optimizer']['gradient_clip']))

    def completed_epoch(self, model, engine, epoch, output, row):
        if callable(previous_epoch):
            previous_epoch(model, engine, epoch, output, row)
        if epoch not in epochs:
            return
        if state['epoch'] != epoch or state['optimizer'] is None or not state['updates']:
            raise RuntimeError('R16-10 diagnostic has no actual optimizer update for this epoch.')
        try:
            result = _diagnose(self, recipe, model, state['optimizer'], epoch, float(engine.config.gradient_clip))
            result['observed_epoch_clipping_fraction'] = state['clipped'] / state['updates']
            result['observed_epoch_optimizer_updates'] = state['updates']
            row['r16_10_actual_update'] = result
        finally:
            state.update(epoch=None, optimizer=None, updates=0, clipped=0)

    provider.after_optimizer_step = MethodType(after_step, provider)
    provider.completed_training_epoch = MethodType(completed_epoch, provider)
    return provider
