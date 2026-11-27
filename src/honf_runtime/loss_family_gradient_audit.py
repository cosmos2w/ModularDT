"""Occasional, state-preserving gradients on a fixed coupled-native TRAIN panel.

Every scalar uses the provider's live native loss terms and declared panel
reductions. Coupled-response terms come from the same native predictions and
physical observation extraction used by the training provider. No optimizer,
projection, clipping, training-state update, or alternate prediction is made.
"""
from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Mapping
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import torch

from .gradient_diagnostics import preserved_state, _norm
from .organization_parameters import partition_organization_parameters
from .unified_training import SamplingKey


THERMAL_NATIVE_FAMILIES = {
    'flow': ('flow/u', 'flow/v', 'flow/p', 'flow/omega'),
    'fluid_temperature': ('temperature/fluid',),
    'member_interface': ('temperature/surface', 'temperature/material', 'q_proxy',
                         'port/outside_temperature'),
}


def _rng_state():
    return (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


@contextmanager
def _preserve_provider(model, provider):
    """Protect persistent provider counters, reader state and model clocks."""
    if not callable(getattr(provider, 'training_state_dict', None)) or not callable(
            getattr(provider, 'load_training_state_dict', None)):
        raise TypeError('The coupled provider must expose its normal recovery state.')
    state = copy.deepcopy(provider.training_state_dict())
    names = ('coupled_native_reader_mode', '_coupled_calibration_panel')
    attributes = {name: (hasattr(provider, name), getattr(provider, name, None)) for name in names}
    core = getattr(model, 'core', model)
    clocks = {name: (hasattr(core, name), getattr(core, name, None))
              for name in ('_joint_epoch', '_joint_update_index')}
    try:
        with preserved_state(model):
            yield
    finally:
        try:
            provider.load_training_state_dict(state)
        finally:
            for target, values in ((provider, attributes), (core, clocks)):
                for name, (existed, value) in values.items():
                    if existed:
                        setattr(target, name, value)
                    elif hasattr(target, name):
                        delattr(target, name)


def _family_members(families, denominator_names):
    if not isinstance(families, Mapping) or not families:
        raise ValueError('Native gradient families must be a nonempty mapping.')
    result = {str(name): tuple(terms) for name, terms in families.items()}
    if any(not name or not terms for name, terms in result.items()):
        raise ValueError('Each native gradient family must contain named loss terms.')
    all_terms = [term for terms in result.values() for term in terms]
    if len(set(all_terms)) != len(all_terms) or set(all_terms) != set(denominator_names):
        raise ValueError('Native gradient families must partition the actual native loss denominators exactly.')
    return result


def _panel_batches(provider, model, cases, key):
    """Preserve the input-selected order while keeping actual complete pairs together."""
    identity = provider.identity_payload().get('observation_response', {})
    declared = tuple(tuple(map(str, pair)) for pair in identity.get('pairs', ()))
    eligible = set(tuple(map(str, pair)) for pair in provider.coupled_native_pairs)
    wanted = set(cases)
    inside = tuple(pair for pair in declared if set(pair) <= wanted)
    selected = tuple(pair for pair in inside if pair in eligible)
    unavailable = tuple(pair for pair in inside if pair not in eligible)
    crossing = tuple(pair for pair in declared if len(set(pair) & wanted) == 1)
    by_case = {case: pair for pair in selected for case in pair}
    if len(by_case) != 2 * len(selected):
        raise ValueError('Configured complete panel pairs must be disjoint.')
    chunks, consumed = [], set()
    for case in cases:
        if case in consumed:
            continue
        chunk = by_case.get(case, (case,))
        chunks.append(chunk)
        consumed.update(chunk)
    if consumed != wanted:
        raise ValueError('Panel batching did not preserve every original case.')
    provider._coupled_calibration_panel = True
    batches = tuple(provider.make_batch(chunk, key) for chunk in chunks)
    for batch, chunk in zip(batches, chunks):
        if tuple(map(str, batch.case_keys)) != tuple(chunk):
            raise ValueError('Provider changed the fixed panel case order.')
    return batches, selected, unavailable, crossing


def _nonnegative_denominator(value, name):
    number = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'Invalid native denominator for {name!r}.')
    return number


def _gradient_groups(model, accumulators, ownership):
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    named_groups = {'global': tuple(names), 'organization': ownership.organization_names,
                    'content_shared': ownership.content_names}
    result = {}
    for group, members in named_groups.items():
        member_set = set(members)
        ids = [index for index, name in enumerate(names) if name in member_set]
        gradients = {label: [values[index] for index in ids] for label, values in accumulators.items()}
        norms = {label: _norm(values) for label, values in gradients.items()}
        cosines = {}
        for left, right in combinations(gradients, 2):
            denominator = norms[left] * norms[right]
            if not denominator:
                cosines[f'{left}:{right}'] = None
                continue
            dot = sum((a.double() * b.double()).sum() for a, b in zip(gradients[left], gradients[right]))
            cosines[f'{left}:{right}'] = float(dot.item()) / denominator
        result[group] = {'parameter_tensors': len(ids), 'norms': norms, 'cosines': cosines}
    return result


def _synchronize(model):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def audit_loss_families(model, provider, *, panel_case_ids, epoch, read_modes=('restricted', 'full'),
                        sampling_epoch=1, native_families=THERMAL_NATIVE_FAMILIES):
    """Measure fixed-panel raw gradients using real native and coupled losses.

    Native denominators cover the exact ordered panel; the response is the
    equal-pair average of the configured eligible pairs wholly inside it. A
    checkpoint whose response updates are disabled still gets this diagnostic
    physical loss with its saved common calibration and no optimizer step.
    Both sides of a pair use the same declared reader and retained native
    sampling stream. The resulting panel gradient is not an actual effective
    training-batch gradient and is never projected or clipped here.
    """
    audit_started = time.perf_counter()
    cases = tuple(map(str, panel_case_ids))
    if (not cases or len(cases) > 15 or len(set(cases)) != len(cases)
            or type(epoch) is not int or epoch < 1 or type(sampling_epoch) is not int or sampling_epoch < 1):
        raise ValueError('Use one to fifteen unique TRAIN cases and positive integer epochs.')
    modes = tuple(read_modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in ('full', 'restricted') for mode in modes):
        raise ValueError('Read modes must be unique full or restricted readers.')
    if getattr(provider, 'model', model) is not model:
        raise ValueError('The native provider must own the measured model.')
    if not all(hasattr(provider, name) for name in
               ('coupled_native_pairs', 'coupled_native_estimates', 'coupled_native_reader_mode')):
        raise TypeError('The provider must expose the maintained coupled-native task.')
    available = tuple(map(str, provider.epoch_cases(sampling_epoch, int(provider.seed))))
    if not set(cases) <= set(available):
        raise ValueError('Every panel case must belong to the checkpoint TRAIN membership.')
    if getattr(provider, 'sampling_version', None) != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError('The audit requires the actual case_epoch_v1 sampling streams.')
    spec = provider.identity_payload().get('observation_response', {})
    probability = float(spec.get('full_reader_probability', .5))
    if not 0 < probability < 1:
        raise ValueError('The saved reader probability must lie strictly between zero and one.')
    state = provider.training_state_dict().get('coupled_native', {})
    calibration = state.get('calibration')
    if not isinstance(calibration, Mapping) or calibration.get('schema') != 'coupled_native_calibration_v1':
        raise ValueError('The checkpoint must contain its shared epoch-zero response calibration.')
    coefficient = float(calibration.get('coefficient', 0))
    multiplier = float(spec.get('response_weight', 1.))
    if (not math.isfinite(coefficient) or coefficient <= 0
            or not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError('The saved response calibration and multiplier must be finite and positive.')
    ownership = partition_organization_parameters(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    key = SamplingKey(int(provider.seed), sampling_epoch, 0, 0, 'joint', model.architecture,
                      sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)
    with _preserve_provider(model, provider):
        model.train(True)
        setup_started = time.perf_counter()
        batches, selected, unavailable, crossing = _panel_batches(provider, model, cases, key)
        core = getattr(model, 'core', model)
        clock = getattr(core, 'set_organization_clock', None)
        if callable(clock):
            clock(epoch, 0)
        planned = {name: _nonnegative_denominator(value, name) for name, value in
                   provider.loss_denominators(batches, phase='joint', arm=model.architecture).items()}
        if not planned or any(value <= 0 for value in planned.values()):
            raise ValueError('Every native panel loss requires a positive declared denominator.')
        families = _family_members(native_families, planned)
        setup_seconds = time.perf_counter() - setup_started
        fixed_rng = _rng_state()
        views = {}
        for mode in modes:
            _restore_rng(fixed_rng)
            provider.coupled_native_reader_mode = mode
            reader_factor = .5 / (probability if mode == 'full' else 1. - probability)
            scale = coefficient * min(epoch / 50., 1.) * multiplier * reader_factor
            labels = (*families, 'response') if selected else tuple(families)
            accumulators = {name: [torch.zeros_like(p) for p in parameters] for name in labels}
            losses = {name: 0. for name in labels}
            observed = {name: 0. for name in planned}
            term_weights, term_numerators, pair_evidence, work = {}, {name: 0. for name in planned}, [], {}
            vjp_calls = 0
            _synchronize(model)
            started = time.perf_counter()
            for batch in batches:
                prediction, auxiliary = provider.predict_native(
                    model, provider.make_scene(batch.scene_inputs), batch.receivers,
                    model.architecture, 'joint', epoch=epoch, temperature=1.)
                terms = provider.loss_terms(prediction, batch.targets, 'joint', auxiliary)
                if not isinstance(terms, Mapping) or set(terms) != set(planned):
                    raise ValueError('Native loss terms differ from the declared denominator prepass.')
                objectives = {}
                for family, members in families.items():
                    pieces = []
                    for name in planned:
                        if name not in members:
                            continue
                        term = terms[name]
                        numerator = term.numerator
                        denominator = _nonnegative_denominator(term.denominator, name)
                        weight = float(term.weight)
                        if not torch.is_tensor(numerator) or numerator.numel() != 1 or not torch.isfinite(numerator).all() or not math.isfinite(weight):
                            raise ValueError(f'Native loss {name!r} must be a finite weighted scalar.')
                        if name in term_weights and term_weights[name] != weight:
                            raise ValueError(f'Native loss weight {name!r} changed within the panel.')
                        term_weights[name] = weight
                        observed[name] += denominator
                        term_numerators[name] += float(numerator.detach())
                        pieces.append(numerator * (weight / planned[name]))
                    objectives[family] = sum(pieces)
                current_pairs = tuple((batch.auxiliary or {}).get('coupled_native', ()))
                if current_pairs:
                    estimates = provider.coupled_native_estimates(batch, prediction)
                    if len(current_pairs) != 1 or len(estimates) != 1 or tuple(estimates[0][0]) not in selected:
                        raise ValueError('A fixed-panel pair did not follow the configured native pairing.')
                    pair, estimate = estimates[0]
                    if estimate.loss is None:
                        raise ValueError('A selected physical response has no eligible weighted component.')
                    objectives['response'] = estimate.loss * (scale / len(selected))
                    pair_evidence.append({'case_ids': list(pair), 'realized_common_draws': estimate.realized_common_draws,
                                          'common_probability': estimate.common_probability,
                                          'eligible_draws': dict(estimate.eligible_draws)})
                active = list(objectives.items())
                for index, (name, objective) in enumerate(active):
                    if not torch.is_tensor(objective) or objective.numel() != 1 or not torch.isfinite(objective).all():
                        raise ValueError(f'Nonfinite gradient objective {name!r}.')
                    losses[name] += float(objective.detach())
                    if not objective.requires_grad:
                        raise ValueError(f'Gradient objective {name!r} is disconnected.')
                    partial = torch.autograd.grad(objective, parameters, allow_unused=True,
                                                  retain_graph=index + 1 < len(active))
                    vjp_calls += 1
                    for destination, value in zip(accumulators[name], partial):
                        if value is not None:
                            destination.add_(value.detach())
                for name, value in provider.work_counts(batch, prediction, auxiliary).items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        if name.endswith('_max_live_tile_bytes_per_call'):
                            work[name] = max(work.get(name, 0.), float(value))
                        else:
                            work[name] = work.get(name, 0.) + float(value)
                del prediction, terms, objectives, active
            if len(pair_evidence) != len(selected):
                raise ValueError('Not every eligible configured panel pair produced one response.')
            for name in planned:
                if not math.isclose(observed[name], planned[name], rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError(f'Native panel denominator mismatch for {name!r}.')
            if any(not torch.isfinite(value).all() for values in accumulators.values() for value in values):
                raise FloatingPointError('A raw loss-family gradient is nonfinite.')
            accumulators['native_total'] = [sum(values) for values in zip(*(accumulators[name] for name in families))]
            losses['native_total'] = sum(losses[name] for name in families)
            _synchronize(model)
            views[mode] = {'losses': losses, 'gradient_groups': _gradient_groups(model, accumulators, ownership),
                           'native_terms': {name: {'numerator': term_numerators[name],
                                                  'denominator': planned[name], 'weight': term_weights[name]}
                                            for name in planned},
                           'response': {'available': bool(selected), 'pairs': pair_evidence,
                                        'pair_denominator': len(selected), 'calibration_coefficient': coefficient,
                                        'response_multiplier': multiplier, 'epoch_ramp': min(epoch / 50., 1.),
                                        'reader_probability_factor': reader_factor, 'total_weight': scale},
                           'work': work, 'native_prediction_calls': len(batches),
                           'scene_presentations': len(cases), 'gradient_vjp_calls': vjp_calls,
                           'gradient_panel_seconds': time.perf_counter() - started}
    return {'schema': 'coupled_native_loss_family_gradient_audit_v1',
            'panel_case_ids': list(cases), 'batch_case_ids': [list(batch.case_keys) for batch in batches],
            'complete_training_pairs_on_panel': [list(pair) for pair in selected],
            'unsupported_complete_pairs': [list(pair) for pair in unavailable],
            'configured_pairs_with_one_panel_case': [list(pair) for pair in crossing],
            'native_sampling_epoch': sampling_epoch, 'checkpoint_epoch': epoch, 'organization_update_index': 0,
            'native_loss_order': list(planned),
            'panel_preparation_seconds': setup_seconds, 'audit_seconds': time.perf_counter() - audit_started,
            'native_families': {name: list(terms) for name, terms in families.items()},
            'parameter_ownership': ownership.metadata(), 'reader_results': views,
            'interpretation': 'Raw weighted panel gradients before projection, clipping or optimizer update; '
                              'not the projected effective-batch or AdamW update. The native panel includes '
                              'every listed case; response averages only complete configured eligible pairs. '
                              'A paired case outside the panel may influence the existing native row sampler '
                              'but is not decoded or added to the panel. The response of a disabled model is '
                              'diagnostic only and never updates that model. Work counters record forward '
                              'execution; elapsed time also includes VJP checkpoint recomputation.',
            'fitting_epochs': 0, 'optimizer_steps': 0}
