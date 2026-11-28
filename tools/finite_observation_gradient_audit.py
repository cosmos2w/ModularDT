#!/usr/bin/env python3
"""Audit finite-observation gradients at a saved checkpoint without updating it.

The reduction is the training objective on a declared pair-complete TRAIN
panel. Each reader is measured separately using its sampled native objective;
the fixed finite rows and their sparse physical extractions are unchanged. The
CLI reconstructs ordinary native sampling: its fixed draws have the same
native marginal, but do not replay pair-coupled training draws.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in ('src', 'Case_ThermalChannel/src', 'tools'):
    sys.path.insert(0, str(ROOT / directory))

import torch

from honf_runtime.finite_observation_task import execute_finite_dependencies
from honf_runtime.finite_observations import (compatible_scene_families, evaluate_observation_scene,
                                               finite_scene_contrast)
from honf_runtime.gradient_diagnostics import preserved_state
from honf_runtime.loss_family_gradient_audit import _restore_rng, _rng_state
from honf_runtime.organization_parameters import partition_organization_parameters
from honf_runtime.unified_training import SamplingKey, _atomic_json

NATIVE_SAMPLING_DESCRIPTION = (
    'Ordinary native case_epoch_v1 source-row draws from the reconstructed native provider; '
    'the marginal and objective match native training, but these are not literal pair-coupled training draws.'
)


@contextmanager
def _preserve_audit_state(model, provider):
    """Restore provider sampling state, temporary reader flags and core clocks."""
    original_rng = _rng_state()
    state = copy.deepcopy(provider.training_state_dict())
    names = ('finite_observation_reader_mode', 'coupled_native_reader_mode', '_coupled_calibration_panel')
    attributes = {name: (hasattr(provider, name), getattr(provider, name, None)) for name in names}
    core = getattr(model, 'core', model)
    clocks = {name: (hasattr(core, name), getattr(core, name, None))
              for name in ('_joint_epoch', '_joint_update_index', '_organization_epoch',
                           '_organization_update', '_frontier_context')}
    try:
        with preserved_state(model):
            yield
    finally:
        try:
            provider.load_training_state_dict(state)
        finally:
            try:
                for target, values in ((provider, attributes), (core, clocks)):
                    for name, (exists, value) in values.items():
                        if exists:
                            setattr(target, name, value)
                        elif hasattr(target, name):
                            delattr(target, name)
            finally:
                _restore_rng(original_rng)


def _calibration_info(calibration, variant, epoch):
    if variant not in ('disabled', 'anchors', 'contrasts', 'legacy_native'):
        raise ValueError('Unknown saved finite-observation variant.')
    ramp = min(epoch / 50., 1.)
    coefficients = None
    if calibration is not None:
        if calibration.get('schema') != 'finite_observation_calibration_v1' or calibration.get('epoch') != 0:
            raise ValueError('The saved finite calibration must be its original epoch-zero record.')
        coefficients = {name: float(calibration.get('coefficients', {}).get(name, 0.))
                        for name in ('anchor', 'contrast')}
        if any(not math.isfinite(value) or value <= 0 for value in coefficients.values()):
            raise ValueError('Saved finite calibration coefficients must be positive and finite.')
    enabled = {'anchor': variant in ('anchors', 'contrasts'), 'contrast': variant == 'contrasts'}
    if coefficients is None and any(enabled.values()):
        raise ValueError('An active saved finite objective requires its saved calibration.')
    return {'source': 'saved epoch-zero calibration' if coefficients else 'unavailable',
            'coefficients': coefficients, 'epoch_ramp': ramp, 'enabled': enabled,
            'prospective_scales': {name: None if coefficients is None else coefficients[name] * ramp
                                   for name in enabled},
            'effective_scales': {name: (coefficients[name] * ramp if enabled[name] else 0.)
                                 for name in enabled}}


def _scalar(value, description):
    if not torch.is_tensor(value) or value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{description} must be a finite scalar tensor.')
    return value


def _norm(values, indices):
    return math.sqrt(sum(float(values[index].double().square().sum()) for index in indices))


def audit_finite_observation_gradients(model, provider, binding, *, panel_case_ids, panel_pairs,
                                       epoch, variant, calibration=None, basis='coordinate',
                                       probe_count=64, probe_seed=130013, sampling_epoch=1,
                                       read_modes=('full', 'restricted'), expected_family_scales=None,
                                       native_sampling_description='Fixed case_epoch_v1 draws from the supplied provider.'):
    """Measure gradients of the exact panel-reduced native and finite scalars.

    Per-family contrast is *unweighted*, divided by the complete eligible-pair
    count; its weighted contribution uses that family's fixed configured
    weight divided by all configured family weights. Missing support contributes
    zero without changing either denominator. No gradient or optimizer state is
    written, and panel gradients are not represented as actual batch updates.
    """
    cases = tuple(map(str, panel_case_ids))
    pairs = tuple(tuple(map(str, pair)) for pair in panel_pairs)
    modes = tuple(read_modes)
    if (not cases or len(set(cases)) != len(cases) or not pairs or
            any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs) or
            len({case for pair in pairs for case in pair}) != 2 * len(pairs) or
            {case for pair in pairs for case in pair} != set(cases)):
        raise ValueError('The audit requires unique cases in disjoint, pair-complete panel order.')
    if (type(epoch) is not int or epoch < 1 or type(sampling_epoch) is not int or sampling_epoch < 1 or
            type(probe_count) is not int or not 1 <= probe_count <= 64 or
            type(probe_seed) is not int or probe_seed < 0):
        raise ValueError('The checkpoint epoch, fixed native sampling epoch and finite probe recipe are invalid.')
    if not modes or len(set(modes)) != len(modes) or any(mode not in ('full', 'restricted') for mode in modes):
        raise ValueError('Use unique full and/or restricted readers.')
    if getattr(provider, 'model', model) is not model:
        raise ValueError('The measured provider must own the restored model.')
    if getattr(provider, 'sampling_version', None) != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError('The checkpoint must use the native case_epoch_v1 sampling stream.')
    info = _calibration_info(calibration, variant, epoch)
    ownership = partition_organization_parameters(model)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    names = [name for name, _ in named]
    group_names = {'global': tuple(names), 'organization': ownership.organization_names,
                   'content_and_output': ownership.content_names}
    groups = {group: [index for index, name in enumerate(names) if name in set(members)]
              for group, members in group_names.items()}
    weights = dict(binding.finite_observation_family_weights())
    total_weight = sum(map(float, weights.values()))
    if not weights or any(not math.isfinite(float(value)) or value <= 0 for value in weights.values()):
        raise ValueError('The finite family weights must be positive and finite.')
    frame = binding.finite_observation_frame()
    frames = {} if frame is None else {frame.identity: frame}
    started = time.perf_counter()
    with _preserve_audit_state(model, provider):
        membership = tuple(map(str, provider.epoch_cases(sampling_epoch, int(provider.seed))))
        if not set(cases) <= set(membership):
            raise ValueError('Every gradient-audit case must belong to saved TRAIN membership.')
        for pair in pairs:
            binding.validate_pair(pair)
        scenes = {case: binding.finite_observation_scene(case, count=probe_count, seed=probe_seed)
                  for case in cases}
        family_metadata = {}
        for scene in scenes.values():
            for family in scene.families:
                family_metadata.setdefault(family.name, {'unit': family.component.unit,
                    'native_scale': float(family.component.scale), 'support': family.support,
                    'frame': family.frame, 'time': family.time, 'extraction': family.extraction_id})
        if expected_family_scales is not None:
            for scene in scenes.values():
                for family in scene.families:
                    if expected_family_scales.get(family.name) != float(family.component.scale):
                        raise ValueError(f'Family {family.name!r} differs from the saved source scale.')
        eligibility = {pair: compatible_scene_families(scenes[pair[0]], scenes[pair[1]],
                        configured_weights=weights) for pair in pairs}
        eligible = tuple(pair for pair in pairs if eligibility[pair].eligible_families)
        if not eligible:
            raise ValueError('The declared panel contains no eligible finite-observation pairs.')
        key = SamplingKey(int(provider.seed), sampling_epoch, 0, 0, 'joint', model.architecture,
                          sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)
        model.train(True)
        provider._coupled_calibration_panel = True
        batches = [provider.make_batch(pair, key) for pair in pairs]
        if any(tuple(map(str, batch.case_keys)) != pair for batch, pair in zip(batches, pairs)):
            raise ValueError('The native provider changed the declared case-pair order.')
        core = getattr(model, 'core', model)
        if callable(getattr(core, 'set_organization_clock', None)):
            core.set_organization_clock(epoch, 0)
        planned = {name: float(value) for name, value in
                   provider.loss_denominators(batches, phase='joint', arm=model.architecture).items()}
        if not planned or any(not math.isfinite(value) or value <= 0 for value in planned.values()):
            raise ValueError('The native panel requires finite positive loss denominators.')
        fixed_rng = _rng_state()
        views = {}
        for mode in modes:
            _restore_rng(fixed_rng)
            provider.coupled_native_reader_mode = mode
            provider.finite_observation_reader_mode = mode
            labels = ('native', 'anchor', *weights)
            accumulated = {label: [torch.zeros(parameter.shape, dtype=torch.float64, device='cpu')
                                   for parameter in parameters] for label in labels}
            losses = {label: 0. for label in labels}
            seen, term_weights = {name: 0. for name in planned}, {}
            family_pairs = {name: 0 for name in weights}
            work, gradient_calls = {}, 0
            for batch, pair in zip(batches, pairs):
                prediction, auxiliary = provider.predict_native(model, provider.make_scene(batch.scene_inputs),
                    batch.receivers, model.architecture, 'joint', epoch=epoch, temperature=1.)
                terms = provider.loss_terms(prediction, batch.targets, 'joint', auxiliary)
                if set(terms) != set(planned):
                    raise ValueError('Native loss terms differ from the declared panel denominators.')
                pieces = []
                for name, term in terms.items():
                    numerator = _scalar(term.numerator, f'Native loss {name!r}')
                    denominator, weight = float(term.denominator), float(term.weight)
                    if not math.isfinite(denominator) or denominator < 0 or not math.isfinite(weight):
                        raise ValueError(f'Invalid native weight or denominator for {name!r}.')
                    if name in term_weights and term_weights[name] != weight:
                        raise ValueError(f'Native loss weight changed for {name!r}.')
                    term_weights[name] = weight
                    seen[name] += denominator
                    pieces.append(numerator * (weight / planned[name]))
                chosen = [scenes[case] for case in pair]
                extracted, measured = execute_finite_dependencies(binding, model, chosen, read_mode=mode,
                    native_batch=batch, native_predictions=prediction)
                for name, value in measured.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        work[name] = work.get(name, 0.) + float(value)
                observations = {case: evaluate_observation_scene(scenes[case], extracted[case],
                    configured_weights=weights, frames=frames, basis=basis) for case in pair}
                objectives = {'native': sum(pieces)}
                anchors = [observations[case].anchor.mean() for case in pair if observations[case].anchor is not None]
                if anchors:
                    objectives['anchor'] = torch.stack(anchors).sum() / len(cases)
                if pair in eligible:
                    contrast = finite_scene_contrast(observations[pair[0]], observations[pair[1]],
                                                      configured_weights=weights)
                    if contrast.loss is None:
                        raise RuntimeError('An eligible pair lost its finite-observation contrast.')
                    for name in contrast.eligible_families:
                        first = observations[pair[0]].families[name].moments
                        second = observations[pair[1]].families[name].moments
                        objectives[name] = (second - first).square().mean() / len(eligible)
                        family_pairs[name] += 1
                active = []
                for name, value in objectives.items():
                    value = _scalar(value, f'Gradient objective {name!r}')
                    losses[name] += float(value.detach())
                    if value.requires_grad:
                        active.append((name, value))
                for index, (name, value) in enumerate(active):
                    gradients = torch.autograd.grad(value, parameters, allow_unused=True,
                                                   retain_graph=index + 1 < len(active))
                    gradient_calls += 1
                    for destination, gradient in zip(accumulated[name], gradients):
                        if gradient is not None:
                            if not bool(torch.isfinite(gradient).all()):
                                raise FloatingPointError(f'Nonfinite gradient in {name!r}.')
                            destination.add_(gradient.detach().to(device='cpu', dtype=torch.float64))
                del prediction, auxiliary, terms, objectives, active, extracted, observations
            for name in planned:
                if not math.isclose(seen[name], planned[name], rel_tol=1.e-6, abs_tol=1.e-7):
                    raise ValueError(f'Native panel denominator mismatch for {name!r}.')
            combined_contrast = [sum(accumulated[name][index] * (float(weights[name]) / total_weight)
                                     for name in weights) for index in range(len(parameters))]
            contrast_loss = sum(losses[name] * (float(weights[name]) / total_weight) for name in weights)
            scale = info['effective_scales']
            combined = [accumulated['native'][index] + accumulated['anchor'][index] * scale['anchor'] +
                        combined_contrast[index] * scale['contrast'] for index in range(len(parameters))]
            group_results = {}
            for group, indices in groups.items():
                native_norm = _norm(accumulated['native'], indices)
                anchor_norm = _norm(accumulated['anchor'], indices)
                contrast_norm = _norm(combined_contrast, indices)
                group_results[group] = {'parameter_tensors': len(indices),
                    'raw_norms': {'native': native_norm, 'anchor': anchor_norm, 'contrast': contrast_norm},
                    'prospective_calibrated_ramped_norms': {
                        'anchor': None if info['prospective_scales']['anchor'] is None else
                            anchor_norm * info['prospective_scales']['anchor'],
                        'contrast': None if info['prospective_scales']['contrast'] is None else
                            contrast_norm * info['prospective_scales']['contrast']},
                    'effective_calibrated_ramped_norms': {'anchor': anchor_norm * scale['anchor'],
                                                          'contrast': contrast_norm * scale['contrast']},
                    'combined_native_plus_enabled_finite_norm': _norm(combined, indices)}
            family_results = {}
            for name, weight in weights.items():
                factor = float(weight) / total_weight
                family_results[name] = {**family_metadata.get(name, {}),
                    'available': family_pairs[name] > 0, 'eligible_pair_count': family_pairs[name],
                    'configured_weight_fraction': factor, 'raw_contrast_loss': losses[name],
                    'raw_gradient_norms': {group: _norm(accumulated[name], indices) for group, indices in groups.items()},
                    'weighted_gradient_norms': {group: _norm(accumulated[name], indices) * factor
                                                for group, indices in groups.items()}}
            views[mode] = {'native_loss': losses['native'], 'anchor_loss': losses['anchor'],
                           'contrast_loss': contrast_loss, 'gradient_groups': group_results,
                           'families': family_results, 'native_term_denominators': planned,
                           'native_term_weights': term_weights, 'finite_dependency_work': work,
                           'native_prediction_calls': len(pairs), 'gradient_vector_products': gradient_calls}
    return {'schema': 'finite_observation_gradient_audit_v1', 'checkpoint_epoch': epoch,
        'native_sampling_epoch': sampling_epoch, 'native_sampling_description': native_sampling_description,
        'panel_case_ids': list(cases),
        'panel_pairs': [list(pair) for pair in pairs], 'eligible_pair_count': len(eligible),
        'pair_support': [{'pair': list(pair), 'eligible_families': list(eligibility[pair].eligible_families),
                          'missing_families': list(eligibility[pair].missing_families)} for pair in pairs],
        'configured_family_weights': weights, 'probe_count': probe_count, 'probe_seed': probe_seed,
        'basis': basis, 'variant': variant, 'calibration': info, 'reader_results': views,
        'parameter_ownership': {'organization': list(group_names['organization']),
                                'content_and_output': list(group_names['content_and_output'])},
        'interpretation': ('Panel gradients are diagnostic, not optimizer updates or a fixed-control causal claim. '
            'Each reader is the conditional single-reader objective; training samples readers equally. '
            'Disabled finite losses and legacy native checkpoints are prospective measurements. '
            'Other training objectives, including legacy response losses, are not included.'),
        'optimizer_updates': 0, 'fitting_epochs': 0, 'audit_seconds': time.perf_counter() - started}


def main(argv=None):
    from channelthermal.training.native_checkpoint import build_native_checkpoint_inference
    from channelthermal.training.observation_response import ThermalObservationBinding
    from finite_observation_evaluate import (_validate_train_panel, _verify_review_storage,
                                             verify_saved_thermal_observation_identity)
    from honf_runtime.compat import load_trusted_checkpoint
    from thermal_field_response_evaluate import ASSESSMENT9_CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--panels', required=True, type=Path, help='Saved matched TRAIN-panel JSON')
    parser.add_argument('--stage', required=True, choices=('development', 'formal'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Retain the existing audit instead of replacing it.')
    if args.cpu_threads < 1:
        raise ValueError('The CPU thread count must be positive.')
    storage = _verify_review_storage(args.output)
    torch.set_num_threads(args.cpu_threads)
    panel = json.loads(args.panels.read_text())[args.stage]
    cases = tuple(map(str, panel['train_case_ids']))
    pairs = tuple(tuple(map(str, pair)) for pair in panel['train_pairs'])
    saved = load_trusted_checkpoint(args.checkpoint, map_location='cpu')
    epoch = saved.get('epoch')
    if type(epoch) is not int or epoch < 1:
        raise ValueError('The checkpoint must record a completed positive epoch.')
    recipe = saved['experiment_identity']['recipe']
    finite = saved['experiment_identity']['provider_identity'].get('finite_observations')
    variant = finite['variant'] if finite else 'legacy_native'
    model, provider = build_native_checkpoint_inference(saved, validation_case_ids=ASSESSMENT9_CASES,
                                                         device=args.device)
    membership_count = len(provider.training_case_ids)
    if ((membership_count, len(cases), len(pairs)) not in ((150, 16, 8), (600, 60, 30)) or
            (args.stage == 'development') != (membership_count == 150)):
        raise ValueError('Use the saved pair-complete TRAIN150/16 or TRAIN600/60 matched panel.')
    _validate_train_panel(cases, pairs, provider.training_case_ids, named_pair_complete_panel=True)
    binding = ThermalObservationBinding(provider, recipe)
    identity = verify_saved_thermal_observation_identity(binding, saved)
    calibration = saved.get('provider_training_state', {}).get('finite_observations', {}).get('calibration')
    basis = finite.get('basis', 'coordinate') if finite else 'coordinate'
    probe_count = finite.get('probe_count', 64) if finite else 64
    probe_seed = finite.get('probe_seed', 130013) if finite else 130013
    result = audit_finite_observation_gradients(model, provider, binding, panel_case_ids=cases, panel_pairs=pairs,
        epoch=epoch, variant=variant, calibration=calibration, basis=basis, probe_count=probe_count,
        probe_seed=probe_seed, expected_family_scales=identity['family_scales'],
        native_sampling_description=NATIVE_SAMPLING_DESCRIPTION)
    device = next(model.parameters()).device
    result['provenance'] = {'checkpoint': str(args.checkpoint.resolve()), 'panels': str(args.panels.resolve()),
        'panel_stage': args.stage, 'training_membership_count': membership_count, 'device': str(device),
        'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
        'source_identity_verification': identity, 'storage': storage}
    if device.type == 'cuda':
        result['provenance']['device_name'] = torch.cuda.get_device_name(device)
    encoded = json.dumps(result, indent=2, allow_nan=False).encode('utf-8')
    if len(encoded) + 1 > storage['new_output_budget_bytes']:
        raise RuntimeError('The audit result exceeds its declared output budget.')
    _atomic_json(args.output, result)
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
