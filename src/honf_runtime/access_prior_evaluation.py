"""Fixed-support access-prior comparison on explicitly supplied observation pairs.

This measures a conditional change to selected within-role prior proportions;
contextual exchange, learned support selection and role masses remain in use.
"""
from __future__ import annotations

import os
import platform
import time
from collections.abc import Mapping

import numpy as np
import torch

from honf_forward_core.interaction_reused_member_routing.anchored import _measure_only_access_directory
from .observation_blocks import (extract_observations, observation_response_losses,
    prepare_observation_sampling, sample_observation_blocks, union_observation_dependencies)
from .observation_response_task import _case_seed


def _numbers(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numbers(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _timed(device, action):
    """Synchronize CUDA and report its allocator baseline and observed peak."""
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        baseline = {'allocated_bytes': int(torch.cuda.memory_allocated(device)),
                    'reserved_bytes': int(torch.cuda.memory_reserved(device))}
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    started = time.perf_counter()
    value = action()
    if cuda:
        torch.cuda.synchronize(device)
        memory = {'baseline': baseline, 'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
                  'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device))}
    else:
        memory = None
    return value, {'seconds': time.perf_counter() - started, 'cuda_allocator': memory}


def _physical(output, scale, offset, count):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != count:
        raise ValueError('The joint decoder must return one scene and every requested field query.')
    raw = output[0]
    scale = torch.as_tensor(scale, device=raw.device)
    offset = torch.as_tensor(offset, device=raw.device)
    if not scale.is_floating_point() or not offset.is_floating_point():
        raise TypeError('Physical field scales and offsets must be floating point.')
    dtype = torch.promote_types(raw.dtype, torch.promote_types(scale.dtype, offset.dtype))
    physical = raw.to(dtype) * scale.to(dtype) + offset.to(dtype)
    if physical.shape != raw.shape or not bool(torch.isfinite(physical).all()):
        raise ValueError('Physical output transforms must yield finite, aligned joint fields.')
    return physical.detach().cpu()


def _selected_rows(prepared, query):
    """Record every located row selected by the same core intervention routine."""
    source = prepared.e_access_directory
    control = _measure_only_access_directory(source, prepared.scene, ('full', 'restricted'))
    if query.batch_size != 1:
        raise ValueError('An observation field request must describe exactly one scene.')
    active = query.anchor_weights > 0
    if query.query_mask is not None:
        active = active & query.query_mask[..., None]
    anchors, counts = torch.unique(query.anchor_indices[active], sorted=True, return_counts=True)
    records = []
    for anchor, visits in zip(anchors.tolist(), counts.tolist()):
        for mode in ('full', 'restricted'):
            learned, measured = getattr(source, mode).row(0, anchor), getattr(control, mode).row(0, anchor)
            same = all(torch.equal(getattr(learned, key), getattr(measured, key))
                       for key in ('member_index', 'original_id', 'is_module', 'structural_self'))
            if not same:
                raise RuntimeError('The fixed-support access intervention changed selected members.')
            roles = {}
            for name, choose in (('M', learned.is_module), ('E', ~learned.is_module)):
                original = learned.access_weight[choose]
                after = measured.access_weight[choose]
                measures = measured.measure[choose]
                before_mass = float(original.sum())
                after_mass = float(after.sum())
                if measures.numel() and float(measures.sum()) > 0:
                    expected = original.sum() * measures / measures.sum()
                    maximum = float((after - expected).abs().max())
                else:
                    maximum = 0.
                roles[name] = {'original_member_ids': _numbers(learned.original_id[choose]),
                               'member_indices': _numbers(learned.member_index[choose]),
                               'selected_count': int(choose.sum()),
                               'learned_prior_mass': before_mass, 'measure_only_prior_mass': after_mass,
                               'mass_absolute_difference': abs(after_mass - before_mass),
                               'mass_consistent': bool(torch.isclose(original.sum(), after.sum(), rtol=1e-5, atol=1e-7)),
                               'maximum_measure_formula_absolute_error': maximum}
            records.append({'anchor_slot': int(anchor),
                            'anchor_original_id': int(prepared.scene.environment_ids[0, anchor]),
                            'query_anchor_reads': int(visits), 'read_mode': mode, 'roles': roles})
    return {'selected_members_identical': True, 'located_rows': records,
            'audit_method': 'The same post-selection access-row transformation used by the decoder; only actually addressed rows are listed.'}


def _pair_panel(pairs):
    pairs = tuple(pairs)
    if any(not isinstance(pair, (tuple, list)) for pair in pairs):
        raise ValueError('Provide distinct, explicitly selected pairs of two different case IDs.')
    pairs = tuple(tuple(map(str, pair)) for pair in pairs)
    if not pairs or len(set(pairs)) != len(pairs) or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError('Provide distinct, explicitly selected pairs of two different case IDs.')
    return pairs


def evaluate_access_prior_panel(model, provider, pairs):
    """Measure each policy on the same sampled requests and prepared scenes.

    The fixed observation sampling matches the existing evaluation namespace.
    Preparation and both paired/standalone decode timings include only the
    named phase; process time includes planning, audits and metric extraction.
    """
    started = time.perf_counter()
    pairs = _pair_panel(pairs)
    if getattr(model.core, 'within_role_access_prior', None) != 'learned':
        raise ValueError('Use a learned-prior checkpoint for the fixed-support comparison.')
    if getattr(model.core, 'reader_capability', None) == 'all_native':
        raise ValueError('The selected reader has no learned anchored access rows.')
    identity = provider.identity_payload().get('observation_response')
    if not isinstance(identity, Mapping):
        raise ValueError('The checkpoint must have an enabled observation-response provider.')
    seed, count = int(identity['pair_seed']), int(identity['seeds_per_pair'])
    if not 1 <= count <= 128:
        raise ValueError('Observation draw count must match the declared bounded profile.')
    evaluation_view = getattr(provider, 'with_observation_evaluation_pairs', None)
    provider = evaluation_view(pairs) if callable(evaluation_view) else provider
    binding = provider.observation_binding
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError('The evaluated model has no device-bearing parameters.') from error
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            pair_records, groups = [], {}
            for pair in pairs:
                planning_started = time.perf_counter()

                def plan_pair():
                    binding.validate_pair(pair)
                    blocks = tuple(binding.observation_blocks(pair))
                    plan = prepare_observation_sampling(blocks, max_neighbors=4)
                    draws = sample_observation_blocks(plan, count, np.random.default_rng(_case_seed(seed, 10, pair)))
                    return blocks, draws, union_observation_dependencies(blocks, draws)

                (blocks, draws, union), planning = _timed(device, plan_pair)
                locations = []
                for dependency in union.dependencies:
                    if dependency.scene_id not in pair:
                        raise ValueError('An observation dependency does not belong to its declared pair.')
                    key = (dependency.scene_id, dependency.frame, dependency.time)
                    group = groups.setdefault(key, {'dependencies': [], 'lookup': {}})
                    if dependency not in group['lookup']:
                        group['lookup'][dependency] = len(group['dependencies'])
                        group['dependencies'].append(dependency)
                    locations.append((key, group['lookup'][dependency]))
                pair_records.append({'pair': pair, 'blocks': blocks, 'draws': draws, 'union': union,
                    'locations': locations, 'planning': planning,
                    'planning_seconds': time.perf_counter() - planning_started})

            scene_records, decoded = [], {}
            for key, group in groups.items():
                case_id, frame, instant = key
                deps = tuple(group['dependencies'])
                (scene, query, scale, offset), field_request = _timed(
                    device, lambda: binding.field_request(model, case_id, deps))
                prepared, preparation = _timed(device, lambda: model.core.prepare(scene))
                audit, audit_time = _timed(device, lambda: _selected_rows(prepared, query))
                evaluations = {}
                decoded[key] = {}
                for policy in ('learned', 'measure_only'):
                    intervention = None if policy == 'learned' else {'within_role_access_prior': 'measure_only'}

                    def paired_action():
                        full, restricted, shared = model.core.decode_paired(prepared, query, intervention=intervention)
                        values = (_physical(full.joint_fields, scale, offset, len(deps)),
                                  _physical(restricted.joint_fields, scale, offset, len(deps)))
                        counts = {'shared': _numbers(shared), 'full': _numbers(full.work_counts),
                                  'restricted': _numbers(restricted.work_counts)}
                        return values, counts

                    (values, paired_counts), timing = _timed(device, paired_action)
                    decoded[key][(policy, 'paired_full')], decoded[key][(policy, 'paired_restricted')] = values
                    restricted_intervention = {'read_mode': 'restricted', **(intervention or {})}

                    def standalone_action():
                        output = model.core.decode(prepared, query, intervention=restricted_intervention)
                        return _physical(output.joint_fields, scale, offset, len(deps)), _numbers(output.work_counts)

                    (values, solo_counts), solo_timing = _timed(device, standalone_action)
                    decoded[key][(policy, 'standalone_restricted')] = values
                    first = decoded[key][(policy, 'paired_restricted')]
                    second = decoded[key][(policy, 'standalone_restricted')]
                    evaluations[policy] = {'paired_full_restricted': {**timing, 'work': paired_counts},
                        'standalone_restricted': {**solo_timing, 'work': solo_counts,
                            'maximum_absolute_field_difference_from_paired': float((first - second).abs().max()),
                            'matches_paired_fields': bool(torch.allclose(first, second, rtol=1e-5, atol=1e-6))}}
                scene_records.append({'case_id': _numbers(case_id), 'frame': _numbers(frame), 'time': _numbers(instant),
                    'dependencies': len(deps), 'field_request': field_request, 'preparation': preparation,
                    'preparation_work': _numbers(prepared.work_counts), 'support_audit': audit,
                    'support_audit_cost': audit_time, 'policies': evaluations})

            rows = []
            view_names = ('paired_full', 'paired_restricted', 'standalone_restricted')
            for record in pair_records:
                blocks, draws, union = record['blocks'], record['draws'], record['union']
                policies = {}
                for policy in ('learned', 'measure_only'):
                    views = {}
                    for view in view_names:
                        values = torch.stack([decoded[key][(policy, view)][index]
                                              for key, index in record['locations']], dim=0)[None]
                        predictions = [torch.stack([extract_observations(block, side, values, union.mappings[bi][side])
                                                   for side in range(2)], dim=1)
                                       for bi, block in enumerate(blocks)]
                        views[view] = _numbers(observation_response_losses(
                            blocks, predictions, draws, include_component_estimates=True))
                    policies[policy] = views
                dependencies = [{'scene_id': _numbers(dep.scene_id), 'frame': _numbers(dep.frame),
                                 'time': _numbers(dep.time), 'native_id': _numbers(dep.native_id),
                                 'coordinates': _numbers(dep.coordinates)} for dep in union.dependencies]
                rows.append({'pair': list(record['pair']), 'draws': _numbers(draws.identities()),
                             'draw_count': len(draws.draws), 'requested_dependencies': union.requested_dependencies,
                             'requested_extraction_terms': union.requested_terms, 'dependencies': dependencies,
                             'planning': record['planning'], 'planning_seconds': record['planning_seconds'],
                             'policies': policies})
    finally:
        model.train(was_training)
    names = ('extra_absolute', 'point_response', 'spatial_response')
    summary = {policy: {view: {name: sum(row['policies'][policy][view][name] for row in rows) / len(rows)
                                      for name in names} for view in view_names}
               for policy in ('learned', 'measure_only')}
    measured_phases = [row['planning'] for row in rows]
    for scene in scene_records:
        measured_phases.extend((scene['field_request'], scene['preparation'], scene['support_audit_cost']))
        measured_phases.extend(value[mode] for value in scene['policies'].values()
                               for mode in ('paired_full_restricted', 'standalone_restricted'))
    peak_allocated = [phase['cuda_allocator']['peak_allocated_bytes'] for phase in measured_phases
                      if phase['cuda_allocator'] is not None]
    peak_reserved = [phase['cuda_allocator']['peak_reserved_bytes'] for phase in measured_phases
                     if phase['cuda_allocator'] is not None]
    device_properties = None
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        device_properties = {'name': properties.name, 'total_memory_bytes': int(properties.total_memory),
                             'compute_capability': [int(properties.major), int(properties.minor)],
                             'uuid': _numbers(getattr(properties, 'uuid', None))}
    return {'schema': 'same_support_access_prior_observation_evaluation_v1',
        'definition': 'Equal-pair mean of fixed-mass sampled observation estimators. The measure-only policy retains learned '
                      'contextual exchange, selected support and per-role masses. Component estimates are raw and exclude training coefficients.',
        'sampling_epoch_namespace': 10, 'pair_seed': seed, 'seeds_per_pair': count,
        'pair_count': len(rows), 'scene_preparations': len(scene_records),
        'scene_grouping': 'One preparation for each distinct case, frame and time across the explicit pair panel.',
        'device': str(device), 'device_name': device_properties['name'] if device_properties else 'CPU',
        'device_properties': device_properties,
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'cuda_runtime': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
        'cuda_allocator_peak_allocated_bytes': max(peak_allocated) if peak_allocated else None,
        'cuda_allocator_peak_reserved_bytes': max(peak_reserved) if peak_reserved else None,
        'timing_scope': 'Synchronized field-request, preparation and support audit are reported separately. Paired/standalone decode timings include physical output transforms and transfer to CPU. Process time includes all orchestration and metric extraction. Timings are single ordered calls, not a controlled speed comparison.',
        'summary': summary, 'pairs': rows, 'scenes': scene_records, 'process_seconds': time.perf_counter() - started}
