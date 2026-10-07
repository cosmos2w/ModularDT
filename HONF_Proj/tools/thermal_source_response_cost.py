#!/usr/bin/env python3
"""Complete native source-response costs and prepared, model-only reuse.

Cold calls rebuild context, native stencils, near/far coefficients and flow.
Prepared scopes reuse only coefficients of the exact geometry/receivers. The
incumbent runs its actual optimized P0/P1/P2 wrapper; the candidate contains no
old thermal fallback. All precision is FP32, except explicitly widened error
summaries. Optional SVD acts on a learned K, never a physical operator.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / 'src', ROOT / 'Case_ThermalChannel/src', ROOT / 'tools'):
    if str(source) not in sys.path: sys.path.insert(0, str(source))

from thermal_campaign_benchmark import differentiable_arguments, optional_fine_work, phase_outputs, tensor_error
from thermal_campaign_heat_inference import snapshot_forward_state, verify_frozen_forward
from thermal_response_refinement_cost import (
    alternating_orders,
    authorized_device,
    memory_result,
    prepare_inputs,
    reset_memory,
    synchronize,
)
from thermal_response_refinement_evaluation import write_json

NATIVE_ROLES = ('pred_field', 'pred_internal_temperature', 'pred_interface', 'pred_port_condition')


def physical_heat(arguments, saved):
    """Uncentered physical source amplitudes; selected-TRAIN scale only."""
    if arguments.get('local_module_params') is not None:
        return arguments['local_module_params'][..., 0]
    value = arguments['structure']['heat_powers']
    if saved['train_config']['dataset'].get('normalize_inputs', False):
        stats = saved['global_normalization_stats']
        return value * value.new_tensor(stats['heat_power_std']).reshape(()) + value.new_tensor(stats['heat_power_mean']).reshape(())
    return value


def incumbent_physical(output, saved):
    """Retained output normalization; ports already use physical units."""
    result = {name: output[name] for name in NATIVE_ROLES}
    if saved['train_config']['dataset'].get('normalize_targets', False):
        stats = saved['global_normalization_stats']
        interface_prefix = 'interface_targets' if 'interface_targets_mean' in stats and 'interface_targets_std' in stats else 'interface_target'
        for name, mean, std in (
                ('pred_field', 'field_mean_by_channel', 'field_std_by_channel'),
                ('pred_internal_temperature', 'internal_temperature_mean', 'internal_temperature_std'),
                ('pred_interface', interface_prefix + '_mean', interface_prefix + '_std')):
            value = result[name]
            result[name] = value * value.new_tensor(stats[std]) + value.new_tensor(stats[mean])
    return result


def prepare_candidate(model, arguments):
    """Full native support; no sampled/teacher material values are read."""
    condition = arguments.get('interface_condition')
    ntheta = int(condition.shape[-2]) if condition is not None else 64
    return model.prepare_native(arguments['structure'], arguments['query_xy'],
        local_query_points=arguments.get('local_query_points'), ntheta=ntheta, chunk_size=512)


def complete_call(name, model, saved, arguments, heat=None):
    if name == 'incumbent':
        return incumbent_physical(model(**arguments), saved)
    prepared = prepare_candidate(model, arguments)
    return model.apply_native(prepared, physical_heat(arguments, saved) if heat is None else heat)


def native_adjoint(output):
    """One fixed nonuniform physical-role adjoint shared across all arms."""
    weights, scales = {}, {}
    for name in NATIVE_ROLES:
        value = output[name]
        channels = value.shape[-1]
        scale = value.detach().reshape(-1, channels).square().mean(0).sqrt().clamp_min(1)
        weights[name] = torch.linspace(.5, 1.5, value.numel(), device=value.device,
            dtype=value.dtype).reshape_as(value) / scale / value.numel()
        scales[name] = scale.cpu().tolist()
    return weights, scales


def candidate_work(model, arguments, saved):
    with torch.no_grad():
        prepared = prepare_candidate(model, arguments)
        output = model.apply_native(prepared, physical_heat(arguments, saved))
    native = prepared['thermal']
    organization = model.thermal.core.export_organization(native.response)
    return {'neural_grid_receiver_rows': int(native.grid_indices.numel()),
        'source_rows': int(native.context.present.sum()),
        'environment_context_rows': int(native.context.environment_present.sum()),
        'near_coefficient_rows': organization['near_work'], 'far_coefficient_rows': organization['far_rows'],
        'far_scope': 'receiver/source nonlinear coefficients' if native.response.mode == 'direct' else 'receiver/mode nonlinear coefficients; factor application and exact near subtraction',
        'allocated_modes': 0 if native.response.group_present is None else int(native.response.group_present.numel()),
        'valid_modes': 0 if native.response.group_present is None else int(native.response.group_present.sum()),
        'output_shapes': {name: list(output[name].shape) for name in NATIVE_ROLES},
        'initial_ports': 'not applicable; no P0/P1/P2 thermal refinement in response candidate',
        'old_thermal_wrapper_calls': 0}


def benchmark(checkpoints, output_dir, *, device='cuda:0', physical_gpu_ids=(0, 2), include_b8=True):
    from channelthermal.evaluation.loading import load_model
    from channelthermal.source_response import load_source_response_model
    from thermal_development import validate_generated_output

    if set(checkpoints) != {'incumbent', 'R-direct', 'R-group'}:
        raise ValueError('Native cost requires incumbent and the matched direct/group pair.')
    device = authorized_device(device, physical_gpu_ids)
    output_dir = validate_generated_output(output_dir)
    if (output_dir / 'receipt.json').exists(): raise FileExistsError('Preserve prior cost attempts.')
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time()
    receipt = {'pid': os.getpid(), 'start_unix': started, 'status': 'running',
        'device': str(device), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'solver_attempts': 0, 'optimizer_updates': 0, 'complete_calls': 0, 'input_vjps': 0}
    write_json(output_dir / 'receipt.json', receipt)
    models, saved, snapshots, rows, prepared_rows = {}, {}, {}, [], []
    try:
        torch.set_num_threads(2)
        for name, path in checkpoints.items():
            loader = load_model if name == 'incumbent' else load_source_response_model
            model, metadata = loader(Path(path), device)
            model.eval().requires_grad_(False)
            if name == 'incumbent':
                model.core.receiver_chunk_size = 512
                model.core.backend.set_execution_mode(model.core.backend.execution_mode, receiver_chunk_size=512)
            models[name], saved[name], snapshots[name] = model, metadata, snapshot_forward_state(model)
        inputs, panels, manifest, stats = prepare_inputs(saved['incumbent'], device)
        if not include_b8: inputs = {name: value for name, value in inputs.items() if name.startswith('B1')}
        for name, metadata in saved.items():
            if name != 'incumbent' and metadata.get('fit_identity', {}).get('recipe', {}).get('manifest_sha256') != manifest['manifest_sha256']:
                raise ValueError(f'Complete cost requires fixed primary membership: {name}')
            for key, value in stats.items():
                if not np.array_equal(np.asarray(metadata['global_normalization_stats'][key], dtype=np.float32), value):
                    raise ValueError(f'Complete cost cannot mix TRAIN normalization: {name}/{key}')
        def call(name, arguments, heat=None):
            receipt['complete_calls'] += 1
            return complete_call(name, models[name], saved[name], arguments, heat)

        for condition, arguments in inputs.items():
            condition_rows = {name: {'arm': name, 'checkpoint': str(Path(checkpoints[name]).resolve()),
                'checkpoint_epoch': saved[name]['epoch'], 'condition': condition, **panels[condition],
                'inference_samples': [], 'heat_query_vjp_samples': [], 'geometry_vjp_samples': []} for name in models}
            with torch.no_grad():
                for name in models:
                    output = call(name, arguments); synchronize(device); del output
            for repetition, order in enumerate(alternating_orders(models, 5)):
                for name in order:
                    baseline = reset_memory(device); before = perf_counter()
                    with torch.no_grad(): output = call(name, arguments)
                    synchronize(device)
                    condition_rows[name]['inference_samples'].append({'repetition': repetition,
                        'seconds': perf_counter() - before, **memory_result(device, baseline)})
                    del output
            adjoints = None
            scopes = ('heat_query', 'geometry') if condition == 'B1Q8192_high' else ('heat_query',)
            for scope in scopes:
                # Warm actual input differentiation, separately from timed calls.
                for name in models:
                    differentiable, leaves = differentiable_arguments(arguments, saved[name])
                    if scope == 'geometry':
                        center = arguments['structure']['module_centers'].detach().clone().requires_grad_()
                        differentiable['structure']['module_centers'] = center
                        leaves = {'input.module_centers': center}
                    output = call(name, differentiable, physical_heat(differentiable, saved[name]))
                    if adjoints is None: adjoints, scales = native_adjoint(output)
                    objective = sum((output[key] * value).sum() for key, value in adjoints.items())
                    gradients = torch.autograd.grad(objective, tuple(leaves.values()), allow_unused=True)
                    receipt['input_vjps'] += 1
                    synchronize(device)
                    del output, objective, gradients, differentiable, leaves
                for repetition, order in enumerate(alternating_orders(models, 2)):
                    for name in order:
                        differentiable, leaves = differentiable_arguments(arguments, saved[name])
                        if scope == 'geometry':
                            center = arguments['structure']['module_centers'].detach().clone().requires_grad_()
                            differentiable['structure']['module_centers'] = center
                            leaves = {'input.module_centers': center}
                        baseline = reset_memory(device); before = perf_counter()
                        output = call(name, differentiable, physical_heat(differentiable, saved[name]))
                        synchronize(device); forward_seconds = perf_counter() - before
                        objective = sum((output[key] * value).sum() for key, value in adjoints.items())
                        synchronize(device); before = perf_counter()
                        gradients = torch.autograd.grad(objective, tuple(leaves.values()), allow_unused=True)
                        synchronize(device); vjp_seconds = perf_counter() - before
                        receipt['input_vjps'] += 1
                        if not all(value is None or bool(value.isfinite().all()) for value in gradients):
                            raise RuntimeError('Nonfinite native physical input VJP.')
                        condition_rows[name][scope + '_vjp_samples'].append({'repetition': repetition,
                            'forward_seconds': forward_seconds, 'vjp_seconds': vjp_seconds,
                            'complete_seconds': forward_seconds + vjp_seconds,
                            'gradient_l2': {key: None if value is None else float(value.norm()) for key, value in zip(leaves, gradients)},
                            **memory_result(device, baseline)})
                        del output, objective, gradients, differentiable, leaves
            for name, model in models.items():
                row = condition_rows[name]
                row['inference_median_seconds'] = float(np.median([sample['seconds'] for sample in row['inference_samples']]))
                row['heat_query_forward_vjp_median_seconds'] = float(np.median([sample['complete_seconds'] for sample in row['heat_query_vjp_samples']]))
                if row['geometry_vjp_samples']:
                    row['geometry_forward_vjp_median_seconds'] = float(np.median([sample['complete_seconds'] for sample in row['geometry_vjp_samples']]))
                row['common_adjoint_scales'] = scales
                if name != 'incumbent':
                    row['actual_work'] = candidate_work(model, arguments, saved[name])
                    receipt['complete_calls'] += 1
                else:
                    with torch.no_grad(), phase_outputs(model.core) as phases, optional_fine_work(model.core.backend) as fine:
                        output = call(name, arguments)
                    synchronize(device)
                    observed = sorted({key.split('.')[0] for key in phases})
                    if observed != ['P0', 'P1', 'P2']: raise RuntimeError('Incumbent cost did not execute actual P0/P1/P2.')
                    row['actual_work'] = {'thermal_phases': observed, 'fine_kernel_work': fine, 'receiver_chunk_size': 512,
                        'output_shapes': {key: list(output[key].shape) for key in NATIVE_ROLES}}
                    del output, phases
                rows.append(row)
                print({'arm': name, 'condition': condition, 'inference': row['inference_median_seconds'],
                    'heat_query_forward_vjp': row['heat_query_forward_vjp_median_seconds']}, flush=True)
            if condition.startswith('B1'):
                for name in ('R-direct', 'R-group'):
                    prepared_rows.extend(prepared_forcing_cost(models[name], saved[name], arguments, condition, device, receipt=receipt))
            write_json(output_dir / 'cost.json', {'rows': rows, 'prepared_rows': prepared_rows})
            gc.collect()
        frozen = {name: verify_frozen_forward(models[name], snapshots[name]) for name in models}
        if not all(value['passed'] for value in frozen.values()): raise RuntimeError('Native cost changed frozen weights.')
        result = {'rows': rows, 'prepared_rows': prepared_rows, 'frozen_state': frozen,
            'manifest_sha256': manifest['manifest_sha256'], 'precision': 'FP32',
            'scope': 'complete declared native field/material/surface/final-port predictions; five alternating warmed cold inference and two warmed full input VJPs',
            'cold_semantics': 'checkpoint/device kernels resident; every call rebuilds context, stencils, thermal coefficients and flow preparation',
            'overlapping_scopes': 'context/coefficients/role extraction and flow are inside complete calls; never add scope costs twice',
            'memory_scope': 'all three models/panels resident; extra allocated subtracts invocation baseline; reserved includes allocator history',
            'solver_attempts': 0, 'optimizer_updates': 0}
        write_json(output_dir / 'cost.json', result)
        receipt.update(status='completed', exit_code=0)
        return result
    except BaseException as error:
        receipt.update(status='failed', exit_code=1, error=repr(error)); raise
    finally:
        receipt['end_unix'] = time()
        receipt['associated_gpu_hours'] = (receipt['end_unix'] - started) / 3600 if device.type == 'cuda' else 0.
        write_json(output_dir / 'receipt.json', receipt)


def prepared_forcing_cost(model, saved, arguments, condition, device, *, receipt=None):
    """Cold versus prepared applications and exact affine-role parity."""
    heat = physical_heat(arguments, saved)
    present = arguments['structure']['module_present']
    delta = torch.zeros_like(heat)
    active = (present[0] > .5).nonzero().flatten()
    if active.numel() >= 2:
        amplitude = heat[0, active].mean() * .05
        delta[0, active[0]], delta[0, active[-1]] = amplitude, -amplitude
    allocations = (heat, heat + delta, heat - delta)
    baseline = reset_memory(device); before = perf_counter()
    with torch.no_grad(): prepared = prepare_candidate(model, arguments)
    synchronize(device)
    preparation = {'scope': 'full native coefficient/context/flow preparation (overlaps cold)',
        'seconds': perf_counter() - before, **memory_result(device, baseline)}
    phase_costs = prepared_native_scope_cost(model,prepared,arguments,heat,device)
    if receipt is not None:
        receipt['complete_calls'] += 1
        receipt['source_context_only_calls'] = receipt.get('source_context_only_calls',0)+1
        receipt['flow_context_only_calls'] = receipt.get('flow_context_only_calls',0)+1
    def apply(preparation, forcing):
        if receipt is not None: receipt['complete_calls'] += 1
        return model.apply_native(preparation, forcing)
    with torch.no_grad():
        reference = apply(prepared, heat)
        changed = apply(prepared, heat + delta)
        cold = apply(prepare_candidate(model, arguments), heat + delta)
        increment = model.thermal.apply_native(prepared['thermal'], delta, increment=True)
        if receipt is not None: receipt['thermal_increment_calls'] = receipt.get('thermal_increment_calls', 0) + 1
    cold_errors = {name: tensor_error(changed[name], cold[name], rtol=2e-5, atol=2e-6) for name in NATIVE_ROLES}
    errors = {name: tensor_error(changed[name] - reference[name], increment[name], rtol=2e-5, atol=2e-6)
        for name in ('fluid_temperature', 'pred_interface', 'pred_internal_temperature')}
    errors['final_port_outside_temperature'] = tensor_error(changed['pred_port_condition'][..., 3:4] - reference['pred_port_condition'][..., 3:4],
        increment['outside_temperature'], rtol=2e-5, atol=2e-6)
    del reference, changed, increment, cold
    def run(scope):
        with torch.no_grad():
            if scope == 'cold_three_forcings':
                return [apply(prepare_candidate(model, arguments), forcing) for forcing in allocations]
            return [apply(prepared, forcing) for forcing in allocations]
    for scope in ('cold_three_forcings', 'prepared_three_forcings'):
        output = run(scope); synchronize(device); del output
    rows = []
    for repetition in range(5):
        order = ('cold_three_forcings', 'prepared_three_forcings')
        if repetition % 2: order = order[::-1]
        for scope in order:
            baseline = reset_memory(device); before = perf_counter(); output = run(scope); synchronize(device)
            rows.append({'arm': saved.get('source_response_mode', model.thermal.core.mode), 'condition': condition,
                'scope': scope, 'allocation_count': 3, 'repetition': repetition,
                'seconds': perf_counter() - before, **memory_result(device, baseline),
                'preparation': preparation, 'cold_prepared_increment_errors': errors,
                'instrumented_prepared_scopes': phase_costs,
                'cold_prepared_full_output_errors': cold_errors,
                'delta_physical_heat': delta.cpu().tolist(),
                'flow_scope': 'prepared flow source states, query flow read repeated per application; no temperature caching'})
            del output
    return rows


def prepared_native_scope_cost(model, prepared, arguments, heat, device):
    """Measured nested scopes of one complete prepared native application.

    Per-method synchronization is confined to this diagnostic call. These
    overlapping scopes are not added to ordinary complete-call timings.
    """
    result = {}
    for name, function in (('source_environment_context',model.thermal.prepare_context),('flow_context',model.flow.prepare_flow)):
        baseline = reset_memory(device); before = perf_counter()
        with torch.no_grad(): value = function(arguments['structure'])
        synchronize(device)
        result[name] = {'seconds':perf_counter()-before,**memory_result(device,baseline)}
        del value
    owners = ((model.thermal.core,'apply_forcing','thermal_coefficient_application'),
        (model.flow,'read_flow','flow_query_read'))
    originals = []
    def timed(function, name):
        def call(*args,**kwargs):
            synchronize(device); before = perf_counter()
            value = function(*args,**kwargs)
            synchronize(device)
            result[name] = {'seconds':perf_counter()-before}
            return value
        return call
    try:
        for owner, method, name in owners:
            existed = method in owner.__dict__
            original = getattr(owner,method)
            originals.append((owner,method,existed,original))
            setattr(owner,method,timed(original,name))
        baseline = reset_memory(device); before = perf_counter()
        with torch.no_grad(): output = model.apply_native(prepared,heat)
        synchronize(device)
        result['complete_prepared_native'] = {'seconds':perf_counter()-before,**memory_result(device,baseline),
            'output_shapes':{name:list(output[name].shape) for name in NATIVE_ROLES}}
        result['native_role_extraction_and_validation_seconds'] = max(0.,result['complete_prepared_native']['seconds']
            - result['thermal_coefficient_application']['seconds'] - result['flow_query_read']['seconds'])
        result['scope'] = 'same call: role extraction includes native interpolation, physical ports, material maximum, validation and field assembly; synchronization instrumented'
        return result
    finally:
        for owner, method, existed, original in reversed(originals):
            if existed: setattr(owner,method,original)
            else: delattr(owner,method)


def balanced_radii(heat, present, train_minimum, train_maximum):
    """Predeclared source-only balanced-box domain; no outcome calibration."""
    count = present.sum(-1, keepdim=True).clamp_min(1)
    cap = .20 * (heat * present).sum(-1, keepdim=True) / count
    return torch.minimum(torch.minimum(heat - train_minimum, train_maximum - heat), cap).clamp_min(0) * present


def low_rank_response(matrix, radii, absolute_budget):
    """Standard SVD control of learned K on balanced forcing, not physical A.

    A Frobenius bound dominates spectral error, then ||d||2 <= ||radii||2
    bounds every output component. Rank choice reads no reference outcome.
    """
    if matrix.ndim != 2 or radii.shape != (matrix.shape[1],):
        raise ValueError('Low-rank response needs [receiver,source] K and aligned radii.')
    if absolute_budget < 0: raise ValueError('Approximation budget must be nonnegative.')
    columns = matrix.shape[1]
    if columns == 0: raise ValueError('SVD control requires active sources.')
    projection = torch.eye(columns, device=matrix.device, dtype=matrix.dtype) - matrix.new_ones(columns, columns) / columns
    balanced = matrix @ projection
    left, singular, right = torch.linalg.svd(balanced, full_matrices=False)
    tails = torch.cat((singular.square().flip(0).cumsum(0).flip(0), singular.new_zeros(1))).sqrt() * radii.norm()
    permitted = (tails <= absolute_budget).nonzero().flatten()
    rank = int(permitted[0]) if permitted.numel() else singular.numel()
    return {'left': left[:, :rank] * singular[:rank], 'right': right[:rank], 'rank': rank,
        'omission_bound': float(tails[rank]), 'matrix': balanced, 'singular_values': singular,
        'physical_operator_factorizations': 0, 'scope': 'SVD of learned response matrix, projected to balanced source increments'}


def native_temperature_vector(thermal, prepared, grid_values):
    """Identical scalar-temperature receiver catalogue for both controls.

    Surface and material interpolation uses nonnegative native stencil weights;
    the grouped grid bound therefore bounds each extracted temperature too.
    Flux and h-effective have distinct units/laws and are not pooled into this
    temperature-only compression budget.
    """
    return torch.cat([thermal._interpolate(grid_values, prepared.stencils[role]).flatten(1)
        for role in ('fluid', 'surface', 'material') if role in prepared.stencils], -1)


def native_temperature_matrix(thermal, prepared, active):
    kernel = prepared.response.dense_kernel()
    matrices = [thermal._interpolate(kernel, prepared.stencils[role]).reshape(kernel.shape[0], -1, kernel.shape[2])
        for role in ('fluid', 'surface', 'material') if role in prepared.stencils]
    return torch.cat(matrices, 1)[0, :, active]


def tensor_storage_bytes(values):
    return sum(value.numel() * value.element_size() for value in values if torch.is_tensor(value))


def compression_probe(model, prepared, heat, physical_increments, radii, train_response_rms, device, *,
                      physical_reference_increments=None, physical_receiver_mask=None, array_sink=None, receipt=None):
    """Input-bounded model-only response compression of explicit saved deltas.

    Budgets are sealed fractions of an externally qualified TRAIN-addendum RMS.
    This probe never reads physical outcomes or applies modes to the absolute
    baseline. Domain failures use the full response and remain explicit.
    """
    thermal, native = model.thermal, prepared['thermal']
    def count(name, amount=1):
        if receipt is not None: receipt[name] = receipt.get(name,0)+amount
    if heat.shape[0] != 1: raise ValueError('Compression pricing currently requires one native layout.')
    if train_response_rms <= 0: raise ValueError('TRAIN response RMS must be positive and predeclared.')
    increments = torch.as_tensor(physical_increments, dtype=heat.dtype, device=heat.device)
    if increments.ndim == 2: increments = increments[:, None]
    if increments.ndim != 3 or increments.shape[1:] != heat.shape:
        raise ValueError('Saved physical increments must be [allocation,B,M] source aligned.')
    active = (native.source_present[0] > .5).nonzero().flatten()
    if not active.numel(): raise ValueError('Compression requires an active source.')
    epsilon = torch.finfo(heat.dtype).eps * 64
    feasible = ((increments * native.source_present).sum(-1).abs() <= epsilon * increments.abs().sum(-1).clamp_min(1)).all(-1)
    feasible &= (increments.abs() <= radii[None] + epsilon).all((1,2))
    if bool((increments * (1-native.source_present)).abs().max() > epsilon):
        raise ValueError('Saved increments may not actuate padded source slots.')
    with torch.no_grad():
        exact = [native_temperature_vector(thermal,native,thermal.core.apply_increment(native.response,delta)) for delta in increments]
        count('temperature_catalogue_applications',len(exact))
        absolute_baseline = native_temperature_vector(thermal,native,thermal.core.apply_forcing(native.response,heat))
        count('absolute_temperature_applications')
    truth = None
    if physical_reference_increments is not None:
        truth = torch.as_tensor(physical_reference_increments,dtype=torch.float64,device=heat.device)
        if truth.shape != (len(exact),*exact[0].shape):
            raise ValueError('Physical finite-response increments must align with the temperature catalogue [allocation,B,R].')
        physical_mask = torch.ones_like(truth,dtype=torch.bool) if physical_receiver_mask is None else torch.broadcast_to(
            torch.as_tensor(physical_receiver_mask,dtype=torch.bool,device=heat.device),truth.shape)
        if not bool(physical_mask.flatten(1).any(-1).all()):
            raise ValueError('Every physical finite response needs a nonempty valid receiver mask.')
        if not bool(truth[physical_mask].isfinite().all()): raise ValueError('Valid physical finite responses must be finite.')
    elif physical_receiver_mask is not None:
        raise ValueError('Physical error masks require saved physical reference increments.')
    if array_sink is not None:
        array_sink['absolute_baseline_temperature'] = absolute_baseline.detach().cpu().numpy()
        array_sink['physical_heat_increments'] = increments.detach().cpu().numpy()
        array_sink['full_model_temperature_increments'] = torch.stack(exact).detach().cpu().numpy()
        if truth is not None:
            array_sink['physical_reference_temperature_increments'] = truth.detach().cpu().numpy()
            array_sink['physical_receiver_mask'] = physical_mask.detach().cpu().numpy()
    response = native.response
    coefficient_bytes = tensor_storage_bytes((response.far_kernel,response.receiver_functions,response.source_membership,
        response.near_weight,response.near_indices,response.near_values,response.offset))
    matrix = None
    matrix_formation = None
    if response.mode == 'direct':
        baseline = reset_memory(device); before = perf_counter()
        with torch.no_grad(): matrix = native_temperature_matrix(thermal,native,active)
        count('learned_temperature_matrix_exports')
        synchronize(device)
        matrix_formation = {'seconds':perf_counter()-before,**memory_result(device,baseline),
            'matrix_bytes':tensor_storage_bytes((matrix,)), 'shape':list(matrix.shape)}
    rows = []
    for fraction in (0.,.005,.01,.02):
        budget = float(train_response_rms)*fraction
        baseline = reset_memory(device); before = perf_counter()
        with torch.no_grad():
            approximation = thermal.core.compress_response(response,radii,budget) if response.mode == 'group' else low_rank_response(matrix,radii[0,active],budget)
        count('group_compression_formations' if response.mode == 'group' else 'learned_K_SVD_formations')
        synchronize(device)
        formation = {'seconds':perf_counter()-before, **memory_result(device,baseline)}
        if response.mode == 'group':
            storage = tensor_storage_bytes((approximation.keep,approximation.omitted_bound,approximation.budget,approximation.forcing_radii))
            bounds = native_temperature_vector(thermal,native,approximation.omitted_bound)
            details = {'allocated_receiver_mode_slots':int(approximation.keep.numel()),
                'retained_receiver_mode_slots':int(approximation.keep.sum()),
                'omitted_receiver_mode_slots':int((~approximation.keep).sum()),
                'nonzero_receiver_functions':int((response.receiver_functions!=0).any(-1).sum()),
                'source_membership_nonzero':int((response.source_membership!=0).sum()),
                'actual_application': 'dense shaped a and b with coefficient omission mask; exact sparse near correction remains',
                'actual_sparse_application_savings':False,
                'near_coefficient_rows':int(response.near_indices.shape[1]),
                'maximum_temperature_bound':float(bounds.max())}
        else:
            retained_full = fraction == 0 or approximation['rank'] == min(matrix.shape)
            storage = tensor_storage_bytes((matrix,)) if retained_full else tensor_storage_bytes((approximation['left'],approximation['right']))
            details = {'rank':approximation['rank'],'maximum_temperature_bound':approximation['omission_bound'],
                'matrix_formation':matrix_formation,'actual_application':'learned K balanced SVD factors, no physical A',
                'retained_full_response':retained_full,
                'formation_complete_seconds':formation['seconds']+matrix_formation['seconds'],
                'physical_operator_factorizations':0}
        def apply(index, compressed, approximation=approximation, fraction=fraction):
            count('temperature_catalogue_applications')
            delta = increments[index]
            if not bool(feasible[index]):
                return native_temperature_vector(thermal,native,thermal.core.apply_increment(response,delta))
            if response.mode == 'group':
                return native_temperature_vector(thermal,native,thermal.core.apply_increment(response,delta,
                    compression=approximation if compressed else None))
            if not compressed or fraction == 0 or approximation['rank'] == min(matrix.shape):
                return (matrix @ delta[0,active])[None]
            return (approximation['left'] @ (approximation['right'] @ delta[0,active]))[None]
        errors = []
        with torch.no_grad():
            for index, reference in enumerate(exact):
                value = apply(index,True)
                if array_sink is not None:
                    array_sink[f'budget_{fraction:g}/allocation_{index}/temperature_increment'] = value.detach().cpu().numpy()
                error = value-reference
                allowed = details['maximum_temperature_bound']
                errors.append({'allocation':index,'inside_declared_balanced_box':bool(feasible[index]),
                    'fallback_full_response':not bool(feasible[index]),'model_response_error_max':float(error.abs().max()),
                    'model_response_error_rms':float(error.square().mean().sqrt()),
                    'bound_holds':bool(error.abs().max()<=allowed+epsilon*reference.abs().max().clamp_min(1)),
                    'physical_reference_scope':'no physical truth read; approximation error only'})
                if truth is not None:
                    mask = physical_mask[index]
                    model_physical = (reference.double()-truth[index])[mask]
                    compressed_physical = (value.double()-truth[index])[mask]
                    masked_approximation = error.double()[mask]
                    triangle_max = float(model_physical.abs().max()+masked_approximation.abs().max())
                    errors[-1].update(physical_reference_scope='saved generator finite differences joined only after ranks/masks are fixed',
                        physical_receiver_count=int(mask.sum()),physical_mask_scope='only physical error reporting; compression bounds/ranks retain the whole input catalogue',
                        masked_model_response_error_max=float(masked_approximation.abs().max()),
                        masked_model_response_error_rms=float(masked_approximation.square().mean().sqrt()),
                        full_model_physical_error_max=float(model_physical.abs().max()),
                        full_model_physical_error_rms=float(model_physical.square().mean().sqrt()),
                        compressed_physical_error_max=float(compressed_physical.abs().max()),
                        compressed_physical_error_rms=float(compressed_physical.square().mean().sqrt()),
                        physical_error_triangle_max_bound=triangle_max,
                        physical_error_triangle_holds=bool(compressed_physical.abs().max()<=triangle_max+epsilon*reference.abs().max().clamp_min(1)))
            # Baseline is always the full operator, irrespective of omitted modes.
            unchanged = native_temperature_vector(thermal,native,thermal.core.apply_forcing(response,heat))
            count('absolute_temperature_applications')
            if not torch.equal(absolute_baseline,unchanged): raise RuntimeError('Compression changed the absolute baseline.')
            for compressed in (False,True):
                warm = [apply(index,compressed) for index in range(len(exact))]
                synchronize(device); del warm
        samples = {'full_response':[],'compressed_response':[]}
        for repetition in range(5):
            order = (False,True) if repetition%2==0 else (True,False)
            for compressed in order:
                baseline = reset_memory(device); before = perf_counter()
                with torch.no_grad(): output = [apply(index,compressed) for index in range(len(exact))]
                synchronize(device)
                samples['compressed_response' if compressed else 'full_response'].append({'repetition':repetition,
                    'seconds':perf_counter()-before,**memory_result(device,baseline)})
                del output
        rows.append({'mode':response.mode,'budget_fraction':fraction,'absolute_temperature_budget':budget,
            'sealed_train_response_rms':float(train_response_rms),'radii_physical_heat':radii.cpu().tolist(),
            'physical_increments':increments.cpu().tolist(),'formation':formation,'approximation_storage_bytes':storage,
            'original_coefficients_storage_bytes':coefficient_bytes,'details':details,'errors':errors,'samples':samples,
            'catalogue':'native fluid temperature + surface temperature + material temperature; identical roles for both arms',
            'baseline_unchanged':True,'physical_factorizations':0,'physical_solves':0,
            'near_work_scope':'group keeps exact source near values/subtraction; direct SVD is learned full K control'})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--direct', type=Path, required=True)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--physical-gpu-ids', default='0,2')
    parser.add_argument('--b1-only', action='store_true')
    args = parser.parse_args(argv)
    benchmark({'incumbent': args.incumbent, 'R-direct': args.direct, 'R-group': args.group},
        args.output_dir, device=args.device, physical_gpu_ids=args.physical_gpu_ids.split(','), include_b8=not args.b1_only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
