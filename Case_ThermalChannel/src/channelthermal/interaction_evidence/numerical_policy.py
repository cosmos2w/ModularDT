"""Versioned numerical-limit checks for Thermal steady-reference records.

This policy is opt-in; it does not alter old corpus or solver behavior. The
caller supplies the exact prescribed endpoint config, including edited member
coordinates, and retains the complete actual solver config separately. A runtime
convergence flag is not a substitute for checking the saved convergence window.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys
from collections.abc import Mapping


POLICY_VERSION = 'thermal_time_cap_v1'
ALLOWED_MAX_SOLVE_TIMES = (200.0, 800.0, 1600.0)
_METADATA = {'runtime', 'generated_at', 'flow_diagnostics', 'physical_assumptions', 'interface_feature_names'}
_RUNTIME_FIELDS = {'warmup_time', 'solve_time', 'max_solve_time', 'total_time', 'num_steps',
    'save_stride', 'expected_saved_frames', 'nu', 'stop_on_convergence', 'convergence_window',
    'convergence_tol', 'convergence_rel_tol', 'thermal_substeps', 'max_steps', 'converged',
    'converged_step', 'converged_time', 'final_step', 'final_time', 'final_delta_inf',
    'final_delta_l2_rel'}
_SECTIONS = {
    'domain': 'DomainConfig', 'flow': 'FlowConfig', 'thermal': 'ThermalConfig',
    'layout': 'LayoutConfig', 'save': 'SaveConfig', 'local_module': 'LocalModuleConfig',
    'execution': 'ExecutionConfig',
}


@lru_cache(maxsize=1)
def _solver_config_module():
    # Load the exact existing solver schema/defaults; it ignores unknown keys,
    # so this module checks all keys explicitly before calling it.
    source = (Path(__file__).resolve().parents[4] / '1_Demo_ChannelThermal/src/_helpers_forward/channelthermal_common.py')
    name = '_honf_thermal_reference_policy_config'
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Thermal solver configuration schema is unavailable: {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _canonical(config, schema, label):
    if not isinstance(config, Mapping):
        raise ValueError(f'{label} config must be a mapping.')
    unknown = set(config) - set(_SECTIONS) - _METADATA
    if unknown:
        raise ValueError(f'{label} config has unknown top-level fields: {sorted(unknown)}')
    for section, class_name in _SECTIONS.items():
        value = config.get(section, {})
        if not isinstance(value, Mapping):
            raise ValueError(f'{label} {section} must be a mapping.')
        allowed = {item.name for item in fields(getattr(schema, class_name))}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f'{label} {section} has unknown controls: {sorted(extra)}')
    layout = config.get('layout', {})
    centers, powers = layout.get('centers'), layout.get('heat_powers')
    if (not isinstance(centers, (list, tuple)) or not centers
            or not isinstance(powers, (list, tuple)) or len(powers) != len(centers)
            or layout.get('num_modules', len(centers)) != len(centers)):
        raise ValueError(f'{label} layout requires explicit matching members, centers and heat.')
    return asdict(schema.config_from_dict(dict(config)))


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number.')
    return float(value)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def validate_numerical_policy(expected_config, actual_config, *, policy=POLICY_VERSION,
                              require_converged=True):
    """Validate an endpoint against the explicit steady-reference time policy.

    Only a 200/800/1600 physical-time cap is varied. The existing adapter may
    populate derived viscosity, use CPU execution metadata, and replace output
    path/name fields. Returned provenance is JSON-safe. ``require_converged``
    checks the recorded runtime flag and final metrics only; callers must also
    validate the full saved convergence window against ``frame_index.csv``.
    """
    if policy != POLICY_VERSION or type(require_converged) is not bool:
        raise ValueError('Unknown numerical policy or invalid convergence requirement.')
    schema = _solver_config_module()
    expected = _canonical(expected_config, schema, 'Expected')
    actual = _canonical(actual_config, schema, 'Actual')
    exp_thermal, act_thermal = expected['thermal'], actual['thermal']
    expected_cap = _finite_number(exp_thermal['max_solve_time'], 'Expected max_solve_time')
    cap = _finite_number(act_thermal['max_solve_time'], 'Actual max_solve_time')
    if expected_cap not in ALLOWED_MAX_SOLVE_TIMES or cap not in ALLOWED_MAX_SOLVE_TIMES:
        raise ValueError('Thermal max_solve_time must be an approved 200, 800 or 1600 cap.')
    if (exp_thermal['enabled'] is not True or act_thermal['enabled'] is not True
            or exp_thermal['stop_on_convergence'] is not True or act_thermal['stop_on_convergence'] is not True
            or exp_thermal['max_steps'] is not None or act_thermal['max_steps'] is not None):
        raise ValueError('Policy requires stop_on_convergence=true and thermal.max_steps=None.')
    if (expected['flow']['dt'] != .01 or expected['flow']['save_stride'] != 50
            or exp_thermal['convergence_window'] != 10 or exp_thermal['convergence_tol'] != 1e-4
            or exp_thermal['convergence_rel_tol'] != 1e-5
            or exp_thermal['require_heat_active_for_convergence'] is not True):
        raise ValueError('Expected convergence, timestep or save cadence is outside this policy.')

    # The adapter's NamedContext resolves an unspecified source viscosity.
    source_nu = expected['flow']['nu']
    if source_nu is None:
        resolved = (float(expected['flow']['viscosity_scale']) * float(expected['flow']['u_in'])
                    * (2.0 * float(expected['domain']['module_radius']))
                    / max(float(expected['flow']['re']), 1.0e-12))
        if actual['flow']['nu'] not in (None, resolved):
            raise ValueError('Actual viscosity differs from the source-derived viscosity.')
        expected['flow']['nu'] = actual['flow']['nu']

    # These adapter-owned fields identify where/how CPU output was written.
    if actual['execution'] not in (expected['execution'], {'device': 'cpu', 'gpu_id': 2}):
        raise ValueError('Actual execution metadata is neither the source nor the known CPU adapter.')
    actual['execution'] = expected['execution']
    for name in ('root_dir', 'case_id', 'tag'):
        actual['save'][name] = expected['save'][name]
    act_thermal['max_solve_time'] = expected_cap
    for section in _SECTIONS:
        if expected[section] != actual[section]:
            changed = sorted(k for k in set(expected[section]) | set(actual[section])
                             if expected[section].get(k) != actual[section].get(k))
            raise ValueError(f'Actual {section} controls differ: {changed}')

    runtime = actual_config.get('runtime')
    source_runtime = expected_config.get('runtime')
    if not isinstance(runtime, Mapping) or not isinstance(source_runtime, Mapping):
        raise ValueError('Expected and actual saved runtime metadata are required.')
    for label, metadata in (('Source', source_runtime), ('Actual', runtime)):
        unknown = set(metadata) - _RUNTIME_FIELDS
        if unknown:
            raise ValueError(f'{label} runtime has unknown fields: {sorted(unknown)}')
    dt = _finite_number(expected['flow']['dt'], 'flow.dt')
    step_limit = max(1, int(math.ceil(cap / max(dt, 1.0e-12))))
    if _positive_int(runtime.get('max_steps'), 'runtime.max_steps') != step_limit:
        raise ValueError('runtime.max_steps differs from the cap-derived step limit.')
    if _finite_number(runtime.get('max_solve_time'), 'runtime.max_solve_time') != cap:
        raise ValueError('runtime.max_solve_time differs from the actual cap.')
    final_step = _positive_int(runtime.get('final_step'), 'runtime.final_step')
    final_time = _finite_number(runtime.get('final_time'), 'runtime.final_time')
    if (final_step > step_limit or not math.isclose(final_time, final_step * dt, rel_tol=1e-12, abs_tol=1e-10)
            or (final_step % expected['flow']['save_stride'] and final_step != step_limit)):
        raise ValueError('Actual final step/time is inconsistent with the cap and save cadence.')
    source_substeps = _positive_int(source_runtime.get('thermal_substeps'), 'Source thermal_substeps')
    substeps = _positive_int(runtime.get('thermal_substeps'), 'Actual thermal_substeps')
    if substeps != source_substeps:
        raise ValueError('Actual thermal substeps differ from the prescribed source.')
    for key, expected_value in (('stop_on_convergence', True), ('save_stride', 50),
                                ('convergence_window', 10), ('convergence_tol', 1e-4),
                                ('convergence_rel_tol', 1e-5)):
        if runtime.get(key) != expected_value:
            raise ValueError(f'Actual runtime {key} differs from the declared numerical controls.')
    if runtime['stop_on_convergence'] is not True:
        raise ValueError('Actual runtime stop_on_convergence must be boolean true.')
    expected_nu = float(source_nu) if source_nu is not None else float(resolved)
    if _finite_number(runtime.get('nu'), 'runtime.nu') != expected_nu:
        raise ValueError('Runtime viscosity differs from the source-derived viscosity.')
    converged = runtime.get('converged')
    if type(converged) is not bool:
        raise ValueError('runtime.converged must be boolean.')
    if require_converged and not converged:
        raise ValueError('The actual reference did not converge.')
    if converged:
        delta_inf = _finite_number(runtime.get('final_delta_inf'), 'runtime.final_delta_inf')
        delta_rel = _finite_number(runtime.get('final_delta_l2_rel'), 'runtime.final_delta_l2_rel')
        if (runtime.get('converged_step') != final_step or runtime.get('converged_time') != final_time
                or final_time < float(exp_thermal['min_solve_time'])
                or not 0 <= delta_inf <= exp_thermal['convergence_tol']
                or not 0 <= delta_rel <= exp_thermal['convergence_rel_tol']):
            raise ValueError('Converged runtime has inconsistent final convergence metadata.')
    elif runtime.get('converged_step') is not None or runtime.get('converged_time') is not None:
        raise ValueError('Unconverged runtime contains a converged step/time.')
    return {'schema': POLICY_VERSION, 'source_max_solve_time': expected_cap,
            'actual_max_solve_time': cap, 'dt': dt, 'runtime_max_steps': step_limit,
            'runtime_final_step': final_step, 'runtime_final_time': final_time,
            'save_stride': expected['flow']['save_stride'], 'thermal_substeps': substeps,
            'runtime_converged': converged, 'convergence_window': exp_thermal['convergence_window'],
            'convergence_verification': 'runtime metadata only; saved full window must be checked separately',
            'resolved_nu': expected_nu}
