import copy
import json
import math
from dataclasses import asdict

import pytest

from channelthermal.interaction_evidence.numerical_policy import (
    _solver_config_module, validate_numerical_policy,
)


def configs(cap=800, *, converged=True):
    schema = _solver_config_module()
    source = asdict(schema.config_from_dict({
        'flow': {'dt': .01, 'save_stride': 50, 're': 20., 'nu': None},
        'thermal': {'max_solve_time': 200., 'max_steps': None, 'stop_on_convergence': True,
                    'convergence_window': 10, 'convergence_tol': 1e-4,
                    'convergence_rel_tol': 1e-5},
        'layout': {'num_modules': 1, 'centers': [[2., 2.]], 'heat_powers': [1.]},
        'execution': {'device': 'gpu', 'gpu_id': 1},
    }))
    source['runtime'] = {'max_solve_time': 200., 'max_steps': 20000, 'nu': .045,
        'thermal_substeps': 1, 'stop_on_convergence': True, 'save_stride': 50,
        'convergence_window': 10, 'convergence_tol': 1e-4, 'convergence_rel_tol': 1e-5}
    source['runtime'].update(thermal_substeps=1, max_steps=20000, final_step=10000,
        final_time=100., converged=True, converged_step=10000, converged_time=100.,
        final_delta_inf=9e-5, final_delta_l2_rel=9e-6)
    actual = copy.deepcopy(source)
    actual['thermal']['max_solve_time'] = cap
    actual['flow']['nu'] = .045
    actual['execution'] = {'device': 'cpu', 'gpu_id': 2}
    actual['save'].update(root_dir='/different/output', case_id='attempt1', tag='reference_solver')
    actual['runtime'].update(max_solve_time=cap, max_steps=int(cap / .01),
        nu=actual['flow']['nu'])
    if not converged:
        actual['runtime'].update(converged=False, converged_step=None, converged_time=None,
                                 final_step=int(cap / .01), final_time=cap,
                                 final_delta_inf=2e-4, final_delta_l2_rel=2e-5)
    # These were absent in some historical source configs and are solver defaults.
    for key in ('solver_type', 'polar_radial_points', 'polar_theta_points', 'boundary_modes_max'):
        source['local_module'].pop(key)
    return source, actual


@pytest.mark.parametrize('cap', [200, 800, 1600])
def test_allowed_caps_runtime_and_existing_defaults(cap):
    source, actual = configs(cap)
    actual['runtime']['num_steps'] = 6000  # Nominal flow duration, not the thermal limit.
    before = copy.deepcopy((source, actual))
    result = validate_numerical_policy(source, actual)
    assert result['actual_max_solve_time'] == cap
    assert result['runtime_max_steps'] == int(cap / .01)
    assert result['runtime_final_step'] == 10000
    assert result['thermal_substeps'] == 1
    assert 'full window must be checked' in result['convergence_verification']
    assert math.isfinite(result['resolved_nu'])
    assert (source, actual) == before
    json.dumps(result, allow_nan=False)


def test_unconverged_requires_explicit_opt_in_and_preserves_false_status():
    source, actual = configs(800, converged=False)
    with pytest.raises(ValueError, match='did not converge'):
        validate_numerical_policy(source, actual)
    result = validate_numerical_policy(source, actual, require_converged=False)
    assert result['runtime_converged'] is False and result['runtime_final_step'] == 80000


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('thermal', 'max_solve_time', 400.), ('thermal', 'max_steps', 80000),
    ('thermal', 'stop_on_convergence', False), ('thermal', 'convergence_tol', 2e-4),
    ('thermal', 'convergence_rel_tol', 2e-5), ('thermal', 'convergence_window', 9),
    ('thermal', 'heat_start_time', 4.), ('thermal', 'fluid_alpha', .03),
    ('flow', 'dt', .02), ('flow', 'save_stride', 25),
    ('flow', 'apply_projection', True), ('flow', 'nu', .999),
    ('domain', 'module_radius', .46), ('local_module', 'solver_tolerance', 2e-6),
    ('layout', 'heat_powers', [2.]), ('layout', 'centers', [[2.1, 2.]]),
    ('save', 'save_pressure', False), ('execution', 'device', 'gpu'),
])
def test_reject_changed_physics_and_undeclared_controls(section, key, value):
    source, actual = configs()
    actual[section][key] = value
    with pytest.raises(ValueError):
        validate_numerical_policy(source, actual)


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('runtime', 'max_steps', 20000), ('runtime', 'max_solve_time', 200.),
    ('runtime', 'final_step', 80001), ('runtime', 'final_time', 101.),
    ('runtime', 'thermal_substeps', 2), ('runtime', 'nu', .123),
    ('runtime', 'convergence_window', 11), ('runtime', 'final_delta_inf', 2e-4),
    ('runtime', 'final_delta_l2_rel', float('nan')),
])
def test_reject_bad_actual_runtime(section, key, value):
    source, actual = configs()
    actual[section][key] = value
    with pytest.raises(ValueError):
        validate_numerical_policy(source, actual)


def test_unknown_fields_and_member_count_cannot_hide_behind_solver_defaults():
    source, actual = configs()
    actual['flow']['new_force'] = 1
    with pytest.raises(ValueError, match='unknown controls'):
        validate_numerical_policy(source, actual)
    del actual['flow']['new_force']
    actual['new_solver_control'] = True
    with pytest.raises(ValueError, match='unknown top-level'):
        validate_numerical_policy(source, actual)
    del actual['new_solver_control']
    actual['runtime']['unknown_solver_mode'] = 1
    with pytest.raises(ValueError, match='runtime has unknown fields'):
        validate_numerical_policy(source, actual)
    del actual['runtime']['unknown_solver_mode']
    actual['layout']['num_modules'] = 2
    with pytest.raises(ValueError, match='matching members'):
        validate_numerical_policy(source, actual)


def test_policy_version_is_explicit_and_source_precision_is_preserved():
    source, actual = configs()
    with pytest.raises(ValueError, match='Unknown numerical policy'):
        validate_numerical_policy(source, actual, policy='legacy')
    actual['layout']['centers'][0][0] += 1e-12
    with pytest.raises(ValueError, match='layout controls differ'):
        validate_numerical_policy(source, actual)
