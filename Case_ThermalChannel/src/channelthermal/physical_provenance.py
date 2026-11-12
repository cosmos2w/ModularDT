"""Verify Thermal endpoint controls and optional saved numerical provenance."""
from __future__ import annotations

import copy

from .interaction_evidence.numerical_policy import _canonical, _solver_config_module, validate_numerical_policy


def validate_thermal_endpoint_config(source_config, actual_config, *, centers,
                                     expected_actual_config=None, numerical_policy=None):
    """Require the complete prescribed endpoint, allowing only declared geometry edits.

    Domain, flow, thermal, local-module and layout controls are checked using
    solver defaults, including resolved viscosity. Save paths and execution
    metadata are not physical controls. When a numerical-policy record is
    supplied, its exact saved solver configuration and policy are also checked.
    """
    if not isinstance(source_config, dict) or not isinstance(actual_config, dict):
        raise TypeError('Thermal source and raw endpoint configurations are required.')
    expected = copy.deepcopy(source_config)
    expected['layout']['centers'] = copy.deepcopy(centers)
    if (expected_actual_config is None) != (numerical_policy is None):
        raise ValueError('Saved actual solver config and numerical policy must be supplied together.')
    if numerical_policy is not None:
        if actual_config != expected_actual_config:
            raise ValueError('Raw endpoint differs from saved actual solver provenance.')
        policy = numerical_policy.get('schema') if isinstance(numerical_policy, dict) else None
        observed = validate_numerical_policy(expected, actual_config, policy=policy, require_converged=True)
        if observed != numerical_policy:
            raise ValueError('Raw endpoint differs from saved numerical-policy provenance.')
        return
    schema = _solver_config_module()
    expected = _canonical(expected, schema, 'Expected endpoint')
    actual = _canonical(actual_config, schema, 'Actual endpoint')
    for config in (expected, actual):
        flow, domain = config['flow'], config['domain']
        if flow['nu'] is None:
            flow['nu'] = (float(flow['viscosity_scale']) * float(flow['u_in'])
                          * (2.0 * float(domain['module_radius']))
                          / max(float(flow['re']), 1.0e-12))
    for section in ('domain', 'flow', 'thermal', 'layout', 'local_module'):
        if expected[section] != actual[section]:
            changed = sorted(key for key in set(expected[section]) | set(actual[section])
                             if expected[section].get(key) != actual[section].get(key))
            raise ValueError(f'Raw endpoint {section} controls differ: {changed}')
