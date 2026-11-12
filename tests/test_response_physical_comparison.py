import numpy as np
import pytest

from channelthermal.physical_response import compare_native_response, summarize_actions
from channelthermal.native_functionals import native_design_functionals


def test_response_compares_endpoints_and_changes_on_common_mask():
    base = {'temperature': {'prediction': np.array([1., 3., 100.]), 'reference': np.array([1., 2., -100.]), 'mask': [True, True, False]}}
    edit = {'temperature': {'prediction': np.array([4., 8., 200.]), 'reference': np.array([3., 6., -200.]), 'mask': [True, False, True]}}
    result = compare_native_response(base, edit)['temperature']
    assert result['response']['count'] == 1
    assert result['baseline']['count'] == 2
    assert result['edited']['count'] == 2
    assert result['baseline']['rmse'] == pytest.approx(np.sqrt(.5))
    assert result['edited']['rmse'] == pytest.approx(np.sqrt((1 + 400**2) / 2))
    assert result['response']['rmse'] == 1
    assert result['response']['reference_rms'] == 2
    assert result['response']['rmse_over_reference_rms'] == .5
    with pytest.raises(ValueError):
        compare_native_response(base, {'other': edit['temperature']})


def test_zero_physical_response_keeps_absolute_error_and_no_fabricated_ratio():
    base = {'attached': {'prediction': [3., 99.], 'reference': [2., 99.], 'mask': [True, False]}}
    edit = {'attached': {'prediction': [4., -99.], 'reference': [2., -99.], 'mask': [True, True]}}
    result = compare_native_response(base, edit)['attached']
    assert result['edited']['count'] == 2
    assert result['response']['count'] == 1
    assert result['response']['rmse'] == 1
    assert result['response']['reference_rms'] == 0
    assert result['response']['rmse_over_reference_rms'] is None


def test_equal_family_response_coverage_and_resolution():
    def row(case, physical, model):
        return {'case_id': case, 'delta': {k: {'physical': physical, 'model': model} for k in ('J', 'P')}}
    result = summarize_actions([row('one', 2., 3.), row('one', -2., -3.), row('two', .1, 4.)],
                               {'J': 1., 'P': 1.}, family_ids=['one', 'two', 'missing'])['J']
    assert result['all_action_equal_family_mae'] == pytest.approx(2.45)
    assert result['resolved_family_count'] == 1
    assert result['families_without_actions'] == 1
    assert result['resolved_sign_correct'] == 2
    with pytest.raises(ValueError, match='finite'):
        summarize_actions([row('one', float('nan'), 1.)], {'J': 1., 'P': 1.})
    with pytest.raises(ValueError, match='finite'):
        summarize_actions([row('one', 1., float('inf'))], {'J': 1., 'P': 1.})


def test_native_functionals_use_masked_physical_receivers():
    import torch
    material = torch.tensor([1., 2., 999.])
    pressure = torch.tensor([4., 999., 1.])
    objective, drop = native_design_functionals(material, [True, True, False], pressure,
        [True, False, True], torch.tensor([0., .5, 1.]), 1., .1)
    assert 1.9 < objective.item() < 2.
    assert drop.item() == pytest.approx(3.)
