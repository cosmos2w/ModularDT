"""Shared state/response operations preserve physical masks and gradients."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_forward_core.read_execution import read_execution, resolve_read_mode
from honf_runtime.state_response import (
    ResponseObjectiveConfig,
    combine_response_losses,
    compare_aligned_response,
    equal_family_mean,
    evaluate_candidate_pool,
    finite_pool_decision_losses,
    normalized_delta_mse,
    normalized_response_mse,
    summarize_response_actions,
    validate_aligned_receiver_ids,
    weighted_mean,
)


def test_response_config_balancing_ramp_and_disable():
    config = ResponseObjectiveConfig.from_mapping({})
    assert config.enabled is True
    assert (config.ramp(10), config.ramp(30), config.ramp(50)) == (0., .5, 1.)
    assert config.coefficients(8., {'endpoint': 2., 'response': 4.}) == {'endpoint': 1., 'response': .2}
    with pytest.raises(ValueError, match='TRAIN-measured'):
        config.coefficients()
    with pytest.raises(ValueError, match='zero'):
        config.coefficients(8., {'endpoint': 0., 'response': 4.})
    explicit = ResponseObjectiveConfig.from_mapping({
        'coefficient_mode': 'fixed', 'endpoint_weight': .3, 'response_weight': .4,
        'field_response_weight': .8, 'functional_weights': {'lift': .2}})
    assert explicit.coefficients() == {'endpoint': .3, 'response': .4}
    assert combine_response_losses(torch.tensor(2.), {'lift': torch.tensor(3.)}, explicit).item() == pytest.approx(2.2)
    assert ResponseObjectiveConfig(enabled=False).coefficients() == {'endpoint': 0., 'response': 0.}
    zero_endpoint = ResponseObjectiveConfig(endpoint_weight=0.)
    assert zero_endpoint.coefficients(8., {'endpoint': 0., 'response': 4.}) == {'endpoint': 0., 'response': .2}
    assert ResponseObjectiveConfig(endpoint_weight=0., response_weight=0.).coefficients() == {'endpoint': 0., 'response': 0.}
    with pytest.raises(ValueError):
        ResponseObjectiveConfig(warmup_epochs=-1)


def test_case_response_weights_keep_historical_float32_addition():
    field = torch.tensor(.0001)
    j = torch.tensor(.0001)
    p = torch.tensor(.0001)
    config = ResponseObjectiveConfig(field_response_weight=.8, functional_weights={'J': .1, 'P': .1})
    actual = combine_response_losses(field, {'J': j, 'P': p}, config)
    expected = .8 * field + .1 * j + .1 * p
    assert torch.equal(actual, expected)


def test_masked_response_is_equal_by_active_group_and_gradient_safe():
    base = torch.zeros((1, 3, 2), dtype=torch.float64)
    edit = torch.tensor([[[1., 3.], [4., 99.], [99., 99.]]], dtype=torch.float64, requires_grad=True)
    reference = torch.zeros_like(base)
    mask = torch.tensor([[[True, True], [True, False], [False, False]]])
    result = normalized_response_mse(base, edit, reference, reference, mask, 2., group_dim=-1)
    assert result.item() == pytest.approx(((1. / 4 + 9. / 4) / 2 + 16. / 4) / 2)
    result.backward()
    torch.testing.assert_close(edit.grad, torch.tensor([[[.125, .375], [1., 0.], [0., 0.]]], dtype=torch.float64))
    nan_reference = reference.clone()
    nan_reference[~mask] = float('nan')
    edit.grad = None
    normalized_response_mse(base, edit, nan_reference, nan_reference, mask, 2., group_dim=-1).backward()
    assert torch.isfinite(edit.grad).all()
    assert normalized_response_mse(base, edit.detach(), reference, reference, mask, 2.).item() == pytest.approx(26. / 12)
    assert normalized_delta_mse(torch.tensor(3.), 1., {'scale': 2.}).item() == 1.
    with pytest.raises(ValueError, match='mask'):
        normalized_response_mse(base, edit, reference, reference, torch.zeros_like(mask), 1.)
    with pytest.raises(ValueError, match='positive'):
        normalized_response_mse(base, edit, reference, reference, mask, 0.)


def test_case_ids_common_support_and_equal_family_metrics():
    base = {'phi': {'receiver_ids': ('a', 'b', 'c'), 'prediction': [1., 3., 100.],
                    'reference': [1., 2., -100.], 'mask': [True, True, False]}}
    edit = {'phi': {'receiver_ids': ('a', 'b', 'c'), 'prediction': [4., 8., 200.],
                    'reference': [3., 6., -200.], 'mask': [True, False, True]}}
    result = compare_aligned_response(base, edit)['phi']
    assert (result['baseline']['count'], result['edited']['count'], result['response']['count']) == (2, 2, 1)
    assert result['response']['rmse'] == 1.
    edit['phi']['receiver_ids'] = ('b', 'a', 'c')
    with pytest.raises(ValueError, match='receiver IDs'):
        compare_aligned_response(base, edit)
    with pytest.raises(ValueError, match='unique'):
        validate_aligned_receiver_ids(('a', 'a'), ('a', 'a'))
    values = [torch.tensor(1.), torch.tensor(3.), torch.tensor(10.)]
    assert equal_family_mean(values, ['a', 'a', 'b']).item() == 6.
    assert weighted_mean(values, [1., 1., 2.]).item() == 6.
    rows = [{'case_id': f, 'delta': {'lift': {'physical': physical, 'model': model}}}
            for f, physical, model in [('a', 2., 3.), ('a', -2., -3.), ('b', .1, 4.)]]
    summary = summarize_response_actions(rows, {'lift': 1.}, family_ids=['a', 'b', 'missing'])['lift']
    assert summary['all_action_equal_family_mae'] == pytest.approx(2.45)
    assert summary['families_without_actions'] == 1


def test_finite_pool_supports_case_defined_observables_actions_and_views():
    labels = np.array([[-1., -.4, .2], [0., .1, -.3], [.8, .5, .7]])
    baseline = torch.tensor([[2., 3., 4.], [-1., 2., 1.]], dtype=torch.float64)
    predictions = torch.cat((baseline[:, None, :], baseline[:, None, :] + torch.tensor(labels)[None]), dim=1)
    predictions = predictions.clone().requires_grad_()
    kwargs = {'observable_names': ('lift', 'drag', 'cost'),
              'sensitivities': {'lift': .02, 'drag': .02, 'cost': .02},
              'research_scales': {'lift': 1., 'drag': 1., 'cost': 1.},
              'response_scales': {'lift': 1., 'drag': 1., 'cost': 1.},
              'requests': [('lift', 'drag', -.25), ('cost', 'lift', 1.)]}
    losses, counts = finite_pool_decision_losses(predictions, labels, **kwargs)
    assert counts['requests'] == 4
    assert counts['ranking_pairs'] > 0
    assert all(float(loss) == pytest.approx(0., abs=1.e-28) for loss in losses.values())
    perturbed = predictions.detach().clone()
    perturbed[:, 1:, :] += torch.tensor([[1., .8, -.4], [.8, -.6, .7], [-1., -.8, .5]])
    perturbed.requires_grad_()
    both, _ = finite_pool_decision_losses(perturbed, labels, **kwargs)
    each = [finite_pool_decision_losses(perturbed[i:i + 1], labels, **kwargs)[0] for i in range(2)]
    for name in both:
        torch.testing.assert_close(both[name], (each[0][name] + each[1][name]) / 2)
    gradient, = torch.autograd.grad(sum(both.values()), perturbed)
    assert gradient.abs().sum() > 0
    torch.testing.assert_close(gradient.sum(dim=1), torch.zeros_like(gradient[:, 0]), atol=1.e-12, rtol=0)


def test_reader_scope_restores_nested_modes_and_exceptions():
    core = SimpleNamespace(_read_execution_override=None)
    assert resolve_read_mode(core) == 'restricted'
    with read_execution(core, 'full'):
        assert resolve_read_mode(core) == 'full'
        with pytest.raises(RuntimeError), read_execution(core, 'restricted'):
            assert resolve_read_mode(core) == 'restricted'
            raise RuntimeError('stop')
        assert resolve_read_mode(core) == 'full'
    assert core._read_execution_override is None
    with pytest.raises(ValueError), read_execution(core, 'invalid'):
        pass


def test_candidate_pool_verifies_selection_fallback_and_finite_pool_regret():
    predicted = {'baseline': {'cost': 0., 'load': 0.},
                 'unsafe': {'cost': -4., 'load': .1},
                 'good': {'cost': -1., 'load': .1},
                 'hidden_best': {'cost': -3., 'load': .4}}
    physical = {'baseline': {'cost': 0., 'load': 0.},
                'unsafe': {'cost': -2., 'load': .5},
                'good': {'cost': -1., 'load': .1},
                'hidden_best': {'cost': -3., 'load': .05}}
    options = {'objective': 'cost', 'constraint_caps': {'load': .2},
               'objective_scale': 2., 'sensitivity': {'cost': .1, 'load': .1}}
    result = evaluate_candidate_pool(predicted, physical, **options)
    assert result['model_selected_action'] == 'unsafe'
    assert result['raw_constraint_violation'] is True
    assert result['raw_constraint_excess']['load'] == pytest.approx(.3)
    assert result['accepted_action'] == 'baseline' and result['fallback_to_baseline'] is True
    assert result['raw_improvement'] == 2. and result['accepted_improvement'] == 0.
    assert result['physical_best_action'] == 'hidden_best'
    assert result['best_admitted_physically_feasible_action'] == 'good'
    assert result['physical_optimum_excluded_by_model'] is True
    assert result['false_feasible_actions'] == ['unsafe']
    assert result['false_rejected_actions'] == result['near_null_false_rejected_actions'] == ['hidden_best']
    assert result['regret'] == 3. and result['normalized_regret'] == 1.5
    unverified = evaluate_candidate_pool(predicted, None, **options)
    assert unverified['model_selected_action'] == 'unsafe'
    assert unverified['physical_verification'] == 'unavailable'
    assert unverified['accepted_action'] is None and unverified['regret'] is None
    with pytest.raises(ValueError, match='action IDs'):
        evaluate_candidate_pool(predicted, {'baseline': physical['baseline']}, **options)


def test_candidate_pool_never_accepts_an_unverified_or_infeasible_baseline():
    predicted = {'baseline': {'cost': 0., 'load': 0.},
                 'candidate': {'cost': -1., 'load': -.2}}
    physical = {'baseline': {'cost': 0., 'load': 0.},
                'candidate': {'cost': -.5, 'load': .1}}
    result = evaluate_candidate_pool(predicted, physical, objective='cost',
        constraint_caps={'load': -.1}, objective_scale=1., sensitivity={'cost': .1, 'load': .1})
    assert result['model_selected_action'] == 'candidate'
    assert result['raw_constraint_violation'] is True
    assert result['accepted_action'] is None
    assert result['physical_best_action'] is None and result['regret'] is None
