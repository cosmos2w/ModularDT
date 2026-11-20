"""Post-calibration response weight on the actual local-response adapter."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.training import thermal_local_response as local
from honf_runtime.local_response_contrasts import CommonObservations, build_uniform_contrast, neighborhood_field_losses


def _adapter(monkeypatch, objective_spec):
    import channelthermal.training.thermal_native_pair_task as native

    ids = [str(i) for i in range(150)]
    neighborhoods = [{'center_id': case_id, 'scene_ids': [case_id, ids[(i + 1) % 150]]}
                     for i, case_id in enumerate(ids)]
    definitions = {'case_ids': ids, 'data_path': 'fixture', 'neighborhoods': neighborhoods,
                   'chart_axes': [], 'input_scales': [], 'max_scenes': 2, 'reynolds_range': 1.}
    model = SimpleNamespace(field_std=torch.ones(5), core=SimpleNamespace(_ANCHOR_COUNT_KEYS=()))
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid')
    provider = SimpleNamespace(model=model, native_weights={name: 1. for name in names},
                               training_case_ids=ids, data_path='fixture',
                               identity_payload=lambda: {'dataset': 'fixture'},
                               training_history_columns=lambda: (), loss_metadata=lambda: {})
    monkeypatch.setattr(native, 'build_thermal_native_pair_task', lambda *a, **k: (model, provider))
    monkeypatch.setattr(local, 'load_thermal_input_definitions', lambda path: definitions)
    monkeypatch.setattr(local, 'build_thermal_input_definitions', lambda *a: definitions)
    monkeypatch.setattr(local, 'scheduled_thermal_neighborhoods', lambda *a: neighborhoods[:1])
    # Two source scenes share three 3D positions; one missing channel and a repeated
    # position retain their individual physical weights (ragged valid observations).
    refs = torch.tensor([[[1., 0.], [2., 0.], [1., 0.]],
                         [[3., 0.], [4., 0.], [3., 0.]]], dtype=torch.float64)
    obs = CommonObservations(refs, torch.tensor([[True, False], [True, True], [True, False]]),
                             torch.tensor([[.25, 0.], [.5, 1.], [.25, 0.]], dtype=torch.float64),
                             {'coordinates': np.array([[0., 1., 2.], [2., 3., 4.], [0., 1., 2.]])})
    prediction = torch.tensor([[[[2., 0.], [4., 1.], [2., 0.]],
                                [[6., 0.], [5., 2.], [6., 0.]]]], dtype=torch.float64,
                              requires_grad=True)

    def components(self, model, neighborhood, *, epoch):
        result = neighborhood_field_losses(prediction, obs, build_uniform_contrast(2), [1., 1.], [1., 1.])
        return {'absolute': result.absolute, 'response': result.response}, {}

    monkeypatch.setattr(local.ThermalLocalResponseObjective, 'components', components)
    spec = {'variant': 'uniform_response', 'input_definitions': 'fixture'}
    if objective_spec is not None:
        spec['objective'] = objective_spec
    _, result = local.build_thermal_local_response_task({'thermal_local_response': spec})
    # Feed already calibrated coefficients; calibration itself is independent of
    # this opt-in post-calibration multiplier and has separate coverage.
    result.validate_local_calibration = lambda record, **kwargs: record
    result.load_training_state_dict({'native': {}, 'local_response': {
        'calibration': {'binding': {'source_checkpoint': 'fixture'},
                        'coefficients': {'absolute': .25, 'response': .1}}}})
    return result, prediction


def _weighted_value_and_gradient(term, prediction):
    value = term.numerator * term.weight / term.denominator
    return value.detach(), torch.autograd.grad(value, prediction, retain_graph=True)[0]


def test_response_multiplier_changes_only_response_loss_and_gradient(monkeypatch):
    results = []
    for config in (None, {'response_weight': 1.}, {'response_weight': .5}, {'response_weight': 0.}):
        provider, prediction = _adapter(monkeypatch, config)
        terms, _ = next(provider.auxiliary_update_terms(None, epoch=50, update_index=0, update_count=1))
        results.append((provider, [_weighted_value_and_gradient(terms[name], prediction)
                                  for name in ('local_response/absolute', 'local_response/response')]))
    baseline = results[0][1]
    for index, multiplier in enumerate((1., 1., .5, 0.)):
        for actual, expected in zip(results[index][1][0], baseline[0]):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(results[index][1][1], baseline[1]):
            torch.testing.assert_close(actual, expected * multiplier)
    assert baseline[1][0] > 0 and baseline[1][1].abs().sum() > 0
    assert 'objective' not in results[0][0].identity_payload()['local_response']
    assert results[1][0].identity_payload()['local_response']['objective']['response_weight'] == 1.


@pytest.mark.parametrize('objective', [{'response_weight': -1.}, {'response_weight': float('nan')},
                                       {'response_weight': float('inf')}, {'unknown': 1.}, None])
def test_invalid_explicit_objective_rejected_before_build(objective):
    recipe = {'thermal_local_response': {'variant': 'uniform_response', 'objective': objective}}
    with pytest.raises((ValueError, TypeError)):
        local.build_thermal_local_response_task(recipe)
