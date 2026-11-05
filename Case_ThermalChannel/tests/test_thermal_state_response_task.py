from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.training.thermal_state_response_task import (
    ACTIONS, FIELD_ROLES, ThermalResponseCorpus, align_role_values,
    fit_response_scales, response_loss_terms,
)


def role(ids, coordinates, values, mask, kind='interface_material_angle', members=None):
    return SimpleNamespace(role='interface', channel_names=('T',), coordinate_kind=kind,
                           query_ids=ids, query_features=np.asarray(coordinates),
                           values=np.asarray(values, dtype=float).reshape(-1, 1),
                           valid_mask=np.asarray(mask, dtype=bool).reshape(-1, 1),
                           receiver_module_ids=members)


def test_alignment_uses_original_ids_and_intersection_for_moving_receivers():
    b = role(('a', 'b'), [[0, 0], [1, 0]], [1, 2], [True, False], members=('m0', 'm1'))
    e = role(('b', 'a'), [[2, 0], [1, 0]], [5, 4], [True, True], members=('m1', 'm0'))
    aligned = align_role_values(b, e)
    np.testing.assert_array_equal(aligned['edited'].ravel(), [4, 5])
    np.testing.assert_array_equal(aligned['common_mask'].ravel(), [True, False])
    e.receiver_module_ids = ('m0', 'm1')
    with pytest.raises(ValueError, match='member IDs'):
        align_role_values(b, e)


def test_eulerian_identity_requires_fixed_coordinates():
    b = role(('a',), [[0, 0]], [1], [True], kind='eulerian')
    e = role(('a',), [[1, 0]], [1], [True], kind='eulerian')
    with pytest.raises(ValueError, match='coordinates'):
        align_role_values(b, e)


def test_scales_equal_family_action_with_train_floors():
    def actions(value):
        item = {n: (np.array([value, 1000]), np.array([True, False])) for n in FIELD_ROLES}
        item.update(J=value, P=value)
        return {action: item for action in ACTIONS}
    fitted = fit_response_scales({'one': actions(1), 'two': actions(3)},
        {n: 500 for n in FIELD_ROLES}, {n: 0 for n in FIELD_ROLES}, {'J': .1, 'P': 4})
    assert fitted['flow/u']['fit_rms'] == pytest.approx(np.sqrt(5))
    assert fitted['flow/u']['scale'] == 5
    assert fitted['J']['scale'] == pytest.approx(np.sqrt(5))
    assert fitted['P']['scale'] == 4


def test_loss_weights_keep_gradients():
    x = torch.tensor(2., requires_grad=True)
    terms = response_loss_terms(x, 2*x, {'J': 3*x, 'P': 4*x}, multiplier=2)
    total = sum(t.numerator * t.weight / t.denominator for t in terms.values())
    total.backward()
    assert x.grad.item() == pytest.approx(2 * (.25 + .08*2 + .01*3 + .01*4))


def test_legacy_and_explicit_default_weights_are_exact():
    def calculate(coefficient):
        x = torch.tensor(1.2345, dtype=torch.float64, requires_grad=True)
        kwargs = {} if coefficient is None else {'finite_response_coefficient': coefficient}
        terms = response_loss_terms(x.square(), 2*x.square(), {'J': 3*x.square(), 'P': 4*x.square()},
                                    multiplier=13 / 16, **kwargs)
        total = sum(term.numerator * term.weight for term in terms.values())
        return terms, total.detach(), torch.autograd.grad(total, x)[0]

    implicit, total, gradient = calculate(None)
    explicit, explicit_total, explicit_gradient = calculate(.1)
    assert [term.weight for term in implicit.values()] == [value * (13 / 16) for value in (.25, .08, .01, .01)]
    assert [term.weight for term in explicit.values()] == [term.weight for term in implicit.values()]
    assert torch.equal(total, explicit_total)
    assert torch.equal(gradient, explicit_gradient)
    x = torch.tensor(1.2345, dtype=torch.float64, requires_grad=True)
    legacy = sum(value * (weight * (13 / 16)) for value, weight in
                 zip((x.square(), 2*x.square(), 3*x.square(), 4*x.square()), (.25, .08, .01, .01)))
    assert torch.equal(total, legacy.detach())
    assert torch.equal(gradient, torch.autograd.grad(legacy, x)[0])


def test_zero_delta_is_detached_but_endpoint_gradient_remains_live():
    endpoint = torch.tensor(3., requires_grad=True)
    delta = torch.tensor(2., requires_grad=True)
    terms = response_loss_terms(endpoint.square(), delta.square(), {'J': delta**3, 'P': delta**4},
                                finite_response_coefficient=0)
    assert terms['response/endpoint'].numerator.requires_grad
    assert all(not term.numerator.requires_grad and term.weight == 0 for term in list(terms.values())[1:])
    total = sum(term.numerator * term.weight for term in terms.values())
    total.backward()
    assert endpoint.grad.item() == pytest.approx(.25 * 2 * 3)
    assert delta.grad is None


@pytest.mark.parametrize('bad', [True, -0.1, float('nan'), float('inf'), '0.1'])
def test_invalid_finite_response_coefficient_rejected(bad):
    value = torch.tensor(1.)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        response_loss_terms(value, value, {'J': value, 'P': value}, finite_response_coefficient=bad)


def test_paired_native_reads_keep_endpoint_gradients_and_detach_only_diagnostics():
    from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
    from channelthermal.training.thermal_state_response_task import predict_response_pair

    def make_batch(factor):
        targets = SimpleNamespace(point_weights=torch.ones(1, 2),
            field_targets=torch.zeros(1, 2, 5), interface_target=torch.zeros(1, 1, 2, 2),
            interface_valid_mask=torch.ones(1, 1, 2), material_targets=torch.zeros(1, 1, 2))
        return SimpleNamespace(scene_inputs=factor, targets=targets,
                               receivers=SimpleNamespace(fluid_xy=torch.tensor([[[0.], [1.]]])))

    class Provider:
        native_weights = dict.fromkeys(FIELD_ROLES, 1.)

        def __init__(self, parameter):
            self.parameter = parameter
            self.visits = []

        def predict_native(self, model, scene, receivers, architecture, mode, *, epoch):
            self.visits.append(scene)
            def branch(factor):
                x = self.parameter * scene * factor
                main = {'pred_field': x * torch.arange(1., 11.).reshape(1, 2, 5),
                        'pred_interface': x * torch.arange(1., 5.).reshape(1, 1, 2, 2),
                        'pred_internal_temperature': x * torch.tensor([[[[1.], [2.]]]])}
                stencil = lambda: SimpleNamespace(valid=torch.ones(2, dtype=torch.bool))
                prep = SimpleNamespace(source_present=torch.ones(1, 1),
                    stencils={name: stencil() for name in ('fluid', 'surface', 'outside', 'material')})
                return SimpleNamespace(native_main=main, native_prepared=prep)
            return ThermalJointPairPredictions(branch(1), branch(.7), {}), {'work': {}}

        def _role_losses(self, branch, targets):
            return {'flow/u': branch.native_main['pred_field'].square().mean()}

    def run(response_grad):
        parameter = torch.tensor(1.2, requires_grad=True)
        provider = Provider(parameter)
        model = SimpleNamespace(architecture='synthetic', core=SimpleNamespace(reader_capability='joint95'))
        pair = {'baseline': make_batch(1.), 'edited': make_batch(1.5),
                'raw_baseline': {'config': {'domain': {'lx': 1.}}},
                'raw_edited': {'config': {'domain': {'lx': 1.}}}}
        result = predict_response_pair(provider, model, pair, dict.fromkeys((*FIELD_ROLES, 'J', 'P'), 1.),
                                       {'J': 0., 'P': 0.}, 1., response_grad=response_grad)
        endpoint_grad = torch.autograd.grad(result['endpoint'], parameter, retain_graph=True)[0]
        return result, endpoint_grad, provider.visits

    zero, zero_grad, zero_visits = run(False)
    weighted, weighted_grad, weighted_visits = run(True)
    assert zero_visits == weighted_visits == [1., 1.5]
    assert torch.equal(zero['endpoint'], weighted['endpoint'])
    assert torch.equal(zero_grad, weighted_grad) and zero_grad.item() > 0
    for name in ('field_delta', 'J_delta', 'P_delta'):
        assert torch.equal(zero[name], weighted[name])
        assert not zero[name].requires_grad and zero[name].grad_fn is None
        assert weighted[name].requires_grad
    assert len(zero['observations']) == len(weighted['observations']) == 2


@pytest.mark.parametrize('update_count', [1, 4, 13])
@pytest.mark.parametrize('coefficient', [0., .1])
def test_actual_family_action_exposure_and_epoch_normalization(monkeypatch, update_count, coefficient):
    from channelthermal.training import thermal_state_response_task as task

    corpus = object.__new__(ThermalResponseCorpus)
    corpus.cases = {str(i): {'role': 'fit', 'feasible': True} for i in range(48)}
    provider = object.__new__(task.ThermalStateResponseTask)
    provider.response_corpus, provider.seed = corpus, 0
    provider.response_spec, provider.response_scales = {'tau': 1.}, {}
    provider.response_labels = {(family, action): {} for family in corpus.family_ids('fit') for action in ACTIONS}
    provider.response_finite_response_coefficient = coefficient
    provider._response_epoch, provider._response_epoch_families = None, []
    visits, values = [], []
    parameter = torch.tensor(1., requires_grad=True)

    def fake_pair(provider, corpus, family, action, *, role):
        assert role == 'fit'
        return family, action

    def fake_predict(provider, model, pair, scales, label, tau, *, response_grad):
        visits.append(pair)
        value = float(len(visits))
        values.append(value)
        assert response_grad == (coefficient != 0)
        delta = parameter * value
        return {'endpoint': parameter * value, 'field_delta': delta, 'J_delta': 2 * delta,
                'P_delta': 3 * delta, 'work': {}}

    monkeypatch.setattr(task, 'response_pair', fake_pair)
    monkeypatch.setattr(task, 'predict_response_pair', fake_predict)
    objective = 0
    diagnostics = dict.fromkeys(provider.auxiliary_loss_names[1:], 0.)
    work_total = {'response_actions': 0, 'response_families': 0}
    assert provider.response_epoch_family_ids(1) == ()
    for update_index in range(update_count):
        for terms, work in provider.auxiliary_update_terms(None, epoch=1, update_index=update_index,
                                                           update_count=update_count):
            objective = objective + sum(term.numerator * term.weight for term in terms.values()) / update_count
            for key, value in provider.auxiliary_diagnostic_values(terms, update_count=update_count).items():
                diagnostics[key] += value / update_count
            for key in work_total:
                work_total[key] += work[key]
    families = corpus.fit_families_for_epoch(1, 0)
    assert visits == [(family, action) for family in families for action in ACTIONS]
    assert provider.response_epoch_family_ids(1) == families
    assert provider.response_epoch_family_ids(2) == ()
    assert work_total == {'response_actions': 16, 'response_families': 4}
    mean = sum(values) / 16
    assert diagnostics == pytest.approx({'response/field_delta': mean, 'response/J_delta': 2 * mean,
                                         'response/P_delta': 3 * mean})
    expected_weight = .25 + coefficient * (.8 + .1 * 2 + .1 * 3)
    assert torch.autograd.grad(objective, parameter)[0].item() == pytest.approx(mean * expected_weight)


def test_four_families_cover_every_fit_family_over_twelve_epochs():
    corpus = object.__new__(ThermalResponseCorpus)
    corpus.cases = {str(i): {'role': 'fit', 'feasible': True} for i in range(48)}
    seen = [x for epoch in range(1, 13) for x in corpus.fit_families_for_epoch(epoch, 0)]
    assert len(seen) == len(set(seen)) == 48
    assert corpus.fit_families_for_epoch(1, 0) == corpus.fit_families_for_epoch(1, 0)


def test_scale_fitting_rejects_holdout_families():
    corpus = object.__new__(ThermalResponseCorpus)
    corpus.cases = {'fit': {'role': 'fit', 'feasible': True},
                    'heldout': {'role': 'response_holdout', 'feasible': True}}
    with pytest.raises(ValueError, match='original FIT'):
        corpus.fit_scales({'heldout': {}}, {}, {}, {})


def test_saved_functional_labels_require_role_membership(tmp_path):
    import json
    from channelthermal.training.thermal_state_response_task import load_saved_functional_responses
    corpus = object.__new__(ThermalResponseCorpus)
    corpus.cases = {'fit': {'role': 'fit', 'feasible': True},
                    'heldout': {'role': 'response_holdout', 'feasible': True}}
    rows = [{'case_id': 'fit', 'action': action,
             'delta': {'J': {'physical': 1}, 'P': {'physical': -2}}} for action in ACTIONS]
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'models': {'source': {'full': {'actions': rows}}}}))
    labels = load_saved_functional_responses(path, corpus, role='fit')
    assert labels[('fit', 'single_plus')] == {'J': 1., 'P': -2.}
    with pytest.raises(ValueError, match='membership'):
        load_saved_functional_responses(path, corpus, role='response_holdout')


def test_alignment_intersects_masks_per_channel():
    b = role(('a',), [[0, 0]], [1], [True])
    e = role(('a',), [[1, 0]], [2], [True])
    b.channel_names = e.channel_names = ('T', 'q')
    b.values, e.values = np.array([[1., 2.]]), np.array([[3., 4.]])
    b.valid_mask, e.valid_mask = np.array([[True, False]]), np.array([[True, True]])
    np.testing.assert_array_equal(align_role_values(b, e)['common_mask'], [[True, False]])


def test_saved_scales_restore_exactly_without_refitting(monkeypatch):
    import json
    from channelthermal.training import thermal_state_response_task as task
    scales = {name: {'scale': 0.45560681823737065, 'fit_rms': 0.45560681823737065,
                     'floor': 0.01} for name in (*FIELD_ROLES, 'J', 'P')}
    saved = json.loads(json.dumps(scales))
    monkeypatch.setattr(task, 'ThermalResponseCorpus', lambda *args: object())
    monkeypatch.setattr(task, 'load_saved_functional_responses', lambda *args, **kwargs: {})
    def unexpected_fit(*args, **kwargs):
        raise AssertionError('Checkpoint restoration must not refit scales')
    monkeypatch.setattr(task, '_fit_native_response_scales', unexpected_fit)
    provider = object.__new__(task.ThermalStateResponseTask)
    provider.configure_state_response({'recipe_path': '/unused/recipe.json', 'ledger_path': '/unused/ledger'},
                                      response_scales=saved)
    assert provider.response_scales == saved
    assert provider.response_scales is not saved
    saved['flow/u']['scale'] = 99
    assert provider.response_scales['flow/u']['scale'] == 0.45560681823737065


@pytest.mark.parametrize('coefficient', [None, 0., .1])
def test_saved_identity_restores_coefficient_weights_and_scales(monkeypatch, coefficient):
    import json
    from channelthermal.training import thermal_state_response_task as task

    scales = {name: {'scale': 0.45560681823737065, 'fit_rms': 0.45560681823737065,
                     'floor': 0.01} for name in (*FIELD_ROLES, 'J', 'P')}
    corpus = SimpleNamespace(family_ids=lambda role: ('01', '02'))
    monkeypatch.setattr(task, 'ThermalResponseCorpus', lambda *args: corpus)
    monkeypatch.setattr(task, 'load_saved_functional_responses', lambda *args, **kwargs: {})
    monkeypatch.setattr(task.ThermalR138AnchoredMemberRoutingTask, 'identity_payload',
                        lambda self: {'objective': {}, 'loss_lineage': {}})
    spec = {'recipe_path': '/unused/recipe.json', 'ledger_path': '/unused/ledger'}
    if coefficient is not None:
        spec['finite_response_coefficient'] = coefficient

    def provider(saved_scales):
        result = object.__new__(task.ThermalStateResponseTask)
        result.model = SimpleNamespace(core=SimpleNamespace(reader_capability='joint95'))
        result.configure_state_response(spec, response_scales=saved_scales)
        return result

    original = provider(scales)
    saved = json.loads(json.dumps(original.identity_payload()))
    restored = provider(saved['response_scales'])
    assert restored.identity_payload() == saved
    assert restored.response_finite_response_coefficient == (.1 if coefficient is None else coefficient)
    assert restored.response_scales == scales
    if coefficient is None:
        assert 'finite_response_coefficient' not in saved['thermal_state_response']
        assert 'response_objective_weights' not in saved
    else:
        assert saved['thermal_state_response']['finite_response_coefficient'] == coefficient
        assert saved['response_objective_weights'] == task._response_objective_weights(coefficient)
    metadata = restored.loss_metadata()
    weights = task._response_objective_weights(restored.response_finite_response_coefficient)
    for role in ('endpoint', 'field_delta', 'J_delta', 'P_delta'):
        assert float(metadata[f'response/{role}']['weight']) == weights[role]
    if coefficient == 0:
        assert all('diagnostic only' in metadata[f'response/{name}']['label']
                   for name in ('field_delta', 'J_delta', 'P_delta'))


@pytest.mark.parametrize('bad', [
    lambda s: s.pop('J'),
    lambda s: s['P'].update(scale=float('nan')),
    lambda s: s['P'].update(scale=0),
    lambda s: s['P'].update(scale=2),
    lambda s: s['P'].update(fit_rms=-1),
    lambda s: s['P'].update(extra=1),
])
def test_invalid_saved_scales_rejected(bad):
    from channelthermal.training.thermal_state_response_task import _restore_response_scales
    scales = {name: {'scale': 1.0, 'fit_rms': 1.0, 'floor': 0.01}
              for name in (*FIELD_ROLES, 'J', 'P')}
    bad(scales)
    with pytest.raises(ValueError):
        _restore_response_scales(scales)
