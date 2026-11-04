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
