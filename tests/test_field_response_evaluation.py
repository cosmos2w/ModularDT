"""Primary nomination score must average physical pair RMSE before channels."""
import pytest
from types import SimpleNamespace
from tools.thermal_field_response_evaluate import (summarize_physical_field_changes, CHANNELS,
                                                    bind_assessment9, ASSESSMENT9_CASES)


def test_assessment_binding_preserves_training_and_original_exposure():
    provider = SimpleNamespace(training_case_ids=('0001',), validation_case_ids=('0686',),
        case_split_by_id={case: 'test' for case in ASSESSMENT9_CASES}, _validation_batches_cache='old')
    bound = bind_assessment9(provider)
    assert bound is not provider
    assert bound.training_case_ids == provider.training_case_ids == ('0001',)
    assert provider.validation_case_ids == ('0686',)
    assert bound.validation_case_ids == ASSESSMENT9_CASES
    assert bound._validation_batches_cache is None
    provider.case_split_by_id['0281'] = 'train'
    with pytest.raises(ValueError, match='original test'):
        bind_assessment9(provider)


def test_physical_score_does_not_pool_pair_mse_or_renormalize_missing_pairs():
    rows = []
    for values in ((1., 8., None, 0., 0.), (3., None, None, 0., 0.)):
        rows.append({'common_field_residuals': [
            {'field': name, 'valid_positions': 7 if value is not None else 0,
             **({'change_rmse': value, 'change_mae': value / 2, 'common_rmse': 2 * value}
                if value is not None else {})}
            for name, value in zip(CHANNELS, values)]})
    measured = summarize_physical_field_changes(rows, [1, 4, 1, 1, 1], [1, 2, 3, 0, 0])
    assert measured['D_delta'] == pytest.approx(2.)
    assert measured['channels']['u']['mean_pair_change_rmse'] == 2.
    assert measured['channels']['u']['normalized_change_mse'] == 5.
    assert measured['channels']['v']['eligible_pairs'] == 1
    assert not measured['channels']['p']['available']
    assert summarize_physical_field_changes([], [1]*5, [1]*5)['D_delta'] is None


def test_cached_field_change_norms_and_near_null_separation():
    import numpy as np
    import torch
    from tools.thermal_field_response_evaluate import field_change_diagnostics
    left = torch.full((2, 5), 1., dtype=torch.float32)
    right = left.clone()
    right[:, 0] = 3.
    right[:, 1] = 1. + torch.finfo(torch.float32).eps
    pred_left = left.clone()
    pred_right = right.clone()
    pred_right[:, 0] = 5.
    pred_right[:, 1] = 4.
    # The second physical position is excluded for both fields.
    right[1, 0] = 100.
    cache = {'first': pred_left, 'second': pred_right}
    targets = {'first': left, 'second': right}
    mask = torch.tensor([True, False])
    coords = np.array([[0., 0.], [1., 0.]])
    result = field_change_diagnostics(cache, targets, {'first': mask, 'second': mask},
                                      {'first': coords, 'second': coords}, [('first', 'second')])
    fields = result['pairs'][0]['channels']
    assert fields['u']['norm_ratio'] == pytest.approx(2.)
    assert fields['u']['cosine'] == pytest.approx(1.)
    assert fields['u']['valid_positions'] == 1
    assert fields['v']['near_null_reference']
    assert fields['v']['norm_ratio'] is None
    assert fields['v']['cosine'] is None
    assert fields['v']['source_rounding_floor_rms'] > 0


def test_cached_triple_contrasts_use_common_support_and_fixed_anchor(monkeypatch):
    import numpy as np
    import torch
    from tools import thermal_field_response_evaluate as evaluation
    ids = evaluation.ASSESSMENT9_TRIPLE
    charts = {'0645': [0., 0.], '0646': [2., 0.], '0281': [0., 1.]}
    seen = []
    def fake_chart(anchor, candidate, *, reynolds_range):
        seen.append((anchor, candidate, reynolds_range))
        return np.array(charts[candidate]), np.zeros(2), [0], None
    monkeypatch.setattr(evaluation, 'thermal_chart', fake_chart)
    targets = {case: torch.zeros((2, 5), dtype=torch.float32) for case in ids}
    cache = {case: torch.full((2, 5), float(i), dtype=torch.float32) for i, case in enumerate(ids)}
    masks = {case: torch.tensor([True, case != '0281']) for case in ids}
    coords = {case: np.array([[0., 0.], [1., 0.]]) for case in ids}
    result = evaluation.triple_contrast_diagnostics(cache, targets, masks, coords,
        {case: case for case in ids}, 180., [1.] * 5, [1.] * 5)
    assert all(anchor == '0645' for anchor, _, _ in seen)
    assert result['channels']['u']['valid_positions'] == 1
    assert result['channels']['u']['uniform_normalized_contrast_mse'] == pytest.approx(2.)
    assert result['rank'] == 2
    assert result['scene_ids'] == list(ids)


@pytest.mark.parametrize('profile', ['local_multisecant_response', 'observation_response'])
def test_local_profile_uses_preserved_train_pairs_for_functional_scales(monkeypatch, profile):
    from tools import thermal_field_response_evaluate as evaluation
    legacy = SimpleNamespace(field_response_pairs=(('0001', '0002'),))
    assert evaluation.evaluation_functional_pairs(legacy) == legacy.field_response_pairs
    provider = SimpleNamespace(training_case_ids=('0001', '0002', '0003'), data_path='original.h5',
                               case_split_by_id={'0001': 'train', '0002': 'train', '0003': 'train'})
    recipe = {'training_profile': profile,
              'thermal_field_response': {'pairs': [['0001', '0002']]}}
    assert evaluation.evaluation_functional_pairs(provider, recipe) == (('0001', '0002'),)
    with pytest.raises(ValueError, match='TRAIN pairs'):
        evaluation.evaluation_functional_pairs(provider, {**recipe, 'thermal_field_response': {'pairs': [['0001', '9999']]}})
    calls = []
    def historical_selector(path, cases, *, case_split_by_id):
        calls.append((path, cases, case_split_by_id))
        return (), (('0002', '0003'),), ()
    monkeypatch.setattr(evaluation, 'select_thermal_field_pairs', historical_selector)
    assert evaluation.evaluation_functional_pairs(provider, {**recipe, 'thermal_field_response': {}}) == (('0002', '0003'),)
    assert calls == [(provider.data_path, provider.training_case_ids, provider.case_split_by_id)]


def test_local_profile_saved_edit_restore_uses_shared_dispatcher(monkeypatch):
    from tools import thermal_state_response_evaluate as restore_module
    recipe = {'training_profile': 'local_multisecant_response'}
    saved = {'experiment_identity': {'recipe': recipe, 'provider_identity': {'local_response': 'saved'}},
             'model_state_dict': {'weight': 1}, 'provider_training_state': {'local_response': 'state'}}
    calls = []
    class Model:
        def load_state_dict(self, state, strict): calls.append(('model', state, strict))
        def to(self, device): calls.append(('device', device)); return self
        def eval(self): calls.append(('eval',))
    class Provider:
        def identity_payload(self): return {'local_response': 'saved'}
        def load_training_state_dict(self, state): calls.append(('provider', state))
    def build(input_recipe, *, device, **options):
        calls.append(('dispatch', input_recipe, device, options))
        return Model(), Provider()
    monkeypatch.setattr(restore_module, 'resolve_checkpoint', lambda run, checkpoint: 'saved.pt')
    monkeypatch.setattr(restore_module, 'load_trusted_checkpoint', lambda path, map_location: saved)
    monkeypatch.setattr(restore_module, 'build', build)
    _, _, actual, path = restore_module.restore_model(recipe, 'run', 'latest', 'cpu')
    assert path == 'saved.pt' and actual is saved
    assert calls[0] == ('dispatch', recipe, 'cpu', {'resume_payload': saved})
    assert ('provider', saved['provider_training_state']) in calls
