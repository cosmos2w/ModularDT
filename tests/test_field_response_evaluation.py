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
