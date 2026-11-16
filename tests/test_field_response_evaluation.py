"""Primary nomination score must average physical pair RMSE before channels."""
import pytest
from tools.thermal_field_response_evaluate import summarize_physical_field_changes, CHANNELS


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
