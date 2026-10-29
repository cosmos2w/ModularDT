from pathlib import Path

import pytest

from honf_runtime.contextual_frontier_training import load_contextual_frontier_config


CONFIGS = Path(__file__).resolve().parents[1] / 'configs' / 'research'


@pytest.mark.parametrize('label,mode,access', [('s', 'relative_quadratic', 'inventory_unit_total'), ('x_s_active', 'relative_quadratic', 'active_domain')])
def test_r13_11_subset_uses_one_scientific_schedule(label, mode, access):
    config, _ = load_contextual_frontier_config(CONFIGS / f'thermal_r13_11{label}_e_anchor_train150_v1.json')
    training = config['training']
    assert (training['local_horizon'], training['first_segment_epochs'], training['optimizer_schedule_horizon']) == (1000, 200, 1000)
    assert training['checkpoint_epochs'] == list(range(100, 1001, 100))
    assert training['curve_every'] == 10
    assert config['optimizer']['schedule'] == 'constant_config_lr'
    model = config['member_routing']['model']
    assert model['access_measure'] == access
    assert model.get('ee_gate_mode') == mode
    if mode:
        assert model['ee_initial_tau'] == pytest.approx(1.4907824993133545)
        assert model['ee_incidence_aggregation'] == 'post_gate'
