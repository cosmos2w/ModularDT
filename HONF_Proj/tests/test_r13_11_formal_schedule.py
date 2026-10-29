import json
from pathlib import Path

import pytest

from honf_runtime.contextual_frontier_training import load_contextual_frontier_config


CONFIGS = Path(__file__).resolve().parents[1] / 'configs' / 'research'


@pytest.mark.parametrize('label', ['s', 'c', 'x_s_active'])
def test_scratch_formal_5000_schedule_round_trips(label, tmp_path):
    config = json.loads((CONFIGS / f'thermal_r13_11{label}_e_anchor_train150_v1.json').read_text())
    config['initialization']['normalization_mode'] = 'scratch_train600_stats'
    config['training'].update(membership='full_train600', local_horizon=5000,
                              optimizer_schedule_horizon=5000,
                              checkpoint_epochs=list(range(100, 5001, 100)))
    config['evaluation']['validation_split'] = 'canonical_VAL89'
    path = tmp_path / 'formal.json'
    path.write_text(json.dumps(config))
    loaded, _ = load_contextual_frontier_config(path)
    assert loaded['training']['checkpoint_epochs'] == list(range(100, 5001, 100))
    assert loaded['optimizer']['schedule'] == 'constant_config_lr'
    assert loaded['member_routing']['model'] == config['member_routing']['model']


@pytest.mark.parametrize('label,access', [('s', 'inventory_unit_total'), ('x_s_active', 'active_domain')])
def test_selected_formal_configs_use_scratch_5000(label, access):
    config, _ = load_contextual_frontier_config(CONFIGS / f'thermal_r13_11{label}_e_anchor_train600_formal_v1.json')
    training = config['training']
    assert (training['membership'], training['local_horizon'], training['first_segment_epochs'],
            training['optimizer_schedule_horizon']) == ('full_train600', 5000, 200, 5000)
    assert training['checkpoint_epochs'] == list(range(100, 5001, 100))
    assert config['initialization']['normalization_mode'] == 'scratch_train600_stats'
    assert config['initialization']['source_checkpoint'] is None
    assert config['evaluation']['validation_split'] == 'canonical_VAL89'
    assert config['member_routing']['model']['access_measure'] == access
