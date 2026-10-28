"""Serialized organizer choices survive Thermal factory, checkpoint and native reads."""
import copy
from pathlib import Path

import pytest
import torch

from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts, load_contextual_checkpoint
from honf_runtime.contextual_frontier_training import (
    _build_task, _run_identity, _validate_loaded_contextual_config, load_contextual_frontier_config,
)

PROJECT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT / 'configs/research'


@pytest.mark.parametrize('arm', ['b', 'p', 'u'])
def test_recipe_and_resume_validation_accept_matched_new_and_historical_schedules(arm):
    config, _ = load_contextual_frontier_config(CONFIGS / f'thermal_r13_10{arm}_e_anchor_train150_v1.json')
    _validate_loaded_contextual_config(config)
    assert config['training']['local_horizon'] == 100
    assert config['training']['checkpoint_epochs'] == [100]
    assert config['training']['curve_every'] == 10
    broken = copy.deepcopy(config)
    broken['member_routing']['model']['ee_incidence_aggregation'] = 'unknown'
    with pytest.raises(ValueError):
        _validate_loaded_contextual_config(broken)
    historical, _ = load_contextual_frontier_config(CONFIGS / 'thermal_r13_9a_e_anchor_inherited_train150_v1.json')
    _validate_loaded_contextual_config(historical)
    formal = copy.deepcopy(config)
    formal['training'].update(membership='full_train600', local_horizon=500, first_segment_epochs=100,
                              optimizer_schedule_horizon=500, checkpoint_epochs=[100, 200, 300, 400, 500])
    formal['initialization']['normalization_mode'] = 'scratch_train600_stats'
    _validate_loaded_contextual_config(formal)


@pytest.mark.parametrize('arm,access,gate', [('p','mean_one','pre_gate'), ('u','inventory_unit_total','post_gate')])
def test_actual_factory_checkpoint_reload_and_native_reads(tmp_path, arm, access, gate):
    config, _ = load_contextual_frontier_config(CONFIGS / f'thermal_r13_10{arm}_e_anchor_train150_v1.json')
    if not Path(config['dataset']['data_path']).exists() or not Path(config['dataset']['manifest_path']).exists():
        pytest.skip('Thermal dataset and fixed manifest are not available on this host')
    torch.set_num_threads(1)
    model, provider = _build_task(config, 'R13-8-A-EAnchor', torch.device('cpu'))
    model.eval()
    assert model.core.access_measure == access
    assert model.core.applied_incidence.ee_incidence_aggregation == gate
    assert model.core.score_calibration == 'inherited'
    identity = _run_identity(config, provider)
    identity['provider_identity'] = provider.identity_payload()
    payload = {'experiment_identity': identity, 'arm': model.architecture,
               'model_state_dict': model.state_dict(), 'epoch': 0}
    checkpoint = tmp_path / 'native.pt'
    torch.save(payload, checkpoint)
    loaded, loaded_provider, _ = load_contextual_checkpoint(checkpoint, 'cpu')
    assert loaded.core.access_measure == access
    assert loaded.core.ee_incidence_aggregation == gate
    assert loaded.core.applied_incidence.ee_incidence_aggregation == gate
    batch = next(provider.native_evaluation_batches(split='validation', case_keys=('0277',), microbatch_size=1))
    loaded_batch = next(loaded_provider.native_evaluation_batches(split='validation', case_keys=('0277',), microbatch_size=1))
    with torch.no_grad():
        a = provider._prepare_for_probe(model, batch)
        b = loaded_provider._prepare_for_probe(loaded, loaded_batch)
        for original, replay in zip(model.core.decode_paired(a.organization, a.query_batch)[:2],
                                    loaded.core.decode_paired(b.organization, b.query_batch)[:2]):
            torch.testing.assert_close(original.joint_fields, replay.joint_fields, rtol=0, atol=0)
    # A saved recipe that disagrees with its adapter cannot silently change the equation.
    bad = copy.deepcopy(payload)
    bad_model = bad['experiment_identity']['provider_identity']['recipe']['member_routing']['model']
    bad_model['access_measure' if arm == 'p' else 'ee_incidence_aggregation'] = (
        'inventory_unit_total' if arm == 'p' else 'pre_gate')
    with pytest.raises(ValueError, match='identities must agree'):
        _contextual_checkpoint_parts(bad)
