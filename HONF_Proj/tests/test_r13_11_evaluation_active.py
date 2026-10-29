"""Shared engine evaluation reload preserves anchored access and EE laws."""
import copy
from pathlib import Path

import pytest
import torch

from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts, load_contextual_checkpoint
from honf_runtime.contextual_frontier_training import _build_task, _run_identity, load_contextual_frontier_config
from honf_runtime.unified_training import SamplingKey

CONFIG = Path(__file__).resolve().parents[1] / 'configs/research/thermal_r13_11x_s_active_e_anchor_train150_v1.json'


@pytest.mark.parametrize('arm,locator', [('R13-8-A-EAnchor', 'e_anchor'), ('R13-8-L-LocalECover', 'local_e_cover')])
def test_active_saved_payload_evaluation_loader_and_mismatched_laws(tmp_path, arm, locator):
    config, _ = load_contextual_frontier_config(CONFIG)
    if not Path(config['dataset']['data_path']).exists() or not Path(config['dataset']['manifest_path']).exists():
        pytest.skip('Thermal dataset and fixed manifest unavailable')
    torch.set_num_threads(1)
    config['member_routing']['arm'] = arm
    config['member_routing']['model'].update(architecture=arm, locator_mode=locator)
    model, provider = _build_task(config, arm, torch.device('cpu'))
    identity = _run_identity(config, provider)
    identity['provider_identity'] = provider.identity_payload()
    payload = {'experiment_identity': identity, 'arm': arm, 'model_state_dict': model.state_dict(), 'epoch': 0}
    _contextual_checkpoint_parts(payload)
    path = tmp_path / 'checkpoint.pt'; torch.save(payload, path)
    loaded, loaded_provider, loaded_payload = load_contextual_checkpoint(path, 'cpu')
    assert loaded.get_config() == model.get_config()
    assert loaded_provider.identity_payload() == provider.identity_payload()
    assert loaded_payload['epoch'] == 0
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    assert loaded.core.access_measure == 'active_domain'
    key = SamplingKey(seed=0, epoch=1, update_index=0, microbatch_index=0, stage='joint',
                      arm=provider.training_sampling_arm, sampling_version=provider.sampling_version,
                      dataset_id=provider.sampling_dataset_id)
    batch = provider.make_batch(('0006',), key)
    model.eval()
    with torch.no_grad():
        before = provider._prepare_for_probe(model, batch)
        after = loaded_provider._prepare_for_probe(loaded, batch)
        a = model.core.decode_paired(before.organization, before.query_batch)
        b = loaded.core.decode_paired(after.organization, after.query_batch)
        for original, reloaded in zip(a[:2], b[:2]):
            torch.testing.assert_close(original.joint_fields, reloaded.joint_fields, rtol=0, atol=0)
    # A historical omitted access/gate option has the established defaults.
    historical = copy.deepcopy(payload)
    p = historical['experiment_identity']['provider_identity']
    for section in (p['recipe']['member_routing']['model'], p['adapter_state']['core']):
        for field in ('access_measure', 'ee_gate_mode', 'ee_initial_tau'): section.pop(field, None)
    _contextual_checkpoint_parts(historical)
    for field, wrong in [('access_measure', 'inventory_unit_total'), ('ee_gate_mode', 'relative_sigmoid'),
                         ('ee_initial_tau', 9.0), ('ee_incidence_aggregation', 'pre_gate')]:
        bad = copy.deepcopy(payload)
        bad['experiment_identity']['provider_identity']['recipe']['member_routing']['model'][field] = wrong
        with pytest.raises(ValueError, match='identities must agree'):
            _contextual_checkpoint_parts(bad)
