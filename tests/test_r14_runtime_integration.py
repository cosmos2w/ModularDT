"""Versioned R14 configuration, adapter, and checkpoint identity contracts."""
import copy
import json
from pathlib import Path

import pytest

from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
from honf_runtime.contextual_frontier_training import (
    R14_ARM, R14_CONFIG_SCHEMA, R14_CORE_SCHEMA, R14_SOURCE_KIND,
    _validate_loaded_contextual_config, load_contextual_frontier_config,
)
from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config = json.loads((ROOT / 'configs/research/thermal_r13_11s_e_anchor_train150_v1.json').read_text())
    config['dataset']['base_recipe_path'] = str(ROOT / 'src/config_core/forward/content_hypergraph/thermal_a2-h6_development.json')
    config['training']['first_segment_epochs'] = 100
    config['initialization']['source_kind'] = R14_SOURCE_KIND
    section = config['member_routing']
    section['schema'] = R14_CONFIG_SCHEMA
    section['arm'] = R14_ARM
    section['model'].update(schema=R14_CORE_SCHEMA, architecture=R14_ARM, locator_mode='local_e_cover', access_measure='active_domain')
    return config


@pytest.mark.parametrize('membership,normalization', [('fixed25_train150_dev22', 'fixed25_train150_stats'), ('full_train600', 'scratch_train600_stats')])
def test_r14_recipe_and_adapter_checkpoint_roundtrip(tmp_path, membership, normalization):
    config = _config()
    config['training']['membership'] = membership
    config['initialization']['normalization_mode'] = normalization
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    loaded, _ = load_contextual_frontier_config(path)
    _validate_loaded_contextual_config(loaded)
    stats = {'field_mean_by_channel': [0]*5, 'field_std_by_channel': [1]*5,
             'interface_targets_std': [1]*2, 'internal_temperature_std': [1]}
    adapter = ThermalDetailAnchoredMemberRoutingAdapter(stats, core_config=config['member_routing']['model'])
    state = adapter.state_payload()
    restored = ThermalDetailAnchoredMemberRoutingAdapter.from_state_payload(state, state_dict=adapter.state_dict())
    assert restored.get_config() == adapter.get_config()
    identity = {'workflow': 'interaction_reused_detail_anchored_member_routing_training_v1',
                'provider_identity': {'schema': 'thermal_r14_detail_anchored_member_routing_provider_v1',
                'architecture': R14_ARM, 'member_routing_schema': R14_CORE_SCHEMA,
                'adapter_state': state, 'recipe': config}}
    _, _, parsed = _contextual_checkpoint_parts({'experiment_identity': identity, 'arm': R14_ARM})
    assert parsed['arm'] == R14_ARM
    identity = copy.deepcopy(identity)
    identity['provider_identity']['schema'] = 'thermal_r13_8_e_anchored_member_routing_provider_v1'
    with pytest.raises(ValueError):
        _contextual_checkpoint_parts({'experiment_identity': identity, 'arm': R14_ARM})


def test_r14_real_native_batch_and_checkpoint_reload(tmp_path):
    import torch
    from honf_runtime.contextual_frontier_training import _build_task, _run_identity
    from honf_runtime.contextual_frontier_evaluation import load_contextual_checkpoint
    from honf_runtime.unified_training import SamplingKey

    config = _config()
    if not Path(config['dataset']['data_path']).exists() or not Path(config['dataset']['manifest_path']).exists():
        pytest.skip('Thermal data and fixed manifest unavailable')
    torch.set_num_threads(1)
    model, provider = _build_task(config, R14_ARM, torch.device('cpu'))
    identity = _run_identity(config, provider)
    identity['provider_identity'] = provider.identity_payload()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'experiment_identity': identity, 'arm': R14_ARM,
                'model_state_dict': model.state_dict(), 'epoch': 0}, path)
    restored, restored_provider, _ = load_contextual_checkpoint(path, 'cpu')
    key = SamplingKey(seed=0, epoch=1, update_index=0, microbatch_index=0, stage='joint',
                      arm=provider.training_sampling_arm, sampling_version=provider.sampling_version,
                      dataset_id=provider.sampling_dataset_id)
    batch = provider.make_batch((provider.training_case_ids[0],), key)
    model.eval()
    with torch.no_grad():
        before = provider._prepare_for_probe(model, batch)
        after = restored_provider._prepare_for_probe(restored, batch)
        original = model.core.decode_paired(before.organization, before.query_batch)
        reloaded = restored.core.decode_paired(after.organization, after.query_batch)
        for a, b in zip(original[:2], reloaded[:2]):
            torch.testing.assert_close(a.joint_fields, b.joint_fields, rtol=0, atol=0)
            assert torch.isfinite(a.joint_fields).all()
            assert sum(a.work_counts[f'query_anchor_coalesced_value_pairs_{role}'] for role in ('M', 'E')) > 0
        for role in ('M', 'E'):
            physical = original[2][f'query_relative_value_pairs_{role}']
            assert physical == reloaded[2][f'query_relative_value_pairs_{role}']
            assert physical >= max(output.work_counts[f'query_anchor_coalesced_value_pairs_{role}']
                                   for output in original[:2])
        assert sum(original[2][f'query_relative_value_pairs_{role}'] for role in ('M', 'E')) > 0
    model.train()
    predictions, auxiliary = provider.predict_native(
        model, provider.make_scene(batch.scene_inputs), batch.receivers,
        R14_ARM, 'joint', epoch=1,
    )
    terms = provider.loss_terms(predictions, batch.targets, 'joint', auxiliary)
    loss = sum(term.numerator * term.weight / term.denominator for term in terms.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(model.core.relative_value[role].weight.grad is not None for role in ('M', 'E'))
