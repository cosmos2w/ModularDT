"""Versioned R14 configuration, adapter, and checkpoint identity contracts."""
import copy
import io
import json
from pathlib import Path

import pytest

from channelthermal.member_routing import (ThermalAnchoredMemberRoutingAdapter,
    ThermalDetailAnchoredMemberRoutingAdapter, ThermalInteractionReusedMemberRoutingAdapter)
from honf_forward_core.interaction_reused_member_routing import InteractionReusedMemberRoutingCore
from honf_runtime.contextual_frontier_training import (
    R14_ARM, R14_CONFIG_SCHEMA, R14_CORE_SCHEMA, R14_SOURCE_KIND,
    _validate_loaded_contextual_config, load_contextual_frontier_config,
)
from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts

ROOT = Path(__file__).resolve().parents[1]
STATS = {'field_mean_by_channel': [0]*5, 'field_std_by_channel': [1]*5,
         'interface_targets_std': [1]*2, 'internal_temperature_std': [1]}


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
@pytest.mark.parametrize('measure', ['inventory_unit_total', 'active_domain'])
def test_r14_recipe_and_adapter_checkpoint_roundtrip(tmp_path, membership, normalization, measure):
    config = _config()
    config['member_routing']['model']['access_measure'] = measure
    config['training']['membership'] = membership
    config['initialization']['normalization_mode'] = normalization
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    loaded, _ = load_contextual_frontier_config(path)
    _validate_loaded_contextual_config(loaded)
    adapter = ThermalDetailAnchoredMemberRoutingAdapter(STATS, core_config=config['member_routing']['model'])
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


def test_historical_member_adapter_reconstructs_its_own_schema():
    base = _config()['member_routing']['model']['base_model_config']
    core = InteractionReusedMemberRoutingCore(base, input_dataset='thermal')
    adapter = ThermalInteractionReusedMemberRoutingAdapter(STATS, core=core)
    restored = ThermalInteractionReusedMemberRoutingAdapter.from_state_payload(
        adapter.state_payload(), state_dict=adapter.state_dict())
    assert restored.get_config() == adapter.get_config()
    with pytest.raises(ValueError):
        ThermalAnchoredMemberRoutingAdapter(STATS, core_config=core.get_config())


@pytest.mark.parametrize('family,measure', [('relation_anchored', 'inventory_unit_total'),
                                            ('relation_anchored_active', 'active_domain')])
def test_public_training_recipe_and_cli_selection(family, measure):
    from honf_research import _parser
    from honf_forward_core.models import relation_anchored_config
    path = ROOT / 'configs/research' / f'thermal_{family}_train600.json'
    config, _ = load_contextual_frontier_config(path)
    _validate_loaded_contextual_config(config)
    model = config['member_routing']['model']
    generated = relation_anchored_config(family, model['base_model_config'], input_dataset='thermal', query_tile_size=512)
    assert model == generated
    assert model.get('access_measure', 'inventory_unit_total') == measure
    assert config['training']['membership'] == 'full_train600'
    assert (config['training']['first_segment_epochs'], config['training']['local_horizon']) == (100, 500)
    assert config['dataset']['base_recipe_path'] == str(ROOT / 'src/config_core/forward/content_hypergraph/thermal_a2-h6_development.json')
    assert config['member_routing']['health']['pause_on_collapse'] is False
    assert config['storage']['formal_forward_run_alias'] is True
    args = _parser().parse_args(['train-member-routing', '--model', family, '--device', 'cpu', '--output', '/tmp/run', '--dry-run'])
    assert args.model == family and args.config is None


def test_formal_forward_alias_uses_run_name_and_never_replaces_existing(tmp_path):
    from honf_runtime.contextual_frontier_training import _formal_forward_alias
    directory = tmp_path / 'project/Trained_Results/ThermalChannel/HONF_Forward_Runs'
    directory.mkdir(parents=True)
    root = tmp_path / 'payloads'
    root.mkdir()
    output = root / 'run'
    alias = _formal_forward_alias(output, root, project=tmp_path / 'project')
    assert alias == directory / 'run'
    output.mkdir()
    alias.symlink_to(output, target_is_directory=True)
    assert alias.resolve(strict=True) == output
    with pytest.raises(FileExistsError):
        _formal_forward_alias(output, root, project=tmp_path / 'project')
    with pytest.raises(ValueError, match='direct child'):
        _formal_forward_alias(root / 'nested/run', root, project=tmp_path / 'project')


@pytest.mark.parametrize('measure', ['inventory_unit_total', 'active_domain'])
def test_detail_synthetic_optimizer_checkpoint_resume(measure):
    import torch
    from dataclasses import fields, replace
    from tests.test_r14_query_relative_values import _model_scene_query
    from honf_forward_core.interaction_reused_member_routing import anchored_core_from_config

    torch.set_num_threads(1)
    torch.manual_seed(7)
    model, scene, query = _model_scene_query(measure)
    def detach(record):
        return replace(record, **{field.name: value.detach() for field in fields(record)
                                  if torch.is_tensor(value := getattr(record, field.name))})
    scene, query = detach(scene), detach(query)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.decode(model.prepare(scene), query).joint_fields
        loss = (prediction - torch.rand_like(prediction)).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    update(model, optimizer)
    stream = io.BytesIO()
    torch.save({'config': model.get_config(), 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state()}, stream)
    expected_loss = update(model, optimizer)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    restored = anchored_core_from_config(saved['config']).double().eval()
    restored.load_state_dict(saved['model'], strict=True)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    resumed_optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng'])
    torch.testing.assert_close(update(restored, resumed_optimizer), expected_loss, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for old, new in zip(optimizer.state.values(), resumed_optimizer.state.values(), strict=True):
        for key in old:
            torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)


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
