"""Observation-response native retention is explicit and resume-identifiable."""
import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

import honf_forward_core.interaction_reused_member_routing.anchored as anchored
from honf_runtime.coupled_initialization import _same, apply_epoch_zero, common_contract, prepare_epoch_zero
from honf_runtime.observation_response_task import build_observation_response_task
from honf_runtime.unified_training import _resume_identity_amendment, _sampling_dataset_id, capture_rng_state
from tests.test_coupled_native_integration import key, recipe
from tools import relation_anchored_train as cli


def _config(fraction=None):
    result = recipe('sampled_reader_response')
    result['training_profile'] = 'observation_response'
    if fraction is not None:
        result['execution_config'] = {'native_retention_fraction': fraction}
    return result


@pytest.mark.parametrize('fraction', [None, 0, 0.0])
def test_default_keeps_existing_provider_identity_and_native_sampling(fraction):
    config = _config(fraction)
    _, baseline = build_observation_response_task(copy.deepcopy(config), device='cpu')
    _, provider = cli.build(copy.deepcopy(config), device='cpu')
    assert provider.identity_payload() == baseline.identity_payload()
    assert 'native_retention_execution' not in provider.identity_payload()
    assert _sampling_dataset_id(provider.identity_payload()) == provider.sampling_dataset_id


def test_nonzero_policy_is_identified_and_rejects_legacy_resume():
    config = _config(.25)
    _, legacy = build_observation_response_task(copy.deepcopy(config), device='cpu')
    _, provider = cli.build(copy.deepcopy(config), device='cpu')
    previous = legacy.identity_payload()
    current = provider.identity_payload()
    assert current == {**previous, 'native_retention_execution': {
        'schema': 'anchored_native_tile_retention_v1', 'fraction': .25,
        'scope': 'native_prediction'}}
    assert _sampling_dataset_id(previous) == _sampling_dataset_id(current) == provider.sampling_dataset_id
    saved = {'workflow': 'relation_anchored_case_training_v1', 'recipe': config,
             'provider_identity': previous}
    rebuilt = {**saved, 'provider_identity': current}
    with pytest.raises(ValueError, match='identity'):
        _resume_identity_amendment(saved, rebuilt)
    assert _resume_identity_amendment(rebuilt, copy.deepcopy(rebuilt)) is None
    assert not cli.provider_identity_matches_for_inference(config, previous, current)


@pytest.mark.parametrize('invalid', [True, -0.1, 1.1, float('nan')])
def test_invalid_observation_fraction_is_rejected(invalid):
    with pytest.raises(ValueError, match='retained fraction'):
        cli.build(_config(invalid), device='cpu')


def test_nonzero_policy_rejects_unimplemented_core():
    provider = types.SimpleNamespace(identity_payload=lambda: {}, predict_native=lambda: None)
    with pytest.raises(ValueError, match='anchored observation-response core'):
        cli.configure_observation_native_retention(torch.nn.Linear(1, 1), provider, .25)
    assert provider.identity_payload() == {}


@pytest.mark.parametrize('empty_execution', [False, True])
def test_shared_epoch_zero_ignores_only_retention_without_mutating_source(empty_execution):
    source_recipe, retained_recipe = _config(), _config(.25)
    source_copy, retained_copy = copy.deepcopy(source_recipe), copy.deepcopy(retained_recipe)
    source_model, source = cli.build(source_recipe, device='cpu')
    retained_model, retained = cli.build(retained_recipe, device='cpu')
    source_identity, retained_identity = source.identity_payload(), retained.identity_payload()
    # Real case adapters may retain the source recipe in provider identity.
    other_execution = {} if empty_execution else {'deterministic_algorithms': True}
    source_identity['recipe'] = {'execution_config': copy.deepcopy(other_execution), 'native_budget': 4}
    retained_identity['recipe'] = {'execution_config': {**other_execution, 'native_retention_fraction': .25},
                                   'native_budget': 4}
    source.identity_payload = lambda: copy.deepcopy(source_identity)
    retained.identity_payload = lambda: copy.deepcopy(retained_identity)
    engine = cli.engine_for(source_recipe, device='cpu')
    retained_engine = cli.engine_for(retained_recipe, device='cpu')
    contract = common_contract(source_model, source, engine)
    assert _same(contract, common_contract(retained_model, retained, retained_engine))
    # Historical saved zero states can contain an empty execution mapping.
    if empty_execution:
        contract['provider']['recipe']['execution_config'] = {}
    state = source.training_state_dict()
    initial = {'schema': 'coupled_native_epoch_zero_v1', 'epoch': 0, 'scheduler_epoch': 0,
               'selector_epoch': 0, 'sampler_state': {'next_epoch': 1},
               'organization_clocks': {'_joint_epoch': 0, '_joint_update_index': 0},
               'contract': contract, 'model_state_dict': copy.deepcopy(source_model.state_dict()),
               'provider_training_state': copy.deepcopy(state), 'rng_state': capture_rng_state(),
               'calibration': copy.deepcopy(state['coupled_native']['calibration']),
               'optimizer_state_by_name': {'groups': [], 'state_by_name': {}}}
    snapshot = copy.deepcopy(initial)
    apply_epoch_zero(retained_model, retained, retained_engine, initial)
    assert _same(initial, snapshot)
    assert source_recipe == source_copy and retained_recipe == retained_copy
    assert _same(source_model.state_dict(), retained_model.state_dict())
    assert _same(source.training_state_dict(), retained.training_state_dict())
    assert 'native_retention_execution' in retained.identity_payload()

    for change in ('reader_seed', 'native_budget', 'deterministic_algorithms'):
        changed = copy.deepcopy(retained_identity)
        if change == 'reader_seed':
            changed['observation_response']['reader_seed'] += 1
        elif change == 'native_budget':
            changed['recipe']['native_budget'] += 1
        else:
            changed['recipe']['execution_config']['deterministic_algorithms'] = False
        retained.identity_payload = lambda changed=changed: copy.deepcopy(changed)
        with pytest.raises(ValueError, match='contract differs'):
            apply_epoch_zero(retained_model, retained, retained_engine, initial)


def test_current_shared_epoch_zero_is_persisted_and_restored_for_retention(tmp_path):
    torch.set_num_threads(1)
    source_recipe, retained_recipe = _config(), _config(.25)
    source_copy, retained_copy = copy.deepcopy(source_recipe), copy.deepcopy(retained_recipe)
    source_model, source = cli.build(source_recipe, device='cpu')
    source_engine = cli.engine_for(source_recipe, device='cpu')
    # This performs gradient calibration only; it never applies an optimizer step.
    initial = prepare_epoch_zero(source_model, source, source_engine)
    assert initial['epoch'] == 0 and not initial['optimizer_state_by_name']['state_by_name']
    path = tmp_path / 'shared_epoch_zero.pt'
    torch.save(initial, path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    snapshot = copy.deepcopy(saved)
    retained_model, retained = cli.build(retained_recipe, device='cpu')
    retained_engine = cli.engine_for(retained_recipe, device='cpu')
    optimizer_seed = apply_epoch_zero(retained_model, retained, retained_engine, saved)
    assert _same(saved, snapshot)
    assert _same(retained_model.state_dict(), saved['model_state_dict'])
    assert _same(optimizer_seed, saved['optimizer_state_by_name'])
    assert _same(retained.training_state_dict(), saved['provider_training_state'])
    assert _same(retained.training_state_dict()['coupled_native']['calibration'], saved['calibration'])
    assert _same(capture_rng_state(), saved['rng_state'])
    assert source_recipe == source_copy and retained_recipe == retained_copy
    assert retained.identity_payload()['native_retention_execution']['fraction'] == .25


@pytest.mark.parametrize('mode', ['full', 'restricted'])
def test_seven_output_single_reader_reduces_checkpoint_calls_and_preserves_gradients(monkeypatch, mode):
    torch.set_num_threads(1)
    checkpoints = []
    original_checkpoint = anchored.checkpoint

    def record_checkpoint(*args, **kwargs):
        checkpoints.append(1)
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(anchored, 'checkpoint', record_checkpoint)
    observed = []
    for fraction in (0., .25):
        model, provider = cli.build(_config(fraction), device='cpu')
        model.core.query_tile_size = 2
        model.train()
        provider.coupled_native_reader_mode = mode
        batch = provider.make_batch(('fixture_0', 'fixture_1'), key(provider))
        original_tile = model.core._decode_tile
        modes, heads = [], []

        def record_tile(_self, prepared, query, read_modes, **kwargs):
            modes.append(tuple(read_modes))
            return original_tile(prepared, query, read_modes, **kwargs)

        model.core._decode_tile = types.MethodType(record_tile, model.core)
        hook = model.core.member_router.joint_head.register_forward_hook(lambda *_: heads.append(1))
        before = len(checkpoints)
        try:
            predictions, _ = provider.predict_native(model, batch.scene_inputs, batch.receivers,
                model.architecture, 'joint', epoch=1, temperature=1.)
            assert all(prediction.shape[-1] == 7 for prediction in predictions)
            assert modes == [(mode,)] * 5 and len(heads) == 5
            # Four and five positions form 2+3 tiles; 25% retains one in each scene.
            assert len(checkpoints) - before == (5 if fraction == 0. else 3)
            assert anchored._decoder_execution.get() == 'checkpoint'
            components = [prediction[..., i].square().mean()
                          for prediction in predictions for i in range(7)]
            gradients = torch.autograd.grad(sum(components), tuple(model.parameters()), allow_unused=True)
            observed.append((predictions, components, gradients))
        finally:
            hook.remove()
    for actual, expected in zip(observed[1], observed[0]):
        for left, right in zip(actual, expected):
            assert (left is None) == (right is None)
            if left is not None:
                torch.testing.assert_close(left, right, rtol=3e-5, atol=2e-6)


def test_cli_saves_fraction_and_fresh_process_reloads_policy_without_fitting(tmp_path, monkeypatch):
    from honf_runtime import state_response_storage
    from honf_runtime.run_layout import RunLayout

    config = _config()
    config['observation_response']['enabled'] = False
    config['storage'] = {}
    recipe_path = tmp_path / 'recipe.json'
    recipe_path.write_text(json.dumps(config))
    output = tmp_path / 'run'
    monkeypatch.setattr(state_response_storage, 'prepare_state_response_storage',
                        lambda *args, **kwargs: {'run_output': str(output)})
    captured = {}

    def no_fit(engine, model, provider, path, **kwargs):
        captured.update(identity=kwargs['identity'], provider=provider.identity_payload())
        return {'completed_epoch': 0}

    monkeypatch.setattr(cli.TrainingEngine, 'fit', no_fit)
    assert cli.main(['fit', '--recipe', str(recipe_path), '--output', str(output), '--stop-after', '1',
                     '--native-retention-fraction', '.25']) == 0
    saved = RunLayout(output).read_path('resolved_config.json')
    saved_recipe = json.loads(saved.read_text())
    assert saved_recipe['execution_config']['native_retention_fraction'] == .25
    assert captured['identity']['recipe'] == saved_recipe
    assert captured['provider']['native_retention_execution']['fraction'] == .25

    root = Path(__file__).resolve().parents[1]
    script = """import json, sys
from tools.relation_anchored_train import build
recipe = json.loads(open(sys.argv[1]).read())
model, provider = build(recipe, device='cpu')
print(json.dumps({'execution': provider.identity_payload()['native_retention_execution'],
                  'dataset_id': provider.sampling_dataset_id}))
"""
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONPATH': os.pathsep.join((str(root), str(root / 'src'),
               str(root / 'Case_ThermalChannel/src'), str(root / 'Case_WindFarm/src')))}
    result = subprocess.run([sys.executable, '-c', script, str(saved)], cwd=root, env=env,
                            check=True, text=True, capture_output=True)
    reloaded = json.loads(result.stdout)
    assert reloaded['execution'] == captured['provider']['native_retention_execution']
    assert reloaded['dataset_id'] == _sampling_dataset_id(captured['provider'])
