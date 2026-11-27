"""A coupled response multiplier changes only its recorded objective weight."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_runtime.coupled_initialization import _same, apply_epoch_zero, common_contract
from honf_runtime.coupled_response_task import attach_coupled_native_task
from honf_runtime.observation_response_fixture import build_observation_capability
from honf_runtime.observation_response_task import build_observation_response_task
from honf_runtime.scalar_history import resolve_training_recording, training_recording_definitions
from honf_runtime.unified_training import _resume_identity_amendment, capture_rng_state
from tests.test_coupled_native_integration import key, recipe
from tests.test_relation_anchored_train_entry import cli


def _build(weight=None):
    config = recipe('sampled_reader_response')
    config['training_profile'] = 'observation_response'
    if weight is not None:
        config['observation_response']['response_weight'] = weight
    model, provider = build_observation_response_task(config, device='cpu')
    return config, model, provider


def _definitions(provider):
    columns = ('epoch', *provider.training_history_columns())
    return training_recording_definitions(columns, provider.loss_metadata(), provider.training_column_definitions())


@pytest.mark.parametrize('value', [None, 0, -0.5, float('inf'), float('-inf'), float('nan'),
                                   10**400, True, False, '0.5'])
def test_response_weight_requires_a_finite_positive_number(value):
    config = recipe()
    config['observation_response']['response_weight'] = value
    with pytest.raises(ValueError, match='response_weight must be a finite positive number'):
        build_observation_response_task(config, device='cpu')


def test_omitted_and_explicit_one_keep_identity_and_recording():
    _, model, baseline = _build()
    _, same_model, explicit = _build(1.0)
    same_model.load_state_dict(model.state_dict())
    assert baseline.identity_payload() == explicit.identity_payload()
    assert baseline.loss_metadata() == explicit.loss_metadata()
    assert _definitions(baseline) == _definitions(explicit)
    left = baseline.make_batch(('fixture_0', 'fixture_1'), key(baseline))
    right = explicit.make_batch(('fixture_0', 'fixture_1'), key(explicit))
    for before, after in zip(left.receivers, right.receivers):
        np.testing.assert_array_equal(before, after)
    for owner, current, batch in ((baseline, model, left), (explicit, same_model, right)):
        owner.coupled_native_reader_mode = 'full'
        prediction, auxiliary = owner.predict_native(current, batch.scene_inputs, batch.receivers,
            current.architecture, 'joint', epoch=25, temperature=1.)
        terms, _ = owner.native_microbatch_terms(batch, prediction, auxiliary, epoch=25,
                                                update_index=0, update_count=3)
        owner._test_response = terms['coupled_native/response']
    assert baseline._test_response.weight == explicit._test_response.weight
    torch.testing.assert_close(baseline._test_response.numerator, explicit._test_response.numerator, rtol=0, atol=0)


def test_explicit_one_is_normalized_in_embedded_source_recipe():
    identities = []
    for weight in (None, 1.0):
        config = recipe()
        if weight is not None:
            config['observation_response']['response_weight'] = weight
        model, provider, binding = build_observation_capability(config, device='cpu')
        original = provider.identity_payload
        provider.identity_payload = lambda: {**original(), 'recipe': copy.deepcopy(config)}
        _, provider = attach_coupled_native_task(config, model, provider, binding)
        identities.append(provider.identity_payload())
    assert identities[0] == identities[1]


@pytest.mark.parametrize('mode', [None, 'full', 'restricted'])
def test_half_weight_scales_only_response_and_its_gradient(mode):
    torch.set_num_threads(1)
    _, model, baseline = _build()
    _, half_model, half = _build(.5)
    half_model.load_state_dict(model.state_dict())
    left = baseline.make_batch(('fixture_0', 'fixture_1'), key(baseline))
    right = half.make_batch(('fixture_0', 'fixture_1'), key(half))
    for before, after in zip(left.receivers, right.receivers):
        np.testing.assert_array_equal(before, after)
    predictions = []
    for owner, current, batch in ((baseline, model, left), (half, half_model, right)):
        owner.coupled_native_reader_mode = mode
        prediction, auxiliary = owner.predict_native(current, batch.scene_inputs, batch.receivers,
            current.architecture, 'joint', epoch=25, temperature=1.)
        predictions.append((prediction, auxiliary))
    native = [owner.loss_terms(prediction, batch.targets, 'joint', auxiliary)['native']
              for owner, batch, (prediction, auxiliary) in
              zip((baseline, half), (left, right), predictions)]
    assert native[0].weight == native[1].weight and native[0].denominator == native[1].denominator
    torch.testing.assert_close(native[0].numerator, native[1].numerator, rtol=0, atol=0)
    for epoch in (1, 25, 50, 1000):
        terms = [owner.native_microbatch_terms(batch, prediction, auxiliary, epoch=epoch,
                    update_index=0, update_count=3)[0]['coupled_native/response']
                 for owner, batch, (prediction, auxiliary) in
                 zip((baseline, half), (left, right), predictions)]
        assert terms[1].weight == .5 * terms[0].weight
        assert terms[0].denominator == terms[1].denominator
        torch.testing.assert_close(terms[0].numerator, terms[1].numerator, rtol=0, atol=0)
        if epoch == 25:
            gradients = [torch.autograd.grad(term.numerator * term.weight, tuple(current.parameters()),
                                            allow_unused=True, retain_graph=True)
                         for term, current in zip(terms, (model, half_model))]
            for before, after in zip(*gradients):
                if before is None:
                    assert after is None
                else:
                    torch.testing.assert_close(after, .5 * before, rtol=1e-6, atol=1e-8)


def test_half_weight_scales_dual_validation_but_not_calibration():
    torch.set_num_threads(1)
    _, model, baseline = _build()
    _, half_model, half = _build(.5)
    half_model.load_state_dict(model.state_dict())
    base_state, half_state = baseline.training_state_dict(), half.training_state_dict()
    assert base_state['coupled_native']['calibration'] == half_state['coupled_native']['calibration']
    for epoch in (1, 25, 50, 1000):
        with torch.no_grad():
            original = baseline.validation_auxiliary_terms(model, epoch=epoch, arm=model.architecture)['coupled_native/response']
            reduced = half.validation_auxiliary_terms(half_model, epoch=epoch, arm=half_model.architecture)['coupled_native/response']
        assert reduced.weight == .5 * original.weight and reduced.denominator == original.denominator
        torch.testing.assert_close(original.numerator, reduced.numerator, rtol=0, atol=0)
    assert baseline.training_state_dict()['coupled_native']['calibration'] == base_state['coupled_native']['calibration']
    assert half.training_state_dict()['coupled_native']['calibration'] == half_state['coupled_native']['calibration']


def test_common_zero_excludes_only_weight_but_resume_retains_it():
    config, model, baseline = _build()
    half_config, half_model, half = _build(.5)
    base_identity, half_identity = baseline.identity_payload(), half.identity_payload()
    # A general adapter may also include its source recipe in its identity.
    base_identity['recipe'] = {'name': 'baseline', 'observation_response': {'mode': 'coupled_native'}}
    half_identity['recipe'] = {'name': 'half', 'observation_response':
                               {'mode': 'coupled_native', 'response_weight': .5}}
    baseline.identity_payload = lambda: copy.deepcopy(base_identity)
    half.identity_payload = lambda: copy.deepcopy(half_identity)
    engine = cli.engine_for(config, device='cpu')
    half_engine = cli.engine_for(half_config, device='cpu')
    contract = common_contract(model, baseline, engine)
    assert _same(contract, common_contract(half_model, half, half_engine))
    initial = {'schema': 'coupled_native_epoch_zero_v1', 'epoch': 0, 'scheduler_epoch': 0,
               'selector_epoch': 0, 'sampler_state': {'next_epoch': 1},
               'organization_clocks': {'_joint_epoch': 0, '_joint_update_index': 0},
               'contract': contract, 'model_state_dict': copy.deepcopy(model.state_dict()),
               'provider_training_state': copy.deepcopy(baseline.training_state_dict()),
               'calibration': copy.deepcopy(baseline.training_state_dict()['coupled_native']['calibration']),
               'rng_state': capture_rng_state(), 'optimizer_state_by_name': {'groups': [], 'state_by_name': {}}}
    optimizer_seed = apply_epoch_zero(half_model, half, half_engine, initial)
    assert _same(model.state_dict(), half_model.state_dict())
    assert _same(initial['optimizer_state_by_name'], optimizer_seed)
    assert _same(initial['provider_training_state'], half.training_state_dict())
    assert half_model.core._joint_epoch == half_model.core._joint_update_index == 0
    changed = copy.deepcopy(half_identity)
    changed['observation_response']['reader_seed'] += 1
    half.identity_payload = lambda: copy.deepcopy(changed)
    with pytest.raises(ValueError, match='contract differs'):
        apply_epoch_zero(half_model, half, half_engine, initial)
    with pytest.raises(ValueError, match='Checkpoint identity'):
        _resume_identity_amendment({'provider_identity': base_identity}, {'provider_identity': half_identity})
    before, after = _definitions(baseline), _definitions(half)
    assert before['training_columns'] == after['training_columns']
    assert before['training_loss_order'] == after['training_loss_order']
    assert 'post-calibration response multiplier 0.5' in after['terms']['coupled_native/response']['weight']
    assert 'response_weight=0.5' in after['training_column_definitions']['train_losses.coupled_native/response']['meaning']
    with pytest.raises(ValueError, match='coupled_native/response'):
        resolve_training_recording(before, after)


def test_common_zero_shares_update_scope_but_keeps_reader_policy():
    config, model, provider = _build()
    engine = cli.engine_for(config, device='cpu')
    identity = copy.deepcopy(provider.identity_payload())
    identity['observation_response'].update(reader_policy='sampled_equal_probability',
        full_reader_probability=.5, response_update_scope='all_parameters',
        response_gradient_composition={'kind': 'native_priority_projection', 'maximum_native_norm_ratio': .1})
    identity['recipe'] = {'name': 'all response', 'observation_response': {
        'mode': 'coupled_native', 'variant': 'observed_response',
        'reader_policy': 'sampled_equal_probability', 'full_reader_probability': .5,
        'response_update_scope': 'all_parameters', 'response_gradient_composition': {
            'kind': 'native_priority_projection', 'maximum_native_norm_ratio': .1}}}
    provider.identity_payload = lambda: copy.deepcopy(identity)
    original = copy.deepcopy(identity)
    contract = common_contract(model, provider, engine)
    for scope in ('disabled', 'organization_parameters'):
        changed = copy.deepcopy(identity)
        changed['observation_response']['response_update_scope'] = scope
        changed['recipe']['name'] = scope
        changed['recipe']['observation_response'].update(
            variant='native_control' if scope == 'disabled' else 'observed_response',
            response_update_scope=scope)
        if scope == 'disabled':
            changed['observation_response'].pop('response_gradient_composition')
            changed['recipe']['observation_response'].pop('response_gradient_composition')
        provider.identity_payload = lambda changed=changed: copy.deepcopy(changed)
        assert _same(contract, common_contract(model, provider, engine))
    changed['observation_response']['reader_policy'] = 'dual_reader'
    provider.identity_payload = lambda: copy.deepcopy(changed)
    assert not _same(contract, common_contract(model, provider, engine))
    assert identity == original


def test_saved_config_reload_accepts_key_reordering_and_rejects_weight_change(tmp_path, monkeypatch):
    config, _, _ = _build(.5)
    saved = {'epoch': 1, 'experiment_identity': {'recipe': copy.deepcopy(config),
                                                 'engine_config': {'total_epochs': 2}}}
    monkeypatch.setattr(cli, 'resolve_checkpoint', lambda *args: tmp_path / 'latest_model.pt')
    monkeypatch.setattr(cli, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    monkeypatch.setattr(cli, 'build', lambda *args, **kwargs: (SimpleNamespace(architecture='fixture'), object()))
    monkeypatch.setattr(cli, 'prepare_profile_fit', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, 'engine_for', lambda *args, **kwargs:
                        SimpleNamespace(config=SimpleNamespace(total_epochs=2),
                                        fit=lambda *args, **kwargs: {'completed_epoch': 1}))
    path = tmp_path / 'saved_recipe.json'
    def reverse(value):
        if isinstance(value, dict):
            return {name: reverse(value[name]) for name in reversed(value)}
        if isinstance(value, list):
            return [reverse(item) for item in value]
        return value
    path.write_text(json.dumps(reverse(config)))
    arguments = ['fit', '--recipe', str(path), '--output', str(tmp_path / 'run'), '--resume', 'latest', '--stop-after', '2']
    assert cli.main(arguments) == 0
    config['observation_response']['response_weight'] = .75
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='recipe differs from the saved run'):
        cli.main(arguments)
