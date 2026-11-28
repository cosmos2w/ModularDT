"""Generic finite objectives on an independent, weighted native sampler."""
import copy
import json

import numpy as np
import pytest
import torch

from honf_runtime.coupled_initialization import _same, apply_epoch_zero, prepare_epoch_zero
from honf_runtime.finite_observation_task import execute_finite_dependencies
from honf_runtime.finite_observations import finite_scene_contrast
from honf_runtime.observation_response_fixture import build_observation_capability
from honf_runtime.observation_response_task import build_observation_response_task
from honf_runtime.organization_parameters import partition_organization_parameters
from honf_runtime.scalar_history import resolve_training_recording, training_recording_definitions
from honf_runtime.unified_training import (EngineConfig, SamplingKey, TrainingEngine, capture_rng_state,
                                           named_optimizer_state)


torch.set_num_threads(1)


def _recipe(variant='contrasts'):
    return {'dataset': 'analytic_fixture', 'seed': 7, 'architecture': 'relation_anchored', 'total_epochs': 2,
            'fixture': {'output_dim': 7, 'finite_only_binding': True,
                        'native_sampling_law': 'weighted_with_replacement', 'native_draw_count': 8},
            'training': {'microbatch_cases': 2, 'effective_batch_size': 4},
            'finite_observations': {'native_sampling': 'native', 'variant': variant, 'probe_count': 64,
                'probe_seed': 130013, 'metadata_cache_bytes': 1024 * 1024,
                'calibration_case_ids': [f'fixture_{index}' for index in range(6)]}}


def _calibration():
    # Choose independent test gradient norms so both coefficients satisfy the
    # actual 0.10/0.05 calibration contract without running calibration per test.
    return {'schema': 'finite_observation_calibration_v1', 'epoch': 0,
            'calibration_case_ids': [f'fixture_{index}' for index in range(6)],
            'coefficients': {'anchor': .4, 'contrast': .2},
            'gradient_diagnostics': {'groups': {'global': {'norms': {'native': 10., 'anchor': 2.5, 'contrast': 2.5}}}}}


def _build(variant='contrasts'):
    model, provider = build_observation_response_task(_recipe(variant), device='cpu')
    provider.set_finite_observation_calibration(_calibration())
    return model, provider


def _key(provider, *, epoch=1):
    return SamplingKey(provider.seed, epoch, 0, 0, 'joint', provider.model.architecture,
                       sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)


def _engine(*, seed=7, total_epochs=2):
    return TrainingEngine(EngineConfig(seed=seed, microbatch_cases=2, effective_cases=4,
        total_epochs=total_epochs, training_mode='joint', sampling_version=SamplingKey.CASE_EPOCH_VERSION),
        device='cpu')


def _prediction(model, provider, batch):
    return provider.predict_native(model, provider.make_scene(batch.scene_inputs), batch.receivers,
                                   model.architecture, 'joint', epoch=1, temperature=1.)


def test_common_initial_state_has_identical_native_draws_predictions_and_extra_probes():
    base_model, base_provider, base_binding = build_observation_capability(_recipe('disabled'), device='cpu')
    models_providers = [_build(variant) for variant in ('disabled', 'anchors', 'contrasts')]
    base_model.train()
    base_provider.coupled_native_reader_mode = 'full'
    pair = ('fixture_0', 'fixture_1')
    base_batch = base_provider.make_batch(pair, _key(base_provider))
    base_prediction, base_aux = _prediction(base_model, base_provider, base_batch)
    base_native = base_provider.loss_terms(base_prediction, base_batch.targets, 'joint', base_aux)['native']
    probe_outputs = []
    for model, provider in models_providers:
        model.load_state_dict(base_model.state_dict())
        model.train()
        provider.finite_observation_reader_mode = 'full'
        assert not hasattr(provider.observation_binding, 'coupled_native_population')
        assert provider.sampling_dataset_id == base_provider.sampling_dataset_id
        batch = provider.make_batch(pair, _key(provider))
        for left, right in zip(base_batch.receivers, batch.receivers):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(base_batch.targets, batch.targets):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        predicted, auxiliary = _prediction(model, provider, batch)
        native = provider.loss_terms(predicted, batch.targets, 'joint', auxiliary)['native']
        assert (native.denominator, native.weight) == (base_native.denominator, base_native.weight)
        torch.testing.assert_close(native.numerator, base_native.numerator, rtol=0, atol=0)
        chosen = tuple(provider.finite_observation_scenes[case] for case in pair)
        probes, counts = execute_finite_dependencies(provider.observation_binding, model, chosen, read_mode='full')
        totals, _, observations = provider.finite_observation_components(model, pair, read_mode='full')
        probe_outputs.append((probes, totals, counts, observations))
    first, second = probe_outputs[1:]
    for case in pair:
        assert set(first[0][case]) == set(second[0][case])
        for family in first[0][case]:
            torch.testing.assert_close(first[0][case][family], second[0][case][family], rtol=0, atol=0)
            torch.testing.assert_close(first[3][case].families[family].moments,
                                       second[3][case].families[family].moments, rtol=0, atol=0)
    for name in ('anchor', 'contrast'):
        torch.testing.assert_close(first[1][name], second[1][name], rtol=0, atol=0)
    for name in ('requested_dependencies', 'dependencies', 'observation_rows', 'neural_queries', 'scene_preparations'):
        assert first[2][name] == second[2][name]
    assert base_binding.identity_payload()['schema'] == 'analytic_finite_observations_v1'


def test_both_observation_losses_reach_relation_and_joint_head_parameters():
    model, provider = _build()
    components, _, _ = provider.finite_observation_components(model, ('fixture_0', 'fixture_1'), read_mode='full')
    ownership = partition_organization_parameters(model)
    named = dict(model.named_parameters())
    head_names = tuple(name for name in ownership.content_names if '.joint_head.' in name)
    assert ownership.organization_names and head_names
    names = (*ownership.organization_names, *head_names)
    for loss_name in ('anchor', 'contrast'):
        gradients = torch.autograd.grad(components[loss_name], [named[name] for name in names],
                                        allow_unused=True, retain_graph=True)
        observed = dict(zip(names, gradients))
        for group in (ownership.organization_names, head_names):
            assert any(observed[name] is not None and torch.isfinite(observed[name]).all()
                       and torch.count_nonzero(observed[name]).item() > 0 for name in group)


def test_epoch_case_and_pair_weights_include_partial_effective_batch_exactly():
    model, provider = _build()
    provider.finite_observation_reader_mode = 'full'
    pairs = (('fixture_0', 'fixture_1'), ('fixture_2', 'fixture_3'), ('fixture_4', 'fixture_5'))
    expected_anchor = expected_contrast = 0.
    actual_anchor = actual_contrast = 0.
    weights = provider.observation_binding.finite_observation_family_weights()
    for index, pair in enumerate(pairs):
        batch = provider.make_batch(pair, _key(provider, epoch=25))
        components, _, observations = provider.finite_observation_components(model, pair, read_mode='full')
        expected_anchor += sum(float(observations[case].anchor.detach()) for case in pair)
        expected_contrast += float(finite_scene_contrast(observations[pair[0]], observations[pair[1]],
                                                         configured_weights=weights).loss.detach())
        terms, _ = provider.native_microbatch_terms(batch, None, None, epoch=25,
                                                    update_index=index // 2, update_count=2)
        anchor = terms['finite_observations/anchor']
        contrast = terms['finite_observations/contrast']
        torch.testing.assert_close(anchor.numerator, components['anchor'], rtol=0, atol=0)
        torch.testing.assert_close(contrast.numerator, components['contrast'], rtol=0, atol=0)
        actual_anchor += float(anchor.numerator.detach()) * anchor.weight / anchor.denominator / 2
        actual_contrast += float(contrast.numerator.detach()) * contrast.weight / contrast.denominator / 2
    # Three pairs and six scenes span one four-scene and one two-scene
    # effective batch. The epoch mean still gives each case/pair equal weight.
    assert actual_anchor == pytest.approx(.4 * .5 * expected_anchor / 6)
    assert actual_contrast == pytest.approx(.2 * .5 * expected_contrast / 3)
    split = provider.make_batch(('fixture_0',), _key(provider))
    with pytest.raises(ValueError, match='split across native microbatches'):
        provider.native_microbatch_terms(split, None, None, epoch=25, update_index=0, update_count=2)


def test_training_engine_uses_one_scalar_backward_per_native_microbatch(monkeypatch):
    model, provider = _build()
    model.train()
    assert getattr(provider, 'native_priority_composition', None) is None
    assert provider.auxiliary_update_terms is None
    engine = TrainingEngine(EngineConfig(seed=7, microbatch_cases=2, effective_cases=4, total_epochs=2,
        training_mode='joint', sampling_version=SamplingKey.CASE_EPOCH_VERSION), device='cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    batches = [provider.make_batch(pair, _key(provider)) for pair in
               (('fixture_0', 'fixture_1'), ('fixture_2', 'fixture_3'))]
    calls = []
    backward = torch.Tensor.backward
    def counted_backward(tensor, *args, **kwargs):
        calls.append(tensor.detach().clone())
        return backward(tensor, *args, **kwargs)
    def no_extra_vector_jacobian(*args, **kwargs):
        raise AssertionError('The training update used a separate gradient or projection pass.')
    monkeypatch.setattr(torch.Tensor, 'backward', counted_backward)
    monkeypatch.setattr(torch.autograd, 'grad', no_extra_vector_jacobian)
    losses, work, _, _ = engine._run_update(model, provider, optimizer, batches, phase='joint',
                                             arm=model.architecture, epoch=1, update_index=0,
                                             epoch_update_count=2)
    assert len(calls) == len(batches)
    assert all(torch.isfinite(value) and value.ndim == 0 for value in calls)
    assert set(('native', 'finite_observations/anchor', 'finite_observations/contrast')) <= set(losses)
    assert not any(name.startswith(('native_priority_', 'state_priority_')) for name in work)
    assert work['finite_observations_scenes'] == 4 and work['finite_observations_pairs'] == 2


def test_calibration_is_independent_and_restores_model_sampler_and_rng(monkeypatch):
    model, provider = _build()
    model.eval()
    before_model = copy.deepcopy(model.state_dict())
    before_provider = copy.deepcopy(provider.training_state_dict())
    before_rng = torch.get_rng_state().clone()
    native_key = _key(provider)
    native_seed = native_key.seed_for('fixture_native_weighted_rows', 'fixture_0')
    source = provider.make_batch
    calibration_seeds = []
    def observe(case_keys, key):
        calibration_seeds.append(key.seed_for('fixture_native_weighted_rows', case_keys[0]))
        return source(case_keys, key)
    monkeypatch.setattr(provider, 'make_batch', observe)
    record = provider.calibrate_finite_observations(model)
    assert len(calibration_seeds) == 3 and calibration_seeds[0] != native_seed
    assert not model.training and provider.training_state_dict() == before_provider
    assert torch.equal(torch.get_rng_state(), before_rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_model[name], rtol=0, atol=0)
    norms = record['gradient_diagnostics']['groups']['global']['norms']
    assert record['coefficients']['anchor'] == pytest.approx(.10 * norms['native'] / norms['anchor'])
    assert record['coefficients']['contrast'] == pytest.approx(.05 * norms['native'] / norms['contrast'])
    assert all(value > 0 for value in norms.values())
    provider.set_finite_observation_calibration(record)


def test_recorded_columns_and_loss_order_are_stable_on_saved_definition_reload(tmp_path):
    _, provider = _build()
    columns = ('epoch', *provider.training_history_columns())
    definition = training_recording_definitions(columns, provider.loss_metadata(), provider.training_column_definitions())
    assert definition['training_loss_order'] == ['native', 'finite_observations/anchor', 'finite_observations/contrast']
    assert len(columns) == len(set(columns))
    for column in columns:
        fields = definition['training_column_definitions'][column]
        assert set(fields) == {'dtype', 'unit', 'meaning'} and all(fields.values())
    path = tmp_path / 'metric_definitions.json'
    path.write_text(json.dumps(definition, sort_keys=True, indent=2))
    saved = json.loads(path.read_text())
    restored_columns, restored_order = resolve_training_recording(saved, definition)
    assert restored_columns == columns
    assert restored_order == tuple(definition['training_loss_order'])
    changed = copy.deepcopy(definition)
    changed['training_column_definitions']['train_losses.finite_observations/anchor']['unit'] = 'wrong units'
    with pytest.raises(ValueError, match='numerical meaning changed=True'):
        resolve_training_recording(saved, changed)
    pointer = provider.write_finite_observation_definitions(tmp_path / 'run')
    source = json.loads(open(pointer['source_definitions']).read())
    assert source['probe_seed'] == 130013 and len(source['scenes']) == 8
    assert source['scenes'][0]['families'][0]['row_ids'] == list(provider.finite_observation_scenes['fixture_0'].families[0].row_ids)
    assert source['scenes'][0]['families'][0]['reference'].startswith('existing adapter source arrays')
    assert provider.write_finite_observation_definitions(tmp_path / 'run') == pointer


def test_resume_preserves_definitions_when_only_process_ram_estimate_changes(tmp_path):
    _, provider = _build()
    run = tmp_path / 'run'
    provider.write_finite_observation_definitions(run)
    path = run / 'configs/finite_observation_objective.json'
    saved = json.loads(path.read_text())
    saved['metadata_bytes'] += 8192
    path.write_text(json.dumps(saved, sort_keys=True))
    before = path.read_bytes()
    provider.write_finite_observation_definitions(run)
    assert path.read_bytes() == before
    saved['objective']['basis'] = 'a changed scientific basis'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="changed=\\['objective'\\]"):
        provider.write_finite_observation_definitions(run)


@pytest.fixture(scope='module')
def finite_epoch_zero():
    model, provider = build_observation_response_task(_recipe(), device='cpu')
    return prepare_epoch_zero(model, provider, _engine())


def test_epoch_zero_v2_restores_identical_native_state_and_preserves_each_objective(finite_epoch_zero, tmp_path):
    assert finite_epoch_zero['schema'] == 'native_observation_epoch_zero_v2'
    path = tmp_path / 'common_initial_state.pt'
    torch.save(finite_epoch_zero, path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    original = copy.deepcopy(payload)
    expected_names = {'disabled': (), 'anchors': ('finite_observations/anchor',),
                      'contrasts': ('finite_observations/anchor', 'finite_observations/contrast')}
    for variant in ('disabled', 'anchors', 'contrasts'):
        model, provider = build_observation_response_task(_recipe(variant), device='cpu')
        provider.training_samples = 19
        torch.rand(4)
        optimizer_seed = apply_epoch_zero(model, provider, _engine(), payload)
        assert _same(model.state_dict(), payload['model_state_dict'])
        assert _same(provider.native_epoch_zero_state(), payload['provider_training_state'])
        assert _same(provider.training_state_dict()['finite_observations']['calibration'], payload['finite_calibration'])
        assert provider.identity_payload()['finite_observations']['variant'] == variant
        assert provider.native_reuse_loss_names == expected_names[variant]
        assert _same(optimizer_seed, payload['optimizer_state_by_name'])
        optimizer, _ = _engine()._make_optimizer(model, provider, model.architecture, 1, optimizer_seed)
        assert _same(named_optimizer_state(model, optimizer), payload['optimizer_state_by_name'])
        assert _same(capture_rng_state(), payload['rng_state'])
    assert _same(payload, original)


def test_epoch_zero_v2_rejects_native_sampling_and_optimizer_contract_changes(finite_epoch_zero):
    changed = _recipe()
    changed['fixture']['native_draw_count'] = 5
    model, provider = build_observation_response_task(changed, device='cpu')
    model_before = copy.deepcopy(model.state_dict())
    provider_before = copy.deepcopy(provider.training_state_dict())
    with pytest.raises(ValueError, match='contract differs'):
        apply_epoch_zero(model, provider, _engine(), finite_epoch_zero)
    assert _same(model.state_dict(), model_before) and _same(provider.training_state_dict(), provider_before)
    model, provider = build_observation_response_task(_recipe(), device='cpu')
    with pytest.raises(ValueError, match='contract differs'):
        apply_epoch_zero(model, provider, _engine(total_epochs=3), finite_epoch_zero)
    model, provider = build_observation_response_task(_recipe(), device='cpu')
    with pytest.raises(ValueError, match='contract differs'):
        apply_epoch_zero(model, provider, _engine(seed=8), finite_epoch_zero)


def test_bare_coupled_all_parameter_recipe_can_load_same_finite_epoch_zero():
    source = _recipe()
    source['fixture'] = {'output_dim': 7}
    source['finite_observations']['native_sampling'] = 'coupled_native'
    source['observation_response'] = {'mode': 'coupled_native', 'variant': 'native_control',
        'anchor_count': 4, 'pair_seed': 7, 'reader_seed': 9,
        'reader_policy': 'sampled_equal_probability', 'full_reader_probability': .5}
    model, provider = build_observation_response_task(source, device='cpu')
    initial = prepare_epoch_zero(model, provider, _engine())
    assert initial['schema'] == 'native_observation_epoch_zero_v2'
    assert initial['calibration'] is not None and initial['finite_calibration'] is not None
    legacy = copy.deepcopy(source)
    legacy.pop('finite_observations')
    legacy['observation_response'].update({'variant': 'sampled_reader_response',
        'response_update_scope': 'all_parameters', 'response_gradient_composition': {
            'kind': 'native_priority_projection', 'maximum_native_norm_ratio': .1}})
    coupled_model, coupled_provider = build_observation_response_task(legacy, device='cpu')
    optimizer_seed = apply_epoch_zero(coupled_model, coupled_provider, _engine(), initial)
    assert _same(coupled_model.state_dict(), initial['model_state_dict'])
    assert _same(coupled_provider.training_state_dict(), initial['provider_training_state'])
    assert _same(optimizer_seed, initial['optimizer_state_by_name'])
    assert coupled_provider.native_priority_composition['scope'] == 'all_parameters'
    assert _same(capture_rng_state(), initial['rng_state'])
