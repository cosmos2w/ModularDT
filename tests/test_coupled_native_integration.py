"""Native graph reuse, matched panels and independent reader execution."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from honf_runtime.coupled_response_task import _fixed_pair_order, attach_coupled_native_task
from honf_runtime.observation_response_fixture import build_observation_capability
from honf_runtime.observation_response_task import build_observation_response_task
from honf_runtime.unified_training import LossTerm, SamplingKey


def recipe(variant='observed_response'):
    return {'dataset': 'analytic_fixture', 'seed': 3, 'architecture': 'relation_anchored',
            'total_epochs': 2, 'microbatch_size': 2, 'effective_batch_size': 2,
            'fixture': {'output_dim': 7}, 'observation_response': {
                'mode': 'coupled_native', 'variant': variant, 'anchor_count': 4,
                'pair_seed': 7, 'reader_seed': 9, 'full_reader_probability': .25,
                'calibration': {'schema': 'coupled_native_calibration_v1', 'epoch': 0, 'coefficient': .3}}}


def key(provider):
    return SamplingKey(provider.seed, 1, 0, 0, 'joint', provider.model.architecture,
                       sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)


def test_disabled_coupled_mode_builds_and_launches_plain_native_without_epoch_zero(tmp_path, monkeypatch):
    from tests.test_relation_anchored_train_entry import cli

    config = recipe()
    config['training_profile'] = 'observation_response'
    config['observation_response']['enabled'] = False
    config['observation_response'].pop('calibration')
    model, provider = build_observation_response_task(config, device='cpu')
    assert not hasattr(provider, 'native_microbatch_terms')
    assert not hasattr(provider, 'coupled_native_reader_mode')
    assert 'observation_response' not in provider.identity_payload()

    path = tmp_path / 'disabled_coupled_recipe.json'
    path.write_text(json.dumps(config))
    calls = []
    def fit(actual_model, actual_provider, output, **kwargs):
        calls.append((actual_model, actual_provider, kwargs))
        assert not hasattr(actual_provider, 'native_microbatch_terms')
        assert 'optimizer_seed' not in kwargs
        return {'completed_epoch': 0}
    monkeypatch.setattr(cli, 'engine_for', lambda *args, **kwargs: SimpleNamespace(fit=fit))
    monkeypatch.setattr(cli, 'prepare_profile_fit', lambda *args, **kwargs: None)
    assert cli.main(['fit', '--recipe', str(path), '--output', str(tmp_path / 'unused_run'),
                     '--stop-after', '1']) == 0
    assert len(calls) == 1


@pytest.mark.parametrize('enabled', [0, 1, 'false', None])
def test_coupled_enabled_must_remain_a_boolean(enabled):
    config = recipe()
    config['observation_response']['enabled'] = enabled
    with pytest.raises(ValueError, match='enabled must be a boolean'):
        build_observation_response_task(config, device='cpu')


def test_pair_packing_preserves_every_case_and_microbatch():
    members = tuple(f'case_{i}' for i in range(9))
    pairs = (('case_0', 'case_1'), ('case_3', 'case_4'), ('case_6', 'case_7'))
    result = _fixed_pair_order(members, pairs, 3, 6)
    assert set(result) == set(members) and len(result) == len(members)
    for left, right in pairs:
        a, b = result.index(left), result.index(right)
        assert a // 6 == b // 6 and (a % 6) // 3 == (b % 6) // 3


def test_native_panels_match_and_response_reuses_the_live_graph():
    torch.set_num_threads(1)
    model, provider = build_observation_response_task(recipe(), device='cpu')
    control_model, control = build_observation_response_task(recipe('native_control'), device='cpu')
    control_model.load_state_dict(model.state_dict())
    sampled_model, sampled = build_observation_response_task(recipe('sampled_reader_response'), device='cpu')
    sampled_model.load_state_dict(model.state_dict())
    pair = ('fixture_0', 'fixture_1')
    batches = [owner.make_batch(pair, key(owner)) for owner in (provider, control, sampled)]
    for batch in batches[1:]:
        for actual, expected in zip(batch.receivers, batches[0].receivers):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(batch.targets, batches[0].targets):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The independent reader stream must not consume or perturb query draws.
    sampled.begin_native_update(epoch=1, update_index=0, update_count=3)
    again = sampled.make_batch(pair, key(sampled))
    for actual, expected in zip(again.receivers, batches[0].receivers):
        np.testing.assert_array_equal(actual, expected)
    batch = batches[0]
    provider.begin_native_update(epoch=1, update_index=0, update_count=3)
    with patch.object(model.core, 'prepare', wraps=model.core.prepare) as prepare, \
            patch.object(model.core, 'decode_paired', wraps=model.core.decode_paired) as decode:
        predictions, auxiliary = provider.predict_native(model, batch.scene_inputs, batch.receivers,
            model.architecture, 'joint', epoch=1, temperature=1.)
        counts = (prepare.call_count, decode.call_count)
        terms, work = provider.native_microbatch_terms(batch, predictions, auxiliary,
                                                       epoch=1, update_index=0, update_count=3)
        assert (prepare.call_count, decode.call_count) == counts == (2, 2)
    assert work['coupled_pair_presentations'] == 1
    loss = terms['coupled_native/response'].numerator * terms['coupled_native/response'].weight
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    assert any(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients)
    control.begin_native_update(epoch=1, update_index=0, update_count=3)
    predictions, auxiliary = control.predict_native(control_model, batches[1].scene_inputs, batches[1].receivers,
        control_model.architecture, 'joint', epoch=1, temperature=1.)
    assert control.native_microbatch_terms(batches[1], predictions, auxiliary,
                                          epoch=1, update_index=0, update_count=3)[0] == {}


def test_probability_weighting_matches_dual_reader_objective_and_response():
    torch.set_num_threads(1)
    model, provider = build_observation_response_task(recipe('sampled_reader_response'), device='cpu')
    batch = provider.make_batch(('fixture_0', 'fixture_1'), key(provider))
    objectives, responses = {}, {}
    for mode in (None, 'full', 'restricted'):
        provider.coupled_native_reader_mode = mode
        predictions, auxiliary = provider.predict_native(model, batch.scene_inputs, batch.receivers,
            model.architecture, 'joint', epoch=1, temperature=1.)
        term = provider.loss_terms(predictions, batch.targets, 'joint', auxiliary)['native']
        objectives[mode] = term.numerator * term.weight / term.denominator
        terms, _ = provider.native_microbatch_terms(batch, predictions, auxiliary,
                                                     epoch=1, update_index=0, update_count=3)
        term = terms['coupled_native/response']
        responses[mode] = term.numerator * term.weight
    for values in (objectives, responses):
        torch.testing.assert_close(values[None], .25 * values['full'] + .75 * values['restricted'],
                                   rtol=2e-5, atol=1e-7)


def test_reader_sampling_leaves_view_independent_term_at_its_original_weight():
    torch.set_num_threads(1)
    config = recipe('sampled_reader_response')
    model, provider, binding = build_observation_capability(config, device='cpu')
    original = provider.loss_terms
    def losses(predictions, targets, phase, auxiliary):
        result = dict(original(predictions, targets, phase, auxiliary))
        result['view_independent'] = LossTerm(next(model.parameters()).square().sum(), 1, .7)
        return result
    provider.loss_terms = losses
    _, provider = attach_coupled_native_task(config, model, provider, binding)
    batch = provider.make_batch(('fixture_0', 'fixture_1'), key(provider))
    provider.coupled_native_reader_mode = 'full'
    predictions, auxiliary = provider.predict_native(model, batch.scene_inputs, batch.receivers,
        model.architecture, 'joint', epoch=1, temperature=1.)
    terms = provider.loss_terms(predictions, batch.targets, 'joint', auxiliary)
    assert terms['native'].weight == 2.0
    assert terms['view_independent'].weight == .7


def test_unsupported_sampling_keeps_other_pairs_available():
    original = recipe()
    model, provider, binding = build_observation_capability(copy.deepcopy(original), device='cpu')
    binding.coupled_native_sampling_law = lambda case: ('weighted_without_replacement'
        if case in ('fixture_0', 'fixture_1') else 'uniform_without_replacement')
    _, provider = attach_coupled_native_task(original, model, provider, binding)
    assert len(provider.coupled_native_pairs) == 2
    assert provider.identity_payload()['observation_response']['unavailable_pairs'][0]['pair'] == ['fixture_0', 'fixture_1']
    rows, carrier = provider.sample_coupled_native_rows(('fixture_0', 'fixture_1'), key(provider))
    assert rows == {} and carrier == ()


def test_existing_wind_binding_reports_unavailable_without_affecting_legacy_binding():
    from windfarm.training.observation_response import WindObservationBinding
    binding = WindObservationBinding.__new__(WindObservationBinding)
    report = binding.coupled_native_capability()
    assert report['available'] is False and 'ordinary native training remains available' in report['reason']


def test_thermal_contextual_override_passes_native_rows_and_carrier(monkeypatch):
    """Exercise the actual Thermal override, not its shadowed parent method."""
    from channelthermal.training import contextual_joint_task

    clock = []
    core = SimpleNamespace(set_organization_clock=lambda epoch, update: clock.append((epoch, update)))
    structure = {'module_centers': torch.zeros(2, 1, 2), 'module_present': torch.ones(2, 1),
                 'material_params': torch.ones(2, 6), 'heat_powers': torch.ones(2, 1)}
    sampled = {'structure': structure, 'query_xy': torch.zeros(2, 2, 2),
               'module_internal_query_points': torch.zeros(2, 1, 2), 'field_targets': torch.zeros(2, 2, 5),
               'point_weights': torch.ones(2, 2), 'interface_target': torch.zeros(2, 1, 2, 2),
               'interface_condition_valid_mask': torch.ones(2, 1, 2),
               'module_internal_temperature_points': torch.zeros(2, 1, 1)}
    declared = {'left': np.array([1, 0]), 'right': np.array([0, 1])}
    carrier = (('pair', 'actual_sample'),)
    seen = {}
    def sample_primary(*args, **kwargs):
        seen.update(kwargs)
        return sampled, 'receipt'
    monkeypatch.setattr(contextual_joint_task, '_sample_primary', sample_primary)
    owner = SimpleNamespace(sampling_version=SamplingKey.CASE_EPOCH_VERSION, sampling_dataset_id='dataset',
        native_only=True, total_epochs=2, model=SimpleNamespace(core=core),
        _train_index_by_id={'left': 0, 'right': 1}, training_cases=[{}, {}], primary_queries=2,
        device=torch.device('cpu'), training_samples=0, sampling_audit=False,
        sample_coupled_native_rows=lambda cases, sampling: (declared, carrier))
    sampling = SamplingKey(3, 1, 0, 0, 'joint', 'fixture',
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id='dataset')
    batch = contextual_joint_task.ThermalContextualJointTask.make_batch(owner, ('left', 'right'), sampling)
    assert seen['fluid_rows_by_case'] is declared
    assert seen['sampling_audit'] is False
    assert batch.auxiliary['coupled_native'] is carrier
    assert clock == [(1, 0)] and owner.training_samples == 2


def test_calibration_predicts_at_epoch_zero_and_restores_sampling_clock(monkeypatch):
    torch.set_num_threads(1)
    model, provider = build_observation_response_task(recipe(), device='cpu')
    original_batch, original_predict = provider.make_batch, provider.predict_native
    observed = []
    def make_batch(cases, sampling):
        model.core.set_organization_clock(sampling.epoch, sampling.update_index)
        return original_batch(cases, sampling)
    def predict(*args, **kwargs):
        observed.append((model.core._joint_epoch, model.core._joint_update_index, kwargs['epoch']))
        return original_predict(*args, **kwargs)
    monkeypatch.setattr(provider, 'make_batch', make_batch)
    monkeypatch.setattr(provider, 'predict_native', predict)
    calibration = provider.calibrate_coupled_native(model)
    assert calibration['epoch'] == 0 and observed and set(observed) == {(0, 0, 0)}
    assert (model.core._joint_epoch, model.core._joint_update_index) == (0, 0)
