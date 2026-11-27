"""Review-time family gradients use the native sampler and preserve state."""
from __future__ import annotations

import copy
import json
import random
from types import MethodType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_runtime.coupled_initialization import _same
from honf_runtime.loss_family_gradient_audit import _gradient_groups, audit_loss_families
from honf_runtime.observation_response_task import build_observation_response_task
from honf_runtime.unified_training import LossTerm
from tests.test_native_priority_composition import scoped_recipe
from tests.test_coupled_native_integration import key


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _split_fixture_terms(provider):
    """Make three independent named, differently weighted native terms."""
    old_terms, old_denominators = provider.loss_terms, provider.loss_denominators
    coefficients = {'native_flow': 2., 'native_fluid_temperature': 3., 'native_member_interface': 5.}
    def loss_terms(self, predictions, targets, phase, auxiliary):
        source = old_terms(predictions, targets, phase, auxiliary)['native']
        return {name: LossTerm(source.numerator, source.denominator, source.weight * factor)
                for name, factor in coefficients.items()}
    def denominators(self, batches, phase, arm):
        source = old_denominators(batches, phase, arm)['native']
        return {name: source for name in coefficients}
    provider.loss_terms = MethodType(loss_terms, provider)
    provider.loss_denominators = MethodType(denominators, provider)
    return {'flow': ('native_flow',), 'fluid_temperature': ('native_fluid_temperature',),
            'member_interface': ('native_member_interface',)}


def _run(scope):
    model, provider = build_observation_response_task(scoped_recipe(scope), device='cpu')
    groups = _split_fixture_terms(provider)
    result = audit_loss_families(model, provider, panel_case_ids=('fixture_0', 'fixture_1', 'fixture_2', 'fixture_3'),
                                 epoch=2, native_families=groups)
    return model, provider, result


def test_native_disabled_still_measures_live_response_without_update():
    _, _, disabled = _run('disabled')
    _, _, focused = _run('organization_parameters')
    assert disabled['complete_training_pairs_on_panel'] == [['fixture_0', 'fixture_1'], ['fixture_2', 'fixture_3']]
    for mode in ('full', 'restricted'):
        left, right = disabled['reader_results'][mode], focused['reader_results'][mode]
        assert left['response']['available'] and left['response']['pair_denominator'] == 2
        assert left['response']['calibration_coefficient'] == .3
        assert left['response']['epoch_ramp'] == pytest.approx(.04)
        assert left['response']['reader_probability_factor'] == 1.
        assert left['losses']['response'] == pytest.approx(right['losses']['response'], rel=1e-7)
        for family in ('global', 'organization', 'content_shared'):
            assert left['gradient_groups'][family]['norms']['response'] == pytest.approx(
                right['gradient_groups'][family]['norms']['response'], rel=1e-7)
        assert left['gradient_groups']['global']['norms']['response'] > 0
    assert disabled['optimizer_steps'] == disabled['fitting_epochs'] == 0
    json.dumps(disabled, allow_nan=False)


def test_recorded_ordered_denominators_and_weighted_gradient_algebra():
    _, _, result = _run('organization_parameters')
    for view in result['reader_results'].values():
        gradient = view['gradient_groups']['global']
        norms, cosines = gradient['norms'], gradient['cosines']
        assert norms['fluid_temperature'] == pytest.approx(1.5 * norms['flow'], rel=1e-6)
        assert norms['member_interface'] == pytest.approx(2.5 * norms['flow'], rel=1e-6)
        assert norms['native_total'] == pytest.approx(5 * norms['flow'], rel=1e-6)
        assert cosines['flow:fluid_temperature'] == pytest.approx(1., abs=1e-7)
        assert cosines['flow:member_interface'] == pytest.approx(1., abs=1e-7)
        terms = view['native_terms']
        assert list(terms) == ['native_flow', 'native_fluid_temperature', 'native_member_interface']
        assert [terms[n]['weight'] for n in terms] == [2., 3., 5.]
        assert len({terms[n]['denominator'] for n in terms}) == 1
        assert view['response']['total_weight'] == pytest.approx(.012)
        assert view['response']['pair_denominator'] == 2


def test_raw_cosines_use_each_resolved_parameter_subset():
    class TwoPart(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.organization = torch.nn.Parameter(torch.zeros(2))
            self.content = torch.nn.Parameter(torch.zeros(1))
    ownership = SimpleNamespace(organization_names=('organization',), content_names=('content',))
    gradients = {'flow': [torch.tensor([1., 0.]), torch.tensor([2.])],
                 'response': [torch.tensor([-1., 1.]), torch.tensor([1.])],
                 'zero': [torch.zeros(2), torch.zeros(1)]}
    groups = _gradient_groups(TwoPart(), gradients, ownership)
    assert groups['global']['cosines']['flow:response'] == pytest.approx(1 / np.sqrt(15))
    assert groups['organization']['cosines']['flow:response'] == pytest.approx(-1 / np.sqrt(2))
    assert groups['content_shared']['cosines']['flow:response'] == pytest.approx(1.)
    assert groups['global']['cosines']['flow:zero'] is None


def test_state_rng_gradients_and_sampler_are_restored_on_success_and_failure():
    model, provider = build_observation_response_task(scoped_recipe('organization_parameters'), device='cpu')
    groups = _split_fixture_terms(provider)
    parameters = list(model.parameters())
    parameters[0].grad = torch.full_like(parameters[0], .75)
    model.eval()
    provider.coupled_native_reader_mode = 'full'
    core = model.core
    core.set_organization_clock(17, 3)
    original_model = copy.deepcopy(model.state_dict())
    original_provider = copy.deepcopy(provider.training_state_dict())
    original_gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
    before = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
    for fail in (False, True):
        if fail:
            bad_groups = {'flow': ('missing',)}
            with pytest.raises(ValueError, match='partition'):
                audit_loss_families(model, provider, panel_case_ids=('fixture_0', 'fixture_1'), epoch=2,
                                    native_families=bad_groups)
        else:
            audit_loss_families(model, provider, panel_case_ids=('fixture_0', 'fixture_1'), epoch=2,
                                native_families=groups)
        assert _same(original_model, model.state_dict())
        assert _same(original_provider, provider.training_state_dict())
        assert not model.training and provider.coupled_native_reader_mode == 'full'
        assert not hasattr(provider, '_coupled_calibration_panel')
        assert (core._joint_epoch, core._joint_update_index) == (17, 3)
        for param, gradient in zip(parameters, original_gradients):
            assert (param.grad is None) == (gradient is None)
            if gradient is not None:
                torch.testing.assert_close(param.grad, gradient, rtol=0, atol=0)
        assert random.getstate() == before[0]
        assert _same(np.random.get_state(), before[1])
        assert torch.equal(torch.get_rng_state(), before[2])


def test_partial_pair_reports_only_complete_configured_pair_and_no_extra_scene():
    model, provider = build_observation_response_task(scoped_recipe('disabled'), device='cpu')
    groups = _split_fixture_terms(provider)
    result = audit_loss_families(model, provider, panel_case_ids=('fixture_0', 'fixture_1', 'fixture_2'),
                                 epoch=2, native_families=groups)
    assert result['panel_case_ids'] == ['fixture_0', 'fixture_1', 'fixture_2']
    assert result['complete_training_pairs_on_panel'] == [['fixture_0', 'fixture_1']]
    assert ['fixture_2', 'fixture_3'] in result['configured_pairs_with_one_panel_case']
    for output in result['reader_results'].values():
        assert output['response']['pair_denominator'] == 1
        assert len(output['response']['pairs']) == 1


def test_missing_calibration_and_unsupported_panel_fail_explicitly():
    config = scoped_recipe('disabled')
    config['observation_response'].pop('calibration')
    model, provider = build_observation_response_task(config, device='cpu')
    groups = _split_fixture_terms(provider)
    with pytest.raises(ValueError, match='shared epoch-zero'):
        audit_loss_families(model, provider, panel_case_ids=('fixture_0', 'fixture_1'), epoch=2,
                            native_families=groups)


def test_checkpoint_entry_restores_saved_calibration_and_refuses_a_different_common_state(tmp_path):
    from tools.thermal_loss_family_gradient_audit import restore_checkpoint

    common_path = tmp_path / 'common_epoch_zero.pt'
    checkpoint_path = tmp_path / 'epoch_0002_model.pt'
    config = scoped_recipe('disabled')
    config['observation_response']['epoch_zero_checkpoint'] = str(common_path)
    model, provider = build_observation_response_task(config, device='cpu')
    saved_state = copy.deepcopy(provider.training_state_dict())
    common = {'schema': 'coupled_native_epoch_zero_v1', 'epoch': 0,
              'calibration': copy.deepcopy(saved_state['coupled_native']['calibration']),
              'provider_training_state': copy.deepcopy(saved_state)}
    saved = {'epoch': 2, 'experiment_identity': {'recipe': config,
             'provider_identity': dict(provider.identity_payload())},
             'model_state_dict': copy.deepcopy(model.state_dict()), 'provider_training_state': saved_state}
    torch.save(common, common_path)
    torch.save(saved, checkpoint_path)
    restored_model, restored_provider, loaded, actual_path, initial_path = restore_checkpoint(
        checkpoint_path, device='cpu')
    assert (actual_path, initial_path) == (checkpoint_path, common_path)
    assert _same(model.state_dict(), restored_model.state_dict())
    assert _same(saved_state, restored_provider.training_state_dict())
    assert loaded['epoch'] == 2
    common['calibration']['coefficient'] *= 2
    torch.save(common, common_path)
    with pytest.raises(ValueError, match='does not match'):
        restore_checkpoint(checkpoint_path, device='cpu')


def test_next_reader_and_native_samples_match_unaudited_provider():
    model, provider = build_observation_response_task(scoped_recipe('organization_parameters'), device='cpu')
    _, control = build_observation_response_task(scoped_recipe('organization_parameters'), device='cpu')
    groups = _split_fixture_terms(provider)
    audit_loss_families(model, provider, panel_case_ids=('fixture_0', 'fixture_1'), epoch=2,
                        native_families=groups)
    next_draw = provider.begin_native_update(epoch=2, update_index=1, update_count=2)
    control_draw = control.begin_native_update(epoch=2, update_index=1, update_count=2)
    assert {k: v for k, v in next_draw.items() if 'reader_' in k} == {
        k: v for k, v in control_draw.items() if 'reader_' in k}
    assert provider.coupled_native_reader_mode == control.coupled_native_reader_mode
    pair = ('fixture_0', 'fixture_1')
    observed = provider.make_batch(pair, key(provider))
    expected = control.make_batch(pair, key(control))
    for actual, reference in zip(observed.receivers, expected.receivers):
        np.testing.assert_array_equal(actual, reference)
    observed_sample = observed.auxiliary['coupled_native'][0][1]
    expected_sample = expected.auxiliary['coupled_native'][0][1]
    np.testing.assert_array_equal(observed_sample.left_rows, expected_sample.left_rows)
    np.testing.assert_array_equal(observed_sample.right_rows, expected_sample.right_rows)


def test_checkpoint_cli_uses_established_panel_and_both_readers(tmp_path, monkeypatch):
    from channelthermal.training.observation_response import _TRAIN15
    from tools import thermal_loss_family_gradient_audit as entry

    panel = tmp_path / 'panel.json'
    output = tmp_path / 'audit.json'
    checkpoint = tmp_path / 'checkpoint.pt'
    common = tmp_path / 'common.pt'
    panel.write_text(json.dumps({'development': {'train_case_ids': list(_TRAIN15)}}))
    model = torch.nn.Linear(1, 1)
    saved = {'epoch': 100, 'experiment_identity': {'recipe': {'name': 'fixture',
             'dataset': {'name': 'thermal'}, 'observation_response': {'response_update_scope': 'disabled'}}}}
    monkeypatch.setattr(entry, 'restore_checkpoint',
                        lambda path, device: (model, object(), saved, checkpoint, common))
    def audit(actual_model, provider, **kwargs):
        assert actual_model is model
        assert tuple(kwargs['panel_case_ids']) == _TRAIN15
        assert kwargs['epoch'] == 100 and kwargs['sampling_epoch'] == 1
        assert kwargs['read_modes'] == ('restricted', 'full')
        return {'reader_results': {'restricted': {}, 'full': {}}}
    monkeypatch.setattr(entry, 'audit_loss_families', audit)
    arguments = ['--checkpoint', str(checkpoint), '--panels', str(panel), '--output', str(output),
                 '--device', 'cpu', '--cpu-threads', '1']
    assert entry.main(arguments) == 0
    result = json.loads(output.read_text())
    assert set(result['reader_results']) == {'restricted', 'full'}
    assert result['provenance']['response_update_scope'] == 'disabled'
    assert result['provenance']['checkpoint'] == str(checkpoint)
    with pytest.raises(FileExistsError):
        entry.main(arguments)
    panel.write_text(json.dumps({'development': {'train_case_ids': list(reversed(_TRAIN15))}}))
    with pytest.raises(ValueError, match='established input-selected'):
        entry.main(arguments[:-4] + ['--device', 'cpu', '--cpu-threads', '1',
                                     '--output', str(tmp_path / 'other.json')])
