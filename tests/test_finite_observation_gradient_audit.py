"""A panel audit reproduces objective gradients and leaves training state alone."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import finite_observation_gradient_audit as audit
from honf_runtime.coupled_initialization import _same
from honf_runtime.organization_parameters import partition_organization_parameters
from honf_runtime.unified_training import SamplingKey
from tests.test_finite_observation_training import _build, _calibration


def _norm(grads, indices):
    return sum(float(grads[index].double().square().sum()) for index in indices if grads[index] is not None) ** .5


def test_panel_gradient_matches_independent_shared_objective_and_restores_state(monkeypatch):
    model, provider = _build()
    model.eval()
    model.core.set_organization_clock(17, 3)
    provider.finite_observation_reader_mode = 'restricted'
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters[0].grad = torch.full_like(parameters[0], .75)
    original_model, original_provider = copy.deepcopy(model.state_dict()), copy.deepcopy(provider.training_state_dict())
    original_rng = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
    pairs = (('fixture_0', 'fixture_1'), ('fixture_2', 'fixture_3'))
    cases = tuple(case for pair in pairs for case in pair)
    arguments = dict(panel_case_ids=cases, panel_pairs=pairs, epoch=25, variant='disabled',
                     calibration=_calibration(), read_modes=('full',))
    result = audit.audit_finite_observation_gradients(model, provider, provider.observation_binding, **arguments)
    assert _same(original_model, model.state_dict()) and _same(original_provider, provider.training_state_dict())
    assert not model.training and (model.core._joint_epoch, model.core._joint_update_index) == (17, 3)
    assert provider.finite_observation_reader_mode == 'restricted'
    assert not hasattr(provider, '_coupled_calibration_panel')
    torch.testing.assert_close(parameters[0].grad, torch.full_like(parameters[0], .75), rtol=0, atol=0)
    assert random.getstate() == original_rng[0] and _same(np.random.get_state(), original_rng[1])
    assert torch.equal(torch.get_rng_state(), original_rng[2])
    view = result['reader_results']['full']
    assert result['checkpoint_epoch'] == 25 and result['native_sampling_epoch'] == 1
    assert result['calibration']['epoch_ramp'] == .5 and result['optimizer_updates'] == 0
    assert result['calibration']['effective_scales'] == {'anchor': 0., 'contrast': 0.}
    assert result['calibration']['prospective_scales'] == {'anchor': .2, 'contrast': .1}
    assert view['families']['field_6']['available'] is False
    assert view['families']['field_6']['raw_gradient_norms']['global'] == 0.
    assert view['families']['field_0']['raw_gradient_norms']['organization'] > 0
    assert view['families']['field_0']['raw_gradient_norms']['content_and_output'] > 0
    global_group = view['gradient_groups']['global']
    assert global_group['effective_calibrated_ramped_norms'] == {'anchor': 0., 'contrast': 0.}
    assert global_group['combined_native_plus_enabled_finite_norm'] == pytest.approx(global_group['raw_norms']['native'])
    active_arguments = {**arguments, 'variant': 'contrasts'}
    active = audit.audit_finite_observation_gradients(
        model, provider, provider.observation_binding, **active_arguments)['reader_results']['full']

    # Reconstruct the same scalar from the provider's existing observation
    # component API, independently of the audit's per-family accumulation.
    model.train()
    model.core.set_organization_clock(25, 0)
    provider.finite_observation_reader_mode = 'full'
    provider.coupled_native_reader_mode = 'full'
    key = SamplingKey(provider.seed, 1, 0, 0, 'joint', model.architecture,
                      sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id=provider.sampling_dataset_id)
    batches = [provider.make_batch(pair, key) for pair in pairs]
    denominators = provider.loss_denominators(batches, phase='joint', arm=model.architecture)
    native = 0.
    for batch in batches:
        prediction, auxiliary = provider.predict_native(model, provider.make_scene(batch.scene_inputs),
            batch.receivers, model.architecture, 'joint', epoch=25, temperature=1.)
        terms = provider.loss_terms(prediction, batch.targets, 'joint', auxiliary)
        native = native + sum(term.numerator * float(term.weight) / float(denominators[name])
                              for name, term in terms.items())
    native_grad = torch.autograd.grad(native, parameters, allow_unused=True)
    total_anchor, total_contrast, first_family = 0., 0., 0.
    for pair in pairs:
        components, _, observations = provider.finite_observation_components(model, pair, read_mode='full')
        total_anchor = total_anchor + components['anchor'] / len(cases)
        total_contrast = total_contrast + components['contrast'] / len(pairs)
        first, second = (observations[case].families['field_0'].moments for case in pair)
        first_family = first_family + (second - first).square().mean() / len(pairs)
    anchor_grad = torch.autograd.grad(total_anchor, parameters, retain_graph=True, allow_unused=True)
    total_grad = torch.autograd.grad(total_contrast, parameters, retain_graph=True, allow_unused=True)
    family_grad = torch.autograd.grad(first_family, parameters, allow_unused=True)
    named = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    ownership = partition_organization_parameters(model)
    subsets = {'global': list(range(len(named))),
               'organization': [index for index, name in enumerate(named) if name in ownership.organization_names],
               'content_and_output': [index for index, name in enumerate(named) if name in ownership.content_names]}
    for group, indices in subsets.items():
        assert view['gradient_groups'][group]['raw_norms']['native'] == pytest.approx(_norm(native_grad, indices), rel=2.e-6)
        assert view['gradient_groups'][group]['raw_norms']['anchor'] == pytest.approx(_norm(anchor_grad, indices), rel=2.e-6)
        assert view['gradient_groups'][group]['raw_norms']['contrast'] == pytest.approx(_norm(total_grad, indices), rel=2.e-6)
        assert view['families']['field_0']['raw_gradient_norms'][group] == pytest.approx(_norm(family_grad, indices), rel=2.e-6)
        combined = [sum((torch.zeros_like(parameter) if gradient is None else gradient) * factor
                        for gradient, factor in zip(gradients, (1., .2, .1)))
                    for parameter, gradients in zip(parameters, zip(native_grad, anchor_grad, total_grad))]
        assert active['gradient_groups'][group]['combined_native_plus_enabled_finite_norm'] == pytest.approx(
            _norm(combined, indices), rel=2.e-6)
    factor = view['families']['field_0']['configured_weight_fraction']
    assert view['families']['field_0']['weighted_gradient_norms']['global'] == pytest.approx(_norm(family_grad, subsets['global']) * factor)
    json.dumps(result, allow_nan=False)

    # Failure during graph construction also restores counters and RNG.
    model.eval()
    provider_before_failure = copy.deepcopy(provider.training_state_dict())
    rng_before_failure = torch.get_rng_state().clone()
    monkeypatch.setattr(audit, 'execute_finite_dependencies', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('fixture failure')))
    with pytest.raises(RuntimeError, match='fixture failure'):
        audit.audit_finite_observation_gradients(model, provider, provider.observation_binding, **arguments)
    assert not model.training and _same(provider_before_failure, provider.training_state_dict())
    assert torch.equal(torch.get_rng_state(), rng_before_failure)


def test_active_and_unavailable_calibration_are_labeled_without_recalibration():
    calibrated = audit._calibration_info(_calibration(), 'contrasts', 1000)
    assert calibrated['epoch_ramp'] == 1.
    assert calibrated['effective_scales'] == {'anchor': .4, 'contrast': .2}
    legacy = audit._calibration_info(None, 'legacy_native', 500)
    assert legacy['source'] == 'unavailable'
    assert legacy['prospective_scales'] == {'anchor': None, 'contrast': None}
    assert legacy['effective_scales'] == {'anchor': 0., 'contrast': 0.}
    with pytest.raises(ValueError, match='requires its saved calibration'):
        audit._calibration_info(None, 'anchors', 500)


def test_cli_uses_declared_panel_checkpoint_epoch_and_saved_calibration(monkeypatch, tmp_path):
    import channelthermal.training.native_checkpoint as checkpoint_module
    import finite_observation_evaluate as review
    import honf_runtime.compat as compat

    membership = tuple(f'case_{index:03}' for index in range(150))
    cases = membership[:16]
    pairs = tuple((cases[index], cases[index + 1]) for index in range(0, 16, 2))
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps({'development': {'train_case_ids': cases, 'train_pairs': pairs}}))
    model = torch.nn.Linear(1, 1)
    provider = SimpleNamespace(training_case_ids=membership)
    calibration = _calibration()
    saved = {'epoch': 1000, 'experiment_identity': {'recipe': {}, 'provider_identity': {
        'finite_observations': {'variant': 'contrasts', 'basis': 'coordinate', 'probe_count': 64,
                                'probe_seed': 130013}}},
        'provider_training_state': {'finite_observations': {'calibration': calibration}}}
    monkeypatch.setattr(review, '_verify_review_storage', lambda output: {'new_output_budget_bytes': 1000000})
    monkeypatch.setattr(review, 'verify_saved_thermal_observation_identity', lambda binding, saved: {'family_scales': {}})
    monkeypatch.setattr(compat, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    monkeypatch.setattr(checkpoint_module, 'build_native_checkpoint_inference', lambda *args, **kwargs: (model, provider))
    def measure(actual_model, actual_provider, binding, **kwargs):
        assert actual_model is model and actual_provider is provider
        assert kwargs['panel_case_ids'] == cases and kwargs['panel_pairs'] == pairs
        assert kwargs['epoch'] == 1000 and kwargs['calibration'] is calibration
        assert kwargs['variant'] == 'contrasts' and kwargs['probe_count'] == 64
        assert 'not literal pair-coupled training draws' in kwargs['native_sampling_description']
        return {'reader_results': {'full': {}, 'restricted': {}}}
    monkeypatch.setattr(audit, 'audit_finite_observation_gradients', measure)
    written = {}
    monkeypatch.setattr(audit, '_atomic_json', lambda path, payload: written.update(payload))
    assert audit.main(['--checkpoint', str(tmp_path / 'model.pt'), '--panels', str(panel), '--stage', 'development',
                       '--cpu-threads', '1', '--output', str(tmp_path / 'out.json')]) == 0
    assert written['provenance']['training_membership_count'] == 150
