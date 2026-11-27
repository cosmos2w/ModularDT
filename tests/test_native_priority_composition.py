"""Effective-batch state-priority composition and its real HONF execution."""
import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
from honf_runtime.coupled_response_task import attach_coupled_native_task
from honf_runtime.native_priority_composer import compose_native_priority
from honf_runtime.observation_response_fixture import build_observation_capability
from honf_runtime.observation_response_task import build_observation_response_task
from honf_runtime.organization_parameters import partition_organization_parameters
from honf_runtime.run_layout import RunLayout
from tests.test_coupled_native_integration import key, recipe
from tests.test_relation_anchored_train_entry import cli


def scoped_recipe(scope):
    config = recipe('native_control' if scope == 'disabled' else 'observed_response')
    config['training_profile'] = 'observation_response'
    config['microbatch_size'], config['effective_batch_size'] = 2, 4
    config['gradient_clip'] = 1000000.
    config['execution'] = {'checkpoint_epochs': [1, 2], 'monitor_every': 1,
                           'monitor_epochs': [1, 2], 'progress_enabled': False}
    spec = config['observation_response']
    spec.update(reader_policy='sampled_equal_probability', full_reader_probability=.5,
                response_update_scope=scope)
    if scope != 'disabled':
        spec['response_gradient_composition'] = {
            'kind': 'native_priority_projection', 'maximum_native_norm_ratio': .1}
    return config


def test_parameter_roles_and_tied_value_path_are_resolved_by_identity():
    model, _ = build_observation_response_task(scoped_recipe('organization_parameters'), device='cpu')
    partition = partition_organization_parameters(model)
    organization, content = set(partition.organization_names), set(partition.content_names)
    assert 'core.context_blocks.0.mm.query.weight' in organization
    assert 'core.context_blocks.0.mm.query_norm.weight' in organization
    assert 'core.context_blocks.0.mm.key_norm.weight' in content
    assert 'core.context_blocks.0.mm.value.weight' in content
    assert 'core.member_router.key_M.weight' in organization
    assert 'core.member_router.value_M.weight' in content
    assert any('typed_read_gates' in name for name in organization)
    assert {id(p) for p in partition.organization_parameters}.isdisjoint(
        {id(p) for p in partition.content_parameters})
    assert {id(p) for p in partition.organization_parameters + partition.content_parameters} == {
        id(p) for p in model.parameters() if p.requires_grad}
    attention = model.core.context_blocks[0].mm
    attention.value.weight = attention.query.weight
    tied = partition_organization_parameters(model)
    assert id(attention.query.weight) in {id(p) for p in tied.content_parameters}
    assert id(attention.query.weight) not in {id(p) for p in tied.organization_parameters}
    with_messages = nn.Sequential(SpatialRouteAttentionV2(64, 2, 3, contextual_message=True), nn.Linear(64, 2))
    other = partition_organization_parameters(with_messages)
    assert '0.query_norm.weight' in other.content_names


def test_global_cap_uses_complete_native_norm_and_projection_uses_selected_subset():
    selected = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    content = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    selected.grad = torch.tensor([3., 4.], dtype=torch.float64)
    content.grad = torch.tensor([12.], dtype=torch.float64)
    stats = compose_native_priority((selected, content), (selected,),
                                    (torch.tensor([-6., 2.], dtype=torch.float64),), .1)
    torch.testing.assert_close(selected.grad, torch.tensor([3. - 1.04, 4. + .78], dtype=torch.float64))
    torch.testing.assert_close(content.grad, torch.tensor([12.], dtype=torch.float64))
    assert stats['native_norm_sum'] == pytest.approx(13.)
    assert stats['selected_native_norm_sum'] == pytest.approx(5.)
    assert stats['effective_auxiliary_norm_sum'] == pytest.approx(1.3)
    assert stats['projection_count'] == stats['cap_count'] == stats['updates'] == 1.


def test_zero_selected_native_keeps_auxiliary_when_complete_native_is_nonzero():
    selected = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    content = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    content.grad = torch.tensor([5.], dtype=torch.float64)
    stats = compose_native_priority((selected, content), (selected,),
                                    (torch.tensor([.2, -.1], dtype=torch.float64),), .1)
    torch.testing.assert_close(selected.grad, torch.tensor([.2, -.1], dtype=torch.float64))
    assert stats['selected_native_norm_sum'] == 0
    assert stats['projection_count'] == stats['cap_count'] == 0


@pytest.mark.parametrize('condition', ['zero_native', 'zero_auxiliary', 'zero_projection'])
def test_zero_correction_preserves_missing_adamw_gradient_and_state(condition):
    def run(compose):
        native, inactive = nn.Parameter(torch.tensor([1.])), nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.AdamW([native, inactive], lr=.01, weight_decay=.1)
        native.grad = torch.tensor([0. if condition == 'zero_native' else 1.])
        if compose:
            raw = torch.tensor([-1. if condition == 'zero_projection' else 0.])
            unused = torch.tensor([1. if condition == 'zero_native' else 0.])
            stats = compose_native_priority((native, inactive), (native, inactive), (raw, unused), .1)
            assert stats['effective_auxiliary_norm_sum'] == 0.
            assert inactive.grad is None
        optimizer.step()
        return native.detach().clone(), inactive.detach().clone(), optimizer.state_dict()
    baseline, inactive_baseline, state = run(False)
    actual, inactive_actual, actual_state = run(True)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(inactive_actual, inactive_baseline, rtol=0, atol=0)
    assert actual_state['param_groups'] == state['param_groups']
    assert actual_state['state'].keys() == state['state'].keys()
    for parameter in state['state']:
        for name in state['state'][parameter]:
            torch.testing.assert_close(actual_state['state'][parameter][name], state['state'][parameter][name])


def _nonuniform_fixture(config, monkeypatch):
    import honf_runtime.observation_response_fixture as fixture
    original_reference = fixture._reference
    signs = np.asarray([1., -1., 1., -1., 1., -1., 1.])
    monkeypatch.setattr(fixture, '_reference',
        lambda case, coordinates, width: original_reference(case, coordinates, width) * signs[:width])
    model, provider, binding = build_observation_capability(config, device='cpu')
    population = binding.coupled_native_population
    def with_measures_and_missing(case):
        original = population(case)
        measures = np.broadcast_to(np.arange(1, len(original.keys) + 1)[:, None],
                                   original.measures.shape).copy()
        valid = original.valid.copy()
        if str(case).endswith(('1', '3', '5', '7')):
            valid[:, -1] = False
        return replace(original, measures=measures, valid=valid)
    monkeypatch.setattr(binding, 'coupled_native_population', with_measures_and_missing)
    return attach_coupled_native_task(config, model, provider, binding)


@pytest.mark.parametrize('scope', ['all_parameters', 'organization_parameters'])
def test_real_honf_complete_batch_gradients_and_single_optimizer_step(scope, monkeypatch):
    torch.set_num_threads(1)
    config = scoped_recipe(scope)
    model, provider = _nonuniform_fixture(config, monkeypatch)
    engine = cli.engine_for(config, device='cpu')
    batches = tuple(provider.make_batch(pair, key(provider)) for pair in
                    (('fixture_0', 'fixture_1'), ('fixture_2', 'fixture_3')))
    assert len(batches[0].receivers[0]) != len(batches[0].receivers[1])
    plan = batches[0].auxiliary['coupled_native'][0][1].plan
    assert plan.eligible_mass[-1] == 0 and np.ptp(plan.left.measures[:, 0]) > 0
    assert (plan.right.references[:4] - plan.left.references[:4]).min() < 0
    assert (plan.right.references[:4] - plan.left.references[:4]).max() > 0
    provider.begin_native_update(epoch=1, update_index=0, update_count=2)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    owned = partition_organization_parameters(model)
    subset_ids = ({id(p) for p in owned.organization_parameters} if scope == 'organization_parameters'
                  else {id(p) for p in parameters})
    selected = tuple(p for p in parameters if id(p) in subset_ids)
    native = [torch.zeros_like(p) for p in parameters]
    auxiliary = [torch.zeros_like(p) for p in selected]
    native_used = [False] * len(parameters)
    auxiliary_used = [False] * len(selected)
    denominator = provider.loss_denominators(batches, 'joint', model.architecture)['native']
    for batch in batches:
        prediction, extra = provider.predict_native(model, batch.scene_inputs, batch.receivers,
            model.architecture, 'joint', epoch=1, temperature=1.)
        state_term = provider.loss_terms(prediction, batch.targets, 'joint', extra)['native']
        response = provider.native_microbatch_terms(batch, prediction, extra, epoch=1,
                                                   update_index=0, update_count=2)[0]['coupled_native/response']
        response_grad = torch.autograd.grad(response.numerator * response.weight, selected,
                                            retain_graph=True, allow_unused=True)
        state_grad = torch.autograd.grad(state_term.numerator * state_term.weight / denominator, parameters,
                                         allow_unused=True)
        for accum, gradient, used in ((native, state_grad, native_used), (auxiliary, response_grad, auxiliary_used)):
            for index, gradient_value in enumerate(gradient):
                if gradient_value is not None:
                    accum[index].add_(gradient_value)
                    used[index] = True
    assert any(bool(value.abs().sum()) for value in auxiliary)
    state_vector = torch.cat([value.double().flatten() for value in native])
    subset_vector = torch.cat([value.double().flatten() for value, parameter in zip(native, parameters)
                               if id(parameter) in subset_ids])
    response_vector = torch.cat([value.double().flatten() for value in auxiliary])
    dot, selected_sq = torch.dot(subset_vector, response_vector), torch.dot(subset_vector, subset_vector)
    projection = response_vector - min(0., float(dot)) / float(selected_sq) * subset_vector
    cap = min(1., .1 * float(state_vector.norm()) / float(projection.norm()))
    expected = {}
    offset = 0
    for parameter, state in zip(parameters, native):
        if id(parameter) in subset_ids:
            size = parameter.numel()
            correction = projection[offset:offset + size].reshape(parameter.shape).to(state) * cap
            expected[id(parameter)] = state + correction
            offset += size
        else:
            expected[id(parameter)] = state
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    captured = {}
    original_step = optimizer.step
    def step(*args, **kwargs):
        captured.update({id(p): None if p.grad is None else p.grad.detach().clone() for p in parameters})
        return original_step(*args, **kwargs)
    with patch.object(optimizer, 'step', wraps=step) as optimizer_step, \
         patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
        _, work, _, _ = engine._run_update(model, provider, optimizer, batches, phase='joint',
            arm=model.architecture, epoch=1, update_index=0, epoch_update_count=2)
    assert optimizer_step.call_count == clip.call_count == 1
    for parameter in parameters:
        actual = captured[id(parameter)]
        if actual is None:
            assert torch.count_nonzero(expected[id(parameter)]) == 0
        else:
            torch.testing.assert_close(actual, expected[id(parameter)], rtol=2e-5, atol=2e-7)
    assert work['native_priority_updates'] == 1 and work['coupled_pair_presentations'] == 2
    assert work.get('coupled_reader_full_batches', 0.) + work.get('coupled_reader_restricted_batches', 0.) == 1


def test_sampled_native_disabled_skips_response_extraction_and_matches_reader_draws(monkeypatch):
    torch.set_num_threads(1)
    native_model, native = _nonuniform_fixture(scoped_recipe('disabled'), monkeypatch)
    response_model, response = build_observation_response_task(scoped_recipe('all_parameters'), device='cpu')
    response_model.load_state_dict(native_model.state_dict())
    for index in range(8):
        native.begin_native_update(epoch=1, update_index=index, update_count=8)
        response.begin_native_update(epoch=1, update_index=index, update_count=8)
        assert native.coupled_native_reader_mode == response.coupled_native_reader_mode
    batch = native.make_batch(('fixture_0', 'fixture_1'), key(native))
    prediction, auxiliary = native.predict_native(native_model, batch.scene_inputs, batch.receivers,
        native_model.architecture, 'joint', epoch=1, temperature=1.)
    native.coupled_native_estimates = lambda *args: pytest.fail('Disabled response was extracted')
    # Patch the binding method used by the closure itself.
    with patch('honf_runtime.coupled_response_task.coupled_response_loss', side_effect=AssertionError):
        terms, work = native.native_microbatch_terms(batch, prediction, auxiliary,
                                                     epoch=1, update_index=0, update_count=8)
    assert terms == {} and work['coupled_pair_presentations'] == 1
    assert native.native_reuse_loss_names == ()


@pytest.mark.parametrize('change', [
    {'reader_policy': 'sampled_equal_probability', 'full_reader_probability': .25},
    {'reader_policy': 'other'},
    {'response_update_scope': 'unknown'},
    {'response_gradient_composition': {'kind': 'native_priority_projection', 'maximum_native_norm_ratio': 0}},
])
def test_new_policy_rejects_incompatible_configuration(change):
    config = scoped_recipe('organization_parameters')
    config['observation_response'].update(change)
    with pytest.raises(ValueError):
        build_observation_response_task(config, device='cpu')


def test_projected_loss_curve_and_recording_are_identified_as_nominal(tmp_path, monkeypatch):
    import matplotlib.axes
    from honf_runtime.unified_training import _render_loss_curves
    model, provider = build_observation_response_task(scoped_recipe('organization_parameters'), device='cpu')
    titles = []
    original = matplotlib.axes.Axes.set_title
    def capture(axis, title, *args, **kwargs):
        titles.append(title)
        return original(axis, title, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, 'set_title', capture)
    row = {'epoch': 1, 'phase': 'joint', 'arm': model.architecture,
           'train_losses': {'native': 1., 'coupled_native/response': .1}}
    _render_loss_curves([row], tmp_path, 'field_score', provider.loss_metadata(),
                        provider.identity_payload(), separate_totals=True)
    assert 'Nominal TRAIN loss sum' in titles
    metadata = json.loads((tmp_path / 'loss_curves_metadata.json').read_text())
    assert 'nominal monitoring only' in metadata['panel_definitions']['weighted_loss_totals']


def test_explicit_disabled_matches_legacy_native_gradients_with_same_reader(monkeypatch):
    torch.set_num_threads(1)
    config = scoped_recipe('disabled')
    legacy_config = recipe('native_control')
    legacy_config.update(training_profile='observation_response', microbatch_size=2, effective_batch_size=4,
                         gradient_clip=1000000.)
    legacy_config['observation_response']['full_reader_probability'] = .5
    new_model, new_provider = build_observation_response_task(config, device='cpu')
    legacy_model, legacy = build_observation_response_task(legacy_config, device='cpu')
    legacy_model.load_state_dict(new_model.state_dict())
    batches = [tuple(owner.make_batch(pair, key(owner)) for pair in
                     (('fixture_0', 'fixture_1'), ('fixture_2', 'fixture_3')))
               for owner in (new_provider, legacy)]
    new_provider.begin_native_update(epoch=1, update_index=0, update_count=2)
    mode = new_provider.coupled_native_reader_mode
    legacy_begin = legacy.begin_native_update
    def begin(**kwargs):
        counts = legacy_begin(**kwargs)
        legacy.coupled_native_reader_mode = mode
        return counts
    monkeypatch.setattr(legacy, 'begin_native_update', begin)
    for model, provider, engine_config, current_batches in (
        (new_model, new_provider, config, batches[0]),
        (legacy_model, legacy, legacy_config, batches[1]),
    ):
        engine = cli.engine_for(engine_config, device='cpu')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        engine._run_update(model, provider, optimizer, current_batches, phase='joint', arm=model.architecture,
                           epoch=1, update_index=0, epoch_update_count=2)
    for actual, expected in zip(new_model.parameters(), legacy_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_no_physical_correspondence_keeps_native_update_and_zero_auxiliary():
    torch.set_num_threads(1)
    config = scoped_recipe('organization_parameters')
    model, provider, binding = build_observation_capability(config, device='cpu')
    original = binding.coupled_native_population
    def population(case):
        value = original(case)
        if str(case) == 'fixture_1':
            return replace(value, keys=(None,) * len(value.keys))
        return value
    binding.coupled_native_population = population
    _, provider = attach_coupled_native_task(config, model, provider, binding)
    assert ('fixture_0', 'fixture_1') not in provider.coupled_native_pairs
    batch = provider.make_batch(('fixture_0', 'fixture_1'), key(provider))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    losses, work, _, _ = cli.engine_for(config, device='cpu')._run_update(
        model, provider, optimizer, (batch,), phase='joint', arm=model.architecture,
        epoch=1, update_index=0, epoch_update_count=2)
    assert losses['native'] > 0 and losses['coupled_native/response'] == 0
    assert work['native_priority_auxiliary_norm_sum'] == 0
    assert work['native_priority_effective_auxiliary_norm_sum'] == 0


def test_checkpointed_tile_sizes_produce_close_effective_batch_gradients():
    torch.set_num_threads(1)
    config = scoped_recipe('organization_parameters')
    first_model, first_provider = build_observation_response_task(config, device='cpu')
    second_model, second_provider = build_observation_response_task(config, device='cpu')
    second_model.load_state_dict(first_model.state_dict())
    first_model.core.query_tile_size, second_model.core.query_tile_size = 2, 32
    results = []
    for model, provider in ((first_model, first_provider), (second_model, second_provider)):
        batches = tuple(provider.make_batch(pair, key(provider)) for pair in
                        (('fixture_0', 'fixture_1'), ('fixture_2', 'fixture_3')))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        gradients = []
        def capture_step():
            gradients.extend(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                             for p in model.parameters())
        with patch.object(optimizer, 'step', side_effect=capture_step):
            losses, work, _, _ = cli.engine_for(config, device='cpu')._run_update(
                model, provider, optimizer, batches, phase='joint', arm=model.architecture,
                epoch=1, update_index=0, epoch_update_count=2)
        results.append((losses, work, torch.cat([value.flatten() for value in gradients])))
    relative_error = (results[0][2] - results[1][2]).norm() / results[1][2].norm()
    assert float(relative_error) < 2e-5
    assert results[0][0]['coupled_native/response'] == pytest.approx(
        results[1][0]['coupled_native/response'], rel=2e-5, abs=1e-7)


def _recovery_worker(output, config_path, stop, resume):
    import random
    import honf_runtime.unified_training as runtime
    torch.set_num_threads(1)
    config = json.loads(Path(config_path).read_text())
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    runtime._render_loss_curves = lambda *args, **kwargs: None
    model, provider = build_observation_response_task(config, device='cpu')
    engine = cli.engine_for(config, device='cpu')
    layout = RunLayout(Path(output))
    if not resume:
        config_destination = Path(output) / 'configs' / 'resolved_config.json'
        config_destination.parent.mkdir(parents=True, exist_ok=True)
        config_destination.write_text(json.dumps(config))
    engine.fit(model, provider, Path(output), identity={'run': 'native-priority-recovery'},
        arm=model.architecture, stop_after=int(stop),
        resume_checkpoint=layout.read_path('latest_model.pt') if resume else None)


def test_fresh_process_saved_configuration_resume_and_recording(tmp_path):
    config = scoped_recipe('organization_parameters')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    code = ('import sys; from tests.test_native_priority_composition import _recovery_worker; '
            '_recovery_worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1")')
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(root / 'src'), str(root), str(root / 'tools'),
        str(root / 'Case_ThermalChannel/src'), str(root / 'Case_WindFarm/src'), os.environ.get('PYTHONPATH', '')))}
    def run(name, path, stop, resume):
        subprocess.run([sys.executable, '-c', code, str(tmp_path / name), str(path), str(stop), str(int(resume))],
                       check=True, env=env, stdout=subprocess.DEVNULL)
    run('full', config_path, 2, False)
    run('resumed', config_path, 1, False)
    saved_config = tmp_path / 'resumed/configs/resolved_config.json'
    run('resumed', saved_config, 2, True)
    full, resumed = RunLayout(tmp_path / 'full'), RunLayout(tmp_path / 'resumed')
    left = torch.load(full.read_path('latest_model.pt'), weights_only=False)
    right = torch.load(resumed.read_path('latest_model.pt'), weights_only=False)
    from honf_runtime.coupled_initialization import _same
    for name in ('model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
                 'provider_training_state', 'best_field_score'):
        assert _same(left[name], right[name]), name
    definitions = [json.loads((tmp_path / name / 'configs/metric_definitions.json').read_text())
                   for name in ('full', 'resumed')]
    assert definitions[0] == definitions[1]
    assert 'Nominal' in definitions[0]['training_column_definitions']['train_loss_total']['meaning']
    histories = []
    for layout in (full, resumed):
        with layout.read_path('history.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            histories.append((reader.fieldnames, list(reader)))
    assert histories[0][0] == histories[1][0] == definitions[0]['training_columns']
    for left_row, right_row in zip(histories[0][1], histories[1][1]):
        for name in histories[0][0]:
            if name not in ('train_seconds', 'work_counts.coupled_plan_build_seconds'):
                assert left_row[name] == right_row[name], name
