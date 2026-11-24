"""Public observation attachment: native-only capability and coupled fixture."""
import copy
from types import SimpleNamespace

import pytest
import torch

from honf_runtime import observation_response_task as shared
from honf_runtime.observation_blocks import DependencyUnion, QueryDependency


def test_disabled_profile_needs_no_observation_binding_or_pairs(monkeypatch):
    model, provider = object(), object()
    calls = []
    def factory(recipe, *, device, resume_payload):
        calls.append((recipe, device, resume_payload))
        return model, provider, None
    monkeypatch.setitem(shared._CAPABILITIES, 'native_without_observations', factory)
    recipe = {'dataset': 'native_without_observations', 'observation_response': {'enabled': False}}
    saved = {'epoch': 2}
    actual = shared.build_observation_response_task(recipe, device='cpu', resume_payload=saved)
    assert actual == (model, provider)
    assert calls == [(recipe, 'cpu', saved)]
    assert not hasattr(provider, 'auxiliary_loss_names')


def test_seven_field_fixture_repeats_draws_and_executes_same_zero_weight_predictions():
    torch.set_num_threads(1)
    recipe = {'dataset':'analytic_fixture','seed':3,'architecture':'relation_anchored','total_epochs':2,
        'fixture':{'output_dim':7}, 'observation_response':{'variant':'anchors','seeds_per_pair':8,
        'fixture_no_ramp':True, 'calibration':{'coefficients':{
            'extra_absolute':.2,'point_response':.1,'spatial_response':.05}}}}
    model, provider = shared.build_observation_response_task(recipe, device='cpu')
    pair = ('fixture_0','fixture_1')
    values, work, detail = provider.observation_components(model, pair, epoch=1, details=True)
    other = copy.deepcopy(recipe)
    other['observation_response']['variant'] = 'spatial_contrasts'
    other_model, other_provider = shared.build_observation_response_task(other, device='cpu')
    other_model.load_state_dict(model.state_dict())
    second, second_work, second_detail = other_provider.observation_components(other_model, pair, epoch=1, details=True)
    assert detail['draws'].identities() == second_detail['draws'].identities()
    assert detail['union'].dependencies == second_detail['union'].dependencies
    for left, right in zip(detail['predictions'], second_detail['predictions']):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        assert left.shape[-1] == 7
    for name in values:
        torch.testing.assert_close(values[name], second[name], rtol=0, atol=0)
    assert work['observation_response_neural_queries'] == second_work['observation_response_neural_queries']
    columns = set(provider.training_history_columns())
    definitions = provider.training_column_definitions()
    for block in provider.observation_binding.block_names():
        actual = detail['loss_result']['coverage'].get(block, {})
        for metric, source in (('draws', 'draws'), ('spatial_edges', 'edges')):
            name = f'observation_response_block_{block}_{metric}'
            column = 'work_counts.' + name
            assert column in columns and definitions[column]['dtype'] == 'float'
            assert work[name] == actual.get(source, 0)
    sum(values.values()).backward()
    gradient = model.core.member_router.joint_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool((gradient.abs().sum(dim=1) > 0).all())


def test_execution_keeps_scene_frame_and_time_separate_and_restores_union_order():
    calls = []
    weight = torch.nn.Parameter(torch.tensor(1.25))

    def request(model, case_id, dependencies):
        keys = {(dep.scene_id, dep.frame, dep.time) for dep in dependencies}
        assert len(keys) == 1
        key = next(iter(keys))
        calls.append(key)
        offset = (10 if key[1] == 'other_frame' else 0) + key[2]
        query = torch.tensor([dep.coordinates for dep in dependencies])
        return offset, query, torch.tensor([2., 3.]), torch.tensor([4., 5.])

    def decode(scene, query):
        fields = (query[:, :1] * weight + scene).expand(-1, 2)[None]
        return (SimpleNamespace(joint_fields=fields, work_counts={}),
                SimpleNamespace(joint_fields=fields + .25, work_counts={}), {})

    model = SimpleNamespace(core=SimpleNamespace(prepare=lambda scene: scene, decode_paired=decode))
    binding = SimpleNamespace(field_request=request)
    dependencies = tuple(QueryDependency(case, frame, instant, index, (float(index),))
                         for index, (case, frame, instant) in enumerate([
                             ('left', 'physical', 2), ('right', 'physical', 2),
                             ('left', 'other_frame', 2), ('left', 'physical', 3),
                             ('right', 'other_frame', 2), ('left', 'physical', 2)]))
    union = DependencyUnion(dependencies, (), 6, 6)
    actual, work = shared.execute_observation_dependencies(binding, model, ('left', 'right'), union)
    assert calls == [('left', 'physical', 2), ('left', 'other_frame', 2),
                     ('left', 'physical', 3), ('right', 'physical', 2), ('right', 'other_frame', 2)]
    base = torch.tensor([2., 2., 12., 3., 12., 2.]) + weight * torch.arange(6)
    expected = torch.stack([base, base + .25])[:, :, None] * torch.tensor([2., 3.]) + torch.tensor([4., 5.])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), weight)[0], torch.tensor(150.))
    assert work['scene_preparations'] == 5 and work['neural_queries'] == 6
    foreign = DependencyUnion((QueryDependency('unrelated', 'physical', 2, 0, (0.,)),), (), 1, 1)
    with pytest.raises(ValueError, match='outside its declared scene pair'):
        shared.execute_observation_dependencies(binding, model, ('left', 'right'), foreign)
    assert len(calls) == 5


def test_single_frame_execution_matches_original_case_grouping_values_and_gradients():
    torch.set_num_threads(1)
    recipe = {'dataset': 'analytic_fixture', 'seed': 3, 'architecture': 'relation_anchored', 'total_epochs': 2,
              'fixture': {'output_dim': 7}, 'observation_response': {'seeds_per_pair': 8}}
    model, provider = shared.build_observation_response_task(recipe, device='cpu')
    pair = ('fixture_0', 'fixture_1')
    _, _, details = provider.observation_components(model, pair, epoch=1, details=True)
    union = details['union']
    actual, work = shared.execute_observation_dependencies(provider.observation_binding, model, pair, union)
    original = None
    for case_id in pair:
        positions = [i for i, dep in enumerate(union.dependencies) if dep.scene_id == case_id]
        dependencies = tuple(union.dependencies[i] for i in positions)
        scene, query, scale, offset = provider.observation_binding.field_request(model, case_id, dependencies)
        full, restricted, _ = model.core.decode_paired(model.core.prepare(scene), query)
        physical = torch.stack((full.joint_fields[0], restricted.joint_fields[0])) * scale + offset
        if original is None:
            original = physical.new_zeros(actual.shape)
        original = original.index_copy(1, torch.tensor(positions), physical)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    parameters = tuple(model.parameters())
    current_gradient = torch.autograd.grad(actual.square().sum(), parameters, allow_unused=True)
    original_gradient = torch.autograd.grad(original.square().sum(), parameters, allow_unused=True)
    for current, previous in zip(current_gradient, original_gradient):
        if current is None or previous is None:
            assert current is previous
        else:
            torch.testing.assert_close(current, previous, rtol=0, atol=0)
    assert work['scene_preparations'] == 2
