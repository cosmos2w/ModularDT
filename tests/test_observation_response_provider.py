"""Public observation attachment: native-only capability and coupled fixture."""
import copy

import torch

from honf_runtime import observation_response_task as shared


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
