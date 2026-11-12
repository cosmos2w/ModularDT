import copy
from types import SimpleNamespace

import pytest
import torch

from channelthermal.training.r16_10_corpus import EligibleFamilySchedule
from channelthermal.training.r16_10_family_task import auxiliary_ramp, attach_family_objective


def test_intersection_cycles_and_exact_resume():
    native = [str(i) for i in range(40)]
    scheduler = EligibleFamilySchedule([str(i) for i in range(15)] + ['excluded'],
                                       [str(i) for i in range(15, 40)] + ['excluded'], native)
    assert auxiliary_ramp(10) == 0 and auxiliary_ramp(11) == .025 and auxiliary_ramp(50) == 1
    assert len(scheduler.families['original']) == 15 and len(scheduler.families['new']) == 25
    for epoch in range(11, 31):
        rows = scheduler.epoch(epoch, {'original': 1, 'new': 2})
        assert rows == scheduler.epoch(epoch, {'original': 1, 'new': 2})
        assert [s for s, _ in rows] == ['original', 'new', 'new']
    saved = scheduler.state_dict()
    restored = EligibleFamilySchedule([str(i) for i in range(15)] + ['excluded'],
                                       [str(i) for i in range(15, 40)] + ['excluded'], native)
    restored.load_state_dict(saved)
    for epoch in range(31, 60):
        assert scheduler.epoch(epoch, {'original': 1, 'new': 2}) == restored.epoch(epoch, {'original': 1, 'new': 2})
    assert scheduler.state_dict() == restored.state_dict()
    broken = copy.deepcopy(saved); broken['cursor']['new'] += 1
    with pytest.raises(ValueError, match='counts'):
        restored.load_state_dict(broken)


def test_auxiliary_epoch_mean_matches_short_batch_update_mean():
    scheduler = EligibleFamilySchedule(['a'], ['b', 'c'], ['a', 'b', 'c'])
    parameter = torch.tensor(2., requires_grad=True)
    class Objective:
        schedule = scheduler
        response_scales = {}; sensitivity = {}; research_scales = {}
        def family_components(self, model, source, family, *, components=None):
            value = {'a': 1., 'b': 2., 'c': 3.}[family]
            return {'endpoint': parameter * value, 'response': parameter * 2 * value,
                    'decision': parameter * 3 * value}, {}
    provider = SimpleNamespace(identity_payload=lambda: {})
    attach_family_objective(provider, Objective(), arm='D', coefficients={'endpoint': 1., 'response': 1., 'decision': 1.},
                            families_per_epoch={'original': 1, 'new': 2})
    for epoch in range(11, 50):
        scheduler.epoch(epoch, {'original': 1, 'new': 2})
    # Four native update means, including a short last batch, are averaged equally.
    epoch_losses = []
    for index in range(4):
        terms = list(provider.auxiliary_update_terms(None, epoch=50, update_index=index, update_count=4))
        epoch_losses.append(sum(t.numerator * t.weight / t.denominator for row, _ in terms for t in row.values()))
    value = sum(epoch_losses) / 4
    assert value.item() == pytest.approx(2. * 2. * (1 + 2 + 3))
    value.backward()
    assert parameter.grad.item() == pytest.approx(12.)
    saved = provider.training_state_dict()
    provider.load_training_state_dict(saved)


def test_five_endpoint_objective_and_response_gradient(monkeypatch):
    import channelthermal.training.r16_10_family_task as module
    import channelthermal.training.thermal_family_objective as family_math
    import channelthermal.training.r16_5_decision_task as decisions
    value = torch.tensor(1., requires_grad=True)
    class Provider:
        native_weights = {name: 1. for name in module.FIELD_ROLES}
        def with_raw_port_targets(self, batch, raw):
            return batch
        def predict_native(self, model, scene, receivers, *args, **kwargs):
            return (scene, scene), {}
        def endpoint_state_loss(self, prediction, batch):
            return value * batch.scene_inputs
    provider = Provider()
    corpus = SimpleNamespace(family_ids=lambda role: ('x',))
    obj = module.R1610FamilyObjective(provider, corpus, corpus, ['x'], {},
        response_scales={**{name: {'scale': 1.} for name in module.FIELD_ROLES}, 'J': {'scale': 1.}, 'P': {'scale': 1.}},
        sensitivity={'J': 1., 'P': 1.}, research_scales={'J': 1., 'P': 1.})
    obj.actions = lambda source, family: ('a', 'b', 'c', 'd')
    obj.label = lambda source, family, action: {'J': 0., 'P': 0.}
    def batch(x):
        return SimpleNamespace(scene_inputs=float(x), receivers=None)
    obj.pair = lambda source, family, action: {'baseline': batch(1), 'edited': batch(2 + 'abcd'.index(action)),
        'raw_baseline': {}, 'raw_edited': {}}
    monkeypatch.setattr(family_math, '_views', lambda prediction: list(prediction))
    monkeypatch.setattr(family_math, '_field_components', lambda prediction, batch: {
        name: (value * prediction * torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))
        for name in module.FIELD_ROLES})
    monkeypatch.setattr(family_math, '_endpoint_functionals', lambda prediction, batch, raw, tau:
        torch.stack([torch.stack([value * v, value * v]) for v in prediction]))
    monkeypatch.setattr(decisions, 'decision_losses', lambda predictions, truth, sensitivity, research, scales:
        ({'feasibility': predictions.sum() * 0, 'ranking': predictions.sum() * 0, 'near_null': predictions.sum() * 0}, {}))
    model = SimpleNamespace(architecture='joint')
    components, _ = obj.family_components(model, 'original', 'x', batched=False)
    assert components['endpoint'].item() == pytest.approx(3.)
    # Differences 1,2,3,4: mean square 7.5 for field, J and P.
    assert components['response'].item() == pytest.approx(7.5)
    components['response'].backward()
    assert value.grad.item() == pytest.approx(15., rel=1e-5)


def test_batch_family_preserves_endpoint_masks_and_gradients():
    from channelthermal.training.r16_10_family_task import _stack_family_batches, _split_family_prediction
    from channelthermal.training.content_hypergraph_task import ThermalContentSceneInputs, ThermalContentReceivers
    from channelthermal.training.r16_10_native_task import NativePortTargets
    from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
    from honf_runtime.unified_training import TaskBatch
    def batch(i):
        x = torch.tensor([[float(i)]])
        target = NativePortTargets(case_ids=(str(i),), case_indices=(0,), epoch=0,
            field_targets=x, point_weights=x, interface_target=x, interface_valid_mask=x,
            material_targets=x, outside_temperature=x, outside_valid=x)
        return TaskBatch(scene_inputs=ThermalContentSceneInputs({'value': x}, case_ids=(str(i),)),
            receivers=ThermalContentReceivers(x, x, x), targets=target,
            auxiliary={}, case_keys=(str(i),))
    combined = _stack_family_batches([batch(i) for i in range(5)])
    assert combined.scene_inputs.case_ids == ('0', '1', '2', '3', '4')
    assert combined.targets.outside_temperature[:, 0].tolist() == [0., 1., 2., 3., 4.]
    source = torch.arange(5., dtype=torch.float64, requires_grad=True).reshape(5, 1)
    prepared = SimpleNamespace(source_present=torch.ones(5, 1), stencils={'fluid': SimpleNamespace(valid=torch.tensor([[1], [0], [1], [0], [1]]))})
    branch = SimpleNamespace(native_prepared=prepared, native_main={'pred_field': source})
    outputs = _split_family_prediction(ThermalJointPairPredictions(branch, branch, {}), 5)
    assert outputs[1].full.native_prepared.stencils['fluid'].valid.item() == 0
    sum(row.full.native_main['pred_field'].sum() * i for i, row in enumerate(outputs)).backward()
    assert torch.equal(source.grad if source.is_leaf else source._base.grad, torch.arange(5., dtype=torch.float64))


def test_active_family_components_match_full_values_and_gradients(monkeypatch):
    """E and R skip unrelated work without changing their objective gradients."""
    import channelthermal.training.r16_10_family_task as legacy
    import channelthermal.training.thermal_family_objective as math_module
    import channelthermal.training.r16_5_decision_task as decisions

    parameter = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    calls = {'fields': 0, 'functionals': 0, 'decision': 0}
    class Provider:
        native_weights = {name: 1. for name in legacy.FIELD_ROLES}
        def with_raw_port_targets(self, batch, raw): return batch
        def predict_native(self, model, scene, receivers, *args, **kwargs): return (scene, scene), {}
        def endpoint_state_loss(self, prediction, batch): return (parameter * batch.scene_inputs).square()
    corpus = SimpleNamespace(family_ids=lambda role: ('x',))
    objective = legacy.R1610FamilyObjective(Provider(), corpus, corpus, ['x'], {},
        response_scales={**{name: 1. for name in legacy.FIELD_ROLES}, 'J': 1., 'P': 1.},
        sensitivity={'J': 1., 'P': 1.}, research_scales={'J': 1., 'P': 1.})
    objective.actions = lambda source, family: ('a', 'b', 'c', 'd')
    objective.label = lambda source, family, action: {'J': .2, 'P': -.1}
    def batch(value): return SimpleNamespace(scene_inputs=float(value), receivers=None)
    objective.pair = lambda source, family, action: {'baseline': batch(1),
        'edited': batch(2 + 'abcd'.index(action)), 'raw_baseline': {}, 'raw_edited': {}}
    monkeypatch.setattr(math_module, '_views', lambda prediction: list(prediction))
    def fields(prediction, batch):
        calls['fields'] += 1
        return {name: (parameter * prediction * torch.ones(1, 1, dtype=torch.float64),
                       torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.bool))
                for name in legacy.FIELD_ROLES}
    def functionals(prediction, batch, raw, tau):
        calls['functionals'] += 1
        return torch.stack([torch.stack((parameter * x, parameter * x / 2)) for x in prediction])
    def decision(values, *args):
        calls['decision'] += 1
        term = values.square().mean()
        return {'feasibility': term, 'ranking': term / 2, 'near_null': term / 4}, {}
    monkeypatch.setattr(math_module, '_field_components', fields)
    monkeypatch.setattr(math_module, '_endpoint_functionals', functionals)
    monkeypatch.setattr(decisions, 'decision_losses', decision)
    model = SimpleNamespace(architecture='joint')
    complete, _ = objective.family_components(model, 'original', 'x', batched=False)
    reference_values = {key: float(value.detach()) for key, value in complete.items()}
    expected = {}
    for names in (('endpoint',), ('endpoint', 'response'), ('endpoint', 'response', 'decision')):
        expected[names] = float(torch.autograd.grad(sum(complete[name] for name in names), parameter,
                                                    retain_graph=True)[0])
    for names in expected:
        before = dict(calls)
        selected, _ = objective.family_components(model, 'original', 'x', batched=False, components=names)
        assert set(selected) == set(names)
        for name in names:
            assert float(selected[name].detach()) == pytest.approx(reference_values[name])
        gradient = float(torch.autograd.grad(sum(selected.values()), parameter)[0])
        assert gradient == pytest.approx(expected[names])
        if 'response' not in names:
            assert calls['fields'] == before['fields'] and calls['functionals'] == before['functionals']
        if 'decision' not in names:
            assert calls['decision'] == before['decision']
