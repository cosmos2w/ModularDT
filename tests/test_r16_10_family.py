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
        def family_components(self, model, source, family):
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
    monkeypatch.setattr(module, '_views', lambda prediction: list(prediction))
    monkeypatch.setattr(module, '_field_components', lambda prediction, batch: {
        name: (value * prediction * torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))
        for name in module.FIELD_ROLES})
    monkeypatch.setattr(module, '_endpoint_functionals', lambda prediction, batch, raw, tau:
        torch.stack([torch.stack([value * v, value * v]) for v in prediction]))
    monkeypatch.setattr(module, 'decision_losses', lambda predictions, truth, sensitivity, research, scales:
        ({'feasibility': predictions.sum() * 0, 'ranking': predictions.sum() * 0, 'near_null': predictions.sum() * 0}, {}))
    model = SimpleNamespace(architecture='joint')
    components, _ = obj.family_components(model, 'original', 'x')
    assert components['endpoint'].item() == pytest.approx(3.)
    # Differences 1,2,3,4: mean square 7.5 for field, J and P.
    assert components['response'].item() == pytest.approx(7.5)
    components['response'].backward()
    assert value.grad.item() == pytest.approx(15., rel=1e-5)
