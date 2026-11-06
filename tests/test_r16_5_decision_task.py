"""Focused tests of R16-5 decision losses and exact full-field replay."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from channelthermal.native_functionals import native_design_functionals
from channelthermal.training import r16_5_decision_task as task
from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
from channelthermal.training.thermal_state_response_task import ACTIONS, FIELD_ROLES, predict_response_pair


SENSITIVITY = {'J': .025, 'P': .0125}
SCALES = {'J': 1., 'P': .5}
RESPONSE_SCALES = {'J': 1.5, 'P': .7}
LABELS = [[-.8, .01], [-.4, -.25], [.5, .3], [0., 0.]]


def test_truth_consistency_equal_view_and_class_reductions():
    physical = torch.tensor(LABELS, dtype=torch.float64)
    base = torch.tensor([[3., 2.], [8., -3.]], dtype=torch.float64)
    predictions = torch.cat((base[:, None], base[:, None] + physical[None]), dim=1).requires_grad_()
    losses, counts = task.decision_losses(predictions, physical, SENSITIVITY, SCALES, RESPONSE_SCALES)
    assert all(float(loss) == pytest.approx(0., abs=1.e-28) for loss in losses.values())
    assert counts['requests'] == 12
    assert counts['ambiguous_actions'] > 0
    assert counts['ranking_pairs'] > 0
    assert counts['null_J'] == 2 and counts['null_P'] == 4
    assert all(value == 0 for name, value in counts.items() if name.startswith('active_'))
    perturbed = predictions.detach().clone()
    perturbed[:, 1:, :] += torch.tensor([[1., .8], [.8, -.6], [-1., -.8], [.5, .4]])
    perturbed.requires_grad_()
    both, active = task.decision_losses(perturbed, physical, SENSITIVITY, SCALES, RESPONSE_SCALES)
    assert active['active_feasible_hinges'] > 0
    assert active['active_infeasible_hinges'] > 0
    assert active['active_ranking_hinges'] > 0
    assert active['active_null_J_hinges'] > 0 and active['active_null_P_hinges'] > 0
    individual = [task.decision_losses(perturbed[i:i + 1], physical, SENSITIVITY, SCALES,
                                       RESPONSE_SCALES)[0] for i in range(2)]
    for name in both:
        torch.testing.assert_close(both[name], (individual[0][name] + individual[1][name]) / 2)
    grad, = torch.autograd.grad(sum(both.values()), perturbed)
    assert grad[:, 0].abs().sum() > 0
    assert all(float(grad[:, i].abs().sum()) > 0 for i in range(1, 5))
    torch.testing.assert_close(grad.sum(dim=1), torch.zeros_like(grad[:, 0]), atol=1.e-12, rtol=0)


def test_empty_resolved_groups_contribute_zero():
    sensitivity = {'J': 1., 'P': 1.}
    predicted = torch.zeros((1, 5, 2), dtype=torch.float64, requires_grad=True)
    losses, counts = task.decision_losses(predicted, torch.zeros((4, 2)), sensitivity,
                                          SCALES, RESPONSE_SCALES)
    assert counts['ranking_pairs'] == 0
    assert all(float(v) == 0 for v in losses.values())
    grads, = torch.autograd.grad(sum(losses.values()), predicted)
    assert torch.count_nonzero(grads) == 0


def test_research_metadata_uses_only_resolved_original_fit_actions():
    labels = {(f'{family:04d}', action): {'J': family + .2 * i, 'P': (family + i) * .03}
              for family in range(48) for i, action in enumerate(ACTIONS)}
    meta = task.decision_research_metadata(labels, SENSITIVITY, RESPONSE_SCALES)
    for name in ('J', 'P'):
        values = np.array([abs(row[name]) for row in labels.values()])
        expected = np.quantile(values[values > 4 * SENSITIVITY[name]], .25)
        assert meta['research_scales'][name] == expected
        assert sum(meta['truth_counts'][name][key] for key in ('feasible', 'infeasible', 'ambiguous')) == 192
    assert meta['fit_families'] == 48 and meta['fit_actions'] == 192


class _StochasticFields(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.9, -.4, .3], dtype=torch.float64))

    def forward(self, design):
        x = torch.linspace(0, 1, 13, dtype=torch.float64)
        material_x = torch.linspace(.1, .9, 7, dtype=torch.float64)
        mask = torch.tensor([True, True, True, False, True, True, True])
        fluid_mask = torch.tensor([True, True, True, True, True, False, True, True, True, True, True, True, True])
        views = []
        for scale in (1., .7):
            noise = torch.rand((), dtype=torch.float64) * .1
            material = self.weight[0] * material_x.square() + design * self.weight[1] + noise
            pressure = self.weight[2] * x + design * self.weight[0] * x.square() + noise * x
            views.append(torch.stack(native_design_functionals(material * scale, mask, pressure * scale,
                                                               fluid_mask, x, 1., .2)))
        return torch.stack(views)


def _same_rng(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2]) and a[3] is b[3] is None)


def test_scalar_replay_matches_full_field_retained_graph_and_rng():
    torch.manual_seed(14)
    model = _StochasticFields()
    initial_rng = task._rng_snapshot('cpu')
    designs = [-.2, .3, -.5, .8, -.9]
    snapshots, originals = [], []
    for design in designs:
        snapshots.append(task._rng_snapshot('cpu'))
        originals.append(model(design))
    end_rng = task._rng_snapshot('cpu')
    full = torch.stack(originals, dim=1)
    losses, _ = task.decision_losses(full, LABELS, SENSITIVITY, SCALES, RESPONSE_SCALES)
    expected = sum(task.DECISION_WEIGHTS[k] * v for k, v in losses.items())
    retained_gradient, = torch.autograd.grad(expected, model.weight)
    pooled, vjp, counts = task.pooled_scalar_gradients(full, LABELS, SENSITIVITY, SCALES, RESPONSE_SCALES)
    assert counts['ranking_pairs'] > 0
    for key in losses:
        torch.testing.assert_close(pooled[key], losses[key].detach(), rtol=0, atol=0)
    replay_gradient = torch.zeros_like(model.weight)
    for i, design in enumerate(designs):
        output, surrogate, difference, _ = task.replay_endpoint(
            lambda design=design: (model(design), {}), full[:, i].detach(), vjp[:, i], snapshots[i], 'cpu')
        assert float(difference.max()) == 0
        contribution, = torch.autograd.grad(surrogate, model.weight)
        replay_gradient += contribution
        assert _same_rng(end_rng, task._rng_snapshot('cpu'))
        del output, surrogate
    torch.testing.assert_close(replay_gradient, retained_gradient, rtol=1.e-12, atol=1.e-12)
    task._set_rng(initial_rng, 'cpu')


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self._joint_update_index = 0
        self._joint_epoch = 0
        self.reader_capability = 'anchored'
        self.clocks = []

    def set_training_clock(self, epoch, index):
        self._joint_epoch = epoch
        self._joint_update_index = index
        self.clocks.append((epoch, index))


class _Model(nn.Module):
    architecture = 'tiny'

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.9, -.4, .3], dtype=torch.float64))
        self.core = _Core()


def _batch(design, name):
    n, k, a = 9, 5, 4
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    scene = SimpleNamespace(design=design, case_ids=(name,))
    receivers = SimpleNamespace(fluid_xy=torch.stack((x, x * 0), dim=-1)[None])
    targets = SimpleNamespace(field_targets=torch.zeros((1, n, 5), dtype=torch.float64),
                              point_weights=torch.ones((1, n), dtype=torch.bool),
                              material_targets=torch.zeros((1, 1, k), dtype=torch.float64),
                              interface_target=torch.zeros((1, 1, a, 2), dtype=torch.float64),
                              interface_valid_mask=torch.ones((1, 1, a), dtype=torch.bool))
    return SimpleNamespace(scene_inputs=scene, receivers=receivers, targets=targets)


def _fake_forward(self, model, scene, receivers, execution_mode, phase, **kwargs):
    x = receivers.fluid_xy[..., 0]
    d = float(scene.design)
    noise = torch.rand((), dtype=torch.float64) * .1
    p = model.weight[0] * x + d * model.weight[2] * x.square() + noise * x
    field = torch.stack((p, p * .5, p, p * .2, p + 1), dim=-1)
    material = (model.weight[1] * torch.linspace(.1, 1, 5, dtype=torch.float64) + d + noise)[None, None, :, None]
    interface = (model.weight[0] * torch.linspace(.1, .9, 4, dtype=torch.float64) + d + noise)[None, None, :, None]
    interface = torch.cat((interface, interface * .2), dim=-1)
    stencil = lambda shape: SimpleNamespace(valid=torch.ones(shape, dtype=torch.bool))
    prep = SimpleNamespace(stencils={'fluid': stencil(x.shape), 'material': stencil(material.shape[:-1]),
                                     'surface': stencil(interface.shape[:-1]),
                                     'outside': stencil(interface.shape[:-1])},
                           source_present=torch.ones((1, 1), dtype=torch.bool))
    def branch(scale):
        return SimpleNamespace(native_prepared=prep, native_main={
            'pred_field': field * scale, 'pred_internal_temperature': material * scale,
            'pred_interface': interface * scale})
    return ThermalJointPairPredictions(branch(1.), branch(.8), {}), {'work': {}}


class _Corpus:
    def family_ids(self, role):
        assert role == 'fit'
        return ('family', 'unused1', 'unused2', 'unused3')

    def fit_families_for_epoch(self, epoch, seed):
        assert epoch == 5102
        return ('family', 'unused1', 'unused2', 'unused3')


def _make_provider(model):
    provider = task.ThermalR165DecisionTask.__new__(task.ThermalR165DecisionTask)
    provider.model = model
    provider.device = torch.device('cpu')
    provider.source_parent_age = 5100
    provider.seed = 0
    provider.native_weights = {name: 1. for name in FIELD_ROLES}
    provider.response_scales = {name: 1. for name in (*FIELD_ROLES, 'J', 'P')}
    provider.response_labels = {('family', action): {'J': LABELS[i][0], 'P': LABELS[i][1]}
                                for i, action in enumerate(ACTIONS)}
    provider.response_spec = {'tau': .2}
    provider.response_corpus = _Corpus()
    provider.decision_spec = {'arm': 'PDeltaD'}
    provider.decision_sensitivity = SENSITIVITY
    provider.decision_scales = SCALES
    provider._decision_epoch = None
    provider._response_epoch = None
    provider._response_epoch_families = []
    provider._response_epoch_actions = []
    provider._snapshot_epoch = None
    provider._clock_epoch = None
    from channelthermal.training.thermal_state_response_task import _field_components
    def role_losses(branch, targets):
        return {name: values[0][values[2]].square().mean().reshape(1)
                for name, values in _field_components(branch, SimpleNamespace(targets=targets)).items()}
    provider._role_losses = role_losses
    return provider


def test_combined_fit_generator_replays_true_losses_and_clocks(monkeypatch):
    monkeypatch.setattr(task._PairedMemberRoutingTask, 'predict_native', _fake_forward)
    def pair(provider, corpus, family, action, role):
        index = ACTIONS.index(action)
        raw = {'config': {'domain': {'lx': 1.}}}
        return {'baseline': _batch(0., 'baseline'), 'edited': _batch((index + 1) * .2, action),
                'raw_baseline': raw, 'raw_edited': raw}
    monkeypatch.setattr(task, 'response_pair', pair)
    model = _Model()
    provider = _make_provider(model)
    torch.manual_seed(45)
    original = []
    for action in ACTIONS:
        paired = pair(provider, provider.response_corpus, 'family', action, 'fit')
        losses = predict_response_pair(task._DirectResponseProvider(provider, 2), model, paired,
            provider.response_scales, provider.response_labels[('family', action)], .2)
        original.append(losses)
    values = torch.stack([torch.stack([torch.stack(v['functionals'][index])
        for v in entry['observations']]) for entry, index in
        [(original[0], 0), *((entry, 1) for entry in original)]], dim=1)
    terms, _ = task.decision_losses(values, LABELS, SENSITIVITY, SCALES, {'J': 1., 'P': 1.})
    common = sum(.25 * v['endpoint'] + .08 * v['field_delta']
                 + .01 * v['J_delta'] + .01 * v['P_delta'] for v in original) / 4
    objective = common + sum(task.DECISION_WEIGHTS[name] * term for name, term in terms.items())
    expected, = torch.autograd.grad(objective, model.weight)
    expected_rng = task._rng_snapshot('cpu')
    expected_logged = {f'decision/{name}': float(term.detach()) * task.DECISION_WEIGHTS[name]
                       for name, term in terms.items()}
    torch.manual_seed(45)
    provider._clock_epoch = None
    model.weight.grad = None
    logged = {}
    work = []
    for group, counts in provider.auxiliary_update_terms(model, epoch=2, update_index=0, update_count=4):
        pieces = [term.numerator * term.weight / term.denominator for term in group.values()]
        for name, term in group.items():
            logged[name] = logged.get(name, 0.) + float(term.numerator.detach()) * term.weight
        sum(pieces).backward()
        work.append(counts)
    torch.testing.assert_close(model.weight.grad, expected, rtol=1.e-11, atol=1.e-11)
    for name, value in expected_logged.items():
        assert logged[name] == pytest.approx(value, abs=1.e-14)
    assert len(work) == 9 and sum(row.get('decision_replayed_endpoints', 0) for row in work) == 5
    assert _same_rng(expected_rng, task._rng_snapshot('cpu'))
    assert set(model.core.clocks) == {(5102, 0)}
    assert all(value == 0 for value in provider._decision_replay_difference.values())
    assert provider._clock_observations['fit_forward'] == {'actual_epochs': {5102}, 'count': 8}
    assert provider._clock_observations['fit_replay'] == {'actual_epochs': {5102}, 'count': 5}


@pytest.mark.parametrize('checkpoint_retained', [False, True])
def test_full_field_qualification_is_read_only_and_checks_each_view(monkeypatch, checkpoint_retained):
    monkeypatch.setattr(task._PairedMemberRoutingTask, 'predict_native', _fake_forward)
    def pair(provider, corpus, family, action, role):
        raw = {'config': {'domain': {'lx': 1.}}}
        return {'baseline': _batch(0., 'baseline'), 'edited': _batch((ACTIONS.index(action) + 1) * .2, action),
                'raw_baseline': raw, 'raw_edited': raw}
    monkeypatch.setattr(task, 'response_pair', pair)
    model = _Model()
    provider = _make_provider(model)
    model.weight.grad = torch.full_like(model.weight, 7.)
    model.eval()
    torch.manual_seed(317)
    before_rng = task._rng_snapshot('cpu')
    report = task.qualify_full_field_replay(provider, model, 'family', local_epoch=2,
                                             checkpoint_retained=checkpoint_retained)
    assert report['full_native_fields'] and report['checkpoint_retained_graph'] is checkpoint_retained
    assert [row['view'] for row in report['views']] == ['full', 'P95']
    for row in report['views']:
        assert all(x == 0 for x in row['max_scalar_abs_difference'].values())
        assert row['gradient_difference_norm'] == pytest.approx(0, abs=1.e-12)
    assert _same_rng(before_rng, task._rng_snapshot('cpu'))
    assert not model.training and model.core._joint_epoch == 0
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 7.))


def test_native_and_port_gradient_snapshots_preserve_accumulated_gradients():
    from honf_runtime.unified_training import LossTerm
    class Route(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Linear(1, 1)
            self.output = nn.Linear(1, 1)
            self.query = nn.Linear(1, 1)
    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_blocks = nn.ModuleList([nn.Module()])
            self.context_blocks[0].mm = Route()
            self.member_router = nn.Module()
            self.member_router.member_M = nn.Linear(1, 1)
            self.member_router.joint_trunk = nn.Linear(1, 1)
            self.member_router.joint_head = nn.Linear(1, 1)
    model = nn.Module()
    model.core = Core()
    provider = task.ThermalR165DecisionTask.__new__(task.ThermalR165DecisionTask)
    provider._snapshot_epoch = None
    params = list(model.parameters())
    for p in params:
        p.grad = torch.full_like(p, 9.)
    a = sum(p.square().sum() for p in params)
    b = sum(p.sum() for p in params)
    terms = {'final/flow/u': LossTerm(a, 2, .5), 'port/outside_temperature': LossTerm(b, 2, .05)}
    provider.diagnose_native_training_terms(model, terms, {'final/flow/u': 2.,
        'port/outside_temperature': 2.}, epoch=1, update_index=0, update_count=4)
    groups = provider._snapshot_groups
    assert all(groups[name] for name in ('context_values', 'relation_calibration', 'joint_trunk'))
    report = task._gradient_report(provider._snapshot_buffers, groups)
    assert all(report[group]['norms']['native'] > 0 for group in groups)
    assert all(report[group]['norms']['port'] > 0 for group in groups)
    assert all(report[group]['norms']['decision'] == 0 for group in groups)
    assert all(report[group]['cosines']['endpoint_vs_decision'] is None for group in groups)
    for p in params:
        torch.testing.assert_close(p.grad, torch.full_like(p, 9.))
    objective = .25 * a + .025 * b
    expected = torch.autograd.grad(objective, params, retain_graph=True)
    objective.backward()
    for p, grad in zip(params, expected):
        torch.testing.assert_close(p.grad, torch.full_like(p, 9.) + grad)


def test_edited_endpoint_uses_saved_physical_targets_without_port_lookup(monkeypatch):
    from channelthermal.data import datasets
    from channelthermal.training.content_hypergraph_task import ThermalContentHypergraphTask
    from channelthermal.training.thermal_state_response_task import _native_endpoint_batch
    def unexpected(*args, **kwargs):
        raise AssertionError('Edited FIT endpoints must not fetch baseline port targets')
    provider = SimpleNamespace(_native_case_batch=unexpected, _with_port_targets=unexpected)
    captured = {}
    def native_batch(self, cases, indices, *, split, ntheta):
        captured.update(case=cases[0], split=split, ntheta=ntheta)
        return 'direct edited batch'
    monkeypatch.setattr(ThermalContentHypergraphTask, '_native_case_batch', native_batch)
    monkeypatch.setattr(datasets, '_local_disk_query_points', lambda mask: np.zeros((2, 2)))
    original = {'case_id': '0007', 'structure': {'module_present': np.array([1.]),
        'module_centers': np.zeros((1, 2)), 'heat_powers': np.array([3.])}}
    raw = {'config': {'layout': {'centers': [[.3, .4]], 'heat_powers': [3.]}},
           'field': np.full((2, 2, 5), 7.), 'fluid_mask': np.array([[True, False], [True, True]]),
           'material_mask': np.array([[True, False], [False, True]]),
           'material': np.full((1, 2, 2), 11.), 'interface_target': np.full((1, 64, 2), 13.)}
    assert _native_endpoint_batch(provider, original, raw, 'edit') == 'direct edited batch'
    case = captured['case']
    assert case['case_id'] == '0007:edit'
    np.testing.assert_allclose(case['structure']['module_centers'], [[.3, .4]])
    np.testing.assert_array_equal(case['steady_field'], raw['field'])
    np.testing.assert_array_equal(case['module_mask'], ~raw['fluid_mask'])
    assert np.all(case['module_internal_temperature_points'] == 11.)
    assert np.all(case['interface_target'] == 13.)
    assert captured['ntheta'] == 64 and captured['split'] == 'train'
