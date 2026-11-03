import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from channelthermal.training.thermal_native_functionals import native_design_functionals
from channelthermal.training.thermal_finite_response_task import (calibrate_response_balance,
    ThermalFiniteResponseTask)
from channelthermal.training import thermal_finite_response_task as finite_task


def test_full_native_functionals_keep_gradient_and_design_masks():
    material = torch.tensor([300., 302., 999.], requires_grad=True)
    pressure = torch.tensor([4., 1., 900., 8.], requires_grad=True)
    j, p = native_design_functionals(material, torch.tensor([1, 1, 0]), pressure,
        torch.tensor([1, 1, 0, 1]), torch.tensor([0., 1., 0., .5]), 1., 2.)
    assert j.item() == pytest.approx(2 * (math.log(math.exp(150) + math.exp(151)) - math.log(2)))
    assert p.item() == 3.
    (j + p).backward()
    assert material.grad[0] > 0 and material.grad[1] > 0 and material.grad[2] == 0
    assert pressure.grad.tolist() == [1., -1., 0., 0.]


def test_empty_band_is_not_silently_replaced():
    with pytest.raises(ValueError, match='both fluid pressure bands'):
        native_design_functionals(torch.ones(1), torch.ones(1, dtype=torch.bool), torch.ones(2),
            torch.ones(2, dtype=torch.bool), torch.tensor([.2, .8]), 1., 1.)


def test_calibration_uses_all_inherited_parameters_and_records_zero():
    x = torch.nn.Parameter(torch.tensor(1.))
    y = torch.nn.Parameter(torch.tensor(1.))
    examples = [(x + 3*y, 2*x + 6*y) for _ in range(11)] + [(x + 3*y, 0*x)]
    coefficient, rows = calibrate_response_balance(examples, [x, y])
    assert coefficient == pytest.approx(.125)
    assert rows[-1]['delta_zero'] and rows[-1]['cosine'] is None
    assert rows[0]['absolute_norm'] == pytest.approx(math.sqrt(10))


def test_pair_denominator_is_one_for_accumulated_microbatches():
    task = object.__new__(ThermalFiniteResponseTask)
    task.native_weights = {'flow/u': 1.}
    task.finite_spec = {'arm': 'B'}
    batches = [SimpleNamespace(targets=SimpleNamespace(case_ids=tuple(range(24)), response_pack={}))]
    batches += [SimpleNamespace(targets=SimpleNamespace(case_ids=tuple(range(24)), response_pack=None))]
    denominators = task.loss_denominators(batches)
    assert denominators['final/flow/u'] == 48
    assert denominators['finite/endpoint'] == denominators['finite/delta'] == 1
    with pytest.raises(ValueError, match='Exactly one pair'):
        task.loss_denominators(batches[:1] * 2)


def test_delta_uses_both_branches_and_backpropagates_to_both_endpoints():
    task = object.__new__(ThermalFiniteResponseTask)
    names = ('flow/u', 'flow/v', 'flow/p', 'flow/omega', 'temperature/fluid',
             'temperature/surface', 'temperature/material', 'q_proxy')
    task.native_weights = {name: 1. for name in names}
    task.model = SimpleNamespace(internal_temperature_std=torch.tensor([2.]))
    def endpoint(value):
        field = torch.full((1, 2, 5), value, requires_grad=True)
        interface = torch.full((1, 1, 2, 2), value, requires_grad=True)
        material = torch.full((1, 1, 2, 1), value, requires_grad=True)
        stencil = lambda shape: SimpleNamespace(valid=torch.ones(shape, dtype=torch.bool))
        prepared = SimpleNamespace(stencils={'fluid': stencil((1, 2)), 'surface': stencil((1, 2)),
            'outside': stencil((1, 2)), 'material': stencil((1, 2))}, source_present=torch.ones((1, 1), dtype=torch.bool))
        return SimpleNamespace(native_main={'pred_field': field, 'pred_interface': interface,
            'pred_internal_temperature': material}, native_prepared=prepared)
    base_full, edit_full, base_restricted, edit_restricted = (endpoint(v) for v in (0., 1., 0., 2.))
    designs = {'baseline': SimpleNamespace(full=base_full, restricted=base_restricted),
               'edited': SimpleNamespace(full=edit_full, restricted=edit_restricted)}
    delta = {}
    for name in names:
        shape = (1, 2) if name in names[:5] else (1, 1, 2)
        delta[name] = {'target': torch.zeros(shape), 'mask': torch.ones(shape, dtype=torch.bool), 'scale': 1.}
    native = {'material_mask': torch.ones((1, 1, 2), dtype=torch.bool),
              'fluid_mask': torch.ones((1, 2), dtype=torch.bool), 'fluid_x': torch.tensor([[0., 1.]]),
              'domain_length_x': 1.}
    pair = {'delta': delta, 'functionals': {'baseline': native, 'edited': native,
        'J': {'scale': 1., 'target_delta': 0.}, 'P': {'scale': 1., 'target_delta': 0.}}}
    loss = task._finite_delta(designs, pair)
    loss.backward()
    for endpoint_value in (base_full, edit_full, base_restricted, edit_restricted):
        assert endpoint_value.native_main['pred_field'].grad.abs().sum() > 0
        assert endpoint_value.native_main['pred_internal_temperature'].grad.abs().sum() > 0


def test_response_scales_are_bound_to_provider_and_checked_on_reload(monkeypatch):
    scales = {'single': {'J': 2.}, 'joint': {'J': 3.}}
    spec = {'schema': finite_task.SCHEMA, 'enabled': True, 'arm': 'B',
            'source_file_epoch': 100, 'inherited_age': 5100, 'source_checkpoint': 'source.pt',
            'scales_path': 'scales.json'}
    task = object.__new__(ThermalFiniteResponseTask)
    task.finite_spec, task.pair_sampler = spec, SimpleNamespace(scales=scales)
    monkeypatch.setattr(finite_task.ThermalPortResponseTask, 'identity_payload', lambda self: {})
    identity = task.identity_payload()
    assert identity['finite_response_scales'] == scales
    scales['single']['J'] = 9.
    assert identity['finite_response_scales']['single']['J'] == 2.
    recipe = {'thermal_finite_response': spec, 'thermal_port_response': {'source_checkpoint': 'source.pt'},
              'member_routing': {'arm': 'detail'}}
    saved = {'epoch': 1, 'arm': 'detail', 'experiment_identity': {
        'workflow': 'relation_anchored_case_training_v1', 'recipe': recipe, 'provider_identity': identity}}
    source = {'epoch': 100, 'experiment_identity': {'recipe': {
        'thermal_port_response': {'arm': 'P64', 'source_epoch': 5000},
        'training': {'membership': 'full_train600'}}}}
    monkeypatch.setattr(finite_task, 'load_trusted_checkpoint', lambda path, **kw: source if path == 'source.pt' else saved)
    monkeypatch.setattr(finite_task.Path, 'read_text', lambda self: '{"scale_by_action_class":{"single":{"J":9},"joint":{"J":3}}}')
    with pytest.raises(ValueError, match='scales differ'):
        finite_task.load_finite_response_checkpoint('run.pt')


def test_teacher_work_is_read_from_children_after_pair_loss(monkeypatch):
    from honf_runtime.unified_training import LossTerm
    @dataclass(frozen=True)
    class Target:
        response_pack: object = None
    baseline=SimpleNamespace(work={'preparations':1,'local_calls':2})
    edited=SimpleNamespace(work={'preparations':1,'local_calls':2})
    pair={'baseline':SimpleNamespace(targets=Target()),'edited':SimpleNamespace(targets=Target())}
    prediction=SimpleNamespace(work={'preparations':3,'local_calls':6,
        'response_baseline_preparations':1,'response_edited_preparations':1},
        response_designs={'baseline':baseline,'edited':edited})
    def parent_terms(self, pred, targets):
        pred.work['teacher_local_calls']=1
        return {'native':LossTerm(torch.tensor(1.),1)}
    monkeypatch.setattr(finite_task.ThermalPortResponseTask,'_terms',parent_terms)
    task=object.__new__(ThermalFiniteResponseTask);task.finite_spec={'arm':'A'}
    before=task.work_counts(None,prediction,{})
    assert before['total_teacher_local_calls']==0 and before['total_local_calls_including_teacher']==6
    task._terms(prediction,Target(response_pack=pair))
    after=task.work_counts(None,prediction,{})
    assert after['teacher_local_calls']==1
    assert after['response_baseline_teacher_local_calls']==after['response_edited_teacher_local_calls']==1
    assert after['total_teacher_local_calls']==3 and after['total_local_calls_including_teacher']==9
    assert after['preparations']==3 and after['local_calls']==6
    assert after['response_baseline_preparations']==after['response_edited_preparations']==1


def test_teacher_work_without_pair_or_teacher_call():
    task=object.__new__(ThermalFiniteResponseTask)
    prediction=SimpleNamespace(work={'preparations':1,'local_calls':1})
    counts=task.work_counts(None,prediction,{})
    assert counts['total_teacher_local_calls']==0 and counts['total_local_calls_including_teacher']==1
    prediction.work['teacher_local_calls']=1
    counts=task.work_counts(None,prediction,{})
    assert counts['total_teacher_local_calls']==1 and counts['total_local_calls_including_teacher']==2
    assert counts['response_baseline_teacher_local_calls']==counts['response_edited_teacher_local_calls']==0
