import copy
import math
import pytest
import torch

from channelthermal.training.r16_8_content_task import (
    COEFFICIENT, action_slots, selected_families, make_content_recipe,
    validate_content_recipe, enable_content, ThermalR168ContentTask)
from channelthermal.training.thermal_state_response_task import ACTIONS, response_loss_terms
from test_r16_5_warm_fork import identities
from relation_anchored_train import engine_for


def test_48_placements_retain_one_mean_and_balanced_cycles():
    for n in (4, 13, 25):
        slots = action_slots(n)
        assert slots == tuple(a * n // 48 for a in range(48))
        assert math.isclose(sum(n / 48 for _ in slots) / n, 1.)
    families = [f'{i:03}' for i in range(96)]
    assert set(sum((list(selected_families(families, e, 0, 8)) for e in range(1, 13)), [])) == set(families)
    scalar = torch.tensor(1.)
    terms = response_loss_terms(scalar, scalar, {'J': scalar, 'P': scalar}, multiplier=13/48,
                                finite_response_coefficient=COEFFICIENT)
    assert [term.weight for term in terms.values()] == [.25*13/48, .8*COEFFICIENT*13/48,
                                                         .1*COEFFICIENT*13/48, .1*COEFFICIENT*13/48]


def test_qualification_schedule_has_full_workload_and_formal_recipe_is_strict():
    class Corpus:
        def family_ids(self, role): return tuple(f'{i:03}' for i in range(48))
        def fit_families_for_epoch(self, epoch, seed): return selected_families(self.family_ids('fit'), epoch, seed, 4)
    task = ThermalR168ContentTask.__new__(ThermalR168ContentTask)
    task.response_corpus, task.qualification, task.seed, task.source_parent_age = Corpus(), True, 0, 5100
    schedule = task._action_schedule(1)
    assert len(schedule) == 48
    assert len({f for _, f, _, _ in schedule}) == 12
    assert [x[3] for x in schedule] == ['original_fit'] * 16 + ['qualification_reuse'] * 32
    assert all([a for _, f, a, _ in schedule[i:i+4]] == list(ACTIONS) for i in range(0, 48, 4))
    p64, h0, _ = identities()
    for arm in ('C8', 'S8'):
        r = make_content_recipe(p64, h0, arm=arm, name=arm, output_root='/tmp',
            membership='full_train600', local_horizon=500, first_segment_epochs=100,
            corpus_manifest='/tmp/physical.json')
        validate_content_recipe(None, r, p64, h0)
        assert r['storage']['max_output_bytes'] == 4 * 1024 ** 3
        assert engine_for(r, device='cpu').config.checkpoint_epochs == (100, 200, 300, 400, 500)
        altered = copy.deepcopy(r); altered['optimizer']['lr'] = .1
        with pytest.raises(ValueError, match='differs'):
            validate_content_recipe(None, altered, p64, h0)
    q = make_content_recipe(p64, h0, arm='C8', name='q', output_root='/tmp',
        membership='fixed25_train150_dev22', local_horizon=4, first_segment_epochs=3)
    assert q['storage']['max_output_bytes'] == p64['recipe']['storage']['max_output_bytes']
    with pytest.raises(ValueError):
        make_content_recipe(p64, h0, arm='C8', name='x', output_root='/tmp', membership='full_train600',
                            local_horizon=500, first_segment_epochs=100)


def test_content_enable_preserves_existing_tensors():
    from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
    model = torch.nn.Sequential(SpatialRouteAttentionV2(8, 4, 2))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    inventory = enable_content(model, 'S8')
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    assert len(inventory['new_names']) == 8
    assert all(torch.count_nonzero(model.state_dict()[k]) == 0 for k in inventory['new_names'])


def test_validation_selection_uses_direct_fields_and_native_masks():
    from types import SimpleNamespace as NS
    from channelthermal.training.content_hypergraph_task import ThermalContentPredictions
    from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
    from channelthermal.training.thermal_port_base import ThermalPortBaseTask

    task = ThermalR168ContentTask.__new__(ThermalR168ContentTask)
    task.adapter = NS(normalization_stats={'field_std_by_channel': [1.] * 5,
        'interface_targets_std': [1., 1.], 'internal_temperature_std': [1.]})
    stencil = lambda mask: NS(valid=torch.tensor(mask, dtype=torch.bool))
    prepared = NS(source_present=torch.tensor([[True]]), stencils={
        'fluid': stencil([[True, False]]), 'surface': stencil([[[True, False]]]),
        'outside': stencil([[[True, False]]]), 'material': stencil([[[True, False]]])})
    targets = NS(case_ids=('one',), field_targets=torch.zeros(1, 2, 5),
        point_weights=torch.ones(1, 2), interface_target=torch.zeros(1, 1, 2, 2),
        interface_valid_mask=torch.ones(1, 1, 2), material_targets=torch.zeros(1, 1, 2))
    direct = ThermalContentPredictions(prepared, {'pred_field': torch.tensor([[[0.]*5, [100.]*5]]),
        'pred_interface': torch.tensor([[[[2., 4.], [100., 100.]]]]),
        'pred_internal_temperature': torch.tensor([[[[3.], [100.]]]])})
    direct.read_mode = 'full'
    direct.local1 = {'full': {'interface_pred': torch.full((1, 1, 64, 2), 20.),
                              'internal_temperature': torch.full((1, 1, 2, 1), 30.)}}
    actual = task.validation_metrics(direct, targets, None)['case_rows'][0]
    local = ThermalPortBaseTask.validation_metrics(task, direct, targets, None)['case_rows'][0]
    assert actual['roles']['temperature_surface']['rmse'] == pytest.approx(2.)
    assert actual['roles']['temperature_material']['rmse'] == pytest.approx(3.)
    assert actual['q_proxy_rmse'] == pytest.approx(4.)
    assert actual['field_score'] == pytest.approx(13. / 6.)
    assert local['field_score'] != actual['field_score']
    pair = ThermalJointPairPredictions(full=ThermalContentPredictions(prepared, {
        k: torch.zeros_like(v) for k, v in direct.native_main.items()}), restricted=direct, work={})
    assert task.validation_metrics(pair, targets, None)['case_rows'][0]['field_score'] == pytest.approx(13. / 6.)


def test_48_action_weighted_loss_and_unweighted_diagnostics_average_once():
    task = ThermalR168ContentTask.__new__(ThermalR168ContentTask)
    for updates in (4, 13, 25):
        recorded = []
        diagnostic = []
        for index in range(48):
            value = torch.tensor(float(index + 1))
            terms = response_loss_terms(value, value, {'J': value, 'P': value},
                multiplier=updates / 48, finite_response_coefficient=COEFFICIENT)
            recorded.append({name: float(term.numerator) * term.weight / term.denominator
                             for name, term in terms.items()})
            diagnostic.append(task.auxiliary_diagnostic_values(terms, update_count=updates))
        expected = sum(range(1, 49)) / 48
        for name, coefficient in [('response/endpoint', .25), ('response/field_delta', .8 * COEFFICIENT),
                                  ('response/J_delta', .1 * COEFFICIENT), ('response/P_delta', .1 * COEFFICIENT)]:
            assert sum(row[name] for row in recorded) / updates == pytest.approx(expected * coefficient)
            if name != 'response/endpoint':
                assert sum(row[name] for row in diagnostic) / updates == pytest.approx(expected)


def test_qualification_retains_dev22_and_formal_requires_p64_val89(monkeypatch):
    from types import SimpleNamespace as NS
    from channelthermal.training.r16_8_content_task import validate_validation_membership
    from channelthermal.data import development_split
    dev = tuple(f'd{i:02}' for i in range(22))
    val = tuple(f'v{i:02}' for i in range(89))
    monkeypatch.setattr(development_split, 'development_case_ids', lambda manifest, split: dev)
    cases = [object() for _ in dev]
    provider = NS(manifest={}, validation_case_ids=dev, validation_cases=cases,
                  catalog_case_ids=val, case_split_by_id={i: 'test' for i in val})
    validate_validation_membership(provider, 'fixed25_train150_dev22', {'validation_case_ids': list(val)})
    assert provider.validation_cases is cases and provider.validation_case_ids is dev
    provider.validation_case_ids = val
    validate_validation_membership(provider, 'full_train600', {'validation_case_ids': list(val)})
    with pytest.raises(ValueError, match='DEV22'):
        validate_validation_membership(provider, 'fixed25_train150_dev22', {'validation_case_ids': list(val)})
    with pytest.raises(ValueError, match='VAL89'):
        validate_validation_membership(provider, 'full_train600', {'validation_case_ids': list(val[:-1])})
