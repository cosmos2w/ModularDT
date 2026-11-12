"""Explicit Wind horizon changes retain every scientific identity field."""
import copy

import pytest

from honf_runtime.unified_training import _resume_identity_amendment


def identities():
    old = {
        "workflow": "relation_anchored_case_training_v1",
        "recipe": {"dataset": "wind", "schema": "wind_relation_anchored_r15_2_v2",
                   "total_epochs": 1000, "seed": 42, "primary_queries": 8192},
        "engine_config": {"total_epochs": 1000, "checkpoint_epochs": list(range(100, 1001, 100)),
                          "microbatch_cases": 24, "effective_cases": 24},
        "provider_identity": {"membership": [1, 2], "training": {"total_epochs": 1000}},
        "optimizer_schedule_contract": [{"name": "wind_decay", "weight_decay": 1e-5,
            "schedule": {"peak_lr": .0003, "final_lr": .0003, "warmup_start_lr": .0003,
                         "warmup_epochs": 0, "total_epochs": 1000}}],
    }
    new = copy.deepcopy(old)
    new["engine_config"].update(total_epochs=2500, checkpoint_epochs=list(range(100, 2501, 100)))
    return old, new


def test_extension_is_opt_in_and_keeps_original_schedule_and_identity():
    old, new = identities()
    snapshot = copy.deepcopy(old)
    with pytest.raises(ValueError):
        _resume_identity_amendment(old, new)
    receipt = _resume_identity_amendment(old, new, allow_horizon_extension=True)
    assert receipt["previous_horizon"] == 1000 and receipt["execution_horizon"] == 2500
    assert receipt["optimizer_schedule_unchanged"] is True
    assert old == snapshot
    assert _resume_identity_amendment(new, new, allow_horizon_extension=True) is None


@pytest.mark.parametrize("path,value", [
    (("recipe", "seed"), 43),
    (("recipe", "primary_queries"), 4096),
    (("recipe", "total_epochs"), 2500),
    (("provider_identity", "membership"), [1, 3]),
    (("provider_identity", "training", "total_epochs"), 2500),
    (("engine_config", "microbatch_cases"), 12),
    (("engine_config", "checkpoint_epochs"), [2500]),
    (("engine_config", "total_epochs"), 2400),
])
def test_extension_rejects_hidden_scientific_or_retention_changes(path, value):
    old, new = identities()
    target = new
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        _resume_identity_amendment(old, new, allow_horizon_extension=True)


def test_extension_rejects_nonconstant_schedule_and_other_case_workloads():
    old, new = identities()
    for value in (old, new):
        value["optimizer_schedule_contract"][0]["schedule"]["final_lr"] = 1e-5
    with pytest.raises(ValueError):
        _resume_identity_amendment(old, new, allow_horizon_extension=True)
    old, new = identities()
    for value in (old, new):
        value["recipe"]["schema"] = "wind_relation_anchored_r15_3_v1"
    with pytest.raises(ValueError):
        _resume_identity_amendment(old, new, allow_horizon_extension=True)


def test_thermal_candidate_extension_keeps_recipe_and_optimizer_contract():
    old, new = identities()
    for value in (old, new):
        value['recipe'] = {'dataset': {'name': 'thermal'},
                           'training': {'membership': 'full_train600'},
                           'thermal_port_response': {'schema': 'thermal_port_response_v1', 'arm': 'P64'}}
    old['engine_config'].update(total_epochs=100, checkpoint_epochs=[100])
    new['engine_config'].update(total_epochs=500, checkpoint_epochs=[100, 200, 300, 400, 500])
    receipt = _resume_identity_amendment(old, new, allow_horizon_extension=True)
    assert receipt['kind'] == 'explicit_thermal_port_horizon_extension'
    for value in (old, new):
        value['recipe']['training']['membership'] = 'fixed25_train150_dev22'
    with pytest.raises(ValueError):
        _resume_identity_amendment(old, new, allow_horizon_extension=True)
    for value in (old, new):
        value['recipe']['training']['membership'] = 'full_train600'
    for arm in ("C0", "R64"):
        for value in (old, new):
            value["recipe"]["thermal_port_response"]["arm"] = arm
        with pytest.raises(ValueError):
            _resume_identity_amendment(old, new, allow_horizon_extension=True)


def r16_10_identities(arm='B0'):
    import json
    from pathlib import Path
    recipe = json.loads((Path(__file__).resolve().parents[1] / 'configs/research' /
                         f'thermal_r16_10_{arm.lower()}_train600.json').read_text())
    old, _ = identities()
    old['recipe'] = recipe
    old['engine_config'].update(total_epochs=2000, checkpoint_epochs=list(range(100, 2001, 100)))
    old['provider_identity'] = {'response_scales': {'J': .123}, 'adapter_state': {'core': {'architecture': 'original'}}}
    old['optimizer_schedule_contract'][0]['schedule'] = {'peak_lr': .0003, 'warmup_start_lr': .0003,
        'warmup_epochs': 0, 'hold_through_epoch': 1000, 'total_epochs': 2000, 'final_lr': .00003}
    new = copy.deepcopy(old)
    new['engine_config'].update(total_epochs=3000, checkpoint_epochs=list(range(100, 3001, 100)))
    return old, new


@pytest.mark.parametrize('arm', ['B0', 'R'])
def test_r16_10_full_lineage_extension_preserves_clamped_schedule(arm):
    from honf_runtime.unified_training import ScheduleSpec
    from tools.relation_anchored_train import engine_for
    old, new = r16_10_identities(arm)
    receipt = _resume_identity_amendment(old, new, allow_horizon_extension=True)
    assert receipt['kind'] == 'explicit_r16_10_horizon_extension'
    assert old['optimizer_schedule_contract'] == new['optimizer_schedule_contract']
    schedule = ScheduleSpec(**old['optimizer_schedule_contract'][0]['schedule'])
    assert all(schedule.value(epoch) == .00003 for epoch in (2000, 2001, 2500, 3000))
    engine = engine_for(old['recipe'], device='cpu', execution_horizon=3000)
    assert engine.config.total_epochs == 3000
    assert engine.config.checkpoint_epochs == tuple(range(100, 3001, 100))


@pytest.mark.parametrize('path,value', [
    (('recipe','r16_10_family','population'),150),
    (('recipe','r16_10_family','arm'),'D'),
    (('recipe','member_routing','model','access_measure'),'active_domain'),
    (('provider_identity','response_scales','J'), .456),
    (('provider_identity','adapter_state','core','architecture'), 'different'),
    (('optimizer_schedule_contract',0,'schedule','final_lr'),.00004),
    (('engine_config','microbatch_cases'),12),
])
def test_r16_10_extension_rejects_identity_changes(path,value):
    old,new = r16_10_identities()
    target=new
    for key in path[:-1]: target=target[key]
    target[path[-1]]=value
    with pytest.raises(ValueError): _resume_identity_amendment(old,new,allow_horizon_extension=True)


def test_r16_10_rejects_nonmatching_saved_schedule_even_when_both_agree():
    old,new=r16_10_identities()
    for value in (old,new): value['optimizer_schedule_contract'][0]['schedule']['total_epochs']=3000
    with pytest.raises(ValueError): _resume_identity_amendment(old,new,allow_horizon_extension=True)
