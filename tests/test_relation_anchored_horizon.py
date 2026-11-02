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
