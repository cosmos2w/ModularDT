"""Explicit execution extensions keep the original scientific schedule sealed."""
import copy
import json
from types import SimpleNamespace

import pytest

from honf_runtime.research_training import _extend_execution_horizon
from honf_runtime.unified_training import EngineConfig, _engine_config_payload, _resume_identity_amendment


def identities():
    engine = SimpleNamespace(config=EngineConfig(seed=0, microbatch_cases=24, effective_cases=48,
        total_epochs=1000, training_mode="joint", monitor_every=10,
        monitor_epochs=tuple(range(10, 1001, 10)), checkpoint_epochs=tuple(range(100, 1001, 100))))
    saved = {"workflow": "ordinary_honf_research_train_v1", "engine_config": _engine_config_payload(engine.config),
             "provider_identity": {"sampling": "frozen", "total_epochs": 1000},
             "optimizer_schedule_contract": {"lr": 0.0003, "total_epochs": 1001}}
    _extend_execution_horizon(engine, 2500)
    current = {**copy.deepcopy(saved), "engine_config": _engine_config_payload(engine.config)}
    return saved, current


def test_execution_extension_requires_explicit_opt_in_and_preserves_schedule():
    saved, current = identities()
    with pytest.raises(ValueError):
        _resume_identity_amendment(saved, current)
    amendment = _resume_identity_amendment(saved, current, allow_horizon_extension=True)
    assert amendment["execution_horizon"] == 2500
    assert amendment["schedule_horizon_unchanged"]
    assert _resume_identity_amendment(current, current) is None
    assert _resume_identity_amendment(saved, json.loads(json.dumps(current)), allow_horizon_extension=True)


@pytest.mark.parametrize("change", ["lr", "sampling", "batch", "cadence", "milestone", "workflow", "shorter"])
def test_extension_rejects_other_changes(change):
    saved, current = identities()
    if change == "lr": current["optimizer_schedule_contract"]["lr"] = 0.001
    elif change == "sampling": current["provider_identity"]["sampling"] = "changed"
    elif change == "batch": current["engine_config"]["effective_cases"] = 24
    elif change == "cadence": current["engine_config"]["monitor_every"] = 20
    elif change == "milestone": current["engine_config"]["checkpoint_epochs"] = (2500,)
    elif change == "workflow": saved["workflow"] = current["workflow"] = "another"
    elif change == "shorter": current["engine_config"]["total_epochs"] = 900
    with pytest.raises(ValueError):
        _resume_identity_amendment(saved, current, allow_horizon_extension=True)
