from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from honf_runtime.contextual_frontier_training import load_contextual_frontier_config


CONFIGS = PROJECT / "configs/research"


def test_explicit_horizon_extension_keeps_provider_and_schedule_identity():
    import copy
    from honf_runtime.unified_training import _resume_identity_amendment

    saved = {
        "workflow": "interaction_reused_member_routing_joint_training_v1",
        "engine_config": {"total_epochs": 100, "checkpoint_epochs": [100]},
        "provider_identity": {"normalizer": "original", "constant_schedule": 3e-4},
    }
    current = copy.deepcopy(saved)
    current["engine_config"].update(total_epochs=200, checkpoint_epochs=[100, 200])
    amendment = _resume_identity_amendment(saved, current, allow_horizon_extension=True)
    assert amendment["optimizer_schedule_unchanged"]
    current["provider_identity"]["normalizer"] = "replacement"
    with pytest.raises(ValueError):
        _resume_identity_amendment(saved, current, allow_horizon_extension=True)


def test_native_only_budget_and_elapsed_state_survive_checkpoint_restore(tmp_path, monkeypatch):
    import time
    from channelthermal.training.member_routing_task import ThermalInteractionReusedMemberRoutingTask

    provider = ThermalInteractionReusedMemberRoutingTask.__new__(ThermalInteractionReusedMemberRoutingTask)
    provider.native_only = True
    provider.organizer = None
    provider.state = {"primary_updates": 0, "training_elapsed_seconds": 4.0}
    provider.training_cases = ("case",)
    provider.effective_batch_size = 48
    provider.training_samples = 150
    provider.validation_samples = 22
    provider._budget_binding = (tmp_path, 20.0, 5.0)
    provider._segment_started = 0.0
    monkeypatch.setattr(time, "monotonic", lambda: 12.0)
    provider.before_optimizer_step(None, None, {}, {}, epoch=1, update_index=0)
    assert (tmp_path / "CLEAN_STOP_REQUEST.json").is_file()
    state = provider.training_state_dict()
    assert state["training_elapsed_seconds"] == 16.0
    provider.load_training_state_dict(state)
    assert provider.training_samples == 150
    assert provider.validation_samples == 22
    assert provider.state["training_elapsed_seconds"] == 16.0


@pytest.mark.parametrize(
    "name,arm,membership,horizon,first_segment,normalization",
    (
        ("thermal_member_routing_r13_3_full_v1.json", "R13-3-Full", "full_train600", 100, 100,
         "scratch_train600_stats"),
        ("thermal_member_routing_r13_3_route_v1.json", "R13-3-Route", "full_train600", 100, 100,
         "scratch_train600_stats"),
        ("thermal_member_routing_r13_3_qual_full_v1.json", "R13-3-Full", "fixed25_train150_dev22", 2, 1,
         "fixed25_train150_stats"),
        ("thermal_member_routing_r13_3_qual_route_v1.json", "R13-3-Route", "fixed25_train150_dev22", 2, 1,
         "fixed25_train150_stats"),
    ),
)
def test_member_routing_configs_bind_native_scratch_contract(
    name, arm, membership, horizon, first_segment, normalization,
):
    config, _ = load_contextual_frontier_config(CONFIGS / name)
    section = config["member_routing"]
    model = section["model"]
    base = model["base_model_config"]

    assert section["arm"] == arm
    assert model["architecture"] == arm
    assert model["member_routing_mode"] == ("full" if arm == "R13-3-Full" else "route")
    assert model["schema"] == "interaction_reused_member_routing_v1"
    assert model["input_capability"] == "joint_prescribed_records_v1"
    assert base["schema"] == "content_hypergraph_core_v2"
    assert (base["module_input_dim"], base["environment_input_dim"],
            base["operating_input_dim"], base["coordinate_dim"], base["output_dim"]) == (9, 6, 12, 2, 4)
    assert config["training"]["membership"] == membership
    assert config["training"]["local_horizon"] == horizon
    assert config["training"]["first_segment_epochs"] == first_segment
    assert config["training"]["optimizer_schedule_horizon"] == horizon
    assert config["initialization"]["normalization_mode"] == normalization
    assert config["initialization"]["source_checkpoint"] is None
    assert config["training"]["seed"] == 0
    assert config["optimizer"]["schedule"] == "constant_config_lr"
    assert config["optimizer"]["lr"] == pytest.approx(3e-4)
    assert config["objective"]["response_training_families"] == []
    assert config["objective"]["response_fit"] is False
    assert config["objective"]["organizer_objective"] is False
    if membership == "full_train600":
        assert config["training"]["checkpoint_epochs"] == [100]
        assert config["training"]["curve_every"] == 10
    else:
        assert config["training"]["checkpoint_epochs"] == [1, 2]
        assert config["training"]["curve_every"] == 1


def test_full_and_route_arms_share_normalizer_optimizer_and_objectives():
    full, _ = load_contextual_frontier_config(CONFIGS / "thermal_member_routing_r13_3_full_v1.json")
    route, _ = load_contextual_frontier_config(CONFIGS / "thermal_member_routing_r13_3_route_v1.json")

    assert full["dataset"] == route["dataset"]
    assert full["initialization"] == route["initialization"]
    assert full["optimizer"] == route["optimizer"]
    assert full["training"] == route["training"]
    assert full["objective"] == route["objective"]
    full_model = full["member_routing"]["model"]
    route_model = route["member_routing"]["model"]
    assert {key: value for key, value in full_model.items()
            if key not in ("architecture", "member_routing_mode")} == {
                key: value for key, value in route_model.items()
                if key not in ("architecture", "member_routing_mode")
            }
