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


def test_relative_adapter_payload_roundtrip_preserves_v2_and_rejects_v1_relabelling():
    import torch
    from channelthermal.member_routing import (
        ThermalInteractionReusedMemberRoutingAdapter, ThermalRelativeMemberRoutingAdapter,
    )
    from honf_forward_core.interaction_reused_member_routing import RelativeMemberRoutingCore
    from tests.test_interaction_reused_member_routing import _base_config
    from tests.test_native_budgeted_postfit import _thermal_stats

    core = RelativeMemberRoutingCore(_base_config(), input_dataset="thermal")
    model = ThermalRelativeMemberRoutingAdapter(_thermal_stats(), core=core)
    restored = ThermalRelativeMemberRoutingAdapter.from_state_payload(
        model.state_payload(), state_dict=model.state_dict(),
    )
    assert restored.get_config() == model.get_config()
    assert restored.architecture == "R13-3-Relative"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Not a Thermal"):
        ThermalInteractionReusedMemberRoutingAdapter.from_state_payload(model.state_payload())


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


def test_native_trace_reports_only_executed_original_ids_and_real_stage_provenance():
    from types import SimpleNamespace
    import torch
    from channelthermal.training.member_routing_task import ThermalInteractionReusedMemberRoutingTask as Task
    from honf_forward_core.content_hypergraph.contracts import QueryBatch

    trace = {
        "original_module_ids": torch.tensor([[101]]),
        "original_environment_ids": torch.tensor([[201, 202]]),
        "B_M": torch.ones(1, 1, 1), "B_E": torch.ones(1, 1, 2),
        "proposal_weights_M": torch.ones(1, 3, 1), "proposal_null": torch.zeros(1, 3),
        "candidate_mask_M": torch.ones(1, 3, 1, dtype=torch.bool),
        "candidate_mask_E": torch.ones(1, 3, 2, dtype=torch.bool),
        "member_weights_M": torch.tensor([[[0.3], [0.0], [0.2]]]),
        "member_weights_E": torch.tensor([[[0.1, 0.0], [0.0, 0.0], [0.4, 0.0]]]),
        "member_null_M": torch.tensor([[0.7, 1.0, 0.8]]),
        "member_null_E": torch.tensor([[0.9, 1.0, 0.6]]),
        "member_logits_M": torch.zeros(1, 3, 1), "member_logits_E": torch.zeros(1, 3, 2),
        "selected_value_indices_M": torch.tensor([[0, 0, 0], [0, 2, 0]]),
        "selected_value_indices_E": torch.tensor([[0, 0, 0], [0, 2, 0]]),
    }
    scene = SimpleNamespace(
        module_coordinates=torch.zeros(1, 1, 2), environment_coordinates=torch.zeros(1, 2, 2),
        module_mask=torch.ones(1, 1, dtype=torch.bool), environment_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    organization = SimpleNamespace(scene=scene, context_stage_metadata=[{
        "stage": "cycle1/me", "route": "me", "source_state_version": "cycle1/after_EE_FFN_E",
        "eligible_edges": 2, "positive_value_edges": 2,
    }])
    query = QueryBatch(coordinates=torch.zeros(1, 3, 2))
    core = SimpleNamespace(capture_interaction_routing_trace=lambda *args, **kwargs: trace)
    captured, rows = Task._trace_rows(SimpleNamespace(core=core),
        SimpleNamespace(organization=organization, query_batch=query), (0, 1, 2))
    assert [row["selected_value_members_E"] for row in rows] == [
        [{"original_id": 201, "weight": pytest.approx(0.1), "candidate": True}], [],
        [{"original_id": 201, "weight": pytest.approx(0.4), "candidate": True}],
    ]
    assert rows[1]["null_weights"]["M"] == 1.0
    assert rows[0]["omitted_final_member_ids_E"] == [202]
    assert rows[0]["context_stage_provenance"][0]["stage"] == "cycle1/me"
    changes = Task._route_weight_changes(captured, captured)
    assert changes["M_selected_ids"]["changed_query_member_pairs"] == 0
    assert changes["E_candidate_ids_unchanged"]


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
