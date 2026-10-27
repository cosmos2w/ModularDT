from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from channelthermal.joint_member_routing import ThermalJointMemberRoutingAdapter
from channelthermal.supplement_member_routing import (
    ThermalOrganizationConditionedMemberRoutingAdapter,
    ThermalSupplementMemberRoutingAdapter,
)
from channelthermal.training.supplement_member_routing_task import (
    ARM,
    B_ARM,
    B_WORKFLOW,
    CONFIG_SECTION_SCHEMA,
    PROVIDER_SCHEMA,
    SupplementMemberRoutingHealth,
)
from channelthermal.training.joint_member_routing_task import JointMemberRoutingHealth
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.interaction_reused_member_routing.joint import JointMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.consistent import (
    ConsistentJointMemberRoutingCore,
)
from honf_forward_core.interaction_reused_member_routing.organization_conditioned import (
    OrganizationConditionedMemberRoutingCore,
)


def test_supplement_cli_dispatches_maintained_evaluator(monkeypatch):
    import importlib.util
    import honf_runtime.supplement_member_routing as evaluation

    script = Path(__file__).parents[1] / "tools" / "honf_research.py"
    spec = importlib.util.spec_from_file_location("supplement_research_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(evaluation, "evaluate", lambda *args: calls.append(args) or {"status": "dispatched"})
    assert cli.main(["evaluate-supplement-member-routing", "--checkpoint", "epoch_0100.pt",
                     "--device", "cuda:0", "--output", "evaluation"]) == 0
    assert calls == [("epoch_0100.pt", "cuda:0", "evaluation")]


@pytest.mark.parametrize("changed", [None, "training_case_ids", "source_binding"])
def test_checkpoint_loader_keeps_saved_provider_data_identity(tmp_path, monkeypatch, changed):
    import copy
    from types import SimpleNamespace
    import honf_runtime.supplement_member_routing as evaluation
    import channelthermal.training.supplement_member_routing_task as task

    model = ThermalSupplementMemberRoutingAdapter(
        _normalization_stats(), core_config=_core_config(ConsistentJointMemberRoutingCore),
    )
    saved = {
        "schema": task.PROVIDER_SCHEMA, "architecture": task.ARM,
        "member_routing_schema": model.CORE_SCHEMA, "adapter_state": model.state_payload(),
        "recipe": {"name": "loader_fixture"}, "training_case_ids": ["0006"],
        "source_binding": {"dataset_path": "fixture.h5", "dataset_mtime_ns": 10},
    }
    rebuilt = copy.deepcopy(saved)
    if changed == "training_case_ids":
        rebuilt[changed] = ["0231"]
    elif changed == "source_binding":
        rebuilt[changed]["dataset_mtime_ns"] = 11
    provider = SimpleNamespace(identity_payload=lambda: rebuilt)
    monkeypatch.setattr(task, "build_thermal_supplement_member_routing_task",
                        lambda *args, **kwargs: (model, provider))
    payload = {"experiment_identity": {"workflow": task.WORKFLOW, "provider_identity": saved},
               "arm": task.ARM, "epoch": 6, "model_state_dict": model.state_dict()}
    monkeypatch.setattr(evaluation, "load_trusted_checkpoint", lambda *args, **kwargs: payload)
    path = tmp_path / "epoch_0006_model.pt"
    path.touch()
    if changed:
        with pytest.raises(ValueError, match="dataset, normalization or provider identity"):
            evaluation.load_checkpoint(path, "cpu")
    else:
        loaded, loaded_provider, _ = evaluation.load_checkpoint(path, "cpu")
        assert loaded_provider.model is loaded
        assert loaded_provider.adapter is loaded
        assert not loaded.training


def test_organization_conditioned_checkpoint_dispatch_rebuilds_b_strictly(tmp_path, monkeypatch):
    import copy
    from types import SimpleNamespace
    import honf_runtime.supplement_member_routing as evaluation
    import channelthermal.training.supplement_member_routing_task as task

    model = ThermalOrganizationConditionedMemberRoutingAdapter(
        _normalization_stats(), core_config=_core_config(OrganizationConditionedMemberRoutingCore),
    )
    saved = {
        "schema": PROVIDER_SCHEMA,
        "architecture": B_ARM,
        "member_routing_schema": model.CORE_SCHEMA,
        "candidate": "B",
        "adapter_state": model.state_payload(),
        "recipe": {"name": "loader_fixture_b"},
        "training_case_ids": ["0006"],
        "source_binding": {"dataset_path": "fixture.h5", "dataset_mtime_ns": 10},
    }
    rebuilt = copy.deepcopy(saved)
    provider = SimpleNamespace(identity_payload=lambda: rebuilt)
    monkeypatch.setattr(
        task,
        "build_thermal_supplement_member_routing_task",
        lambda *args, **kwargs: (model, provider),
    )
    payload = {
        "experiment_identity": {
            "workflow": B_WORKFLOW,
            "provider_identity": saved,
        },
        "arm": B_ARM,
        "epoch": 6,
        "model_state_dict": model.state_dict(),
    }
    monkeypatch.setattr(evaluation, "load_trusted_checkpoint", lambda *args, **kwargs: payload)
    path = tmp_path / "epoch_0006_model_b.pt"
    path.touch()

    parts = evaluation._checkpoint_parts(payload)
    assert parts[-1]["candidate"] == "B"
    loaded, loaded_provider, loaded_payload = evaluation.load_checkpoint(path, "cpu")
    assert isinstance(loaded, ThermalOrganizationConditionedMemberRoutingAdapter)
    assert loaded.architecture == B_ARM
    assert loaded.core.schema == model.CORE_SCHEMA
    assert loaded_provider.model is loaded
    assert loaded_provider.adapter is loaded
    assert loaded_payload["candidate"] == "B"
    assert not loaded.training

    rebuilt["training_case_ids"] = ["0231"]
    with pytest.raises(ValueError, match="dataset, normalization or provider identity"):
        evaluation.load_checkpoint(path, "cpu")
    rebuilt["training_case_ids"] = ["0006"]

    incomplete = dict(payload)
    incomplete["model_state_dict"] = dict(model.state_dict())
    incomplete["model_state_dict"].pop(next(iter(incomplete["model_state_dict"])))
    monkeypatch.setattr(evaluation, "load_trusted_checkpoint", lambda *args, **kwargs: incomplete)
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluation.load_checkpoint(path, "cpu")


def test_task_variant_dispatch_selects_organization_conditioned_adapter(monkeypatch):
    import channelthermal.training.supplement_member_routing_task as task
    from channelthermal.training.member_routing_task import ThermalInteractionReusedMemberRoutingTask
    from channelthermal.training.joint_member_routing_task import ThermalJointMemberRoutingTask

    variant = task._SUPPLEMENT_VARIANTS[OrganizationConditionedMemberRoutingCore.schema]
    assert variant["adapter"] is ThermalOrganizationConditionedMemberRoutingAdapter
    assert variant["arm"] == B_ARM
    assert variant["workflow"] == B_WORKFLOW
    assert variant["candidate"] == "B"

    def fake_base_init(self, **kwargs):
        self.model = kwargs["model"]
        self.recipe = kwargs["recipe"]

    model = ThermalOrganizationConditionedMemberRoutingAdapter(
        _normalization_stats(), core_config=_core_config(OrganizationConditionedMemberRoutingCore),
    )
    recipe = {
        "initialization": {
            "source_kind": "scratch_r13_5_supplement",
            "source_parent_age": 0,
            "source_checkpoint": None,
            "pretrained_weights": False,
        },
        "training": {"sampling_arm": "R13-3-Relative"},
        "member_routing": {},
    }
    monkeypatch.setattr(ThermalInteractionReusedMemberRoutingTask, "__init__", fake_base_init)
    monkeypatch.setattr(ThermalJointMemberRoutingTask, "identity_payload", lambda self: {"objective": {}})
    provider = task.ThermalSupplementMemberRoutingTask(model=model, recipe=recipe)
    assert provider.candidate == "B"
    assert provider.arm == B_ARM
    assert provider.workflow == B_WORKFLOW
    assert provider.provider_schema == PROVIDER_SCHEMA
    identity = provider.identity_payload()
    assert identity["candidate"] == "B"
    assert identity["workflow"] == B_WORKFLOW
    assert identity["architecture"] == B_ARM
    assert identity["member_routing_schema"] == model.CORE_SCHEMA
    assert identity["contribution_rule"] == task.B_CONTRIBUTION_RULE


def _base_config():
    return ContentHypergraphCoreV2(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=12,
        output_dim=4,
        architecture="A2-D",
        context_dim=16,
        num_heads=4,
        context_depth=2,
        local_hidden_dim=16,
        dropout=0.0,
        init_seed=731,
    ).get_config()


def _normalization_stats():
    return {
        "field_mean_by_channel": np.zeros(5, dtype=np.float32),
        "field_std_by_channel": np.ones(5, dtype=np.float32),
        "interface_targets_mean": np.zeros(2, dtype=np.float32),
        "interface_targets_std": np.ones(2, dtype=np.float32),
        "internal_temperature_std": np.ones(1, dtype=np.float32),
    }


def _core_config(core_type):
    if core_type is JointMemberRoutingCore:
        core = core_type(_base_config(), address_mode="ME", input_dataset="thermal", query_tile_size=8)
    else:
        core = core_type(
            _base_config(), address_mode="ME", input_dataset="thermal", query_tile_size=8,
            retrieval_calibration="normalized", score_calibration="inherited",
        )
    return core.get_config()


def test_supplement_adapter_roundtrip_keeps_compatible_scratch_tensors():
    stats = _normalization_stats()
    torch.manual_seed(903)
    old = ThermalJointMemberRoutingAdapter(stats, core_config=_core_config(JointMemberRoutingCore))
    torch.manual_seed(903)
    new = ThermalSupplementMemberRoutingAdapter(
        stats, core_config=_core_config(ConsistentJointMemberRoutingCore),
    )

    assert new.architecture == ARM
    assert new.core.address_mode == "ME"
    assert tuple(old.state_dict()) == tuple(new.state_dict())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name

    restored = ThermalSupplementMemberRoutingAdapter.from_state_payload(
        new.state_payload(), state_dict=new.state_dict(),
    )
    assert restored.get_config() == new.get_config()
    for name, value in new.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    with pytest.raises(ValueError, match="Not a Thermal"):
        ThermalJointMemberRoutingAdapter.from_state_payload(new.state_payload())


def test_organization_conditioned_adapter_keeps_the_same_194_scratch_tensors():
    stats = _normalization_stats()
    torch.manual_seed(903)
    a_model = ThermalSupplementMemberRoutingAdapter(
        stats, core_config=_core_config(ConsistentJointMemberRoutingCore),
    )
    torch.manual_seed(903)
    b_model = ThermalOrganizationConditionedMemberRoutingAdapter(
        stats, core_config=_core_config(OrganizationConditionedMemberRoutingCore),
    )

    assert a_model.CORE_SCHEMA == "interaction_reused_consistent_joint_member_routing_v1"
    assert b_model.SCHEMA == "thermal_r13_5_organization_conditioned_member_routing_adapter_v1"
    assert b_model.CORE_SCHEMA == "interaction_reused_organization_conditioned_member_routing_v1"
    assert b_model.architecture == B_ARM
    assert len(a_model.state_dict()) == len(b_model.state_dict()) == 194
    assert len(a_model.core.state_dict()) == len(b_model.core.state_dict()) == 190
    assert tuple(a_model.state_dict()) == tuple(b_model.state_dict())
    for name, value in a_model.state_dict().items():
        assert torch.equal(value, b_model.state_dict()[name]), name

    restored = ThermalOrganizationConditionedMemberRoutingAdapter.from_state_payload(
        b_model.state_payload(), state_dict=b_model.state_dict(),
    )
    assert restored.get_config() == b_model.get_config()
    for name, value in b_model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    with pytest.raises(ValueError, match="Not a Thermal"):
        ThermalSupplementMemberRoutingAdapter.from_state_payload(b_model.state_payload())


def test_whole_joint_contribution_mass_is_normalized_before_typed_strengths():
    before_m = torch.tensor([[[0.25, 0.25], [0.0, 0.0], [0.0, 0.0]]])
    before_e = torch.tensor([[[0.25, 0.25], [0.4, 0.6], [0.0, 0.0]]])
    after_m = before_m * 0.6
    after_e = before_e * 0.4
    summary = SupplementMemberRoutingHealth._mass_summary({
        "member_contribution_pre_strength_M": before_m,
        "member_contribution_pre_strength_E": before_e,
        "member_contribution_M": after_m,
        "member_contribution_E": after_e,
    })
    assert summary["joint_pre_strength_mass_mean"] == pytest.approx(2 / 3)
    assert summary["joint_pre_strength_mass_max_abs_error"] == pytest.approx(0.0)
    assert summary["empty_joint_query_count"] == 1
    assert summary["role_conditional_mass_max_abs_error"] == {"M": pytest.approx(0.0), "E": pytest.approx(0.0)}
    assert summary["role_post_strength_mass_mean"]["M"] == pytest.approx(0.1)
    assert summary["role_post_strength_mass_mean"]["E"] == pytest.approx(0.2)


def test_whole_joint_contribution_mass_rejects_rolewise_renormalization():
    with pytest.raises(RuntimeError, match="sum to one jointly across M and E"):
        SupplementMemberRoutingHealth._mass_summary({
            "member_contribution_pre_strength_M": torch.tensor([[[1.0, 0.0]]]),
            "member_contribution_pre_strength_E": torch.tensor([[[1.0, 0.0]]]),
            "member_contribution_M": torch.tensor([[[0.5, 0.0]]]),
            "member_contribution_E": torch.tensor([[[0.5, 0.0]]]),
        })


def test_health_releases_graph_traces_after_successful_summary(monkeypatch):
    trace = {
        "member_contribution_pre_strength_M": torch.tensor([[[1.0, 0.0]]]),
        "member_contribution_pre_strength_E": torch.zeros(1, 1, 2),
        "member_contribution_M": torch.tensor([[[0.5, 0.0]]]),
        "member_contribution_E": torch.zeros(1, 1, 2),
    }

    def fake_observe(self, model, case_id):
        del model, case_id
        self._captured_traces.update({"full": trace, "restricted": trace})
        return {"modes": {"full": {}, "restricted": {}}}

    monkeypatch.setattr(JointMemberRoutingHealth, "_observe_case", fake_observe)
    health = object.__new__(SupplementMemberRoutingHealth)
    health._captured_traces = {}

    result = health._observe_case(None, "0003")

    assert "joint_contribution_mass" in result["modes"]["full"]
    assert health._captured_traces == {}


def test_health_releases_graph_traces_after_failed_summary(monkeypatch):
    bad_trace = {
        "member_contribution_pre_strength_M": torch.tensor([[[1.0, 0.0]]]),
        "member_contribution_pre_strength_E": torch.tensor([[[1.0, 0.0]]]),
        "member_contribution_M": torch.tensor([[[0.5, 0.0]]]),
        "member_contribution_E": torch.tensor([[[0.5, 0.0]]]),
    }

    def fake_observe(self, model, case_id):
        del model, case_id
        self._captured_traces.update({"full": bad_trace, "restricted": bad_trace})
        return {"modes": {"full": {}, "restricted": {}}}

    monkeypatch.setattr(JointMemberRoutingHealth, "_observe_case", fake_observe)
    health = object.__new__(SupplementMemberRoutingHealth)
    health._captured_traces = {}

    with pytest.raises(RuntimeError, match="sum to one jointly"):
        health._observe_case(None, "0003")
    assert health._captured_traces == {}


def _organization_conditioned_health_trace():
    return {
        "member_contribution_pre_strength_M": torch.tensor([[[0.25, 0.25], [0.2, 0.1]]]),
        "member_contribution_pre_strength_E": torch.tensor([[[0.5], [0.7]]]),
        "member_contribution_M": torch.tensor([[[0.125, 0.125], [0.1, 0.05]]]),
        "member_contribution_E": torch.tensor([[[0.25], [0.35]]]),
        "proposal_weights_M": torch.ones(1, 2, 2) / 2,
        "proposal_weights_E": torch.ones(1, 2, 1),
        "organization_conditioned_retained_view_mass": torch.tensor(
            [[[0.4, 0.6, 0.0], [0.0, 0.2, 0.8]]]
        ),
        "organization_conditioned_scaled_view_likelihood_mass": torch.tensor(
            [[[0.2, 0.3, 0.0], [0.0, 0.1, 0.4]]]
        ),
        "organization_conditioned_retained_prior_mass": torch.tensor([[1.0, 0.7]]),
        "organization_conditioned_effective_view_weights": torch.tensor(
            [[[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]]]
        ),
        "organization_conditioned_view_mixture_mass": torch.tensor(
            [[[0.25, 0.25, 0.0], [0.0, 0.175, 0.525]]]
        ),
        "organization_conditioned_pre_normalization_mass": torch.ones(1, 2),
        "organization_conditioned_pre_normalization_error": torch.zeros(1, 2),
    }


def test_organization_conditioned_health_reports_compact_view_statistics_and_releases_trace(monkeypatch):
    trace = _organization_conditioned_health_trace()

    def fake_observe(self, model, case_id):
        del model, case_id
        self._captured_traces.update({"full": trace, "restricted": trace})
        return {"modes": {"full": {}, "restricted": {}}}

    monkeypatch.setattr(JointMemberRoutingHealth, "_observe_case", fake_observe)
    health = object.__new__(SupplementMemberRoutingHealth)
    health._captured_traces = {}

    result = health._observe_case(None, "0003")
    summary = result["modes"]["full"]["joint_contribution_mass"]["organization_conditioned_view_summary"]

    assert summary["retained_view_mass_d"]["all"]["count"] == 6
    assert summary["retained_view_mass_d"]["M"]["sum"] == pytest.approx(1.2)
    assert summary["retained_view_mass_d"]["E"]["sum"] == pytest.approx(0.8)
    assert summary["scaled_view_likelihood_mass_Z"]["all"]["sum"] == pytest.approx(1.0)
    assert summary["retained_prior_mass_D"]["mean"] == pytest.approx(0.85)
    assert summary["pre_normalization_mass_minus_one"]["max"] == pytest.approx(0.0)
    assert health._captured_traces == {}


def test_organization_conditioned_health_failure_releases_graph_trace(monkeypatch):
    trace = _organization_conditioned_health_trace()
    trace["organization_conditioned_effective_view_weights"] = torch.zeros(1, 2, 2)

    def fake_observe(self, model, case_id):
        del model, case_id
        self._captured_traces.update({"full": trace, "restricted": trace})
        return {"modes": {"full": {}, "restricted": {}}}

    monkeypatch.setattr(JointMemberRoutingHealth, "_observe_case", fake_observe)
    health = object.__new__(SupplementMemberRoutingHealth)
    health._captured_traces = {}

    with pytest.raises(RuntimeError, match="view statistics do not share the M/E address inventory"):
        health._observe_case(None, "0003")
    assert health._captured_traces == {}


@pytest.mark.parametrize(
    "filename,membership,horizon,normalization",
    (
        ("thermal_r13_5a_consistent_joint95_qual_train150_v1.json", "fixed25_train150_dev22", 6,
         "fixed25_train150_stats"),
        ("thermal_r13_5a_consistent_joint95_train600_formal_v1.json", "full_train600", 100,
         "scratch_train600_stats"),
    ),
)
def test_supplement_configs_bind_fresh_matched_contract(filename, membership, horizon, normalization):
    path = Path(__file__).parents[1] / "configs" / "research" / filename
    config = json.loads(path.read_text())
    section = config["member_routing"]
    model = section["model"]

    assert section["arm"] == model["architecture"] == ARM
    assert section["schema"] == "thermal_r13_5_supplement_member_routing_v1"
    assert model["schema"] == "interaction_reused_consistent_joint_member_routing_v1"
    assert model["address_mode"] == "ME"
    assert model["retrieval_calibration"] == "normalized"
    assert model["score_calibration"] == "inherited"
    assert config["initialization"]["source_kind"] == "scratch_r13_5_supplement"
    assert config["initialization"]["source_checkpoint"] is None
    assert config["initialization"]["pretrained_weights"] is False
    assert config["training"]["membership"] == membership
    assert config["training"]["local_horizon"] == horizon
    assert config["training"]["first_segment_epochs"] == horizon
    assert config["training"]["optimizer_schedule_horizon"] == horizon
    assert (config["training"]["effective_batch_size"], config["training"]["microbatch_cases"],
            config["training"]["primary_queries"]) == (48, 24, 1024)
    assert config["training"]["checkpoint_epochs"] == [horizon]
    assert config["training"]["curve_every"] == 10
    assert config["training"]["progress_enabled"] is True
    assert config["initialization"]["normalization_mode"] == normalization
    assert config["optimizer"]["lr"] == pytest.approx(3e-4)
    assert config["optimizer"]["schedule"] == "constant_config_lr"
    assert config["storage"]["output_root"] == "/data2/wanglz/ModularDT/round13_5_supplement_20261027"
    assert config["storage"]["max_peak_temporary_bytes"] == 256 * 1024**2

    from honf_runtime.contextual_frontier_training import (
        load_contextual_frontier_config, _validate_loaded_contextual_config,
        _member_routing_task_factory, _member_routing_workflow, _run_identity,
    )
    loaded, _ = load_contextual_frontier_config(path)
    _validate_loaded_contextual_config(loaded)
    assert _member_routing_workflow(loaded["member_routing"]) == (
        "interaction_reused_consistent_joint_member_routing_training_v1"
    )
    assert _member_routing_task_factory(loaded).__name__ == "build_thermal_supplement_member_routing_task"
    identity = _run_identity(loaded, None)
    assert identity["schema"] == "honf_supplement_member_routing_training_v1"


@pytest.mark.parametrize(
    "b_filename,a_filename,membership,horizon,normalization",
    (
        ("thermal_r13_5b_organization_conditioned_joint95_qual_train150_v1.json",
         "thermal_r13_5a_consistent_joint95_qual_train150_v1.json",
         "fixed25_train150_dev22", 6, "fixed25_train150_stats"),
        ("thermal_r13_5b_organization_conditioned_joint95_train600_formal_v1.json",
         "thermal_r13_5a_consistent_joint95_train600_formal_v1.json",
         "full_train600", 100, "scratch_train600_stats"),
    ),
)
def test_organization_conditioned_configs_are_matched_fresh_lineages(
        b_filename, a_filename, membership, horizon, normalization):
    config_dir = Path(__file__).parents[1] / "configs" / "research"
    b_config = json.loads((config_dir / b_filename).read_text())
    a_config = json.loads((config_dir / a_filename).read_text())
    section = b_config["member_routing"]
    model = section["model"]

    assert section["schema"] == CONFIG_SECTION_SCHEMA
    assert section["arm"] == model["architecture"] == B_ARM
    assert model["schema"] == OrganizationConditionedMemberRoutingCore.schema
    assert model["address_mode"] == "ME"
    assert model["retrieval_calibration"] == "normalized"
    assert model["score_calibration"] == "inherited"
    assert b_config["initialization"]["source_kind"] == "scratch_r13_5_supplement"
    assert b_config["initialization"]["source_checkpoint"] is None
    assert b_config["initialization"]["pretrained_weights"] is False
    assert b_config["initialization"]["seed"] == a_config["initialization"]["seed"] == 0
    assert b_config["initialization"]["normalization_mode"] == normalization
    assert b_config["training"]["membership"] == membership
    assert b_config["training"]["local_horizon"] == horizon
    assert b_config["training"]["first_segment_epochs"] == horizon
    assert b_config["training"]["optimizer_schedule_horizon"] == horizon
    assert (b_config["training"]["effective_batch_size"], b_config["training"]["microbatch_cases"],
            b_config["training"]["primary_queries"]) == (48, 24, 1024)
    assert b_config["training"]["checkpoint_epochs"] == [horizon]
    assert b_config["optimizer"]["lr"] == pytest.approx(3e-4)

    from honf_runtime.contextual_frontier_training import (
        load_contextual_frontier_config, _validate_loaded_contextual_config,
        _member_routing_task_factory, _member_routing_workflow, _run_identity,
    )
    loaded, _ = load_contextual_frontier_config(config_dir / b_filename)
    _validate_loaded_contextual_config(loaded)
    assert _member_routing_workflow(loaded["member_routing"]) == B_WORKFLOW
    assert _member_routing_task_factory(loaded).__name__ == "build_thermal_supplement_member_routing_task"
    identity = _run_identity(loaded, None)
    assert identity["schema"] == "honf_supplement_member_routing_training_v1"
    assert identity["workflow"] == B_WORKFLOW
    assert identity["model_family"] == "organized_joint95_member_routing_joint_field_v1"

    matched = json.loads(json.dumps(b_config))
    matched["name"] = a_config["name"]
    matched["member_routing"]["arm"] = a_config["member_routing"]["arm"]
    matched["member_routing"]["model"]["schema"] = a_config["member_routing"]["model"]["schema"]
    matched["member_routing"]["model"]["architecture"] = a_config["member_routing"]["model"]["architecture"]
    assert matched == a_config
