from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.joint_member_routing import ThermalJointMemberRoutingAdapter
from channelthermal.member_routing import (
    ThermalFaithfulMemberRoutingAdapter,
    ThermalInteractionReusedMemberRoutingAdapter,
    ThermalTypedOrganizationMemberRoutingAdapter,
)
from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
from channelthermal.training.r13_6_member_routing_task import (
    ThermalR136MemberRoutingTask,
)
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing.faithful import FaithfulMECore
from honf_forward_core.interaction_reused_member_routing.joint import JointMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.typed_organization_core import (
    TypedOrganizationMemberRoutingCore,
)
from honf_runtime.contextual_frontier_training import (
    _member_routing_task_factory,
    _run_identity,
    _validate_loaded_contextual_config,
    load_contextual_frontier_config,
)


ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs" / "research"
CONFIG_NAMES = (
    "thermal_r13_6c_faithful_train150_6_v1.json",
    "thermal_r13_6c_faithful_train600_formal_v1.json",
    "thermal_r13_6h_typed_organization_train150_6_v1.json",
    "thermal_r13_6h_typed_organization_train600_formal_v1.json",
)
O_ARM = "R13-6-O-ME-FaithfulJoint95"
H_ARM = "R13-6-H-ME-TypedOrganizationJoint95"
O_WORKFLOW = "interaction_reused_faithful_member_routing_training_v1"
H_WORKFLOW = "interaction_reused_typed_organization_member_routing_training_v1"


def _base_config() -> dict:
    return ContentHypergraphCoreV2(
        module_input_dim=9, environment_input_dim=6, coordinate_dim=2,
        query_input_dim=0, operating_input_dim=12, output_dim=4,
        architecture="A2-D", context_dim=16, num_heads=4, context_depth=2,
        local_hidden_dim=16, dropout=0.0, init_seed=731,
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
    return core_type(
        _base_config(), address_mode="ME", input_dataset="thermal",
        input_capability="joint_prescribed_records_v1", query_tile_size=8,
    ).get_config()


def _scene_query():
    torch.manual_seed(13606)
    scene = SceneBatch(
        module_features=torch.randn(1, 2, 9),
        environment_features=torch.randn(1, 2, 6),
        module_coordinates=torch.tensor([[[0.0, 0.0], [0.7, 0.1]]]),
        environment_coordinates=torch.tensor([[[0.2, 0.4], [0.9, 0.4]]]),
        module_mask=torch.ones(1, 2, dtype=torch.bool),
        environment_mask=torch.ones(1, 2, dtype=torch.bool),
        module_measure=torch.ones(1, 2), environment_measure=torch.ones(1, 2),
        module_ids=torch.tensor([[11, 12]]), environment_ids=torch.tensor([[21, 22]]),
        local_radius=torch.full((1, 2), 0.2), operating_features=torch.randn(1, 12),
        coordinate_origin=torch.zeros(1, 2), coordinate_scale=torch.ones(1, 2),
        module_measure_reference=torch.ones(1), environment_measure_reference=torch.ones(1),
        scene_key="r13-6-integration-scene", normalizer_key="r13-6-integration-normalizer",
        coordinate_frame_key="r13-6-integration-frame",
        module_characteristic_lengths=torch.full((1, 2, 2), 0.1),
        environment_characteristic_lengths=torch.full((1, 2, 2), 0.1),
    )
    query = QueryBatch(
        coordinates=torch.tensor([[[0.1, 0.1], [0.8, 0.2]]]),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
        query_ids=torch.tensor([[101, 102]]),
        scene_key="r13-6-integration-scene", coordinate_frame_key="r13-6-integration-frame",
    )
    return scene, query


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_r13_6_configs_are_strictly_routed_and_match_declared_horizons(name):
    config, _path = load_contextual_frontier_config(CONFIGS / name)
    _validate_loaded_contextual_config(config)
    section = config["member_routing"]
    core_schema = section["model"]["schema"]
    expected_arm = O_ARM if core_schema == FaithfulMECore.schema else H_ARM
    expected_workflow = O_WORKFLOW if expected_arm == O_ARM else H_WORKFLOW
    assert section["arm"] == expected_arm
    assert config["training"]["curve_every"] == 10
    assert config["execution"]["preferred_devices"] == ["cuda:0"]
    assert _member_routing_task_factory(config).__name__ == "build_thermal_r13_6_member_routing_task"

    fake_provider = SimpleNamespace(identity_payload=lambda: {})
    identity = _run_identity(config, fake_provider)
    assert identity["workflow"] == expected_workflow
    assert identity["training_membership"] == config["training"]["membership"]
    if config["training"]["membership"] == "fixed25_train150_dev22":
        assert (config["training"]["local_horizon"], config["training"]["first_segment_epochs"],
                config["training"]["optimizer_schedule_horizon"],
                config["training"]["checkpoint_epochs"]) == (6, 3, 6, [3, 6])
    else:
        assert (config["training"]["local_horizon"], config["training"]["first_segment_epochs"],
                config["training"]["optimizer_schedule_horizon"],
                config["training"]["checkpoint_epochs"]) == (100, 100, 500, [100])


def test_c_h_adapters_keep_original_joint_parameter_inventory_and_load_strictly():
    stats = _normalization_stats()
    baseline = ThermalJointMemberRoutingAdapter(stats, core_config=_core_config(JointMemberRoutingCore))
    torch.manual_seed(90136)
    o_model = ThermalFaithfulMemberRoutingAdapter(stats, core_config=_core_config(FaithfulMECore))
    torch.manual_seed(90136)
    h_model = ThermalTypedOrganizationMemberRoutingAdapter(
        stats, core_config=_core_config(TypedOrganizationMemberRoutingCore),
    )

    assert tuple(baseline.state_dict()) == tuple(o_model.state_dict()) == tuple(h_model.state_dict())
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, o_model.state_dict()[name]), name
        assert torch.equal(tensor, h_model.state_dict()[name]), name
    assert np.array_equal(o_model.normalization_stats["field_mean_by_channel"],
                          h_model.normalization_stats["field_mean_by_channel"])

    o_restored = ThermalFaithfulMemberRoutingAdapter.from_state_payload(
        o_model.state_payload(), state_dict=o_model.state_dict(),
    )
    h_restored = ThermalTypedOrganizationMemberRoutingAdapter.from_state_payload(
        h_model.state_payload(), state_dict=h_model.state_dict(),
    )
    assert o_restored.core.get_config() == o_model.core.get_config()
    assert h_restored.core.get_config() == h_model.core.get_config()
    with pytest.raises(ValueError, match="Not a Thermal"):
        ThermalInteractionReusedMemberRoutingAdapter.from_state_payload(h_model.state_payload())


def test_h_member_read_uses_cross_role_addresses_without_changing_o_gate_law():
    core = TypedOrganizationMemberRoutingCore(_base_config())
    retrieval = torch.tensor([[[0.9, 0.1, 0.1, 0.9],
                               [0.2, 0.8, 0.3, 0.7],
                               [0.1, 0.9, 0.4, 0.6],
                               [0.8, 0.2, 0.7, 0.3]]])
    # Both query-address weight and M-role support live only at E-address row 2.
    proposal = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    prior = torch.tensor([[[0.25, 0.75, 0.60, 0.40]]])
    ell = torch.zeros_like(prior)
    candidate = torch.ones_like(prior, dtype=torch.bool)
    gate, value, trace = core._role_read_contributions(
        torch.zeros_like(prior), prior, ell, candidate,
        full_candidate=candidate, proposal=proposal, retrieval_R=retrieval,
        row_mass=torch.ones(1, 4), module_count=2,
    )
    torch.testing.assert_close(gate, prior)
    torch.testing.assert_close(value[..., :2], torch.tensor([[[0.1, 0.9]]]))
    torch.testing.assert_close(value[..., 2:], torch.tensor([[[0.4, 0.6]]]))
    torch.testing.assert_close(trace["typed_organization_member_contribution_M"], value[..., :2])


def test_h_decode_reports_three_contractions_summed_over_both_roles():
    core = TypedOrganizationMemberRoutingCore(_base_config(), query_tile_size=8)
    scene, query = _scene_query()
    prepared = core.prepare(scene)
    full, restricted, shared = core.decode_paired(prepared, query)
    assert set(shared) >= {"query_M_proposal_score_rows", "query_E_proposal_score_rows"}
    for output in (full, restricted):
        counts = output.work_counts
        keys = (
            "query_H_retained_view_mass_contraction_pairs",
            "query_H_view_likelihood_mass_contraction_pairs",
            "query_H_coalesced_member_base_contraction_pairs",
        )
        assert all(counts[key] > 0 for key in keys)
        assert len({counts[key] for key in keys}) == 1
        assert sum(counts[key] for key in keys) == 3 * counts[keys[0]]
    # Full execution sees A=M+E=4 address rows and N=M+E=4 members.
    expected_per_contraction = query.batch_size * query.num_queries * 4 * 4
    assert full.work_counts[keys[0]] == expected_per_contraction
    assert sum(full.work_counts[key] for key in keys) == 3 * expected_per_contraction


def test_r13_6_training_task_calls_actual_paired_core_and_counts_shared_work_once():
    calls = []

    class FakeCore:
        def decode_paired(self, organization, query_batch):
            calls.append((organization, query_batch))
            output = SimpleNamespace(
                joint_fields=torch.ones(1, 2, 4),
                work_counts={"native_member_pairs": 7},
            )
            restricted = SimpleNamespace(
                joint_fields=torch.zeros(1, 2, 4),
                work_counts={"native_member_pairs": 3},
            )
            return output, restricted, {"shared_query_rows": 2}

    class FakeModel:
        CORE_SCHEMA = ThermalFaithfulMemberRoutingAdapter.CORE_SCHEMA
        architecture = O_ARM
        core = FakeCore()

        def validate_prepared(self, prepared):
            assert prepared.organization is organization
            self.validated = True

        def _extract_native_roles(self, prepared, temperature, flow):
            del prepared
            return torch.cat((flow, temperature), dim=-1)

    organization = SimpleNamespace(work_counts={"prepared_pair_rows": 4})
    prepared = SimpleNamespace(
        organization=organization,
        scene_batch=SimpleNamespace(module_features=torch.zeros(1, 1, 1)),
        structure={"heat_powers": torch.zeros(1, 2)},
        fluid_xy=torch.zeros(1, 2, 2),
        union=SimpleNamespace(unique_counts=torch.tensor([5])),
    )
    prepared.query_batch = SimpleNamespace(local_weights=torch.ones(1, 2, 3))
    task = object.__new__(ThermalR136MemberRoutingTask)
    task.model = FakeModel()
    paired = task._paired_predictions(prepared, None)

    assert isinstance(paired, ThermalJointPairPredictions)
    assert calls == [(organization, prepared.query_batch)]
    assert task.model.validated is True
    assert paired.work["core_decode_joint_shared_shared_query_rows"] == 2
    assert paired.work["core_prepare_primary_prepared_pair_rows"] == 4
    assert paired.work["core_decode_joint_full_native_member_pairs"] == 7
    assert paired.work["core_decode_joint_restricted_native_member_pairs"] == 3


@pytest.mark.parametrize("core_type,adapter_type,workflow,provider_schema,provider_field", [
    (FaithfulMECore, ThermalFaithfulMemberRoutingAdapter, O_WORKFLOW,
     "thermal_r13_6_faithful_member_routing_provider_v1", "R13-6-O-ME-FaithfulJoint95"),
    (TypedOrganizationMemberRoutingCore, ThermalTypedOrganizationMemberRoutingAdapter, H_WORKFLOW,
     "thermal_r13_6_typed_organization_member_routing_provider_v1",
     "R13-6-H-ME-TypedOrganizationJoint95"),
])
def test_maintained_checkpoint_loader_reconstructs_each_r13_6_arm_strictly(
    tmp_path, monkeypatch, core_type, adapter_type, workflow, provider_schema, provider_field,
):
    import honf_runtime.contextual_frontier_evaluation as evaluation
    import honf_runtime.contextual_frontier_training as training

    model = adapter_type(_normalization_stats(), core_config=_core_config(core_type))
    provider_identity = {
        "schema": provider_schema,
        "architecture": provider_field,
        "member_routing_schema": core_type.schema,
        "adapter_state": model.state_payload(),
        "recipe": {"fixture": "strict-r13-6"},
    }
    payload = {
        "experiment_identity": {
            "workflow": workflow,
            "provider_identity": provider_identity,
        },
        "arm": provider_field,
        "epoch": 100,
        "model_state_dict": model.state_dict(),
    }
    provider = SimpleNamespace(model=model, adapter=model, device=torch.device("cpu"))
    monkeypatch.setattr(evaluation, "load_trusted_checkpoint", lambda *args, **kwargs: payload)
    monkeypatch.setattr(training, "_build_task", lambda *args, **kwargs: (model, provider))
    checkpoint = tmp_path / "epoch_0100_model.pt"
    checkpoint.touch()

    restored, restored_provider, restored_payload = evaluation.load_contextual_checkpoint(checkpoint, "cpu")
    assert isinstance(restored, adapter_type)
    assert restored.core.schema == core_type.schema
    assert restored_provider.model is restored
    assert restored_provider.adapter is restored
    assert restored_payload["checkpoint_path"] == str(checkpoint.resolve())
    assert not restored.training

    original_state = payload["model_state_dict"]
    incomplete = dict(original_state)
    incomplete.pop(next(iter(incomplete)))
    payload["model_state_dict"] = incomplete
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluation.load_contextual_checkpoint(checkpoint, "cpu")


def test_all_val89_evaluator_runs_full_and_restricted_native_panels(tmp_path, monkeypatch):
    import honf_runtime.r13_6_member_routing_evaluation as evaluation

    output_root = tmp_path / "round-output"
    output_root.mkdir()
    validation_ids = ["0274", "0293", "0678"] + [f"v{index:02d}" for index in range(86)]
    training_ids = ["0003"] + [f"t{index:03d}" for index in range(599)]
    recipe = {
        "storage": {
            "output_root": str(output_root), "max_output_bytes": 1_000_000,
            "max_peak_temporary_bytes": 1_000_000, "minimum_free_headroom_bytes": 0,
        },
        "evaluation": {"max_output_bytes": 1_000_000},
    }
    provider = SimpleNamespace(
        training_case_ids=training_ids,
        validation_case_ids=validation_ids,
        validation_scope="canonical_VAL89_exposed",
        recipe=recipe,
    )

    class FakeCore:
        schema = TypedOrganizationMemberRoutingCore.schema
        address_mode = "ME"
        joint_member_coverage = 0.95
        retained_prior_law = "none"

        @contextlib.contextmanager
        def read_execution(self, mode):
            yield mode

    model = SimpleNamespace(
        architecture=H_ARM,
        core=FakeCore(),
    )
    payload = {"epoch": 100, "experiment_identity": {"workflow": H_WORKFLOW,
                                                         "training_membership": "full_train600"}}
    calls = []

    def fake_evaluate(_model, _provider, _payload, split, case_ids, *, export_ids):
        ids = tuple(case_ids)
        calls.append((split, ids, set(export_ids)))
        return {
            "case_count": len(ids),
            "representative_arrays": {
                case_id: {"field": np.asarray([1.0], dtype=np.float32)} for case_id in export_ids
            },
        }

    monkeypatch.setattr(evaluation, "load_contextual_checkpoint",
                        lambda *args: (model, provider, payload))
    monkeypatch.setattr(evaluation, "_storage_snapshot", lambda *_args, **_kwargs: {"fixture": True})
    monkeypatch.setattr(evaluation, "_evaluate_native_split", fake_evaluate)
    output = output_root / "eval_H"
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.touch()
    result = evaluation.evaluate(checkpoint, "cpu", output)

    assert result["status"] == "completed"
    assert len(calls) == 4
    assert all(len(ids) == 89 for split, ids, _exports in calls if split == "validation")
    assert all(ids == ("0003",) for split, ids, _exports in calls if split == "train")
    assert all(exports == {"0274", "0293", "0678"} for split, _ids, exports in calls
               if split == "validation")
    assert all(exports == {"0003"} for split, _ids, exports in calls if split == "train")
    assert (output / "result.json").is_file()
    assert len(list(output.rglob("*.npz"))) == 8


@pytest.mark.parametrize("name,workflow,manifest_schema", [
    (CONFIG_NAMES[1], O_WORKFLOW, "honf_r13_6_faithful_member_routing_run_manifest_v1"),
    (CONFIG_NAMES[3], H_WORKFLOW, "honf_r13_6_typed_organization_member_routing_run_manifest_v1"),
])
def test_resume_member_routing_extends_the_same_epoch100_lineage_to500(
    tmp_path, monkeypatch, name, workflow, manifest_schema,
):
    import honf_runtime.compat as compat
    import honf_runtime.contextual_frontier_training as training

    config, config_path = load_contextual_frontier_config(CONFIGS / name)
    arm = config["member_routing"]["arm"]
    run = tmp_path / arm
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "latest_model.pt").touch()
    manifest = {
        "schema": manifest_schema,
        "artifact_layout_version": 1,
        "config": config,
        "config_path": str(config_path),
        "arm": arm,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    latest_payload = {
        "epoch": 100,
        "arm": arm,
        "experiment_identity": {"workflow": workflow, "training_membership": "full_train600"},
    }
    model = SimpleNamespace(architecture=arm)
    provider = SimpleNamespace(model=model)
    captured = {}

    monkeypatch.setattr(compat, "load_trusted_checkpoint", lambda *_args, **_kwargs: latest_payload)
    monkeypatch.setattr(training, "_ensure_workspace_storage_bindings", lambda: {})
    monkeypatch.setattr(training.research, "verify_storage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(training.research, "_permitted_order", lambda *_args, **_kwargs: ["cpu"])
    monkeypatch.setattr(training.research, "resolve_device", lambda *_args, **_kwargs: (torch.device("cpu"), {}))
    monkeypatch.setattr(training, "_build_task", lambda *_args, **_kwargs: (model, provider))
    monkeypatch.setattr(training, "_storage_estimate", lambda *_args, **_kwargs: (0, 0))
    monkeypatch.setattr(training, "_prepare_storage", lambda *_args, **_kwargs: {"fixture": True})
    monkeypatch.setattr(training, "_write_run_manifest", lambda *_args, **_kwargs: None)

    def fake_fit(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "resumed_fixture"}

    monkeypatch.setattr(training, "_fit", fake_fit)
    result = training.resume_contextual_frontier(
        run, device="cpu", devices=["cpu"], stop_after_epoch=500,
        extend_horizon=True, verify_bindings=False,
    )

    assert result == {"status": "resumed_fixture"}
    assert captured["resume_checkpoint"] == run / "checkpoints" / "latest_model.pt"
    assert captured["stop_after"] == 500
    assert captured["engine_horizon"] == 500
    assert captured["engine_checkpoint_epochs"] == (100, 200, 300, 400, 500)
    assert captured["allow_horizon_extension"] is True
    assert config["training"]["local_horizon"] == 100
    assert config["training"]["optimizer_schedule_horizon"] == 500


def test_resume_member_routing_cli_passes_extend_horizon_flag(monkeypatch):
    import honf_runtime.contextual_frontier_training as training

    script = ROOT / "tools" / "honf_research.py"
    spec = importlib.util.spec_from_file_location("r13_6_resume_research_cli", script)
    cli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(training, "resume_contextual_frontier",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "done"})
    assert cli.main(["resume-member-routing", "--run", "lineage", "--device", "cuda:0",
                     "--stop-after-epoch", "500", "--extend-horizon"]) == 0
    assert calls == [(("lineage",), {
        "device": "cuda:0", "devices": None, "stop_after_epoch": 500,
        "extend_horizon": True, "wall_clock_budget_seconds": None,
    })]


def test_r13_6_evaluator_cli_dispatches_maintained_all_val89_evaluator(monkeypatch):
    import honf_runtime.r13_6_member_routing_evaluation as evaluation

    script = ROOT / "tools" / "honf_research.py"
    spec = importlib.util.spec_from_file_location("r13_6_research_cli", script)
    cli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(evaluation, "evaluate", lambda *args: calls.append(args) or {"status": "done"})
    assert cli.main(["evaluate-r13-6-member-routing", "--checkpoint", "epoch_0100.pt",
                     "--device", "cuda:0", "--output", "eval"]) == 0
    assert calls == [("epoch_0100.pt", "cuda:0", "eval")]
