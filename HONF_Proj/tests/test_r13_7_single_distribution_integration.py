from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.member_routing import (
    ThermalSingleDistributionMemberRoutingAdapter,
    ThermalTypedOrganizationMemberRoutingAdapter,
)
from channelthermal.joint_member_routing import ThermalJointMemberRoutingAdapter
from channelthermal.training.joint_member_routing_task import ThermalJointPairPredictions
from channelthermal.training.r13_7_single_distribution_task import (
    ARM,
    ThermalSingleDistributionMemberRoutingTask,
)
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.interaction_reused_member_routing.joint import JointMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.single_distribution_core import (
    SingleDistributionMemberRoutingCore,
)
from honf_forward_core.interaction_reused_member_routing.typed_organization_core import (
    TypedOrganizationMemberRoutingCore,
)
from honf_runtime.contextual_frontier_training import (
    R13_7_SINGLE_DISTRIBUTION_WORKFLOW,
    _member_routing_task_factory,
    _run_identity,
    _validate_loaded_contextual_config,
    load_contextual_frontier_config,
)


ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs" / "research"
CONFIG_NAMES = (
    "thermal_r13_7u_single_distribution_train150_6_v1.json",
    "thermal_r13_7u_single_distribution_train600_formal_v1.json",
)


def _base_config() -> dict:
    return ContentHypergraphCoreV2(
        module_input_dim=9, environment_input_dim=6, coordinate_dim=2,
        query_input_dim=0, operating_input_dim=12, output_dim=4,
        architecture="A2-D", context_dim=16, num_heads=4, context_depth=2,
        local_hidden_dim=16, dropout=0.0, init_seed=731,
    ).get_config()


def _stats():
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


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_r13_7_u_configs_route_strictly_and_keep_the_declared_horizons(name):
    config, _path = load_contextual_frontier_config(CONFIGS / name)
    _validate_loaded_contextual_config(config)
    section = config["member_routing"]
    assert section["schema"] == "thermal_r13_7_member_routing_v1"
    assert section["arm"] == ARM
    assert section["model"]["schema"] == SingleDistributionMemberRoutingCore.schema
    assert section["model"]["retained_prior_law"] == "none"
    assert _member_routing_task_factory(config).__name__ == "build_thermal_single_distribution_member_routing_task"
    identity = _run_identity(config, SimpleNamespace(identity_payload=lambda: {}))
    assert identity["workflow"] == R13_7_SINGLE_DISTRIBUTION_WORKFLOW
    assert identity["training_membership"] == config["training"]["membership"]
    if config["training"]["membership"] == "fixed25_train150_dev22":
        assert (config["training"]["local_horizon"], config["training"]["first_segment_epochs"],
                config["training"]["optimizer_schedule_horizon"],
                config["training"]["checkpoint_epochs"]) == (6, 3, 6, [3, 6])
    else:
        assert (config["training"]["local_horizon"], config["training"]["first_segment_epochs"],
                config["training"]["optimizer_schedule_horizon"],
                config["training"]["checkpoint_epochs"]) == (100, 100, 500, [100])


def test_u_adapter_has_no_new_learned_tensors_and_loads_strictly():
    stats = _stats()
    torch.manual_seed(7317)
    baseline = ThermalJointMemberRoutingAdapter(stats, core_config=_core_config(JointMemberRoutingCore))
    torch.manual_seed(7317)
    h_model = ThermalTypedOrganizationMemberRoutingAdapter(
        stats, core_config=_core_config(TypedOrganizationMemberRoutingCore),
    )
    torch.manual_seed(7317)
    u_model = ThermalSingleDistributionMemberRoutingAdapter(
        stats, core_config=_core_config(SingleDistributionMemberRoutingCore),
    )
    assert tuple(baseline.state_dict()) == tuple(h_model.state_dict()) == tuple(u_model.state_dict())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, u_model.state_dict()[name]), name
    restored = ThermalSingleDistributionMemberRoutingAdapter.from_state_payload(
        u_model.state_payload(), state_dict=u_model.state_dict(),
    )
    assert restored.core.get_config() == u_model.core.get_config()
    with pytest.raises(ValueError, match="Not a Thermal"):
        ThermalTypedOrganizationMemberRoutingAdapter.from_state_payload(u_model.state_payload())


def test_u_task_dispatches_the_maintained_paired_native_core_call():
    calls = []

    class FakeCore:
        def decode_paired(self, organization, query_batch):
            calls.append((organization, query_batch))
            return (
                SimpleNamespace(joint_fields=torch.ones(1, 2, 4), work_counts={"pairs": 7}),
                SimpleNamespace(joint_fields=torch.zeros(1, 2, 4), work_counts={"pairs": 3}),
                {"query_rows": 2},
            )

    class FakeModel:
        CORE_SCHEMA = SingleDistributionMemberRoutingCore.schema
        architecture = ARM
        core = FakeCore()

        def validate_prepared(self, prepared):
            assert prepared.organization is organization
            self.validated = True

        def _extract_native_roles(self, prepared, temperature, flow):
            del prepared
            return torch.cat((flow, temperature), dim=-1)

    organization = SimpleNamespace(work_counts={"prepared_rows": 4})
    prepared = SimpleNamespace(
        organization=organization,
        scene_batch=SimpleNamespace(module_features=torch.zeros(1, 1, 1)),
        structure={"heat_powers": torch.zeros(1, 2)},
        fluid_xy=torch.zeros(1, 2, 2),
        union=SimpleNamespace(unique_counts=torch.tensor([5])),
        query_batch=SimpleNamespace(local_weights=torch.ones(1, 2, 3)),
    )
    task = object.__new__(ThermalSingleDistributionMemberRoutingTask)
    task.model = FakeModel()
    paired = task._paired_predictions(prepared, None)
    assert isinstance(paired, ThermalJointPairPredictions)
    assert calls == [(organization, prepared.query_batch)]
    assert task.model.validated is True
    assert paired.work["core_decode_joint_shared_query_rows"] == 2
    assert paired.work["core_prepare_primary_prepared_rows"] == 4


def test_maintained_checkpoint_loader_reconstructs_u_adapter_strictly(tmp_path, monkeypatch):
    import honf_runtime.contextual_frontier_evaluation as evaluation
    import honf_runtime.contextual_frontier_training as training

    model = ThermalSingleDistributionMemberRoutingAdapter(
        _stats(), core_config=_core_config(SingleDistributionMemberRoutingCore),
    )
    provider_identity = {
        "schema": "thermal_r13_7_single_distribution_member_routing_provider_v1",
        "architecture": ARM,
        "member_routing_schema": SingleDistributionMemberRoutingCore.schema,
        "adapter_state": model.state_payload(),
        "recipe": {"fixture": "strict-r13-7-u"},
    }
    payload = {
        "experiment_identity": {
            "workflow": R13_7_SINGLE_DISTRIBUTION_WORKFLOW,
            "provider_identity": provider_identity,
        },
        "arm": ARM,
        "epoch": 100,
        "model_state_dict": model.state_dict(),
    }
    provider = SimpleNamespace(model=model, adapter=model, device=torch.device("cpu"))
    monkeypatch.setattr(evaluation, "load_trusted_checkpoint", lambda *args, **kwargs: payload)
    monkeypatch.setattr(training, "_build_task", lambda *args, **kwargs: (model, provider))
    checkpoint = tmp_path / "epoch_0100_model.pt"
    checkpoint.touch()
    restored, restored_provider, restored_payload = evaluation.load_contextual_checkpoint(checkpoint, "cpu")
    assert isinstance(restored, ThermalSingleDistributionMemberRoutingAdapter)
    assert restored.core.schema == SingleDistributionMemberRoutingCore.schema
    assert restored_provider.model is restored and restored_provider.adapter is restored
    assert restored_payload["checkpoint_path"] == str(checkpoint.resolve())
    assert not restored.training

    saved_core = provider_identity["adapter_state"]["core"]
    saved_core["address_temperature"] = 0.5
    with pytest.raises(ValueError, match="temperature"):
        evaluation.load_contextual_checkpoint(checkpoint, "cpu")
    saved_core["address_temperature"] = 1.0

    incomplete = dict(payload["model_state_dict"])
    incomplete.pop(next(iter(incomplete)))
    payload["model_state_dict"] = incomplete
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluation.load_contextual_checkpoint(checkpoint, "cpu")


def test_u_cli_aliases_dispatch_to_the_maintained_runner_and_evaluator(monkeypatch, capsys):
    import honf_runtime.contextual_frontier_training as training
    import honf_runtime.r13_6_member_routing_evaluation as evaluation

    script = ROOT / "tools" / "honf_research.py"
    spec = importlib.util.spec_from_file_location("r13_7_u_research_cli", script)
    cli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cli)
    config, _path = load_contextual_frontier_config(CONFIGS / CONFIG_NAMES[0])
    model = SimpleNamespace(architecture=ARM)
    provider = SimpleNamespace(
        training_case_ids=tuple(f"t{i:03d}" for i in range(150)),
        validation_case_ids=tuple(f"v{i:02d}" for i in range(22)),
        validation_scope="fixed25_VAL22_exposed",
        identity_payload=lambda: {"schema": "u-provider"},
    )
    monkeypatch.setattr(training, "load_contextual_frontier_config", lambda _path: (config, Path("u.json")))
    monkeypatch.setattr(training, "_build_task", lambda *_args, **_kwargs: (model, provider))
    assert cli.main(["train-single-distribution-member-routing", "--config", "u.json",
                     "--device", "cpu", "--output", "unused", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["event"] == "member_routing_dry_run_ready"

    resume_calls = []
    monkeypatch.setattr(
        training, "resume_contextual_frontier",
        lambda *args, **kwargs: resume_calls.append((args, kwargs)) or {"status": "done"},
    )
    assert cli.main(["resume-single-distribution-member-routing", "--run", "run-u",
                     "--device", "cpu", "--stop-after-epoch", "101", "--extend-horizon"]) == 0
    assert len(resume_calls) == 1
    assert resume_calls[0][0] == ("run-u",)
    for name, value in {
        "device": "cpu", "devices": None, "stop_after_epoch": 101,
        "extend_horizon": True,
    }.items():
        assert resume_calls[0][1][name] == value

    calls = []
    monkeypatch.setattr(evaluation, "evaluate", lambda *args: calls.append(args) or {"status": "done"})
    assert cli.main(["evaluate-r13-7-single-distribution-member-routing", "--checkpoint", "e100.pt",
                     "--device", "cpu", "--output", "eval-u"]) == 0
    assert calls == [("e100.pt", "cpu", "eval-u")]
