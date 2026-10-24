from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from honf_runtime import research_training as research
from honf_runtime.unified_training import EngineConfig, SamplingKey, TaskBatch, TrainingEngine


PROJECT = Path(__file__).resolve().parents[1]
PRESETS = PROJECT / "configs" / "research"


def test_research_presets_keep_membership_and_execution_contracts_separate():
    expected = {
        "thermal_honf_full.json": ("original_train600_fixed25_dev22", 24, 1024, "values"),
        "thermal_dense_full.json": ("original_train600_fixed25_dev22", 24, 1024, "values"),
        "thermal_direct_full.json": ("original_train600_fixed25_dev22", 24, 1024, "values"),
        "thermal_honf_dev150.json": ("fixed25_train150_dev22", 8, 512, "tile"),
        "thermal_dense_dev150.json": ("fixed25_train150_dev22", 8, 512, "tile"),
        "thermal_direct_dev150.json": ("fixed25_train150_dev22", 8, 512, "tile"),
    }
    for name, (membership, microbatch, tile, checkpoint) in expected.items():
        config, _ = research.load_research_config(PRESETS / name)
        assert config["training"]["membership"] == membership
        assert config["execution"]["options"] == {
            "backend": "dense_weighted",
            "query_tile_size": tile,
            "checkpoint": checkpoint,
            "context_backend": "segment",
        }
        assert config["storage"]["output_root"].startswith("/data2/")
        assert config["training"]["effective_batch_size"] == 48
        assert config["training"]["primary_queries"] == 1024
        assert config["training"]["microbatch_cases"] == microbatch

    fresh, _ = research.load_research_config(PRESETS / "thermal_honf_full_fresh_train600_normalization.json")
    assert fresh["training"]["membership"] == "original_train600_fixed25_dev22"
    assert fresh["initialization"]["normalization_mode"] == "fresh_full_train600_stats"
    assert fresh["initialization"]["fresh_model_schedule"]["ramp_epochs"] == 40
    assert fresh["training"]["microbatch_cases"] == 24
    assert fresh["execution"]["options"] == {
        "backend": "dense_weighted",
        "query_tile_size": 1024,
        "checkpoint": "values",
        "context_backend": "segment",
    }
    assert fresh["initialization"]["fresh_controller_lambda0"] == {
        "M": 1.007182002067566,
        "E": 1.59217369556427,
        "context": 1.3138799667358398,
    }
    assert "U10-H" in fresh["initialization"]["fresh_controller_lambda0_source"]
    assert fresh["optimizer"]["schedule"] == "constant_source_terminal_lr"


def test_round12_presets_keep_literal_uh_lineage_cut_and_device_scope():
    for name, role, arm in (
        ("thermal_compact_round12_common_v1.json", "common", "common"),
        ("thermal_compact_round12_r0_v1.json", "R0", "singleton"),
        ("thermal_compact_round12_r1_v1.json", "R1", "merge"),
    ):
        config, _ = research.load_research_config(PRESETS / name)
        assert config["initialization"]["source_arm"] == "U-H"
        assert config["initialization"]["source_parent_age"] == 1000
        assert config["initialization"]["source_checkpoint"].endswith(
            "thermal_u_h_full5000_v1/checkpoints/epoch_1000_model.pt"
        )
        assert config["initialization"]["source_sha256"] == (
            "2efd008f4135c7e5c7d909e0174b31e6ad5b05a058c0e68914ffc727d29134d7"
        )
        assert "expected_source_checkpoint_sha256" not in config["round12"]
        assert config["execution"]["preferred_devices"] == ["cuda:2"]
        assert config["round12"]["role"] == role
        assert config["round12"]["compact_arm"] == arm
        assert config["training"]["checkpoint_epochs"] == [20, 40, 100]
        assert config["training"]["local_horizon"] == 100


def test_round12_validator_accepts_only_declared_forward_horizon_extensions():
    config, _ = research.load_research_config(PRESETS / "thermal_compact_round12_r1_v1.json")
    extended = json.loads(json.dumps(config))
    extended["training"]["local_horizon"] = 200
    extended["training"]["checkpoint_epochs"] = [20, 40, 100, 200]
    research._validate_research_config(extended)
    invalid = json.loads(json.dumps(extended))
    invalid["training"]["checkpoint_epochs"] = [20, 40, 60, 100, 200]
    with pytest.raises(ValueError, match="checkpoints at 20/40/100"):
        research._validate_research_config(invalid)


def test_round12_frozen_and_new_parameter_names_map_through_adapter_prefix():
    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = nn.Parameter(torch.ones(()))
            self.new = nn.Parameter(torch.ones(()))
            self.compact_frozen_parameter_names = ("compact_retrieval.compatibility.policy",)
            self.compact_parameter_migration = {"new_names": ("compact_retrieval.new",)}

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()

    model = Adapter()
    model.core.compact_frozen_parameter_names = ("core.policy",)
    model.core.compact_parameter_migration = {"new_names": ("core.new",)}
    assert research._round12_model_parameter_names(model) == {"core.policy"}
    assert research._round12_new_parameter_names(model) == ("core.new",)


def _round12_compact_adapter():
    from tests.test_thermal_compact_adapter import _compact_adapter

    return _compact_adapter(12)[0]


def _round12_migration_identity(model, config, sampling_dataset_id):
    return {
        "family": research.TASK_SCHEMA,
        "config_name": config["name"],
        "sampling_dataset_id": sampling_dataset_id,
        "provider_identity": {
            "adapter_state": model.state_payload(),
            "recipe": {
                "compact_adaptation": {
                    "parameter_migration": model.core.compact_parameter_migration,
                },
            },
        },
    }


def _round12_optimizer_spec(model, horizon):
    from honf_runtime.unified_training import OptimizerGroupSpec, ScheduleSpec

    rate = 3.0e-4
    schedule = ScheduleSpec(
        peak_lr=rate, warmup_start_lr=rate, warmup_epochs=0,
        hold_through_epoch=horizon, total_epochs=horizon + 1, final_lr=rate,
    )
    return OptimizerGroupSpec(
        "formal_joint", tuple(name for name, _ in model.named_parameters()), schedule,
        weight_decay=1.0e-5,
    )


def _round12_provider_for_test(model, config, sampling_dataset_id):
    class Provider:
        def __init__(self):
            self.model = model
            self.sampling_dataset_id = sampling_dataset_id
            self._round12_local_epoch = 0
            self._research_config = config

        def load_training_state_dict(self, state):
            self.model.core.load_compact_state(state["compact_state"])

        def training_state_dict(self):
            return {
                "schema": "thermal_compact_round12_provider_state_v1",
                "base": {},
                "compact_state": self.model.core.export_compact_state(),
                "clocks": {
                    **config["initialization"]["clock_mapping"],
                    "local_epoch": int(self.model.core.compact_local_epoch),
                    "controller_epoch": int(self.model.core.joint_epoch),
                    "macro_update_index": int(self.model.core.joint_update_index),
                    "joint_macro_clock": int(self.model.core.joint_macro_clock),
                    "optimizer_updates_since_parent": int(self.model.core.joint_macro_clock) - 13000,
                },
            }

    return Provider()


def _round12_test_configs(tmp_path):
    import copy
    import hashlib

    parent = tmp_path / "source" / "epoch_1000_model.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"temporary U-H source binding for runner tests")
    sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    result = {}
    for role in ("common", "r0", "r1"):
        filename = f"thermal_compact_round12_{role}_v1.json"
        config, _ = research.load_research_config(PRESETS / filename)
        config = copy.deepcopy(config)
        config["initialization"]["source_checkpoint"] = str(parent)
        config["initialization"]["source_run_dir"] = str(parent.parent)
        config["initialization"]["source_alias"] = str(parent)
        config["initialization"]["source_sha256"] = sha
        path = tmp_path / filename
        path.write_text(json.dumps(config), encoding="utf-8")
        result[role] = (research.load_research_config(path)[0], path)
    return parent, sha, result


def test_round12_restores_migration_metadata_from_ordinary_checkpoint_identity():
    model = _round12_compact_adapter()
    model.core.set_compact_phase(20, arm="common")
    identity = _round12_migration_identity(model, {"name": "round12-test"}, "sample-id")
    payload = {
        "experiment_identity": identity,
        "model_state_dict": model.state_dict(),
    }
    restored = research._model_from_run_payload(payload, torch.device("cpu"))
    assert restored.core.compact_parameter_migration == model.core.compact_parameter_migration
    assert set(research._round12_new_parameter_names(restored)) == {
        f"core.{name}" for name in model.core.compact_parameter_migration["new_names"]
    }


def test_round12_calibration_matches_native_norm_on_each_term_participant_set():
    native = {"tie_only": torch.tensor([3.0]), "capacity_only": torch.tensor([4.0])}
    tie = {"tie_only": torch.tensor([2.0])}
    capacity = {"capacity_only": torch.tensor([10.0])}
    tie_native, tie_term, tie_coefficient = research._round12_term_gradient_calibration(
        native, tie, 0.02,
    )
    cap_native, cap_term, cap_coefficient = research._round12_term_gradient_calibration(
        native, capacity, 0.02,
    )
    assert (tie_native, tie_term, tie_coefficient) == pytest.approx((3.0, 2.0, 0.03))
    assert (cap_native, cap_term, cap_coefficient) == pytest.approx((4.0, 10.0, 0.008))
    assert research._round12_term_gradient_calibration(native, {"missing": torch.tensor([1.0])}, 0.02) == (
        0.0, 1.0, 0.0,
    )


def test_round12_native_calibration_combines_task_and_existing_work_vjps():
    parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    combined = research._round12_merge_native_work_gradients(
        {"compact": torch.tensor([1.0, 2.0])}, ("compact",), (parameter,),
    )
    torch.testing.assert_close(combined["compact"], torch.tensor([4.0, 6.0]))
    parameter.grad = None
    combined = research._round12_merge_native_work_gradients(
        {"compact": torch.tensor([1.0, 2.0])}, ("compact",), (parameter,),
    )
    torch.testing.assert_close(combined["compact"], torch.tensor([1.0, 2.0]))


def test_round12_resume41_activates_cut_before_advancing_compact_phase(monkeypatch):
    model = _round12_compact_adapter()
    model.core.set_compact_phase(40, arm="merge")
    provider = SimpleNamespace(_round12_local_epoch=40)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    engine = object.__new__(research.ResearchTrainingEngine)
    engine.controller_parent_coordinate = 4000
    engine.fresh_model_schedule = None
    engine.compact_arm = "merge"
    engine.compact_cut_epoch = 40
    engine._execution_runtime_dirty = False
    engine.runtime_microbatch_cases = 24

    def parent_update(_self, _model, _provider, _optimizer, _microbatches, *, epoch, **_kwargs):
        assert epoch == 4041
        assert bool(model.core.compact_cut_active)
        assert int(model.core.compact_local_epoch) == 41
        return {}, {}, {}, None

    monkeypatch.setattr(TrainingEngine, "_run_update", parent_update)
    batch = SimpleNamespace(case_keys=("TRAIN0006",))
    engine._run_update(
        model, provider, optimizer, (batch,), phase="joint", arm="A2-D",
        epoch=41, update_index=0,
    )
    assert provider._round12_local_epoch == 41
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name in research._round12_model_parameter_names(model))


def test_round12_soft_merge_keeps_both_structural_terms_in_history_input(monkeypatch):
    model = _round12_compact_adapter()
    model.core.set_compact_phase(20, arm="merge")
    provider = SimpleNamespace(_round12_local_epoch=20)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    engine = object.__new__(research.ResearchTrainingEngine)
    engine.controller_parent_coordinate = 4000
    engine.fresh_model_schedule = None
    engine.compact_arm = "merge"
    engine.compact_cut_epoch = 40
    engine._execution_runtime_dirty = False
    engine.runtime_microbatch_cases = 24
    losses = {"compact_tie": 0.125, "compact_capacity": 0.25}

    def parent_update(_self, *_args, **_kwargs):
        return losses, {}, {}, None

    monkeypatch.setattr(TrainingEngine, "_run_update", parent_update)
    batch = SimpleNamespace(case_keys=("TRAIN0006",))
    result = engine._run_update(
        model, provider, optimizer, (batch,), phase="joint", arm="A2-D",
        epoch=21, update_index=0,
    )
    assert result[0] == losses

    def missing_terms(_self, *_args, **_kwargs):
        return {}, {}, {}, None

    monkeypatch.setattr(TrainingEngine, "_run_update", missing_terms)
    model.core.set_compact_phase(21, arm="merge")
    with pytest.raises(RuntimeError, match="additive losses were not returned"):
        engine._run_update(
            model, provider, optimizer, (batch,), phase="joint", arm="A2-D",
            epoch=21, update_index=1,
        )


def test_round12_common_fork_uses_adapted_a2_d_and_preserves_real_compact_state(tmp_path, monkeypatch):
    import copy
    from honf_runtime.unified_training import capture_rng_state
    from honf_runtime.unified_training import _optimizer_spec_payload

    source_path, source_sha, configs = _round12_test_configs(tmp_path)
    common_config, _ = configs["common"]
    model = _round12_compact_adapter()
    model.core.set_compact_phase(20, arm="common")
    sampling_id = common_config["round12"]["expected_parent_sampling_dataset_id"]
    identity = _round12_migration_identity(model, common_config, sampling_id)
    spec = _round12_optimizer_spec(model, 100)
    optimizer_contract = [_optimizer_spec_payload(spec)]
    checkpoint_root = tmp_path / "common20"
    checkpoint_root.mkdir()
    latest_path = checkpoint_root / "latest_model.pt"
    best_path = checkpoint_root / "best_by_field_mse_model.pt"
    latest_path.write_bytes(b"latest placeholder")
    best_path.write_bytes(b"best placeholder")
    compact_state = model.core.export_compact_state()
    base_payload = {
        "experiment_identity": identity,
        "epoch": 20,
        "arm": "A2-D",
        "model_state_dict": research._cpu_state_snapshot(model.state_dict()),
        "provider_training_state": {
            "schema": "thermal_compact_round12_provider_state_v1",
            "base": {},
            "compact_state": compact_state,
            "clocks": {**common_config["initialization"]["clock_mapping"],
                       "local_epoch": 20, "controller_epoch": 4020,
                       "macro_update_index": 12, "joint_macro_clock": 13260,
                       "optimizer_updates_since_parent": 260},
        },
        "history": [{"epoch": epoch, "case_visits": 600, "optimizer_updates": 13}
                    for epoch in range(1, 21)],
        "optimizer_group_names": ["formal_joint"],
        "optimizer_schedule_contract": optimizer_contract,
        "optimizer_state_by_name": {
            "groups": [{"name": "formal_joint", "parameter_names": list(spec.parameter_names)}],
            "state": {"compact": {"exp_avg": torch.tensor([1.25]), "step": torch.tensor(260)}},
        },
        "rng_state": capture_rng_state(),
        "sampler_state": {"epoch": 20, "cursor": 600},
    }
    best_payload = copy.deepcopy(base_payload)
    best_payload["epoch"] = 10
    common_best_model = _round12_compact_adapter()
    common_best_model.core.set_compact_phase(10, arm="common")
    best_payload["model_state_dict"] = research._cpu_state_snapshot(common_best_model.state_dict())
    best_payload["provider_training_state"]["compact_state"] = common_best_model.core.export_compact_state()
    best_payload["provider_training_state"]["clocks"]["local_epoch"] = 10
    common_manifest = {
        "config": common_config,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "optimizer_schedule_contract": optimizer_contract,
        "run_case_ids": None,
        "diagnostic_only": False,
    }
    monkeypatch.setattr(research, "_load_run_manifest", lambda *_: (common_manifest, None))
    monkeypatch.setattr(research, "_read_run_checkpoint", lambda _run, selector: (
        (best_payload, best_path) if selector == "best_by_field_mse"
        else (base_payload, latest_path)
    ))
    monkeypatch.setattr(research, "_round12_calibrate_structural_losses", lambda *_: {
        "record": {"schema": "thermal_compact_structural_loss_calibration_v1",
                   "terms": {"tie": {}, "capacity": {}}},
        "coefficients": {"tie": 0.012, "capacity": 0.018},
    })
    monkeypatch.setattr(research, "verify_storage", lambda *_args, **_kwargs: {"test": True})
    monkeypatch.setattr(research, "_estimate_run_bytes", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(research, "resolve_device", lambda _device: (torch.device("cpu"), {"device": "cpu"}))

    def build_provider(config, checkpoint_payload, *_args, **_kwargs):
        built = research._model_from_run_payload(checkpoint_payload, torch.device("cpu"))
        return built, _round12_provider_for_test(built, config, sampling_id)

    monkeypatch.setattr(research, "_build_provider_for_resume", build_provider)

    def sealed_identity(config, child_model, _provider, _engine, _source, _sha):
        assert child_model.core.compact_arm == config["round12"]["compact_arm"]
        child_identity = _round12_migration_identity(child_model, config, sampling_id)
        child_identity["round12_role"] = config["round12"]["role"]
        child_identity["compact_arm"] = config["round12"]["compact_arm"]
        return child_identity, (spec,)

    monkeypatch.setattr(research, "_round12_sealed_identity", sealed_identity)
    result = research.fork_round12_common(
        checkpoint_root,
        r0_config_path=configs["r0"][1], r0_output=tmp_path / "R0",
        r1_config_path=configs["r1"][1], r1_output=tmp_path / "R1",
        device="cpu",
    )
    assert result["status"] == "round12_children_forked_from_common20"
    assert [child["optimizer_updates_preserved"] for child in result["children"]] == [260, 260]
    monkeypatch.undo()
    for role, arm in (("R0", "singleton"), ("R1", "merge")):
        child_payload, _ = research._read_run_checkpoint(tmp_path / role, "latest")
        assert child_payload["epoch"] == 20
        assert child_payload["provider_training_state"]["compact_state"]["arm"] == arm
        assert child_payload["provider_training_state"]["compact_state"]["cut_active"] is False
        assert child_payload["optimizer_state_by_name"]["groups"] == base_payload["optimizer_state_by_name"]["groups"]
        torch.testing.assert_close(
            child_payload["optimizer_state_by_name"]["state"]["compact"]["exp_avg"],
            base_payload["optimizer_state_by_name"]["state"]["compact"]["exp_avg"],
        )
        assert child_payload["sampler_state"] == base_payload["sampler_state"]
        assert child_payload["rng_state"]["torch_cpu"].equal(base_payload["rng_state"]["torch_cpu"])
        assert not any("sha256" in key for key in child_payload["branch_parent"])
        best_child, _ = research._read_run_checkpoint(tmp_path / role, "best_by_field_mse")
        assert best_child["experiment_identity"]["round12_role"] == role
        loaded_prefix = research._model_from_run_payload(best_child, torch.device("cpu"))
        assert loaded_prefix.core.compact_arm == "common"
        assert int(loaded_prefix.core.compact_local_epoch) == 10


def test_round12_cut_writes_distinct_before_and_after_states_and_restores_frozen_policy(tmp_path, monkeypatch):
    from honf_runtime.unified_training import capture_rng_state

    config, _ = research.load_research_config(PRESETS / "thermal_compact_round12_r1_v1.json")
    model = _round12_compact_adapter()
    model.core.set_compact_phase(40, arm="merge")
    identity = _round12_migration_identity(
        model, config, config["round12"]["expected_parent_sampling_dataset_id"],
    )
    run = tmp_path / "R1"
    run.mkdir()
    (run / "artifact_layout.json").write_text(json.dumps({"artifact_layout_version": 1}), encoding="utf-8")
    layout = research.RunLayout(run)
    layout.ensure()
    research._atomic_json(layout.write_path("experiment_identity.json"), identity)
    before_payload = {
        "experiment_identity": identity,
        "epoch": 40,
        "arm": "A2-D",
        "model_state_dict": research._cpu_state_snapshot(model.state_dict()),
        "provider_training_state": {
            "schema": "thermal_compact_round12_provider_state_v1",
            "base": {}, "compact_state": model.core.export_compact_state(),
            "clocks": {**config["initialization"]["clock_mapping"],
                       "local_epoch": 40, "controller_epoch": 4040,
                       "macro_update_index": 12, "joint_macro_clock": 13520,
                       "optimizer_updates_since_parent": 520},
        },
        "optimizer_state_by_name": {"groups": [], "state": {"moment": torch.tensor([8.0])}},
        "rng_state": capture_rng_state(),
        "sampler_state": {"epoch": 1040, "cursor": 600},
    }
    torch.save(before_payload, layout.write_path("epoch_0040_model.pt"))
    torch.save(before_payload, layout.write_path("latest_model.pt"))
    monkeypatch.setattr(research, "_load_run_manifest", lambda *_: ({"config": config}, None))
    monkeypatch.setattr(research, "verify_storage", lambda *_args, **_kwargs: {"test": True})

    result = research.activate_round12_cut(run)
    assert result["status"] == "round12_hard_cut_activated"
    before, _ = research._read_run_checkpoint(run, "epoch_0040_model.pt")
    after, _ = research._read_run_checkpoint(run, "epoch_0040_after_cut_model.pt")
    latest, _ = research._read_run_checkpoint(run, "latest")
    assert before["provider_training_state"]["compact_state"]["cut_active"] is False
    assert after["provider_training_state"]["compact_state"]["cut_active"] is True
    assert after["provider_training_state"]["compact_state"]["last_topology"] is None
    assert latest["provider_training_state"]["compact_state"] == after["provider_training_state"]["compact_state"]
    assert after["optimizer_state_by_name"] == before["optimizer_state_by_name"]
    assert after["sampler_state"] == before["sampler_state"]
    assert after["rng_state"]["torch_cpu"].equal(before["rng_state"]["torch_cpu"])
    restored = research._model_from_run_payload(after, torch.device("cpu"))
    frozen = research._round12_model_parameter_names(restored)
    assert frozen
    assert all(not parameter.requires_grad for name, parameter in restored.named_parameters() if name in frozen)


def test_round12_terminal_fork_keeps_cut_optimizer_rng_sampler_and_native_clocks(tmp_path, monkeypatch):
    import copy
    import hashlib
    from honf_runtime.unified_training import capture_rng_state
    from honf_runtime.unified_training import _optimizer_spec_payload

    source_path, source_sha, configs = _round12_test_configs(tmp_path)
    config, _ = configs["r1"]
    model = _round12_compact_adapter()
    model.core.set_compact_phase(40, arm="merge")
    model.core.activate_hard_cut()
    model.core.set_compact_phase(100, arm="merge")
    sampling_id = config["round12"]["expected_parent_sampling_dataset_id"]
    identity = _round12_migration_identity(model, config, sampling_id)
    spec100 = _round12_optimizer_spec(model, 100)
    contract100 = [_optimizer_spec_payload(spec100)]
    source_run = tmp_path / "R1-terminal"
    source_run.mkdir()
    latest_path = source_run / "latest_model.pt"
    best_path = source_run / "best_by_field_mse_model.pt"
    latest_path.write_bytes(b"terminal latest checkpoint placeholder")
    best_path.write_bytes(b"terminal best checkpoint placeholder")
    latest_payload = {
        "experiment_identity": identity,
        "epoch": 100,
        "arm": "A2-D",
        "model_state_dict": research._cpu_state_snapshot(model.state_dict()),
        "provider_training_state": {
            "schema": "thermal_compact_round12_provider_state_v1",
            "base": {}, "compact_state": model.core.export_compact_state(),
            "clocks": {**config["initialization"]["clock_mapping"],
                       "local_epoch": 100, "controller_epoch": 4100,
                       "macro_update_index": 12, "joint_macro_clock": 14300,
                       "optimizer_updates_since_parent": 1300},
        },
        "history": [{"epoch": epoch} for epoch in range(1, 101)],
        "optimizer_group_names": ["formal_joint"],
        "optimizer_schedule_contract": contract100,
        "optimizer_state_by_name": {
            "groups": [{"name": "formal_joint", "parameter_names": list(spec100.parameter_names)}],
            "state": {"compact": {"exp_avg": torch.tensor([3.5]), "step": torch.tensor(14300)}},
        },
        "rng_state": capture_rng_state(),
        "sampler_state": {"epoch": 1100, "cursor": 600},
    }
    best_payload = copy.deepcopy(latest_payload)
    best_payload["epoch"] = 50
    calibration = {"schema": "thermal_compact_structural_loss_calibration_v1",
                   "terms": {"tie": {}, "capacity": {}}}
    (source_run / "round12_calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    manifest = {
        "config": config,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "optimizer_schedule_contract": contract100,
        "run_case_ids": None,
        "diagnostic_only": False,
    }
    monkeypatch.setattr(research, "_load_run_manifest", lambda *_: (manifest, None))
    monkeypatch.setattr(research, "_read_run_checkpoint", lambda _run, selector: (
        (best_payload, best_path) if selector == "best_by_field_mse"
        else (latest_payload, latest_path)
    ))
    monkeypatch.setattr(research, "verify_storage", lambda *_args, **_kwargs: {"test": True})
    monkeypatch.setattr(research, "_estimate_run_bytes", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(research, "resolve_device", lambda _device: (torch.device("cpu"), {"device": "cpu"}))

    def build_provider(child_config, checkpoint_payload, *_args, **_kwargs):
        built = research._model_from_run_payload(checkpoint_payload, torch.device("cpu"))
        return built, _round12_provider_for_test(built, child_config, sampling_id)

    monkeypatch.setattr(research, "_build_provider_for_resume", build_provider)

    def sealed_identity(child_config, child_model, _provider, _engine, _source, _sha):
        child_identity = _round12_migration_identity(child_model, child_config, sampling_id)
        child_identity["round12_role"] = child_config["round12"]["role"]
        child_identity["compact_arm"] = child_config["round12"]["compact_arm"]
        return child_identity, (_round12_optimizer_spec(child_model, child_config["training"]["local_horizon"]),)

    monkeypatch.setattr(research, "_round12_sealed_identity", sealed_identity)
    output = tmp_path / "R1-continue-200"
    result = research.fork_round12_terminal(
        source_run, output=output, local_horizon=200, device="cpu",
    )
    assert result["status"] == "round12_terminal_fork_created"
    assert result["resume_from_local_epoch"] == 100
    assert result["next_sampler_epoch"] == 1101
    assert result["next_controller_epoch"] == 4101
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_sha
    monkeypatch.undo()
    child_payload, _ = research._read_run_checkpoint(output, "latest")
    assert child_payload["epoch"] == 100
    assert child_payload["provider_training_state"]["compact_state"]["cut_active"] is True
    assert child_payload["optimizer_state_by_name"] == latest_payload["optimizer_state_by_name"]
    assert child_payload["sampler_state"] == latest_payload["sampler_state"]
    assert child_payload["rng_state"]["torch_cpu"].equal(latest_payload["rng_state"]["torch_cpu"])
    assert child_payload["optimizer_schedule_contract"] != contract100
    assert child_payload["branch_parent"]["native_clocks"] == latest_payload["provider_training_state"]["clocks"]
    assert not any("sha256" in key for key in child_payload["branch_parent"])


def test_round12_update_recovery_clears_structural_graph_cache_and_restores_adam_state():
    import copy
    from honf_runtime.unified_training import capture_rng_state

    config, _ = research.load_research_config(PRESETS / "thermal_compact_round12_r1_v1.json")
    model = _round12_compact_adapter()
    model.core.set_compact_phase(40, arm="merge")
    model.core.activate_hard_cut()
    trainable_name = next(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    provider = _round12_provider_for_test(
        model, config, config["round12"]["expected_parent_sampling_dataset_id"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    trainable = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    trainable.grad = torch.ones_like(trainable)
    optimizer.step()
    model_state = research._cpu_state_snapshot(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng_state = capture_rng_state()
    provider_state = provider.training_state_dict()
    model.core._compact_loss_entries = [torch.tensor(99.0, requires_grad=True)]
    with torch.no_grad():
        trainable.add_(5.0)
    optimizer.state.clear()
    torch.rand(5)

    research.ResearchTrainingEngine._restore_update_boundary(
        model, optimizer, provider, model_state, optimizer_state, rng_state,
        provider_state,
        provider.load_training_state_dict,
    )
    assert model.core._compact_loss_entries == []
    assert optimizer.state
    torch.testing.assert_close(trainable, model_state[trainable_name].to(trainable.device))
    assert model.core.compact_cut_active
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name in research._round12_model_parameter_names(model))
    assert torch.equal(capture_rng_state()["torch_cpu"], rng_state["torch_cpu"])


def test_classic_full5000_profiles_retain_formal_data_and_execution_contracts():
    configs = [research.load_research_config(PRESETS / f"thermal_u_{arm}_full5000_v1.json")[0]
               for arm in "dhs"]
    assert {config["training"]["membership"] for config in configs} == {
        "classic_train600_exposed_val89"
    }
    assert {config["training"]["local_horizon"] for config in configs} == {5000}
    assert {config["training"]["microbatch_cases"] for config in configs} == {24}
    assert {config["training"]["effective_batch_size"] for config in configs} == {48}
    assert {config["training"]["primary_queries"] for config in configs} == {1024}
    assert {config["training"]["seed"] for config in configs} == {0}
    assert {tuple(config["training"]["checkpoint_epochs"]) for config in configs} == {
        (10, 50, 100, 250, 500, 1000, 2500, 5000)
    }
    assert {config["initialization"]["fresh_routing_mode"] for config in configs} == {
        "dense", "hypergraph", "direct"
    }
    assert {config["initialization"]["normalization_mode"] for config in configs} == {
        "classic_train600_stats"
    }
    assert all(research._parent_age(config) == 0 for config in configs)
    assert {research._fresh_model_schedule_epoch(config, 1) for config in configs} == {3001}
    assert {research._fresh_model_schedule_epoch(config, 5000) for config in configs} == {8000}
    for config in configs:
        assert config["execution"]["options"] == {
            "backend": "dense_weighted",
            "query_tile_size": 1024,
            "checkpoint": "values",
            "context_backend": "segment",
        }
        assert config["optimizer"] == {
            "kind": "fresh_adamw_from_classic_formal_contract",
            "schedule": "constant_classic_formal_lr",
        }
        assert config["storage"]["output_root"].startswith(
            "/data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_u_"
        )


def test_classic_fresh_models_use_metadata_only_and_common_seed0_architecture(monkeypatch):
    from honf_runtime import compat

    def forbid_checkpoint_deserialization(*args, **kwargs):
        pytest.fail("Fresh classic initialization must not deserialize mature checkpoint weights.")

    monkeypatch.setattr(compat, "load_trusted_checkpoint", forbid_checkpoint_deserialization)
    models = {}
    expected_routes = {"d": "dense", "h": "hypergraph", "s": "direct"}
    for arm, route in expected_routes.items():
        config, _ = research.load_research_config(PRESETS / f"thermal_u_{arm}_full5000_v1.json")
        model, source_payload, _, _ = research._source_components(config, torch.device("cpu"))
        assert model.core.routing_mode == route
        assert model.core.get_config()["base_model_config"]["init_seed"] == 0
        assert model.state_payload()["normalizer_key"] == (
            "7ab3441ba2616c68224f9d829a92e2db4c59658b45d3f1acf4c535333896939b"
        )
        assert "model_state_dict" not in source_payload
        assert "optimizer_state_dict" not in source_payload
        assert source_payload["classic_sampling_dataset_id"].startswith("ThermalChannel:")
        models[arm] = model
    states = [{name: value.detach().cpu() for name, value in model.state_dict().items()
               if isinstance(value, torch.Tensor)}
              for model in models.values()]
    common = set(states[0]) & set(states[1]) & set(states[2])
    assert len(common) > 100
    assert all(torch.equal(states[0][name], states[index][name]) for index in (1, 2) for name in common)


def test_legacy_u_identity_omits_inactive_round12_fields_and_keeps_strict_resume(monkeypatch):
    from honf_runtime.unified_training import _resume_identity_amendment

    config, _ = research.load_research_config(PRESETS / "thermal_u_h_full5000_v1.json")
    monkeypatch.setattr(research.ThermalContentHypergraphTaskV2, "identity_payload", lambda self: {})
    provider = object.__new__(research.ResearchThermalProvider)
    provider.__dict__.update(
        training_case_ids=("0001", "0002"), validation_case_ids=("0003",),
        _classic_provider_identity={"subset_manifest_sha256": "a" * 64,
                                    "normalization_stats_sha256": "b" * 64},
        _research_case_limit=None, _research_membership=config["training"]["membership"],
        _research_normalization_mode=config["initialization"]["normalization_mode"],
        _research_config=config, _research_source_payload={}, _round12=None,
        protocol="formal_d_full_train_v1", validation_scope="canonical89_exposed",
        seed=0, total_epochs=5000, effective_batch_size=48, _identity_microbatch_size=24,
        primary_queries=1024,
    )
    payload = provider.identity_payload()
    assert "compact_adaptation" not in payload["recipe"]
    identity = research._experiment_identity(config, provider, "/classic/source.pt", "c" * 64,
                                             diagnostic=False)
    assert "initialization_recipe" not in identity
    assert "initialization_recipe_sha256" not in identity
    assert "compact_adaptation" not in identity["recipe"]
    assert _resume_identity_amendment(identity, json.loads(json.dumps(identity))) is None
    changed = json.loads(json.dumps(identity))
    changed["normalization_mode"] = "different_training_statistics"
    with pytest.raises(ValueError, match="identity.*differs"):
        _resume_identity_amendment(identity, changed)


@pytest.mark.parametrize("name", ["thermal_compact_round12_r1_v1.json",
                                 "thermal_compact_tiestop_full1000_scratch_v1.json"])
def test_round12_initialization_identity_layout_is_unchanged(name):
    config, _ = research.load_research_config(PRESETS / name)
    provider = SimpleNamespace(identity_payload=lambda: {"recipe": {"compact_adaptation": {"role": "R1"}}})
    identity = research._experiment_identity(config, provider, "/declared/source", "d" * 64,
                                             diagnostic=False)
    assert identity["recipe"]["compact_adaptation"] == {"role": "R1"}
    scratch = config["initialization"]["source_kind"] == "scratch_compact_config"
    assert identity["initialization_recipe"] == ("/declared/source" if scratch else None)
    assert identity["initialization_recipe_sha256"] == ("d" * 64 if scratch else None)


def test_sampling_membership_fingerprint_is_shared_across_h_d_s_weights():
    train_ids = ("0001", "0002", "0003")
    dev_ids = ("0277", "0279")
    configs = [research.load_research_config(PRESETS / f"thermal_{arm}_full.json")[0]
               for arm in ("honf", "dense", "direct")]
    assert {config["initialization"]["source_parent_age"] for config in configs} == {5000}
    assert {config["initialization"]["normalization_mode"] for config in configs} == {
        "inherit_source_train150"
    }
    dataset_identity = {
        "dataset_id": "thermal_channel_global_v1",
        "dataset_size_bytes": 1234,
        "dataset_mtime_ns": 5678,
        "catalog_metadata_sha256": "c" * 64,
        "base_manifest_sha256": "d" * 64,
    }
    fingerprints = {
        config["initialization"]["source_arm"]: research.research_membership_fingerprint(
            dataset_identity=dataset_identity,
            training_case_ids=train_ids,
            validation_case_ids=dev_ids,
        )
        for config in configs
    }
    assert len(set(fingerprints.values())) == 1
    assert next(iter(fingerprints.values())) != research.research_membership_fingerprint(
        dataset_identity=dataset_identity,
        training_case_ids=("0001", "0002"),
        validation_case_ids=dev_ids,
    )


def test_fresh_work_ramp_uses_local_sampler_clock_and_native_controller_clock():
    config, _ = research.load_research_config(PRESETS / "thermal_honf_full_fresh_train600_normalization.json")
    schedule = config["initialization"]["fresh_model_schedule"]
    assert research._fresh_model_schedule_epoch(config, 1) == 3025
    assert research._fresh_model_schedule_epoch(config, 40) == 4000
    engine = research.ResearchTrainingEngine(
        EngineConfig(seed=0, microbatch_cases=2, effective_cases=2, total_epochs=1000,
                     training_mode="joint"),
        device="cpu",
        parent_age=0,
        fresh_model_schedule=schedule,
    )
    assert engine.sampler_epoch(1) == 1
    assert engine.sampler_epoch(40) == 40
    contract = engine.sampler_epoch_contract()
    assert contract["parent_age"] == 0
    assert contract["fresh_model_schedule"] == schedule
    assert "coordinate_start_epoch" in contract["fit_epoch_to_model_clock"]


def test_fresh_controller_configuration_rejects_disabled_work_penalties():
    config, _ = research.load_research_config(
        PRESETS / "thermal_honf_full_fresh_train600_normalization.json"
    )
    config["initialization"]["fresh_controller_lambda0"]["E"] = 0.0
    with pytest.raises(ValueError, match="positive calibrated strengths"):
        research._validate_research_config(config)


class _TinyDual:
    def __init__(self, lambda0: float):
        self.lambda0 = torch.tensor(lambda0)
        self.dual_lambda = torch.tensor(lambda0)
        self.zeta = torch.tensor(lambda0)
        self.lambda_max = torch.tensor(10 * lambda0)
        self.eta_lambda = torch.tensor(lambda0)
        self.macro_step = torch.tensor(23)
        self.window_step_count = torch.tensor(4)
        self.dual_update_count = torch.tensor(7)
        self.window_residual_sum = torch.tensor(2.5)

    def set_calibrated_lambda0(self, value):
        self.lambda0.fill_(value)
        self.dual_lambda.fill_(value)
        self.zeta.fill_(value)
        self.lambda_max.fill_(10 * value)
        self.eta_lambda.fill_(value)


class _TinyBudget:
    def schedule(self):
        return {"absolute_epoch": 3000, "sparse_age": 0}

    def __init__(self):
        self.duals = {name: _TinyDual(9.0) for name in ("M", "E", "context")}


class _TinyCore(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([weight], dtype=torch.float32))
        self.joint_epoch = 3000
        self.joint_macro_clock = 0
        self.joint_budget = _TinyBudget()
        self.execution_options = None

    def get_config(self):
        return {"architecture": "tiny"}

    @classmethod
    def from_config(cls, config):
        assert config == {"architecture": "tiny"}
        return cls(weight=1.0)

    def set_execution_options(self, **options):
        self.execution_options = options

    def set_training_clock(self, epoch, update_index):
        self.joint_epoch = int(epoch)
        self.joint_macro_clock = int(update_index)


def test_fresh_core_uses_config_initialization_and_resets_native_state():
    source = _TinyCore(weight=99.0)
    fresh = research._fresh_core_from_template(
        source, seed=17,
        execution_options={"backend": "dense_weighted", "query_tile_size": 512,
                           "checkpoint": "tile", "context_backend": "segment"},
        controller_lambda0={"M": 1.0, "E": 1.5, "context": 1.25},
    )
    assert fresh is not source
    assert float(fresh.weight.item()) == 1.0
    assert float(source.weight.item()) == 99.0
    assert fresh.joint_epoch == 3000
    assert fresh.joint_macro_clock == 0
    assert fresh.execution_options["backend"] == "dense_weighted"
    for name, expected in {"M": 1.0, "E": 1.5, "context": 1.25}.items():
        dual = fresh.joint_budget.duals[name]
        assert float(dual.lambda0) == expected
        assert float(dual.dual_lambda) == expected
        assert float(dual.zeta) == expected
        assert float(dual.lambda_max) == 10 * expected
        assert float(dual.eta_lambda) == expected
        assert int(dual.macro_step) == 0
        assert int(dual.window_step_count) == 0
        assert int(dual.dual_update_count) == 0
        assert float(dual.window_residual_sum) == 0.0


def test_run_manifest_is_self_contained_after_initial_source_is_gone(tmp_path):
    config, _ = research.load_research_config(PRESETS / "thermal_honf_full.json")
    config["initialization"]["source_checkpoint"] = str(tmp_path / "removed-parent.pt")
    config_path = tmp_path / "removed-config.json"
    run = tmp_path / "run"
    run.mkdir()
    research._write_run_manifest(
        run,
        config=config,
        config_path=config_path,
        source_path=Path(config["initialization"]["source_checkpoint"]),
        source_sha="a" * 64,
        optimizer_contract=[{"name": "source_read_joint", "parameter_names": [], "schedule": {}}],
        case_ids=("0001", "0002"),
        storage_snapshot={},
        diagnostic=True,
        batch=2,
        microbatch=2,
        identity_microbatch=2,
    )
    loaded, saved_config_path = research._load_run_manifest(run)
    assert saved_config_path == config_path
    assert loaded["source_checkpoint"] == str(config_path.parent / "removed-parent.pt")
    assert loaded["config"]["initialization"]["source_checkpoint"].endswith("removed-parent.pt")
    assert not Path(loaded["source_checkpoint"]).exists()


def test_run_checkpoint_rejects_foreign_paths_and_mismatched_identity(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    identity = {"family": research.TASK_SCHEMA, "config_name": "thermal_u_h_full5000_v1"}
    (run / "experiment_identity.json").write_text(json.dumps(identity))
    checkpoint = run / "latest_model.pt"
    torch.save({"experiment_identity": identity, "epoch": 10}, checkpoint)
    payload, selected = research._read_run_checkpoint(run)
    assert payload["epoch"] == 10 and selected == checkpoint
    foreign = tmp_path / "foreign.pt"
    torch.save({"experiment_identity": identity, "epoch": 10}, foreign)
    with pytest.raises(ValueError, match="outside the declared research run"):
        research._read_run_checkpoint(run, str(foreign))
    torch.save({"experiment_identity": {**identity, "config_name": "thermal_u_d_full5000_v1"}}, checkpoint)
    with pytest.raises(ValueError, match="identity differs"):
        research._read_run_checkpoint(run)


class _RetryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0]))


class _InjectedOOMEngine:
    def __init__(self, model, initial_rng, *, fail):
        self.model = model
        self.initial_rng = initial_rng
        self.fail = fail
        self.device = torch.device("cpu")
        self.parent_age = 5
        self.config = SimpleNamespace(microbatch_cases=2, effective_cases=2)

    def fit(self, model, provider, output, **kwargs):
        del kwargs
        if self.fail:
            with torch.no_grad():
                model.weight.add_(10)
            provider.mutable_state += 1
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("CUDA out of memory: injected retry-boundary check")
        assert torch.equal(model.weight.detach(), torch.tensor([3.0]))
        assert provider.mutable_state == 0
        state = research.capture_rng_state()
        assert state["python"] == self.initial_rng["python"]
        assert state["numpy"][0] == self.initial_rng["numpy"][0]
        assert np.array_equal(state["numpy"][1], self.initial_rng["numpy"][1])
        assert torch.equal(state["torch_cpu"], self.initial_rng["torch_cpu"])
        return {"completed_epoch": 1, "status": "injected_retry_passed"}


def test_first_epoch_oom_retries_from_model_and_all_rng_boundary(tmp_path, monkeypatch):
    previous_rng = research.capture_rng_state()
    try:
        random.seed(73)
        np.random.seed(73)
        torch.manual_seed(73)
        expected_rng = research.capture_rng_state()
        model = _RetryModel()
        provider = SimpleNamespace(
            device=torch.device("cpu"),
            microbatch_size=2,
            _identity_microbatch_size=2,
            _research_case_limit=2,
            _research_config={"training": {"microbatch_cases": 2}},
            mutable_state=0,
        )
        provider.training_state_dict = lambda: {"mutable_state": provider.mutable_state}
        provider.load_training_state_dict = lambda state: setattr(provider, "mutable_state", state["mutable_state"])
        output = tmp_path / "retry-run"
        output.mkdir()
        (output / research._CONFIG_NAME).write_text(json.dumps({"microbatch_cases": 2}))
        first = _InjectedOOMEngine(model, expected_rng, fail=True)
        second = _InjectedOOMEngine(model, expected_rng, fail=False)
        engines = iter((second,))
        monkeypatch.setattr(research, "_engine_for", lambda *args, **kwargs: next(engines))
        result = research._fit_with_retries(
            first, model, provider, output,
            identity={}, stop_after=1, arm="A2-D",
        )
        assert result["status"] == "injected_retry_passed"
        assert (tmp_path / "retry-run.oom_attempt_01" / "oom_retry_01.json").is_file()
    finally:
        research.restore_rng_state(previous_rng)


def test_outer_oom_downshift_is_not_overwritten_by_initial_cli_microbatch(tmp_path, monkeypatch):
    options = {
        "backend": "dense_weighted",
        "query_tile_size": 1024,
        "checkpoint": "values",
        "context_backend": "segment",
    }
    config = EngineConfig(
        seed=4, microbatch_cases=8, effective_cases=48, total_epochs=10,
        training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION,
    )

    class _OutOfMemoryEngine(research.ResearchTrainingEngine):
        def fit(self, model, provider, output, **kwargs):
            del model, provider, kwargs
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "latest_model.pt").write_bytes(b"sealed latest checkpoint marker")
            raise RuntimeError("CUDA out of memory: injected outer-retry boundary")

    class _RetryEngine(research.ResearchTrainingEngine):
        def fit(self, model, provider, output, **kwargs):
            del model, kwargs
            assert self.nominal_microbatch_cases == 8
            assert self.runtime_microbatch_cases == 12
            assert self.config.microbatch_cases == 8
            assert provider.microbatch_size == 12
            runtime = json.loads((Path(output) / "execution_runtime.json").read_text())
            assert runtime["nominal_microbatch_cases"] == 8
            assert runtime["runtime_microbatch_cases"] == 12
            assert runtime["runtime_execution_options"] == options
            return {"status": "outer_oom_downshift_retained"}

    initial = _OutOfMemoryEngine(config, device="cpu", parent_age=5000, execution_options=options)
    retry = _RetryEngine(config, device="cpu", parent_age=5000, execution_options=options)
    model = _RetryModel()
    provider = SimpleNamespace(
        device=torch.device("cpu"),
        microbatch_size=8,
        _identity_microbatch_size=8,
        _research_config={},
        _research_case_limit=None,
    )
    monkeypatch.setattr(research, "_engine_for", lambda *args, **kwargs: retry)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = research._fit_with_retries(
        initial, model, provider, tmp_path / "outer-retry",
        identity={}, stop_after=5, arm="A2-H",
        runtime_microbatch_cases=24,
        runtime_execution_options=options,
    )
    assert result["status"] == "outer_oom_downshift_retained"


def test_local_oom_retry_restores_the_last_complete_optimizer_macro(tmp_path, monkeypatch):
    import fcntl

    previous_rng = research.capture_rng_state()
    try:
        random.seed(109)
        np.random.seed(109)
        torch.manual_seed(109)
        model = _RetryModel()
        provider = SimpleNamespace(
            sampling_dataset_id="thermal-membership-test",
            mutable_state=0,
            query_draws={"case-a": [], "case-b": []},
        )

        def make_batch(case_chunk, key):
            for case_id in case_chunk:
                provider.query_draws[case_id].append(tuple(map(int, key.native_indices(
                    1000, 8, case_id, "fluid", "primary",
                ))))
            return TaskBatch(None, None, None, {"sampling_key": key}, tuple(case_chunk))

        def training_state_dict():
            return {"mutable_state": provider.mutable_state}

        def load_training_state_dict(state):
            provider.mutable_state = int(state["mutable_state"])

        provider.make_batch = make_batch
        provider.training_state_dict = training_state_dict
        provider.load_training_state_dict = load_training_state_dict
        provider.loss_denominators = lambda microbatches, phase, arm: {
            "field": sum(len(batch.case_keys) for batch in microbatches),
        }
        engine = research.ResearchTrainingEngine(
            EngineConfig(seed=13, microbatch_cases=2, effective_cases=2, total_epochs=4,
                         training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION),
            device="cpu", parent_age=5,
        )
        output = tmp_path / "local-retry"
        output.mkdir()
        engine.bind_execution_runtime(output / "execution_runtime.json", nominal_microbatch_cases=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        order = ("case-a", "case-b")
        first_batches = engine.prepare_update_microbatches(
            provider, arm="A2-H", phase="joint", epoch=1, update_index=0,
            update_cases=order, sampling_dataset_id=provider.sampling_dataset_id,
        )

        def optimizer_snapshot():
            return research._cpu_state_snapshot(optimizer.state_dict())

        def assert_nested_equal(left, right):
            if torch.is_tensor(left):
                assert torch.equal(left, right)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    assert_nested_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for left_item, right_item in zip(left, right):
                    assert_nested_equal(left_item, right_item)
            else:
                assert left == right

        first_macro_weight = None
        first_macro_optimizer = None
        failed_entry_rng = None
        seen_cases = []
        second_macro_calls = 0

        def injected_parent_update(self, active_model, active_provider, active_optimizer, microbatches,
                                   *, phase, arm, epoch, update_index, **kwargs):
            nonlocal first_macro_weight, first_macro_optimizer, failed_entry_rng, second_macro_calls
            del self, phase, arm, epoch, kwargs
            cases = tuple(case for batch in microbatches for case in batch.case_keys)
            seen_cases.append(cases)
            denominator = active_provider.loss_denominators(microbatches, "joint", "A2-H")["field"]
            if update_index == 1:
                second_macro_calls += 1
                if second_macro_calls == 1:
                    assert torch.equal(active_model.weight.detach(), first_macro_weight)
                    assert_nested_equal(optimizer_snapshot(), first_macro_optimizer)
                    assert active_provider.mutable_state == 1
                    failed_entry_rng = research._cpu_state_snapshot(research.capture_rng_state())
                else:
                    assert cases == order
                    assert torch.equal(active_model.weight.detach(), first_macro_weight)
                    assert_nested_equal(optimizer_snapshot(), first_macro_optimizer)
                    assert active_provider.mutable_state == 1
                    current_rng = research.capture_rng_state()
                    assert current_rng["python"] == failed_entry_rng["python"]
                    assert current_rng["numpy"][0] == failed_entry_rng["numpy"][0]
                    assert np.array_equal(current_rng["numpy"][1], failed_entry_rng["numpy"][1])
                    assert torch.equal(current_rng["torch_cpu"], failed_entry_rng["torch_cpu"])
            active_optimizer.zero_grad(set_to_none=True)
            loss = sum(
                active_model.weight.square().sum() * (1.0 if case == "case-a" else 2.0)
                for case in cases
            ) / denominator
            loss.backward()
            active_optimizer.step()
            active_provider.mutable_state += 1
            if update_index == 0:
                first_macro_weight = active_model.weight.detach().clone()
                first_macro_optimizer = optimizer_snapshot()
            elif second_macro_calls == 1:
                with torch.no_grad():
                    active_model.weight.add_(17.0)
                random.random()
                np.random.random()
                torch.rand(4)
                raise RuntimeError("CUDA out of memory: injected partial second-macro update")
            return {}, {}, {"field": float(denominator)}, None

        monkeypatch.setattr(TrainingEngine, "_run_update", injected_parent_update)
        lock_path = output / ".training.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            engine._run_update(
                model, provider, optimizer, first_batches,
                phase="joint", arm="A2-H", epoch=1, update_index=0,
            )
            second_batches = engine.prepare_update_microbatches(
                provider, arm="A2-H", phase="joint", epoch=1, update_index=1,
                update_cases=order, sampling_dataset_id=provider.sampling_dataset_id,
            )
            engine._run_update(
                model, provider, optimizer, second_batches,
                phase="joint", arm="A2-H", epoch=1, update_index=1,
            )
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        runtime = json.loads((output / "execution_runtime.json").read_text())
        assert engine.config.microbatch_cases == 2
        assert engine.nominal_microbatch_cases == 2
        assert engine.runtime_microbatch_cases == 1
        assert engine.local_oom_retry_count == 1
        assert runtime["nominal_microbatch_cases"] == 2
        assert runtime["runtime_microbatch_cases"] == 1
        assert runtime["local_oom_retry_count"] == 1
        assert runtime["last_oom"]["status"] == "repacked_same_optimizer_macro"
        assert runtime["last_oom"]["ordered_case_keys"] == list(order)
        resumed_engine = research.ResearchTrainingEngine(
            EngineConfig(seed=13, microbatch_cases=2, effective_cases=2, total_epochs=4,
                         training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION),
            device="cpu", parent_age=5,
        )
        resumed_engine.bind_execution_runtime(
            output / "execution_runtime.json", nominal_microbatch_cases=2,
        )
        assert resumed_engine.config.microbatch_cases == 2
        assert resumed_engine.runtime_microbatch_cases == 1
        assert resumed_engine.local_oom_retry_count == 1
        mismatched_engine = research.ResearchTrainingEngine(
            EngineConfig(seed=13, microbatch_cases=1, effective_cases=2, total_epochs=4,
                         training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION),
            device="cpu", parent_age=5,
        )
        with pytest.raises(ValueError, match="nominal microbatch identity"):
            mismatched_engine.bind_execution_runtime(
                output / "execution_runtime.json", nominal_microbatch_cases=1,
            )
        assert seen_cases == [order, order, order]
        assert all(len(set(draws)) == 1 for draws in provider.query_draws.values())
        assert int(optimizer.state[model.weight]["step"]) == 2
        assert provider.mutable_state == 2
    finally:
        research.restore_rng_state(previous_rng)


def test_local_oom_exhaustion_does_not_enter_epoch_checkpoint_fallback(tmp_path, monkeypatch):
    output = tmp_path / "local-oom-exhausted"
    output.mkdir()
    model = _RetryModel()
    provider = SimpleNamespace(
        _identity_microbatch_size=2,
        training_state_dict=lambda: {},
    )

    class _ExhaustedEngine:
        device = torch.device("cpu")
        config = SimpleNamespace(microbatch_cases=2, effective_cases=2)

        def fit(self, *args, **kwargs):
            del args, kwargs
            raise research.ResearchLocalOOMExhausted("local macro CUDA OOM retries exhausted")

    monkeypatch.setattr(
        research, "_engine_for",
        lambda *args, **kwargs: pytest.fail("Local update exhaustion must not rewind from an epoch checkpoint."),
    )
    with pytest.raises(research.ResearchLocalOOMExhausted):
        research._fit_with_retries(
            _ExhaustedEngine(), model, provider, output,
            identity={}, stop_after=1, arm="A2-H",
        )
    assert not list(output.glob("oom_retry_*.json"))


def test_clean_stop_handler_writes_a_resumable_request(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    previous = research._install_clean_stop_handlers(output)
    try:
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
    finally:
        research._restore_handlers(previous)
    request = json.loads((output / "CLEAN_STOP_REQUEST.json").read_text())
    assert request["signal"] == int(signal.SIGINT)
    assert request["request_id"]


def test_resume_stop_supports_an_absolute_total_epoch_target():
    assert research._resolve_resume_stop(
        current_epoch=120, local_horizon=600, additional_epochs=None, stop_after_epoch=200,
    ) == 200
    assert research._resolve_resume_stop(
        current_epoch=120, local_horizon=600, additional_epochs=8, stop_after_epoch=None,
    ) == 128
    with pytest.raises(ValueError, match="Specify exactly one"):
        research._resolve_resume_stop(
            current_epoch=120, local_horizon=600, additional_epochs=8, stop_after_epoch=200,
        )
    with pytest.raises(ValueError, match="greater than the current checkpoint"):
        research._resolve_resume_stop(
            current_epoch=200, local_horizon=600, additional_epochs=None, stop_after_epoch=200,
        )


def test_runtime_pack_larger_than_nominal_preserves_identity_and_query_keys(tmp_path):
    options = {
        "backend": "dense_weighted",
        "query_tile_size": 512,
        "checkpoint": "tile",
        "context_backend": "segment",
    }
    engine_config = EngineConfig(
        seed=31, microbatch_cases=8, effective_cases=48, total_epochs=10,
        training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION,
    )
    runtime_path = tmp_path / "execution_runtime.json"
    engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    engine.bind_execution_runtime(
        runtime_path, nominal_microbatch_cases=8, default_execution_options=options,
    )

    case_order = tuple(f"thermal-{index:03d}" for index in range(48))

    class _BatchProvider:
        def __init__(self):
            self.microbatch_size = 8
            self._identity_microbatch_size = 8
            self.draws = {}

        def make_batch(self, cases, key):
            assert len(cases) <= self.microbatch_size
            for case_id in cases:
                self.draws[case_id] = tuple(map(int, key.native_indices(
                    128, 16, case_id, "fluid", "primary",
                )))
            return TaskBatch(None, None, None, {"sampling_key": key}, tuple(cases))

        def identity_payload(self):
            current_options = self.core.execution_options
            payload = {
                "microbatch_size": self._identity_microbatch_size,
                "model_config": {"execution_options": current_options},
                "adapter_state": {"core_config": {"execution_options": current_options}},
            }
            return research._restore_checkpoint_identity_metadata(self, payload)

    identity_core = _TinyCore(weight=1.0)
    identity_core.set_execution_options(**options)
    runtime_provider = _BatchProvider()
    runtime_provider.core = identity_core
    runtime_provider._research_identity_model_config = {"execution_options": options}
    runtime_provider._research_identity_adapter_state = {
        "core_config": {"execution_options": options},
    }
    identity_before = research._canonical_sha256(runtime_provider.identity_payload())
    runtime_options = {**options, "query_tile_size": 1024, "checkpoint": "values"}
    research._apply_runtime_execution_options(identity_core, runtime_provider, runtime_options)
    engine.set_runtime_overrides(
        provider=runtime_provider, microbatch_cases=24, execution_options=runtime_options,
    )
    runtime_batches = engine.prepare_update_microbatches(
        runtime_provider, arm="A2-H", phase="joint", epoch=3, update_index=11,
        update_cases=case_order, sampling_dataset_id="thermal-fixed-membership",
    )
    identity_after = research._canonical_sha256(runtime_provider.identity_payload())
    assert identity_after == identity_before
    assert identity_core.execution_options == runtime_options
    assert [len(batch.case_keys) for batch in runtime_batches] == [24, 24]
    assert tuple(case for batch in runtime_batches for case in batch.case_keys) == case_order
    runtime_draws = dict(runtime_provider.draws)

    nominal_engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    nominal_provider = _BatchProvider()
    nominal_batches = nominal_engine.prepare_update_microbatches(
        nominal_provider, arm="A2-H", phase="joint", epoch=3, update_index=11,
        update_cases=case_order, sampling_dataset_id="thermal-fixed-membership",
    )
    assert [len(batch.case_keys) for batch in nominal_batches] == [8, 8, 8, 8, 8, 8]
    assert tuple(case for batch in nominal_batches for case in batch.case_keys) == case_order
    assert nominal_provider.draws == runtime_draws
    assert sum(len(batch.case_keys) for batch in runtime_batches) == 48
    assert sum(len(batch.case_keys) for batch in nominal_batches) == 48

    engine._execution_runtime_path = runtime_path
    engine._write_execution_runtime(last_oom={"status": "execution_override_test"})
    saved = json.loads(runtime_path.read_text())
    assert saved["nominal_microbatch_cases"] == 8
    assert saved["runtime_microbatch_cases"] == 24
    assert saved["runtime_execution_options"] == runtime_options
    loaded_options = research._execution_options_for_run(
        tmp_path,
        {"execution": {"options": options}},
    )
    assert loaded_options == runtime_options

    resumed_engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    resumed_engine.bind_execution_runtime(
        runtime_path, nominal_microbatch_cases=8, default_execution_options=options,
    )
    resumed_provider = _BatchProvider()
    resumed_engine.set_runtime_overrides(provider=resumed_provider)
    assert resumed_engine.config.microbatch_cases == 8
    assert resumed_engine.nominal_microbatch_cases == 8
    assert resumed_engine.runtime_microbatch_cases == 24
    assert resumed_provider.microbatch_size == 24
    assert resumed_engine.runtime_execution_options == runtime_options
    mismatched_engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    with pytest.raises(ValueError, match="nominal microbatch identity"):
        mismatched_engine.bind_execution_runtime(
            runtime_path, nominal_microbatch_cases=16, default_execution_options=options,
        )
