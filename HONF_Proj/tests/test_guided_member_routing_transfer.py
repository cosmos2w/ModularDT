"""State and membership behavior of the narrow relation-guided attachment."""
import copy
import json

import pytest
import torch

from honf_runtime.guided_member_routing_child import prepare_guided_transfer, same_state, WORKFLOW
from honf_runtime.unified_training import _resume_identity_amendment


def attachment(tmp_path):
    model = torch.nn.Linear(2, 1)
    model.architecture = "R13-4-Guide"
    adapter = {
        "normalization_stats": {"u": {"mean": 1., "std": 2.}}, "heat_scale": 3.,
        "native_grid_nx_ny": [128, 64], "environment_grid_nx_ny": [24, 8],
        "core": {"base_model_config": {"context_depth": 2}, "input_dataset": "thermal",
                 "input_capability": "joint_prescribed_records_v1", "entmax_temperature": 1.,
                 "query_tile_size": 512},
    }
    identity = {
        "provider_identity": {"training_case_ids": ["0001"], "validation_case_ids": ["0274"],
                              "adapter_state": adapter,
                              "source_binding": {"dataset_path": "/original/packed_dataset.h5",
                                                 "dataset_size_bytes": 1234,
                                                 "dataset_mtime_ns": 5678,
                                                 "manifest_path": "/original/manifest.json"}},
        "optimizer": {"lr": 3e-4}, "optimizer_schedule_contract": [{"kind": "constant"}],
        "sampling_dataset_id": "same-original-dataset",
        "engine_config": {"seed": 0, "effective_cases": 48, "microbatch_cases": 24},
    }
    parent = {"epoch": 500, "arm": "R13-3-Relative",
              "workflow": "interaction_reused_member_routing_joint_training_v2",
              "model_state_dict": copy.deepcopy(model.state_dict()), "experiment_identity": identity}
    current = copy.deepcopy(identity)
    current.update(workflow=WORKFLOW, training_membership="full_train600")
    source_path = tmp_path / "epoch_0500_model.pt"
    declaration = {"source_epoch": 500, "source_checkpoint": str(source_path)}
    return model, parent, current, declaration, source_path


def test_attachment_preserves_parent_state_and_declares_changed_reader(tmp_path):
    model, parent, current, declaration, source = attachment(tmp_path)
    before = copy.deepcopy(parent)
    result = prepare_guided_transfer(model, None, parent, current, declaration, source_path=source)
    assert same_state(before, parent)
    assert result["new_parameter_names"] == []
    assert result["source_epoch"] == 500
    assert result["target_arm"] == "R13-4-Guide"


@pytest.mark.parametrize("change", ["normalization", "membership", "packing", "optimizer", "parent_age",
                                  "dataset_binding", "missing_dataset_binding"])
def test_attachment_rejects_silent_training_state_changes(tmp_path, change):
    model, parent, current, declaration, source = attachment(tmp_path)
    if change == "normalization":
        current["provider_identity"]["adapter_state"]["normalization_stats"]["u"]["std"] = 4.
    elif change == "membership":
        current["provider_identity"]["training_case_ids"] = ["another"]
    elif change == "packing":
        current["engine_config"]["microbatch_cases"] = 12
    elif change == "optimizer":
        current["optimizer"]["lr"] = 1e-3
    elif change == "dataset_binding":
        current["provider_identity"]["source_binding"]["dataset_mtime_ns"] += 1
    elif change == "missing_dataset_binding":
        del current["provider_identity"]["source_binding"]
    else:
        parent["epoch"] = 501
    with pytest.raises(ValueError):
        prepare_guided_transfer(model, None, parent, current, declaration, source_path=source)


def test_scratch_extension_interface_changes_only_execution_horizon():
    saved = {"workflow": WORKFLOW, "initialization": {"source_parent_age": 0},
             "engine_config": {"total_epochs": 100, "checkpoint_epochs": [100]},
             "optimizer": {"lr": 3e-4}, "data": "unchanged"}
    current = copy.deepcopy(saved)
    current["engine_config"].update(total_epochs=200, checkpoint_epochs=[100, 200])
    amendment = _resume_identity_amendment(saved, current, allow_horizon_extension=True)
    assert amendment["optimizer_schedule_unchanged"]
    current["optimizer"]["lr"] = 1e-3
    with pytest.raises(ValueError):
        _resume_identity_amendment(saved, current, allow_horizon_extension=True)


def output_extension_identities():
    saved = {"workflow": WORKFLOW, "initialization": {"source_parent_age": 0},
             "engine_config": {"total_epochs": 100, "checkpoint_epochs": [100],
                               "monitor_every": 100, "latest_every": 1, "seed": 0},
             "optimizer": {"lr": 3e-4}, "provider_identity": {"normalization": "TRAIN600"}}
    current = copy.deepcopy(saved)
    current["engine_config"].update(total_epochs=500, checkpoint_epochs=[200, 500], curve_every=1)
    return saved, current


def test_guided_extension_accepts_explicit_output_policy_without_mutating_source():
    saved, current = output_extension_identities()
    original = copy.deepcopy((saved, current))
    amendment = _resume_identity_amendment(saved, current, allow_horizon_extension=True)
    assert amendment["output_policy"] == {"checkpoint_epochs": [200, 500], "curve_every": 1}
    assert (saved, current) == original
    with pytest.raises(ValueError):
        _resume_identity_amendment(saved, current)


@pytest.mark.parametrize("change", ["optimizer", "normalization", "validation", "seed", "bad_curves",
                                  "duplicate_checkpoint", "checkpoint_after_horizon", "legacy_reader"])
def test_guided_output_extension_keeps_scientific_and_historical_resume_checks(change):
    saved, current = output_extension_identities()
    if change == "optimizer":
        current["optimizer"]["lr"] = 1e-3
    elif change == "normalization":
        current["provider_identity"]["normalization"] = "changed"
    elif change == "validation":
        current["engine_config"]["monitor_every"] = 1
    elif change == "seed":
        current["engine_config"]["seed"] = 1
    elif change == "bad_curves":
        current["engine_config"]["curve_every"] = 0
    elif change == "duplicate_checkpoint":
        current["engine_config"]["checkpoint_epochs"] = [200, 200, 500]
    elif change == "checkpoint_after_horizon":
        current["engine_config"]["checkpoint_epochs"] = [200, 600]
    else:
        saved["workflow"] = current["workflow"] = "interaction_reused_member_routing_joint_training_v2"
    with pytest.raises(ValueError):
        _resume_identity_amendment(saved, current, allow_horizon_extension=True)


def test_guided_resume_restores_custom_output_policy_after_interruption(tmp_path, monkeypatch):
    from honf_runtime import guided_member_routing_training as training

    config = {"member_routing": {"arm": "R13-4-Guide"},
              "training": {"local_horizon": 100, "checkpoint_epochs": [100], "curve_every": 10,
                            "wall_clock_budget_seconds": 18000},
              "initialization": {"source_parent_age": 0}}
    manifest = {"schema": training.MANIFEST_SCHEMA, "config": config, "config_path": "scratch.json"}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    payload = {"epoch": 100, "experiment_identity": {"engine_config": {
        "total_epochs": 100, "checkpoint_epochs": [100]}}}
    checkpoint = tmp_path / "latest_model.pt"
    monkeypatch.setattr(training, "resolve_checkpoint", lambda *a: checkpoint)
    monkeypatch.setattr(training, "load_trusted_checkpoint", lambda *a, **kw: payload)
    monkeypatch.setattr(training.research, "resolve_device", lambda *a, **kw: ("cpu", {}))
    monkeypatch.setattr(training.maintained, "_ensure_workspace_storage_bindings", lambda: {})
    monkeypatch.setattr(training, "_build", lambda *a: (object(), object()))
    monkeypatch.setattr(training.maintained, "_storage_estimate", lambda *a, **kw: (0, 0))
    monkeypatch.setattr(training.maintained, "_prepare_storage", lambda *a, **kw: {})
    monkeypatch.setattr(training, "training_identity", lambda c: {"recipe_budget": c["training"]["wall_clock_budget_seconds"]})
    calls = []
    monkeypatch.setattr(training.maintained, "_fit", lambda *a, **kw: calls.append((a[2], kw)))
    training.resume(tmp_path, stop_after_epoch=500, extend_horizon=True,
                    checkpoint_epochs=[200, 500], curve_every=1, wall_clock_budget_seconds=36000)
    assert calls[-1][1]["engine_checkpoint_epochs"] == [200, 500]
    assert calls[-1][0]["training"]["curve_every"] == 1
    assert calls[-1][1]["wall_clock_budget_seconds"] == 36000
    assert calls[-1][1]["training_identity"]["recipe_budget"] == 18000
    # A clean pause does not shrink the saved horizon or restore default outputs.
    payload.update(epoch=201, experiment_identity={"engine_config": {
        "total_epochs": 500, "checkpoint_epochs": [200, 500], "curve_every": 1}})
    training.resume(tmp_path, stop_after_epoch=500)
    assert calls[-1][1]["engine_horizon"] == 500
    assert calls[-1][1]["engine_checkpoint_epochs"] == [200, 500]
    assert calls[-1][0]["training"]["curve_every"] == 1
    assert calls[-1][1]["wall_clock_budget_seconds"] == 36000
    assert calls[-1][1]["allow_horizon_extension"] is False
