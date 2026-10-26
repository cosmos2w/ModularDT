"""State and membership behavior of the narrow relation-guided attachment."""
import copy

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
                              "adapter_state": adapter},
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


@pytest.mark.parametrize("change", ["normalization", "membership", "packing", "optimizer", "parent_age"])
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
