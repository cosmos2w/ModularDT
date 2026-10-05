"""Named optimizer transplant keeps history and cannot silently change physics."""

import copy

import pytest
import torch
from channelthermal.training.lean_attachment import (
    MANIFEST_SHA256,
    attach_lean_checkpoint,
    transplant_optimizer_by_name,
    validate_lean_parent,
)
from torch import nn


class Fixture(nn.Module):
    def __init__(self, residual=False, wrong=False):
        super().__init__()
        self.core = nn.Module()
        self.core.backend = nn.Module()
        # Reverse registration order in the child: transplantation must use
        # names rather than zip the source and target moment positions.
        if residual:
            self.core.backend.tensor_residual = nn.Linear(2, 1)
        self.core.backend.physical = nn.Linear(2, 2 if wrong else 1)


def initialized_parent():
    source = Fixture()
    optimizer = torch.optim.AdamW(source.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad()
        source.core.backend.physical(torch.tensor([[1., 2.]])).square().sum().backward()
        optimizer.step()
    return source, optimizer.state_dict()


def test_named_transplant_preserves_each_common_moment_and_new_state_is_empty():
    source, saved = initialized_parent()
    before = copy.deepcopy(saved)
    target = Fixture(residual=True)
    optimizer = torch.optim.AdamW(target.parameters(), lr=1.)
    receipt = transplant_optimizer_by_name(source, target, saved, optimizer)
    source_named = dict(source.named_parameters())
    source_ids = dict(zip(source_named, saved["param_groups"][0]["params"], strict=True))
    for name, parameter in target.named_parameters():
        if "tensor_residual" in name:
            assert parameter not in optimizer.state
        else:
            for key, value in saved["state"][source_ids[name]].items():
                torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
    assert receipt["new_learning_rate"] == 3e-4
    assert receipt["restored_moment_tensors"] == 2
    for key, value in before["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(saved["state"][key][field], tensor, rtol=0, atol=0)


def test_transplant_rejects_changed_common_shape():
    source, saved = initialized_parent()
    target = Fixture(residual=True, wrong=True)
    with pytest.raises(ValueError, match="common parameter shape"):
        transplant_optimizer_by_name(source, target, saved, torch.optim.AdamW(target.parameters()))


def test_transplant_rejects_unaccounted_parameters_and_optimizer_groups():
    source, saved = initialized_parent()
    target = Fixture(residual=True)
    target.unrelated = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="outside the declared"):
        transplant_optimizer_by_name(source, target, saved, torch.optim.AdamW(target.parameters()))
    target = Fixture(residual=True)
    optimizer = torch.optim.AdamW([{ "params": target.core.backend.physical.parameters()},
                                  { "params": target.core.backend.tensor_residual.parameters()}])
    with pytest.raises(ValueError, match="single native"):
        transplant_optimizer_by_name(source, target, saved, optimizer)


def parent_contract():
    return {"checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
        "workflow": "forward", "epoch": 500, "selection_state": {"epoch": 500, "total_epochs": 1000},
        "model_config": {"core_honf": {"forward_architecture": "native_context_global_control_honf",
                                        "interface_model": {"hypergraph_options": {}}}},
        "optimizer_state_dict": {"state": "fixture"}, "rng_state": {"torch": "fixture"},
        "train_config": {"dataset": {"development_manifest_sha256": MANIFEST_SHA256,
            "development_subset": {"partitions": {"train": {"selected_count": 150}, "test": {"selected_count": 22}}}},
            "training": {"amp": False, "campaign": {"schedule_total_epochs": 1000,
                "physical_loss_policy_version": 2, "native_loss_denominators_start_epoch": 101,
                "organizer_gradient_policy": "ordinary_task_v1", "heat_null_response": {"coefficient": 0.0}}}}}


@pytest.mark.parametrize("change", ["optimizer", "rng", "manifest", "full", "age", "horizon", "amp", "already_attached"])
def test_parent_contract_rejects_nonmatching_or_nonresumable_input(change):
    checkpoint = parent_contract()
    validate_lean_parent(checkpoint)
    if change == "optimizer":
        checkpoint["optimizer_state_dict"] = None
    elif change == "rng":
        checkpoint["rng_state"] = None
    elif change == "manifest":
        checkpoint["train_config"]["dataset"]["development_manifest_sha256"] = "bad"
    elif change == "full":
        checkpoint["train_config"]["dataset"]["development_subset"] = None
    elif change == "age":
        checkpoint["epoch"] = 400
    elif change == "horizon":
        checkpoint["selection_state"]["total_epochs"] = 5000
    elif change == "amp":
        checkpoint["train_config"]["training"]["amp"] = True
    else:
        checkpoint["model_config"]["core_honf"]["interface_model"]["hypergraph_options"]["tensor_source_residual"] = True
    with pytest.raises(ValueError):
        validate_lean_parent(checkpoint)


class ConfigFixture:
    def __init__(self, payload):
        self.payload = copy.deepcopy(payload)

    def to_dict(self):
        return copy.deepcopy(self.payload)


def attachment_fixture():
    checkpoint = parent_contract()
    source, saved = initialized_parent()
    source.config = ConfigFixture(checkpoint["model_config"])
    target = Fixture()
    target.core.backend.tensor_residual = nn.Module()
    target.core.backend.tensor_residual.gamma = nn.ModuleDict({"MM": nn.Linear(2, 1)})
    for parameter in target.core.backend.tensor_residual.gamma.parameters():
        nn.init.zeros_(parameter)
    payload = copy.deepcopy(checkpoint["model_config"])
    payload["core_honf"]["interface_model"]["hypergraph_options"] = {
        "global_fast_reader": True, "tensor_source_residual": True, "residual_parent_epoch": 500}
    target.config = ConfigFixture(payload)
    checkpoint.update(model_state_dict=copy.deepcopy(source.state_dict()), optimizer_state_dict=saved,
                      campaign_training_state={"calibration": {"completed": True}},
                      global_normalization_stats={"mean": torch.tensor([2.])},
                      local_normalization_stats={"mean": torch.tensor([4.])},
                      rng_state={"torch": torch.get_rng_state()}, best_metrics={"field": 1.})
    checkpoint["train_config"]["training"].update(learning_rate=3e-4, weight_decay=1e-5)
    return checkpoint, source, target


def test_attachment_preserves_exact_common_weights_rng_calibration_normalizers_and_age():
    checkpoint, source, target = attachment_fixture()
    before = copy.deepcopy(checkpoint)
    child, receipt = attach_lean_checkpoint(checkpoint, source_model=source, target_model=target,
                                           target_config=target.config, source_sha256="fixture", source_path="/fixture.pt")
    for name, value in before["model_state_dict"].items():
        torch.testing.assert_close(child["model_state_dict"][name], value, rtol=0, atol=0)
    torch.testing.assert_close(child["rng_state"]["torch"], before["rng_state"]["torch"], rtol=0, atol=0)
    assert child["campaign_training_state"]["calibration"] == before["campaign_training_state"]["calibration"]
    for key in ("global_normalization_stats", "local_normalization_stats"):
        torch.testing.assert_close(child[key]["mean"], before[key]["mean"], rtol=0, atol=0)
    assert child["selection_state"] == before["selection_state"]
    assert child["epoch"] == 500
    assert receipt["new_moments"] == "empty"
    assert child["best_metrics"] == {}
    assert checkpoint["best_metrics"] == before["best_metrics"]


@pytest.mark.parametrize("change", ["nonzero_gamma", "undesired_option"])
def test_attachment_rejects_nonzero_residual_or_other_model_changes(change):
    checkpoint, source, target = attachment_fixture()
    if change == "nonzero_gamma":
        with torch.no_grad():
            target.core.backend.tensor_residual.gamma["MM"].weight.fill_(0.1)
    else:
        target.config.payload["core_honf"]["interface_model"]["hypergraph_options"]["other_math"] = True
    with pytest.raises(ValueError):
        attach_lean_checkpoint(checkpoint, source_model=source, target_model=target,
                               target_config=target.config, source_sha256="fixture", source_path="/fixture.pt")


def test_cli_rejects_wrong_parent_hash_before_checkpoint_deserialization(tmp_path, monkeypatch):
    import thermal_lean_continuation as preparation

    parent = tmp_path / "Run_3402_fixture" / "epoch_0500_model.pt"
    parent.parent.mkdir()
    parent.write_bytes(b"not the declared checkpoint")
    monkeypatch.setattr(preparation, "validate_generated_output", lambda path: tmp_path / "output")

    def unexpected_load(*args, **kwargs):
        pytest.fail("A mismatched checkpoint must reject before trusted deserialization.")

    monkeypatch.setattr(preparation, "load_trusted_checkpoint", unexpected_load)
    with pytest.raises(ValueError, match="content hash"):
        preparation.main(["--parent-checkpoint", str(parent), "--output-root", str(tmp_path / "output")])
