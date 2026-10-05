"""Explicit frozen-host fit state cannot become a physical-model continuation."""

import copy

import pytest
import torch
from channelthermal.training.campaign import validate_campaign
from channelthermal.training.interface_fit import (
    FIT_KIND,
    FIT_PARAMETER_PREFIX,
    attach_interface_fit_checkpoint,
    freeze_interface_fit_model,
    validate_interface_fit_parent,
    validate_interface_fit_resume,
)
from channelthermal.training.lean_attachment import MANIFEST_SHA256, validate_lean_parent
from torch import nn


class Config:
    def __init__(self, value):
        self.value = copy.deepcopy(value)

    def to_dict(self):
        return copy.deepcopy(self.value)


class Host(nn.Module):
    def __init__(self, new=False):
        super().__init__()
        self.core = nn.Module()
        self.core.backend = nn.Module()
        self.core.backend.physical = nn.Linear(2, 1)
        self.register_buffer("normalizer", torch.tensor([3.]))
        if new:
            block = nn.Module()
            block.content_encoder = nn.Linear(2, 2)
            block.gamma = nn.ModuleDict({"QM": nn.Linear(2, 1)})
            for p in block.gamma.parameters():
                nn.init.zeros_(p)
            self.core.backend.tensor_query_interaction = block

    def set_training_progress(self, **kwargs):
        self.progress = kwargs

    def forward(self, x):
        block = self.core.backend.tensor_query_interaction
        correction = block.gamma["QM"](block.content_encoder(x))
        return self.core.backend.physical(x) * (1 + correction)


def fixture():
    source = Host()
    model = {"core_honf": {"forward_architecture": "native_context_global_control_honf",
        "interface_model": {"hypergraph_options": {"global_fast_reader": True}}},
        "channelthermal": {"global_feature_schema": "source_local_v3"}}
    dataset = {"development_manifest_sha256": MANIFEST_SHA256,
        "development_subset": {"partitions": {"train": {"selected_count": 150}, "test": {"selected_count": 22}}}}
    parent_training = {"amp": False, "learning_rate": 3e-4, "weight_decay": 1e-5,
                       "campaign": {"physical_loss_policy_version": 2}}
    parent = {"checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
        "workflow": "forward", "epoch": 1000, "selection_state": {"epoch": 1000, "total_epochs": 1000},
        "model_config": model, "model_state_dict": copy.deepcopy(source.state_dict()),
        "train_config": {"dataset": dataset, "loss": {"field_weight": 1.}, "training": parent_training},
        "rng_state": {"torch": torch.get_rng_state()}, "optimizer_state_dict": {"state": {0: "old moment"}},
        "global_normalization_stats": {"mean": torch.tensor([2.])},
        "local_normalization_stats": {"mean": torch.tensor([4.])},
        "campaign_training_state": {"response_scale": .1, "response_scale_samples": [.1] * 5,
                                     "matched_continuation_attachment": {"kind": "old lineage"}}}
    target = Host(new=True)
    payload = copy.deepcopy(model)
    payload["core_honf"]["interface_model"]["hypergraph_options"].update(
        tensor_query_interaction=True, query_interaction_mode="add", query_interface_parent_epoch=1000)
    target.config = Config(payload)
    fit = copy.deepcopy(parent["train_config"])
    fit["model"] = payload
    fit["training"].update(seed=0, epochs=100, campaign={"name": "fixture", "arm": "H-add",
        "parent": {"checkpoint": "/Run_3601_fixture/epoch_1000_model.pt", "epoch": 1000, "kind": FIT_KIND},
        "interface_fit": {"backbone_epoch": 1000, "parameter_prefix": FIT_PARAMETER_PREFIX},
        "matched_fresh_initialization": False, "schedule_total_epochs": 1000,
        "organizer_gradient_policy": "ordinary_task_v1", "physical_loss_policy_version": 2,
        "native_loss_denominators_start_epoch": 101, "structural_weight": 0.0,
        "heat_null_response": {"coefficient": 0.0}})
    return parent, target, fit


def attach(parent, target, fit):
    return attach_interface_fit_checkpoint(parent, target_model=target, target_config=target.config,
        fit_config=fit, source_path="/Run_3601_fixture/epoch_1000_model.pt", source_sha256="provenance")


def test_fit_attachment_preserves_host_and_normalizers_and_starts_empty_new_optimizer():
    parent, target, fit = fixture()
    before = copy.deepcopy(parent)
    child, receipt = attach(parent, target, fit)
    assert child["epoch"] == 0 and child["selection_state"] == {"epoch": 0, "total_epochs": 1000}
    assert child["optimizer_state_dict"]["state"] == {}
    assert all(n.startswith(FIT_PARAMETER_PREFIX) for n in receipt["new_parameter_names"])
    assert receipt["inherited_backbone_epoch"] == 1000
    assert child["campaign_training_state"]["response_scale_samples"] == [.1] * 5
    assert "matched_continuation_attachment" not in child["campaign_training_state"]
    for key in ("global_normalization_stats", "local_normalization_stats", "rng_state"):
        torch.testing.assert_close(child[key][next(iter(child[key]))], before[key][next(iter(before[key]))], rtol=0, atol=0)
    validate_interface_fit_resume(child, target, fit["training"]["campaign"])
    optimizer = torch.optim.AdamW([p for p in target.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
    x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    for _ in range(3):
        optimizer.zero_grad()
        target(x).square().sum().backward()
        optimizer.step()
    assert x.grad is not None and torch.count_nonzero(x.grad)
    assert target.core.backend.tensor_query_interaction.content_encoder.weight.grad.abs().sum() > 0
    for n, value in before["model_state_dict"].items():
        torch.testing.assert_close(target.state_dict()[n], value, rtol=0, atol=0)
    assert len(optimizer.state) == len(receipt["new_parameter_names"])


@pytest.mark.parametrize("change", ["age", "formal", "tensor_h", "missing_rng", "base_not_fast"])
def test_fit_parent_rejects_other_backbones(change):
    parent, _, _ = fixture()
    if change == "age": parent["epoch"] = 500
    elif change == "formal": parent["train_config"]["dataset"]["development_subset"] = None
    elif change == "tensor_h": parent["model_config"]["core_honf"]["interface_model"]["hypergraph_options"]["tensor_source_residual"] = True
    elif change == "missing_rng": parent["rng_state"] = None
    else: parent["model_config"]["core_honf"]["interface_model"]["hypergraph_options"]["global_fast_reader"] = False
    with pytest.raises(ValueError): validate_interface_fit_parent(parent)


def test_e500_lean_contract_stays_distinct_and_lazy_freeze_is_supported():
    parent, target, _ = fixture()
    with pytest.raises(ValueError): validate_lean_parent(parent)
    target.lazy_physical = nn.LazyLinear(1)
    freeze_interface_fit_model(target)
    assert target.lazy_physical.weight.requires_grad is False


@pytest.mark.parametrize("change", ["gamma", "physical_option", "loss", "unexpected_lineage", "mode"])
def test_fit_rejects_silent_changes(change):
    parent, target, fit = fixture()
    if change == "gamma":
        target.core.backend.tensor_query_interaction.gamma["QM"].bias.data.fill_(.1)
    elif change == "physical_option": target.config.value["core_honf"]["hidden_dim"] = 99
    elif change == "loss": fit["loss"]["field_weight"] = 2.
    elif change == "unexpected_lineage": fit["training"]["campaign"]["matched_fresh_initialization"] = True
    else: fit["model"]["core_honf"]["interface_model"]["hypergraph_options"]["query_interaction_mode"] = "joint"
    with pytest.raises(ValueError): attach(parent, target, fit)


def test_campaign_accepts_only_declared_frozen_fit_without_weakening_ordinary_native():
    _, _, fit = fixture()
    validate_campaign(fit)
    fit["training"]["campaign"].pop("interface_fit")
    with pytest.raises(ValueError): validate_campaign(fit)
