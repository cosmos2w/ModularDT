from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField

from channelthermal.workflows.train_forward import (
    _validate_optimizer_resume_compatibility,
    build_forward_optimizer,
    should_save_milestone_checkpoint,
)


class _ToyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.organizer = nn.Linear(3, 4)
        self.decoder = nn.Linear(4, 2)


class _ToyForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _ToyCore()
        self.output = nn.Linear(2, 1)
        self.frozen_local_surrogate = nn.Linear(3, 3)
        self.frozen_local_surrogate.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.core.organizer(inputs)
        return self.output(self.core.decoder(hidden))


def _config(organizer_learning_rate: float | None = None) -> dict[str, float | None]:
    return {
        "learning_rate": 3.0e-4,
        "organizer_learning_rate": organizer_learning_rate,
        "weight_decay": 1.0e-5,
    }


@pytest.mark.parametrize("include_null", [False, True])
def test_absent_or_null_organizer_lr_preserves_one_group(include_null: bool) -> None:
    model = _ToyForwardModel()
    config = {"learning_rate": 3.0e-4, "weight_decay": 1.0e-5}
    if include_null:
        config["organizer_learning_rate"] = None
    optimizer, inventory = build_forward_optimizer(model, config)

    assert len(optimizer.param_groups) == 1
    assert inventory["mode"] == "single"
    assert inventory["groups"][0]["name"] == "all"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-4)
    assert optimizer.param_groups[0]["params"] == [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]


def test_split_optimizer_membership_rates_and_frozen_exclusion() -> None:
    model = _ToyForwardModel()
    optimizer, inventory = build_forward_optimizer(model, _config(1.0e-4))

    assert len(optimizer.param_groups) == 2
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1.0e-4, 3.0e-4])
    organizer_names = inventory["groups"][0]["parameter_names"]
    prediction_names = inventory["groups"][1]["parameter_names"]
    assert organizer_names
    assert all(name.startswith("core.organizer.") for name in organizer_names)
    assert all(not name.startswith("core.organizer.") for name in prediction_names)
    assert all("frozen_local_surrogate" not in name for name in organizer_names + prediction_names)
    assert len(set(organizer_names + prediction_names)) == len(organizer_names + prediction_names)
    assert set(organizer_names + prediction_names) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def test_split_optimizer_finite_step_updates_both_groups() -> None:
    torch.manual_seed(7)
    model = _ToyForwardModel()
    optimizer, _ = build_forward_optimizer(model, _config(1.0e-4))
    organizer_before = model.core.organizer.weight.detach().clone()
    output_before = model.output.weight.detach().clone()

    loss = model(torch.randn(5, 3)).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(organizer_before, model.core.organizer.weight)
    assert not torch.equal(output_before, model.output.weight)


def test_split_resume_accepts_exact_inventory_and_rejects_mode_or_lr_drift() -> None:
    model = _ToyForwardModel()
    optimizer, inventory = build_forward_optimizer(model, _config(1.0e-4))
    checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_group_inventory": copy.deepcopy(inventory),
    }
    _validate_optimizer_resume_compatibility(checkpoint, inventory)

    _, one_group = build_forward_optimizer(_ToyForwardModel(), _config(None))
    with pytest.raises(ValueError, match="--initialize-checkpoint"):
        _validate_optimizer_resume_compatibility(checkpoint, one_group)

    _, changed_lr = build_forward_optimizer(_ToyForwardModel(), _config(2.0e-4))
    with pytest.raises(ValueError, match="--initialize-checkpoint"):
        _validate_optimizer_resume_compatibility(checkpoint, changed_lr)


def test_historical_one_group_checkpoint_can_resume_without_inventory() -> None:
    model = _ToyForwardModel()
    optimizer, inventory = build_forward_optimizer(model, _config(None))
    _validate_optimizer_resume_compatibility(
        {"optimizer_state_dict": optimizer.state_dict()},
        inventory,
    )


def test_model_only_initialization_is_independent_of_optimizer_grouping() -> None:
    source = _ToyForwardModel()
    source_state = copy.deepcopy(source.state_dict())
    build_forward_optimizer(source, _config(None))
    target = _ToyForwardModel()
    build_forward_optimizer(target, _config(1.0e-4))

    incompatible = target.load_state_dict(source_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


class _FixedHONFWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = HONFNeuralField(
            UnifiedForwardConfig.from_dict(
                {
                    "field_dim": 2,
                    "domain_length_x": 4.0,
                    "domain_length_y": 2.0,
                    "num_env_tokens_x": 3,
                    "num_env_tokens_y": 2,
                    "num_hyperedges": 3,
                    "hidden_dim": 16,
                    "dropout": 0.0,
                    "decoder_mode": "enhanced_honf_pairwise",
                    "pairwise_kernel_hidden_dim": 16,
                    "organizer_mode": "fixed_projection",
                    "mechanism_state_mode": "descriptor_first",
                    "field_assembly_mode": "edge_additive",
                }
            )
        )

    def forward(self, batch: BatchData) -> torch.Tensor:
        return self.core(batch)["pred_field"]


def test_fixed_projection_split_optimizer_updates_fixed_organizer_and_prediction() -> None:
    torch.manual_seed(17)
    model = _FixedHONFWrapper()
    optimizer, inventory = build_forward_optimizer(model, _config(1.0e-4))
    organizer_names = inventory["groups"][0]["parameter_names"]
    assert "core.organizer.module_score.weight" in organizer_names
    assert "core.organizer.env_score.weight" in organizer_names

    batch = BatchData(
        module_centers=torch.rand(1, 3, 2),
        module_present=torch.tensor([[1.0, 1.0, 0.0]]),
        module_features=torch.randn(1, 3, 4),
        global_context=torch.randn(1, 3),
        query_xy=torch.rand(1, 7, 2),
        query_time=None,
        target_field=None,
        case_name="fixed-split-optimizer",
        metadata={},
    )
    organizer_before = model.core.organizer.module_score.weight.detach().clone()
    prediction_before = model.core.decoder.edge_head.net[-1].weight.detach().clone()
    loss = model(batch).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(organizer_before, model.core.organizer.module_score.weight)
    assert not torch.equal(prediction_before, model.core.decoder.edge_head.net[-1].weight)


def test_explicit_milestone_checkpoint_schedule_retains_only_named_epochs() -> None:
    config = {"save_epoch_milestones": [250, 500, 1000, 1500, 2500]}

    assert should_save_milestone_checkpoint(250, config)
    assert should_save_milestone_checkpoint(1500, config)
    assert not should_save_milestone_checkpoint(1499, config)
    assert not should_save_milestone_checkpoint(2000, config)
