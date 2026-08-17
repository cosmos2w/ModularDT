from __future__ import annotations

import pytest
import torch

from channelthermal.training_tools.losses import (
    channelthermal_field_channel_weights,
    channelthermal_field_mse,
)


def test_channelthermal_temperature_weight_follows_field_name_not_index() -> None:
    field_names = ["temperature", "u", "v", "p", "omega"]
    pred = torch.zeros(1, 1, len(field_names))
    target = torch.ones_like(pred)

    loss = channelthermal_field_mse(
        pred,
        target,
        {"temperature_weight": 6.0},
        field_names=field_names,
    )

    assert loss.item() == pytest.approx(2.0)


def test_channelthermal_explicit_weights_are_complete_override() -> None:
    weights = channelthermal_field_channel_weights(
        ["u", "v", "p", "omega", "temperature"],
        {
            "temperature_weight": 100.0,
            "field_channel_weights": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(weights, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_channelthermal_requires_one_explicit_weight_per_named_field() -> None:
    with pytest.raises(ValueError, match="exactly one value for each"):
        channelthermal_field_channel_weights(
            ["u", "v", "p", "omega", "temperature"],
            {"field_channel_weights": [1.0, 2.0]},
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
