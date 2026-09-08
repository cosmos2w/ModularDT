from __future__ import annotations

import pytest
import torch

from channelthermal.training_tools.losses import (
    channelthermal_field_channel_weights,
    channelthermal_field_mse,
)


def test_group_gradient_diagnostics_separate_bypasses_without_double_counting_total() -> None:
    from channelthermal.training.epoch import _fp64_group_norm

    values = [
        ("core.backend.module_message.weight", torch.tensor([3.0])),
        ("core.backend.receiver_query.weight", torch.tensor([4.0])),
        ("core.common.coarse_query.weight", torch.tensor([12.0])),
        ("core.common.local_message.weight", torch.tensor([0.0])),
        ("core.common.field_head.weight", torch.tensor([5.0])),
    ]
    total, groups = _fp64_group_norm(values)
    assert groups["backend"] == pytest.approx(13.0)
    assert groups["group_prepare"] == pytest.approx(3.0)
    assert groups["group_receiver"] == pytest.approx(4.0)
    assert groups["coarse"] == pytest.approx(12.0)
    assert groups["local"] == 0.0
    assert groups["head"] == pytest.approx(5.0)
    assert total == pytest.approx(194.0 ** 0.5)


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
