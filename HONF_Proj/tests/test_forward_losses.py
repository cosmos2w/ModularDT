from __future__ import annotations

import pytest
import torch

from honf_forward_core.training.losses import weighted_channel_mse


def test_weighted_channel_mse_uses_every_explicit_channel_weight() -> None:
    pred = torch.zeros(1, 2, 5)
    target = torch.ones_like(pred)

    loss = weighted_channel_mse(pred, target, [1.0, 2.0, 3.0, 4.0, 5.0])

    assert loss.item() == pytest.approx(3.0)


def test_weighted_channel_mse_combines_channel_and_point_weights() -> None:
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.zeros_like(pred)

    loss = weighted_channel_mse(
        pred,
        target,
        channel_weights=[2.0, 0.5],
        point_weights=torch.tensor([[1.0, 3.0]]),
    )

    expected = (1.0 * (2.0 * 1.0**2 + 0.5 * 2.0**2) + 3.0 * (2.0 * 3.0**2 + 0.5 * 4.0**2)) / 8.0
    assert loss.item() == pytest.approx(expected)


def test_weighted_channel_mse_requires_complete_weight_vector() -> None:
    pred = torch.zeros(1, 2, 5)

    with pytest.raises(ValueError, match="exactly one entry per output channel"):
        weighted_channel_mse(pred, pred, [1.0, 2.0])


def test_weighted_channel_mse_rejects_incompatible_point_weights() -> None:
    pred = torch.zeros(2, 3, 4)

    with pytest.raises(ValueError, match="not broadcastable"):
        weighted_channel_mse(pred, pred, [1.0] * 4, torch.ones(2, 2))
