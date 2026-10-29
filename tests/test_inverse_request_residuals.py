"""Directional residual channel for the one-pass inverse corrector."""

from __future__ import annotations

import pytest
import torch

from channelthermal.inverse.differentiable_functionals import normalized_request_residuals


def test_request_residuals_encode_too_low_vs_too_high() -> None:
    request = {
        "type_id": torch.zeros(1, 4, dtype=torch.long),
        "relation_id": torch.tensor([[1, 0, 2, 2]], dtype=torch.long),
        "target_normalized": torch.zeros(1, 4),
        "tolerance_normalized": torch.zeros(1, 4),
        "range_normalized": torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]]),
        "active_mask": torch.ones(1, 4),
        "weight": torch.ones(1, 4),
    }
    residual, loss = normalized_request_residuals(
        torch.tensor([[-2.0, 2.0, -2.0, 2.0]]),
        request,
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(residual, torch.tensor([[-2.0, 2.0, -1.0, 1.0]]))
    assert float(loss) == pytest.approx(2.5)
