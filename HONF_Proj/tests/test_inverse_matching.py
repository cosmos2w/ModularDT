"""Canonical/default and optional matching for `G` versus `G_hat`."""

from __future__ import annotations

import torch
import numpy as np

from honf_inverse_core.models.matching import match_tokens
from channelthermal.inverse.evaluation.scoring import compact_plan_distance


def test_canonical_and_sinkhorn_matching_contracts() -> None:
    target = torch.tensor([[[0.0], [1.0], [2.0]]])
    prediction = target[:, [2, 0, 1]]
    assert torch.equal(match_tokens(prediction, target, method="canonical"), target)
    sinkhorn = match_tokens(prediction, target, method="sinkhorn", sinkhorn_temperature=0.01, sinkhorn_iterations=30)
    torch.testing.assert_close(sinkhorn, prediction, atol=1e-3, rtol=1e-3)


def test_optional_matching_is_wired_into_plan_distance() -> None:
    target = np.zeros((3, 12), dtype=np.float32)
    target[:, 1] = [0.0, 0.5, 1.0]
    prediction = target[[2, 0, 1]]
    canonical = compact_plan_distance(prediction, target)
    sinkhorn = compact_plan_distance(prediction, target, matching_mode="sinkhorn")
    assert canonical > 0.0
    assert sinkhorn < canonical
