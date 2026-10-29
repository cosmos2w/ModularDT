"""Supervised one-pass correction targets respect immutable module topology."""

from __future__ import annotations

import torch

from channelthermal.inverse.differentiable_verifier import bounded_layout_correction_target


def test_bounded_layout_correction_target_ignores_unmatched_sampled_slots() -> None:
    sampled = torch.tensor(
        [[[0.20, 0.30, 0.10], [0.80, 0.70, -0.10], [0.60, 0.40, 0.20]]]
    )
    target = torch.tensor(
        [[[0.40, 0.10, 0.30], [0.75, 0.65, -0.20], [0.0, 0.0, 0.0]]]
    )
    sampled_present = torch.tensor([[1.0, 1.0, 1.0]])
    target_present = torch.tensor([[1.0, 1.0, 0.0]])

    delta, overlap = bounded_layout_correction_target(
        target, sampled, sampled_present, target_present, max_delta=0.05
    )

    torch.testing.assert_close(overlap, torch.tensor([[1.0, 1.0, 0.0]]))
    torch.testing.assert_close(
        delta,
        torch.tensor([[[0.05, -0.05, 0.05], [-0.05, -0.05, -0.05], [0.0, 0.0, 0.0]]]),
    )


def test_bounded_layout_correction_target_validates_contract_shapes() -> None:
    layout = torch.zeros(1, 2, 3)
    try:
        bounded_layout_correction_target(
            layout, layout[:, :1], torch.ones(1, 2), torch.ones(1, 2), max_delta=0.05
        )
    except ValueError as exc:
        assert "share shape" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("Mismatched correction target shapes were accepted.")
