"""Small deterministic tensor fixtures for inverse-model unit tests."""

from __future__ import annotations

import torch


def request_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    type_id = torch.tensor([[0, 3, -1, -1]], dtype=torch.long).repeat(batch, 1)
    relation_id = torch.tensor([[0, 2, -1, -1]], dtype=torch.long).repeat(batch, 1)
    active = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32).repeat(batch, 1)
    return {
        "type_id": type_id,
        "relation_id": relation_id,
        "target_normalized": torch.tensor([[0.2, -0.4, 0.0, 0.0]]).repeat(batch, 1),
        "target_mask": active.clone(),
        "tolerance_normalized": torch.full((batch, 4), 0.05) * active,
        "range_normalized": torch.zeros(batch, 4, 2),
        "range_mask": torch.tensor([[0, 1, 0, 0]], dtype=torch.float32).repeat(batch, 1),
        "priority": torch.tensor([[2, 3, 0, 0]], dtype=torch.float32).repeat(batch, 1),
        "weight": torch.tensor([[1.0, 2.0, 0.0, 0.0]]).repeat(batch, 1),
        "region": torch.zeros(batch, 4, 4),
        "region_mask": torch.zeros(batch, 4),
        "active_mask": active,
    }


def permute_request(request: dict[str, torch.Tensor], order: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[:, order] for name, value in request.items()}
