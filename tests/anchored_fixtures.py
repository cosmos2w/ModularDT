"""Small native scenes and independent historical read oracle for anchored tests."""
from __future__ import annotations
import torch
from torch import Tensor
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.interaction_reused_member_routing.core import _positive_softmax

def _base_config() -> dict:
    return ContentHypergraphCoreV2(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=12,
        output_dim=4,
        architecture="A2-D",
        context_dim=16,
        num_heads=4,
        context_depth=2,
        local_hidden_dim=16,
        dropout=0.0,
        init_seed=731,
    ).get_config()

def _scene() -> SceneBatch:
    torch.manual_seed(7601)
    batch, modules, environments = 2, 3, 4
    module_mask = torch.tensor([[True, True, True], [True, False, False]])
    environment_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    module_coordinates = torch.tensor([
        [[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]],
        [[0.2, -0.1], [0.0, 0.0], [0.0, 0.0]],
    ])
    environment_coordinates = torch.tensor([
        [[0.0, 0.4], [0.4, 0.4], [0.8, 0.4], [1.2, 0.4]],
        [[0.1, 0.5], [0.8, 0.5], [0.0, 0.0], [0.0, 0.0]],
    ])
    return SceneBatch(
        module_features=torch.randn(batch, modules, 9),
        environment_features=torch.randn(batch, environments, 6),
        module_coordinates=module_coordinates,
        environment_coordinates=environment_coordinates,
        module_mask=module_mask,
        environment_mask=environment_mask,
        module_measure=torch.ones(batch, modules),
        environment_measure=torch.tensor([[0.4, 0.5, 0.6, 0.7], [0.3, 0.8, 0.0, 0.0]]),
        module_ids=torch.tensor([[27, 34, 58], [81, -1, -1]]),
        environment_ids=torch.tensor([[110, 114, 121, 130], [210, 211, -1, -1]]),
        local_radius=torch.full((batch, modules), 0.2),
        operating_features=torch.randn(batch, 12),
        coordinate_origin=torch.zeros(batch, 2),
        coordinate_scale=torch.ones(batch, 2),
        module_measure_reference=torch.ones(batch),
        environment_measure_reference=torch.ones(batch),
        scene_key="r13-6-faithful-test-scene",
        normalizer_key="r13-6-faithful-test-normalizer",
        coordinate_frame_key="r13-6-faithful-test-frame",
        module_characteristic_lengths=torch.full((batch, modules, 2), 0.12),
        environment_characteristic_lengths=torch.full((batch, environments, 2), 0.1),
    )

def _query() -> QueryBatch:
    torch.manual_seed(7602)
    batch, queries = 2, 4
    x = torch.linspace(0.05, 1.1, queries)
    coordinates = torch.stack((x, torch.linspace(-0.1, 0.2, queries)), dim=-1)
    return QueryBatch(
        coordinates=coordinates[None].expand(batch, -1, -1).clone(),
        query_mask=torch.ones(batch, queries, dtype=torch.bool),
        query_ids=torch.arange(500, 500 + queries)[None].expand(batch, -1).clone(),
        scene_key="r13-6-faithful-test-scene",
        coordinate_frame_key="r13-6-faithful-test-frame",
    )

def _normalize_nonnegative(value: Tensor, allowed: Tensor) -> Tensor:
    """Normalize a nonnegative tensor, returning zeros for an empty row."""
    masked = value * allowed.to(value.dtype)
    total = masked.sum(-1, keepdim=True)
    return torch.where(total > 0, masked / total.clamp_min(torch.finfo(value.dtype).tiny),
                       torch.zeros_like(masked))

def original_role_conditional_likelihood(prior: Tensor, ell: Tensor, candidate: Tensor) -> Tensor:
    """Return the original ME ``P exp(ell)`` law, normalized within one role.

    ``ell`` is already calibrated against this candidate set. This follows the
    original reader's conditional-prior normalization and positive softmax,
    including exact zero output for empty candidate roles.
    """
    if prior.shape != ell.shape or prior.shape != candidate.shape or candidate.dtype != torch.bool:
        raise ValueError("Role prior, likelihood and boolean candidates must have equal shapes.")
    if not prior.is_floating_point() or not ell.is_floating_point():
        raise TypeError("Role prior and likelihood must be floating point.")
    conditional_prior = _normalize_nonnegative(prior, candidate)
    tiny = torch.finfo(prior.dtype).tiny
    log_prior = torch.where(
        candidate,
        conditional_prior.clamp_min(tiny).log(),
        torch.zeros_like(conditional_prior),
    )
    return _positive_softmax(ell + log_prior, candidate)
