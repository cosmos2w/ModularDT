"""Query-selected max-product retrieval over native hypergraph memberships."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .contracts import (
    NativeQueryRetrievalPlan,
    query_batch_identity,
    query_batch_tensor_versions,
)
from .hypergraph import NativeRetrievalHypergraph


def _threshold(value: float, name: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold >= 0.5:
        raise ValueError(f"{name} must be finite and lie in [0,0.5).")
    return threshold


def _s5(value: Tensor) -> Tensor:
    t = value.clamp(0.0, 1.0)
    return (t.pow(3) * (10.0 + t * (-15.0 + 6.0 * t))).clamp(0.0, 1.0)


def query_seed_permissions(
    query_scores: Tensor,
    seed_valid: Tensor,
    query_mask: Tensor,
    threshold: float,
) -> Tensor:
    """Apply the declared density-relative query-to-seed permission rule."""
    tau = _threshold(threshold, "tau_query")
    if query_scores.ndim != 4:
        raise ValueError("query_scores must have shape [B,Hd,Q,S].")
    batch, _heads, queries, seeds = query_scores.shape
    if seed_valid.shape != (batch, seeds) or seed_valid.dtype != torch.bool:
        raise ValueError("seed_valid must be boolean [B,S].")
    if query_mask.shape != (batch, queries) or query_mask.dtype != torch.bool:
        raise ValueError("query_mask must be boolean [B,Q].")
    if seeds == 0:
        return torch.zeros_like(query_scores)
    eligible = seed_valid[:, None, None, :] & query_mask[:, None, :, None]
    row_has_seed = eligible.any(dim=-1, keepdim=True)
    masked = query_scores.masked_fill(~eligible, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    row_max = torch.where(row_has_seed, row_max, torch.zeros_like(row_max))
    delta = torch.where(eligible & row_has_seed, query_scores - row_max, torch.zeros_like(query_scores))
    relative = torch.exp(delta) * eligible.to(query_scores.dtype)
    if tau == 0.0:
        return eligible.to(query_scores.dtype).expand_as(query_scores)
    return _s5((relative - tau) / tau) * eligible.to(query_scores.dtype)


def max_product_union(seed_permission: Tensor, member_permission: Tensor) -> Tensor:
    """Combine query-selected seed memberships without duplicate accumulation.

    ``seed_permission`` is ``[B,Hd,Q,S]`` and ``member_permission`` is
    ``[B,S,N]``. Preparation-head reduction, when needed, is performed by the
    caller as an exact max over that independent axis.
    """
    if seed_permission.ndim != 4 or member_permission.ndim != 3:
        raise ValueError("Expected seed [B,Hd,Q,S] and member [B,S,N] permissions.")
    if seed_permission.shape[0] != member_permission.shape[0] or seed_permission.shape[-1] != member_permission.shape[1]:
        raise ValueError("Seed/member batch and group axes must match.")
    if seed_permission.shape[-1] == 0:
        return seed_permission.new_zeros(
            seed_permission.shape[0], seed_permission.shape[1], seed_permission.shape[2], member_permission.shape[-1]
        )
    candidate = seed_permission.unsqueeze(-1) * member_permission[:, None, None, :, :]
    return candidate.amax(dim=-2)


def _tiled_max_product_union(
    seed_permission: Tensor,
    member_permission: Tensor,
    *,
    max_intermediate_elements: int,
) -> Tensor:
    batch, heads, queries, seeds = seed_permission.shape
    if member_permission.shape[:2] != (batch, seeds):
        raise ValueError("Seed/member batch and group axes must match.")
    if queries == 0 or seeds == 0:
        return seed_permission.new_zeros(batch, heads, queries, member_permission.shape[-1])
    elements_per_query = max(1, batch * heads * seeds * member_permission.shape[-1])
    chunk_size = max(1, int(max_intermediate_elements) // elements_per_query)
    pieces = []
    for start in range(0, queries, chunk_size):
        stop = min(queries, start + chunk_size)
        pieces.append(max_product_union(seed_permission[:, :, start:stop], member_permission))
    return torch.cat(pieces, dim=2)


def build_query_retrieval_plan(
    query_scores: Tensor,
    hypergraph: NativeRetrievalHypergraph,
    *,
    tau_q: float | None = None,
    tau_query: float | None = None,
    query_mask: Tensor | None = None,
    query_ids: Tensor | None = None,
    query_seed_ids: Tensor | None = None,
    protected_module: Tensor | None = None,
    query_batch=None,
    token_store=None,
    include_environment: bool = False,
    environment_value_consumer: bool = False,
    max_intermediate_elements: int = 4_000_000,
) -> NativeQueryRetrievalPlan:
    """Select native members by live max-product permissions.

    The decoder's existing raw query/module scores are the only query selector.
    Environment membership remains available in the graph, while the potentially
    large query/E union is materialized only for a caller that opts into it.
    """
    if max_intermediate_elements < 1:
        raise ValueError("max_intermediate_elements must be positive.")
    if tau_q is None and tau_query is None:
        raise ValueError("Supply the sealed query threshold as tau_q.")
    if tau_q is not None and tau_query is not None and float(tau_q) != float(tau_query):
        raise ValueError("tau_q and tau_query aliases disagree.")
    tau_query = _threshold(tau_q if tau_q is not None else tau_query, "tau_q")
    batch, _decoder_heads, queries, seeds = query_scores.shape
    hypergraph.assert_current()
    if seeds != hypergraph.seed_ids.shape[1] or batch != hypergraph.seed_ids.shape[0]:
        raise ValueError("query score batch/seed axes must match the native graph.")
    if query_batch is not None:
        if query_batch.coordinates.ndim != 3 or query_batch.coordinates.shape[:2] != (batch, queries):
            raise ValueError("query_batch coordinates must match the score [B,Q] axes.")
        if query_batch.query_mask is None:
            if query_mask is not None:
                raise ValueError("A query-bound plan must take its mask from query_batch or use its all-open default.")
        elif query_mask is not None and query_mask is not query_batch.query_mask:
            raise ValueError("A query-bound plan must use the exact query_batch.query_mask tensor.")
        if query_batch.query_ids is None:
            if query_ids is not None:
                raise ValueError("A query-bound plan must take IDs from query_batch.")
        elif query_ids is not None and query_ids is not query_batch.query_ids:
            raise ValueError("A query-bound plan must use the exact query_batch.query_ids tensor.")
        query_mask = query_batch.query_mask
        query_ids = query_batch.query_ids
    if query_seed_ids is not None and (
        query_seed_ids.shape != hypergraph.seed_ids.shape or not torch.equal(query_seed_ids, hypergraph.seed_ids)
    ):
        raise ValueError("Query score module axes must use the graph's native seed ordering.")
    if query_mask is None:
        query_mask = torch.ones(batch, queries, dtype=torch.bool, device=query_scores.device)
    elif query_mask.shape != (batch, queries) or query_mask.dtype != torch.bool:
        raise ValueError("query_mask must be boolean [B,Q].")
    if query_ids is not None and query_ids.shape != (batch, queries):
        raise ValueError("query_ids must have shape [B,Q].")

    seed_permission = query_seed_permissions(
        query_scores, hypergraph.seed_valid, query_mask, tau_query
    )
    # Preparation heads are a separate axis and are conservatively unioned by
    # max. Their per-head views remain available in the graph and snapshots.
    module_members = hypergraph.module_member_permissions.amax(dim=1)
    module_permission = _tiled_max_product_union(
        seed_permission,
        module_members,
        max_intermediate_elements=max_intermediate_elements,
    )
    module_valid = hypergraph.module_valid
    eligible_module_rows = query_mask[:, :, None] & module_valid[:, None, :]
    module_permission = module_permission * eligible_module_rows[:, None, :, :].to(module_permission.dtype)
    if protected_module is not None:
        if protected_module.shape != (batch, queries, module_valid.shape[-1]):
            raise ValueError("protected_module must have shape [B,Q,M].")
        if protected_module.dtype != torch.bool and not protected_module.is_floating_point():
            raise TypeError("protected_module must be boolean or floating [0,1] permissions.")
        if protected_module.dtype == torch.bool:
            protected_module = protected_module.to(module_permission.dtype)
        valid_protection = (
            torch.isfinite(protected_module) & (protected_module >= 0) & (protected_module <= 1)
        ).all()
        if protected_module.device.type == "cuda":
            torch._assert_async(valid_protection, "protected_module permissions must lie in [0,1].")
        elif not bool(valid_protection):
            raise ValueError("protected_module permissions must be finite and lie in [0,1].")
        module_permission = torch.maximum(module_permission, protected_module[:, None, :, :])
    module_support = (module_permission.amax(dim=1) > 0) & eligible_module_rows
    unique_module_indices = torch.nonzero(module_support, as_tuple=False)

    environment_permission = environment_support = unique_environment_indices = None
    environment_ids = eligible_environment_rows = None
    retrieved_environment_pairs = None
    if include_environment:
        environment_members = hypergraph.environment_member_permissions.amax(dim=1)
        environment_permission = _tiled_max_product_union(
            seed_permission,
            environment_members,
            max_intermediate_elements=max_intermediate_elements,
        )
        environment_valid = hypergraph.environment_valid
        eligible_environment_rows = query_mask[:, :, None] & environment_valid[:, None, :]
        environment_permission = environment_permission * eligible_environment_rows[:, None, :, :].to(
            environment_permission.dtype
        )
        environment_support = (environment_permission.amax(dim=1) > 0) & eligible_environment_rows
        unique_environment_indices = torch.nonzero(environment_support, as_tuple=False)
        environment_ids = hypergraph.environment_ids
        retrieved_environment_pairs = environment_support.sum()
    elif environment_value_consumer:
        raise ValueError("environment_value_consumer requires include_environment=True.")

    mm_snapshot = hypergraph.mm_snapshot
    graph_binding = (
        mm_snapshot.owner_token,
        mm_snapshot.parameter_versions,
        mm_snapshot.buffer_versions,
        mm_snapshot.scene_identity,
        mm_snapshot.scene_tensor_versions,
        mm_snapshot.model_config_key,
    )
    if token_store is not None:
        if token_store.modules.native_ids is not hypergraph.module_ids:
            raise ValueError("Native token store module IDs do not match the captured graph records.")
        if token_store.environments.native_ids is not hypergraph.environment_ids:
            raise ValueError("Native token store environment IDs do not match the captured graph records.")
        owner, parameter_versions, buffer_versions, scene_identity, scene_tensor_versions, model_config_key = (
            graph_binding
        )
        token_store.assert_current(
            model_owner=owner,
            model_parameter_versions=parameter_versions,
            model_buffer_versions=buffer_versions,
            scene_tensor_versions=scene_tensor_versions,
            scene_identity=scene_identity,
            model_config_key=model_config_key,
        )

    return NativeQueryRetrievalPlan(
        seed_permission=seed_permission,
        module_permission=module_permission,
        module_support=module_support,
        unique_module_indices=unique_module_indices,
        module_ids=hypergraph.module_ids,
        seed_ids=hypergraph.seed_ids,
        query_ids=query_ids,
        query_mask=query_mask,
        eligible_module_rows=eligible_module_rows,
        environment_permission=environment_permission,
        environment_support=environment_support,
        unique_environment_indices=unique_environment_indices,
        environment_ids=environment_ids,
        eligible_environment_rows=eligible_environment_rows,
        include_environment=include_environment,
        environment_value_consumer=environment_value_consumer,
        retrieved_module_pairs=module_support.sum(),
        retrieved_environment_pairs=retrieved_environment_pairs,
        query_tensor_versions=(query_batch_tensor_versions(query_batch) if query_batch is not None else None),
        query_identity=(query_batch_identity(query_batch) if query_batch is not None else None),
        model_owner=(mm_snapshot.owner_token if query_batch is not None or token_store is not None else None),
        graph_binding=graph_binding,
        _hypergraph=hypergraph,
        _token_store=token_store,
    )


__all__ = [
    "build_query_retrieval_plan",
    "max_product_union",
    "query_seed_permissions",
]
