"""Opt-in query-factor reuse for separately applicable affine controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .contracts import CoreOutput, PreparedOrganization, QueryBatch
from .decoding import LocalPairPack


def _versions(values) -> tuple[tuple[int, int, int], ...]:
    return tuple((id(x), x.data_ptr(), x._version) for x in values if x is not None)


def _query_versions(query: QueryBatch):
    return _versions((query.coordinates, query.features, query.query_mask, query.query_ids,
                      query.local_batch_indices, query.local_query_indices,
                      query.local_module_indices, query.local_weights,
                      query.anchor_indices, query.anchor_weights))


@dataclass(frozen=True)
class PreparedAffineQuery:
    """Retained spatial factors; valid only for the bound scene/model/query.

    A-H6 stores Phi and selected local coefficients, without dense source K.
    A-D necessarily retains its direct source coefficients. Preparation keeps
    its autograd graph live; inference callers may prepare under no_grad.
    """

    organization: PreparedOrganization
    query: QueryBatch
    basis: Tensor | None
    direct_coefficients: Tensor | None
    local_pairs: LocalPairPack
    local_coefficients: Tensor
    nonlinear_field: Tensor
    query_versions: tuple
    factor_versions: tuple
    query_identity: tuple[Any, ...]
    edge_gates: Tensor | None
    uniform_access: bool


def _factor_values(prepared: PreparedAffineQuery):
    pack = prepared.local_pairs
    return (prepared.organization.source_coefficients,
            prepared.basis, prepared.direct_coefficients, prepared.local_coefficients,
            prepared.nonlinear_field, pack.batch_index, pack.query_index, pack.module_index,
            pack.support_weight, prepared.edge_gates)


def prepare_affine_query(core, organization: PreparedOrganization, query: QueryBatch, *,
                         edge_gates: Tensor | None = None, uniform_access: bool = False) -> PreparedAffineQuery:
    """Read spatial factors once, before any separately applicable heating."""
    if not core.thermal_response_rank:
        raise ValueError("This model has no affine control capability.")
    core.validate_prepared_state(organization, query)
    core._validate_edge_intervention(organization, edge_gates, uniform_access)
    field, _, _, basis, direct, coefficients, pairs, _ = core._decode_features(
        organization, query, return_trace=False, edge_gates=edge_gates, uniform_access=uniform_access)
    prepared = PreparedAffineQuery(
        organization, query, basis, direct, pairs, coefficients, field,
        _query_versions(query), (),
        (query.scene_key, query.coordinate_frame_key, query.receiver_key), edge_gates, uniform_access)
    return PreparedAffineQuery(**{**vars(prepared), "factor_versions": _versions(_factor_values(prepared))})


def validate_affine_query(core, prepared: PreparedAffineQuery) -> None:
    if not isinstance(prepared, PreparedAffineQuery):
        raise TypeError("Expected a prepared affine query.")
    core.validate_prepared_state(prepared.organization, prepared.query)
    if _query_versions(prepared.query) != prepared.query_versions:
        raise ValueError("Receiver/support tensors changed after query-factor preparation.")
    identity = (prepared.query.scene_key, prepared.query.coordinate_frame_key, prepared.query.receiver_key)
    if identity != prepared.query_identity or _versions(_factor_values(prepared)) != prepared.factor_versions:
        raise ValueError("Prepared query identity, factors or access policy changed; rebuild.")


def apply_prepared_affine(core, prepared: PreparedAffineQuery, controls: Tensor, *,
                          accumulation_dtype: torch.dtype | None = None) -> CoreOutput:
    """Contract retained factors with new controls, with zero neural reads."""
    validate_affine_query(core, prepared)
    organization, query = prepared.organization, prepared.query
    if controls.shape != (query.batch_size, organization.scene.num_modules):
        raise ValueError("Affine controls must match the prepared physical source axis.")
    if controls.device != query.coordinates.device:
        raise ValueError("Affine controls must be on the prepared query device.")
    dtype = core._accumulation_dtype(accumulation_dtype, controls.dtype)
    if prepared.basis is not None:
        temperature = core._factorized_apply(organization, query, prepared.basis,
                                             prepared.local_pairs, prepared.local_coefficients,
                                             controls, accumulation_dtype=dtype)
    else:
        temperature = torch.einsum("bqm,bm->bq", prepared.direct_coefficients.to(dtype), controls.to(dtype))
        temperature = temperature + core._local_apply(prepared.local_pairs, prepared.local_coefficients,
                                                      controls, query.batch_size, query.num_queries, dtype=dtype)
    if query.query_mask is not None:
        temperature = temperature * query.query_mask.to(dtype)
    return CoreOutput(
        nonlinear_field=prepared.nonlinear_field, thermal_temperature=temperature,
        work_counts={"receiver_rows": query.batch_size * query.num_queries, "edge_rows": 0,
                     "direct_source_rows": 0, "local_rows": 0,
                     "local_contraction_rows": prepared.local_pairs.count,
                     "spatial_factor_entries": 0 if prepared.basis is None else prepared.basis.numel()},
    )


def heat_vjp_prepared_affine(core, prepared: PreparedAffineQuery, cotangent: Tensor, *,
                             accumulation_dtype: torch.dtype | None = None) -> Tensor:
    """Transpose contraction of the retained operator without dense K."""
    validate_affine_query(core, prepared)
    organization, query = prepared.organization, prepared.query
    if cotangent.shape != (query.batch_size, query.num_queries) or cotangent.device != query.coordinates.device:
        raise ValueError("Affine cotangent must match the prepared receiver axis and device.")
    dtype = core._accumulation_dtype(accumulation_dtype, cotangent.dtype)
    weights = cotangent.to(dtype)
    if query.query_mask is not None:
        weights = weights * query.query_mask.to(dtype)
    if prepared.basis is not None:
        modes = torch.einsum("bq,bqhr->bhr", weights, prepared.basis.to(dtype))
        gradient = torch.einsum("bhr,bhrm->bm", modes, organization.source_coefficients.to(dtype))
    else:
        gradient = torch.einsum("bq,bqm->bm", weights, prepared.direct_coefficients.to(dtype))
    pack = prepared.local_pairs
    if pack.count:
        b, q, m = pack.batch_index, pack.query_index, pack.module_index
        values = weights[b, q] * pack.support_weight.to(dtype) * prepared.local_coefficients.to(dtype)
        gradient = gradient.reshape(-1).index_add(0, b * organization.scene.num_modules + m, values).reshape(
            query.batch_size, organization.scene.num_modules)
    return gradient
