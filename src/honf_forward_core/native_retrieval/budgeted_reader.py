"""Round8 hard native-member references and grouped execution.

These paths retain the original A2-D per-query/per-source value function. The
dense backend is a training reference with a hard forward and a deliberately
biased router gradient; the two inference references consume binary support.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..content_hypergraph.contracts import PreparedOrganization, QueryBatch
from ..content_hypergraph.decoding import _query_mask
from ..content_hypergraph.readers_v2 import DirectSourceReaderV2
from ..content_hypergraph.source_read_response import SourceReadAffineFactorsV1
from ..content_hypergraph.spatial_features_v2 import query_source_features
from .reader import NativeModuleScores, NativeSelectedRead, NativeSelectedReader


def dense_hard_router_gate(
    protected: Tensor,
    dispatch_st: Tensor,
    membership_st: Tensor,
) -> Tensor:
    """Build the dense hard-forward router estimator from protection and ST routes.

    ``dispatch_st`` is ``[B,Q,S]`` and ``membership_st`` is ``[B,S,M]``. Their
    forwards must be one-hot and binary respectively. The straight-through
    gradient is intentionally a training estimator, not a derivative of the
    discrete inference route.
    """
    if protected.ndim != 3 or protected.dtype != torch.bool:
        raise ValueError("protected must be boolean [B,Q,M].")
    if dispatch_st.ndim != 3 or membership_st.ndim != 3:
        raise ValueError("dispatch_st and membership_st must have shapes [B,Q,S] and [B,S,M].")
    batch, queries, seeds = dispatch_st.shape
    if tuple(membership_st.shape[:2]) != (batch, seeds):
        raise ValueError("dispatch and membership batch/seed axes do not match.")
    if tuple(protected.shape) != (batch, queries, membership_st.shape[-1]):
        raise ValueError("protected batch/query/module axes do not match the route tensors.")
    if dispatch_st.device != membership_st.device or protected.device != dispatch_st.device:
        raise ValueError("All router gate tensors must share one device.")
    dispatched_members = torch.matmul(dispatch_st, membership_st)
    protected_float = protected.to(dtype=dispatched_members.dtype)
    return protected_float + (1.0 - protected_float) * dispatched_members


def _raw_scores(
    reader: DirectSourceReaderV2,
    organization: PreparedOrganization,
    query: QueryBatch,
    query_states: Tensor,
    scores: NativeModuleScores | None,
) -> NativeModuleScores:
    if scores is None:
        scores = NativeSelectedReader.score_queries(reader, organization, query, query_states)
    expected = (
        query_states.shape[0], reader.heads, query_states.shape[1], organization.scene.num_modules
    )
    if tuple(scores.raw_logits.shape) != expected:
        raise ValueError(f"raw native module scores must have shape {expected}.")
    return scores


def _valid_rows(
    organization: PreparedOrganization,
    query: QueryBatch,
    query_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    scene = organization.scene
    module_measure = scene.module_measure.to(query_states.dtype)
    module_present = scene.module_mask
    query_present = _query_mask(query)
    info_valid = module_present & (module_measure > 0)
    pair_valid = query_present[:, :, None] & module_present[:, None, :]
    return module_measure, info_valid, pair_valid


def _protection_exception_count(
    support: Tensor,
    organization: PreparedOrganization,
    query: QueryBatch,
    graph: Any | None,
    dispatch_plan: Any | None,
    protected_mask: Tensor | None,
) -> Tensor | None:
    """Count selected protected rows outside the dispatched seed membership.

    ``None`` means route inputs were not supplied, so callers must expose the
    numeric field as unavailable rather than treating it as a measured zero.
    """
    membership = getattr(graph, "membership_hard", None)
    dispatch = getattr(dispatch_plan, "dispatch_seed", None)
    if (not isinstance(membership, Tensor) or not isinstance(dispatch, Tensor)
            or not isinstance(protected_mask, Tensor)):
        return None
    batch, queries, modules = support.shape
    if (membership.ndim != 3 or membership.shape[0] != batch
            or membership.shape[2] != modules or membership.dtype != torch.bool
            or dispatch.shape != (batch, queries) or dispatch.dtype != torch.long
            or protected_mask.shape != support.shape or protected_mask.dtype != torch.bool):
        raise ValueError("Protection-exception telemetry inputs do not match the native request.")
    if membership.device != support.device or dispatch.device != support.device or protected_mask.device != support.device:
        raise ValueError("Protection-exception telemetry inputs must share the native request device.")
    seeds = membership.shape[1]
    dispatch_valid = (dispatch >= 0) & (dispatch < seeds)
    safe_dispatch = dispatch.clamp(0, max(0, seeds - 1))
    chosen_members = membership.gather(
        1, safe_dispatch[..., None].expand(batch, queries, modules)
    )
    seed_valid = getattr(graph, "seed_valid", None)
    if isinstance(seed_valid, Tensor):
        if seed_valid.shape != (batch, seeds) or seed_valid.dtype != torch.bool:
            raise ValueError("graph.seed_valid does not match the native request.")
        dispatch_valid = dispatch_valid & seed_valid.gather(1, safe_dispatch)
    query_valid = _query_mask(query)
    valid = dispatch_valid[:, :, None] & query_valid[:, :, None]
    chosen_members = chosen_members & valid & organization.scene.module_mask[:, None, :]
    return (support & protected_mask & ~chosen_members).sum()


def _multiplicative_alpha(
    raw_logits: Tensor,
    gate: Tensor,
    module_measure: Tensor,
    info_valid: Tensor,
    query_valid: Tensor,
    measure_reference: Tensor,
) -> Tensor:
    """Native information normalization with a derivative at zero gates."""
    batch = raw_logits.shape[0]
    score_valid = info_valid[:, None, None, :] & query_valid[:, None, :, None]
    masked_scores = torch.where(score_valid, raw_logits, torch.full_like(raw_logits, -torch.inf))
    score_max = masked_scores.amax(dim=-1, keepdim=True)
    score_max = torch.where(torch.isfinite(score_max), score_max, torch.zeros_like(score_max))
    exponent = torch.exp(torch.where(score_valid, raw_logits - score_max, torch.full_like(raw_logits, -torch.inf)))
    reference = measure_reference.to(dtype=raw_logits.dtype, device=raw_logits.device).reshape(batch, 1, 1, 1)
    measure = torch.where(info_valid, module_measure, torch.zeros_like(module_measure))[:, None, None, :]
    info_weight = exponent * (measure / reference.clamp_min(torch.finfo(raw_logits.dtype).tiny))
    info_weight = info_weight * score_valid.to(raw_logits.dtype)
    gate_head_major = gate[:, None, :, :]
    weighted = info_weight * gate_head_major
    denominator = weighted.sum(dim=-1, keepdim=True)
    normalized = weighted / denominator.clamp_min(torch.finfo(raw_logits.dtype).tiny)
    normalized = torch.where(denominator > 0, normalized, torch.zeros_like(normalized))
    return normalized.permute(0, 2, 3, 1)


def _full_native_values(
    reader: DirectSourceReaderV2,
    organization: PreparedOrganization,
    query: QueryBatch,
    query_states: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate original A2-D nonlinear hidden/messages on every rectangular row."""
    scene = organization.scene
    query_valid = _query_mask(query)
    safe_query_coordinates = torch.where(
        query_valid[..., None], query.coordinates.to(query_states.dtype),
        torch.zeros_like(query.coordinates, dtype=query_states.dtype),
    )
    safe_module_coordinates = torch.where(
        scene.module_mask[..., None], scene.module_coordinates.to(query_states.dtype),
        torch.zeros_like(scene.module_coordinates, dtype=query_states.dtype),
    )
    safe_module_lengths = torch.where(
        scene.module_mask[..., None], scene.module_characteristic_lengths.to(query_states.dtype),
        torch.ones_like(scene.module_characteristic_lengths, dtype=query_states.dtype),
    )
    geometry, relative_gamma = query_source_features(
        safe_query_coordinates,
        safe_module_coordinates,
        organization.frame_scale,
        safe_module_lengths,
    )
    if organization.direct_source_values is None:
        if scene.num_modules and query_states.shape[1]:
            raise ValueError("Prepared A2-D native module values are missing.")
        prepared_values = query_states.new_zeros((query_states.shape[0], scene.num_modules, reader.width))
    else:
        prepared_values = organization.direct_source_values
        prepared_values = torch.where(
            scene.module_mask[..., None], prepared_values, torch.zeros_like(prepared_values)
        )
    relative = reader.relative_value(relative_gamma)
    if reader.source_scale_projection is not None:
        relative = relative + reader.source_scale_projection(geometry[..., -reader.coordinate_dim :])
    safe_query_states = torch.where(
        query_valid[..., None], query_states, torch.zeros_like(query_states)
    )
    hidden = F.silu(
        reader.query_value(safe_query_states)[:, :, None, :]
        + prepared_values[:, None, :, :]
        + relative
    )
    # Preserve the original projection and bias placement exactly.
    messages = reader.message_output(hidden)
    return hidden, messages


def _packed_native_values(
    reader: DirectSourceReaderV2,
    organization: PreparedOrganization,
    query_coordinates: Tensor,
    query_projection: Tensor,
    pair_indices: Tensor,
) -> Tensor:
    """Gather exact nonlinear A2-D messages for ordered ``(b,q,m)`` rows."""
    scene = organization.scene
    if pair_indices.numel() == 0:
        hidden = query_projection.new_empty((0, reader.width))
        return reader.message_output(hidden)
    b, q, m = pair_indices.unbind(dim=1)
    delta = query_coordinates[b, q].to(query_projection.dtype) - scene.module_coordinates[b, m].to(
        query_projection.dtype
    )
    normalized = delta / organization.frame_scale[b]
    pieces = [normalized]
    for frequency in (1.0, 2.0, 4.0):
        phase = (2.0 * torch.pi * frequency) * normalized
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    gamma = torch.cat(pieces, dim=-1)
    relative = reader.relative_value(gamma)
    if reader.source_scale_projection is not None:
        scale_feature = torch.asinh(
            delta / scene.module_characteristic_lengths[b, m].to(query_projection.dtype)
        )
        relative = relative + reader.source_scale_projection(scale_feature)
    query_value = query_projection[b, q]
    source_value = organization.direct_source_values[b, m]
    hidden = F.silu(query_value + source_value + relative)
    return reader.message_output(hidden)


def _response_from_rows(
    reader: DirectSourceReaderV2,
    organization: PreparedOrganization,
    query: QueryBatch,
    query_states: Tensor,
    pair_indices: Tensor,
    messages: Tensor,
    gate_rows: Tensor,
) -> SourceReadAffineFactorsV1 | None:
    if reader.thermal_output is None:
        return None
    batch, queries, _ = query_states.shape
    scene = organization.scene
    if pair_indices.shape[0]:
        b, q, m = pair_indices.unbind(dim=1)
        # A2-D's scalar law is unnormalized and signed. Module measure applies
        # only in the information denominator, never a second time here.
        raw = reader.thermal_output(messages).squeeze(-1)
        coefficients = raw * gate_rows.to(raw.dtype)
        coefficients = coefficients * _query_mask(query)[b, q].to(raw.dtype)
        coefficients = coefficients * scene.module_mask[b, m].to(raw.dtype)
    else:
        b = q = m = pair_indices.new_empty((0,))
        coefficients = query_states.new_empty((0,))
    return SourceReadAffineFactorsV1(
        batch_size=batch,
        query_count=queries,
        source_count=scene.num_modules,
        selected_batch=b,
        selected_query=q,
        selected_source=m,
        residual_coefficients=coefficients,
        present_mask=scene.module_mask,
        heating_envelope=query_states.new_tensor(torch.finfo(query_states.dtype).max),
        scalar_bias=0.0,
    )


def _masked_dense_messages(messages: Tensor, support: Tensor) -> Tensor:
    return messages * support[..., None].to(messages.dtype)


def _empty_read(
    reader: DirectSourceReaderV2,
    organization: PreparedOrganization,
    query_states: Tensor,
    backend: str,
    *,
    return_messages: bool = False,
) -> NativeSelectedRead:
    batch, queries, _ = query_states.shape
    modules = organization.scene.num_modules
    alpha = query_states.new_zeros((batch, queries, modules, reader.heads))
    response = None
    if reader.thermal_output is not None:
        empty = torch.empty(0, dtype=torch.long, device=query_states.device)
        response = SourceReadAffineFactorsV1(
            batch_size=batch,
            query_count=queries,
            source_count=modules,
            selected_batch=empty,
            selected_query=empty,
            selected_source=empty,
            residual_coefficients=query_states.new_empty((0,)),
            present_mask=organization.scene.module_mask,
            heating_envelope=query_states.new_tensor(torch.finfo(query_states.dtype).max),
        )
    counts = {
        "native_useful_value_rows": 0,
        "native_value_rows": 0,
        "native_padded_rows": 0,
        "native_dense_value_rows": 0,
        "native_protection_exception_rows": 0,
        "native_protection_exception_rows_available": 1,
        "native_dispatch_rows": 0,
        "native_dispatch_groups": 0,
        "native_group_launches": 0,
        "native_request_syncs": 0,
    }
    trace: dict[str, Tensor] = {
        "native_hard_support": torch.zeros((batch, queries, modules), dtype=torch.bool, device=query_states.device),
    }
    return NativeSelectedRead(
        query_states.new_zeros((batch, queries, reader.width)),
        alpha,
        response,
        query_states.new_zeros((batch, queries, modules, reader.width)) if return_messages else None,
        counts,
        trace,
        backend,
    )


class BudgetedNativeReader:
    """W2/W3 native member readers with one common binary scientific function."""

    @staticmethod
    def evaluate(
        reader: DirectSourceReaderV2,
        organization: PreparedOrganization,
        query: QueryBatch,
        query_states: Tensor,
        gate: Tensor,
        *,
        backend: str = "dense_hard_reference",
        scores: NativeModuleScores | None = None,
        graph: Any | None = None,
        dispatch_plan: Any | None = None,
        protected_mask: Tensor | None = None,
        return_messages: bool = False,
        measure_reference: Tensor | None = None,
        query_tile_size: int = 32,
    ) -> NativeSelectedRead:
        """Evaluate one hard-support native function.

        Dense mode accepts a straight-through gate whose forward is binary;
        flat/grouped modes require boolean binary support. Grouped mode also
        requires the native seed graph and its one-seed query dispatch.
        """
        if backend not in {"dense_hard_reference", "flat_selected_reference", "grouped_native"}:
            raise ValueError("Unsupported Round8 native reader backend.")
        if backend == "grouped_native":
            return BudgetedNativeReader._grouped_native(
                reader, organization, query, query_states, gate, graph=graph,
                dispatch_plan=dispatch_plan, protected_mask=protected_mask,
                scores=scores, return_messages=return_messages,
                measure_reference=measure_reference, query_tile_size=query_tile_size,
            )
        return BudgetedNativeReader._dense_or_flat(
            reader, organization, query, query_states, gate, backend=backend,
            scores=scores, graph=graph, dispatch_plan=dispatch_plan,
            protected_mask=protected_mask, return_messages=return_messages,
            measure_reference=measure_reference,
        )

    @staticmethod
    def _dense_or_flat(
        reader: DirectSourceReaderV2,
        organization: PreparedOrganization,
        query: QueryBatch,
        query_states: Tensor,
        gate: Tensor,
        *,
        backend: str,
        scores: NativeModuleScores | None,
        graph: Any | None,
        dispatch_plan: Any | None,
        protected_mask: Tensor | None,
        return_messages: bool,
        measure_reference: Tensor | None,
    ) -> NativeSelectedRead:
        scene = organization.scene
        batch, queries, _ = query_states.shape
        modules = scene.num_modules
        expected = (batch, queries, modules)
        if tuple(gate.shape) != expected:
            raise ValueError(f"gate must have shape {expected}.")
        if gate.device != query_states.device:
            raise ValueError("gate and query states must share one device.")
        if backend == "flat_selected_reference" and gate.dtype != torch.bool:
            raise TypeError("flat_selected_reference requires boolean binary support.")
        if backend == "dense_hard_reference" and gate.dtype == torch.bool:
            gate = gate.to(query_states.dtype)
        elif backend == "dense_hard_reference" and not gate.is_floating_point():
            raise TypeError("dense_hard_reference requires a floating hard gate or straight-through gate.")
        if backend == "dense_hard_reference":
            hard_gate = gate.detach()
            binary_gate = torch.isfinite(hard_gate).all() & ((hard_gate == 0) | (hard_gate == 1)).all()
            if binary_gate.is_cuda:
                torch._assert_async(binary_gate, "dense_hard_reference requires a binary hard forward.")
            elif not bool(binary_gate):
                raise ValueError("dense_hard_reference requires a binary hard forward.")
        if queries == 0 or modules == 0:
            return _empty_read(
                reader, organization, query_states, backend, return_messages=return_messages
            )
        prepared = _raw_scores(reader, organization, query, query_states, scores)
        if measure_reference is None:
            measure_reference = scene.module_measure_reference
        if measure_reference is None:
            measure_reference = query_states.new_ones((batch,))
        measure, info_valid, pair_valid = _valid_rows(organization, query, query_states)
        query_valid = _query_mask(query)
        support = (gate.detach() > 0) & pair_valid
        exception_tensor = _protection_exception_count(
            support, organization, query, graph, dispatch_plan, protected_mask
        )
        gate_live = gate.to(query_states.dtype) * pair_valid.to(query_states.dtype)
        alpha = _multiplicative_alpha(
            prepared.raw_logits, gate_live, measure, info_valid, query_valid, measure_reference
        )

        if backend == "dense_hard_reference":
            _, dense_messages = _full_native_values(reader, organization, query, query_states)
            effective_messages = dense_messages * pair_valid[..., None].to(dense_messages.dtype)
            blocks = effective_messages.reshape(batch, queries, modules, reader.heads, reader.head_width)
            context = (blocks * alpha[..., None]).sum(dim=2).reshape(batch, queries, reader.width)
            # The reference physically evaluates all valid native message rows.
            # Its Thermal factors keep zero-gate coefficients connected to the
            # router estimator, so omitted values can return during learning.
            present_pairs = (
                torch.nonzero(pair_valid, as_tuple=False)
                if reader.thermal_output is not None
                else torch.empty((0, 3), dtype=torch.long, device=query_states.device)
            )
            present_messages = dense_messages[
                present_pairs[:, 0], present_pairs[:, 1], present_pairs[:, 2]
            ] if present_pairs.numel() else dense_messages.new_empty((0, reader.width))
            gate_rows = gate_live[
                present_pairs[:, 0], present_pairs[:, 1], present_pairs[:, 2]
            ] if present_pairs.numel() else gate_live.new_empty((0,))
            response = _response_from_rows(
                reader, organization, query, query_states, present_pairs, present_messages, gate_rows
            )
            messages = _masked_dense_messages(dense_messages, support) if return_messages else None
            row_stats = [support.sum(), pair_valid.sum()]
            if exception_tensor is not None:
                row_stats.append(exception_tensor)
            values = torch.stack(row_stats).to(dtype=torch.long).detach().cpu().tolist()
            support_count, valid_value_rows = map(int, values[:2])
            protection_exceptions = int(values[2]) if exception_tensor is not None else 0
            # The dense reference evaluates every padded source/query slot before
            # masking; valid rows are reported separately from executed rows.
            value_rows = batch * queries * modules
            dense_rows = value_rows
            # Response compaction (when Thermal is enabled) and the integer
            # telemetry each require one request-level CUDA synchronization.
            request_syncs = (1 + int(reader.thermal_output is not None)) if query_states.is_cuda else 0
            selected_indices = present_pairs
        else:
            selected_indices = torch.nonzero(support, as_tuple=False)
            query_projection = reader.query_value(query_states)
            selected_messages = _packed_native_values(
                reader, organization, query.coordinates, query_projection, selected_indices
            )
            flat_dest = selected_indices[:, 0] * queries + selected_indices[:, 1]
            if selected_indices.shape[0]:
                b, q, m = selected_indices.unbind(1)
                row_alpha = alpha[b, q, m, :]
                blocks = selected_messages.reshape(-1, reader.heads, reader.head_width)
                reduced = selected_messages.new_zeros((batch * queries, reader.heads, reader.head_width))
                reduced = reduced.index_add(0, flat_dest, blocks * row_alpha[..., None])
                context = reduced.reshape(batch, queries, reader.width)
                gate_rows = gate_live[b, q, m]
            else:
                context = query_states.new_zeros((batch, queries, reader.width))
                gate_rows = gate_live.new_empty((0,))
            response = _response_from_rows(
                reader, organization, query, query_states, selected_indices, selected_messages, gate_rows
            )
            if return_messages:
                messages = query_states.new_zeros((batch, queries, modules, reader.width))
                if selected_indices.shape[0]:
                    b, q, m = selected_indices.unbind(1)
                    messages[b, q, m] = selected_messages
            else:
                messages = None
            support_count = int(selected_indices.shape[0])
            value_rows = support_count
            dense_rows = 0
            protection_exceptions = 0
            if exception_tensor is not None:
                protection_exceptions = int(exception_tensor.detach().cpu())
            request_syncs = int(query_states.is_cuda) * (
                1 + int(exception_tensor is not None)
            )

        counts = _base_counts(
            support_count=support_count,
            value_rows=value_rows,
            valid_rows=valid_value_rows if backend == "dense_hard_reference" else support_count,
            scalar_rows=(
                valid_value_rows if backend == "dense_hard_reference" and reader.thermal_output is not None
                else value_rows if reader.thermal_output is not None else 0
            ),
            padded_rows=0,
            dense_rows=dense_rows,
            protection_exceptions=protection_exceptions,
            protection_exceptions_available=exception_tensor is not None,
            dispatch_rows=queries * batch,
            dispatch_groups=0,
            group_launches=0,
            request_syncs=request_syncs,
            candidate_rows=batch * queries * modules,
            coordinate_dim=reader.coordinate_dim,
            thermal=reader.thermal_output is not None,
            hidden_width=reader.width,
        )
        trace: dict[str, Tensor] = {"native_hard_support": support, "native_module_ids": scene.module_ids}
        if backend == "flat_selected_reference":
            selected_batches, selected_queries, selected_sources = selected_indices.unbind(1) if selected_indices.shape[0] else (
                selected_indices.new_empty((0,)), selected_indices.new_empty((0,)), selected_indices.new_empty((0,))
            )
            trace.update(
                native_selected_batch=selected_batches,
                native_selected_query=selected_queries,
                native_selected_source=selected_sources,
                native_selected_source_ids=scene.module_ids[selected_batches, selected_sources],
            )
        return NativeSelectedRead(context, alpha, response, messages, counts, trace, backend)

    @staticmethod
    def _grouped_native(
        reader: DirectSourceReaderV2,
        organization: PreparedOrganization,
        query: QueryBatch,
        query_states: Tensor,
        gate: Tensor,
        *,
        graph: Any | None,
        dispatch_plan: Any | None,
        protected_mask: Tensor | None,
        scores: NativeModuleScores | None,
        return_messages: bool,
        measure_reference: Tensor | None,
        query_tile_size: int,
    ) -> NativeSelectedRead:
        if graph is None or dispatch_plan is None or protected_mask is None:
            raise ValueError("grouped_native requires graph, dispatch_plan, and protected_mask.")
        if query_tile_size < 1:
            raise ValueError("query_tile_size must be positive.")
        scene = organization.scene
        batch, queries, _ = query_states.shape
        modules = scene.num_modules
        dispatch = getattr(dispatch_plan, "dispatch_seed", None)
        member_ids = getattr(graph, "member_indices", None)
        member_valid = getattr(graph, "member_valid", None)
        seed_valid = getattr(graph, "seed_valid", None)
        if not isinstance(dispatch, Tensor) or dispatch.shape != (batch, queries):
            raise ValueError("dispatch_plan.dispatch_seed must have shape [B,Q].")
        if not isinstance(member_ids, Tensor) or not isinstance(member_valid, Tensor):
            raise TypeError("graph must expose padded member_indices and member_valid tensors.")
        if member_ids.ndim != 3 or member_valid.shape != member_ids.shape:
            raise ValueError("graph member indices and masks must have shape [B,S,K].")
        seeds = member_ids.shape[1]
        if member_ids.shape[0] != batch or member_ids.dtype != torch.long or member_valid.dtype != torch.bool:
            raise ValueError("graph member lists have incompatible batch, index, or mask types.")
        if protected_mask.shape != (batch, queries, modules) or protected_mask.dtype != torch.bool:
            raise ValueError("protected_mask must be boolean [B,Q,M].")
        if gate.shape != (batch, queries, modules) or gate.dtype != torch.bool:
            raise ValueError("grouped_native gate must be boolean [B,Q,M].")
        if dispatch.device != query_states.device or member_ids.device != query_states.device:
            raise ValueError("group, dispatch, query, and value tensors must share a device.")
        if queries == 0 or modules == 0 or seeds == 0:
            return _empty_read(
                reader, organization, query_states, "grouped_native", return_messages=return_messages
            )
        tie = getattr(dispatch_plan, "tie_indeterminate", None)
        if isinstance(tie, Tensor):
            tie_valid = ~tie.any()
            if tie_valid.is_cuda:
                torch._assert_async(tie_valid, "grouped_native cannot resolve an indeterminate geometric seed tie.")
            elif not bool(tie_valid):
                raise ValueError("grouped_native cannot resolve an indeterminate geometric seed tie.")
        query_valid = _query_mask(query)
        dispatch_valid = (dispatch >= 0) & (dispatch < seeds) & query_valid
        safe_dispatch = dispatch.clamp(0, seeds - 1)
        batch_index = torch.arange(batch, device=query_states.device)[:, None, None]
        membership_count = torch.zeros((batch, seeds, modules), dtype=torch.int32, device=query_states.device)
        if member_ids.shape[-1]:
            safe_member_ids = member_ids.clamp(0, max(0, modules - 1))
            valid_members = member_valid & (member_ids >= 0) & (member_ids < modules)
            membership_count.scatter_add_(2, safe_member_ids, valid_members.to(torch.int32))
        membership = (membership_count > 0) & scene.module_mask[:, None, :]
        if isinstance(seed_valid, Tensor):
            membership = membership & seed_valid[:, :, None]
            dispatch_seed_valid = seed_valid.gather(1, safe_dispatch)
            dispatch_valid = dispatch_valid & dispatch_seed_valid
        chosen_members = membership.gather(
            1, safe_dispatch[..., None].expand(batch, queries, modules)
        )
        support = (
            (chosen_members | protected_mask)
            & scene.module_mask[:, None, :]
            & query_valid[:, :, None]
            & dispatch_valid[:, :, None]
        )
        # The supplied gate is the public plan's deduped support; reject stale
        # or mismatched route material before doing any nonlinear reads.
        gate_match = (support == gate).all()
        if gate_match.is_cuda:
            torch._assert_async(gate_match, "grouped hard support does not match the graph/dispatch/protection inputs.")
        elif not bool(gate_match):
            raise ValueError("grouped hard support does not match the graph/dispatch/protection inputs.")

        if measure_reference is None:
            measure_reference = scene.module_measure_reference
        if measure_reference is None:
            measure_reference = query_states.new_ones((batch,))
        measure, info_valid, _ = _valid_rows(organization, query, query_states)
        prepared = _raw_scores(reader, organization, query, query_states, scores)
        alpha = _multiplicative_alpha(
            prepared.raw_logits, support.to(query_states.dtype), measure, info_valid,
            query_valid, measure_reference,
        )

        one_hot = F.one_hot(safe_dispatch, num_classes=seeds).to(torch.bool) & dispatch_valid[..., None]
        if isinstance(seed_valid, Tensor):
            one_hot = one_hot & seed_valid[:, None, :]
        query_counts = one_hot.sum(dim=1)
        # Union protection exceptions within each dispatched group. The pair
        # gate below still suppresses an exception for queries that do not own it.
        protected_counts = torch.zeros((batch, seeds, modules), dtype=torch.int32, device=query_states.device)
        scatter_seed = safe_dispatch[..., None].expand(batch, queries, modules)
        protected_counts.scatter_add_(
            1,
            scatter_seed,
            (
                protected_mask
                & scene.module_mask[:, None, :]
                & query_valid[:, :, None]
                & dispatch_valid[:, :, None]
            ).to(torch.int32),
        )
        group_sources = membership | (protected_counts > 0)
        source_counts = group_sources.sum(dim=-1)
        support_count_tensor = support.sum()
        dispatch_count_tensor = dispatch_valid.sum()
        group_count_tensor = ((query_counts > 0) & (source_counts > 0)).sum()
        max_query_tensor = query_counts.max()
        max_source_tensor = source_counts.max()
        protection_exception_tensor = (support & protected_mask & ~chosen_members).sum()
        stats = torch.stack((
            max_query_tensor, max_source_tensor, support_count_tensor,
            dispatch_count_tensor, group_count_tensor, protection_exception_tensor,
        )).to(dtype=torch.long)
        if stats.is_cuda:
            max_queries, max_sources, support_count, dispatch_count, group_count, protection_exceptions = (
                int(value) for value in stats.cpu().tolist()
            )
            request_syncs = 1
        else:
            max_queries, max_sources, support_count, dispatch_count, group_count, protection_exceptions = (
                int(value) for value in stats.tolist()
            )
            request_syncs = 0
        if max_queries == 0 or max_sources == 0:
            return _empty_read(
                reader, organization, query_states, "grouped_native", return_messages=return_messages
            )

        # Stable grouping by seed and query order; dynamic extents are discovered
        # in the single request-level transfer above.
        query_slot = one_hot.to(torch.long).cumsum(dim=1) - 1
        query_slot = query_slot.permute(0, 2, 1).clamp_min(0)
        query_values = torch.arange(queries, device=query_states.device)[None, None, :].expand(
            batch, seeds, queries
        ) + 1
        packed_values = torch.where(one_hot.permute(0, 2, 1), query_values, torch.zeros_like(query_values))
        packed_queries_plus = torch.zeros(
            (batch, seeds, max_queries), dtype=torch.long, device=query_states.device
        ).scatter_add(2, query_slot, packed_values)
        packed_queries = packed_queries_plus - 1
        packed_query_valid = packed_queries_plus > 0

        # Stable argsort preserves the native source axis order in each list.
        source_order = torch.argsort((~group_sources).to(torch.int8), dim=-1, stable=True)[..., :max_sources]
        packed_sources = source_order
        packed_source_valid = group_sources.gather(-1, packed_sources)
        source_coordinates = scene.module_coordinates.to(query_states.dtype)
        source_values = organization.direct_source_values
        if source_values is None:
            raise ValueError("Prepared A2-D native module values are missing.")
        query_projection = reader.query_value(query_states)
        context_groups = query_states.new_zeros((batch, seeds, max_queries, reader.heads, reader.head_width))
        scalar_grid = query_states.new_zeros((batch, queries, modules)) if reader.thermal_output is not None else None
        message_grid = query_states.new_zeros((batch, queries, modules, reader.width)) if return_messages else None
        value_rows = 0
        padded_rows = 0
        launches = 0
        frame_scale = organization.frame_scale

        for query_start in range(0, max_queries, query_tile_size):
            query_stop = min(max_queries, query_start + query_tile_size)
            qids = packed_queries[:, :, query_start:query_stop]
            qvalid = packed_query_valid[:, :, query_start:query_stop]
            tile_queries = query_stop - query_start
            q_coordinates = query.coordinates.to(query_states.dtype)[:, None, :, :].expand(
                batch, seeds, queries, query.coordinates.shape[-1]
            ).gather(2, qids.clamp_min(0)[..., None].expand(batch, seeds, tile_queries, query.coordinates.shape[-1]))
            q_values = query_projection[:, None, :, :].expand(
                batch, seeds, queries, reader.width
            ).gather(2, qids.clamp_min(0)[..., None].expand(batch, seeds, tile_queries, reader.width))
            q_coordinates = torch.where(qvalid[..., None], q_coordinates, torch.zeros_like(q_coordinates))
            q_values = torch.where(qvalid[..., None], q_values, torch.zeros_like(q_values))
            q_source_ids = packed_sources
            source_coords = source_coordinates[:, None, :, :].expand(
                batch, seeds, modules, source_coordinates.shape[-1]
            ).gather(2, q_source_ids[..., None].expand(batch, seeds, max_sources, source_coordinates.shape[-1]))
            source_value_rows = source_values[:, None, :, :].expand(
                batch, seeds, modules, reader.width
            ).gather(2, q_source_ids[..., None].expand(batch, seeds, max_sources, reader.width))
            source_value_rows = torch.where(
                packed_source_valid[..., None], source_value_rows, torch.zeros_like(source_value_rows)
            )
            source_lengths = scene.module_characteristic_lengths.to(query_states.dtype)[:, None, :, :].expand(
                batch, seeds, modules, reader.coordinate_dim
            ).gather(2, q_source_ids[..., None].expand(batch, seeds, max_sources, reader.coordinate_dim))
            source_lengths = torch.where(
                packed_source_valid[..., None], source_lengths, torch.ones_like(source_lengths)
            )
            source_coords = torch.where(
                packed_source_valid[..., None], source_coords, torch.zeros_like(source_coords)
            )
            delta = q_coordinates[:, :, :, None, :] - source_coords[:, :, None, :, :]
            normalized = delta / frame_scale[:, None, None, None, :]
            pieces = [normalized]
            for frequency in (1.0, 2.0, 4.0):
                phase = (2.0 * torch.pi * frequency) * normalized
                pieces.extend((torch.sin(phase), torch.cos(phase)))
            gamma = torch.cat(pieces, dim=-1)
            relative = reader.relative_value(gamma)
            if reader.source_scale_projection is not None:
                source_scale = torch.asinh(
                    delta / source_lengths[:, :, None, :, :]
                )
                relative = relative + reader.source_scale_projection(source_scale)
            hidden = F.silu(q_values[:, :, :, None, :] + source_value_rows[:, :, None, :, :] + relative)
            message_block = reader.message_output(hidden)
            source_gate = support[batch_index, qids.clamp_min(0)]
            source_gate = source_gate.gather(
                -1,
                q_source_ids[:, :, None, :].expand(batch, seeds, tile_queries, max_sources),
            )
            live = qvalid[..., None] & packed_source_valid[:, :, None, :] & source_gate
            message_block = message_block * live[..., None].to(message_block.dtype)
            row_alpha = alpha[batch_index, qids.clamp_min(0)].gather(
                3,
                q_source_ids[:, :, None, :, None].expand(batch, seeds, tile_queries, max_sources, reader.heads),
            )
            blocks = message_block.reshape(
                batch, seeds, tile_queries, max_sources, reader.heads, reader.head_width
            )
            context_groups[:, :, query_start:query_stop] = (blocks * row_alpha[..., None]).sum(dim=3)
            value_rows += batch * seeds * tile_queries * max_sources
            launches += 1

            if scalar_grid is not None:
                # Preserve original message_output then thermal_output placement.
                scalar = reader.thermal_output(message_block).squeeze(-1)
                bidx = torch.arange(batch, device=query_states.device)[:, None, None, None]
                qidx = qids.clamp_min(0)[..., None].expand(batch, seeds, tile_queries, max_sources)
                midx = q_source_ids[:, :, None, :].expand(batch, seeds, tile_queries, max_sources)
                linear = (bidx * queries + qidx) * modules + midx
                scalar_grid.view(-1).scatter_add_(
                    0, linear.reshape(-1), (scalar * live.to(scalar.dtype)).reshape(-1)
                )
            if message_grid is not None:
                bidx = torch.arange(batch, device=query_states.device)[:, None, None, None]
                qidx = qids.clamp_min(0)[..., None].expand(batch, seeds, tile_queries, max_sources)
                midx = q_source_ids[:, :, None, :].expand(batch, seeds, tile_queries, max_sources)
                linear = (bidx * queries + qidx) * modules + midx
                message_grid.view(-1, reader.width).index_add_(
                    0, linear.reshape(-1), (message_block * live[..., None]).reshape(-1, reader.width)
                )

        # Restore receiver order after grouped execution.
        flat_query_ids = packed_queries.clamp_min(0).reshape(batch, -1)
        flat_group_context = context_groups.reshape(batch, seeds * max_queries, reader.heads, reader.head_width)
        context_flat = query_states.new_zeros((batch, queries, reader.heads, reader.head_width))
        context_flat.scatter_add_(
            1,
            flat_query_ids[..., None, None].expand(batch, seeds * max_queries, reader.heads, reader.head_width),
            flat_group_context * packed_query_valid.reshape(batch, -1, 1, 1),
        )
        context = context_flat.reshape(batch, queries, reader.width)
        padded_rows = value_rows - support_count
        pair_indices = torch.nonzero(support, as_tuple=False) if reader.thermal_output is not None else None
        response = None
        if reader.thermal_output is not None:
            assert pair_indices is not None and scalar_grid is not None
            b, q, m = pair_indices.unbind(dim=1) if pair_indices.shape[0] else (
                pair_indices.new_empty((0,)), pair_indices.new_empty((0,)), pair_indices.new_empty((0,))
            )
            coefficients = scalar_grid[b, q, m] if pair_indices.shape[0] else query_states.new_empty((0,))
            response = SourceReadAffineFactorsV1(
                batch_size=batch,
                query_count=queries,
                source_count=modules,
                selected_batch=b,
                selected_query=q,
                selected_source=m,
                residual_coefficients=coefficients,
                present_mask=scene.module_mask,
                heating_envelope=query_states.new_tensor(torch.finfo(query_states.dtype).max),
                scalar_bias=0.0,
            )
        counts = _base_counts(
            support_count=support_count,
            value_rows=value_rows,
            valid_rows=support_count,
            scalar_rows=value_rows if reader.thermal_output is not None else 0,
            padded_rows=padded_rows,
            dense_rows=0,
            protection_exceptions=protection_exceptions,
            protection_exceptions_available=True,
            dispatch_rows=dispatch_count,
            dispatch_groups=group_count,
            group_launches=launches,
            request_syncs=request_syncs + int(query_states.is_cuda and reader.thermal_output is not None),
            candidate_rows=batch * queries * modules,
            coordinate_dim=reader.coordinate_dim,
            thermal=reader.thermal_output is not None,
            hidden_width=reader.width,
        )
        trace = {
            "native_hard_support": support,
            "native_module_ids": scene.module_ids,
            "native_dispatch_seed": dispatch,
            "native_group_member_ids": packed_sources,
            "native_group_member_valid": packed_source_valid,
            "native_group_query_ids": packed_queries,
            "native_group_query_valid": packed_query_valid,
            "native_protected_exceptions": support & protected_mask & ~chosen_members,
        }
        if return_messages:
            assert message_grid is not None
            messages = message_grid
        else:
            messages = None
        return NativeSelectedRead(context, alpha, response, messages, counts, trace, "grouped_native")


def _base_counts(
    *,
    support_count: int,
    value_rows: int,
    valid_rows: int,
    scalar_rows: int,
    padded_rows: int,
    dense_rows: int,
    protection_exceptions: int,
    protection_exceptions_available: bool,
    dispatch_rows: int,
    dispatch_groups: int,
    group_launches: int,
    request_syncs: int,
    candidate_rows: int,
    coordinate_dim: int,
    thermal: bool,
    hidden_width: int,
) -> dict[str, int]:
    return {
        "fine_selected_rows": support_count,
        "fine_executed_rows": value_rows,
        "fine_message_projection_rows": value_rows,
        "fine_scalar_projection_rows": scalar_rows if thermal else 0,
        "fine_padding_rows": padded_rows,
        "fine_fourier_feature_entries": value_rows * 7 * coordinate_dim,
        "fine_hidden_entries": value_rows * hidden_width,
        "source_score_rows": candidate_rows,
        "source_full_fastpath": 0,
        "positive_permission_rows": support_count,
        "unique_nonlinear_rows": value_rows,
        "padded_rows": padded_rows,
        "dense_masked_rows": dense_rows,
        "native_useful_value_rows": support_count,
        "native_value_rows": value_rows,
        "native_valid_value_rows": valid_rows,
        "native_invalid_candidate_rows": max(0, value_rows - valid_rows),
        "native_padded_rows": padded_rows,
        "native_unique_nonlinear_rows": support_count,
        "native_dense_value_rows": dense_rows,
        "native_protection_exception_rows": protection_exceptions,
        "native_protection_exception_rows_available": int(protection_exceptions_available),
        "native_dispatch_rows": dispatch_rows,
        "native_dispatch_groups": dispatch_groups,
        "native_group_launches": group_launches,
        "native_request_syncs": request_syncs,
        "native_candidate_rows": candidate_rows,
        "native_flat_selected_backend": int(value_rows == support_count and dense_rows == 0 and group_launches == 0),
        "native_dense_hard_reference_backend": int(dense_rows > 0),
        "native_grouped_backend": int(group_launches > 0),
    }


__all__ = ["BudgetedNativeReader", "dense_hard_router_gate"]
