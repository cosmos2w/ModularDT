"""Selected native-token reads using the original A2-D value function.

Scoring uses the small original geometry descriptor on the full query/module
axes. Relative Fourier features, nonlinear messages, and signed Thermal
coefficients are then evaluated only for unique permission-positive native
module rows. The dense-masked backend is a same-function correctness and cost
reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..content_hypergraph.contracts import PreparedOrganization, QueryBatch, SceneBatch
from ..content_hypergraph.decoding import _query_mask
from ..content_hypergraph.readers_v2 import DirectSourceReaderV2
from ..content_hypergraph.source_read_response import SourceReadAffineFactorsV1
from ..content_hypergraph.spatial_features_v2 import (
    query_source_features,
    query_source_score_features,
)
from ..content_hypergraph.staged_attention_v2 import safe_masked_softmax


@dataclass(frozen=True)
class NativeModuleScores:
    """Narrow original A2-D scores; no value Fourier features are stored."""

    raw_logits: Tensor  # [B, H, Q, M], before physical measure and permission
    score_descriptor: Tensor  # [B, Q, M, 2D+1]


@dataclass(frozen=True)
class NativeSelectedRead:
    context: Tensor
    alpha: Tensor  # [B, Q, M, H], permission/measure-normalized information weights
    response: SourceReadAffineFactorsV1 | None
    messages: Tensor | None
    counts: dict[str, int]
    trace: dict[str, Tensor]
    backend: str


def score_native_modules(
    reader: DirectSourceReaderV2,
    scene: SceneBatch,
    query: QueryBatch,
    query_states: Tensor,
    frame_scale: Tensor,
    prepared_keys: Tensor | None,
) -> NativeModuleScores:
    """Compute original decoder logits without physical measure or wide values."""
    batch, query_count, _ = query_states.shape
    module_count = scene.num_modules
    descriptor = query_source_score_features(
        query.coordinates.to(query_states.dtype),
        scene.module_coordinates.to(query_states.dtype),
        frame_scale,
        scene.module_characteristic_lengths.to(query_states.dtype),
    )
    if query_count == 0 or module_count == 0:
        return NativeModuleScores(
            raw_logits=query_states.new_zeros((batch, reader.heads, query_count, module_count)),
            score_descriptor=descriptor,
        )
    if prepared_keys is None:
        raise ValueError("Prepared A2-D native module keys are missing.")
    query_key = reader.query_key(reader.query_norm(query_states)).reshape(
        batch, query_count, reader.heads, reader.head_width
    )
    module_key = prepared_keys.reshape(batch, module_count, reader.heads, reader.head_width)
    query_key = F.normalize(query_key, dim=-1, eps=1.0e-6)
    module_key = F.normalize(module_key, dim=-1, eps=1.0e-6)
    logits = torch.einsum("bqhd,bmhd->bhqm", query_key, module_key)
    logits = logits * reader.temperature[None, :, None, None]
    logits = logits + reader.geometry_score(descriptor).permute(0, 3, 1, 2)
    return NativeModuleScores(logits, descriptor)


def _relative_gamma_rows(delta: Tensor, frame_scale: Tensor) -> Tensor:
    """Match relative_gamma for packed [N,D] query--module displacements."""
    normalized = delta / frame_scale
    pieces = [normalized]
    for frequency in (1.0, 2.0, 4.0):
        phase = (2.0 * torch.pi * frequency) * normalized
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(pieces, dim=-1)


def _selected_value_rows(
    reader: DirectSourceReaderV2,
    scene: SceneBatch,
    query: QueryBatch,
    query_states: Tensor,
    frame_scale: Tensor,
    prepared_values: Tensor,
    score_descriptor: Tensor,
    pair_indices: Tensor,
) -> Tensor:
    """Apply the unchanged native-token nonlinear formula to packed rows."""
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 3:
        raise ValueError("unique_module_indices must have shape [N,3]=(batch,query,module).")
    b, q, m = pair_indices.to(dtype=torch.long).unbind(dim=1)
    delta = query.coordinates.to(query_states.dtype)[b, q] - scene.module_coordinates.to(query_states.dtype)[b, m]
    relative_gamma = _relative_gamma_rows(delta, frame_scale[b])
    relative_value = reader.relative_value(relative_gamma)
    if reader.source_scale_projection is not None:
        score_scale = score_descriptor[b, q, m, -reader.coordinate_dim :]
        relative_value = relative_value + reader.source_scale_projection(score_scale)
    query_value = reader.query_value(query_states)
    hidden = F.silu(query_value[b, q] + prepared_values[b, m] + relative_value)
    return hidden


def _dense_masked_value_rows(
    reader: DirectSourceReaderV2,
    scene: SceneBatch,
    query: QueryBatch,
    query_states: Tensor,
    frame_scale: Tensor,
    prepared_values: Tensor,
    pair_indices: Tensor,
) -> Tensor:
    """Evaluate the same selected-permission function through a dense reference."""
    geometry, relative_gamma = query_source_features(
        query.coordinates.to(query_states.dtype),
        scene.module_coordinates.to(query_states.dtype),
        frame_scale,
        scene.module_characteristic_lengths.to(query_states.dtype),
    )
    relative_value = reader.relative_value(relative_gamma)
    if reader.source_scale_projection is not None:
        relative_value = relative_value + reader.source_scale_projection(
            geometry[..., -reader.coordinate_dim :]
        )
    query_value = reader.query_value(query_states)
    all_hidden = F.silu(query_value[:, :, None, :] + prepared_values[:, None, :, :] + relative_value)
    b, q, m = pair_indices.to(dtype=torch.long).unbind(dim=1)
    return all_hidden[b, q, m]


def _permission_tensors(
    plan: Any,
    scene: SceneBatch,
    query: QueryBatch,
    query_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    permission = getattr(plan, "module_permission", None)
    indices = getattr(plan, "unique_module_indices", None)
    if not isinstance(permission, Tensor) or permission.ndim != 4:
        raise ValueError("retrieval plan module_permission must have shape [B,H,Q,M].")
    batch, queries = query_states.shape[:2]
    modules = scene.num_modules
    if tuple(permission.shape) != (batch, permission.shape[1], queries, modules):
        raise ValueError("module_permission dimensions do not match the scene and query axes.")
    if permission.shape[1] <= 0:
        raise ValueError("module_permission must contain at least one decoder head.")
    if indices is None or not isinstance(indices, Tensor) or indices.dtype != torch.long:
        raise ValueError("retrieval plan unique_module_indices must be int64 [N,3].")
    if indices.ndim != 2 or indices.shape[-1] != 3:
        raise ValueError("unique_module_indices must have shape [N,3].")
    if not permission.is_floating_point():
        raise TypeError("module_permission must be a floating tensor so fractional gates remain live.")
    if indices.device != query_states.device or permission.device != query_states.device:
        raise ValueError("Native module permissions and indices must share the query device.")
    permission_valid = torch.isfinite(permission).all() & (permission >= 0).all() & (permission <= 1).all()
    if permission_valid.device.type == "cuda":
        torch._assert_async(permission_valid, "module_permission must be finite and lie in [0,1].")
    elif not bool(permission_valid):
        raise ValueError("module_permission must be finite and lie in [0,1].")
    permission = permission.to(dtype=query_states.dtype)
    valid = scene.module_mask[:, None, :] & _query_mask(query)[:, :, None]
    support = (permission > 0).any(dim=1) & valid
    return permission, indices, support


class NativeSelectedReader:
    """Stateless opt-in executor around the historical ``DirectSourceReaderV2``."""

    @staticmethod
    def score_queries(
        reader: DirectSourceReaderV2,
        organization: PreparedOrganization,
        query: QueryBatch,
        query_states: Tensor,
    ) -> NativeModuleScores:
        return score_native_modules(
            reader,
            organization.scene,
            query,
            query_states,
            organization.frame_scale,
            organization.direct_source_keys,
        )

    @staticmethod
    def evaluate(
        reader: DirectSourceReaderV2,
        organization: PreparedOrganization,
        query: QueryBatch,
        query_states: Tensor,
        plan: Any,
        *,
        backend: str = "selected",
        return_messages: bool = False,
        measure_reference: Tensor | None = None,
        scores: NativeModuleScores | None = None,
    ) -> NativeSelectedRead:
        """Evaluate selected native values and reduce with live permissions.

        ``selected`` evaluates one nonlinear row per unique native
        ``(batch,query,module)``. ``dense_masked`` evaluates every row but uses
        the same permissions, physical measures, and output law.
        """
        if backend not in {"selected", "dense_masked", "compiled_selected"}:
            raise ValueError("backend must be 'selected', 'dense_masked', or 'compiled_selected'.")
        if getattr(plan, "environment_value_consumer", False):
            raise ValueError("This M-only reader cannot consume E values; the direct-E pilot is a separate architecture.")
        if callable(getattr(plan, "assert_query_current", None)):
            plan.assert_query_current(
                query, organization_model_owner=getattr(organization, "model_owner", None),
                organization_token_store=getattr(organization, "native_token_store", None))
        scene = organization.scene
        batch, query_count, _ = query_states.shape
        module_count = scene.num_modules
        if (
            (organization.direct_source_keys is None or organization.direct_source_values is None)
            and module_count
            and query_count
        ):
            raise ValueError("Prepared A2-D native module projections are missing.")
        permission, pair_indices, support = _permission_tensors(plan, scene, query, query_states)
        if permission.shape[1] != reader.heads:
            raise ValueError("Native module permissions have the wrong decoder-head count.")
        if scores is None:
            scores = score_native_modules(
                reader, scene, query, query_states, organization.frame_scale,
                organization.direct_source_keys,
            )
        raw_logits = scores.raw_logits
        if tuple(raw_logits.shape) != (batch, reader.heads, query_count, module_count):
            raise ValueError("Native module scores have the wrong shape.")

        measure = scene.module_measure.to(query_states.dtype)
        if measure_reference is None:
            measure_reference = scene.module_measure_reference
        if measure_reference is None:
            measure_reference = query_states.new_ones((batch,))
        reference = measure_reference.to(query_states.dtype).reshape(batch, 1)
        info_valid = (scene.module_mask & (measure > 0))[:, None, None, :]
        query_valid = _query_mask(query)[:, :, None, None]
        head_permission = permission.permute(0, 2, 1, 3)  # [B,Q,H,M]
        safe_measure = torch.where(scene.module_mask & (measure > 0), measure, torch.ones_like(measure))
        log_measure = torch.log(safe_measure) - torch.log(
            reference.clamp_min(torch.finfo(query_states.dtype).tiny)
        )
        safe_permission = torch.where(
            head_permission > 0, head_permission, torch.ones_like(head_permission)
        )
        log_permission = torch.log(safe_permission)
        attention_logits = raw_logits.permute(0, 2, 1, 3) + log_measure[:, None, None, :] + log_permission
        allowed = info_valid & query_valid & (head_permission > 0)
        alpha_head_major = safe_masked_softmax(attention_logits, allowed, dim=-1)
        alpha = alpha_head_major.permute(0, 1, 3, 2)

        if pair_indices.numel():
            selected_rows = (
                pair_indices[:, 0] * query_count + pair_indices[:, 1]
            ) * module_count + pair_indices[:, 2]
        else:
            selected_rows = pair_indices.new_empty((0,))
        compiled_coefficients = None
        actual_backend = backend
        if backend == "compiled_selected" and (
            torch.is_grad_enabled() or pair_indices.shape[0] == 0
        ):
            # Training and native derivative probes keep the exact eager path.
            # Empty support has no selected arithmetic to compile.
            actual_backend = "selected"
        if module_count == 0 or query_count == 0:
            hidden = query_states.new_empty((0, reader.width))
            message_rows = reader.message_output(hidden)
            executed_rows = 0
            context = query_states.new_zeros((batch, query_count, reader.width))
        elif actual_backend == "compiled_selected":
            from .execution import compiled_selected_values

            context, message_rows, compiled_coefficients = compiled_selected_values(
                reader, organization, query, query_states, pair_indices, alpha, permission, scores
            )
            executed_rows = pair_indices.shape[0]
        elif actual_backend == "selected":
            hidden = _selected_value_rows(
                reader, scene, query, query_states, organization.frame_scale,
                organization.direct_source_values, scores.score_descriptor, pair_indices,
            )
            message_rows = reader.message_output(hidden)
            executed_rows = pair_indices.shape[0]
        else:
            hidden = _dense_masked_value_rows(
                reader, scene, query, query_states, organization.frame_scale,
                organization.direct_source_values, pair_indices,
            )
            message_rows = reader.message_output(hidden)
            executed_rows = batch * query_count * module_count

        if actual_backend != "compiled_selected" and pair_indices.shape[0]:
            b, q, m = pair_indices.unbind(dim=1)
            row_alpha = alpha[b, q, m, :]
            blocks = message_rows.reshape(-1, reader.heads, reader.head_width)
            weighted = blocks * row_alpha[..., None]
            query_rows = b * query_count + q
            reduced = weighted.new_zeros((batch * query_count, reader.heads, reader.head_width))
            reduced = reduced.index_add(0, query_rows, weighted)
            context = reduced.reshape(batch, query_count, reader.width)
        elif actual_backend != "compiled_selected":
            context = query_states.new_zeros((batch, query_count, reader.width))

        response = None
        if reader.thermal_output is not None:
            if pair_indices.shape[0]:
                b, q, m = pair_indices.unbind(dim=1)
                query_gate = _query_mask(query)[b, q].to(message_rows.dtype)
                source_gate = scene.module_mask[b, m].to(message_rows.dtype)
                if compiled_coefficients is None:
                    scalar_rows = reader.thermal_output(message_rows).squeeze(-1)
                    scalar_gate = permission.amax(dim=1)[b, q, m]
                    coefficients = scalar_rows * scalar_gate
                else:
                    coefficients = compiled_coefficients
                coefficients = coefficients * query_gate * source_gate
            else:
                b = q = m = pair_indices.new_empty((0,))
                coefficients = query_states.new_empty((0,))
            envelope = query_states.new_tensor(torch.finfo(query_states.dtype).max)
            response = SourceReadAffineFactorsV1(
                batch_size=batch,
                query_count=query_count,
                source_count=module_count,
                selected_batch=b,
                selected_query=q,
                selected_source=m,
                residual_coefficients=coefficients,
                present_mask=scene.module_mask,
                heating_envelope=envelope,
                query_mask=_query_mask(query),
                scalar_bias=0.0,
            )

        messages = None
        if return_messages:
            flat_messages = message_rows.new_zeros((batch * query_count * module_count, reader.width))
            if pair_indices.shape[0]:
                flat_messages = flat_messages.index_copy(0, selected_rows, message_rows)
            messages = flat_messages.reshape(batch, query_count, module_count, reader.width)

        # Counts are shape-derived so this path does not synchronize CUDA for
        # telemetry. The plan owns unique row compaction and its associated cost.
        selected_count = int(pair_indices.shape[0])
        candidates = batch * query_count * module_count
        dense_rows = candidates if actual_backend == "dense_masked" else 0
        environment_indices = getattr(plan, "unique_environment_indices", None)
        environment_ids = int(environment_indices.shape[0]) if isinstance(environment_indices, Tensor) else 0
        counts = {
            "fine_selected_rows": selected_count,
            "fine_executed_rows": int(executed_rows),
            "fine_message_projection_rows": selected_count,
            "fine_scalar_projection_rows": selected_count if reader.thermal_output is not None else 0,
            "fine_padding_rows": 0,
            "fine_fourier_feature_entries": int(executed_rows * 7 * reader.coordinate_dim),
            "fine_hidden_entries": int(executed_rows * reader.width),
            "source_score_rows": int(candidates),
            "source_utility_score_rows": 0,
            "source_full_fastpath": 0,
            "positive_permission_rows": selected_count,
            "unique_nonlinear_rows": int(executed_rows),
            "padded_rows": 0,
            "dense_masked_rows": int(dense_rows),
            "environment_metadata_ids": environment_ids,
            "environment_consumed_ids": 0,
            # This is the scored rectangular candidate count before native
            # validity masks. It is deliberately not called an eligible count.
            "native_scored_candidate_rows": int(candidates),
            "native_positive_permission_rows": selected_count,
            "native_unique_nonlinear_rows": selected_count,
            "native_padded_rows": 0,
            "native_dense_masked_rows": int(dense_rows),
            "native_selected_backend": int(actual_backend == "selected"),
            "native_compiled_selected_backend": int(actual_backend == "compiled_selected"),
            "native_dense_masked_backend": int(actual_backend == "dense_masked"),
            "native_environment_metadata_ids": environment_ids,
            "native_environment_consumed_ids": 0,
        }
        trace = {
            "module_permission": permission,
            "module_support": support,
            "unique_module_indices": pair_indices,
            "native_module_permission": permission,
        }
        seed_permission = getattr(plan, "seed_permission", None)
        if isinstance(seed_permission, Tensor):
            trace["native_seed_permission"] = seed_permission
        if pair_indices.shape[0]:
            b, q, m = pair_indices.unbind(dim=1)
            trace.update(
                selected_module_batch_indices=b,
                selected_module_query_indices=q,
                selected_module_indices=m,
                selected_module_native_ids=scene.module_ids[b, m],
                native_selected_batch=b,
                native_selected_query=q,
                native_selected_source=m,
                native_selected_source_ids=scene.module_ids[b, m],
            )
        else:
            trace.update(
                native_selected_batch=pair_indices.new_empty((0,)),
                native_selected_query=pair_indices.new_empty((0,)),
                native_selected_source=pair_indices.new_empty((0,)),
                native_selected_source_ids=scene.module_ids.new_empty((0,)),
            )
        return NativeSelectedRead(context, alpha, response, messages, counts, trace, actual_backend)


__all__ = [
    "NativeModuleScores",
    "NativeSelectedRead",
    "NativeSelectedReader",
    "score_native_modules",
]
