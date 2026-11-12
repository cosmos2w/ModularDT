"""Ordered, geometry-aware token exchange for content-hypergraph v2."""

from __future__ import annotations

from dataclasses import replace

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..inherited_attention.contracts import RouteScoreSnapshot
from ..native_retrieval.route_capture import NativeRouteSnapshot, make_native_route_snapshot
from .spatial_features_v2 import pair_score_features


def bounded_temperature(raw: Tensor) -> Tensor:
    """Map live unconstrained parameters to the declared closed interval."""
    return 1.0 + 31.0 * torch.sigmoid(raw)


def _initial_temperature_logit(head_width: int = 32) -> float:
    fraction = (math.sqrt(float(head_width)) - 1.0) / 31.0
    return math.log(fraction / (1.0 - fraction))


def safe_masked_softmax(logits: Tensor, allowed: Tensor, dim: int = -1) -> Tensor:
    """Softmax with exact zero for empty rows and no all-masked NaNs."""
    if logits.shape[dim] == 0:
        return torch.zeros_like(logits)
    allowed = allowed.to(torch.bool)
    row_has_donor = allowed.any(dim=dim, keepdim=True)
    masked = logits.masked_fill(~allowed, -torch.inf)
    masked = torch.where(row_has_donor, masked, torch.zeros_like(masked))
    weights = torch.softmax(masked, dim=dim) * allowed.to(logits.dtype)
    return torch.where(row_has_donor, weights, torch.zeros_like(weights))


def explicit_spatial_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    allowed: Tensor,
    geometry_bias: Tensor,
    donor_measure: Tensor,
    measure_reference: Tensor,
    temperature: Tensor,
) -> tuple[Tensor, Tensor]:
    """Simple reference for the complete normalized-QK attention law."""
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query/key/value must have shape [B,heads,N,width].")
    if allowed.shape != (query.shape[0], query.shape[2], key.shape[2]):
        raise ValueError("allowed has the wrong destination/donor axes.")
    q = F.normalize(query, dim=-1, eps=1.0e-6)
    k = F.normalize(key, dim=-1, eps=1.0e-6)
    logits = torch.matmul(q * temperature[None, :, None, None], k.transpose(-2, -1))
    logits = logits + geometry_bias
    measure = donor_measure.to(logits.dtype)
    safe_measure = torch.where(measure > 0, measure, torch.ones_like(measure))
    ref = measure_reference.to(logits.dtype).reshape(logits.shape[0], 1)
    log_measure = torch.log(safe_measure) - torch.log(ref.clamp_min(torch.finfo(logits.dtype).tiny))
    logits = logits + log_measure[:, None, None, :]
    expanded = allowed[:, None, :, :]
    weights = safe_masked_softmax(logits, expanded, dim=-1)
    return torch.matmul(weights, value), weights


class ContextualPairMessage(nn.Module):
    """Per-head nonlinear receiver/donor/geometry residual, reduced by edges."""

    def __init__(self, width: int, heads: int, geometry_width: int) -> None:
        super().__init__()
        self.receiver = nn.Parameter(torch.empty(heads, width, 32))
        self.donor = nn.Parameter(torch.empty(heads, width, 32))
        self.geometry = nn.Parameter(torch.empty(heads, geometry_width, 32))
        self.bias = nn.Parameter(torch.zeros(heads, 32))
        self.output = nn.Parameter(torch.zeros(heads, 32, width // heads))
        # Standalone route construction also preserves the caller's RNG.
        generator = torch.Generator(device="cpu").manual_seed(0)
        for parameter in (self.receiver, self.donor, self.geometry):
            nn.init.xavier_uniform_(parameter, generator=generator)

    def forward(self, receiver: Tensor, donor: Tensor, geometry: Tensor,
                coefficients: Tensor, eligible: Tensor, tile_size: int = 4096) -> Tensor:
        batch, receivers, _ = receiver.shape
        receiver_projection = torch.einsum('brc,hcp->brhp', receiver, self.receiver)
        donor_projection = torch.einsum('bdc,hcp->bdhp', donor, self.donor)
        rows = eligible.nonzero(as_tuple=False)
        reduced = receiver.new_zeros(batch * receivers, self.bias.shape[0], 32)
        for start in range(0, rows.shape[0], tile_size):
            b, r, d = rows[start:start + tile_size].unbind(-1)
            a = (receiver_projection[b, r] +
                 torch.einsum('eg,hgp->ehp', geometry[b, r, d], self.geometry) + self.bias)
            psi = F.silu(a + donor_projection[b, d]) - F.silu(a)
            reduced = reduced.index_add(0, b * receivers + r, psi * coefficients[b, r, d, :, None])
        reduced = reduced.reshape(batch, receivers, self.bias.shape[0], 32)
        return torch.einsum('brhp,hpv->brhv', reduced, self.output).reshape(batch, receivers, -1)


class ContextualContentReplacement(nn.Module):
    """Two independent zero-initialized shifts on the actual value path.

    Optional work_trace counts forward tensor elements, not timings or FLOPs;
    projection rows count both signs. Checkpoint replay calls are observed
    separately and are updated in place when backward recomputes a tile.
    """

    def __init__(self, width: int, heads: int, geometry_width: int, mode: str):
        super().__init__()
        if mode not in ('c8', 's8'):
            raise ValueError('Content replacement mode must be c8 or s8.')
        self.mode = mode
        self.capture_messages = False
        self.message_trace = None
        self.capture_work = False
        self.work_trace = None
        value_width = width // heads
        for sign in ('plus', 'minus'):
            for name, shape in (('q', (heads, width, value_width)),
                                ('p', (heads, width, value_width)),
                                ('r', (heads, geometry_width, value_width)),
                                ('b', (heads, value_width))):
                self.register_parameter(f'{name}_{sign}', nn.Parameter(torch.zeros(shape),
                    requires_grad=(mode == 's8' or name not in ('p', 'r'))))

    def forward(self, receiver, donor, geometry, value, coefficients, eligible, tile_size=1024):
        batch, receivers, _ = receiver.shape
        self.message_trace = None
        self.work_trace = None
        v = value.transpose(1, 2)
        donor_plus = torch.einsum('bdc,hcv->bdhv', donor, self.q_plus) + self.b_plus
        donor_minus = torch.einsum('bdc,hcv->bdhv', donor, self.q_minus) + self.b_minus
        heads, vector_width = value.shape[1], value.shape[-1]
        donors = donor.shape[1]
        if self.capture_work:
            self.work_trace = dict(mode=self.mode, donor_projection_rows=2 * batch * donors,
                receiver_projection_rows=0, geometry_projection_rows=0,
                donor_nonlinear_heads=0, evaluated_edge_heads=0,
                coefficient_reduction_head_pairs=0, nonlinear_vector_elements=0,
                reduction_vector_elements=0, forward_tile_calls=0, tile_calls=0,
                checkpoint_replay_calls=0)
        if self.mode == 'c8':
            messages = F.silu(v + donor_plus) - F.silu(-v + donor_minus)
            if self.work_trace is not None:
                self.work_trace.update(donor_nonlinear_heads=batch * donors * heads,
                    coefficient_reduction_head_pairs=batch * receivers * donors * heads,
                    nonlinear_vector_elements=2 * batch * donors * heads * vector_width,
                    reduction_vector_elements=batch * receivers * donors * heads * vector_width)
            if self.capture_messages:
                rows = eligible.nonzero(as_tuple=False)
                b, r, d = rows.unbind(-1)
                self.message_trace = dict(rows=rows.detach(), messages=messages[b, d].detach(),
                                          coefficients=coefficients[b, r, d].detach())
            return torch.einsum('brdh,bdhv->brhv', coefficients * eligible[..., None], messages)
        receiver_plus = torch.einsum('brc,hcv->brhv', receiver, self.p_plus)
        receiver_minus = torch.einsum('brc,hcv->brhv', receiver, self.p_minus)
        rows = eligible.nonzero(as_tuple=False)
        work = self.work_trace
        if work is not None:
            edge_count = rows.shape[0]
            work.update(receiver_projection_rows=2 * batch * receivers,
                geometry_projection_rows=2 * edge_count,
                evaluated_edge_heads=edge_count * heads,
                coefficient_reduction_head_pairs=edge_count * heads,
                nonlinear_vector_elements=2 * edge_count * heads * vector_width,
                reduction_vector_elements=edge_count * heads * vector_width)
        reduced = value.new_zeros(batch * receivers, value.shape[1], value.shape[-1])
        captured = [] if self.capture_messages else None

        def tile(vv, dp, dm, rp, rm, geo, weights, rplus, rminus, b, r, d):
            if work is not None:
                work['tile_calls'] += 1
                if work['forward_tile_calls']:
                    work['checkpoint_replay_calls'] += 1
            plus = vv[b, d] + dp[b, d] + rp[b, r] + torch.einsum('eg,hgv->ehv', geo[b, r, d], rplus)
            minus = -vv[b, d] + dm[b, d] + rm[b, r] + torch.einsum('eg,hgv->ehv', geo[b, r, d], rminus)
            return (F.silu(plus) - F.silu(minus)) * weights[b, r, d, :, None]

        for start in range(0, rows.shape[0], tile_size):
            b, r, d = rows[start:start + tile_size].unbind(-1)
            # Checkpoint full endpoint tensors and gather inside the tile so
            # autograd does not retain a separate edge-wide endpoint bank.
            args = (v, donor_plus, donor_minus, receiver_plus, receiver_minus,
                    geometry, coefficients, self.r_plus, self.r_minus, b, r, d)
            if torch.is_grad_enabled() and any(x.requires_grad for x in args):
                from torch.utils.checkpoint import checkpoint
                contribution = checkpoint(tile, *args, use_reentrant=False)
            else:
                contribution = tile(*args)
            if captured is not None:
                # Recover the message directly, including zero coefficients per head.
                with torch.no_grad():
                    vv, dp, dm, rp, rm, geo, _, rplus, rminus, _, _, _ = args
                    captured.append((F.silu(vv[b, d] + dp[b, d] + rp[b, r] + torch.einsum('eg,hgv->ehv', geo[b, r, d], rplus))
                        - F.silu(-vv[b, d] + dm[b, d] + rm[b, r] + torch.einsum('eg,hgv->ehv', geo[b, r, d], rminus))).detach())
            reduced = reduced.index_add(0, b * receivers + r, contribution)
        if work is not None:
            work['forward_tile_calls'] = work['tile_calls']
        if not rows.numel():
            reduced = reduced + (value.sum() + donor_plus.sum() + donor_minus.sum()
                                 + receiver_plus.sum() + receiver_minus.sum()
                                 + self.r_plus.sum() + self.r_minus.sum()) * 0
        if captured is not None:
            b, r, d = rows.unbind(-1)
            messages = torch.cat(captured, dim=0) if captured else value.new_empty(0, value.shape[1], value.shape[-1])
            self.message_trace = dict(rows=rows.detach(), messages=messages,
                                      coefficients=coefficients[b, r, d].detach())
        return reduced.reshape(batch, receivers, value.shape[1], value.shape[-1])


class SpatialRouteAttentionV2(nn.Module):
    """One SDPA route with normalized Q/K and live bounded head temperatures."""

    def __init__(self, width: int, heads: int, coordinate_dim: int, dropout: float = 0.0, *, contextual_message: bool = False) -> None:
        super().__init__()
        if contextual_message and (width // heads != 32 or dropout != 0.0):
            raise ValueError("Contextual messages require head width 32 and zero dropout.")
        if width % heads:
            raise ValueError("context width must be divisible by the attention head count.")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.coordinate_dim = int(coordinate_dim)
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        score_input = 2 * coordinate_dim + 1
        self.geometry_score = nn.Sequential(
            nn.Linear(score_input, 32), nn.SiLU(), nn.Linear(32, heads, bias=False)
        )
        self.pair_message = ContextualPairMessage(width, heads, score_input) if contextual_message else None
        self.content_replacement = None
        self.capture_pair_diagnostics = False
        self.pair_diagnostics = None
        self.raw_temperature = nn.Parameter(torch.full((heads,), _initial_temperature_logit(self.head_width)))

    def enable_content_replacement(self, mode: str) -> None:
        if self.pair_message is not None or self.content_replacement is not None:
            raise ValueError('Content replacement requires a route without another message modification.')
        if self.dropout:
            raise ValueError('Content replacement requires zero attention dropout.')
        self.content_replacement = ContextualContentReplacement(
            self.width, self.heads, 2 * self.coordinate_dim + 1, mode
        ).to(device=self.value.weight.device, dtype=self.value.weight.dtype)

    def record_pair_diagnostics(self, stage_identity, base, correction, coefficients, eligible):
        """Opt-in, last-call trace; counters distinguish forward and backward use."""
        if not self.capture_pair_diagnostics:
            return
        positive = eligible[..., None] & (coefficients > 0)
        positive_heads = int(positive.sum().detach().cpu())
        edges = int(positive.any(-1).sum().detach().cpu()) * self.heads
        batch, receivers, donors = eligible.shape
        record = dict(stage=stage_identity, base=base.detach(), correction=correction.detach(),
                      dense_score_head_pairs=batch * receivers * donors * self.heads,
                      dense_geometry_pairs=batch * receivers * donors,
                      positive_edge_heads=positive_heads, evaluated_edge_heads=edges,
                      forward_reductions=edges,
                      backward_edge_heads=0, backward_reductions=0,
                      grad_enabled=torch.is_grad_enabled())
        self.pair_diagnostics = record
        if correction.requires_grad:
            def backward_count(grad):
                record["backward_edge_heads"] += edges
                record["backward_reductions"] += edges
                return grad
            correction.register_hook(backward_count)

    @property
    def temperature(self) -> Tensor:
        return bounded_temperature(self.raw_temperature)

    def forward(
        self,
        destination: Tensor,
        donor: Tensor,
        destination_coordinates: Tensor,
        donor_coordinates: Tensor,
        destination_lengths: Tensor,
        donor_lengths: Tensor,
        donor_measure: Tensor,
        measure_reference: Tensor,
        donor_mask: Tensor,
        destination_mask: Tensor,
        frame_scale: Tensor,
        *,
        remove_self: bool = False,
        return_attention: bool = False,
        export_score_snapshot: bool = False,
        route_name: str | None = None,
        destination_ids: Tensor | None = None,
        donor_ids: Tensor | None = None,
        geometry_features: Tensor | None = None,
        capture_native_route_snapshot: bool = False,
        stage_identity: str | None = None,
        owner_token: int | None = None,
        parameter_versions: tuple[tuple[int, int], ...] = (),
        buffer_versions: tuple[tuple[int, int], ...] = (),
        scene_identity: tuple[object, object, object] | None = None,
        scene_tensor_versions: tuple[tuple[int, int, int], ...] = (),
        model_config_key: tuple[object, ...] = (),
        value_policy=None,
    ) -> (
        tuple[Tensor, Tensor | None]
        | tuple[Tensor, Tensor | None, RouteScoreSnapshot]
        | tuple[Tensor, Tensor | None, NativeRouteSnapshot]
    ):
        batch, query_count, _ = destination.shape
        key_count = donor.shape[1]
        if key_count == 0:
            attention = destination.new_zeros(batch, self.heads, query_count, 0) if return_attention else None
            update = destination.new_zeros(batch, query_count, self.width)
            if not export_score_snapshot and not capture_native_route_snapshot:
                return update, attention
            empty_query = destination.new_zeros(batch, self.heads, query_count, self.head_width)
            empty_key = destination.new_zeros(batch, self.heads, 0, self.head_width)
            empty_bias = destination.new_zeros(batch, self.heads, query_count, 0)
            if export_score_snapshot and capture_native_route_snapshot:
                raise ValueError("Legacy and native route snapshots use separate versioned contracts.")
            if route_name is None or destination_ids is None or donor_ids is None:
                raise ValueError("An exported route snapshot requires a route name and native IDs.")
            if capture_native_route_snapshot:
                if route_name not in ("mm", "me", "em", "ee") or not stage_identity:
                    raise ValueError("A native route snapshot requires a typed route and stage identity.")
                snapshot = make_native_route_snapshot(
                    route=route_name,
                    query_scaled=empty_query,
                    key_normalized=empty_key,
                    geometry_bias=empty_bias,
                    allowed=donor_mask.new_zeros(batch, query_count, 0),
                    destination_mask=destination_mask,
                    donor_mask=donor_mask,
                    destination_ids=destination_ids,
                    donor_ids=donor_ids,
                    donor_measure=donor_measure,
                    stage_identity=stage_identity,
                    owner_token=owner_token,
                    parameter_versions=parameter_versions,
                    buffer_versions=buffer_versions,
                    scene_identity=scene_identity,
                    scene_tensor_versions=scene_tensor_versions,
                    model_config_key=model_config_key,
                )
            else:
                if route_name not in ("em", "ee"):
                    raise ValueError("The legacy route snapshot remains restricted to EM/EE.")
                snapshot = RouteScoreSnapshot(
                    route=route_name,
                    query_scaled=empty_query,
                    key_normalized=empty_key,
                    geometry_bias=empty_bias,
                    raw_logits=empty_bias,
                    donor_mask=donor_mask & (donor_measure > 0),
                    destination_mask=destination_mask,
                    donor_measure=donor_measure,
                    anchor_ids=destination_ids,
                    donor_ids=donor_ids,
                    score_origin=f"last_{route_name}",
                )
            if value_policy is not None and capture_native_route_snapshot:
                snapshot = replace(snapshot, applied_incidence=destination.new_zeros(batch, query_count, 0),
                    applied_message_mass=destination.new_zeros(batch,query_count,self.heads))
            return update, attention, snapshot

        receiver_normalized = self.query_norm(destination)
        q = self.query(receiver_normalized)
        donor_normalized = self.key_norm(donor)
        k = self.key(donor_normalized)
        v = self.value(donor_normalized)
        q = q.view(batch, query_count, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, key_count, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, key_count, self.heads, self.head_width).transpose(1, 2)
        q = F.normalize(q, dim=-1, eps=1.0e-6)
        k = F.normalize(k, dim=-1, eps=1.0e-6)
        # Scaling Q by a live tensor and setting SDPA scale=1 avoids the
        # implementation's default 1/sqrt(head_width) a second time.
        q = q * self.temperature[None, :, None, None]

        if geometry_features is None:
            geometry_features, _ = pair_score_features(
                destination_coordinates,
                donor_coordinates,
                frame_scale,
                destination_lengths,
                donor_lengths,
            )
        geometry_bias = self.geometry_score(geometry_features).permute(0, 3, 1, 2)
        measure = donor_measure.to(dtype=destination.dtype)
        valid_donors = donor_mask & (measure > 0)
        allowed = valid_donors[:, None, :].expand(batch, query_count, key_count)
        allowed = allowed & destination_mask[:, :, None]
        if remove_self:
            if query_count != key_count:
                raise ValueError("Self-exclusion requires equal destination and donor axes.")
            allowed = allowed & ~torch.eye(query_count, dtype=torch.bool, device=destination.device)[None, :, :]

        safe_measure = torch.where(valid_donors, measure, torch.ones_like(measure))
        reference = measure_reference.to(dtype=destination.dtype).reshape(batch, 1)
        log_measure = torch.log(safe_measure) - torch.log(reference.clamp_min(torch.finfo(destination.dtype).tiny))
        additive = geometry_bias + log_measure[:, None, None, :]
        additive = additive.masked_fill(~allowed[:, None, :, :], -torch.inf)
        row_has_donor = allowed.any(dim=-1, keepdim=True)
        additive = torch.where(row_has_donor[:, None, :, :], additive, torch.zeros_like(additive))
        snapshot = None
        native_snapshot = None
        if export_score_snapshot:
            if route_name not in ("em", "ee") or destination_ids is None or donor_ids is None:
                raise ValueError("An exported route snapshot requires typed EM/EE names and native IDs.")
            raw_logits = torch.matmul(q, k.transpose(-2, -1)) + geometry_bias
            snapshot = RouteScoreSnapshot(
                route=route_name,
                query_scaled=q,
                key_normalized=k,
                geometry_bias=geometry_bias,
                raw_logits=raw_logits,
                donor_mask=donor_mask,
                destination_mask=destination_mask,
                donor_measure=donor_measure,
                anchor_ids=destination_ids,
                donor_ids=donor_ids,
                score_origin=f"last_{route_name}",
            )
        if capture_native_route_snapshot:
            if export_score_snapshot:
                raise ValueError("Legacy and native route snapshots use separate versioned contracts.")
            if route_name not in ("mm", "me", "em", "ee") or destination_ids is None or donor_ids is None:
                raise ValueError("A native route snapshot requires a typed route and native IDs.")
            if not stage_identity:
                raise ValueError("A native route snapshot requires a stage identity.")
            native_snapshot = make_native_route_snapshot(
                route=route_name,
                query_scaled=q,
                key_normalized=k,
                geometry_bias=geometry_bias,
                allowed=allowed,
                destination_mask=destination_mask,
                # Keep native scene presence distinct from contextualizer
                # eligibility: a present zero-measure module can still be an
                # exact native thermal source and retrieval self-member.
                donor_mask=donor_mask,
                destination_ids=destination_ids,
                donor_ids=donor_ids,
                donor_measure=donor_measure,
                stage_identity=stage_identity,
                owner_token=owner_token,
                parameter_versions=parameter_versions,
                buffer_versions=buffer_versions,
                scene_identity=scene_identity,
                scene_tensor_versions=scene_tensor_versions,
                model_config_key=model_config_key,
            )
        if value_policy is not None:
            from ..continuous_joint.preparation import contextual_value_update
            update, weights, applied, applied_mass = contextual_value_update(
                self, q, k, v, geometry_bias, donor_measure, allowed,
                destination_mask, value_policy, route_name=route_name,
                stage_identity=stage_identity, destination_ids=destination_ids,
                donor_ids=donor_ids, receiver_normalized=receiver_normalized,
                donor_normalized=donor_normalized, geometry_features=geometry_features)
            if native_snapshot is not None:
                # Snapshot binds scores to their actual source-state stage;
                # coefficients are exported separately, never recomputed afterward.
                native_snapshot = replace(native_snapshot, applied_incidence=applied, applied_message_mass=applied_mass)
            exported = native_snapshot if native_snapshot is not None else snapshot
            return (update, weights if return_attention else None, exported) if exported is not None else (
                update, weights if return_attention else None)
        if self.content_replacement is not None:
            raise ValueError('Content replacement requires the contextual value policy.')
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=1.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_count, self.width)
        update = self.output(attended)
        if self.pair_message is not None:
            logits = torch.matmul(q, k.transpose(-2, -1)) + additive
            coefficients = safe_masked_softmax(logits, allowed[:, None], dim=-1).permute(0, 2, 3, 1)
            residual = self.pair_message(receiver_normalized, donor_normalized, geometry_features,
                                         coefficients, allowed & (coefficients > 0).any(-1))
            correction = F.linear(residual, self.output.weight, None)
            valid_trace = (row_has_donor & destination_mask[:, :, None]).to(update.dtype)
            correction = correction * valid_trace
            self.record_pair_diagnostics(stage_identity, update * valid_trace,
                                         correction, coefficients, allowed)
            update = update + correction
        valid_rows = row_has_donor & destination_mask[:, :, None]
        update = update * valid_rows.to(update.dtype)
        if not return_attention:
            exported = native_snapshot if native_snapshot is not None else snapshot
            return (update, None, exported) if exported is not None else (update, None)
        # Trace-only explicit weights; the production path remains SDPA.
        logits = torch.matmul(q, k.transpose(-2, -1)) + additive
        weights = safe_masked_softmax(logits, allowed[:, None, :, :], dim=-1)
        exported = native_snapshot if native_snapshot is not None else snapshot
        return (update, weights, exported) if exported is not None else (update, weights)


class TokenFeedForwardV2(nn.Module):
    """One ordinary tokenwise pre-normalized residual FFN."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        updated = values + self.dropout(self.net(self.norm(values)))
        return updated * mask[:, :, None].to(updated.dtype)


class StagedContextCycleV2(nn.Module):
    """MM → EM(new M) → EE(new E) → ME(new E), followed by one/type FFN."""

    def __init__(self, width: int, heads: int, coordinate_dim: int, dropout: float = 0.0, *, contextual_message: bool = False) -> None:
        super().__init__()
        self.mm = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout, contextual_message=contextual_message)
        self.em = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout, contextual_message=contextual_message)
        self.ee = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout, contextual_message=contextual_message)
        self.me = SpatialRouteAttentionV2(width, heads, coordinate_dim, dropout, contextual_message=contextual_message)
        self.environment_ff = TokenFeedForwardV2(width, dropout)
        self.module_ff = TokenFeedForwardV2(width, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        module_states: Tensor,
        environment_states: Tensor,
        scene,
        frame_scale: Tensor,
        *,
        return_attention: bool = False,
        export_route_scores: bool = False,
        capture_native_routes: bool = False,
        native_capture_route_names: tuple[str, ...] = ("mm", "me"),
        cycle_index: int = 0,
        owner_token: int | None = None,
        parameter_versions: tuple[tuple[int, int], ...] = (),
        buffer_versions: tuple[tuple[int, int], ...] = (),
        scene_identity: tuple[object, object, object] | None = None,
        scene_tensor_versions: tuple[tuple[int, int, int], ...] = (),
        model_config_key: tuple[object, ...] = (),
        geometry_features: dict[str, Tensor] | None = None,
        value_policy=None,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        module_lengths = scene.module_characteristic_lengths
        environment_lengths = scene.environment_characteristic_lengths
        module_measure = scene.module_measure
        environment_measure = scene.environment_measure
        module_reference = scene.module_measure_reference
        environment_reference = scene.environment_measure_reference
        batch = module_states.shape[0]
        if module_reference is None:
            module_reference = module_measure.new_ones(batch)
        if environment_reference is None:
            environment_reference = environment_measure.new_ones(batch)

        mm_result = self.mm(
            module_states, module_states,
            scene.module_coordinates, scene.module_coordinates,
            module_lengths, module_lengths,
            module_measure, module_reference,
            scene.module_mask, scene.module_mask, frame_scale,
            remove_self=True, return_attention=return_attention,
            route_name="mm",
            destination_ids=scene.module_ids,
            donor_ids=scene.module_ids,
            capture_native_route_snapshot=capture_native_routes and "mm" in native_capture_route_names,
            stage_identity=f"cycle{cycle_index}/mm",
            owner_token=owner_token,
            parameter_versions=parameter_versions,
            buffer_versions=buffer_versions,
            scene_identity=scene_identity,
            scene_tensor_versions=scene_tensor_versions,
            model_config_key=model_config_key,
            geometry_features=None if geometry_features is None else geometry_features["mm"],
            value_policy=value_policy,
        )
        if capture_native_routes and "mm" in native_capture_route_names:
            mm, mm_weights, mm_snapshot = mm_result
        else:
            mm, mm_weights = mm_result
            mm_snapshot = None
        module_after_mm = module_states + self.dropout(mm)
        module_after_mm = module_after_mm * scene.module_mask[:, :, None].to(module_after_mm.dtype)

        em_result = self.em(
            environment_states, module_after_mm,
            scene.environment_coordinates, scene.module_coordinates,
            environment_lengths, module_lengths,
            module_measure, module_reference,
            scene.module_mask, scene.environment_mask, frame_scale,
            return_attention=return_attention,
            export_score_snapshot=export_route_scores,
            route_name="em",
            destination_ids=scene.environment_ids,
            donor_ids=scene.module_ids,
            capture_native_route_snapshot=capture_native_routes and "em" in native_capture_route_names,
            stage_identity=f"cycle{cycle_index}/em",
            owner_token=owner_token,
            parameter_versions=parameter_versions,
            buffer_versions=buffer_versions,
            scene_identity=scene_identity,
            scene_tensor_versions=scene_tensor_versions,
            model_config_key=model_config_key,
            geometry_features=None if geometry_features is None else geometry_features["em"],
            value_policy=value_policy,
        )
        if export_route_scores or (capture_native_routes and "em" in native_capture_route_names):
            em, em_weights, em_snapshot = em_result
        else:
            em, em_weights = em_result
            em_snapshot = None
        environment_after_em = environment_states + self.dropout(em)
        environment_after_em = environment_after_em * scene.environment_mask[:, :, None].to(
            environment_after_em.dtype
        )

        ee_result = self.ee(
            environment_after_em, environment_after_em,
            scene.environment_coordinates, scene.environment_coordinates,
            environment_lengths, environment_lengths,
            environment_measure, environment_reference,
            scene.environment_mask, scene.environment_mask, frame_scale,
            return_attention=return_attention,
            export_score_snapshot=export_route_scores,
            route_name="ee",
            destination_ids=scene.environment_ids,
            donor_ids=scene.environment_ids,
            capture_native_route_snapshot=capture_native_routes and "ee" in native_capture_route_names,
            stage_identity=f"cycle{cycle_index}/ee",
            owner_token=owner_token,
            parameter_versions=parameter_versions,
            buffer_versions=buffer_versions,
            scene_identity=scene_identity,
            scene_tensor_versions=scene_tensor_versions,
            model_config_key=model_config_key,
            geometry_features=None if geometry_features is None else geometry_features["ee"],
            value_policy=value_policy,
        )
        if export_route_scores or (capture_native_routes and "ee" in native_capture_route_names):
            ee, ee_weights, ee_snapshot = ee_result
        else:
            ee, ee_weights = ee_result
            ee_snapshot = None
        environment_after_ee = environment_after_em + self.dropout(ee)
        environment_next = self.environment_ff(environment_after_ee, scene.environment_mask)

        me_result = self.me(
            module_after_mm, environment_next,
            scene.module_coordinates, scene.environment_coordinates,
            module_lengths, environment_lengths,
            environment_measure, environment_reference,
            scene.environment_mask, scene.module_mask, frame_scale,
            return_attention=return_attention,
            route_name="me",
            destination_ids=scene.module_ids,
            donor_ids=scene.environment_ids,
            capture_native_route_snapshot=capture_native_routes and "me" in native_capture_route_names,
            stage_identity=f"cycle{cycle_index}/me",
            owner_token=owner_token,
            parameter_versions=parameter_versions,
            buffer_versions=buffer_versions,
            scene_identity=scene_identity,
            scene_tensor_versions=scene_tensor_versions,
            model_config_key=model_config_key,
            geometry_features=None if geometry_features is None else geometry_features["me"],
            value_policy=value_policy,
        )
        if capture_native_routes and "me" in native_capture_route_names:
            me, me_weights, me_snapshot = me_result
        else:
            me, me_weights = me_result
            me_snapshot = None
        module_after_me = module_after_mm + self.dropout(me)
        module_next = self.module_ff(module_after_me, scene.module_mask)
        trace = {
            "module_after_mm": module_after_mm,
            "environment_after_em": environment_after_em,
            "environment_after_ee": environment_next,
            "module_after_me": module_next,
        }
        if return_attention:
            for name, value in (
                ("mm_attention", mm_weights),
                ("em_attention", em_weights),
                ("ee_attention", ee_weights),
                ("me_attention", me_weights),
            ):
                if value is not None:
                    trace[name] = value
        if export_route_scores:
            trace["em_score_snapshot"] = em_snapshot
            trace["ee_score_snapshot"] = ee_snapshot
        if value_policy is not None:
            for route, snapshot in (("mm", mm_snapshot), ("em", em_snapshot),
                                    ("ee", ee_snapshot), ("me", me_snapshot)):
                if snapshot is not None and hasattr(snapshot, "applied_incidence"):
                    trace[f"{route}_applied_incidence"] = snapshot.applied_incidence
        if capture_native_routes:
            trace["mm_native_route_snapshot"] = mm_snapshot
            trace["me_native_route_snapshot"] = me_snapshot
            if "em" in native_capture_route_names:
                trace["em_native_route_snapshot"] = em_snapshot
            if "ee" in native_capture_route_names:
                trace["ee_native_route_snapshot"] = ee_snapshot
        return module_next, environment_next, trace
