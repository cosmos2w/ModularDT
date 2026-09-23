"""Run-1503 adaptive hyperedge opening environmental reader.

The Run-1503 reader keeps the Run-1502 source organizer, query router, and
module reader intact.  Environmental sources are additionally pooled after
the existing physical MM/ME/EM preparation.  Every group is therefore
available through one cheap aggregate, while only groups with positive query
sparsemax support are evaluated against their fine source rectangle.

There is deliberately no second router here.  ``route.logits`` is the one
query/group score bank: it produces the existing sparse opening assignment
``alpha`` and, after the environmental-mass prior is added, the dense coarse
mixture ``p``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from .sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)
from .sparse_incidence_router import SparseIncidencePreparedGroupControl
from .types import EncodedInterfaceCase

ADAPTIVE_HYPEREDGE_EPSILON = 1.0e-8
# The K-batched executor trades common-padding memory for much lower launch
# overhead.  Measurements show that tradeoff is favorable for the training
# tile (32) and standard small-chunk inference (128), but not for larger
# receiver tiles.  Keep this as a runtime policy constant: it is not a model
# or checkpoint configuration field.
ADAPTIVE_HYPEREDGE_BATCHED_QUERY_LIMIT = 128


@dataclass(frozen=True)
class AdaptiveEnvironmentAggregation:
    """Mass-weighted environmental hyperedge state used by Run-1503."""

    mass: torch.Tensor
    centroids: torch.Tensor
    radius_sq: torch.Tensor
    states: torch.Tensor
    source_mass: torch.Tensor


def mass_weighted_environment_aggregates(
    environment_states: torch.Tensor,
    environment_coordinates: torch.Tensor,
    environment_membership: torch.Tensor,
    environment_measure: torch.Tensor,
    *,
    epsilon: float = ADAPTIVE_HYPEREDGE_EPSILON,
) -> AdaptiveEnvironmentAggregation:
    """Pool contextualized environmental states by physical hyperedge mass.

    ``environment_measure`` is the normalized quadrature measure ``nu`` and
    ``environment_membership`` is the nonnegative, row-normalized ``A^E``.
    The source mass ``nu_j A^E_jk`` is retained explicitly so fine group
    readers can partition (rather than duplicate) the physical measure.
    Empty groups return finite zero state/geometry rows.
    """

    if not isinstance(environment_states, torch.Tensor) or environment_states.ndim != 3:
        raise ValueError("environment_states must have shape [B,E,H].")
    if not isinstance(environment_coordinates, torch.Tensor) or environment_coordinates.ndim != 3:
        raise ValueError("environment_coordinates must have shape [B,E,d].")
    if not isinstance(environment_membership, torch.Tensor) or environment_membership.ndim != 3:
        raise ValueError("environment_membership must have shape [B,E,K].")
    if not isinstance(environment_measure, torch.Tensor) or environment_measure.ndim != 2:
        raise ValueError("environment_measure must have shape [B,E].")
    batch, sources, _ = environment_states.shape
    if tuple(environment_coordinates.shape[:2]) != (batch, sources):
        raise ValueError("environment_coordinates must align with environment_states.")
    if tuple(environment_membership.shape[:2]) != (batch, sources):
        raise ValueError("environment_membership must align with environment_states.")
    if tuple(environment_measure.shape) != (batch, sources):
        raise ValueError("environment_measure must align with environment_states.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0 or not torch.isfinite(environment_states).all():
        raise ValueError("environment_states must be finite and epsilon must be positive.")
    if not torch.isfinite(environment_coordinates).all():
        raise ValueError("environment_coordinates must be finite.")
    if not torch.isfinite(environment_membership).all() or bool((environment_membership < 0.0).any()):
        raise ValueError("environment_membership must be finite and nonnegative.")
    if not torch.isfinite(environment_measure).all() or bool((environment_measure < 0.0).any()):
        raise ValueError("environment_measure must be finite and nonnegative.")

    source_mass = environment_measure[..., None] * environment_membership
    mass = source_mass.sum(dim=1)
    denominator = (mass + float(epsilon)).unsqueeze(-1)
    states = torch.einsum("bek,beh->bkh", source_mass, environment_states) / denominator
    centroids = torch.einsum("bek,bed->bkd", source_mass, environment_coordinates) / denominator
    displacement = environment_coordinates[:, :, None, :] - centroids[:, None, :, :]
    radius_numerator = (source_mass * displacement.square().sum(dim=-1)).sum(dim=1)
    radius_sq = radius_numerator / (mass + float(epsilon))
    nonempty = mass > 0.0
    states = torch.where(nonempty[..., None], states, torch.zeros_like(states))
    centroids = torch.where(nonempty[..., None], centroids, torch.zeros_like(centroids))
    radius_sq = torch.where(nonempty, radius_sq, torch.zeros_like(radius_sq))
    return AdaptiveEnvironmentAggregation(
        mass=mass,
        centroids=centroids,
        radius_sq=radius_sq,
        states=states,
        source_mass=source_mass,
    )


# The longer descriptive name is useful at call sites; retain a compact alias
# for focused algebra tests and external evidence utilities.
compute_environment_group_aggregates = mass_weighted_environment_aggregates


def masked_mass_softmax(
    logits: torch.Tensor,
    mass: torch.Tensor,
    *,
    epsilon: float = ADAPTIVE_HYPEREDGE_EPSILON,
) -> torch.Tensor:
    """Form ``p = masked_softmax(logits + log(mass + epsilon))``.

    Zero-mass groups are masked before softmax.  Empty rows return zeros rather
    than a NaN vector, which keeps padded/diagnostic cases finite.
    """

    if logits.ndim < 1 or mass.ndim != 2:
        raise ValueError("logits must end in K and mass must have shape [B,K].")
    if int(logits.shape[-1]) != int(mass.shape[-1]):
        raise ValueError("mass must align with the logits group dimension.")
    if logits.ndim < 2 or int(logits.shape[0]) != int(mass.shape[0]):
        raise ValueError("logits and mass must share the batch dimension.")
    mass_for_logits = mass.reshape(
        int(mass.shape[0]),
        *([1] * (logits.ndim - 2)),
        int(mass.shape[-1]),
    )
    try:
        expanded_mass = torch.broadcast_to(mass_for_logits, logits.shape)
    except RuntimeError as error:
        raise ValueError("mass must broadcast over logits query dimensions.") from error
    valid = expanded_mass > 0.0
    safe_mass = expanded_mass + float(epsilon)
    masked_logits = logits + torch.log(safe_mass.clamp_min(float(epsilon)))
    masked_logits = masked_logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    has_support = valid.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_support, masked_logits, torch.zeros_like(masked_logits))
    probabilities = torch.softmax(safe_logits, dim=-1)
    probabilities = probabilities * valid.to(dtype=probabilities.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )


def opening_blend(
    alpha: torch.Tensor,
    p: torch.Tensor,
    *,
    epsilon: float = ADAPTIVE_HYPEREDGE_EPSILON,
) -> torch.Tensor:
    """Return the continuous fine-opening blend ``clamp(alpha/(p+eps),0,1)``."""

    if alpha.shape != p.shape:
        raise ValueError("alpha and p must have the same shape.")
    return torch.clamp(alpha / (p + float(epsilon)), min=0.0, max=1.0)


def combine_opened_group_responses(
    coarse_response: torch.Tensor,
    fine_response: torch.Tensor,
    p: torch.Tensor,
    alpha: torch.Tensor,
    *,
    epsilon: float = ADAPTIVE_HYPEREDGE_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend and mix group responses, retaining intervention-ready branches.

    ``coarse_response`` and ``fine_response`` have shape ``[B,Q,K,H]``;
    ``p`` and ``alpha`` have shape ``[B,Q,K]``.  The returned tuple is
    ``(total, coarse_contribution, fine_contribution, opening)``.
    """

    if coarse_response.shape != fine_response.shape:
        raise ValueError("coarse_response and fine_response must have matching shapes.")
    if coarse_response.ndim != 4 or p.ndim != 3 or alpha.ndim != 3:
        raise ValueError("Responses must be [B,Q,K,H] and routes must be [B,Q,K].")
    if tuple(coarse_response.shape[:3]) != tuple(p.shape) or p.shape != alpha.shape:
        raise ValueError("Group response and route dimensions must align.")
    opening = opening_blend(alpha, p, epsilon=epsilon)
    coarse_contribution = (p[..., None] * (1.0 - opening[..., None]) * coarse_response).sum(dim=2)
    fine_contribution = (p[..., None] * opening[..., None] * fine_response).sum(dim=2)
    return coarse_contribution + fine_contribution, coarse_contribution, fine_contribution, opening


def _pack_positive_rows(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack variable positive rows into one padded group-major rectangle."""

    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("support mask must have shape [B,S] and boolean dtype.")
    batch = int(mask.shape[0])
    counts = mask.sum(dim=1, dtype=torch.long)
    width = int(counts.max().item()) if batch else 0
    indices = torch.zeros((batch, width), dtype=torch.long, device=mask.device)
    if width == 0:
        return indices, torch.zeros_like(indices, dtype=torch.bool)
    support = torch.nonzero(mask, as_tuple=False)
    ranks = mask.cumsum(dim=1) - 1
    indices[support[:, 0], ranks[support[:, 0], support[:, 1]]] = support[:, 1]
    valid = torch.arange(width, device=mask.device)[None, :] < counts[:, None]
    return indices, valid


def _pack_positive_group_rows(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[B, S, K]`` support rows into one K-batched rectangle.

    The scalar Run-1503 reader packs one ``[B, S]`` mask at a time.  The
    batched executor uses the same stable row order, but packs all groups in
    one pass.  A common padded width is intentional: it lets the fine score
    and value contractions run as one ordinary PyTorch batched GEMM over
    ``[batch, group, head]`` without introducing a custom kernel.
    """

    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("group support mask must have shape [B,S,K] and boolean dtype.")
    batch, rows, groups = (int(value) for value in mask.shape)
    packed, valid = _pack_positive_rows(mask.permute(0, 2, 1).reshape(batch * groups, rows))
    width = int(packed.shape[1])
    return (
        packed.reshape(batch, groups, width),
        valid.reshape(batch, groups, width),
    )


class AdaptiveHyperedgeOpeningPairwiseField(SparseIncidenceGroupControlPairwiseField):
    """Run-1503 grouped environmental reader with the Run-1502 module path."""

    executor_policy = "group_major"
    diagnostic_executor_independent = True
    ledger_rectangular_rows = False

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        group_count: int = 12,
        group_control_dim: int = 16,
        spatial_dim: int = 2,
        module_temperature: float = 1.0,
        environment_temperature: float = 1.0,
        query_temperature: float = 1.0,
        activation_checkpointing: bool = False,
        query_tile_size: int = 128,
        source_tile_size: int = 128,
        environment_refinement_normalizer: str = "sparsemax",
        epsilon: float = ADAPTIVE_HYPEREDGE_EPSILON,
    ) -> None:
        if int(group_count) != 12:
            raise ValueError("adaptive_hyperedge_opening_honf requires group_count=12.")
        if int(group_control_dim) != 16:
            raise ValueError("adaptive_hyperedge_opening_honf requires group_control_dim=16.")
        if str(environment_refinement_normalizer) != "sparsemax":
            raise ValueError(
                "adaptive_hyperedge_opening_honf requires environment_refinement_normalizer='sparsemax'."
            )
        if float(epsilon) <= 0.0 or not torch.isfinite(torch.tensor(float(epsilon))):
            raise ValueError("epsilon must be finite and positive.")
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            group_count=group_count,
            group_control_dim=group_control_dim,
            spatial_dim=spatial_dim,
            module_temperature=module_temperature,
            environment_temperature=environment_temperature,
            query_temperature=query_temperature,
            activation_checkpointing=activation_checkpointing,
            query_tile_size=query_tile_size,
            source_tile_size=source_tile_size,
            environment_refinement_normalizer=environment_refinement_normalizer,
        )
        self.adaptive_epsilon = float(epsilon)

    @staticmethod
    def _raw_environment_values(state: dict[str, Any], *, num_heads: int, head_dim: int) -> torch.Tensor:
        """Recover the inherited raw ``V(e*_j)`` without a second projection."""

        values = state["environment_values"]
        value_gain = state["environment_value_gain"]
        gain = value_gain.reshape(
            int(value_gain.shape[0]),
            int(value_gain.shape[1]),
            int(num_heads),
            int(head_dim),
        ).permute(0, 2, 1, 3)
        return values / gain.clamp_min(ADAPTIVE_HYPEREDGE_EPSILON)

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Reuse Run-1502 preparation and pool only the contextual E sources."""

        state = super().prepare(
            encoded,
            module_states,
            return_routing_maps=bool(return_routing_maps),
        )
        controls = state["group_control_state"]
        if not isinstance(controls, SparseIncidencePreparedGroupControl):
            raise TypeError("Run-1503 requires SparseIncidencePreparedGroupControl state.")
        aggregation = mass_weighted_environment_aggregates(
            state["env_tokens"],
            encoded.env_coords,
            controls.environment_membership,
            controls.environment_measure,
            epsilon=self.adaptive_epsilon,
        )
        raw_values = self._raw_environment_values(
            state,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        _, coarse_values = self.env_attention.project_source(aggregation.states)
        coarse_gain = 1.0 + torch.tanh(
            self.environment_value_control(controls.group_control)
        )
        coarse_gain_heads = coarse_gain.reshape(
            int(coarse_gain.shape[0]),
            int(coarse_gain.shape[1]),
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        coarse_values = coarse_values * coarse_gain_heads
        state.update(
            {
                "adaptive_environment_mass": aggregation.mass,
                "adaptive_environment_centroids": aggregation.centroids,
                "adaptive_environment_radius_sq": aggregation.radius_sq,
                "adaptive_environment_group_states": aggregation.states,
                "adaptive_environment_source_mass": aggregation.source_mass,
                "adaptive_environment_keys": state["environment_keys"],
                "adaptive_environment_raw_values": raw_values,
                "adaptive_coarse_values": coarse_values,
                "adaptive_coarse_value_gain": coarse_gain,
                "adaptive_preparation_aux": {
                    "group_control_adaptive_environment_mass": aggregation.mass.detach(),
                    "group_control_adaptive_environment_centroids": aggregation.centroids.detach(),
                    "group_control_adaptive_environment_radius_sq": aggregation.radius_sq.detach(),
                    "group_control_adaptive_environment_group_states": aggregation.states.detach(),
                    "group_control_adaptive_environment_source_projection_rows": module_states.new_full(
                        (int(module_states.shape[0]),), float(encoded.env_coords.shape[1])
                    ).detach(),
                    "group_control_adaptive_coarse_source_projection_rows": module_states.new_full(
                        (int(module_states.shape[0]),), float(aggregation.mass.shape[1])
                    ).detach(),
                },
            }
        )
        return state

    def preparation_aux(
        self,
        state: dict[str, Any],
        *,
        include_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        summary = super().preparation_aux(state, include_diagnostics=include_diagnostics)
        summary.update(state.get("adaptive_preparation_aux", {}))
        if include_diagnostics:
            summary.update(
                {
                    "group_control_adaptive_environment_source_mass": state[
                        "adaptive_environment_source_mass"
                    ].detach(),
                    "group_control_adaptive_environment_group_states": state[
                        "adaptive_environment_group_states"
                    ].detach(),
                    "group_control_adaptive_environment_keys": state[
                        "adaptive_environment_keys"
                    ].detach(),
                }
            )
        return summary

    def _coarse_group_response(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        controls: SparseIncidencePreparedGroupControl,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``G_qk`` from the controlled aggregate physical value.

        The inherited query router already forms the sole query/group logit
        bank ``ell_qk`` from projected query state, group state/centroid
        geometry, and ``h_k``.  Reuse those exact logits as a bounded,
        non-normalized per-group interaction gate rather than constructing a
        second Q/K score bank:

        ``G_qk = O((1 + tanh(ell_qk) / sqrt(head_dim)) V(bar_e_k; h_k))``.

        ``p`` remains the only query/group routing distribution.  The value
        is still the mass-weighted aggregate physical state, projected once by
        the inherited source/value/output path.
        """

        del receiver_features, controls, encoded
        values = state["adaptive_coarse_values"]
        gate = 1.0 + torch.tanh(route_logits) / (float(self.head_dim) ** 0.5)
        group_values = values.permute(0, 2, 1, 3)
        gated_values = group_values[:, None, :, :, :] * gate[..., None, None]
        return self.env_attention.output(
            gated_values.reshape(
                int(values.shape[0]),
                int(route_logits.shape[1]),
                int(values.shape[2]),
                self.hidden_dim,
            )
        )

    def _fine_group_response_block(
        self,
        query: torch.Tensor,
        environment_keys: torch.Tensor,
        environment_raw_values: torch.Tensor,
        environment_coordinates: torch.Tensor,
        coordinate_scale: torch.Tensor,
        receivers: torch.Tensor,
        query_indices: torch.Tensor,
        query_valid: torch.Tensor,
        source_indices: torch.Tensor,
        source_valid: torch.Tensor,
        source_mass: torch.Tensor,
        group_control: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one padded fine rectangle.

        The complete group calculation is the activation-checkpoint boundary.
        In particular, the score, geometry, masked-softmax, and value
        contraction tensors are recomputed in backward instead of remaining
        live for every group in a receiver tile.  Only the compact scattered
        ``[B,Q,H]`` group response is kept by the forward graph.
        """

        batch = int(receivers.shape[0])
        padded_queries = int(query_indices.shape[1])
        padded_sources = int(source_indices.shape[1])
        query_gather = query_indices[:, None, :, None].expand(
            batch, self.num_heads, padded_queries, self.head_dim
        )
        source_gather = source_indices[:, None, :, None].expand(
            batch, self.num_heads, padded_sources, self.head_dim
        )
        query_block = torch.gather(query, 2, query_gather)
        key_block = torch.gather(environment_keys, 2, source_gather)
        value_block = torch.gather(environment_raw_values, 2, source_gather)
        group_gain = 1.0 + torch.tanh(self.environment_value_control(group_control))
        group_gain = group_gain.reshape(batch, self.num_heads, self.head_dim)
        value_block = value_block * group_gain[:, :, None, :]

        query_coordinates = torch.gather(
            receivers,
            1,
            query_indices[..., None].expand(batch, padded_queries, self.spatial_dim),
        )
        source_coordinates = torch.gather(
            environment_coordinates,
            1,
            source_indices[..., None].expand(batch, padded_sources, self.spatial_dim),
        )
        scale = coordinate_scale
        if scale.ndim == 1:
            scale = scale[None, None, None, :]
        elif scale.ndim == 2:
            scale = scale[:, None, None, :]
        elif scale.ndim == 3:
            scale = scale[:, :, None, :]
        relative = (query_coordinates[:, :, None, :] - source_coordinates[:, None, :, :]) / scale
        geometry = self._mlp(
            self.env_geometry_bias,
            self.relative_fourier(relative),
        ).permute(0, 3, 1, 2)
        scores = torch.matmul(query_block, key_block.transpose(-1, -2)) / (float(self.head_dim) ** 0.5)
        score_control = self.environment_score_control(group_control).reshape(
            batch, self.num_heads, 1, 1
        )
        scores = scores * (1.0 + torch.tanh(score_control)) + geometry
        selected_mass = torch.gather(source_mass, 1, source_indices)
        selected_mass = torch.where(source_valid, selected_mass, torch.ones_like(selected_mass))
        scores = scores + torch.log(selected_mass.clamp_min(self.adaptive_epsilon))[:, None, None, :]
        valid = query_valid[:, None, :, None] & source_valid[:, None, None, :]
        masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        row_has_support = valid.any(dim=-1, keepdim=True)
        safe_scores = torch.where(row_has_support, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * valid.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        response = torch.matmul(weights, value_block).transpose(1, 2).reshape(
            batch, padded_queries, self.hidden_dim
        )
        group_gate = 1.0 + torch.tanh(route_logits) / (float(self.head_dim) ** 0.5)
        response = response * group_gate[..., None]
        response = self.env_attention.output(response)
        fine = receivers.new_zeros((batch, int(receivers.shape[1]), self.hidden_dim))
        fine.scatter_add_(
            1,
            query_indices[..., None].expand(batch, padded_queries, self.hidden_dim),
            response * query_valid[..., None].to(dtype=response.dtype),
        )
        return fine

    def _fine_group_response(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        query: torch.Tensor,
        controls: SparseIncidencePreparedGroupControl,
        alpha: torch.Tensor,
        route_logits: torch.Tensor,
        group_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read one positive-support group as one padded dense rectangle."""

        source_mass = state["adaptive_environment_source_mass"][..., group_index]
        query_indices, query_valid = _pack_positive_rows(alpha[..., group_index] > 0.0)
        source_indices, source_valid = _pack_positive_rows(source_mass > 0.0)
        batch = int(receivers.shape[0])
        query_count = int(receivers.shape[1])
        if int(query_indices.shape[1]) == 0 or int(source_indices.shape[1]) == 0:
            return (
                receivers.new_zeros((batch, query_count, self.hidden_dim)),
                query_valid.sum(dim=1).to(receivers.dtype),
                source_valid.sum(dim=1).to(receivers.dtype),
                query_valid,
                source_valid,
            )

        block_args = (
            query,
            state["adaptive_environment_keys"],
            state["adaptive_environment_raw_values"],
            encoded.env_coords,
            encoded.coordinate_scale,
            receivers,
            query_indices,
            query_valid,
            source_indices,
            source_valid,
            source_mass,
            controls.group_control[:, group_index, :],
            torch.gather(route_logits[..., group_index], 1, query_indices),
        )
        if self._checkpoint_active():
            fine = checkpoint(
                lambda *values_: self._fine_group_response_block(*values_),
                *block_args,
                use_reentrant=False,
            )
        else:
            fine = self._fine_group_response_block(*block_args)
        return (
            fine,
            query_valid.sum(dim=1).to(receivers.dtype),
            source_valid.sum(dim=1).to(receivers.dtype),
            query_valid,
            source_valid,
        )

    def _fine_group_response_batched_block(
        self,
        query: torch.Tensor,
        environment_keys: torch.Tensor,
        environment_raw_values: torch.Tensor,
        environment_coordinates: torch.Tensor,
        coordinate_scale: torch.Tensor,
        receivers: torch.Tensor,
        query_indices: torch.Tensor,
        query_valid: torch.Tensor,
        source_indices: torch.Tensor,
        source_valid: torch.Tensor,
        source_mass: torch.Tensor,
        group_control: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate every positive-support group in one padded batched read.

        The leading dimensions are ``[B,K,H]`` for the attention tensors and
        ``[B,K]`` for the packed row maps.  The source and query projections
        are therefore shared across the group axis; only the group-local
        score/value controls are expanded.  A single flattened scatter puts
        the ``K`` responses back into ``[B,Q,K,H]``.  This is the hot path for
        Run-1503; :meth:`_fine_group_response` remains the scalar reference
        used by CPU parity tests.
        """

        batch = int(receivers.shape[0])
        groups = int(query_indices.shape[1])
        padded_queries = int(query_indices.shape[2])
        padded_sources = int(source_indices.shape[2])
        if padded_queries == 0 or padded_sources == 0:
            return receivers.new_zeros(
                (batch, int(receivers.shape[1]), groups, self.hidden_dim)
            )

        # Gather the already projected source/query banks without repeating
        # either affine projection across K.  The [B,K,H] layout is chosen so
        # torch.matmul sees one ordinary batched GEMM for all groups/heads.
        query_bank = query[:, None, :, :, :].expand(
            batch, groups, self.num_heads, int(query.shape[2]), self.head_dim
        )
        query_gather = query_indices[:, :, None, :, None].expand(
            batch, groups, self.num_heads, padded_queries, self.head_dim
        )
        query_block = torch.gather(query_bank, 3, query_gather)

        key_bank = environment_keys[:, None, :, :, :].expand(
            batch, groups, self.num_heads, int(environment_keys.shape[2]), self.head_dim
        )
        value_bank = environment_raw_values[:, None, :, :, :].expand(
            batch, groups, self.num_heads, int(environment_raw_values.shape[2]), self.head_dim
        )
        source_gather = source_indices[:, :, None, :, None].expand(
            batch, groups, self.num_heads, padded_sources, self.head_dim
        )
        key_block = torch.gather(key_bank, 3, source_gather)
        value_block = torch.gather(value_bank, 3, source_gather)

        group_gain = 1.0 + torch.tanh(
            self.environment_value_control(group_control.reshape(batch * groups, -1))
        )
        group_gain = group_gain.reshape(batch, groups, self.num_heads, self.head_dim)
        value_block = value_block * group_gain[:, :, :, None, :]

        receiver_bank = receivers[:, None, :, :].expand(
            batch, groups, int(receivers.shape[1]), self.spatial_dim
        )
        query_coordinates = torch.gather(
            receiver_bank,
            2,
            query_indices[..., None].expand(batch, groups, padded_queries, self.spatial_dim),
        )
        source_bank = environment_coordinates[:, None, :, :].expand(
            batch, groups, int(environment_coordinates.shape[1]), self.spatial_dim
        )
        source_coordinates = torch.gather(
            source_bank,
            2,
            source_indices[..., None].expand(batch, groups, padded_sources, self.spatial_dim),
        )
        scale = coordinate_scale
        if scale.ndim == 1:
            scale = scale[None, None, None, None, :]
        elif scale.ndim == 2:
            scale = scale[:, None, None, None, :]
        elif scale.ndim == 3:
            scale = scale[:, :, None, None, :]
        else:
            raise ValueError("coordinate_scale must have shape [d], [B,d], or [B,1,d].")
        relative = (
            query_coordinates[:, :, :, None, :] - source_coordinates[:, :, None, :, :]
        ) / scale
        geometry = self._mlp(
            self.env_geometry_bias,
            self.relative_fourier(relative),
        ).permute(0, 1, 4, 2, 3)

        scores = torch.matmul(
            query_block,
            key_block.transpose(-1, -2),
        ) / (float(self.head_dim) ** 0.5)
        score_control = self.environment_score_control(
            group_control.reshape(batch * groups, -1)
        ).reshape(batch, groups, self.num_heads, 1, 1)
        scores = scores * (1.0 + torch.tanh(score_control)) + geometry

        source_mass_by_group = source_mass.permute(0, 2, 1)
        selected_mass = torch.gather(source_mass_by_group, 2, source_indices)
        selected_mass = torch.where(
            source_valid,
            selected_mass,
            torch.ones_like(selected_mass),
        )
        scores = scores + torch.log(
            selected_mass.clamp_min(self.adaptive_epsilon)
        )[:, :, None, None, :]
        valid = query_valid[:, :, :, None] & source_valid[:, :, None, :]
        masked_scores = scores.masked_fill(~valid[:, :, None, :, :], torch.finfo(scores.dtype).min)
        row_has_support = valid.any(dim=-1, keepdim=True)
        safe_scores = torch.where(
            row_has_support[:, :, None, :, :],
            masked_scores,
            torch.zeros_like(masked_scores),
        )
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * valid[:, :, None, :, :].to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        response = torch.matmul(weights, value_block).permute(0, 1, 3, 2, 4)
        response = response.reshape(batch, groups, padded_queries, self.hidden_dim)
        group_gate = 1.0 + torch.tanh(route_logits) / (float(self.head_dim) ** 0.5)
        response = response * group_gate[..., None]

        # ``output`` is one shared affine.  Flattening B*K keeps it as one
        # call while retaining the exact per-group output transformation.
        response = self.env_attention.output(
            response.reshape(batch * groups, padded_queries, self.hidden_dim)
        ).reshape(batch, groups, padded_queries, self.hidden_dim)

        # Scatter all groups in one reduction.  Invalid padded rows carry zero
        # response, so their harmless index 0 never changes a valid result.
        group_offsets = torch.arange(groups, device=receivers.device)[None, :, None]
        target = (query_indices * groups + group_offsets).reshape(batch, groups * padded_queries)
        values = (
            response * query_valid[..., None].to(dtype=response.dtype)
        ).reshape(batch, groups * padded_queries, self.hidden_dim)
        fine = receivers.new_zeros(
            (batch, int(receivers.shape[1]) * groups, self.hidden_dim)
        )
        fine.scatter_add_(
            1,
            target[..., None].expand(batch, groups * padded_queries, self.hidden_dim),
            values,
        )
        return fine.reshape(batch, int(receivers.shape[1]), groups, self.hidden_dim)

    def _fine_group_responses_batched(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        query: torch.Tensor,
        controls: SparseIncidencePreparedGroupControl,
        alpha: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Prepare one K-batched fine rectangle and its accounting maps."""

        source_mass = state["adaptive_environment_source_mass"]
        query_indices, query_valid = _pack_positive_group_rows(alpha > 0.0)
        source_indices, source_valid = _pack_positive_group_rows(source_mass > 0.0)
        query_counts = query_valid.sum(dim=-1).to(receivers.dtype)
        source_counts = source_valid.sum(dim=-1).to(receivers.dtype)
        active_groups = (query_counts > 0.0) & (source_counts > 0.0)
        active_count = active_groups.any()
        if bool(active_count):
            block_args = (
                query,
                state["adaptive_environment_keys"],
                state["adaptive_environment_raw_values"],
                encoded.env_coords,
                encoded.coordinate_scale,
                receivers,
                query_indices,
                query_valid,
                source_indices,
                source_valid,
                source_mass,
                controls.group_control,
                torch.gather(route_logits.transpose(1, 2), 2, query_indices),
            )
            if self._checkpoint_active():
                fine = checkpoint(
                    lambda *values_: self._fine_group_response_batched_block(*values_),
                    *block_args,
                    use_reentrant=False,
                )
            else:
                fine = self._fine_group_response_batched_block(*block_args)
        else:
            fine = receivers.new_zeros(
                (int(receivers.shape[0]), int(receivers.shape[1]), self.group_count, self.hidden_dim)
            )

        # Keep the historical per-group scalar-K row ledger unchanged.  The
        # new batched executor also emits its true common-padding area so a
        # report can show the memory/work tradeoff explicitly.
        query_width = query_counts.max(dim=0).values
        source_width = source_counts.max(dim=0).values
        scalar_rows = query_width[None, :] * source_width[None, :]
        scalar_rows = scalar_rows.expand(int(receivers.shape[0]), -1)
        batched_rows = receivers.new_full(
            (int(receivers.shape[0]), self.group_count),
            float(query_indices.shape[-1] * source_indices.shape[-1]),
        )
        if not bool(active_count):
            batched_rows.zero_()
        scalar_calls = active_groups.any(dim=0).sum().to(receivers.dtype)
        batched_calls = active_count.to(receivers.dtype)
        diagnostics = {
            "scalar_block_calls": scalar_calls,
            "batched_block_calls": batched_calls,
            "block_call_reduction": scalar_calls - batched_calls,
            "scalar_gemm_launches": scalar_calls,
            "batched_gemm_launches": batched_calls,
            "gemm_launch_reduction": scalar_calls - batched_calls,
            "scalar_rows": scalar_rows,
            "batched_rows": batched_rows,
            "batched_checkpoint_calls": (
                batched_calls if self._checkpoint_active() else receivers.new_zeros(())
            ),
            "batched_rows_recompute": (
                batched_rows.sum()
                if self._checkpoint_active()
                else receivers.new_zeros(())
            ),
        }
        return (
            fine,
            query_counts,
            source_counts,
            query_valid,
            source_valid,
            diagnostics,
        )

    def _read_environment(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: Any,
        *,
        include_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._read_environment_impl(
            state,
            encoded,
            receivers,
            receiver_features,
            route,
            include_diagnostics=include_diagnostics,
            use_batched_executor=(
                int(receivers.shape[1]) <= ADAPTIVE_HYPEREDGE_BATCHED_QUERY_LIMIT
            ),
        )

    def _read_environment_scalar_reference(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: Any,
        *,
        include_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reference reader retaining the pre-optimization scalar-K loop.

        This is intentionally private and is not selected by any config.  It
        gives the CPU parity tests a direct old/new comparison without
        changing the Run-1503 checkpoint or experiment identity.
        """

        return self._read_environment_impl(
            state,
            encoded,
            receivers,
            receiver_features,
            route,
            include_diagnostics=include_diagnostics,
            use_batched_executor=False,
        )

    def _read_environment_impl(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        route: Any,
        *,
        include_diagnostics: bool,
        use_batched_executor: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        controls = state["group_control_state"]
        if not isinstance(controls, SparseIncidencePreparedGroupControl):
            raise TypeError("Run-1503 requires SparseIncidencePreparedGroupControl state.")
        alpha = route.assignment
        p = masked_mass_softmax(
            route.logits,
            state["adaptive_environment_mass"],
            epsilon=self.adaptive_epsilon,
        )
        opening = opening_blend(alpha, p, epsilon=self.adaptive_epsilon)
        query = self.env_attention.project_query(self.env_query(receiver_features))
        coarse = self._coarse_group_response(
            state,
            encoded,
            receivers,
            receiver_features,
            controls,
            route.logits,
        )

        if use_batched_executor:
            (
                fine,
                query_count_per_group,
                source_count_per_group,
                query_valid,
                source_valid,
                fine_executor_diagnostics,
            ) = self._fine_group_responses_batched(
                state,
                encoded,
                receivers,
                query,
                controls,
                alpha,
                route.logits,
            )
            query_valid_maps = [query_valid[:, group_index] for group_index in range(self.group_count)]
            source_valid_maps = [source_valid[:, group_index] for group_index in range(self.group_count)]
        else:
            fine_groups: list[torch.Tensor] = []
            query_counts: list[torch.Tensor] = []
            source_counts: list[torch.Tensor] = []
            query_valid_maps = []
            source_valid_maps = []
            for group_index in range(self.group_count):
                fine, q_count, e_count, query_valid, source_valid = self._fine_group_response(
                    state,
                    encoded,
                    receivers,
                    query,
                    controls,
                    alpha,
                    route.logits,
                    group_index,
                )
                fine_groups.append(fine)
                query_counts.append(q_count)
                source_counts.append(e_count)
                query_valid_maps.append(query_valid)
                source_valid_maps.append(source_valid)
            fine = torch.stack(fine_groups, dim=2)
            query_count_per_group = torch.stack(query_counts, dim=1)
            source_count_per_group = torch.stack(source_counts, dim=1)
            active_groups = (query_count_per_group > 0.0) & (source_count_per_group > 0.0)
            scalar_calls = active_groups.any(dim=0).sum().to(receivers.dtype)
            fine_executor_diagnostics = {
                "scalar_block_calls": scalar_calls,
                "batched_block_calls": receivers.new_zeros(()),
                "block_call_reduction": receivers.new_zeros(()),
                "scalar_gemm_launches": scalar_calls,
                "batched_gemm_launches": receivers.new_zeros(()),
                "gemm_launch_reduction": receivers.new_zeros(()),
                "scalar_rows": torch.stack(
                    [
                        receivers.new_full(
                            (int(receivers.shape[0]),),
                            float(query_valid.shape[1] * source_valid.shape[1]),
                        )
                        for query_valid, source_valid in zip(
                            query_valid_maps, source_valid_maps, strict=True
                        )
                    ],
                    dim=1,
                ),
                "batched_rows": receivers.new_zeros(
                    (int(receivers.shape[0]), self.group_count)
                ),
                "batched_checkpoint_calls": receivers.new_zeros(()),
                "batched_rows_recompute": receivers.new_zeros(()),
            }
        total, coarse_contribution, fine_contribution, opening = combine_opened_group_responses(
            coarse,
            fine,
            p,
            alpha,
            epsilon=self.adaptive_epsilon,
        )
        # Keep the logical per-case support area Q_k E_k distinct from the
        # packed group GEMM area.  The latter uses maximum positive widths in
        # the batch, so its actual forward area is Q_pad,k E_pad,k for every
        # batch item.
        fine_group_rows_logical = query_count_per_group * source_count_per_group
        fine_group_rows_forward = fine_executor_diagnostics["scalar_rows"]
        fine_group_rows_padded = fine_group_rows_forward - fine_group_rows_logical
        opened = (alpha > 0.0).to(dtype=receivers.dtype)
        fine_rows_per_query = torch.einsum(
            "bqk,bk->bq",
            opened,
            source_count_per_group,
        )
        logical_paths = self._logical_paths(alpha, controls.environment_membership)
        physical_support = (logical_paths > 0.0).to(dtype=receivers.dtype)
        unique_pairs_per_query = physical_support.sum(dim=-1)
        coarse_group_rows = receivers.new_full(
            (int(receivers.shape[0]),),
            float(receivers.shape[1] * self.group_count),
        )
        full_rows = receivers.new_full(
            (int(receivers.shape[0]),),
            float(receivers.shape[1] * encoded.env_coords.shape[1]),
        )
        fine_rows_logical = fine_group_rows_logical.sum(dim=1)
        fine_rows_forward = fine_group_rows_forward.sum(dim=1)
        fine_rows_padded = fine_group_rows_padded.sum(dim=1)
        aux: dict[str, torch.Tensor] = {
            "group_control_environment_overlap_mass_per_query": p.sum(dim=-1),
            "group_control_environment_unique_pairs_per_query": unique_pairs_per_query,
            "group_control_environment_logical_paths_per_query": logical_paths.sum(dim=-1),
            "group_control_environment_unique_pairs": unique_pairs_per_query.sum(),
            "group_control_environment_logical_paths": logical_paths.sum(),
            "group_control_environment_support_rows": unique_pairs_per_query.sum(),
            "group_control_environment_fine_rows": fine_rows_forward.sum(),
            "group_control_environment_fine_rows_forward": fine_rows_forward.sum(),
            "group_control_environment_fine_rows_padded": fine_rows_padded.sum(),
            "group_control_environment_fine_forward_rows": fine_rows_forward.sum(),
            "group_control_environment_padded_rows": fine_rows_padded.sum(),
            "group_control_environment_valid_pair_denominator": full_rows.sum(),
            "group_control_environment_padded_pair_denominator": receivers.new_zeros(()),
            "group_control_environment_geometry_rows_forward": fine_rows_forward.sum(),
            "group_control_environment_content_dot_rows_forward": fine_rows_forward.sum()
            * float(self.num_heads),
            "group_control_environment_scalar_control_rows": receivers.new_tensor(
                float(receivers.numel() / max(int(receivers.shape[-1]), 1))
            ),
            "group_control_environment_checkpoint_recomputations": (
                fine_rows_forward.sum()
                if self._checkpoint_active()
                else receivers.new_zeros(())
            ),
            "group_control_environment_fine_rows_recompute": (
                fine_rows_forward.sum()
                if self._checkpoint_active()
                else receivers.new_zeros(())
            ),
            "group_control_environment_complete_support": receivers.new_zeros(()),
            "group_control_environment_partial_support": receivers.new_ones(()),
            "group_control_environment_executor_selected": receivers.new_ones(()),
            "group_control_adaptive_query_count_per_group": query_count_per_group,
            "group_control_adaptive_source_count_per_group": source_count_per_group,
            "group_control_adaptive_fine_group_rows": fine_group_rows_forward,
            "group_control_adaptive_fine_group_rows_logical": fine_group_rows_logical,
            "group_control_adaptive_fine_group_rows_forward": fine_group_rows_forward,
            "group_control_adaptive_fine_group_rows_padded": fine_group_rows_padded,
            "group_control_adaptive_fine_rows_per_query": fine_rows_per_query,
            "group_control_adaptive_fine_rows_logical": fine_rows_logical.sum(),
            "group_control_adaptive_fine_rows": fine_rows_forward.sum(),
            "group_control_adaptive_fine_rows_forward": fine_rows_forward.sum(),
            "group_control_adaptive_fine_rows_padded": fine_rows_padded.sum(),
            "group_control_adaptive_coarse_rows": coarse_group_rows.sum(),
            "group_control_adaptive_full_rectangle_rows": full_rows.sum(),
            "group_control_adaptive_fine_work_ratio": fine_rows_forward.sum()
            / full_rows.sum().clamp_min(1.0),
            "group_control_adaptive_fine_scalar_block_calls": fine_executor_diagnostics[
                "scalar_block_calls"
            ],
            "group_control_adaptive_fine_batched_block_calls": fine_executor_diagnostics[
                "batched_block_calls"
            ],
            "group_control_adaptive_fine_block_call_reduction": fine_executor_diagnostics[
                "block_call_reduction"
            ],
            "group_control_adaptive_fine_scalar_gemm_launches": fine_executor_diagnostics[
                "scalar_gemm_launches"
            ],
            "group_control_adaptive_fine_batched_gemm_launches": fine_executor_diagnostics[
                "batched_gemm_launches"
            ],
            "group_control_adaptive_fine_gemm_launch_reduction": fine_executor_diagnostics[
                "gemm_launch_reduction"
            ],
            "group_control_adaptive_fine_batched_checkpoint_calls": fine_executor_diagnostics[
                "batched_checkpoint_calls"
            ],
            # One value per receiver makes mixed-size outer reads explicit:
            # the last short tile may use the batched executor even when an
            # earlier large tile uses the scalar executor.
            "group_control_adaptive_fine_executor_selected": receivers.new_full(
                (int(receivers.shape[0]), int(receivers.shape[1])),
                float(use_batched_executor),
            ),
            "group_control_adaptive_fine_executor_batch_limit": receivers.new_tensor(
                float(ADAPTIVE_HYPEREDGE_BATCHED_QUERY_LIMIT)
            ),
            "group_control_adaptive_batched_fine_group_rows_forward": fine_executor_diagnostics[
                "batched_rows"
            ],
            "group_control_adaptive_batched_fine_rows": fine_executor_diagnostics[
                "batched_rows"
            ].sum(),
            "group_control_adaptive_batched_fine_rows_padded": (
                fine_executor_diagnostics["batched_rows"] - fine_group_rows_logical
            ).sum()
            if use_batched_executor
            else receivers.new_zeros(()),
            "group_control_adaptive_batched_fine_rows_recompute": fine_executor_diagnostics[
                "batched_rows_recompute"
            ],
            "group_control_adaptive_coarse_contribution": coarse_contribution,
            "group_control_adaptive_fine_contribution": fine_contribution,
        }
        if include_diagnostics:
            aux.update(
                {
                    "group_control_adaptive_environment_p": p,
                    "group_control_adaptive_environment_alpha": alpha,
                    "group_control_adaptive_opening_blend": opening,
                    "group_control_adaptive_coarse_group_response": coarse,
                    "group_control_adaptive_fine_group_response": fine,
                    "group_control_adaptive_environment_logits": route.logits,
                }
            )
        return total, aux

    def read_environment(
        self,
        state: dict[str, Any],
        encoded: EncodedInterfaceCase,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        route: Any | None = None,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Expose the grouped environmental term through the historical API."""

        if route is None:
            route = self._route(state, encoded, receivers, receiver_features)
        context, _ = self._read_environment(
            state,
            encoded,
            receivers,
            receiver_features,
            route,
            include_diagnostics=bool(return_routing_maps),
        )
        return context, None


# Naming aliases make the architecture discoverable without changing the
# registry-facing class name used by the factory.
AdaptiveHyperedgeOpeningField = AdaptiveHyperedgeOpeningPairwiseField
AdaptiveHyperedgeOpeningHONF = AdaptiveHyperedgeOpeningPairwiseField


__all__ = [
    "ADAPTIVE_HYPEREDGE_BATCHED_QUERY_LIMIT",
    "ADAPTIVE_HYPEREDGE_EPSILON",
    "AdaptiveEnvironmentAggregation",
    "AdaptiveHyperedgeOpeningField",
    "AdaptiveHyperedgeOpeningHONF",
    "AdaptiveHyperedgeOpeningPairwiseField",
    "combine_opened_group_responses",
    "compute_environment_group_aggregates",
    "masked_mass_softmax",
    "mass_weighted_environment_aggregates",
    "opening_blend",
]
