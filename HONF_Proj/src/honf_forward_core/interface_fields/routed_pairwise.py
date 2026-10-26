"""Sparse routed execution with Dense's fine source preparation.

The router in this module is deliberately a scalar index.  Candidate hub
descriptors and the resulting incidences are used to decide which receiver
and source pairs are instantiated; they are never passed to the field head as
source values.  The contextual module/environment states and the nonlinear
Dense readers remain the only fine physical values.

The implementation keeps the expensive pair readers gathered.  It computes
source memberships and query densities first, joins positive two-hop paths,
coalesces repeated ``(batch, receiver, source)`` keys, and only then evaluates
the inherited QM/QE networks.  All floating point route values remain live in
the autograd graph; integer support and sort decisions are metadata only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from .dense_pairwise import DensePairwiseField
from .routing_index.geometry import (
    geometry_features,
    geometry_length_scale,
    geometry_resistance,
    routing_affinity,
)
from .routing_index.pair_join import compile_two_hop_pairs
from .routing_index.router import RoutedRoutingRouter, fixed_data_mean_shift
from .routing_index.sparse_projection import (
    build_typed_source_incidence,
    environment_source_measure,
    module_source_measure,
    ordinary_source_sparsemax,
    source_measure_sparsemax,
)
from .routing_index.types import (
    PackedPairs,
    PreparedRoutingIndex,
    RoutingCandidates,
    TypedSourceIncidence,
)


def _option(options: Any, name: str, default: Any) -> Any:
    if options is None:
        return default
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def _as_bool_mask(values: torch.Tensor) -> torch.Tensor:
    return values if values.dtype == torch.bool else values > 0.5


def compile_positive_pairs(
    density: torch.Tensor,
    incidence: TypedSourceIncidence,
) -> PackedPairs:
    """Join positive query densities with positive source incidences.

    The join is performed by integer ``(batch, hub)`` keys.  The path values
    are gathered from the live source membership/density tensors and are then
    coalesced by integer ``(batch, receiver, source)`` keys.  Thus a pair that
    shares multiple hubs invokes a fine reader once and receives the sum of
    all its two-hop priors.
    """

    return compile_two_hop_pairs(density, incidence)


def _scatter_sum_rows(values: torch.Tensor, pairs: PackedPairs, batch: int, receivers: int) -> torch.Tensor:
    """Reduce live weighted rows into ``[B,Q,...]`` without query loops."""

    output = values.new_zeros((int(batch) * int(receivers), *values.shape[1:]))
    if int(values.shape[0]) == 0:
        return output.reshape(int(batch), int(receivers), *values.shape[1:])
    keys = pairs.batch_index * int(receivers) + pairs.receiver_index
    prior = pairs.prior.to(dtype=values.dtype)
    output.index_add_(0, keys, values * prior.reshape((-1,) + (1,) * (values.ndim - 1)))
    return output.reshape(int(batch), int(receivers), *values.shape[1:])


def _segment_softmax(scores: torch.Tensor, group: torch.Tensor, group_count: int) -> torch.Tensor:
    """Stable softmax over packed rows sharing an integer receiver key."""

    if scores.ndim != 2:
        raise ValueError("Packed scores must have shape [I,heads].")
    if int(scores.shape[0]) == 0:
        return scores
    heads = int(scores.shape[1])
    maxima = scores.new_full((int(group_count), heads), -torch.inf)
    expanded_group = group[:, None].expand(-1, heads)
    maxima.scatter_reduce_(0, expanded_group, scores, reduce="amax", include_self=True)
    exponent = torch.exp(scores - maxima.index_select(0, group))
    denominator = scores.new_zeros((int(group_count), heads))
    denominator.index_add_(0, group, exponent)
    return exponent / denominator.index_select(0, group).clamp_min(torch.finfo(scores.dtype).tiny)


class RoutedPairwiseField(DensePairwiseField):
    """Dense preparation plus sparse two-hop dispatch of fine QM and QE."""

    def __init__(
        self,
        hidden_dim: int,
        message_hidden_dim: int,
        num_heads: int,
        fourier_frequencies: int,
        *,
        routing_config: Any = None,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            activation_checkpointing=activation_checkpointing,
        )
        self.routing_strategy = str(_option(routing_config, "strategy", "module_hubs"))
        if self.routing_strategy not in {"module_hubs", "mean_shift"}:
            raise ValueError(
                "routed_pairwise_honf supports routing.strategy='module_hubs' or 'mean_shift'."
            )
        self.routing_descriptor_dim = int(_option(routing_config, "descriptor_dim", 32))
        self.routing_hidden_dim = int(_option(routing_config, "router_hidden_dim", 64))
        self.routing_temperature = float(_option(routing_config, "temperature", 1.0))
        self.routing_content_scale = float(_option(routing_config, "content_scale", 2.0))
        self.routing_geometry_scale = float(_option(routing_config, "geometry_scale", 0.25))
        self.routing_propensity_scale = float(_option(routing_config, "propensity_scale", 0.25))
        self.fine_pair_chunk_size = int(_option(routing_config, "fine_pair_chunk_size", 16384))
        self.mean_shift_steps = int(_option(routing_config, "mean_shift_steps", 3))
        self.mean_shift_feature_bandwidth = float(
            _option(routing_config, "mean_shift_feature_bandwidth", 1.0)
        )
        if self.routing_temperature <= 0.0:
            raise ValueError("routing temperature must be positive.")
        if self.fine_pair_chunk_size <= 0:
            raise ValueError("routing fine_pair_chunk_size must be positive.")
        if self.routing_strategy == "mean_shift" and not 0 <= self.mean_shift_steps <= 3:
            # The typed configuration/profile requires exactly three steps;
            # direct backend construction permits T=0..2 numerical fixtures
            # that verify the fixed-data update independently.
            raise ValueError("routing mean-shift steps must be between 0 and 3.")
        if self.mean_shift_feature_bandwidth <= 0.0:
            raise ValueError("routing mean-shift feature bandwidth must be positive.")
        self.router = RoutedRoutingRouter(
            hidden_dim,
            descriptor_dim=self.routing_descriptor_dim,
            hidden_dim=self.routing_hidden_dim,
        )

    def _geometry(self, encoded: Any, points: torch.Tensor, *, fallback_features: torch.Tensor | None = None) -> torch.Tensor:
        provider = getattr(encoded, "routing_geometry", None)
        if provider is not None:
            return geometry_features(provider, points)
        if fallback_features is not None:
            values = fallback_features
            if values.ndim == 2:
                values = values.unsqueeze(0).expand(points.shape[0], -1, -1)
            if tuple(values.shape[:-1]) != tuple(points.shape[:-1]):
                raise ValueError("Fallback routing geometry features must align with points.")
            return values.to(device=points.device, dtype=points.dtype)
        scale = encoded.coordinate_scale.reshape(-1).to(device=points.device, dtype=points.dtype)
        return points / scale.reshape(*([1] * (points.ndim - 1)), -1).clamp_min(1.0e-8)

    def _length_scale(self, encoded: Any, points: torch.Tensor) -> torch.Tensor:
        provider = getattr(encoded, "routing_geometry", None)
        fallback = encoded.coordinate_scale.reshape(-1)
        return geometry_length_scale(provider, points, fallback=fallback)

    def _resistance(self, encoded: Any, starts: torch.Tensor, ends: torch.Tensor, relation_type: str) -> torch.Tensor:
        provider = getattr(encoded, "routing_geometry", None)
        return geometry_resistance(provider, starts, ends, relation_type=relation_type)

    def _descriptor_module(self, module_tokens: torch.Tensor, module_geometry: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
        return self.router.module_map(module_tokens, module_geometry, global_token)

    def _descriptor_environment(self, env_tokens: torch.Tensor, env_geometry: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
        return self.router.environment_map(env_tokens, env_geometry, global_token)

    def _descriptor_query(self, receiver_features: torch.Tensor, query_geometry: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
        return self.router.query_map(receiver_features, query_geometry, global_token)

    def _mean_shift_candidates(
        self,
        encoded: Any,
        module_centers: torch.Tensor,
        module_descriptors: torch.Tensor,
        module_valid: torch.Tensor,
        length_scale: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Generate all module-seeded candidates with three live updates."""

        def resistance_fn(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
            return self._resistance(encoded, starts, ends, "source_module")

        return fixed_data_mean_shift(
            module_centers,
            module_descriptors,
            length_scale,
            source_valid=_as_bool_mask(module_valid),
            steps=self.mean_shift_steps,
            feature_bandwidth=self.mean_shift_feature_bandwidth,
            resistance_fn=resistance_fn,
            return_trajectory=return_trajectory,
        )

    def _build_candidates_with_sources(
        self,
        encoded: Any,
        module_tokens: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> tuple[RoutingCandidates, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, module_count, spatial_dim = module_tokens.shape[0], module_tokens.shape[1], encoded.module_centers.shape[-1]
        module_geometry = self._geometry(encoded, encoded.module_centers)
        module_descriptors = self._descriptor_module(module_tokens, module_geometry, encoded.global_token)
        candidate_trajectory = None
        if self.routing_strategy == "mean_shift":
            length_scale = self._length_scale(encoded, encoded.module_centers)
            shifted, candidate_trajectory = self._mean_shift_candidates(
                encoded,
                encoded.module_centers,
                module_descriptors,
                encoded.module_present,
                length_scale,
                return_trajectory=return_trajectory,
            )
            candidate_coords = shifted[..., :spatial_dim] * length_scale.reshape(1, 1, spatial_dim)
            candidate_descriptors = shifted[..., spatial_dim:]
            candidate_geometry = self._geometry(encoded, candidate_coords)
        else:
            candidate_coords = encoded.module_centers
            candidate_descriptors = module_descriptors
            candidate_geometry = module_geometry
        candidate_valid = _as_bool_mask(encoded.module_present)
        candidate_origin = torch.arange(module_count, device=module_tokens.device, dtype=torch.long)[None].expand(batch, -1)
        candidate_propensity = self.router.hub_propensity(
            candidate_descriptors, candidate_geometry, encoded.global_token
        )

        # A neutral background candidate is materialized only when at least one
        # case has no active module.  It serves the environmental route; it is
        # never a module source and has no field value of its own.
        empty_cases = ~candidate_valid.any(dim=1)
        if bool(empty_cases.any()):
            background_coords = encoded.module_centers.new_zeros((batch, 1, spatial_dim))
            background_descriptors = encoded.module_centers.new_zeros((batch, 1, self.routing_descriptor_dim))
            background_geometry = self._geometry(encoded, background_coords)
            background_propensity = encoded.module_centers.new_zeros((batch, 1))
            if candidate_trajectory is not None:
                background_state = torch.cat(
                    (
                        background_coords / self._length_scale(encoded, encoded.module_centers).reshape(1, 1, spatial_dim),
                        background_descriptors,
                    ),
                    dim=-1,
                )
                background_trajectory = background_state[:, None, :, :].expand(
                    -1, candidate_trajectory.shape[1], -1, -1
                )
                candidate_trajectory = torch.cat(
                    (candidate_trajectory, background_trajectory), dim=2
                )
            candidate_coords = torch.cat([candidate_coords, background_coords], dim=1)
            candidate_descriptors = torch.cat([candidate_descriptors, background_descriptors], dim=1)
            candidate_geometry = torch.cat([candidate_geometry, background_geometry], dim=1)
            candidate_propensity = torch.cat([candidate_propensity, background_propensity], dim=1)
            candidate_valid = torch.cat([candidate_valid, empty_cases[:, None]], dim=1)
            candidate_origin = torch.cat(
                [candidate_origin, torch.full((batch, 1), -1, device=module_tokens.device, dtype=torch.long)],
                dim=1,
            )
        candidates = RoutingCandidates(
            coords=candidate_coords,
            descriptors=candidate_descriptors,
            propensity=candidate_propensity,
            valid=candidate_valid,
            candidate_origin=candidate_origin,
        )
        return candidates, candidate_geometry, module_geometry, module_descriptors, candidate_trajectory

    def _build_candidates(
        self,
        encoded: Any,
        module_tokens: torch.Tensor,
    ) -> tuple[RoutingCandidates, torch.Tensor, torch.Tensor]:
        """Backward-compatible candidate builder for small numerical fixtures."""

        candidates, candidate_geometry, module_geometry, _, _ = self._build_candidates_with_sources(
            encoded, module_tokens
        )
        return candidates, candidate_geometry, module_geometry

    def _source_logits(
        self,
        encoded: Any,
        source_coords: torch.Tensor,
        source_descriptors: torch.Tensor,
        candidates: RoutingCandidates,
        relation_type: str,
    ) -> torch.Tensor:
        length_scale = self._length_scale(encoded, source_coords)
        starts = source_coords[:, :, None, :].expand(-1, -1, candidates.coords.shape[1], -1)
        ends = candidates.coords[:, None, :, :].expand(-1, source_coords.shape[1], -1, -1)
        resistance = self._resistance(encoded, starts, ends, relation_type)
        return routing_affinity(
            source_descriptors,
            source_coords,
            candidates.descriptors,
            candidates.coords,
            candidates.propensity,
            length_scale,
            resistance=resistance,
            content_scale=self.routing_content_scale,
            geometry_scale=self.routing_geometry_scale,
            propensity_scale=self.routing_propensity_scale,
        )

    def _make_source_incidence(
        self,
        logits: torch.Tensor,
        source_weights: torch.Tensor,
        source_valid: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> TypedSourceIncidence:
        valid = _as_bool_mask(source_valid)[..., None] & candidate_valid[:, None, :]
        membership = ordinary_source_sparsemax(
            logits / self.routing_temperature,
            valid,
        )
        return build_typed_source_incidence(
            source_weights,
            membership,
            source_valid=source_valid,
        )

    def prepare(
        self,
        encoded: Any,
        module_states: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        fine = self.prepare_fine_messages(encoded, module_states)
        global_env = encoded.global_token[:, None, :].expand(-1, fine["env_tokens"].shape[1], -1)
        contextual_env = fine["env_tokens"] + self.env_update(
            torch.cat([fine["env_tokens"], fine["environment_messages"], global_env], dim=-1)
        )

        with torch.profiler.record_function("routing.candidate_generation"):
            (
                candidates,
                candidate_geometry,
                module_geometry,
                module_descriptors,
                candidate_trajectory,
            ) = self._build_candidates_with_sources(
                encoded,
                fine["module_tokens"],
                return_trajectory=bool(return_routing_maps),
            )
        with torch.profiler.record_function("routing.source_route_preparation"):
            env_geometry = self._geometry(
                encoded,
                encoded.env_coords,
                fallback_features=getattr(encoded, "env_features", None),
            )
            env_descriptors = self._descriptor_environment(
                contextual_env, env_geometry, encoded.global_token
            )
            module_logits = self._source_logits(
                encoded,
                encoded.module_centers,
                module_descriptors,
                candidates,
                "source_module",
            )
            environment_logits = self._source_logits(
                encoded,
                encoded.env_coords,
                env_descriptors,
                candidates,
                "source_environment",
            )
            module_weights = module_source_measure(encoded.module_present)
            environment_weights = environment_source_measure(encoded.env_weights)
            module_incidence = self._make_source_incidence(
                module_logits,
                module_weights,
                encoded.module_present,
                candidates.valid,
            )
            environment_incidence = self._make_source_incidence(
                environment_logits,
                environment_weights,
                torch.ones_like(environment_weights, dtype=torch.bool),
                candidates.valid,
            )
        routing_diagnostics: dict[str, Any] = {
            "candidate_geometry": candidate_geometry,
            "module_geometry": module_geometry,
            "environment_geometry": env_geometry,
            # For mean_shift this remains the fixed source descriptor map;
            # candidates.descriptors may have moved in joint space.
            "module_descriptors": module_descriptors,
            "environment_descriptors": env_descriptors,
            "module_logits": module_logits,
            "environment_logits": environment_logits,
        }
        if candidate_trajectory is not None:
            routing_scale = self._length_scale(encoded, encoded.module_centers)
            routing_diagnostics["candidate_trajectory"] = {
                "coords": candidate_trajectory[..., : encoded.module_centers.shape[-1]]
                * routing_scale.reshape(1, 1, 1, -1),
                "descriptors": candidate_trajectory[..., encoded.module_centers.shape[-1] :],
            }
        routing_index = PreparedRoutingIndex(
            candidates=candidates,
            module_incidence=module_incidence,
            environment_incidence=environment_incidence,
            diagnostics=routing_diagnostics,
        )
        module_support = (module_incidence.membership > 0.0).sum(dim=-1)
        environment_support = (environment_incidence.membership > 0.0).sum(dim=-1)
        active_modules = _as_bool_mask(encoded.module_present).sum(dim=-1)
        prep_aux: dict[str, torch.Tensor] = {
            "routing_candidate_count": candidates.valid.sum(dim=-1).detach(),
            "routing_source_active_hubs": (
                (module_incidence.hub_measure + environment_incidence.hub_measure) > 0.0
            ).sum(dim=-1).detach(),
            "routing_module_source_nnz": module_support.sum(dim=-1).detach(),
            "routing_environment_source_nnz": environment_support.sum(dim=-1).detach(),
            "routing_module_source_mean_support": module_support.float().sum(dim=-1).detach()
            / active_modules.float().clamp_min(1.0),
            "routing_environment_source_mean_support": environment_support.float().mean(dim=-1).detach(),
            "routing_module_source_singleton_fraction": (
                (module_support == 1).sum(dim=-1).float() / active_modules.float().clamp_min(1.0)
            ).detach(),
            "routing_environment_source_singleton_fraction": (environment_support == 1).float().mean(dim=-1).detach(),
            "routing_module_source_active_fraction": (
                (module_support > 0).sum(dim=-1).float() / active_modules.float().clamp_min(1.0)
            ).detach(),
            "routing_environment_source_active_fraction": (environment_support > 0).float().mean(dim=-1).detach(),
        }
        env_keys, env_values = self.env_attention.project_source(contextual_env)
        return {
            "module_tokens": fine["module_tokens"],
            "env_tokens": contextual_env,
            "env_keys": env_keys,
            "env_values": env_values,
            "routing_index": routing_index,
            "routing_preparation_aux": prep_aux,
        }

    def preparation_aux(self, state: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return detached scalar/source support summaries for core aggregation."""

        return dict(state.get("routing_preparation_aux", {}))

    def _query_descriptors(
        self,
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
    ) -> torch.Tensor:
        query_geometry = self._geometry(encoded, receivers)
        return self._descriptor_query(receiver_features, query_geometry, encoded.global_token)

    def _query_logits(
        self,
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        candidates: RoutingCandidates,
        source_type: str,
        query_descriptors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_descriptors is None:
            query_descriptors = self._query_descriptors(encoded, receivers, receiver_features)
        return self._source_logits(
            encoded,
            receivers,
            query_descriptors,
            candidates,
            source_type,
        )

    def _query_density(
        self,
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        candidates: RoutingCandidates,
        incidence: TypedSourceIncidence,
        source_type: str,
        query_descriptors: torch.Tensor | None = None,
    ) -> Any:
        with torch.profiler.record_function("routing.query_hub_scores"):
            logits = self._query_logits(
                encoded,
                receivers,
                receiver_features,
                candidates,
                source_type,
                query_descriptors=query_descriptors,
            )
        with torch.profiler.record_function("routing.query_source_measure_sparsemax"):
            valid = (
                candidates.valid[:, None, :] & (incidence.hub_measure[:, None, :] > 0.0)
            ).expand_as(logits)
            projection = source_measure_sparsemax(
                logits / self.routing_temperature,
                incidence.hub_measure,
                valid,
            )
        return logits, projection

    def _tile(self, function: Any, *args: Any) -> torch.Tensor:
        # Gather inside the checkpointed function: retaining gathered Q/K/V or
        # concatenated MLP inputs across tiles defeats bounded activation memory.
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _module_tile(
        self, sources, centers, receivers, global_token, scale, batch_index,
        receiver_index, source_index, prior,
    ):
        batch, query_count = receivers.shape[:2]
        with torch.profiler.record_function("routing.qm_fine_kernel"):
            relative = (receivers[batch_index, receiver_index] - centers[batch_index, source_index]) / scale.reshape(-1)
            messages = self.query_module_message(torch.cat((
                sources[batch_index, source_index], self.relative_fourier(relative), global_token[batch_index],
            ), dim=-1))
        with torch.profiler.record_function("routing.qm_reduce"):
            result = sources.new_zeros((batch * query_count, sources.shape[-1]))
            weights = prior.to(messages.dtype)
            result.index_add_(0, batch_index * query_count + receiver_index, messages * weights[:, None])
        return result.reshape(batch, query_count, -1)

    def read_module_pairs(self, state, encoded, receivers, pairs: PackedPairs) -> torch.Tensor:
        """Evaluate each selected QM pair once, in bounded neural tiles."""
        batch, query_count = receivers.shape[:2]
        reduced = state["module_tokens"].new_zeros(batch, query_count, self.hidden_dim)
        for start in range(0, pairs.unique_pair_count, self.fine_pair_chunk_size):
            end = min(start + self.fine_pair_chunk_size, pairs.unique_pair_count)
            reduced = reduced + self._tile(
                self._module_tile, state["module_tokens"], encoded.module_centers, receivers,
                encoded.global_token, encoded.coordinate_scale, pairs.batch_index[start:end],
                pairs.receiver_index[start:end], pairs.source_index[start:end], pairs.prior[start:end],
            )
        count = encoded.module_present.sum(dim=1)
        # Output bias remains present even for M=0, as in Dense.
        return self.query_module_output(reduced * (count / (1.0 + count))[:, None, None])

    def _environment_score_tile(
        self, query, key, receivers, coordinates, scale, batch_index, receiver_index, source_index,
    ):
        with torch.profiler.record_function("routing.qe_fine_geometry_content"):
            relative = (receivers[batch_index, receiver_index] - coordinates[batch_index, source_index]) / scale.reshape(-1)
            bias = self.env_geometry_bias(self.relative_fourier(relative))
            q = query[batch_index, :, receiver_index, :]
            k = key[batch_index, :, source_index, :]
            return (q * k).sum(-1) / (float(self.env_attention.head_dim) ** 0.5) + bias

    def _environment_value_tile(
        self, value, attention, batch_index, source_index, group, group_count,
    ):
        with torch.profiler.record_function("routing.qe_reduce"):
            selected = value[batch_index, :, source_index, :]
            weighted = attention.to(selected.dtype)[..., None] * selected
            result = value.new_zeros((group_count, self.env_attention.num_heads, self.env_attention.head_dim))
            return result.index_add(0, group, weighted)

    def read_environment_pairs(self, state, encoded, receivers, receiver_features, pairs: PackedPairs) -> torch.Tensor:
        """Selected QE geometry/content, global receiver normalization, tiled V reads."""
        batch, query_count = receivers.shape[:2]
        if "env_keys" in state and "env_values" in state:
            key, value = state["env_keys"], state["env_values"]
        else:
            key, value = self.project_environment_sources(state["env_tokens"])
        query = self.env_attention.project_query(self.env_query(receiver_features))
        if pairs.unique_pair_count == 0:
            return self.env_attention.output(value.new_zeros(batch, query_count, self.hidden_dim))
        scores = []
        for start in range(0, pairs.unique_pair_count, self.fine_pair_chunk_size):
            end = min(start + self.fine_pair_chunk_size, pairs.unique_pair_count)
            scores.append(self._tile(
                self._environment_score_tile, query, key, receivers, encoded.env_coords,
                encoded.coordinate_scale, pairs.batch_index[start:end],
                pairs.receiver_index[start:end], pairs.source_index[start:end],
            ))
        group = pairs.batch_index * query_count + pairs.receiver_index
        with torch.profiler.record_function("routing.qe_weighted_normalization"):
            # Scalar FP64 log-priors avoid an FP32 1/Pi backward overflow;
            # quadrature is already in Pi and is not applied again here.
            weighted_scores = torch.cat(scores).double() + pairs.prior.double().log()[:, None]
            attention = _segment_softmax(weighted_scores, group, batch * query_count)
        reduced = value.new_zeros((batch * query_count, self.env_attention.num_heads, self.env_attention.head_dim))
        for start in range(0, pairs.unique_pair_count, self.fine_pair_chunk_size):
            end = min(start + self.fine_pair_chunk_size, pairs.unique_pair_count)
            reduced = reduced + self._tile(
                self._environment_value_tile, value, attention[start:end], pairs.batch_index[start:end],
                pairs.source_index[start:end], group[start:end], batch * query_count,
            )
        return self.env_attention.output(reduced.reshape(batch, query_count, self.hidden_dim))

    def read(
        self,
        state: dict[str, Any],
        encoded: Any,
        receivers: torch.Tensor,
        receiver_features: torch.Tensor,
        *,
        return_routing_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        routing_index: PreparedRoutingIndex = state["routing_index"]
        candidates = routing_index.candidates
        module_incidence = routing_index.module_incidence
        environment_incidence = routing_index.environment_incidence
        if module_incidence is None or environment_incidence is None:
            raise ValueError("RoutedPairwiseField requires both module and environment incidences.")
        query_descriptors = self._query_descriptors(encoded, receivers, receiver_features)
        module_logits, module_projection = self._query_density(
            encoded,
            receivers,
            receiver_features,
            candidates,
            module_incidence,
            "query_module",
            query_descriptors=query_descriptors,
        )
        environment_logits, environment_projection = self._query_density(
            encoded,
            receivers,
            receiver_features,
            candidates,
            environment_incidence,
            "query_environment",
            query_descriptors=query_descriptors,
        )
        with torch.profiler.record_function("routing.pair_join_deduplicate"):
            module_pairs = compile_positive_pairs(module_projection.density, module_incidence)
            environment_pairs = compile_positive_pairs(
                environment_projection.density, environment_incidence
            )
        module_context = self.read_module_pairs(state, encoded, receivers, module_pairs)
        environment_context = self.read_environment_pairs(
            state, encoded, receivers, receiver_features, environment_pairs
        )

        def receiver_counts(pairs: PackedPairs) -> torch.Tensor:
            values = receivers.new_zeros((int(receivers.shape[0]) * int(receivers.shape[1]),))
            if pairs.unique_pair_count:
                keys = pairs.batch_index * int(receivers.shape[1]) + pairs.receiver_index
                values.index_add_(0, keys, receivers.new_ones(pairs.prior.shape))
            return values.reshape(receivers.shape[0], receivers.shape[1]).detach()

        def scalar(value: float) -> torch.Tensor:
            return receivers.new_tensor(float(value)).detach()

        aux: dict[str, torch.Tensor] = {
            "routing_module_query_hub_support": module_projection.support.sum(dim=-1).detach(),
            "routing_environment_query_hub_support": environment_projection.support.sum(dim=-1).detach(),
            "routing_module_fine_pair_count": receiver_counts(module_pairs),
            "routing_environment_fine_pair_count": receiver_counts(environment_pairs),
            "routing_module_raw_path_count": scalar(module_pairs.raw_path_count),
            "routing_environment_raw_path_count": scalar(environment_pairs.raw_path_count),
            "routing_module_unique_pair_count": scalar(module_pairs.unique_pair_count),
            "routing_environment_unique_pair_count": scalar(environment_pairs.unique_pair_count),
            "routing_module_duplicate_expansion": scalar(module_pairs.duplicate_expansion),
            "routing_environment_duplicate_expansion": scalar(environment_pairs.duplicate_expansion),
        }
        if return_routing_maps:
            aux.update(
                {
                    "routing_module_query_logits": module_logits,
                    "routing_module_query_density": module_projection.density,
                    "routing_module_query_probability": module_projection.probability,
                    "routing_environment_query_logits": environment_logits,
                    "routing_environment_query_density": environment_projection.density,
                    "routing_environment_query_probability": environment_projection.probability,
                    "routing_module_pair_batch": module_pairs.batch_index,
                    "routing_module_pair_receiver": module_pairs.receiver_index,
                    "routing_module_pair_source": module_pairs.source_index,
                    "routing_module_pair_prior": module_pairs.prior,
                    "routing_environment_pair_batch": environment_pairs.batch_index,
                    "routing_environment_pair_receiver": environment_pairs.receiver_index,
                    "routing_environment_pair_source": environment_pairs.source_index,
                    "routing_environment_pair_prior": environment_pairs.prior,
                }
            )
        return module_context + environment_context, aux


__all__ = ["RoutedPairwiseField", "compile_positive_pairs"]
