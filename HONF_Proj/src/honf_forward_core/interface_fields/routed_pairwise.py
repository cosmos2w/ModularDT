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
from .routing_index.pair_join import (
    compile_two_hop_pairs,
    compile_two_hop_pairs_batched,
)
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
        # Execution-only opt-in during the measured optimization study. The
        # sparse union is always constructed first; this path is eligible only
        # when every environmental pair is present, with its original prior.
        self.routing_execution = str(_option(routing_config, "execution", "gathered"))
        if self.routing_execution not in {"gathered", "optimized_exact"}:
            raise ValueError("routing.execution must be 'gathered' or 'optimized_exact'.")
        self.dense_environment_fast_path = self.routing_execution == "optimized_exact"
        self.dense_environment_pair_tile_size = 262144
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

    @staticmethod
    def _select_environment_batch(
        values: torch.Tensor,
        batch_index: torch.Tensor,
        total_batch: int,
    ) -> torch.Tensor:
        """Select case rows while allowing geometry tensors to broadcast from one case."""

        if values.ndim == 0:
            return values
        if int(values.shape[0]) == 1 and int(total_batch) != 1:
            values = values.expand((int(total_batch), *values.shape[1:]))
        return values.index_select(0, batch_index)

    @staticmethod
    def _select_environment_scale(
        scale: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Select per-case scales, retaining one-dimensional broadcast scales."""

        if scale.ndim <= 1 or int(scale.shape[0]) == 1:
            return scale
        return scale.index_select(0, batch_index)

    @staticmethod
    def _environment_scale_rows(scale: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Normalize coordinate scales to ``[B, 1, d]`` for pairwise reads."""

        if scale.ndim == 1:
            return scale.reshape(1, 1, -1).expand(int(batch_size), -1, -1)
        if scale.ndim == 2:
            if int(scale.shape[0]) == 1:
                return scale[:, None, :].expand(int(batch_size), -1, -1)
            return scale[:, None, :]
        if scale.ndim == 3:
            if int(scale.shape[0]) == 1:
                return scale.expand(int(batch_size), -1, -1)
            return scale
        raise ValueError("coordinate_scale must have one, two, or three dimensions.")

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

    def _read_environment_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        receivers: torch.Tensor,
        coordinates: torch.Tensor,
        scale: torch.Tensor,
        pairs: PackedPairs,
    ) -> torch.Tensor:
        """Reduce selected QE pairs into unprojected contexts.

        The projections are supplied by the caller so a mixed complete/partial
        batch read can evaluate each projected source and query exactly once.
        This method deliberately returns before ``env_attention.output``;
        callers combine packed and complete rows and apply that biased output
        projection once, including rows with no selected pairs.
        """

        batch, query_count = receivers.shape[:2]
        if pairs.unique_pair_count == 0:
            return value.new_zeros(batch, query_count, self.hidden_dim)
        scores = []
        for start in range(0, pairs.unique_pair_count, self.fine_pair_chunk_size):
            end = min(start + self.fine_pair_chunk_size, pairs.unique_pair_count)
            scores.append(self._tile(
                self._environment_score_tile,
                query,
                key,
                receivers,
                coordinates,
                scale,
                pairs.batch_index[start:end],
                pairs.receiver_index[start:end],
                pairs.source_index[start:end],
            ))
        group = pairs.batch_index * query_count + pairs.receiver_index
        with torch.profiler.record_function("routing.qe_weighted_normalization"):
            # Scalar FP64 log-priors avoid an FP32 1/Pi backward overflow;
            # quadrature is already in Pi and is not applied again here.
            weighted_scores = torch.cat(scores).double() + pairs.prior.double().log()[:, None]
            attention = _segment_softmax(weighted_scores, group, batch * query_count)
        reduced = value.new_zeros(
            (batch * query_count, self.env_attention.num_heads, self.env_attention.head_dim)
        )
        for start in range(0, pairs.unique_pair_count, self.fine_pair_chunk_size):
            end = min(start + self.fine_pair_chunk_size, pairs.unique_pair_count)
            reduced = reduced + self._tile(
                self._environment_value_tile,
                value,
                attention[start:end],
                pairs.batch_index[start:end],
                pairs.source_index[start:end],
                group[start:end],
                batch * query_count,
            )
        return reduced.reshape(batch, query_count, self.hidden_dim)

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
        source_count = int(key.shape[-2])
        if (
            self.dense_environment_fast_path
            and pairs.unique_pair_count == batch * query_count * source_count
        ):
            # The union is complete before any fine score is evaluated. Use
            # pair indices rather than assuming an externally supplied packed
            # list has canonical order. No omitted fine pair is evaluated.
            pair_key = (pairs.batch_index * query_count + pairs.receiver_index) * source_count + pairs.source_index
            prior = pairs.prior.new_zeros(batch * query_count * source_count).scatter(
                0, pair_key, pairs.prior
            ).reshape(batch, query_count, source_count)
            query_tile = max(1, self.dense_environment_pair_tile_size // (batch * source_count))
            chunks = []
            for start in range(0, query_count, query_tile):
                end = min(start + query_tile, query_count)
                chunks.append(self._tile(
                    self._environment_complete_tile,
                    query[:, :, start:end], key, value, receivers[:, start:end],
                    encoded.env_coords, encoded.coordinate_scale, prior[:, start:end],
                ))
            reduced = torch.cat(chunks, dim=1)
            return self.env_attention.output(reduced)
        if not self.dense_environment_fast_path:
            reduced = self._read_environment_packed(
                query,
                key,
                value,
                receivers,
                encoded.env_coords,
                encoded.coordinate_scale,
                pairs,
            )
            return self.env_attention.output(reduced)

        # A complete environmental support can occur for individual batch
        # elements even when the global packed union is sparse.  Count packed
        # rows by batch and run the exact dense QE path only for those cases;
        # the remaining selected pairs keep the bounded packed evaluator.
        pair_counts = torch.bincount(pairs.batch_index, minlength=batch)
        full_batch = pair_counts == query_count * source_count
        full_indices = torch.nonzero(full_batch, as_tuple=False).flatten()
        if int(full_indices.numel()) == 0:
            reduced = self._read_environment_packed(
                query,
                key,
                value,
                receivers,
                encoded.env_coords,
                encoded.coordinate_scale,
                pairs,
            )
            return self.env_attention.output(reduced)

        full_pair_mask = full_batch.index_select(0, pairs.batch_index)
        partial_indices = torch.nonzero(~full_batch, as_tuple=False).flatten()

        full_remap = pairs.batch_index.new_full((batch,), -1)
        full_remap[full_indices] = torch.arange(
            int(full_indices.numel()), device=full_indices.device, dtype=full_indices.dtype
        )
        full_pair_batch = full_remap.index_select(0, pairs.batch_index[full_pair_mask])
        full_prior = pairs.prior[full_pair_mask]
        full_pair_key = (
            full_pair_batch * query_count + pairs.receiver_index[full_pair_mask]
        ) * source_count + pairs.source_index[full_pair_mask]
        full_prior_dense = full_prior.new_zeros(
            int(full_indices.numel()) * query_count * source_count
        ).scatter(0, full_pair_key, full_prior).reshape(
            int(full_indices.numel()), query_count, source_count
        )
        full_query = self._select_environment_batch(query, full_indices, batch)
        full_key = self._select_environment_batch(key, full_indices, batch)
        full_value = self._select_environment_batch(value, full_indices, batch)
        full_receivers = self._select_environment_batch(receivers, full_indices, batch)
        full_coordinates = self._select_environment_batch(encoded.env_coords, full_indices, batch)
        full_scale = self._select_environment_scale(encoded.coordinate_scale, full_indices)
        query_tile = max(
            1,
            self.dense_environment_pair_tile_size
            // (int(full_indices.numel()) * source_count),
        )
        full_chunks = []
        for start in range(0, query_count, query_tile):
            end = min(start + query_tile, query_count)
            full_chunks.append(self._tile(
                self._environment_complete_tile,
                full_query[:, :, start:end],
                full_key,
                full_value,
                full_receivers[:, start:end],
                full_coordinates,
                full_scale,
                full_prior_dense[:, start:end],
            ))
        full_reduced = torch.cat(full_chunks, dim=1)

        reduced = value.new_zeros(batch, query_count, self.hidden_dim)
        reduced = reduced.index_copy(0, full_indices, full_reduced)
        if int(partial_indices.numel()) != 0:
            partial_remap = pairs.batch_index.new_full((batch,), -1)
            partial_remap[partial_indices] = torch.arange(
                int(partial_indices.numel()),
                device=partial_indices.device,
                dtype=partial_indices.dtype,
            )
            partial_pair_mask = ~full_pair_mask
            partial_prior = pairs.prior[partial_pair_mask]
            partial_count = int(partial_prior.numel())
            partial_pairs = PackedPairs(
                partial_remap.index_select(0, pairs.batch_index[partial_pair_mask]),
                pairs.receiver_index[partial_pair_mask],
                pairs.source_index[partial_pair_mask],
                partial_prior,
                partial_count,
                partial_count,
            )
            partial_reduced = self._read_environment_packed(
                self._select_environment_batch(query, partial_indices, batch),
                self._select_environment_batch(key, partial_indices, batch),
                self._select_environment_batch(value, partial_indices, batch),
                self._select_environment_batch(receivers, partial_indices, batch),
                self._select_environment_batch(encoded.env_coords, partial_indices, batch),
                self._select_environment_scale(encoded.coordinate_scale, partial_indices),
                partial_pairs,
            )
            reduced = reduced.index_copy(0, partial_indices, partial_reduced)
        return self.env_attention.output(reduced)

    def _environment_complete_tile(self, query, key, value, receivers, coordinates, scale, prior):
        """Exact full-support QE with batched products, bounded receiver tiles.

        Geometry and prior arithmetic match the packed reader, including FP64
        normalization and casting attention back to the value dtype. The
        callback is checkpointed as a whole, so its pairwise geometry MLP
        activations are not retained across all receiver chunks in training.
        """
        with torch.profiler.record_function("routing.qe_complete_support"):
            scale_rows = self._environment_scale_rows(scale, receivers.shape[0])
            relative = (
                receivers[:, :, None, :] - coordinates[:, None, :, :]
            ) / scale_rows[:, :, None, :]
            bias = self.env_geometry_bias(self.relative_fourier(relative)).permute(0, 3, 1, 2)
            scores = torch.matmul(query, key.transpose(-1, -2)) / (float(self.env_attention.head_dim) ** 0.5)
            weighted = (scores + bias).double() + prior.double().log()[:, None, :, :]
            attention = torch.softmax(weighted, dim=-1).to(value.dtype)
            reduced = torch.matmul(attention, value).transpose(1, 2)
            return reduced.reshape(receivers.shape[0], receivers.shape[1], self.hidden_dim)

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
            if self.routing_execution == "optimized_exact":
                module_pairs = compile_two_hop_pairs_batched(
                    module_projection.density,
                    module_incidence,
                    scalar_tile_size=262144,
                    use_checkpoint=False,
                )
                environment_pairs = compile_two_hop_pairs_batched(
                    environment_projection.density,
                    environment_incidence,
                    scalar_tile_size=262144,
                    use_checkpoint=False,
                )
            else:
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
