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
from torch import nn
from torch.utils.checkpoint import checkpoint

from .dense_pairwise import DensePairwiseField

try:
    from .kernels import fused_qe_reader, is_triton_qe_available
except (ImportError, ModuleNotFoundError):  # pragma: no cover - package fallback
    fused_qe_reader = None

    def is_triton_qe_available(device: torch.device | str | None = None) -> bool:
        del device
        return False

from .routing_index.geometry import (
    geometry_features,
    geometry_length_scale,
    geometry_resistance,
    routing_affinity,
)
from .routing_index.pair_join import (
    compile_two_hop_pairs,
    compile_two_hop_pairs_batched,
    compile_two_hop_pairs_compiled,
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
    CompiledPairs,
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
        typed_log_temperatures: Any = None,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_hidden_dim,
            num_heads,
            fourier_frequencies,
            activation_checkpointing=activation_checkpointing,
        )
        # Ownership stays on InterfaceFieldCore: avoid registering the same
        # four parameters again under backend state-dict keys.
        object.__setattr__(self, "_typed_log_temperatures", typed_log_temperatures)
        self.sparsification = _option(routing_config, "sparsification", None)
        self.paircost_enabled = bool(_option(self.sparsification, "enabled", False))
        if bool(_option(self.sparsification, "learn_typed_temperatures", False)) and typed_log_temperatures is None:
            raise ValueError("Typed temperatures require the four core-owned log-temperature parameters.")
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
        if self.routing_execution not in {"gathered", "optimized_exact", "compiled_exact"}:
            raise ValueError(
                "routing.execution must be 'gathered', 'optimized_exact', or 'compiled_exact'."
            )
        self.qe_backend = str(_option(routing_config, "qe_backend", "torch"))
        if self.qe_backend not in {"torch", "triton"}:
            raise ValueError("routing.qe_backend must be 'torch' or 'triton'.")
        # The requested backend is mutable for isolated benchmark runs.  Keep
        # the actual backend and fallback reason observable after every read so
        # a requested Triton run can never be reported as a measured Triton
        # execution when the optional kernel fell back to Torch.
        self.last_qe_backend = "torch"
        self.last_qe_backend_reason = "not_run"
        self._qe_backend_seen: set[str] = set()
        self._qe_backend_reasons: list[str] = []
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

    def _temperature(self, relation: str) -> float | torch.Tensor:
        if self._typed_log_temperatures is None:
            return self.routing_temperature
        # Exponentiation uses scalar route precision; no floor or schedule.
        return self._typed_log_temperatures["log_temperature_" + relation].double().exp()

    def _paircost_components(self, density, incidence):
        """Live, tiled Eq. (18) components; every positive route is retained."""
        membership = incidence.membership
        support = torch.zeros_like(membership, dtype=torch.bool)
        edges = incidence.inverted
        support[edges.batch_index, edges.source_index, edges.hub_index] = True
        valid = support.any(dim=-1) & (incidence.source_weights > 0)
        a = torch.where(support, membership, torch.zeros_like(membership)).double()
        d = torch.where(density > 0, density, torch.zeros_like(density)).double()
        batch, queries, hubs = d.shape
        sources = a.shape[1]
        epsilon = float(_option(self.sparsification, "relative_density_epsilon", .05))
        batch_numerators = []
        budget = 262144
        batch_tile = max(1, min(batch, budget // max(hubs, 1)))
        for b0 in range(0, batch, batch_tile):
            b1 = min(batch, b0 + batch_tile)
            source_tile = max(1, min(sources, budget // (b1-b0)))
            query_tile = max(1, budget // ((b1-b0) * max(source_tile, hubs, 1)))
            query_numerators = []
            for q0 in range(0, queries, query_tile):
                numerator = d[b0:b1, q0:q0+query_tile].sum(dim=-1) * 0.0
                for n0 in range(0, sources, source_tile):
                    relative_density = torch.bmm(
                        d[b0:b1, q0:q0+query_tile], a[b0:b1, n0:n0+source_tile].transpose(1, 2),
                    )
                    numerator = numerator + ((relative_density / (relative_density + epsilon))
                        * valid[b0:b1, None, n0:n0+source_tile]).sum(dim=-1)
                query_numerators.append(numerator)
            batch_numerators.append(torch.cat(query_numerators, dim=1))
        # Preserve receiver ownership until case coupling excludes padded
        # port slots. The case loss sums these live components only once.
        numerator = torch.cat(batch_numerators, dim=0)
        denominator = valid.sum(dim=-1).to(dtype=d.dtype)[:, None].expand(batch, queries)
        return numerator, denominator

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
        source_type: str = "source_module",
    ) -> TypedSourceIncidence:
        valid = _as_bool_mask(source_valid)[..., None] & candidate_valid[:, None, :]
        membership = ordinary_source_sparsemax(
            logits / self._temperature(source_type),
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
                "source_module",
            )
            environment_incidence = self._make_source_incidence(
                environment_logits,
                environment_weights,
                torch.ones_like(environment_weights, dtype=torch.bool),
                candidates.valid,
                "source_environment",
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
                logits / self._temperature(source_type),
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

    @staticmethod
    def _is_uninitialized_lazy(module: nn.Module) -> bool:
        """Return whether a lazy layer still needs its input width inferred."""

        return bool(
            isinstance(module, nn.LazyLinear)
            and module.has_uninitialized_params()
        )

    def _ensure_qm_initialized(
        self,
        sources: torch.Tensor,
        global_token: torch.Tensor,
        spatial_dim: int,
    ) -> None:
        """Infer the historical QM first-layer width before affine reuse.

        ``query_module_message`` is a LazyMLP because adapter feature widths
        are runtime-defined.  Compiled complete rows should still take the
        factored path on their first read, so initialize the lazy layer from
        its shape with a detached one-row probe.  Parameter initialization is
        exactly the same LazyLinear initialization used by the ordinary
        reader; the probe contributes no graph or field value.
        """

        network = self.query_module_message.net
        if not network or not self._is_uninitialized_lazy(network[0]):
            return
        if sources.ndim != 3 or global_token.ndim != 2:
            raise ValueError("QM source/global tensors must have shapes [B,N,H] and [B,H].")
        probe_relative = sources.new_zeros((1, spatial_dim))
        relative_width = int(self.relative_fourier(probe_relative).shape[-1])
        input_width = int(sources.shape[-1]) + relative_width + int(global_token.shape[-1])
        probe = sources.new_zeros((1, input_width))
        # LazyLinear infers shape and initializes parameters from the probe;
        # no numerical result is used and no autograd edge is retained.  Call
        # only the lazy layer so a future nonzero-dropout configuration does
        # not consume an extra RNG draw during shape inference.
        with torch.no_grad():
            network[0](probe)

    def _qm_affine_terms(
        self,
        sources: torch.Tensor,
        global_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Prepare ``W_z z`` and ``W_g g+b`` for exact pairwise QM reuse."""

        network = self.query_module_message.net
        if not network or not isinstance(network[0], nn.Linear):
            return None
        first = network[0]
        source_width = int(sources.shape[-1])
        global_width = int(global_token.shape[-1])
        relative_width = int(first.weight.shape[1]) - source_width - global_width
        if relative_width < 0:
            return None
        source_term = torch.nn.functional.linear(
            sources,
            first.weight[:, :source_width],
            None,
        )
        global_term = torch.nn.functional.linear(
            global_token,
            first.weight[:, source_width + relative_width :],
            first.bias,
        )
        return source_term, global_term, relative_width

    def _read_module_complete_direct(
        self,
        state: dict[str, torch.Tensor],
        encoded: Any,
        receivers: torch.Tensor,
        pairs: CompiledPairs,
        reduced: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate complete rows through the original QM network.

        This defensive path handles a custom or incompletely materialized
        first layer.  It retains the compiled representation's implicit
        complete dispatch and only forms rectangular ``[row, source]`` tiles
        over authoritative valid sources; it never expands complete rows to a
        global ``(batch, receiver, source)`` pair list.
        """

        batch, query_count = receivers.shape[:2]
        complete_count = int(pairs.complete_row_index.numel())
        if complete_count == 0:
            return reduced
        complete_batches = torch.div(
            pairs.complete_row_index,
            query_count,
            rounding_mode="floor",
        )
        scale = encoded.coordinate_scale.reshape(-1)
        for batch_index in range(batch):
            case_positions = torch.nonzero(
                complete_batches == batch_index,
                as_tuple=False,
            ).flatten()
            valid_sources = torch.nonzero(
                pairs.source_valid[batch_index],
                as_tuple=False,
            ).flatten()
            if int(case_positions.numel()) == 0 or int(valid_sources.numel()) == 0:
                continue
            source_tile = max(1, min(int(valid_sources.numel()), self.fine_pair_chunk_size))
            row_tile = max(1, self.fine_pair_chunk_size // source_tile)
            for row_start in range(0, int(case_positions.numel()), row_tile):
                row_positions = case_positions[row_start : row_start + row_tile]
                row_ids = pairs.complete_row_index.index_select(0, row_positions)
                row_receiver = torch.remainder(row_ids, query_count)
                for source_start in range(0, int(valid_sources.numel()), source_tile):
                    source_ids = valid_sources[source_start : source_start + source_tile]
                    prior = pairs.complete_prior.index_select(0, row_positions)
                    prior = prior[:, source_ids]
                    weighted = self._tile(
                        self._qm_complete_direct_tile,
                        state["module_tokens"][batch_index, source_ids],
                        encoded.module_centers[batch_index, source_ids],
                        receivers[batch_index, row_receiver],
                        encoded.global_token[batch_index],
                        scale,
                        prior,
                    )
                    reduced.index_add_(
                        0,
                        row_ids,
                        weighted,
                    )
        return reduced

    def _qm_reused_tile(
        self,
        sources: torch.Tensor,
        centers: torch.Tensor,
        receivers: torch.Tensor,
        source_affine: torch.Tensor,
        global_affine: torch.Tensor,
        coordinate_scale: torch.Tensor,
        relative_width: int,
        batch_index: torch.Tensor,
        receiver_index: torch.Tensor,
        source_index: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a compact QM tile after exact first-affine reuse."""

        batch, query_count = receivers.shape[:2]
        first = self.query_module_message.net[0]
        source_width = int(sources.shape[-1])
        relative = (
            receivers[batch_index, receiver_index]
            - centers[batch_index, source_index]
        ) / coordinate_scale.reshape(-1)
        relative_features = self.relative_fourier(relative)
        relative_term = torch.nn.functional.linear(
            relative_features,
            first.weight[:, source_width : source_width + int(relative_width)],
            None,
        )
        preactivation = (
            source_affine[batch_index, source_index]
            + global_affine[batch_index]
            + relative_term
        )
        messages = self.query_module_message.net[1](preactivation)
        for layer in self.query_module_message.net[2:]:
            messages = layer(messages)
        result = sources.new_zeros((batch * query_count, sources.shape[-1]))
        result.index_add_(
            0,
            batch_index * query_count + receiver_index,
            messages * prior.to(messages.dtype)[:, None],
        )
        return result.reshape(batch, query_count, -1)

    def _qm_complete_affine_tile(
        self,
        source_affine: torch.Tensor,
        global_affine: torch.Tensor,
        centers: torch.Tensor,
        receivers: torch.Tensor,
        relative_weights: torch.Tensor,
        coordinate_scale: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce one complete QM rectangle after first-layer reuse.

        The rectangle is the natural checkpoint boundary: source and global
        affine terms stay factored while only the nonlinear activation for the
        current ``[receiver, source]`` tile is recomputed in backward.
        """

        with torch.profiler.record_function("routing.qm_complete_affine_tile"):
            relative = (
                receivers[:, None, :] - centers[None, :, :]
            ) / coordinate_scale.reshape(-1)
            relative_term = torch.nn.functional.linear(
                self.relative_fourier(relative),
                relative_weights,
                None,
            )
            preactivation = (
                source_affine[None, :, :]
                + global_affine[None, None, :]
                + relative_term
            )
            messages = self.query_module_message.net[1](preactivation)
            for layer in self.query_module_message.net[2:]:
                messages = layer(messages)
            return (
                messages * prior.to(messages.dtype)[..., None]
            ).sum(dim=1)

    def _qm_pair_affine_tile(
        self,
        source_affine: torch.Tensor,
        global_affine: torch.Tensor,
        centers: torch.Tensor,
        receivers: torch.Tensor,
        relative_weights: torch.Tensor,
        coordinate_scale: torch.Tensor,
        prior: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate and weight one CSR QM tile after first-layer reuse."""

        with torch.profiler.record_function("routing.qm_pair_affine_tile"):
            if valid is not None:
                safe_centers = torch.where(
                    valid[..., None], centers, torch.zeros_like(centers)
                )
                safe_affine = torch.where(
                    valid[..., None], source_affine, torch.zeros_like(source_affine)
                )
            else:
                safe_centers = centers
                safe_affine = source_affine
            relative = (receivers - safe_centers) / coordinate_scale.reshape(-1)
            relative_term = torch.nn.functional.linear(
                self.relative_fourier(relative),
                relative_weights,
                None,
            )
            messages = self.query_module_message.net[1](
                safe_affine + global_affine + relative_term
            )
            for layer in self.query_module_message.net[2:]:
                messages = layer(messages)
            if valid is not None:
                messages = torch.where(valid[..., None], messages, torch.zeros_like(messages))
            return messages * prior.to(messages.dtype)[:, None]

    def _qm_complete_direct_tile(
        self,
        sources: torch.Tensor,
        centers: torch.Tensor,
        receivers: torch.Tensor,
        global_token: torch.Tensor,
        coordinate_scale: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce one direct complete QM rectangle for fallback readers."""

        relative = (
            receivers[:, None, :] - centers[None, :, :]
        ) / coordinate_scale.reshape(-1)
        relative_features = self.relative_fourier(relative)
        source_values = sources[None, :, :].expand(
            int(receivers.shape[0]), -1, -1
        )
        global_values = global_token[None, None, :].expand(
            int(receivers.shape[0]), int(sources.shape[0]), -1
        )
        messages = self.query_module_message(
            torch.cat((source_values, relative_features, global_values), dim=-1)
            .reshape(-1, source_values.shape[-1] + relative_features.shape[-1] + global_values.shape[-1])
        ).reshape(
            int(receivers.shape[0]), int(sources.shape[0]), self.hidden_dim
        )
        return (messages * prior.to(messages.dtype)[..., None]).sum(dim=1)

    def _qm_complete_batch_affine_tile(
        self, source_affine, global_affine, centers, receivers, scale, prior,
        relative_width: int, source_width: int,
    ):
        """Rectangular exact QM with shared source/global affine terms."""
        relative = (receivers[:, :, None, :] - centers[:, None, :, :]) / scale.reshape(-1)
        first = self.query_module_message.net[0]
        relative_term = torch.nn.functional.linear(
            self.relative_fourier(relative),
            first.weight[:, source_width:source_width + relative_width],
        )
        messages = source_affine[:, None, :, :] + global_affine[:, None, None, :] + relative_term
        for layer in self.query_module_message.net[1:]:
            messages = layer(messages)
        return (messages * prior.to(messages.dtype)[..., None]).sum(dim=2)

    def _qm_complete_rows_affine_tile(
        self, source_affine, global_affine, centers, receivers, scale, prior,
        row_ids, relative_width: int, source_width: int,
    ):
        batch_ids = torch.div(row_ids, receivers.shape[1], rounding_mode="floor")
        query_ids = torch.remainder(row_ids, receivers.shape[1])
        return self._qm_complete_batch_affine_tile(
            source_affine[batch_ids], global_affine[batch_ids], centers[batch_ids],
            receivers[batch_ids, query_ids, None], scale, prior[:, None],
            relative_width, source_width,
        )[:, 0]

    def _read_module_compiled(self, state, encoded, receivers, pairs: CompiledPairs) -> torch.Tensor:
        """Read complete rows implicitly and partial rows from CSR."""

        batch, query_count = receivers.shape[:2]
        source_count = int(state["module_tokens"].shape[1])
        if source_count != pairs.source_count:
            raise ValueError("Compiled module source count does not match module state.")
        # Keep this cache on the physical prepared state.  It remains live for
        # gradients and is naturally discarded when P0/P1/P2 prepare a new
        # state.  The coordinate scale is read by the compact tile helper and
        # set only for this call, avoiding an extra public state key.
        self._ensure_qm_initialized(
            state["module_tokens"],
            encoded.global_token,
            int(receivers.shape[-1]),
        )
        affine = state.get("_qm_first_affine")
        if affine is None:
            affine = self._qm_affine_terms(state["module_tokens"], encoded.global_token)
            if affine is not None:
                state["_qm_first_affine"] = affine
        if (affine is not None and int(pairs.complete_row_index.numel()) == batch * query_count
                and bool(pairs.source_valid.all())):
            source_affine, global_affine, relative_width = affine
            # Physical source banks are shared over queries, not gathered
            # once per pair. Checkpoint only compact inputs to each tile.
            q_tile = max(1, self.fine_pair_chunk_size // max(batch * source_count, 1))
            prior = pairs.complete_prior.reshape(batch, query_count, source_count)
            chunks = []
            for q0 in range(0, query_count, q_tile):
                q1 = min(query_count, q0 + q_tile)
                chunks.append(self._tile(
                    self._qm_complete_batch_affine_tile, source_affine, global_affine,
                    encoded.module_centers, receivers[:, q0:q1], encoded.coordinate_scale,
                    prior[:, q0:q1], relative_width, int(state["module_tokens"].shape[-1]),
                ))
            context = torch.cat(chunks, dim=1)
            count = encoded.module_present.sum(dim=1)
            return self.query_module_output(context * (count / (1.0 + count))[:, None, None])
        reduced = state["module_tokens"].new_zeros(batch * query_count, self.hidden_dim)
        if affine is None:
            # A custom reader can lack a splittable first Linear.  Preserve
            # exact complete support through rectangular direct tiles rather
            # than silently returning zero for those rows.
            reduced = self._read_module_complete_direct(
                state,
                encoded,
                receivers,
                pairs,
                reduced,
            )
        else:
            source_affine, global_affine, relative_width = affine
            valid_source = pairs.source_valid
            # Complete source ranges are evaluated in bounded rectangular
            # tiles over the authoritative valid source list.  This avoids
            # evaluating padded module slots and never emits complete-row
            # pair-index triplets.
            complete_batches = torch.div(
                pairs.complete_row_index,
                query_count,
                rounding_mode="floor",
            )
            first = self.query_module_message.net[0]
            source_width = int(state["module_tokens"].shape[-1])
            relative_weights = first.weight[:, source_width : source_width + int(relative_width)]
            scale = encoded.coordinate_scale.reshape(-1)
            whole_case = pairs.complete_rows.all(dim=1) & pairs.source_valid.all(dim=1)
            whole_indices = torch.nonzero(whole_case, as_tuple=False).flatten()
            if int(whole_indices.numel()):
                group_count = int(whole_indices.numel())
                position_map = (pairs.complete_rows.reshape(-1).long().cumsum(0) - 1).reshape(batch, query_count)
                positions = position_map.index_select(0, whole_indices)
                group_prior = pairs.complete_prior.index_select(0, positions.reshape(-1)).reshape(group_count, query_count, source_count)
                group_source = source_affine.index_select(0, whole_indices)
                group_global = global_affine.index_select(0, whole_indices)
                group_centers = encoded.module_centers.index_select(0, whole_indices)
                group_receivers = receivers.index_select(0, whole_indices)
                group_tile = max(1, self.fine_pair_chunk_size // max(group_count * source_count, 1))
                group_results = []
                for q0 in range(0, query_count, group_tile):
                    q1 = min(query_count, q0 + group_tile)
                    group_results.append(self._tile(
                        self._qm_complete_batch_affine_tile, group_source, group_global,
                        group_centers, group_receivers[:, q0:q1], encoded.coordinate_scale,
                        group_prior[:, q0:q1], relative_width, source_width,
                    ))
                group_rows = whole_indices[:, None] * query_count + torch.arange(query_count, device=receivers.device)[None]
                reduced = reduced.index_copy(0, group_rows.reshape(-1), torch.cat(group_results, dim=1).reshape(-1, self.hidden_dim))
            # Irregular receiver sets can still share an implicit complete
            # source range. Pack only row IDs, and gather reused affine terms
            # inside checkpointed fine tiles; no (q,k,i) paths or pair triples.
            row_pack_cases = (~whole_case) & pairs.source_valid.all(dim=1)
            row_positions = torch.nonzero(row_pack_cases.index_select(0, complete_batches), as_tuple=False).flatten()
            row_tile = max(1, self.fine_pair_chunk_size // max(source_count, 1))
            for r0 in range(0, int(row_positions.numel()), row_tile):
                positions = row_positions[r0:r0+row_tile]
                rows = pairs.complete_row_index.index_select(0, positions)
                context = self._tile(
                    self._qm_complete_rows_affine_tile, source_affine, global_affine,
                    encoded.module_centers, receivers, encoded.coordinate_scale,
                    pairs.complete_prior.index_select(0, positions), rows,
                    relative_width, source_width,
                )
                reduced = reduced.index_copy(0, rows, context)
            handled_cases = whole_case | row_pack_cases
            remaining = torch.unique(complete_batches[~handled_cases.index_select(0, complete_batches)]).tolist()
            for batch_index in remaining:
                case_positions = torch.nonzero(
                    complete_batches == batch_index,
                    as_tuple=False,
                ).flatten()
                valid_sources = torch.nonzero(
                    valid_source[batch_index],
                    as_tuple=False,
                ).flatten()
                if int(case_positions.numel()) == 0 or int(valid_sources.numel()) == 0:
                    continue
                source_tile = max(1, min(int(valid_sources.numel()), self.fine_pair_chunk_size))
                row_tile = max(1, self.fine_pair_chunk_size // source_tile)
                for row_start in range(0, int(case_positions.numel()), row_tile):
                    row_positions = case_positions[row_start : row_start + row_tile]
                    row_ids = pairs.complete_row_index.index_select(0, row_positions)
                    row_receiver = torch.remainder(row_ids, query_count)
                    for source_start in range(0, int(valid_sources.numel()), source_tile):
                        source_ids = valid_sources[source_start : source_start + source_tile]
                        prior = pairs.complete_prior.index_select(0, row_positions)
                        weighted = self._tile(
                            self._qm_complete_affine_tile,
                            source_affine[batch_index, source_ids],
                            global_affine[batch_index],
                            encoded.module_centers[batch_index, source_ids],
                            receivers[batch_index, row_receiver],
                            relative_weights,
                            scale,
                            prior[:, source_ids],
                        )
                        reduced.index_add_(0, row_ids, weighted)

        partial_counts = pairs.partial_row_ptr[1:] - pairs.partial_row_ptr[:-1]
        partial_count = int(pairs.partial_pair_count)
        if partial_count:
            row_ids = torch.repeat_interleave(
                torch.arange(batch * query_count, device=receivers.device, dtype=torch.long),
                partial_counts,
            )
            batch_index = torch.div(row_ids, query_count, rounding_mode="floor")
            receiver_index = torch.remainder(row_ids, query_count)
            if affine is None:
                partial_pairs = PackedPairs(
                    batch_index,
                    receiver_index,
                    pairs.partial_source_index,
                    pairs.partial_prior,
                    partial_count,
                    partial_count,
                )
                partial_result = self.read_module_pairs(state, encoded, receivers, partial_pairs)
                reduced = reduced + partial_result.reshape(batch * query_count, -1)
            else:
                source_affine, global_affine, relative_width = affine
                for start in range(0, partial_count, self.fine_pair_chunk_size):
                    end = min(start + self.fine_pair_chunk_size, partial_count)
                    tile_rows = row_ids[start:end]
                    tile_batch = batch_index[start:end]
                    tile_receiver = receiver_index[start:end]
                    tile_source = pairs.partial_source_index[start:end]
                    valid = pairs.source_valid[tile_batch, tile_source]
                    weighted = self._tile(
                        self._qm_pair_affine_tile,
                        source_affine[tile_batch, tile_source],
                        global_affine[tile_batch],
                        encoded.module_centers[tile_batch, tile_source],
                        receivers[tile_batch, tile_receiver],
                        relative_weights,
                        scale,
                        pairs.partial_prior[start:end],
                        valid,
                    )
                    reduced.index_add_(0, tile_rows, weighted)
        count = encoded.module_present.sum(dim=1)
        return self.query_module_output(
            reduced.reshape(batch, query_count, -1)
            * (count / (1.0 + count))[:, None, None]
        )

    def read_module_pairs(
        self,
        state,
        encoded,
        receivers,
        pairs: PackedPairs | CompiledPairs,
    ) -> torch.Tensor:
        """Evaluate each selected QM pair once, in bounded neural tiles."""
        if isinstance(pairs, CompiledPairs):
            return self._read_module_compiled(state, encoded, receivers, pairs)
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

    def _begin_qe_backend_observation(self) -> None:
        """Reset per-read QE backend accounting without touching model state."""

        self._qe_backend_seen = set()
        self._qe_backend_reasons = []
        self.last_qe_backend = "torch"
        self.last_qe_backend_reason = "not_run"
        if self.qe_backend == "torch":
            self._note_qe_backend("torch", "configured_torch")

    def _note_qe_backend(self, backend: str, reason: str) -> None:
        """Record the backend actually used by the latest QE read."""

        seen = getattr(self, "_qe_backend_seen", set())
        reasons = getattr(self, "_qe_backend_reasons", [])
        seen.add(str(backend))
        if reason not in reasons:
            reasons.append(str(reason))
        self._qe_backend_seen = seen
        self._qe_backend_reasons = reasons
        self.last_qe_backend = next(iter(seen)) if len(seen) == 1 else "mixed"
        self.last_qe_backend_reason = ";".join(reasons)

    def _qe_triton_eligible(self, device: torch.device) -> bool:
        """Check the explicit opt-in and optional runtime without changing defaults."""

        if self.qe_backend != "triton":
            self._note_qe_backend("torch", "configured_torch")
            return False
        if fused_qe_reader is None or not is_triton_qe_available(device):
            self._note_qe_backend("torch", "triton_unavailable")
            return False
        return True

    def _environment_pair_geometry_bias(
        self,
        receivers: torch.Tensor,
        coordinates: torch.Tensor,
        scale: torch.Tensor,
        batch_index: torch.Tensor,
        receiver_index: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the existing geometry MLP for CSR scalar pairs only."""

        batch = int(receivers.shape[0])
        if coordinates.ndim == 2:
            source_coordinates = coordinates[source_index]
        elif coordinates.ndim == 3:
            source_coordinates = coordinates[batch_index, source_index]
        else:
            raise ValueError("environment coordinates must have shape [N,d] or [B,N,d].")
        scale_rows = self._environment_scale_rows(scale, batch)
        if int(scale_rows.shape[1]) != 1:
            raise ValueError("coordinate_scale must provide one vector per case for CSR QE.")
        relative = (
            receivers[batch_index, receiver_index] - source_coordinates
        ) / scale_rows[batch_index, 0, :]
        return self.env_geometry_bias(self.relative_fourier(relative))

    def _read_environment_csr_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        receivers: torch.Tensor,
        coordinates: torch.Tensor,
        scale: torch.Tensor,
        row_offsets: torch.Tensor,
        source_index: torch.Tensor,
        prior: torch.Tensor,
        row_batch: torch.Tensor,
        row_receiver: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run Triton for exact CSR rows after Torch geometry-bias evaluation."""

        if not self._qe_triton_eligible(query.device):
            return None
        if int(source_index.numel()) == 0:
            self._note_qe_backend("torch", "empty_csr_support")
            return value.new_zeros((int(receivers.shape[0]), int(receivers.shape[1]), self.hidden_dim))
        bias = self._environment_pair_geometry_bias(
            receivers,
            coordinates,
            scale,
            row_batch,
            row_receiver,
            source_index,
        )
        try:
            context = fused_qe_reader(
                query,
                key,
                value,
                bias,
                prior,
                mode="csr",
                row_offsets=row_offsets,
                source_indices=source_index,
            )
        except RuntimeError as exc:
            # A missing/unsupported Triton runtime must leave the explicit
            # Torch path usable.  Record the concrete exception class so
            # measurements cannot call the fallback a Triton execution.
            self._note_qe_backend(
                "torch",
                f"triton_runtime_error:{type(exc).__name__}",
            )
            return None
        self._note_qe_backend("triton", "fused_csr")
        return context.transpose(1, 2).reshape(
            int(receivers.shape[0]), int(receivers.shape[1]), self.hidden_dim
        )

    def _read_environment_complete_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        receivers: torch.Tensor,
        prior: torch.Tensor,
        source_mask: torch.Tensor | None,
        bias: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run Triton for a complete rectangular QE tile when explicitly enabled."""

        if not self._qe_triton_eligible(query.device):
            return None
        try:
            context = fused_qe_reader(
                query,
                key,
                value,
                bias,
                prior,
                mode="complete",
                source_mask=source_mask,
            )
        except RuntimeError as exc:
            self._note_qe_backend(
                "torch",
                f"triton_runtime_error:{type(exc).__name__}",
            )
            return None
        self._note_qe_backend("triton", "fused_complete")
        return context.transpose(1, 2).reshape(
            int(receivers.shape[0]), int(receivers.shape[1]), self.hidden_dim
        )

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
        if self.qe_backend == "triton":
            # Legacy PackedPairs do not carry authoritative CSR row pointers;
            # retain their readable Torch reducer rather than rebuilding and
            # sorting an index list solely to force the optional backend.
            self._note_qe_backend("torch", "packed_rows_torch_only")
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

    def _read_environment_compiled(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        receivers: torch.Tensor,
        coordinates: torch.Tensor,
        scale: torch.Tensor,
        pairs: CompiledPairs,
    ) -> torch.Tensor:
        """Read mixed complete/partial QE rows without complete pair indices."""

        batch, query_count = receivers.shape[:2]
        reduced = value.new_zeros(batch * query_count, self.hidden_dim)
        complete_count = int(pairs.complete_row_index.numel())
        if complete_count == batch * query_count:
            # Whole rectangular support retains the original batched K/V
            # sharing and reader tile budget; never copy a bank per receiver.
            source_count = int(pairs.source_count)
            query_tile = max(1, self.dense_environment_pair_tile_size // max(batch * source_count, 1))
            prior = pairs.complete_prior.reshape(batch, query_count, source_count)
            source_mask = None if bool(pairs.source_valid.all()) else pairs.source_valid
            chunks = []
            for start in range(0, query_count, query_tile):
                end = min(query_count, start + query_tile)
                chunks.append(self._tile(
                    self._environment_complete_tile,
                    query[:, :, start:end], key, value, receivers[:, start:end],
                    coordinates, scale, prior[:, start:end], source_mask,
                ))
            return torch.cat(chunks, dim=1)
        if complete_count:
            # A complete row carries an implicit source range.  Group complete
            # receivers by physical case so each prepared K/V bank is kept
            # rectangular and loaded once per receiver tile; gathering a full
            # source bank separately for every row would erase the benefit of
            # the implicit dispatch.
            source_count = int(pairs.source_count)
            row_tile = max(1, self.dense_environment_pair_tile_size // max(source_count, 1))
            complete_batches = torch.div(
                pairs.complete_row_index,
                query_count,
                rounding_mode="floor",
            )
            # Most mixed batches have whole complete cases and only a few
            # cases with irregular receiver support. Batch the rectangular
            # cases together before handling the genuinely irregular rows.
            whole_case = pairs.complete_rows.all(dim=1)
            whole_indices = torch.nonzero(whole_case, as_tuple=False).flatten()
            if int(whole_indices.numel()):
                group_count = int(whole_indices.numel())
                position_map = (pairs.complete_rows.reshape(-1).long().cumsum(0) - 1).reshape(batch, query_count)
                positions = position_map.index_select(0, whole_indices)
                group_prior = pairs.complete_prior.index_select(0, positions.reshape(-1)).reshape(
                    group_count, query_count, source_count,
                )
                group_query = query.index_select(0, whole_indices)
                group_key = key.index_select(0, whole_indices)
                group_value = value.index_select(0, whole_indices)
                group_receivers = receivers.index_select(0, whole_indices)
                group_coordinates = self._select_environment_batch(coordinates, whole_indices, batch)
                group_scale = self._select_environment_scale(scale, whole_indices)
                group_mask = pairs.source_valid.index_select(0, whole_indices)
                if bool(group_mask.all()):
                    group_mask = None
                group_tile = max(1, self.dense_environment_pair_tile_size // (group_count * source_count))
                group_results = []
                for q0 in range(0, query_count, group_tile):
                    q1 = min(query_count, q0 + group_tile)
                    group_results.append(self._tile(
                        self._environment_complete_tile, group_query[:, :, q0:q1],
                        group_key, group_value, group_receivers[:, q0:q1],
                        group_coordinates, group_scale, group_prior[:, q0:q1], group_mask,
                    ))
                group_rows = whole_indices[:, None] * query_count + torch.arange(query_count, device=query.device)[None]
                reduced = reduced.index_copy(0, group_rows.reshape(-1), torch.cat(group_results, dim=1).reshape(-1, self.hidden_dim))
            remaining = torch.unique(complete_batches[~whole_case.index_select(0, complete_batches)]).tolist()
            for batch_index in remaining:
                case_mask = complete_batches == batch_index
                case_positions = torch.nonzero(case_mask, as_tuple=False).flatten()
                if int(case_positions.numel()) == 0:
                    continue
                row_ids = pairs.complete_row_index.index_select(0, case_positions)
                row_receiver = torch.remainder(row_ids, query_count)
                case_index = torch.tensor(
                    [batch_index],
                    device=receivers.device,
                    dtype=torch.long,
                )
                case_key = self._select_environment_batch(key, case_index, batch)
                case_value = self._select_environment_batch(value, case_index, batch)
                case_coordinates = self._select_environment_batch(
                    coordinates,
                    case_index,
                    batch,
                )
                case_scale = self._select_environment_scale(
                    scale,
                    case_index,
                )
                case_mask_sources = pairs.source_valid[batch_index : batch_index + 1]
                for start in range(0, int(case_positions.numel()), row_tile):
                    end = min(start + row_tile, int(case_positions.numel()))
                    positions = case_positions[start:end]
                    local_rows = pairs.complete_row_index.index_select(0, positions)
                    local_receiver = torch.remainder(local_rows, query_count)
                    row_query = query[batch_index : batch_index + 1, :, local_receiver]
                    row_receivers = receivers[batch_index : batch_index + 1, local_receiver]
                    row_prior = pairs.complete_prior.index_select(0, positions).unsqueeze(0)
                    complete_result = self._tile(
                        self._environment_complete_tile,
                        row_query,
                        case_key,
                        case_value,
                        row_receivers,
                        case_coordinates,
                        case_scale,
                        row_prior,
                        case_mask_sources,
                    )
                    reduced.index_copy_(0, local_rows, complete_result.reshape(-1, self.hidden_dim))

        partial_count = int(pairs.partial_pair_count)
        if partial_count:
            partial_counts = pairs.partial_row_ptr[1:] - pairs.partial_row_ptr[:-1]
            row_ids = torch.repeat_interleave(
                torch.arange(batch * query_count, device=receivers.device, dtype=torch.long),
                partial_counts,
            )
            row_batch = torch.div(row_ids, query_count, rounding_mode="floor")
            row_receiver = torch.remainder(row_ids, query_count)
            partial_result = self._read_environment_csr_triton(
                query,
                key,
                value,
                receivers,
                coordinates,
                scale,
                pairs.partial_row_ptr,
                pairs.partial_source_index,
                pairs.partial_prior,
                row_batch,
                row_receiver,
            )
            if partial_result is None:
                partial_pairs = PackedPairs(
                    row_batch,
                    row_receiver,
                    pairs.partial_source_index,
                    pairs.partial_prior,
                    partial_count,
                    partial_count,
                )
                partial_result = self._read_environment_packed(
                    query,
                    key,
                    value,
                    receivers,
                    coordinates,
                    scale,
                    partial_pairs,
                )
            reduced = reduced + partial_result.reshape(batch * query_count, -1)
        return reduced.reshape(batch, query_count, self.hidden_dim)

    def read_environment_pairs(
        self,
        state,
        encoded,
        receivers,
        receiver_features,
        pairs: PackedPairs | CompiledPairs,
    ) -> torch.Tensor:
        """Selected QE geometry/content, global receiver normalization, tiled V reads."""
        self._begin_qe_backend_observation()
        batch, query_count = receivers.shape[:2]
        if "env_keys" in state and "env_values" in state:
            key, value = state["env_keys"], state["env_values"]
        else:
            key, value = self.project_environment_sources(state["env_tokens"])
        query = self.env_attention.project_query(self.env_query(receiver_features))
        if isinstance(pairs, CompiledPairs):
            if pairs.unique_pair_count == 0:
                self._note_qe_backend("torch", "empty_support")
            reduced = self._read_environment_compiled(
                query,
                key,
                value,
                receivers,
                encoded.env_coords,
                encoded.coordinate_scale,
                pairs,
            )
            return self.env_attention.output(reduced)
        if pairs.unique_pair_count == 0:
            self._note_qe_backend("torch", "empty_support")
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

    def _environment_complete_tile(self, query, key, value, receivers, coordinates, scale, prior, source_mask=None):
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
            fused = self._read_environment_complete_triton(
                query,
                key,
                value,
                receivers,
                prior,
                source_mask,
                bias,
            )
            if fused is not None:
                return fused
            scores = torch.matmul(query, key.transpose(-1, -2)) / (float(self.env_attention.head_dim) ** 0.5)
            safe_prior = torch.where(prior > 0.0, prior, torch.ones_like(prior))
            weighted = (scores + bias).double() + safe_prior.double().log()[:, None, :, :]
            if source_mask is not None:
                valid = source_mask[:, None, None, :].to(dtype=torch.bool)
                weighted = weighted.masked_fill(~valid, torch.finfo(weighted.dtype).min)
            attention = torch.softmax(weighted, dim=-1)
            if source_mask is not None:
                valid_float = valid.to(attention.dtype)
                attention = attention * valid_float
                attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(attention.dtype).tiny
                )
            attention = attention.to(value.dtype)
            reduced = torch.matmul(attention, value).transpose(1, 2)
            return reduced.reshape(receivers.shape[0], receivers.shape[1], self.hidden_dim)

    @staticmethod
    def _single_candidate_pairs(
        projection: Any,
        incidence: TypedSourceIncidence,
    ) -> CompiledPairs | None:
        """Build the constant one-candidate route without a general join.

        With one candidate there is no hub-level duplicate to join.  The
        shortcut still evaluates the exact live scalar
        ``Pi=omega * A * d`` in FP64, including finite source weights and
        query density.  Production singleton projections often reduce this
        algebra to ``Pi=omega`` after normalization, but retaining every live
        factor preserves gradients for loaded or hand-authored incidences.
        """

        if incidence.inverted.hub_count != 1:
            return None
        weights = incidence.source_weights
        membership = incidence.membership[..., 0]
        source_valid = (weights > 0.0) & (membership > 0.0)
        selected_weights = weights.masked_select(source_valid)
        if selected_weights.numel() and bool(
            (~torch.isfinite(selected_weights)).any()
        ):
            raise FloatingPointError(
                "single-candidate route has a nonfinite positive source measure"
            )
        selected_membership = membership.masked_select(source_valid)
        if selected_membership.numel() and bool(
            (~torch.isfinite(selected_membership)).any()
        ):
            raise FloatingPointError(
                "single-candidate route has a nonfinite positive membership"
            )
        density = projection.density
        if density.ndim != 3 or int(density.shape[-1]) != 1:
            return None
        batch, receiver_count, _ = (int(v) for v in density.shape)
        query_valid = density[..., 0] > 0.0
        complete_rows = query_valid & source_valid.any(dim=-1)[:, None]
        complete_row_index = torch.nonzero(
            complete_rows.reshape(-1),
            as_tuple=False,
        ).flatten()
        source_count = int(weights.shape[1])
        # Keep the exact live two-hop scalar, including finite precision
        # source mass and query density.  The masks are applied before the
        # product so inactive source/query entries do not acquire gradients.
        source_route = torch.where(
            source_valid,
            weights.to(dtype=torch.float64) * membership.to(dtype=torch.float64),
            torch.zeros_like(weights, dtype=torch.float64),
        )
        query_route = torch.where(
            query_valid,
            density[..., 0].to(dtype=torch.float64),
            torch.zeros_like(density[..., 0], dtype=torch.float64),
        )
        prior_matrix = (query_route[..., None] * source_route[:, None, :]).reshape(
            batch * receiver_count,
            source_count,
        )
        complete_prior = prior_matrix.index_select(0, complete_row_index)
        complete_batch = torch.div(
            complete_row_index,
            receiver_count,
            rounding_mode="floor",
        ) if int(complete_row_index.numel()) else complete_row_index
        if int(complete_row_index.numel()):
            selected_prior = prior_matrix.index_select(0, complete_row_index).masked_select(
                source_valid.index_select(0, complete_batch)
            )
            if selected_prior.numel() and bool(
                (~torch.isfinite(selected_prior) | (selected_prior <= 0.0)).any()
            ):
                raise FloatingPointError(
                    "single-candidate route produced a nonpositive or nonfinite prior"
                )
        complete_pair_count = int(source_valid.index_select(0, complete_batch).sum()) if int(complete_row_index.numel()) else 0
        raw_path_count = int(
            (query_valid.sum(dim=1) * source_valid.sum(dim=1)).sum()
        )
        empty_index = torch.empty(
            (0,),
            device=density.device,
            dtype=torch.long,
        )
        return CompiledPairs(
            complete_rows=complete_rows,
            complete_row_index=complete_row_index,
            complete_prior=complete_prior,
            partial_row_ptr=torch.zeros(
                (batch * receiver_count + 1,),
                device=density.device,
                dtype=torch.long,
            ),
            partial_source_index=empty_index,
            partial_prior=density.new_empty((0,), dtype=torch.float64),
            source_valid=source_valid,
            raw_path_count=raw_path_count,
            unique_pair_count=complete_pair_count,
            support_exam_count=0,
            scalar_exam_count=complete_pair_count,
            complete_pair_count_hint=complete_pair_count,
        )

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
            if self.routing_execution == "compiled_exact":
                module_pairs = None
                environment_pairs = None
                if not return_routing_maps:
                    module_pairs = self._single_candidate_pairs(
                        module_projection,
                        module_incidence,
                    )
                    environment_pairs = self._single_candidate_pairs(
                        environment_projection,
                        environment_incidence,
                    )
                if module_pairs is None:
                    module_pairs = compile_two_hop_pairs_compiled(
                        module_projection.density,
                        module_incidence,
                        scalar_tile_size=262144,
                        support_tile_size=262144,
                    )
                if environment_pairs is None:
                    environment_pairs = compile_two_hop_pairs_compiled(
                        environment_projection.density,
                        environment_incidence,
                        scalar_tile_size=262144,
                        support_tile_size=262144,
                    )
            elif self.routing_execution == "optimized_exact":
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

        def receiver_counts(pairs: PackedPairs | CompiledPairs) -> torch.Tensor:
            if isinstance(pairs, CompiledPairs):
                return pairs.row_counts.to(device=receivers.device).detach()
            values = receivers.new_zeros((int(receivers.shape[0]) * int(receivers.shape[1]),))
            if pairs.unique_pair_count:
                keys = pairs.batch_index * int(receivers.shape[1]) + pairs.receiver_index
                values.index_add_(0, keys, receivers.new_ones(pairs.prior.shape))
            return values.reshape(receivers.shape[0], receivers.shape[1]).detach()

        def scalar(value: float) -> torch.Tensor:
            if isinstance(module_pairs, CompiledPairs) and isinstance(value, int):
                # Exact route counts can exceed the FP32 integer range.
                return torch.tensor(value, device=receivers.device, dtype=torch.int64)
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
        if self.paircost_enabled:
            module_num, module_den = self._paircost_components(module_projection.density, module_incidence)
            env_num, env_den = self._paircost_components(environment_projection.density, environment_incidence)
            module_cost = float(_option(self.sparsification, "module_pair_cost", 1.0))
            environment_cost = float(_option(self.sparsification, "environment_pair_cost", 1.0))
            aux["routing_paircost_numerator"] = module_cost * module_num + environment_cost * env_num
            aux["routing_paircost_denominator"] = module_cost * module_den + environment_cost * env_den
        if self._typed_log_temperatures is not None:
            for relation in ("source_module", "source_environment", "query_module", "query_environment"):
                aux["routing_temperature_" + relation] = self._temperature(relation).detach()
        if return_routing_maps:
            if isinstance(module_pairs, CompiledPairs):
                aux.update(
                    {
                        # Keep the query-side maps available under both
                        # executor representations.  The compiled path has
                        # implicit complete rows, but its source-measure
                        # projections remain the authoritative diagnostics.
                        "routing_module_query_logits": module_logits,
                        "routing_module_query_density": module_projection.density,
                        "routing_module_query_probability": module_projection.probability,
                        "routing_environment_query_logits": environment_logits,
                        "routing_environment_query_density": environment_projection.density,
                        "routing_environment_query_probability": environment_projection.probability,
                        "routing_module_complete_rows": module_pairs.complete_rows,
                        "routing_module_complete_row_index": module_pairs.complete_row_index,
                        "routing_module_complete_prior": module_pairs.complete_prior,
                        "routing_module_partial_row_ptr": module_pairs.partial_row_ptr,
                        "routing_module_partial_source": module_pairs.partial_source_index,
                        "routing_module_partial_prior": module_pairs.partial_prior,
                        "routing_environment_complete_rows": environment_pairs.complete_rows,
                        "routing_environment_complete_row_index": environment_pairs.complete_row_index,
                        "routing_environment_complete_prior": environment_pairs.complete_prior,
                        "routing_environment_partial_row_ptr": environment_pairs.partial_row_ptr,
                        "routing_environment_partial_source": environment_pairs.partial_source_index,
                        "routing_environment_partial_prior": environment_pairs.partial_prior,
                    }
                )
                return module_context + environment_context, aux
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
