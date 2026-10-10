"""Joint fields with a preserved nonlinear pair path and a shallow graph residual.

The direct path is the maintained :class:`NonlinearFieldReadout`: two rounds of
measure-aware MM/ME/EM messages, followed by nonlinear reads of every physical
source. P-G and P-H insert one typed collective block between the two pair
rounds. Its node and receiver projections start at zero, so the direct model is
an exact submodel at initialization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .interaction_core import (
    DependencySpec,
    InteractionScene,
    NonlinearFieldReadout,
    PreparedResponseContext,
    _geometry,
    _score_mlp,
    _versions,
)
from .source_response_operator import PreparedSourceResponse, ResponseApplication


def _masked_measure_softmax(scores: torch.Tensor, measure: torch.Tensor,
                            present: torch.Tensor,
                            edge_present: torch.Tensor) -> torch.Tensor:
    """Normalize donor mass per edge, applying physical measure exactly once."""
    if scores.shape[-1] == 0:
        return scores
    mass = measure * present
    active = mass > 0
    log_mass = mass.clamp_min(torch.finfo(scores.dtype).tiny).log()
    logits = scores + log_mass[:, None]
    valid = active[:, None] & (edge_present[..., None] > 0)
    logits = logits.masked_fill(~valid, -torch.inf)
    has_donor = valid.any(-1, keepdim=True)
    maximum = torch.where(has_donor, logits.amax(-1, keepdim=True),
                          torch.zeros_like(logits[..., :1]))
    weights = torch.exp(logits - maximum) * valid
    denominator = weights.sum(-1, keepdim=True)
    return torch.where(denominator > 0,
                       weights / denominator.clamp_min(torch.finfo(scores.dtype).tiny),
                       torch.zeros_like(weights))


def _geometry_log_prior(nodes: torch.Tensor, edges: torch.Tensor,
                        scales: torch.Tensor) -> torch.Tensor:
    if edges.shape[1] == 0 or nodes.shape[1] == 0:
        return nodes.new_empty(nodes.shape[0], edges.shape[1], nodes.shape[1])
    relative = (nodes[:, None] - edges[:, :, None]) / scales[:, :, None].clamp_min(1e-12)
    return -0.5 * relative.square().sum(-1)


def _edge_to_node(membership: torch.Tensor, edge_states: torch.Tensor,
                  node_present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the existing normalized transpose as a context-sharing return rule."""
    if membership.shape[1] == 0:
        weights = membership.transpose(1, 2)
        return edge_states.new_zeros(edge_states.shape[0], membership.shape[2],
                                     edge_states.shape[-1]), weights
    weights = membership.transpose(1, 2)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny)
    weights = weights * node_present[..., None]
    return torch.bmm(weights, edge_states), weights


def _matched_mass_shuffle(membership: torch.Tensor, measure: torch.Tensor,
                          present: torch.Tensor,
                          physical_ids: torch.Tensor) -> torch.Tensor:
    """Reassign donor masses within equal-measure/presence buckets by physical ID."""
    if membership.shape[-1] == 0:
        return membership
    batches, edge_count, node_count = membership.shape
    if measure.shape != (batches, node_count) or present.shape != (batches, node_count):
        raise ValueError('Matched-mass shuffle measure/presence must align with donors.')
    if physical_ids.shape != (batches, node_count):
        raise ValueError('Matched-mass shuffle IDs must align with donor nodes.')
    rows = []
    for batch_index in range(batches):
        permutation = torch.arange(node_count, dtype=torch.long,
                                   device=membership.device)[None].expand(edge_count, -1).clone()
        active = (measure[batch_index] > 0) & (present[batch_index] > 0)
        if bool(active.any()):
            keys = torch.stack((measure[batch_index], present[batch_index]), -1)
            _unique, bucket_ids = torch.unique(keys, sorted=True, return_inverse=True, dim=0)
            for bucket_id in torch.unique(bucket_ids[active], sorted=True):
                bucket = torch.nonzero(active & (bucket_ids == bucket_id), as_tuple=False).flatten()
                if bucket.numel() < 2:
                    continue
                ordered = bucket[torch.argsort(physical_ids[batch_index, bucket], stable=True)]
                for edge_index in range(edge_count):
                    shift = 1 + edge_index % (bucket.numel() - 1)
                    permutation[edge_index, ordered] = ordered.roll(shift)
        rows.append(membership[batch_index].gather(-1, permutation))
    return torch.stack(rows, 0)


def _zero_linear(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


@dataclass
class InteractionPreservingPreparedContext(PreparedResponseContext):
    """Prepared pair context plus the one executed typed collective block."""

    source_measures_raw: torch.Tensor | None = None
    environment_measures_raw: torch.Tensor | None = None
    environment_ids: torch.Tensor | None = None
    context: torch.Tensor | None = None
    domain_origin: torch.Tensor | None = None
    edge_scale_vectors: torch.Tensor | None = None
    edge_type: torch.Tensor | None = None
    edge_source_ids: torch.Tensor | None = None
    edge_region_ids: torch.Tensor | None = None
    edge_anchor_states: torch.Tensor | None = None
    group_to_source: torch.Tensor | None = None
    group_to_environment: torch.Tensor | None = None
    source_update_delta: torch.Tensor | None = None
    environment_update_delta: torch.Tensor | None = None
    source_membership_density: torch.Tensor | None = None
    environment_membership_density: torch.Tensor | None = None
    _input_args: dict[str, Any] | None = None
    _incidence_mode: str = 'none'
    _access_mode: str = 'none'
    _collective_enabled: bool = False
    _matched_mass_shuffle: bool = False
    _intervention_receipt: dict[str, Any] | None = None
    _owner_id: int | None = None
    _model_config_snapshot: tuple | None = None
    _model_semantic_snapshot: tuple | None = None
    _prepared_semantic_snapshot: tuple | None = None

    def assert_fresh(self) -> None:
        super().assert_fresh()
        owner = self.model_reference() if self.model_reference is not None else None
        if owner is None or self._owner_id != id(owner):
            raise ValueError('Prepared interaction-preserving context belongs to another model.')
        snapshot = tuple(sorted(owner.config.items()))
        semantics = (owner.mode, owner.locality_prior_strength, owner.regional_anchors,
                     owner.collective_width, owner.field_outputs, owner.affine_outputs,
                     owner.forcing_scale, owner.zero_offset, owner.query_width,
                     owner.spatial_dim)
        prepared_semantics = (self._incidence_mode, self._access_mode,
                              self._collective_enabled, self._matched_mass_shuffle)
        if (snapshot != self._model_config_snapshot or
                semantics != self._model_semantic_snapshot or
                prepared_semantics != self._prepared_semantic_snapshot):
            raise ValueError('Prepared interaction-preserving model semantics changed; rebuild this request.')


class InteractionPreservingJointCore(NonlinearFieldReadout):
    """Opt-in P, P-G and P-H shared joint core.

    ``P`` executes only the maintained two-round source-conditioned path.
    ``P-G`` adds one fixed smooth geometric collective block. ``P-H`` uses the
    same block and geometry prior with learned residual donor and receiver
    scores. Thermal adapters may add a separately applicable affine response
    head; Wind uses only the inherited nonlinear field head.
    """

    prepared_context_type = InteractionPreservingPreparedContext

    def __init__(self, source_width: int, context_width: int,
                 environment_width: int, *, spatial_dim: int = 2,
                 hidden: int = 128, message: int = 128, mode: str = 'P',
                 collective_width: int = 64, regional_anchors: int = 0,
                 field_outputs: int | None = None, output_width: int | None = None,
                 affine_outputs: int = 0, query_width: int = 0,
                 max_sources: int | None = None, forcing_scale: float = 1.0,
                 zero_offset: bool = True, initialization_seed: int = 0,
                 locality_prior_strength: float | None = None):
        if mode not in ('P', 'P-G', 'P-H'):
            raise ValueError("mode must be 'P', 'P-G' or 'P-H'.")
        if field_outputs is None:
            field_outputs = 3 if output_width is None else output_width
        elif output_width is not None and output_width != field_outputs:
            raise ValueError('field_outputs and output_width must agree when both are supplied.')
        if field_outputs < 1 or affine_outputs < 0:
            raise ValueError('Field outputs must be positive and affine outputs nonnegative.')
        if collective_width < 1 or regional_anchors < 0:
            raise ValueError('Collective width must be positive and anchor count nonnegative.')
        if mode == 'P' and regional_anchors != 0:
            raise ValueError('P has no collective anchor bank; set regional_anchors=0.')
        if mode != 'P' and regional_anchors < 1:
            raise ValueError('P-G and P-H require at least one regional anchor.')
        if mode == 'P' and locality_prior_strength not in (None, 0, 0.0):
            raise ValueError('P has no collective locality prior.')
        if mode != 'P' and (locality_prior_strength is None or
                             not math.isfinite(locality_prior_strength) or
                             locality_prior_strength < 0):
            raise ValueError('P-G and P-H require an explicit finite nonnegative locality prior.')
        if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
            raise TypeError('initialization_seed must be an integer.')
        if not math.isfinite(forcing_scale) or forcing_scale <= 0:
            raise ValueError('forcing_scale must be finite and positive.')

        # Reset a private CPU RNG so all common direct tensors match across P,
        # P-G and P-H, independent of how many mode-specific modules follow.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            super().__init__(source_width, context_width, environment_width,
                             spatial_dim=spatial_dim, hidden=hidden,
                             message=message, output_width=field_outputs,
                             query_width=query_width, max_sources=max_sources)
            self.mode = mode
            self.collective_width = collective_width
            self.regional_anchors = regional_anchors
            self.field_outputs = field_outputs
            self.affine_outputs = affine_outputs
            self.forcing_scale = float(forcing_scale)
            self.zero_offset = bool(zero_offset)
            self.initialization_seed = initialization_seed
            self.locality_prior_strength = (0.0 if locality_prior_strength is None
                                            else float(locality_prior_strength))
            self.output_law = 'joint'
            self.field_output_law = 'nonlinear'
            self.response_output_law = 'affine' if affine_outputs else None
            self.output_capabilities = {
                'configuration_fields': {'law': 'nonlinear', 'outputs': field_outputs},
                'affine_control': ({'law': 'affine', 'outputs': affine_outputs}
                                   if affine_outputs else None),
            }

            self.affine_head = (nn.Sequential(nn.Linear(message, hidden), nn.SiLU(),
                                              nn.Linear(hidden, affine_outputs))
                                if affine_outputs else None)
            affine_read_width = 2 * hidden + 3 * spatial_dim * 9 + query_width
            self.affine_source_read = (nn.Sequential(
                nn.Linear(affine_read_width, hidden), nn.SiLU(),
                nn.Linear(hidden, message)) if affine_outputs else None)
            self.offset_head = (nn.Sequential(nn.Linear(hidden + spatial_dim * 9 + query_width,
                                                        hidden), nn.SiLU(),
                                              nn.Linear(hidden, affine_outputs))
                                if affine_outputs and not zero_offset else None)

            if mode != 'P':
                geometry_width = spatial_dim * 9
                access_width = min(collective_width, 64)
                self.collective_anchor = nn.Sequential(
                    nn.Linear(2 * geometry_width + 1 + context_width, collective_width),
                    nn.SiLU(), nn.Linear(collective_width, collective_width))
                self.collective_source_anchor = nn.Linear(hidden, collective_width, bias=False)
                self.collective_source_value = nn.Sequential(
                    nn.Linear(hidden + geometry_width, hidden), nn.SiLU(),
                    nn.Linear(hidden, collective_width))
                self.collective_environment_value = nn.Sequential(
                    nn.Linear(hidden + geometry_width, hidden), nn.SiLU(),
                    nn.Linear(hidden, collective_width))
                self.collective_update = nn.Sequential(
                    nn.Linear(3 * collective_width + context_width + 1, hidden), nn.SiLU(),
                    nn.Linear(hidden, collective_width))
                self.collective_to_source = nn.Linear(collective_width, hidden)
                self.collective_to_environment = nn.Linear(collective_width, hidden)
                self.collective_to_receiver = nn.Linear(collective_width, hidden)
                _zero_linear(self.collective_to_source)
                _zero_linear(self.collective_to_environment)
                _zero_linear(self.collective_to_receiver)

                if mode == 'P-H':
                    self.source_membership_score = _score_mlp(
                        hidden + collective_width + geometry_width, hidden)
                    self.environment_membership_score = _score_mlp(
                        hidden + collective_width + geometry_width, hidden)
                    self.receiver_query = nn.Sequential(
                        nn.Linear(context_width + geometry_width + query_width, hidden),
                        nn.SiLU(), nn.Linear(hidden, access_width, bias=False))
                    self.edge_key = nn.Linear(collective_width, access_width, bias=False)
                    self.access_geometry = _score_mlp(geometry_width, hidden)
                    # A zero score residual makes learned and geometric routes
                    # identical initially while retaining trainable upstream nets.
                    with torch.no_grad():
                        self.source_membership_score[-1].weight.zero_()
                        self.environment_membership_score[-1].weight.zero_()
                        self.receiver_query[-1].weight.zero_()
                        self.access_geometry[-1].weight.zero_()

            self.config.update({
                'mode': mode, 'collective_width': collective_width,
                'regional_anchors': regional_anchors, 'field_outputs': field_outputs,
                'output_width': field_outputs, 'affine_outputs': affine_outputs,
                'forcing_scale': float(forcing_scale), 'zero_offset': bool(zero_offset),
                'initialization_seed': initialization_seed,
            })
            if mode != 'P':
                self.config['locality_prior_strength'] = self.locality_prior_strength

    @staticmethod
    def _validate_ids(ids: torch.Tensor, present: torch.Tensor, shape: tuple[int, int],
                      name: str) -> None:
        if tuple(ids.shape) != shape or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f'{name} must be an integer tensor with shape {shape}.')
        for batch_index in range(ids.shape[0]):
            active = ids[batch_index][present[batch_index] > 0]
            if bool((active < 0).any()) or torch.unique(active).numel() != active.numel():
                raise ValueError(f'{name} must uniquely identify every active physical record.')

    @staticmethod
    def _regional_anchors(lengths: torch.Tensor, domain_origin: torch.Tensor,
                          count: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, dimension = lengths.shape
        if count == 0:
            return lengths.new_empty(batch, 0, dimension), lengths.new_empty(batch, 0, dimension)
        primes = (2, 3, 5)

        def radical_inverse(index: int, base: int) -> float:
            value, denominator, current = 0.0, 1.0, index
            while current:
                current, remainder = divmod(current, base)
                denominator *= base
                value += remainder / denominator
            return value

        unit = [[radical_inverse(index + 1, primes[axis]) for axis in range(dimension)]
                for index in range(count)]
        anchors_unit = lengths.new_tensor(unit)
        anchors = domain_origin[:, None] + anchors_unit[None] * lengths[:, None]
        scale = lengths / (count ** (1.0 / dimension))
        scales = scale[:, None].expand(-1, count, -1).clamp_min(
            torch.finfo(lengths.dtype).eps)
        return anchors, scales

    def _collective_block(self, source_states: torch.Tensor,
                          environment_states: torch.Tensor, *, context: torch.Tensor,
                          centers: torch.Tensor, present: torch.Tensor,
                          source_lengths: torch.Tensor, source_measures: torch.Tensor,
                          environment_coords: torch.Tensor,
                          environment_present: torch.Tensor,
                          environment_measures: torch.Tensor,
                          lengths: torch.Tensor, domain_origin: torch.Tensor,
                          source_ids: torch.Tensor, environment_ids: torch.Tensor,
                          incidence_mode: str, matched_mass_shuffle: bool,
                          collective_enabled: bool
                          ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        batch, modules, dimension = centers.shape
        environments = environment_coords.shape[1]
        regional_centers, regional_scales = self._regional_anchors(
            lengths, domain_origin, self.regional_anchors)
        edge_centers = torch.cat((centers, regional_centers), 1)
        edge_scales = torch.cat((source_lengths[..., None].expand(-1, -1, dimension),
                                 regional_scales), 1)
        edge_present = torch.cat((present, present.new_ones(batch, self.regional_anchors)), 1)
        edge_type = torch.cat((present.new_zeros(batch, modules),
                               present.new_ones(batch, self.regional_anchors)), 1)
        edge_source_ids = torch.cat((torch.where(present > 0, source_ids,
            torch.full_like(source_ids, -1)),
            torch.full((batch, self.regional_anchors), -1, dtype=source_ids.dtype,
                       device=source_ids.device)), 1)
        edge_region_ids = torch.cat((torch.full((batch, modules), -1, dtype=torch.long,
                                                device=centers.device),
            torch.arange(self.regional_anchors, dtype=torch.long, device=centers.device)[None]
                .expand(batch, -1)), 1)
        edge_features = torch.cat((
            _geometry((edge_centers - domain_origin[:, None]) / lengths[:, None]),
            _geometry(edge_scales / lengths[:, None]), edge_type[..., None],
            context[:, None].expand(-1, edge_centers.shape[1], -1)), -1)
        anchor_states = self.collective_anchor(edge_features)
        if modules:
            anchor_states = torch.cat((anchor_states[:, :modules] +
                                       self.collective_source_anchor(source_states),
                                       anchor_states[:, modules:]), 1)
        anchor_states = anchor_states * edge_present[..., None]

        source_relative = _geometry((centers[:, None] - edge_centers[:, :, None]) /
                                    edge_scales[:, :, None].clamp_min(1e-12))
        environment_relative = _geometry(
            (environment_coords[:, None] - edge_centers[:, :, None]) /
            edge_scales[:, :, None].clamp_min(1e-12))
        source_prior = self.locality_prior_strength * _geometry_log_prior(
            centers, edge_centers, edge_scales)
        environment_prior = self.locality_prior_strength * _geometry_log_prior(
            environment_coords, edge_centers, edge_scales)
        source_scores = source_prior
        environment_scores = environment_prior
        if incidence_mode == 'learned':
            source_inputs = torch.cat((
                source_states[:, None].expand(-1, edge_centers.shape[1], -1, -1),
                anchor_states[:, :, None].expand(-1, -1, modules, -1), source_relative), -1)
            environment_inputs = torch.cat((
                environment_states[:, None].expand(-1, edge_centers.shape[1], -1, -1),
                anchor_states[:, :, None].expand(-1, -1, environments, -1),
                environment_relative), -1)
            source_scores = source_scores + self.source_membership_score(source_inputs).squeeze(-1)
            environment_scores = (environment_scores +
                                  self.environment_membership_score(environment_inputs).squeeze(-1))
        source_membership = _masked_measure_softmax(
            source_scores, source_measures, present, edge_present)
        environment_membership = _masked_measure_softmax(
            environment_scores, environment_measures, environment_present, edge_present)
        if matched_mass_shuffle:
            source_membership = _matched_mass_shuffle(
                source_membership, source_measures, present, source_ids)
            environment_membership = _matched_mass_shuffle(
                environment_membership, environment_measures, environment_present,
                environment_ids)

        source_values = self.collective_source_value(torch.cat((
            source_states[:, None].expand(-1, edge_centers.shape[1], -1, -1),
            source_relative), -1))
        environment_values = self.collective_environment_value(torch.cat((
            environment_states[:, None].expand(-1, edge_centers.shape[1], -1, -1),
            environment_relative), -1))
        source_pool = (source_membership[..., None] * source_values).sum(2)
        environment_pool = (environment_membership[..., None] * environment_values).sum(2)
        collective_states = anchor_states + self.collective_update(torch.cat((
            anchor_states, source_pool, environment_pool,
            context[:, None].expand(-1, edge_centers.shape[1], -1), edge_type[..., None]), -1))
        collective_states = collective_states * edge_present[..., None]
        source_return, group_to_source = _edge_to_node(
            source_membership, collective_states, present)
        environment_return, group_to_environment = _edge_to_node(
            environment_membership, collective_states, environment_present)
        source_delta = self.collective_to_source(source_return) * present[..., None]
        environment_delta = self.collective_to_environment(environment_return) * environment_present[..., None]
        if not collective_enabled:
            source_delta = torch.zeros_like(source_delta)
            environment_delta = torch.zeros_like(environment_delta)
        source_result = source_states + source_delta
        environment_result = environment_states + environment_delta
        source_density = torch.where((source_measures * present > 0)[:, None],
            source_membership / source_measures.clamp_min(torch.finfo(source_measures.dtype).tiny)[:, None],
            torch.zeros_like(source_membership))
        environment_density = torch.where((environment_measures * environment_present > 0)[:, None],
            environment_membership / environment_measures.clamp_min(
                torch.finfo(environment_measures.dtype).tiny)[:, None],
            torch.zeros_like(environment_membership))
        return source_result, environment_result, {
            'group_states': collective_states, 'group_centers': edge_centers,
            'group_lengths': edge_scales.mean(-1), 'group_present': edge_present,
            'edge_scale_vectors': edge_scales, 'edge_type': edge_type,
            'edge_source_ids': edge_source_ids, 'edge_region_ids': edge_region_ids,
            'edge_anchor_states': anchor_states,
            'source_membership': source_membership,
            'environment_membership': environment_membership,
            'source_membership_density': source_density,
            'environment_membership_density': environment_density,
            'group_to_source': group_to_source,
            'group_to_environment': group_to_environment,
            'source_update_delta': source_delta,
            'environment_update_delta': environment_delta,
            'source_return': source_return, 'environment_return': environment_return,
            'incidence_mode': incidence_mode,
            'collective_enabled': bool(collective_enabled),
            'matched_mass_shuffle': bool(matched_mass_shuffle),
            'return_rule': 'normalized transpose of typed donor incidence; context sharing only',
            'typed_pair_slots': {
                'MM': int(batch * modules * modules),
                'ME': int(batch * modules * environments),
                'EM': int(batch * environments * modules),
            },
        }

    def prepare_context(self, sources: torch.Tensor, context: torch.Tensor,
                        centers: torch.Tensor, present: torch.Tensor,
                        lengths: torch.Tensor, source_lengths: torch.Tensor,
                        source_measures: torch.Tensor | None = None,
                        environment_tokens: torch.Tensor | None = None,
                        environment_coords: torch.Tensor | None = None,
                        environment_present: torch.Tensor | None = None,
                        environment_measures: torch.Tensor | None = None,
                        source_ids: torch.Tensor | None = None,
                        environment_embedding: torch.Tensor | None = None,
                        domain_origin: torch.Tensor | None = None,
                        environment_ids: torch.Tensor | None = None,
                        incidence_mode: str | None = None,
                        receiver_access: str | None = None,
                        matched_mass_shuffle: bool = False,
                        collective_enabled: bool = True) -> InteractionPreservingPreparedContext:
        batch, modules, dimension = centers.shape
        if source_ids is None:
            raise ValueError('The interaction-preserving family requires explicit physical source_ids.')
        self._validate_ids(source_ids, present, (batch, modules), 'source_ids')
        if source_lengths.ndim == 3:
            source_lengths = source_lengths.squeeze(-1)
        source_measures = torch.ones_like(present) if source_measures is None else source_measures
        if environment_tokens is None:
            environment_tokens = sources.new_empty(batch, 0, self.config['environment_width'])
            environment_coords = centers.new_empty(batch, 0, dimension)
        if environment_coords is None:
            raise ValueError('Environmental tokens require their physical coordinates.')
        environments = environment_tokens.shape[1]
        environment_present = (sources.new_ones(batch, environments) if environment_present is None
                               else environment_present)
        environment_measures = (torch.ones_like(environment_present) if environment_measures is None
                                else environment_measures)
        environment_ids = (torch.arange(environments, device=centers.device, dtype=torch.long)[None]
                           .expand(batch, -1) if environment_ids is None else environment_ids)
        self._validate_ids(environment_ids, environment_present, (batch, environments),
                           'environment_ids')
        domain_origin = centers.new_zeros(batch, dimension) if domain_origin is None else domain_origin
        self._validate_shape(domain_origin, (batch, dimension), 'domain_origin')

        if self.mode == 'P':
            if incidence_mode not in (None, 'none') or receiver_access not in (None, 'none'):
                raise ValueError('P has no collective incidence or receiver access to intervene on.')
            if matched_mass_shuffle or not collective_enabled:
                raise ValueError('P has no collective block to shuffle or disable.')
            chosen_incidence, chosen_access = 'none', 'none'
        else:
            default_incidence = 'geometry' if self.mode == 'P-G' else 'learned'
            default_access = 'geometry' if self.mode == 'P-G' else 'learned'
            chosen_incidence = default_incidence if incidence_mode is None else incidence_mode
            chosen_access = default_access if receiver_access is None else receiver_access
            if chosen_incidence not in ('geometry', 'learned'):
                raise ValueError("incidence_mode must be 'geometry' or 'learned'.")
            if chosen_access not in ('geometry', 'learned'):
                raise ValueError("receiver_access must be 'geometry' or 'learned'.")
            if self.mode == 'P-G' and (chosen_incidence == 'learned' or chosen_access == 'learned'):
                raise ValueError('P-G has no learned incidence or receiver-score parameters.')

        input_args = {
            'sources': sources, 'context': context, 'centers': centers, 'present': present,
            'lengths': lengths, 'source_lengths': source_lengths,
            'source_measures': source_measures, 'environment_tokens': environment_tokens,
            'environment_coords': environment_coords, 'environment_present': environment_present,
            'environment_measures': environment_measures, 'source_ids': source_ids,
            'environment_embedding': environment_embedding, 'domain_origin': domain_origin,
            'environment_ids': environment_ids,
        }
        if self.mode == 'P':
            prepared = super().prepare_context(
                sources, context, centers, present, lengths, source_lengths,
                source_measures, environment_tokens, environment_coords,
                environment_present, environment_measures, source_ids,
                environment_embedding, _extra_version_tensors=(domain_origin, environment_ids))
            collective = None
        else:
            holder: dict[str, Any] = {}

            def between_rounds(pair_sources: torch.Tensor,
                               pair_environment: torch.Tensor):
                updated_sources, updated_environment, auxiliary = self._collective_block(
                    pair_sources, pair_environment, context=context, centers=centers,
                    present=present, source_lengths=source_lengths,
                    source_measures=source_measures, environment_coords=environment_coords,
                    environment_present=environment_present,
                    environment_measures=environment_measures, lengths=lengths,
                    domain_origin=domain_origin, source_ids=source_ids,
                    environment_ids=environment_ids, incidence_mode=chosen_incidence,
                    matched_mass_shuffle=matched_mass_shuffle,
                    collective_enabled=collective_enabled)
                holder['collective'] = auxiliary
                return updated_sources, updated_environment, auxiliary

            prepared = super().prepare_context(
                sources, context, centers, present, lengths, source_lengths,
                source_measures, environment_tokens, environment_coords,
                environment_present, environment_measures, source_ids,
                environment_embedding, _between_rounds=between_rounds,
                _extra_version_tensors=(domain_origin, environment_ids))
            collective = holder['collective']

        prepared.source_measures_raw = source_measures
        prepared.environment_measures_raw = environment_measures
        prepared.environment_ids = environment_ids
        prepared.context = context
        prepared.domain_origin = domain_origin
        prepared._input_args = input_args
        prepared._incidence_mode = chosen_incidence
        prepared._access_mode = chosen_access
        prepared._collective_enabled = bool(collective_enabled and self.mode != 'P')
        prepared._matched_mass_shuffle = bool(matched_mass_shuffle)
        prepared._owner_id = id(self)
        prepared._model_config_snapshot = tuple(sorted(self.config.items()))
        prepared._model_semantic_snapshot = (
            self.mode, self.locality_prior_strength, self.regional_anchors,
            self.collective_width, self.field_outputs, self.affine_outputs,
            self.forcing_scale, self.zero_offset, self.query_width, self.spatial_dim)
        prepared._prepared_semantic_snapshot = (
            prepared._incidence_mode, prepared._access_mode,
            prepared._collective_enabled, prepared._matched_mass_shuffle)
        prepared._intervention_receipt = {
            'incidence': chosen_incidence,
            'receiver_access': chosen_access,
            'collective_enabled': prepared._collective_enabled,
            'matched_mass_shuffle': bool(matched_mass_shuffle),
            'return_rule': 'normalized transpose context sharing' if collective is not None else 'none',
        }
        if collective is not None:
            for name, value in collective.items():
                if name in ('incidence_mode', 'collective_enabled', 'matched_mass_shuffle',
                            'return_rule', 'typed_pair_slots'):
                    continue
                setattr(prepared, name, value)
        return prepared

    def prepare(self, scene: InteractionScene) -> InteractionPreservingPreparedContext:
        if not isinstance(scene, InteractionScene):
            raise TypeError('prepare expects an InteractionScene with a declared dependency.')
        if not isinstance(scene.dependency, DependencySpec):
            raise TypeError('InteractionScene dependency must be a DependencySpec.')
        if scene.dependency.output_law == 'affine' and self.affine_outputs == 0:
            raise ValueError('Scene requests an affine response block that this model does not provide.')
        prepared = self.prepare_context(**scene.tensors())
        prepared.dependency = scene.dependency
        return prepared

    def assert_owned(self, prepared: InteractionPreservingPreparedContext) -> None:
        if not isinstance(prepared, InteractionPreservingPreparedContext):
            raise TypeError('Prepared context must come from InteractionPreservingJointCore.')
        prepared.assert_fresh()
        if prepared._owner_id != id(self):
            raise ValueError('Prepared interaction-preserving context belongs to another model.')

    def intervene_context(self, prepared: InteractionPreservingPreparedContext, *,
                          incidence_mode: str | None = None,
                          receiver_access: str | None = None,
                          matched_mass_shuffle: bool = False,
                          collective_enabled: bool = True) -> InteractionPreservingPreparedContext:
        """Rebuild a same-weight context under declared graph interventions."""
        self.assert_owned(prepared)
        if self.mode == 'P':
            raise ValueError('P has no collective graph to intervene on.')
        assert prepared._input_args is not None
        rebuilt = self.prepare_context(
            **prepared._input_args, incidence_mode=incidence_mode,
            receiver_access=receiver_access, matched_mass_shuffle=matched_mass_shuffle,
            collective_enabled=collective_enabled)
        return self._preserve_context_metadata(prepared, rebuilt)

    def restore(self, prepared: InteractionPreservingPreparedContext) -> InteractionPreservingPreparedContext:
        """Rebuild the model's declared default incidence and receiver access."""
        self.assert_owned(prepared)
        if prepared._input_args is None:
            raise ValueError('Prepared context lacks the original scene inputs.')
        rebuilt = self.prepare_context(**prepared._input_args)
        return self._preserve_context_metadata(prepared, rebuilt)

    @staticmethod
    def _preserve_context_metadata(original: InteractionPreservingPreparedContext,
                                   rebuilt: InteractionPreservingPreparedContext
                                   ) -> InteractionPreservingPreparedContext:
        declared = set(InteractionPreservingPreparedContext.__dataclass_fields__)
        for name, value in vars(original).items():
            if name not in declared:
                setattr(rebuilt, name, value)
        return rebuilt

    @staticmethod
    def _receiver_access(prepared: InteractionPreservingPreparedContext,
                         receivers: torch.Tensor, features: torch.Tensor,
                         mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if prepared.group_states is None or prepared.group_centers is None:
            raise ValueError('Receiver collective access requires P-G or P-H context.')
        relative_raw = ((receivers[:, :, None] - prepared.group_centers[:, None]) /
                        prepared.edge_scale_vectors[:, None].clamp_min(1e-12))
        owner = prepared.model_reference()
        logits = -0.5 * owner.locality_prior_strength * relative_raw.square().sum(-1)
        valid = prepared.group_present[:, None] > 0
        if mode == 'learned':
            # Match the maintained direct query convention: absolute
            # coordinates divided by domain lengths, with no origin shift.
            query_geometry = _geometry(receivers / prepared.lengths[:, None])
            query_inputs = torch.cat((query_geometry,
                                      prepared.context[:, None].expand(-1, receivers.shape[1], -1),
                                      features), -1)
            query = owner.receiver_query(query_inputs)
            edge_key = owner.edge_key(prepared.group_states)
            learned = torch.einsum('bqd,bgd->bqg', query, edge_key) / math.sqrt(query.shape[-1])
            geometry_residual = owner.access_geometry(_geometry(relative_raw)).squeeze(-1)
            logits = logits + learned + geometry_residual
        logits = logits.masked_fill(~valid, -torch.inf)
        any_valid = valid.any(-1, keepdim=True)
        maximum = torch.where(any_valid, logits.amax(-1, keepdim=True),
                              torch.zeros_like(logits[..., :1]))
        weights = torch.exp(logits - maximum) * valid
        access = weights / weights.sum(-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny)
        collective = torch.bmm(access, prepared.group_states)
        return access, collective

    def _read_source_features(self, prepared: InteractionPreservingPreparedContext,
                              receivers: torch.Tensor, features: torch.Tensor,
                              start: int) -> tuple[torch.Tensor, torch.Tensor | None,
                                                   torch.Tensor | None]:
        if self.mode == 'P' or not prepared._collective_enabled:
            source_features = self.source_read(self._read_features(
                prepared, receivers, prepared.source_states, prepared.centers,
                prepared.source_lengths, features))
            return source_features, None, None
        access, receiver_collective = self._receiver_access(
            prepared, receivers, features, prepared._access_mode)
        inputs = self._read_features(prepared, receivers, prepared.source_states,
                                     prepared.centers, prepared.source_lengths, features)
        hidden_pre = self.source_read[0](inputs)
        hidden_pre = hidden_pre + self.collective_to_receiver(receiver_collective)[:, :, None]
        source_features = self.source_read[2](self.source_read[1](hidden_pre))
        return source_features, access, receiver_collective

    def _read_affine_source_features(self, prepared: InteractionPreservingPreparedContext,
                                    receivers: torch.Tensor, features: torch.Tensor,
                                    start: int) -> tuple[torch.Tensor, torch.Tensor | None,
                                                         torch.Tensor | None]:
        if self.affine_source_read is None:
            raise ValueError('This model has no affine source-specific read head.')
        if self.mode == 'P' or not prepared._collective_enabled:
            source_features = self.affine_source_read(self._read_features(
                prepared, receivers, prepared.source_states, prepared.centers,
                prepared.source_lengths, features))
            return source_features, None, None
        access, receiver_collective = self._receiver_access(
            prepared, receivers, features, prepared._access_mode)
        inputs = self._read_features(prepared, receivers, prepared.source_states,
                                     prepared.centers, prepared.source_lengths, features)
        hidden_pre = self.affine_source_read[0](inputs)
        hidden_pre = hidden_pre + self.collective_to_receiver(receiver_collective)[:, :, None]
        source_features = self.affine_source_read[2](self.affine_source_read[1](hidden_pre))
        return source_features, access, receiver_collective

    def predict_fields(self, prepared: InteractionPreservingPreparedContext,
                       receivers: torch.Tensor, receiver_features: torch.Tensor | None = None, *,
                       chunk_size: int = 512, return_messages: bool = False,
                       return_organization: bool = False):
        self.assert_owned(prepared)
        if self.mode == 'P' and not return_organization:
            return super().predict(prepared, receivers, receiver_features,
                                   chunk_size=chunk_size, return_messages=return_messages)
        batch, queries, _ = receivers.shape
        self._validate_shape(receivers, (prepared.centers.shape[0], queries, self.spatial_dim),
                             'receivers')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive.')
        features = (receivers.new_empty(batch, queries, 0) if receiver_features is None
                    else receiver_features)
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        outputs, messages, accesses, collective_reads = [], [], [], []
        for start in range(0, queries, chunk_size):
            query = receivers[:, start:start + chunk_size]
            current_features = features[:, start:start + chunk_size]
            source_features, access, collective = self._read_source_features(
                prepared, query, current_features, start)
            source_pool = (source_features * prepared.source_measure[:, None, :, None]).sum(2)
            field_input = torch.cat((source_pool,
                                     prepared.global_state[:, None].expand(-1, query.shape[1], -1),
                                     _geometry(query / prepared.lengths[:, None]),
                                     current_features), -1)
            outputs.append(self.field_head(field_input))
            if return_messages:
                messages.append(source_features * prepared.present[:, None, :, None])
            if return_organization:
                if access is None:
                    access = query.new_empty(batch, query.shape[1], 0)
                    collective = query.new_empty(batch, query.shape[1], 0)
                accesses.append(access)
                collective_reads.append(collective)
        fields = (torch.cat(outputs, 1) if outputs else
                  receivers.new_empty(batch, 0, self.field_outputs))
        if return_organization:
            organization = {
                'receiver_access': (torch.cat(accesses, 1) if accesses else
                                    receivers.new_empty(batch, 0,
                                        0 if prepared.group_states is None else prepared.group_states.shape[1])),
                'receiver_collective_state': (torch.cat(collective_reads, 1) if collective_reads else
                                               receivers.new_empty(batch, 0, self.collective_width)),
                'source_ids': prepared.source_ids,
                'group_centers': prepared.group_centers,
                'group_states': prepared.group_states,
                'source_membership': prepared.source_membership,
                'environment_membership': prepared.environment_membership,
                'incidence_mode': prepared._incidence_mode,
                'receiver_access_mode': prepared._access_mode,
                'collective_enabled': prepared._collective_enabled,
                'typed_pair_slots_per_batch': (None if self.mode == 'P' else {
                    'MM': int(prepared.centers.shape[1] ** 2),
                    'ME': int(prepared.centers.shape[1] * prepared.environment_coords.shape[1]),
                    'EM': int(prepared.environment_coords.shape[1] * prepared.centers.shape[1]),
                }),
                'return_rule': 'normalized transpose context sharing',
            }
            if return_messages:
                organization['source_messages'] = (torch.cat(messages, 1) if messages else
                    receivers.new_empty(batch, 0, prepared.centers.shape[1], self.message))
            return fields, organization
        if return_messages:
            exported = (torch.cat(messages, 1) if messages else
                        receivers.new_empty(batch, 0, prepared.centers.shape[1], self.message))
            return fields, exported
        return fields

    def predict(self, prepared, receivers, receiver_features=None, **kwargs):
        return self.predict_fields(prepared, receivers, receiver_features, **kwargs)

    def prepare_receivers(self, prepared: InteractionPreservingPreparedContext,
                          receivers: torch.Tensor,
                          receiver_features: torch.Tensor | None = None,
                          receiver_ids: torch.Tensor | None = None,
                          chunk_size: int = 512) -> PreparedSourceResponse:
        """Prepare the source-resolved affine kernel with existing response semantics."""
        self.assert_owned(prepared)
        if self.affine_outputs == 0 or self.affine_head is None:
            raise ValueError('This model has no separately applicable affine output capability.')
        batch, queries, _ = receivers.shape
        self._validate_shape(receivers, (prepared.centers.shape[0], queries, self.spatial_dim),
                             'receivers')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive.')
        features = (receivers.new_empty(batch, queries, 0) if receiver_features is None
                    else receiver_features)
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        ids = (torch.arange(queries, device=receivers.device)[None].expand(batch, -1)
               if receiver_ids is None else receiver_ids)
        self._validate_shape(ids, (batch, queries), 'receiver_ids')
        kernels, offsets, accesses, collective_reads = [], [], [], []
        for start in range(0, queries, chunk_size):
            query = receivers[:, start:start + chunk_size]
            current_features = features[:, start:start + chunk_size]
            source_features, access, collective = self._read_affine_source_features(
                prepared, query, current_features, start)
            kernels.append(self.affine_head(source_features) * prepared.present[:, None, :, None])
            if self.zero_offset:
                offsets.append(query.new_zeros(batch, query.shape[1], self.affine_outputs))
            else:
                geometry = _geometry(query / prepared.lengths[:, None])
                offsets.append(self.offset_head(torch.cat((
                    prepared.global_state[:, None].expand(-1, query.shape[1], -1),
                    geometry, current_features), -1)))
            if access is not None:
                accesses.append(access)
                collective_reads.append(collective)
        kernel = (torch.cat(kernels, 1) if kernels else
                  receivers.new_empty(batch, 0, prepared.centers.shape[1], self.affine_outputs))
        offset = (torch.cat(offsets, 1) if offsets else
                  receivers.new_empty(batch, 0, self.affine_outputs))
        response = PreparedSourceResponse(
            context=prepared, receivers=receivers, receiver_ids=ids, offset=offset,
            near_weight=receivers.new_zeros(batch, queries, prepared.centers.shape[1]),
            near_indices=torch.empty(3, 0, dtype=torch.long, device=receivers.device),
            near_values=receivers.new_empty(0, self.affine_outputs), far_kernel=kernel,
            receiver_functions=None, source_membership=prepared.source_membership,
            environment_membership=prepared.environment_membership,
            group_present=prepared.group_present, forcing_scale=self.forcing_scale,
            mode='direct', receiver_versions=_versions((receivers, features, ids)))
        response.receiver_access = (torch.cat(accesses, 1) if accesses else None)
        response.receiver_collective_state = (torch.cat(collective_reads, 1)
                                              if collective_reads else None)
        response.source_features = None
        response.access_mode = prepared._access_mode
        response.affine_capability = True
        return response

    def predict_responses(self, prepared: InteractionPreservingPreparedContext,
                          receivers: torch.Tensor,
                          receiver_features: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        response = self.prepare_receivers(prepared, receivers, receiver_features, **kwargs)
        kernel = response.dense_kernel()
        return kernel[..., 0] if self.affine_outputs == 1 else kernel

    def read_kernel(self, prepared, receivers, **kwargs) -> torch.Tensor:
        return self.predict_responses(prepared, receivers, **kwargs)

    def _assert_response_owned(self, response: PreparedSourceResponse) -> None:
        if not isinstance(response, PreparedSourceResponse):
            raise TypeError('Response must be a PreparedSourceResponse from this core.')
        self.assert_owned(response.context)
        response.assert_fresh()

    def apply_forcing(self, response: PreparedSourceResponse, forcing: torch.Tensor, *,
                      forcing_map=None, return_contributions: bool = False,
                      accumulation_dtype=None):
        self._assert_response_owned(response)
        if forcing.shape != response.context.present.shape or not bool(torch.isfinite(forcing).all()):
            raise ValueError('Forcing must be finite [B,M] aligned with prepared source IDs.')
        mapped = forcing if forcing_map is None else forcing_map(forcing)
        if mapped.shape != forcing.shape or not bool(torch.isfinite(mapped).all()):
            raise ValueError('Mapped forcing must remain finite [B,M].')
        if accumulation_dtype not in (None, torch.float32, torch.float64):
            raise ValueError('Response accumulation dtype must be float32 or float64.')
        kernel = response.dense_kernel(accumulation_dtype=accumulation_dtype)
        if accumulation_dtype is not None:
            offset = response.offset.to(accumulation_dtype)
            values = offset + torch.einsum('bqma,bm->bqa', kernel,
                                           mapped.to(accumulation_dtype))
        else:
            values = response.offset + torch.einsum('bqma,bm->bqa', kernel, mapped)
        result = values[..., 0] if self.affine_outputs == 1 else values
        if not return_contributions:
            return result
        contributions = kernel * mapped[:, None, :, None].to(kernel.dtype)
        return ResponseApplication(result, contributions, {
            'forcing_law': 'affine' if forcing_map is None else 'explicit caller nonlinear map',
            'kernel_coordinates': 'physical forcing' if forcing_map is None else 'mapped forcing',
            'source_ids': response.context.source_ids, 'physical_solves': 0,
        })

    def apply_increment(self, response: PreparedSourceResponse, delta_forcing: torch.Tensor, *,
                        forcing_map=None, baseline_forcing=None, accumulation_dtype=None):
        if self.affine_outputs == 0:
            raise ValueError('Nonlinear Wind outputs have no exact affine finite-increment capability.')
        self._assert_response_owned(response)
        if forcing_map is not None:
            if baseline_forcing is None:
                raise ValueError('Nonlinear forcing increments require an explicit baseline.')
            delta_forcing = (forcing_map(baseline_forcing + delta_forcing) -
                             forcing_map(baseline_forcing))
        if delta_forcing.shape != response.context.present.shape or not bool(
                torch.isfinite(delta_forcing).all()):
            raise ValueError('Forcing increment must be finite [B,M] aligned with prepared source IDs.')
        kernel = response.dense_kernel(accumulation_dtype=accumulation_dtype)
        values = torch.einsum('bqma,bm->bqa', kernel,
                              delta_forcing if accumulation_dtype is None else
                              delta_forcing.to(accumulation_dtype))
        return values[..., 0] if self.affine_outputs == 1 else values
