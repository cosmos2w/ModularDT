"""Joint, source-preserving fields with explicit regional hypergraph context.

The core is dataset independent: adapters supply typed source/environment
records, geometry, physical measures and receiver features; they own units,
output laws and target extraction.  Context preparation never accepts an
applied affine control or an observed field.
"""
from __future__ import annotations

import math
import weakref
import zlib
from dataclasses import dataclass, fields, replace

import torch
from torch import nn

from .interaction_core import (
    DependencySpec,
    InteractionScene,
    PreparedResponseContext,
    _check_versions,
    _geometry,
    _parameter_signature,
    _score_mlp,
    _versions,
)
from .source_response_operator import PreparedSourceResponse, ResponseApplication


def _mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def _normalized_measure(measure: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    mass = measure * present
    total = mass.sum(-1, keepdim=True)
    return torch.where(total > 0, mass / total.clamp_min(torch.finfo(mass.dtype).tiny), torch.zeros_like(mass))


def _masked_measure_softmax(scores: torch.Tensor, measure: torch.Tensor,
                            present: torch.Tensor, edge_present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized donor mass and density, applying physical measure once."""
    if measure.shape[-1] == 0:
        return scores, scores
    mass = measure * present
    active = mass > 0
    log_mass = mass.clamp_min(torch.finfo(scores.dtype).tiny).log()
    logits = scores + log_mass[:, None]
    valid = active[:, None] & (edge_present[..., None] > 0)
    logits = logits.masked_fill(~valid, -torch.inf)
    has_donor = valid.any(-1, keepdim=True)
    maximum = torch.where(has_donor, logits.amax(-1, keepdim=True), torch.zeros_like(logits[..., :1]))
    unnormalized = torch.exp(logits - maximum) * valid
    denominator = unnormalized.sum(-1, keepdim=True)
    donor_mass = torch.where(denominator > 0,
                             unnormalized / denominator.clamp_min(torch.finfo(scores.dtype).tiny),
                             torch.zeros_like(unnormalized))
    density = torch.where(active[:, None],
                          donor_mass / measure.clamp_min(torch.finfo(scores.dtype).tiny)[:, None],
                          torch.zeros_like(donor_mass))
    return donor_mass, density


def _stable_linear_initialization(module: nn.Module, seed: int) -> None:
    """Initialize each Linear from its qualified name, independent of mode-only modules."""
    with torch.no_grad():
        for name, child in module.named_modules():
            if not isinstance(child, nn.Linear):
                continue
            qualified = f'{name}:{child.in_features}:{child.out_features}'.encode()
            local_seed = (int(seed) + zlib.crc32(qualified)) & 0x7FFF_FFFF_FFFF_FFFF
            generator = torch.Generator(device='cpu')
            generator.manual_seed(local_seed)
            nn.init.kaiming_uniform_(child.weight, a=math.sqrt(5), generator=generator)
            if child.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(child.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(child.bias, -bound, bound, generator=generator)


@dataclass
class JointRegionalPreparedContext(PreparedResponseContext):
    """Prepared tensors and inspectable factors for one fixed configuration."""

    source_measures_raw: torch.Tensor | None = None
    environment_measures_raw: torch.Tensor | None = None
    context: torch.Tensor | None = None
    domain_origin: torch.Tensor | None = None
    edge_scale_vectors: torch.Tensor | None = None
    edge_type: torch.Tensor | None = None
    edge_source_ids: torch.Tensor | None = None
    edge_region_ids: torch.Tensor | None = None
    edge_anchor_states: torch.Tensor | None = None
    edge_to_source: torch.Tensor | None = None
    edge_to_environment: torch.Tensor | None = None
    source_update_delta: torch.Tensor | None = None
    environment_update_delta: torch.Tensor | None = None
    source_membership_density: torch.Tensor | None = None
    environment_membership_density: torch.Tensor | None = None
    source_membership_history: tuple[torch.Tensor, ...] = ()
    environment_membership_history: tuple[torch.Tensor, ...] = ()
    source_membership_density_history: tuple[torch.Tensor, ...] = ()
    environment_membership_density_history: tuple[torch.Tensor, ...] = ()
    edge_state_history: tuple[torch.Tensor, ...] = ()
    edge_to_source_history: tuple[torch.Tensor, ...] = ()
    edge_to_environment_history: tuple[torch.Tensor, ...] = ()
    source_update_history: tuple[torch.Tensor, ...] = ()
    environment_update_history: tuple[torch.Tensor, ...] = ()
    _input_args: dict | None = None
    _access_mode: str = 'learned'
    _scene_average_access: torch.Tensor | None = None
    _scene_average_catalogue_ids: torch.Tensor | None = None
    _intervention_receipt: dict | None = None
    _owner_id: int | None = None
    _locality_prior_strength_snapshot: float = 0.0

    def assert_fresh(self):
        super().assert_fresh()
        owner = self.model_reference() if self.model_reference is not None else None
        if owner is not None and owner.locality_prior_strength != self._locality_prior_strength_snapshot:
            raise ValueError('Prepared joint locality configuration changed; rebuild this request.')


class JointRegionalFieldCore(nn.Module):
    """Two typed node-to-edge-to-node blocks and receiver-addressable reads.

    ``J-H`` learns typed incidence and receiver access. ``J-geometry`` uses
    fixed smooth geometry for both. ``J-direct`` updates typed nodes through a
    pooled scene context and has no receiver-addressable edge bank. All modes
    preserve individual physical sources at the receiver readout.
    """

    def __init__(self, source_width: int, context_width: int, environment_width: int, *,
                 spatial_dim: int = 2, hidden: int = 128, message: int = 128,
                 mode: str = 'J-H', regional_anchors: int = 16, depth: int = 2,
                 field_outputs: int = 4, affine_outputs: int = 1,
                 query_width: int = 0, max_sources: int | None = None,
                 forcing_scale: float = 1.0, zero_offset: bool = True,
                 initialization_seed: int | None = None,
                 locality_prior_strength: float = 0.0):
        super().__init__()
        if mode not in ('J-H', 'J-geometry', 'J-direct'):
            raise ValueError("mode must be 'J-H', 'J-geometry' or 'J-direct'.")
        if spatial_dim not in (2, 3):
            raise ValueError('spatial_dim must be 2 or 3.')
        if min(source_width, context_width, environment_width, hidden, message, depth, field_outputs) < 1:
            raise ValueError('Input, hidden, message, depth and field widths must be positive.')
        if query_width < 0 or affine_outputs < 0 or regional_anchors < 0:
            raise ValueError('query_width, affine_outputs and regional_anchors must be nonnegative.')
        if mode != 'J-direct' and regional_anchors < 1:
            raise ValueError('Regional hypergraph modes require at least one registered regional anchor.')
        if max_sources is not None and max_sources < 1:
            raise ValueError('max_sources must be positive when specified.')
        if not math.isfinite(forcing_scale) or forcing_scale <= 0:
            raise ValueError('forcing_scale must be finite and positive.')
        if not math.isfinite(locality_prior_strength) or locality_prior_strength < 0:
            raise ValueError('locality_prior_strength must be finite and nonnegative.')
        if locality_prior_strength and mode != 'J-H':
            raise ValueError('A learned locality prior applies only to J-H.')

        self.spatial_dim = spatial_dim
        self.hidden = hidden
        self.message = message
        self.mode = mode
        self.depth = depth
        self.regional_anchors = regional_anchors
        self.field_outputs = field_outputs
        self.affine_outputs = affine_outputs
        self.output_law = 'joint'
        self.field_output_law = 'nonlinear'
        self.response_output_law = 'affine' if affine_outputs else None
        self.output_capabilities = {
            'configuration_fields': {'law': 'nonlinear', 'outputs': field_outputs},
            'affine_control': ({'law': 'affine', 'outputs': affine_outputs}
                               if affine_outputs else None),
        }
        self.query_width = query_width
        self.max_sources = max_sources
        self.forcing_scale = float(forcing_scale)
        self.zero_offset = bool(zero_offset)
        self.locality_prior_strength = float(locality_prior_strength)
        self.config = {
            'source_width': source_width, 'context_width': context_width,
            'environment_width': environment_width, 'spatial_dim': spatial_dim,
            'hidden': hidden, 'message': message, 'mode': mode,
            'regional_anchors': regional_anchors, 'depth': depth,
            'field_outputs': field_outputs, 'affine_outputs': affine_outputs,
            'query_width': query_width, 'max_sources': max_sources,
            'forcing_scale': float(forcing_scale), 'zero_offset': bool(zero_offset),
        }
        # Omit the optional zero value so v1 checkpoint identities and exact
        # resumes retain their original configuration bytes.
        if self.locality_prior_strength:
            self.config['locality_prior_strength'] = self.locality_prior_strength
        dgeom = spatial_dim * 9

        # Common input and readout modules keep identical qualified names and
        # initialization across J-H and J-geometry.
        self.source_encoder = _mlp(source_width + context_width, hidden, hidden)
        self.environment_encoder = _mlp(environment_width + context_width, hidden, hidden)
        self.global_encoder = _mlp(2 * hidden + context_width + 3, hidden, hidden)
        self.source_read = _mlp(3 * hidden + 3 * dgeom + query_width, message, hidden)
        self.field_head = _mlp(message + 2 * hidden + dgeom + query_width, field_outputs, hidden)
        self.affine_head = _mlp(message, affine_outputs, hidden) if affine_outputs else None
        self.offset_head = (None if zero_offset else
                            _mlp(2 * hidden + dgeom + query_width, affine_outputs, hidden)) if affine_outputs else None

        if mode == 'J-direct':
            self.direct_source_updates = nn.ModuleList(
                [_mlp(3 * hidden + context_width, hidden, hidden) for _ in range(depth)])
            self.direct_environment_updates = nn.ModuleList(
                [_mlp(3 * hidden + context_width, hidden, hidden) for _ in range(depth)])
            self.direct_global_updates = nn.ModuleList(
                [_mlp(3 * hidden + context_width + 3, hidden, hidden) for _ in range(depth)])
        else:
            self.anchor_encoder = _mlp(2 * dgeom + 1 + context_width, hidden, hidden)
            self.source_content = nn.ModuleList([_mlp(hidden + dgeom, hidden, hidden) for _ in range(depth)])
            self.environment_content = nn.ModuleList([_mlp(hidden + dgeom, hidden, hidden) for _ in range(depth)])
            self.edge_updates = nn.ModuleList([_mlp(4 * hidden + context_width + 3, hidden, hidden)
                                               for _ in range(depth)])
            self.source_node_updates = nn.ModuleList([_mlp(3 * hidden + context_width, hidden, hidden)
                                                      for _ in range(depth)])
            self.environment_node_updates = nn.ModuleList([_mlp(3 * hidden + context_width, hidden, hidden)
                                                           for _ in range(depth)])
            self.global_updates = nn.ModuleList([_mlp(3 * hidden + context_width + 3, hidden, hidden)
                                                 for _ in range(depth)])
            if mode == 'J-H':
                score_width = 2 * hidden + dgeom
                self.source_membership_score = _score_mlp(score_width, hidden)
                self.environment_membership_score = _score_mlp(score_width, hidden)
                access_width = min(hidden, 64)
                self.receiver_query = _mlp(context_width + dgeom + query_width, access_width, hidden)
                self.edge_key = nn.Linear(hidden, access_width, bias=False)
                # A scalar additive bias cancels exactly inside the edge
                # softmax, so omit the final bias to avoid an inert parameter.
                self.access_geometry = _score_mlp(dgeom, hidden)

        seed = torch.initial_seed() if initialization_seed is None else int(initialization_seed)
        _stable_linear_initialization(self, seed)

    @staticmethod
    def _validate_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f'{name} shape must be {tuple(shape)}, received {tuple(value.shape)}.')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must contain finite values.')

    @staticmethod
    def _type_mass(measure: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        return _normalized_measure(measure, present)

    @staticmethod
    def _scene_statistics(source_measures: torch.Tensor, present: torch.Tensor,
                          environment_measures: torch.Tensor,
                          environment_present: torch.Tensor) -> torch.Tensor:
        source_mass = source_measures * present
        environment_mass = environment_measures * environment_present
        source_count = present.sum(-1)
        return torch.stack((torch.log1p(source_count),
                            torch.log1p(source_mass.sum(-1)),
                            torch.log1p(environment_mass.sum(-1))), -1)

    @staticmethod
    def _pool(nodes: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        return (nodes * mass[..., None]).sum(1)

    def _regional_anchors(self, lengths: torch.Tensor,
                          domain_origin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create deterministic low-discrepancy anchors and scales in domain units."""
        batch, dimension = lengths.shape
        count = self.regional_anchors
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
        scales = scale[:, None].expand(-1, count, -1).clamp_min(torch.finfo(lengths.dtype).eps)
        return anchors, scales

    def _make_edges(self, context: torch.Tensor, centers: torch.Tensor,
                    source_lengths: torch.Tensor, source_states: torch.Tensor,
                    lengths: torch.Tensor, domain_origin: torch.Tensor,
                    present: torch.Tensor, source_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, modules, dimension = centers.shape
        regional_centers, regional_scales = self._regional_anchors(lengths, domain_origin)
        source_scales = source_lengths[..., None].expand(-1, -1, dimension)
        edge_centers = torch.cat((centers, regional_centers), 1)
        edge_scales = torch.cat((source_scales, regional_scales), 1)
        edge_present = torch.cat((present, present.new_ones(batch, self.regional_anchors)), 1)
        edge_type = torch.cat((present.new_zeros(batch, modules), present.new_ones(batch, self.regional_anchors)), 1)
        edge_ids = torch.cat((torch.where(present > 0, source_ids,
                                          torch.full((batch, modules), -1, device=centers.device, dtype=torch.long)),
                              torch.full((batch, self.regional_anchors), -1, device=centers.device, dtype=torch.long)), 1)
        region_ids = torch.cat((torch.full((batch, modules), -1, device=centers.device, dtype=torch.long),
                                torch.arange(self.regional_anchors, device=centers.device)[None].expand(batch, -1)), 1)
        scale_safe = edge_scales / lengths[:, None].clamp_min(torch.finfo(lengths.dtype).eps)
        anchor_inputs = torch.cat((_geometry((edge_centers - domain_origin[:, None]) / lengths[:, None]),
                                  _geometry(scale_safe), edge_type[..., None],
                                  context[:, None].expand(-1, edge_centers.shape[1], -1)), -1)
        anchor_states = self.anchor_encoder(anchor_inputs)
        if modules:
            anchor_states = torch.cat((anchor_states[:, :modules] + source_states,
                                       anchor_states[:, modules:]), 1)
        anchor_states = anchor_states * edge_present[..., None]
        return edge_centers, edge_scales, edge_present, edge_type, edge_ids, region_ids, anchor_states

    @staticmethod
    def _geometry_scores(node_coords: torch.Tensor, edge_centers: torch.Tensor,
                         edge_scales: torch.Tensor) -> torch.Tensor:
        if edge_centers.shape[1] == 0 or node_coords.shape[1] == 0:
            return node_coords.new_empty(node_coords.shape[0], edge_centers.shape[1], node_coords.shape[1])
        relative = (node_coords[:, None] - edge_centers[:, :, None]) / edge_scales[:, :, None].clamp_min(1e-12)
        return -0.5 * relative.square().sum(-1)

    def _memberships(self, source_states: torch.Tensor, environment_states: torch.Tensor,
                     centers: torch.Tensor, source_lengths: torch.Tensor,
                     environment_coords: torch.Tensor, edge_centers: torch.Tensor,
                     edge_scales: torch.Tensor, edge_states: torch.Tensor,
                     edge_present: torch.Tensor, source_measures: torch.Tensor,
                     present: torch.Tensor, environment_measures: torch.Tensor,
                     environment_present: torch.Tensor, incidence_mode: str
                     ) -> tuple[torch.Tensor, ...]:
        if edge_centers.shape[1] == 0:
            b_source = source_states.new_empty(source_states.shape[0], 0, source_states.shape[1])
            b_environment = environment_states.new_empty(environment_states.shape[0], 0, environment_states.shape[1])
            return b_source, b_environment, b_source, b_environment
        if incidence_mode not in ('learned', 'geometry'):
            raise ValueError("incidence_mode must be 'learned' or 'geometry'.")
        if incidence_mode == 'geometry':
            score_source = self._geometry_scores(centers, edge_centers, edge_scales)
            score_environment = self._geometry_scores(environment_coords, edge_centers, edge_scales)
        else:
            source_relative = _geometry((centers[:, None] - edge_centers[:, :, None]) /
                                         edge_scales[:, :, None].clamp_min(1e-12))
            environment_relative = _geometry((environment_coords[:, None] - edge_centers[:, :, None]) /
                                             edge_scales[:, :, None].clamp_min(1e-12))
            source_inputs = torch.cat((source_states[:, None].expand(-1, edge_centers.shape[1], -1, -1),
                                       edge_states[:, :, None].expand(-1, -1, source_states.shape[1], -1),
                                       source_relative), -1)
            environment_inputs = torch.cat((environment_states[:, None].expand(-1, edge_centers.shape[1], -1, -1),
                                            edge_states[:, :, None].expand(-1, -1, environment_states.shape[1], -1),
                                            environment_relative), -1)
            score_source = self.source_membership_score(source_inputs).squeeze(-1)
            score_environment = self.environment_membership_score(environment_inputs).squeeze(-1)
            if self.locality_prior_strength:
                score_source = score_source + self.locality_prior_strength * self._geometry_scores(
                    centers, edge_centers, edge_scales)
                score_environment = score_environment + self.locality_prior_strength * self._geometry_scores(
                    environment_coords, edge_centers, edge_scales)
        b_source, d_source = _masked_measure_softmax(score_source, source_measures, present, edge_present)
        b_environment, d_environment = _masked_measure_softmax(
            score_environment, environment_measures, environment_present, edge_present)
        return b_source, b_environment, d_source, d_environment

    @staticmethod
    def _matched_mass_shuffle(membership: torch.Tensor, measure: torch.Tensor,
                              present: torch.Tensor,
                              physical_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Permute donors per edge within equal-measure/equal-presence buckets.

        The per-edge mass histogram is unchanged exactly.  Restricting each
        permutation to equal physical measures and presence weights keeps the
        physical-measure density well defined after donor reassignment. Donor
        ordering is deterministic from physical IDs, with a stable slot-order
        tie break.
        """
        if membership.shape[-1] == 0:
            return membership
        batches, _edges, nodes = membership.shape
        if measure.shape != (batches, nodes) or present.shape != (batches, nodes):
            raise ValueError('Matched-mass shuffle measure/presence must align with donor nodes.')
        if physical_ids is None:
            physical_ids = torch.arange(nodes, device=membership.device)[None].expand(batches, -1)
        if physical_ids.shape != (batches, nodes):
            raise ValueError('Matched-mass shuffle IDs must align with donor nodes.')
        rows = []
        for batch_index in range(batches):
            donor_permutation = torch.arange(nodes, dtype=torch.long, device=membership.device)[None].expand(
                membership.shape[1], -1).clone()
            active = (measure[batch_index] > 0) & (present[batch_index] > 0)
            if bool(active.any()):
                bucket_keys = torch.stack((measure[batch_index], present[batch_index]), -1)
                _unique_keys, bucket_ids = torch.unique(
                    bucket_keys, sorted=True, return_inverse=True, dim=0
                )
                for bucket_id in torch.unique(bucket_ids[active], sorted=True):
                    bucket = torch.nonzero(active & (bucket_ids == bucket_id), as_tuple=False).flatten()
                    if bucket.numel() < 2:
                        continue
                    stable_ids = physical_ids[batch_index, bucket]
                    ordered = bucket[torch.argsort(stable_ids, stable=True)]
                    for edge_index in range(membership.shape[1]):
                        shift = 1 + edge_index % (bucket.numel() - 1)
                        donor_permutation[edge_index, ordered] = ordered.roll(shift)
            rows.append(membership[batch_index].gather(-1, donor_permutation))
        return torch.stack(rows, 0)

    @staticmethod
    def _preserve_context_metadata(original: JointRegionalPreparedContext,
                                   rebuilt: JointRegionalPreparedContext) -> JointRegionalPreparedContext:
        """Carry adapter-owned identity/dependency sidecars across rebuilds."""
        declared = {field.name for field in fields(rebuilt)}
        for name, value in vars(original).items():
            if name not in declared:
                setattr(rebuilt, name, value)
        return rebuilt

    @staticmethod
    def _membership_density(membership: torch.Tensor, measure: torch.Tensor,
                             present: torch.Tensor) -> torch.Tensor:
        """Recover typed physical density from incidence mass and raw measure."""
        active = (measure * present) > 0
        density = membership / measure.clamp_min(torch.finfo(membership.dtype).tiny)[:, None]
        return torch.where(active[:, None], density, torch.zeros_like(density))

    @staticmethod
    def _edge_to_node(membership: torch.Tensor, edge_states: torch.Tensor,
                      node_present: torch.Tensor) -> torch.Tensor:
        if membership.shape[1] == 0:
            return edge_states.new_zeros(edge_states.shape[0], membership.shape[2], edge_states.shape[-1])
        incoming = membership.transpose(1, 2)
        incoming = incoming / incoming.sum(-1, keepdim=True).clamp_min(torch.finfo(incoming.dtype).tiny)
        incoming = incoming * node_present[..., None]
        return torch.bmm(incoming, edge_states)

    def _prepare_context_impl(self, *, sources, context, centers, present, lengths, source_lengths,
                              source_measures=None, environment_tokens=None, environment_coords=None,
                              environment_present=None, environment_measures=None, source_ids=None,
                              environment_embedding=None, domain_origin=None, incidence_mode=None,
                              remove_collective_content=False, remove_collective_update=False,
                              matched_mass_shuffle=False, block_index=None,
                              frozen_incidence_plan=None, receiver_access='learned',
                              scene_average_access=None, scene_average_catalogue_ids=None,
                              scene_average_catalogue_versions=()):
        batch, modules, dimension = centers.shape
        selected_block_index = block_index
        if block_index is not None:
            if isinstance(block_index, bool) or not isinstance(block_index, int):
                raise TypeError('block_index must be a zero-based integer or None for all blocks.')
            if not (remove_collective_content or remove_collective_update):
                raise ValueError('block_index is only valid for collective content/update ablations.')
            if block_index < 0 or block_index >= self.depth:
                raise ValueError(f'block_index must lie in [0,{self.depth - 1}].')
        if matched_mass_shuffle:
            if (incidence_mode is not None or remove_collective_content or remove_collective_update or
                    block_index is not None):
                raise ValueError('matched_mass_shuffle is a separate intervention using the prepared incidence histories.')
            if frozen_incidence_plan is None:
                raise ValueError('matched_mass_shuffle requires a frozen baseline incidence plan.')
            if len(frozen_incidence_plan) != self.depth:
                raise ValueError('Frozen incidence plan must provide source/environment masses for every block.')
            if self.mode == 'J-direct':
                raise ValueError('J-direct has no typed regional incidence to shuffle.')
        elif frozen_incidence_plan is not None:
            raise ValueError('A frozen incidence plan is only valid for matched_mass_shuffle.')
        if dimension != self.spatial_dim or (self.max_sources is not None and modules > self.max_sources):
            raise ValueError('Source coordinates/capacity disagree with the model configuration.')
        if source_lengths.ndim == 3:
            source_lengths = source_lengths.squeeze(-1)
        self._validate_shape(centers, (batch, modules, self.spatial_dim), 'centers')
        self._validate_shape(sources, (batch, modules, self.config['source_width']), 'sources')
        self._validate_shape(context, (batch, self.config['context_width']), 'context')
        self._validate_shape(present, (batch, modules), 'present')
        self._validate_shape(lengths, (batch, dimension), 'lengths')
        self._validate_shape(source_lengths, (batch, modules), 'source_lengths')
        if bool((lengths <= 0).any()) or bool(((source_lengths <= 0) & (present > 0)).any()):
            raise ValueError('Domain lengths and active source characteristic lengths must be positive.')
        if bool(((present < 0) | (present > 1)).any()):
            raise ValueError('present must lie between zero and one.')
        source_measures = torch.ones_like(present) if source_measures is None else source_measures
        self._validate_shape(source_measures, present.shape, 'source_measures')
        if bool((source_measures < 0).any()) or bool(((source_measures <= 0) & (present > 0)).any()):
            raise ValueError('Source measures must be positive for active sources and nonnegative elsewhere.')
        if environment_tokens is None:
            environment_tokens = sources.new_empty(batch, 0, self.config['environment_width'])
            environment_coords = centers.new_empty(batch, 0, dimension)
        if environment_coords is None:
            raise ValueError('Environmental tokens require their physical coordinates.')
        environments = environment_tokens.shape[1]
        environment_present = sources.new_ones(batch, environments) if environment_present is None else environment_present
        environment_measures = torch.ones_like(environment_present) if environment_measures is None else environment_measures
        self._validate_shape(environment_tokens, (batch, environments, self.config['environment_width']), 'environment_tokens')
        self._validate_shape(environment_coords, (batch, environments, dimension), 'environment_coords')
        self._validate_shape(environment_present, (batch, environments), 'environment_present')
        self._validate_shape(environment_measures, (batch, environments), 'environment_measures')
        if bool((environment_measures < 0).any()) or bool(((environment_present < 0) | (environment_present > 1)).any()):
            raise ValueError('Environmental measures must be nonnegative and presence must lie in [0,1].')
        if bool(((environment_measures <= 0) & (environment_present > 0)).any()):
            raise ValueError('Environment measures must be positive for active records.')
        if environment_embedding is not None:
            self._validate_shape(environment_embedding, (batch, environments, self.hidden), 'environment_embedding')
        source_ids = (torch.arange(modules, device=centers.device)[None].expand(batch, -1)
                      if source_ids is None else source_ids)
        self._validate_shape(source_ids, (batch, modules), 'source_ids')
        if source_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('source_ids must be integer physical identities.')
        domain_origin = centers.new_zeros(batch, dimension) if domain_origin is None else domain_origin
        self._validate_shape(domain_origin, (batch, dimension), 'domain_origin')
        if receiver_access not in ('learned', 'geometry', 'scene-average', 'uniform-average'):
            raise ValueError('Unknown receiver access mode.')
        edge_slots = 0 if self.mode == 'J-direct' else self.regional_anchors + modules
        if scene_average_access is not None:
            self._validate_shape(scene_average_access, (batch, edge_slots), 'scene_average_access')
            if bool((scene_average_access < 0).any()):
                raise ValueError('scene_average_access must be nonnegative.')
            valid_edges = (torch.cat((present > 0,
                                      torch.ones(batch, self.regional_anchors,
                                                 dtype=torch.bool, device=present.device)), -1)
                           if self.mode != 'J-direct' else
                           torch.empty(batch, 0, dtype=torch.bool, device=present.device))
            if bool((scene_average_access.masked_select(~valid_edges) != 0).any()):
                raise ValueError('scene_average_access must be zero for invalid edge slots.')
            totals = scene_average_access.sum(-1)
            if valid_edges.shape[-1] and bool((valid_edges.any(-1) & ~torch.isclose(
                    totals, torch.ones_like(totals), rtol=1e-5, atol=1e-6)).any()):
                raise ValueError('scene_average_access must sum to one for each nonempty scene.')
        elif receiver_access == 'scene-average':
            raise ValueError('scene-average access requires an explicit fixed-catalogue average.')
        input_versions = _versions((sources, context, centers, present, lengths, source_lengths,
                                    source_measures, environment_tokens, environment_coords,
                                    environment_present, environment_measures, source_ids,
                                    environment_embedding, domain_origin, scene_average_access,
                                    scene_average_catalogue_ids)) + tuple(scene_average_catalogue_versions)
        input_args = {
            'sources': sources, 'context': context, 'centers': centers, 'present': present,
            'lengths': lengths, 'source_lengths': source_lengths, 'source_measures': source_measures,
            'environment_tokens': environment_tokens, 'environment_coords': environment_coords,
            'environment_present': environment_present, 'environment_measures': environment_measures,
            'source_ids': source_ids, 'environment_embedding': environment_embedding,
            'domain_origin': domain_origin,
        }
        source_mass = self._type_mass(source_measures, present)
        environment_mass = self._type_mass(environment_measures, environment_present)
        source_context = context[:, None].expand(-1, modules, -1)
        environment_context = context[:, None].expand(-1, environments, -1)
        source_states = self.source_encoder(torch.cat((sources, source_context), -1)) * present[..., None]
        environment_states = self.environment_encoder(torch.cat((environment_tokens, environment_context), -1))
        if environment_embedding is not None:
            environment_states = environment_states + environment_embedding
        environment_states = environment_states * environment_present[..., None]
        stats = self._scene_statistics(source_measures, present, environment_measures, environment_present)
        source_pool = self._pool(source_states, source_mass)
        environment_pool = self._pool(environment_states, environment_mass)
        global_state = self.global_encoder(torch.cat((source_pool, environment_pool, context, stats), -1))

        if self.mode == 'J-direct':
            if receiver_access not in ('learned', 'geometry', 'scene-average'):
                raise ValueError('Unknown receiver access intervention.')
            for source_update, environment_update, global_update in zip(
                    self.direct_source_updates, self.direct_environment_updates, self.direct_global_updates):
                source_delta = source_update(torch.cat((source_states,
                    environment_pool[:, None].expand(-1, modules, -1),
                    global_state[:, None].expand(-1, modules, -1), source_context), -1)) * present[..., None]
                environment_delta = environment_update(torch.cat((environment_states,
                    source_pool[:, None].expand(-1, environments, -1),
                    global_state[:, None].expand(-1, environments, -1), environment_context), -1)) * environment_present[..., None]
                source_states = (source_states + source_delta) * present[..., None]
                environment_states = (environment_states + environment_delta) * environment_present[..., None]
                source_pool = self._pool(source_states, source_mass)
                environment_pool = self._pool(environment_states, environment_mass)
                global_state = global_state + global_update(torch.cat((source_pool, environment_pool,
                                                                       global_state, context, stats), -1))
            edges = centers.new_empty(batch, 0, dimension)
            edge_scales = centers.new_empty(batch, 0, dimension)
            edge_present = present.new_empty(batch, 0)
            edge_type = present.new_empty(batch, 0)
            edge_ids = torch.empty(batch, 0, dtype=torch.long, device=centers.device)
            region_ids = torch.empty(batch, 0, dtype=torch.long, device=centers.device)
            edge_states = source_states.new_empty(batch, 0, self.hidden)
            edge_anchor_states = edge_states
            b_source = source_states.new_empty(batch, 0, modules)
            b_environment = environment_states.new_empty(batch, 0, environments)
            d_source, d_environment = b_source, b_environment
            edge_to_source = source_states.new_zeros(batch, modules, self.hidden)
            edge_to_environment = environment_states.new_zeros(batch, environments, self.hidden)
            source_delta = source_states.new_zeros(batch, modules, self.hidden)
            environment_delta = environment_states.new_zeros(batch, environments, self.hidden)
            source_history, environment_history = (), ()
            source_density_history, environment_density_history = (), ()
            edge_state_history, edge_to_source_history, edge_to_environment_history = (), (), ()
            source_update_history, environment_update_history = (), ()
            group_length = centers.new_empty(batch, 0)
        else:
            edges, edge_scales, edge_present, edge_type, edge_ids, region_ids, edge_anchor_states = self._make_edges(
                context, centers, source_lengths, source_states, lengths, domain_origin, present, source_ids)
            incidence_mode = incidence_mode or ('learned' if self.mode == 'J-H' else 'geometry')
            if incidence_mode == 'learned' and self.mode != 'J-H':
                raise ValueError('This model has no learned incidence parameters.')
            if incidence_mode == 'learned' and self.mode == 'J-geometry':
                raise ValueError('J-geometry cannot use learned incidence.')
            source_history_list, environment_history_list = [], []
            source_density_history_list, environment_density_history_list = [], []
            edge_state_history_list, edge_to_source_history_list, edge_to_environment_history_list = [], [], []
            source_update_history_list, environment_update_history_list = [], []
            edge_to_source = source_states.new_zeros(batch, modules, self.hidden)
            edge_to_environment = environment_states.new_zeros(batch, environments, self.hidden)
            source_delta = source_states.new_zeros(batch, modules, self.hidden)
            environment_delta = environment_states.new_zeros(batch, environments, self.hidden)
            edge_states = edge_anchor_states
            blocks = zip(self.source_content, self.environment_content, self.edge_updates,
                         self.source_node_updates, self.environment_node_updates, self.global_updates,
                         strict=True)
            for current_block_index, (source_content, environment_content, edge_update, source_update,
                                      environment_update, global_update) in enumerate(blocks):
                if frozen_incidence_plan is None:
                    b_source, b_environment, d_source, d_environment = self._memberships(
                        source_states, environment_states, centers, source_lengths, environment_coords,
                        edges, edge_scales, edge_states, edge_present, source_measures, present,
                        environment_measures, environment_present, incidence_mode)
                else:
                    baseline_source, baseline_environment = frozen_incidence_plan[current_block_index]
                    if baseline_source.shape != (batch, edges.shape[1], modules):
                        raise ValueError('Frozen source incidence does not match this block and scene.')
                    if baseline_environment.shape != (batch, edges.shape[1], environments):
                        raise ValueError('Frozen environment incidence does not match this block and scene.')
                    b_source = self._matched_mass_shuffle(
                        baseline_source, source_measures, present, source_ids)
                    environment_ids = torch.arange(environments, device=centers.device)[None].expand(
                        batch, -1)
                    b_environment = self._matched_mass_shuffle(
                        baseline_environment, environment_measures, environment_present, environment_ids)
                    d_source = self._membership_density(b_source, source_measures, present)
                    d_environment = self._membership_density(
                        b_environment, environment_measures, environment_present)
                source_history_list.append(b_source)
                environment_history_list.append(b_environment)
                source_density_history_list.append(d_source)
                environment_density_history_list.append(d_environment)
                relative_source = _geometry((centers[:, None] - edges[:, :, None]) /
                                            edge_scales[:, :, None].clamp_min(1e-12))
                relative_environment = _geometry((environment_coords[:, None] - edges[:, :, None]) /
                                                 edge_scales[:, :, None].clamp_min(1e-12))
                content_source = source_content(torch.cat((source_states[:, None].expand(-1, edges.shape[1], -1, -1),
                                                          relative_source), -1))
                content_environment = environment_content(torch.cat((environment_states[:, None].expand(-1, edges.shape[1], -1, -1),
                                                                      relative_environment), -1))
                pool_source = (b_source[..., None] * content_source).sum(2)
                pool_environment = (b_environment[..., None] * content_environment).sum(2)
                edge_delta = edge_update(torch.cat((edge_states, pool_source, pool_environment,
                                                    global_state[:, None].expand(-1, edges.shape[1], -1),
                                                    context[:, None].expand(-1, edges.shape[1], -1),
                                                    stats[:, None].expand(-1, edges.shape[1], -1)), -1))
                edge_states = (edge_states + edge_delta) * edge_present[..., None]
                block_ablation = (selected_block_index is None or
                                  selected_block_index == current_block_index)
                if block_ablation and remove_collective_content:
                    edge_states = edge_anchor_states
                edge_state_history_list.append(edge_states)
                edge_to_source = self._edge_to_node(b_source, edge_states, present)
                edge_to_environment = self._edge_to_node(b_environment, edge_states, environment_present)
                if block_ablation and remove_collective_update:
                    edge_to_source = torch.zeros_like(edge_to_source)
                    edge_to_environment = torch.zeros_like(edge_to_environment)
                edge_to_source_history_list.append(edge_to_source)
                edge_to_environment_history_list.append(edge_to_environment)
                source_delta = source_update(torch.cat((source_states, edge_to_source,
                    global_state[:, None].expand(-1, modules, -1), source_context), -1)) * present[..., None]
                environment_delta = environment_update(torch.cat((environment_states, edge_to_environment,
                    global_state[:, None].expand(-1, environments, -1), environment_context), -1)) * environment_present[..., None]
                if block_ablation and remove_collective_update:
                    source_delta = torch.zeros_like(source_delta)
                    environment_delta = torch.zeros_like(environment_delta)
                source_update_history_list.append(source_delta)
                environment_update_history_list.append(environment_delta)
                source_states = (source_states + source_delta) * present[..., None]
                environment_states = (environment_states + environment_delta) * environment_present[..., None]
                source_pool = self._pool(source_states, source_mass)
                environment_pool = self._pool(environment_states, environment_mass)
                global_state = global_state + global_update(torch.cat((source_pool, environment_pool,
                                                                       global_state, context, stats), -1))
            source_history, environment_history = tuple(source_history_list), tuple(environment_history_list)
            source_density_history = tuple(source_density_history_list)
            environment_density_history = tuple(environment_density_history_list)
            edge_state_history = tuple(edge_state_history_list)
            edge_to_source_history = tuple(edge_to_source_history_list)
            edge_to_environment_history = tuple(edge_to_environment_history_list)
            source_update_history = tuple(source_update_history_list)
            environment_update_history = tuple(environment_update_history_list)
            group_length = edge_scales.mean(-1)

        prepared = JointRegionalPreparedContext(
            source_states=source_states, environment_states=environment_states, global_state=global_state,
            centers=centers, source_lengths=source_lengths, present=present, source_measure=source_mass,
            environment_coords=environment_coords, environment_present=environment_present,
            environment_measure=environment_mass, lengths=lengths, source_ids=source_ids,
            group_states=edge_states, group_centers=edges, group_lengths=group_length,
            group_present=edge_present, source_membership=b_source, environment_membership=b_environment,
            input_versions=input_versions, model_reference=weakref.ref(self),
            parameter_signature=_parameter_signature(self), source_measures_raw=source_measures,
            environment_measures_raw=environment_measures, context=context, domain_origin=domain_origin,
            edge_scale_vectors=edge_scales,
            edge_type=edge_type, edge_source_ids=edge_ids, edge_region_ids=region_ids,
            edge_anchor_states=edge_anchor_states, edge_to_source=edge_to_source,
            edge_to_environment=edge_to_environment, source_update_delta=source_delta,
            environment_update_delta=environment_delta, source_membership_density=d_source,
            environment_membership_density=d_environment, source_membership_history=source_history,
            environment_membership_history=environment_history,
            source_membership_density_history=source_density_history,
            environment_membership_density_history=environment_density_history,
            edge_state_history=edge_state_history, edge_to_source_history=edge_to_source_history,
            edge_to_environment_history=edge_to_environment_history,
            source_update_history=source_update_history,
            environment_update_history=environment_update_history, _input_args=input_args,
            _access_mode=receiver_access, _scene_average_access=scene_average_access,
            _scene_average_catalogue_ids=scene_average_catalogue_ids,
            _intervention_receipt={
                'incidence': incidence_mode if self.mode != 'J-direct' else 'none',
                'remove_collective_content': bool(remove_collective_content),
                'remove_collective_update': bool(remove_collective_update),
                'block_index': selected_block_index,
                'ablation_scope': (
                    'none' if not (remove_collective_content or remove_collective_update) else
                    'all_blocks' if selected_block_index is None else 'single_block'
                ),
                'matched_mass_shuffle': bool(matched_mass_shuffle),
                'frozen_plan': bool(matched_mass_shuffle),
                'propagated_updates': bool(matched_mass_shuffle),
                'receiver_access': receiver_access,
                'scene_average_catalogue_ids': scene_average_catalogue_ids,
            }, _owner_id=id(self),
            _locality_prior_strength_snapshot=self.locality_prior_strength)
        return prepared

    def prepare_context(self, sources, context, centers, present, lengths, source_lengths,
                        source_measures=None, environment_tokens=None, environment_coords=None,
                        environment_present=None, environment_measures=None, source_ids=None,
                        environment_embedding=None, domain_origin=None):
        """Prepare a configuration from explicit target-free tensors.

        The input widths and tensor contract match ``InteractionContextCore``:
        sources[B,M,F_S], context[B,C], centers[B,M,D], present[B,M],
        lengths[B,D], source_lengths[B,M], and optional environmental records
        [B,E,F_E] with coordinates/measures. ``domain_origin[B,D]`` is optional
        and defaults to zero. No forcing, observed output or query-dependent
        graph input enters this prepared state.
        """
        return self._prepare_context_impl(sources=sources, context=context, centers=centers,
            present=present, lengths=lengths, source_lengths=source_lengths,
            source_measures=source_measures, environment_tokens=environment_tokens,
            environment_coords=environment_coords, environment_present=environment_present,
            environment_measures=environment_measures, source_ids=source_ids,
            environment_embedding=environment_embedding, domain_origin=domain_origin)

    def prepare(self, scene: InteractionScene) -> JointRegionalPreparedContext:
        if not isinstance(scene, InteractionScene):
            raise TypeError('prepare expects an InteractionScene with a declared dependency.')
        if not isinstance(scene.dependency, DependencySpec):
            raise TypeError('InteractionScene dependency must be a DependencySpec.')
        if scene.dependency.output_law == 'affine' and self.response_output_law != 'affine':
            raise ValueError('Scene requests an affine response block that this model does not provide.')
        prepared = self.prepare_context(**scene.tensors())
        prepared.dependency = scene.dependency
        return prepared

    def assert_owned(self, prepared: JointRegionalPreparedContext) -> None:
        if not isinstance(prepared, JointRegionalPreparedContext):
            raise TypeError('Prepared context must come from JointRegionalFieldCore.')
        prepared.assert_fresh()
        if prepared._owner_id != id(self):
            raise ValueError('Prepared regional context belongs to another model.')

    def intervene_context(self, prepared: JointRegionalPreparedContext, *,
                          incidence_mode: str | None = None,
                          receiver_access: str | None = None,
                          scene_average_access=None,
                          remove_collective_content: bool = False,
                          remove_collective_update: bool = False,
                          matched_mass_shuffle: bool = False,
                          block_index: int | None = None) -> JointRegionalPreparedContext:
        """Rebuild an actual fixed-weight intervention from the same scene inputs."""
        self.assert_owned(prepared)
        allowed_access = ('learned', 'geometry', 'scene-average', 'uniform-average')
        if receiver_access is not None and receiver_access not in allowed_access:
            raise ValueError(f'receiver_access must be one of {allowed_access}.')
        if matched_mass_shuffle and incidence_mode is not None:
            raise ValueError('matched_mass_shuffle uses the prepared baseline incidence histories.')
        if matched_mass_shuffle and (remove_collective_content or remove_collective_update or block_index is not None):
            raise ValueError('matched_mass_shuffle must be evaluated separately from collective ablations.')
        scene_ids = None
        scene_versions = ()
        if isinstance(scene_average_access, dict):
            metadata = scene_average_access
            if metadata.get('_owner_id') != id(self):
                raise ValueError('Scene-average catalogue result belongs to another core.')
            if metadata.get('_prepared_context') is not prepared:
                raise ValueError('Scene-average catalogue result belongs to another prepared context.')
            scene_average_access = metadata['access']
            scene_ids = metadata['receiver_ids']
            scene_versions = tuple(metadata['_catalogue_versions'])
            _check_versions(scene_versions)
        elif scene_average_access is not None:
            raise TypeError('scene_average_access must be the result of average_receiver_access().')
        selected_mode = receiver_access or prepared._access_mode
        if selected_mode == 'scene-average':
            if scene_average_access is None:
                if prepared._access_mode != 'scene-average' or prepared._scene_average_access is None:
                    raise ValueError('scene-average access requires an explicit fixed-catalogue average.')
                scene_average_access = prepared._scene_average_access
                scene_ids = prepared._scene_average_catalogue_ids
            elif scene_ids is None:
                scene_ids = torch.empty(0, dtype=torch.long, device=scene_average_access.device)
        elif scene_average_access is not None:
            raise ValueError('scene_average_access is only valid with receiver_access="scene-average".')
        if incidence_mode is None and not (remove_collective_content or remove_collective_update or
                                           matched_mass_shuffle or block_index is not None):
            new_versions = _versions((scene_average_access, scene_ids)) + tuple(scene_versions)
            return self._preserve_context_metadata(prepared, replace(prepared, _access_mode=selected_mode,
                           _scene_average_access=scene_average_access,
                           _scene_average_catalogue_ids=scene_ids,
                           input_versions=prepared.input_versions + new_versions,
                           _intervention_receipt={**(prepared._intervention_receipt or {}),
                                                  'receiver_access': selected_mode,
                                                  'scene_average_catalogue_ids': scene_ids}))
        frozen_plan = None
        if matched_mass_shuffle:
            if self.mode == 'J-direct':
                raise ValueError('J-direct has no typed regional incidence to shuffle.')
            if len(prepared.source_membership_history) != self.depth or len(
                    prepared.environment_membership_history) != self.depth:
                raise ValueError('Prepared context does not contain a complete baseline incidence history.')
            frozen_plan = tuple((source.detach(), environment.detach()) for source, environment in zip(
                prepared.source_membership_history, prepared.environment_membership_history, strict=True))
        rebuilt = self._prepare_context_impl(**prepared._input_args,
            incidence_mode=incidence_mode,
            remove_collective_content=remove_collective_content,
            remove_collective_update=remove_collective_update,
            matched_mass_shuffle=matched_mass_shuffle,
            block_index=block_index,
            frozen_incidence_plan=frozen_plan,
            receiver_access=selected_mode,
            scene_average_access=scene_average_access,
            scene_average_catalogue_ids=scene_ids,
            scene_average_catalogue_versions=scene_versions)
        return self._preserve_context_metadata(prepared, rebuilt)

    def restore(self, prepared: JointRegionalPreparedContext) -> JointRegionalPreparedContext:
        """Rebuild the ordinary model state from the original scene tensors."""
        self.assert_owned(prepared)
        return self._preserve_context_metadata(prepared, self.prepare_context(**prepared._input_args))

    def average_receiver_access(self, prepared: JointRegionalPreparedContext,
                                catalogue_receivers: torch.Tensor,
                                receiver_features: torch.Tensor | None = None,
                                receiver_ids: torch.Tensor | None = None,
                                receiver_mask: torch.Tensor | None = None, *,
                                chunk_size: int = 512) -> dict:
        """Average access over a fixed, caller-selected receiver catalogue.

        The catalogue must be selected from inputs before evaluation and reused
        across later query minibatches. Rows are equally weighted after masking.
        """
        self.assert_owned(prepared)
        batch, receivers, features = self._validate_receivers(
            prepared, catalogue_receivers, receiver_features, chunk_size)
        ids = (torch.arange(receivers, device=catalogue_receivers.device)[None].expand(batch, -1)
               if receiver_ids is None else receiver_ids)
        self._validate_shape(ids, (batch, receivers), 'receiver_ids')
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('receiver_ids must be integer catalogue identities.')
        mask = (catalogue_receivers.new_ones(batch, receivers) if receiver_mask is None
                else receiver_mask)
        self._validate_shape(mask, (batch, receivers), 'receiver_mask')
        if bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError('receiver_mask must lie in [0,1].')
        if bool((mask.sum(-1) <= 0).any()):
            raise ValueError('Every scene needs at least one selected catalogue receiver.')
        access_sums = catalogue_receivers.new_zeros(batch, prepared.group_states.shape[1])
        for start in range(0, receivers, chunk_size):
            stop = min(start + chunk_size, receivers)
            current, _regional, _receipt = self._receiver_access(
                prepared, catalogue_receivers[:, start:stop], features[:, start:stop],
                start, 'learned', None, None)
            access_sums = access_sums + (current * mask[:, start:stop, None]).sum(1)
        average = access_sums / mask.sum(-1, keepdim=True).clamp_min(1)
        catalogue_versions = _versions((catalogue_receivers, features, ids, mask, average))
        return {
            'access': average,
            'receiver_ids': ids,
            'receiver_mask': mask,
            'catalogue_count': mask.sum(-1),
            'catalogue_semantics': 'fixed input-selected receiver catalogue; equal-row average; independent of later query minibatches',
            'access_mode': ('geometry' if self.mode == 'J-geometry' else
                            'learned' if self.mode == 'J-H' else 'none'),
            '_catalogue_versions': catalogue_versions,
            '_prepared_context': prepared,
            '_owner_id': id(self),
        }

    def _truncate_receiver_access(self, prepared: JointRegionalPreparedContext,
                                  access: torch.Tensor, retained_access_mass=None,
                                  retained_edge_counts: torch.Tensor | None = None,
                                  fixed_receiver_edge_mask: torch.Tensor | None = None):
        """Select and renormalize an approximation independently of its executor.

        A fixed mask holds physical edge identities for conditional derivatives;
        mass/count selection rebuilds support and can switch at a boundary.
        """
        if sum(value is not None for value in (
                retained_access_mass, retained_edge_counts, fixed_receiver_edge_mask)) > 1:
            raise ValueError('Choose one of retained_access_mass, retained_edge_counts or fixed_receiver_edge_mask.')
        allowed_masses = (1.0, 0.99, 0.95, 0.90)
        if retained_access_mass is not None:
            target = float(retained_access_mass)
            if not any(math.isclose(target, level, rel_tol=0.0, abs_tol=1e-9)
                       for level in allowed_masses):
                raise ValueError('retained_access_mass must be one of 1.0, 0.99, 0.95 or 0.90.')
        else:
            target = None
        batch, queries, edges = access.shape
        valid_edges = prepared.group_present > 0
        valid_edge_counts = valid_edges.sum(-1)[:, None].expand(-1, queries)
        requested = None
        if retained_edge_counts is not None:
            if retained_edge_counts.shape != (batch, queries):
                raise ValueError('retained_edge_counts must have shape [B,Q].')
            if (retained_edge_counts.dtype not in (torch.int32, torch.int64) and
                    (not bool(torch.isfinite(retained_edge_counts).all()) or
                     not bool((retained_edge_counts == retained_edge_counts.round()).all()))):
                raise ValueError('retained_edge_counts must contain integer values.')
            requested = retained_edge_counts.to(device=access.device, dtype=torch.long)
            if bool((requested < 0).any()) or bool((requested > valid_edge_counts).any()):
                raise ValueError('retained_edge_counts must lie within each receiver valid-edge count.')
        if fixed_receiver_edge_mask is not None:
            if fixed_receiver_edge_mask.shape != access.shape or fixed_receiver_edge_mask.dtype != torch.bool:
                raise ValueError('fixed_receiver_edge_mask must be boolean [B,Q,G].')
            if fixed_receiver_edge_mask.device != access.device:
                raise ValueError('fixed_receiver_edge_mask must use the receiver device.')
            if bool((fixed_receiver_edge_mask & ~valid_edges[:, None]).any()):
                raise ValueError('Fixed receiver support cannot select an absent edge.')
            retained = access * fixed_receiver_edge_mask
            actual_mass = retained.sum(-1)
            if edges and bool((actual_mass <= 0).any()):
                raise ValueError('Every fixed receiver support must retain positive access mass.')
            selected = retained / actual_mass[..., None].clamp_min(torch.finfo(access.dtype).tiny)
            counts = fixed_receiver_edge_mask.sum(-1)
            source = 'fixed-edge-identities'
        elif edges == 0:
            actual_mass = access.sum(-1)
            counts = torch.zeros(batch, queries, dtype=torch.long, device=access.device)
            selected = access
            source = ('exact-count' if requested is not None else
                      'mass-threshold' if target is not None and target < 1.0 else
                      'all-valid-edges' if target is not None else 'dense-default')
        elif requested is None and (target is None or target == 1.0):
            # Preserve the original path exactly for the default and 100% study.
            selected = access
            actual_mass = access.sum(-1)
            counts = (access > 0).sum(-1)
            source = 'dense-default' if target is None else 'all-valid-edges'
        else:
            # Canonicalize edge columns by typed physical identity first. The
            # subsequent stable score sort then resolves exact ties identically
            # after a source-slot permutation.
            edge_type = prepared.edge_type.to(torch.long)
            stable_id = torch.where(edge_type == 0, prepared.edge_source_ids,
                                    prepared.edge_region_ids)
            id_order = torch.argsort(stable_id, dim=-1, stable=True)
            ordered_types = edge_type.gather(-1, id_order)
            type_order = torch.argsort(ordered_types, dim=-1, stable=True)
            canonical_order = id_order.gather(-1, type_order)
            canonical_access = access.gather(
                -1, canonical_order[:, None, :].expand(-1, queries, -1))
            canonical_valid = valid_edges.gather(-1, canonical_order)
            valid_by_query = canonical_valid[:, None, :].expand(-1, queries, -1)
            sorting_values = canonical_access.masked_fill(~valid_by_query, -torch.inf)
            score_order = torch.argsort(sorting_values, dim=-1, descending=True, stable=True)
            sorted_access = canonical_access.gather(-1, score_order)
            sorted_valid = valid_by_query.gather(-1, score_order)

            if requested is not None:
                rank = torch.arange(edges, device=access.device)[None, None, :]
                retained_by_rank = rank < requested[..., None]
                retained_by_canonical = torch.zeros_like(canonical_access).scatter(
                    -1, score_order, retained_by_rank.to(access.dtype))
                counts = requested
                source = 'exact-count'
            else:
                cumulative = sorted_access.masked_fill(~sorted_valid, 0).cumsum(-1)
                crossed = cumulative >= target
                first_crossing = crossed.to(torch.int64).argmax(-1)
                base_count = (first_crossing + 1).masked_fill(~crossed.any(-1), edges)
                has_valid = canonical_valid.any(-1)[:, None]
                base_count = base_count.masked_fill(~has_valid, 0)
                cutoff = sorted_access.gather(-1, (base_count - 1).clamp_min(0)[..., None]).squeeze(-1)
                retained_by_rank = (sorted_valid & (sorted_access >= cutoff[..., None]) &
                                    (base_count > 0)[..., None])
                counts = retained_by_rank.sum(-1)
                retained_by_canonical = torch.zeros_like(canonical_access).scatter(
                    -1, score_order, retained_by_rank.to(access.dtype))
                source = 'mass-threshold'

            retained_canonical = canonical_access * retained_by_canonical
            actual_mass = retained_canonical.sum(-1)
            normalized_canonical = retained_canonical / actual_mass[..., None].clamp_min(
                torch.finfo(access.dtype).tiny)
            selected = torch.zeros_like(access).scatter(
                -1, canonical_order[:, None, :].expand(-1, queries, -1), normalized_canonical)
        receipt = {
            'retained_access_mass_target': target,
            'retained_access_mass_actual': actual_mass,
            'retained_edge_counts': counts,
            'valid_edge_counts': valid_edge_counts,
            'selection': source,
            'execution': 'dense receiver-edge scoring and reading; access truncation is a model approximation, not sparse execution',
        }
        return selected, receipt

    @staticmethod
    def _regional_read(prepared, access, receipt, executor):
        """Execute either a dense contraction or an actual packed pair read.

        Scores, context formation and every physical-source read remain dense.
        The packed path gathers only positive retained receiver/edge pairs; it
        does not expand a padded Q x K x H value array. Index selection is a
        conditional support operation, not a smooth support derivative.
        """
        if executor not in ('dense', 'subset'):
            raise ValueError("receiver_edge_executor must be 'dense' or 'subset'.")
        edges = prepared.group_states
        batch, queries, edge_count = access.shape
        approximate = receipt['selection'] in ('mass-threshold', 'exact-count', 'fixed-edge-identities')
        use_subset = executor == 'subset' and approximate and edge_count > 0
        if not edge_count:
            regional = prepared.global_state[:, None].expand(-1, queries, -1)
            pairs = 0
        elif use_subset:
            indices = torch.nonzero(access > 0, as_tuple=False)
            batch_indices, query_indices, edge_indices = indices.unbind(-1)
            values = edges.flatten(0, 1).index_select(0, batch_indices * edge_count + edge_indices)
            weights = access[batch_indices, query_indices, edge_indices]
            regional = edges.new_zeros(batch * queries, edges.shape[-1]).index_add(
                0, batch_indices * queries + query_indices, values * weights[:, None])
            regional = regional.reshape(batch, queries, edges.shape[-1])
            pairs = indices.shape[0]
        else:
            regional = torch.einsum('bqe,beh->bqh', access, edges)
            pairs = batch * queries * edge_count
        executed = 'packed-subset' if use_subset else 'dense'
        receipt.update({
            'receiver_edge_executor_requested': executor,
            'receiver_edge_executor_executed': executed,
            'full_access_fallback': executor == 'subset' and not use_subset,
            'receiver_edge_score_pairs': batch * queries * edge_count,
            'receiver_edge_value_pairs': pairs,
            'physical_source_value_pairs': batch * queries * prepared.centers.shape[1],
            'execution': (f'dense receiver-edge scoring; {executed} collective value read; '
                          'all original physical-source slots read'),
        })
        return regional

    @staticmethod
    def _receiver_execution_summary(receipts):
        return {
            name: sum(receipt[name] for receipt in receipts)
            for name in ('receiver_edge_score_pairs', 'receiver_edge_value_pairs', 'physical_source_value_pairs')
        } | {
            'executors': sorted({receipt['receiver_edge_executor_executed'] for receipt in receipts}),
            'full_access_fallback': any(receipt['full_access_fallback'] for receipt in receipts),
            'scoring_and_context': 'dense',
            'physical_source_path': 'all original source slots retained',
        }

    def _receiver_access(self, prepared: JointRegionalPreparedContext,
                         receivers: torch.Tensor, features: torch.Tensor,
                         start: int, access_mode: str | None,
                         retained_access_mass=None,
                         retained_edge_counts: torch.Tensor | None = None,
                         receiver_edge_executor: str = 'dense',
                         fixed_receiver_edge_mask: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        edges = prepared.group_states
        batch, queries, _dimension = receivers.shape
        edge_count = edges.shape[1]
        mode = prepared._access_mode if access_mode is None else access_mode
        if mode not in ('learned', 'geometry', 'scene-average', 'uniform-average'):
            raise ValueError("Unknown receiver access mode.")
        if edge_count == 0:
            access = receivers.new_empty(batch, queries, 0)
        elif mode == 'scene-average':
            if prepared._scene_average_access is None:
                raise ValueError('scene-average access requires an explicit fixed-catalogue average.')
            access = prepared._scene_average_access[:, None].expand(-1, queries, -1)
        elif mode == 'uniform-average':
            valid = (prepared.group_present > 0).to(receivers.dtype)
            access = valid[:, None].expand(-1, queries, -1)
            access = access / access.sum(-1, keepdim=True).clamp_min(1)
        elif mode == 'geometry' or self.mode == 'J-geometry':
            relative = ((receivers[:, :, None] - prepared.group_centers[:, None]) /
                        prepared.edge_scale_vectors[:, None].clamp_min(1e-12))
            logits = -0.5 * relative.square().sum(-1)
            valid = prepared.group_present[:, None] > 0
            logits = logits.masked_fill(~valid, -torch.inf)
            any_valid = valid.any(-1, keepdim=True)
            maximum = torch.where(any_valid, logits.amax(-1, keepdim=True), torch.zeros_like(logits[..., :1]))
            weights = torch.exp(logits - maximum) * valid
            access = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        else:
            query_input = torch.cat((_geometry((receivers - prepared.domain_origin[:, None]) /
                                                prepared.lengths[:, None]),
                                     prepared.context[:, None].expand(-1, queries, -1), features), -1)
            query = self.receiver_query(query_input)
            keys = self.edge_key(edges)
            relative = ((receivers[:, :, None] - prepared.group_centers[:, None]) /
                        prepared.edge_scale_vectors[:, None].clamp_min(1e-12))
            geometry_bias = self.access_geometry(_geometry(relative)).squeeze(-1)
            logits = torch.einsum('bqd,bed->bqe', query, keys) / math.sqrt(query.shape[-1]) + geometry_bias
            if self.locality_prior_strength:
                logits = logits - 0.5 * self.locality_prior_strength * relative.square().sum(-1)
            valid = prepared.group_present[:, None] > 0
            logits = logits.masked_fill(~valid, -torch.inf)
            any_valid = valid.any(-1, keepdim=True)
            maximum = torch.where(any_valid, logits.amax(-1, keepdim=True), torch.zeros_like(logits[..., :1]))
            weights = torch.exp(logits - maximum) * valid
            access = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        access, receipt = self._truncate_receiver_access(
            prepared, access, retained_access_mass, retained_edge_counts, fixed_receiver_edge_mask)
        regional = self._regional_read(prepared, access, receipt, receiver_edge_executor)
        return access, regional, receipt

    def _read_source_features(self, prepared: JointRegionalPreparedContext,
                              receivers: torch.Tensor, features: torch.Tensor,
                              regional: torch.Tensor) -> torch.Tensor:
        delta = receivers[:, :, None] - prepared.centers[:, None]
        relative = torch.cat((_geometry(delta / prepared.lengths[:, None, None]),
                              _geometry(delta / prepared.source_lengths[:, None, :, None].clamp_min(1e-12))), -1)
        query_geometry = _geometry((receivers - prepared.domain_origin[:, None]) / prepared.lengths[:, None])
        inputs = torch.cat((prepared.source_states[:, None].expand(-1, receivers.shape[1], -1, -1),
                            regional[:, :, None].expand(-1, -1, prepared.centers.shape[1], -1),
                            prepared.global_state[:, None, None].expand(-1, receivers.shape[1], prepared.centers.shape[1], -1),
                            query_geometry[:, :, None].expand(-1, -1, prepared.centers.shape[1], -1),
                            relative, features[:, :, None].expand(-1, -1, prepared.centers.shape[1], -1)), -1)
        return self.source_read(inputs) * prepared.present[:, None, :, None]

    def _validate_receivers(self, prepared, receivers, receiver_features, chunk_size):
        batch, queries, _dimension = receivers.shape
        self._validate_shape(receivers, (prepared.centers.shape[0], queries, self.spatial_dim), 'receivers')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive.')
        features = (receivers.new_empty(batch, queries, 0) if receiver_features is None
                    else receiver_features)
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        return batch, queries, features

    def _effective_access_mode(self, prepared, access_mode):
        selected = prepared._access_mode if access_mode is None else access_mode
        if self.mode == 'J-direct':
            return 'none'
        if self.mode == 'J-geometry' and selected == 'learned':
            return 'geometry'
        return selected

    @staticmethod
    def _validate_receiver_read_options(context, batch, queries, executor, mask):
        if executor not in ('dense', 'subset'):
            raise ValueError("receiver_edge_executor must be 'dense' or 'subset'.")
        if mask is not None and (
                mask.shape != (batch, queries, context.group_states.shape[1]) or
                mask.dtype != torch.bool):
            raise ValueError('fixed_receiver_edge_mask must be boolean [B,Q,G].')

    def read_receiver(self, context: JointRegionalPreparedContext, receivers: torch.Tensor,
                      receiver_features: torch.Tensor | None = None, *,
                      chunk_size: int = 512, access_mode: str | None = None,
                      retained_access_mass=None,
                      retained_edge_counts: torch.Tensor | None = None,
                      receiver_edge_executor: str = 'dense',
                      fixed_receiver_edge_mask: torch.Tensor | None = None) -> dict:
        """Expose actual receiver-to-edge/source reads and primitive work counts."""
        self.assert_owned(context)
        batch, queries, features = self._validate_receivers(context, receivers, receiver_features, chunk_size)
        self._validate_receiver_read_options(context, batch, queries, receiver_edge_executor, fixed_receiver_edge_mask)
        if retained_edge_counts is not None and retained_edge_counts.shape != (batch, queries):
            raise ValueError('retained_edge_counts must have shape [B,Q].')
        accesses, regional_values, source_values = [], [], []
        actual_masses, retained_counts, valid_counts = [], [], []
        execution_receipts = []
        for start in range(0, queries, chunk_size):
            stop = min(start + chunk_size, queries)
            query = receivers[:, start:stop]
            current_features = features[:, start:stop]
            current_counts = (None if retained_edge_counts is None
                              else retained_edge_counts[:, start:stop])
            access, regional, receipt = self._receiver_access(
                context, query, current_features, start, access_mode,
                retained_access_mass, current_counts, receiver_edge_executor,
                None if fixed_receiver_edge_mask is None else fixed_receiver_edge_mask[:, start:stop])
            source_features = self._read_source_features(context, query, current_features, regional)
            accesses.append(access)
            regional_values.append(regional)
            source_values.append(source_features)
            actual_masses.append(receipt['retained_access_mass_actual'])
            retained_counts.append(receipt['retained_edge_counts'])
            valid_counts.append(receipt['valid_edge_counts'])
            execution_receipts.append(receipt)
        cat = lambda values, shape: (torch.cat(values, 1) if values else
                                     receivers.new_empty(shape))
        access = cat(accesses, (batch, 0, context.group_states.shape[1]))
        regional = cat(regional_values, (batch, 0, self.hidden))
        source_features = cat(source_values, (batch, 0, context.centers.shape[1], self.message))
        actual_mass = cat(actual_masses, (batch, 0))
        retained_counts_tensor = cat(retained_counts, (batch, 0)).long()
        valid_counts_tensor = cat(valid_counts, (batch, 0)).long()
        return {
            'receiver_access': access,
            'receiver_regional_state': regional,
            'source_features': source_features,
            'source_ids': context.source_ids,
            'retained_access_mass_target': (None if retained_access_mass is None
                                           else float(retained_access_mass)),
            'retained_access_mass_actual': actual_mass,
            'retained_edge_counts': retained_counts_tensor,
            'valid_edge_counts': valid_counts_tensor,
            'access_mode': self._effective_access_mode(context, access_mode),
            'execution': 'dense receiver-edge scoring/context and all-source reads; collective value executor reported separately',
            'executor_receipt': self._receiver_execution_summary(execution_receipts),
        }

    def predict_fields(self, context: JointRegionalPreparedContext, receivers: torch.Tensor,
                       receiver_features: torch.Tensor | None = None, *, chunk_size: int = 512,
                       access_mode: str | None = None, retained_access_mass=None,
                       retained_edge_counts: torch.Tensor | None = None,
                       receiver_edge_executor: str = 'dense',
                       fixed_receiver_edge_mask: torch.Tensor | None = None,
                       return_organization: bool = False):
        """Read nonlinear configuration fields independently at each receiver."""
        self.assert_owned(context)
        batch, queries, features = self._validate_receivers(context, receivers, receiver_features, chunk_size)
        self._validate_receiver_read_options(context, batch, queries, receiver_edge_executor, fixed_receiver_edge_mask)
        if retained_edge_counts is not None and retained_edge_counts.shape != (batch, queries):
            raise ValueError('retained_edge_counts must have shape [B,Q].')
        outputs = []
        accesses, actual_masses, retained_counts, valid_counts = [], [], [], []
        execution_receipts = []
        for start in range(0, queries, chunk_size):
            stop = min(start + chunk_size, queries)
            query = receivers[:, start:start + chunk_size]
            current_features = features[:, start:start + chunk_size]
            current_counts = (None if retained_edge_counts is None
                              else retained_edge_counts[:, start:stop])
            access, regional, receipt = self._receiver_access(
                context, query, current_features, start, access_mode,
                retained_access_mass, current_counts, receiver_edge_executor,
                None if fixed_receiver_edge_mask is None else fixed_receiver_edge_mask[:, start:stop])
            source_features = self._read_source_features(context, query, current_features, regional)
            source_pool = (source_features * context.source_measure[:, None, :, None]).sum(2)
            field_input = torch.cat((source_pool, regional,
                                     context.global_state[:, None].expand(-1, query.shape[1], -1),
                                     _geometry((query - context.domain_origin[:, None]) /
                                               context.lengths[:, None]), current_features), -1)
            outputs.append(self.field_head(field_input))
            if return_organization:
                accesses.append(access)
                actual_masses.append(receipt['retained_access_mass_actual'])
                retained_counts.append(receipt['retained_edge_counts'])
                valid_counts.append(receipt['valid_edge_counts'])
                execution_receipts.append(receipt)
        fields = torch.cat(outputs, 1) if outputs else receivers.new_empty(batch, 0, self.field_outputs)
        if not return_organization:
            return fields
        cat = lambda values, shape: (torch.cat(values, 1) if values else
                                     receivers.new_empty(shape))
        return fields, {
            'receiver_access': cat(accesses, (batch, 0, context.group_states.shape[1])),
            'receiver_access_mode': self._effective_access_mode(context, access_mode),
            'retained_access_mass_target': (None if retained_access_mass is None
                                           else float(retained_access_mass)),
            'retained_access_mass_actual': cat(actual_masses, (batch, 0)),
            'retained_edge_counts': cat(retained_counts, (batch, 0)).long(),
            'valid_edge_counts': cat(valid_counts, (batch, 0)).long(),
            'source_ids': context.source_ids,
            'execution': 'dense receiver-edge scoring/context and all-source reads; collective value executor reported separately',
            'executor_receipt': self._receiver_execution_summary(execution_receipts),
        }

    def predict(self, context, receivers, receiver_features=None, **kwargs):
        return self.predict_fields(context, receivers, receiver_features, **kwargs)

    def prepare_receivers(self, context: JointRegionalPreparedContext, receivers: torch.Tensor,
                          receiver_features: torch.Tensor | None = None,
                          receiver_ids: torch.Tensor | None = None, chunk_size: int = 512,
                          access_mode: str | None = None, retained_access_mass=None,
                          retained_edge_counts: torch.Tensor | None = None,
                          receiver_edge_executor: str = 'dense',
                          fixed_receiver_edge_mask: torch.Tensor | None = None) -> PreparedSourceResponse:
        """Prepare the source-resolved affine coefficient K(q,i)."""
        self.assert_owned(context)
        if self.affine_outputs == 0:
            raise ValueError('This model has no separately applicable affine output capability.')
        batch, queries, features = self._validate_receivers(context, receivers, receiver_features, chunk_size)
        self._validate_receiver_read_options(context, batch, queries, receiver_edge_executor, fixed_receiver_edge_mask)
        ids = (torch.arange(queries, device=receivers.device)[None].expand(batch, -1)
               if receiver_ids is None else receiver_ids)
        self._validate_shape(ids, (batch, queries), 'receiver_ids')
        if retained_edge_counts is not None and retained_edge_counts.shape != (batch, queries):
            raise ValueError('retained_edge_counts must have shape [B,Q].')
        kernels, offsets, accesses, regional_values = [], [], [], []
        actual_masses, retained_counts, valid_counts = [], [], []
        execution_receipts = []
        for start in range(0, queries, chunk_size):
            stop = min(start + chunk_size, queries)
            query = receivers[:, start:start + chunk_size]
            current_features = features[:, start:start + chunk_size]
            current_counts = (None if retained_edge_counts is None
                              else retained_edge_counts[:, start:stop])
            access, regional, receipt = self._receiver_access(
                context, query, current_features, start, access_mode,
                retained_access_mass, current_counts, receiver_edge_executor,
                None if fixed_receiver_edge_mask is None else fixed_receiver_edge_mask[:, start:stop])
            source_features = self._read_source_features(context, query, current_features, regional)
            kernel = self.affine_head(source_features) * context.present[:, None, :, None]
            kernels.append(kernel)
            accesses.append(access)
            regional_values.append(regional)
            actual_masses.append(receipt['retained_access_mass_actual'])
            retained_counts.append(receipt['retained_edge_counts'])
            valid_counts.append(receipt['valid_edge_counts'])
            execution_receipts.append(receipt)
            if self.zero_offset:
                offsets.append(query.new_zeros(batch, query.shape[1], self.affine_outputs))
            else:
                offset_input = torch.cat((regional, context.global_state[:, None].expand(-1, query.shape[1], -1),
                                          _geometry((query - context.domain_origin[:, None]) /
                                                    context.lengths[:, None]), current_features), -1)
                offsets.append(self.offset_head(offset_input))
        far_kernel = (torch.cat(kernels, 1) if kernels else
                      receivers.new_empty(batch, 0, context.centers.shape[1], self.affine_outputs))
        offset = (torch.cat(offsets, 1) if offsets else
                  receivers.new_empty(batch, 0, self.affine_outputs))
        access = torch.cat(accesses, 1) if accesses else receivers.new_empty(batch, 0, context.group_states.shape[1])
        regional = (torch.cat(regional_values, 1) if regional_values else
                    receivers.new_empty(batch, 0, self.hidden))
        no_near = receivers.new_zeros(batch, queries, context.centers.shape[1])
        empty_index = torch.empty(3, 0, dtype=torch.long, device=receivers.device)
        empty_values = receivers.new_empty(0, self.affine_outputs)
        response = PreparedSourceResponse(
            context=context, receivers=receivers, receiver_ids=ids, offset=offset,
            near_weight=no_near, near_indices=empty_index, near_values=empty_values,
            far_kernel=far_kernel, receiver_functions=None,
            source_membership=None, environment_membership=None, group_present=None,
            forcing_scale=self.forcing_scale, mode='direct',
            receiver_versions=_versions((receivers, features, ids, fixed_receiver_edge_mask, retained_edge_counts)))
        response.receiver_access = access
        response.receiver_regional_state = regional
        response.source_features = None
        response.access_mode = self._effective_access_mode(context, access_mode)
        response.affine_capability = True
        response.retention_receipt = {
            'retained_access_mass_target': (None if retained_access_mass is None
                                           else float(retained_access_mass)),
            'retained_access_mass_actual': (torch.cat(actual_masses, 1) if actual_masses else
                                            receivers.new_empty(batch, 0)),
            'retained_edge_counts': (torch.cat(retained_counts, 1).long() if retained_counts else
                                     torch.empty(batch, 0, dtype=torch.long, device=receivers.device)),
            'valid_edge_counts': (torch.cat(valid_counts, 1).long() if valid_counts else
                                  torch.empty(batch, 0, dtype=torch.long, device=receivers.device)),
            'selection': ('fixed-edge-identities' if fixed_receiver_edge_mask is not None
                          else 'dense-default' if retained_access_mass is None and retained_edge_counts is None
                          else 'mass-threshold' if retained_access_mass is not None and float(retained_access_mass) < 1
                          else 'all-valid-edges' if retained_access_mass is not None
                          else 'exact-count'),
            'execution': 'dense receiver-edge scoring/context and all-source reads; collective value executor reported separately',
            'executor_receipt': self._receiver_execution_summary(execution_receipts),
        }
        return response

    def read_kernel(self, context, receivers, **kwargs) -> torch.Tensor:
        """Return scalar K as [B,Q,M], or multiple affine heads as [B,Q,M,A]."""
        response = self.prepare_receivers(context, receivers, **kwargs)
        kernel = response.dense_kernel()
        return kernel[..., 0] if self.affine_outputs == 1 else kernel

    def apply_forcing(self, response: PreparedSourceResponse, forcing: torch.Tensor, *,
                      forcing_map=None, return_contributions: bool = False,
                      accumulation_dtype=None):
        response.assert_fresh()
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
            values = offset + torch.einsum('bqma,bm->bqa', kernel, mapped.to(accumulation_dtype))
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
        response.assert_fresh()
        if forcing_map is not None:
            if baseline_forcing is None:
                raise ValueError('Nonlinear forcing increments require an explicit baseline.')
            delta_forcing = forcing_map(baseline_forcing + delta_forcing) - forcing_map(baseline_forcing)
        if delta_forcing.shape != response.context.present.shape or not bool(torch.isfinite(delta_forcing).all()):
            raise ValueError('Forcing increment must be finite [B,M] aligned with prepared source IDs.')
        kernel = response.dense_kernel(accumulation_dtype=accumulation_dtype)
        values = torch.einsum('bqma,bm->bqa', kernel,
                              delta_forcing if accumulation_dtype is None else delta_forcing.to(accumulation_dtype))
        return values[..., 0] if self.affine_outputs == 1 else values

    def export_organization(self, prepared: JointRegionalPreparedContext,
                            response: PreparedSourceResponse | None = None,
                            receiver_read: dict | None = None) -> dict:
        """Export the actual executed typed incidences, updates and receiver reads."""
        self.assert_owned(prepared)
        receiver_access = None
        access_mode = None
        retention_receipt = None
        if response is not None:
            response.assert_fresh()
            if response.context is not prepared:
                raise ValueError('Receiver response belongs to a different prepared context.')
            receiver_access = response.receiver_access
            access_mode = response.access_mode
            retention_receipt = getattr(response, 'retention_receipt', None)
        if receiver_read is not None:
            receiver_access = receiver_read['receiver_access']
            access_mode = receiver_read['access_mode']
            retention_receipt = {
                'retained_access_mass_target': receiver_read['retained_access_mass_target'],
                'retained_access_mass_actual': receiver_read['retained_access_mass_actual'],
                'retained_edge_counts': receiver_read['retained_edge_counts'],
                'valid_edge_counts': receiver_read['valid_edge_counts'],
                'execution': receiver_read['execution'],
                'executor_receipt': receiver_read.get('executor_receipt'),
            }
        source_measure = prepared.source_measures_raw
        environment_measure = prepared.environment_measures_raw
        return {
            'mode': self.mode,
            'source_ids': prepared.source_ids,
            'source_id_catalogue': getattr(prepared, 'source_id_catalogue', None),
            'source_states': prepared.source_states,
            'environment_states': prepared.environment_states,
            'source_measures': source_measure,
            'environment_measures': environment_measure,
            'source_membership_density': prepared.source_membership_density,
            'environment_membership_density': prepared.environment_membership_density,
            'source_membership_mass': prepared.source_membership,
            'environment_membership_mass': prepared.environment_membership,
            'source_membership_mass_history': prepared.source_membership_history,
            'environment_membership_mass_history': prepared.environment_membership_history,
            'source_membership_density_history': prepared.source_membership_density_history,
            'environment_membership_density_history': prepared.environment_membership_density_history,
            'edge_states': prepared.group_states,
            'edge_state_history': prepared.edge_state_history,
            'edge_anchor_states': prepared.edge_anchor_states,
            'edge_centers': prepared.group_centers,
            'edge_scales': prepared.edge_scale_vectors,
            'edge_type': prepared.edge_type,
            'edge_source_ids': prepared.edge_source_ids,
            'edge_region_ids': prepared.edge_region_ids,
            'edge_present': prepared.group_present,
            'edge_to_source': prepared.edge_to_source,
            'edge_to_environment': prepared.edge_to_environment,
            'edge_to_source_history': prepared.edge_to_source_history,
            'edge_to_environment_history': prepared.edge_to_environment_history,
            'source_update_delta': prepared.source_update_delta,
            'environment_update_delta': prepared.environment_update_delta,
            'source_update_history': prepared.source_update_history,
            'environment_update_history': prepared.environment_update_history,
            'receiver_access': receiver_access,
            'receiver_access_mode': access_mode,
            'receiver_retention': retention_receipt,
            'global_state': prepared.global_state,
            'bypasses': ('source-preserving receiver read', 'pooled scene context'),
            'intervention': prepared._intervention_receipt,
            'semantics': 'executed typed regional organization; memberships are descriptive learned weights, not causal supports',
        }

    def forward(self, prepared, receivers, receiver_features=None, **kwargs):
        return self.predict_fields(prepared, receivers, receiver_features, **kwargs)
