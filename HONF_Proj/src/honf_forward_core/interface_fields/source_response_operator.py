"""Whole-layout response coefficients with explicit, separately applied forcing.

Modules are actuated sources; environmental tokens are context donors. This
module owns no case semantics, physical solver, observed-field input, or heat
allocation. Direct and grouped readouts share the contextual and near kernels.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from torch import nn


def _mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def _score_mlp(inputs, hidden):
    # A final scalar bias would cancel exactly in normalized memberships.
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False))


def _geometry(value):
    frequency = value.new_tensor([math.pi * 2 ** index for index in range(4)])
    phase = value[..., None] * frequency
    return torch.cat((value, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


def _normalized_measure(measure, present):
    mass = measure * present
    return mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)


def _versions(tensors):
    return tuple((tensor, tensor._version) for tensor in tensors if tensor is not None)


def _check_versions(versions):
    if any(tensor._version != version for tensor, version in versions):
        raise ValueError('Prepared response context/receiver tensors changed; rebuild this request.')


@dataclass
class PreparedResponseContext:
    source_states: torch.Tensor
    environment_states: torch.Tensor
    global_state: torch.Tensor
    centers: torch.Tensor
    source_lengths: torch.Tensor
    present: torch.Tensor
    source_measure: torch.Tensor
    environment_coords: torch.Tensor
    environment_present: torch.Tensor
    environment_measure: torch.Tensor
    lengths: torch.Tensor
    source_ids: torch.Tensor
    group_states: torch.Tensor | None
    group_centers: torch.Tensor | None
    group_lengths: torch.Tensor | None
    group_present: torch.Tensor | None
    source_membership: torch.Tensor | None
    environment_membership: torch.Tensor | None
    input_versions: tuple

    def assert_fresh(self):
        _check_versions(self.input_versions)


@dataclass
class ResponseCompression:
    keep: torch.Tensor
    omitted_bound: torch.Tensor
    budget: torch.Tensor
    forcing_radii: torch.Tensor
    semantics: str = 'model-only balanced forcing increment bound; absolute baseline is unchanged'


@dataclass
class PreparedSourceResponse:
    context: PreparedResponseContext
    receivers: torch.Tensor
    receiver_ids: torch.Tensor
    offset: torch.Tensor
    near_weight: torch.Tensor
    near_indices: torch.Tensor
    near_values: torch.Tensor
    far_kernel: torch.Tensor | None
    receiver_functions: torch.Tensor | None
    source_membership: torch.Tensor | None
    environment_membership: torch.Tensor | None
    group_present: torch.Tensor | None
    forcing_scale: float
    mode: str
    receiver_versions: tuple

    def assert_fresh(self):
        self.context.assert_fresh()
        _check_versions(self.receiver_versions)

    def dense_kernel(self):
        """Explicit source-resolved export, in derivative units per forcing."""
        self.assert_fresh()
        if self.mode == 'direct':
            far = self.far_kernel
        else:
            far = torch.einsum('bqeo,bem->bqmo', self.receiver_functions, self.source_membership)
        kernel = far * (1 - self.near_weight[..., None])
        if self.near_indices.numel():
            batch, query, source = self.near_indices
            kernel = kernel.index_put((batch, query, source),
                kernel[batch, query, source] + self.near_weight[batch, query, source, None] * self.near_values)
        return kernel / self.forcing_scale


@dataclass
class ResponseApplication:
    values: torch.Tensor
    source_contributions: torch.Tensor | None
    receipt: dict


class SourceResponseOperator(nn.Module):
    """Prepare nonlinear context/kernel once; apply affine physical forcing.

    Current forcing is absent from ``prepare_context`` and receiver reads.
    ``forcing_scale`` is one frozen, positive training scale. Optional caller
    forcing maps make the *forcing law* nonlinear, and are explicitly labeled;
    the exported kernel then differentiates the mapped forcing coordinates.
    """
    def __init__(self, source_width, context_width, environment_width, *, mode='direct',
                 spatial_dim=2, hidden=64, message=64, output_width=1, query_width=0,
                 background_mode=True, zero_offset=True, forcing_scale=1., far_hidden=None,
                 max_sources=None):
        super().__init__()
        if mode not in ('direct', 'group') or spatial_dim not in (2, 3):
            raise ValueError('Response mode must be direct/group with 2-D or 3-D geometry.')
        if min(source_width, context_width, environment_width, hidden, message, output_width) < 1 or query_width < 0:
            raise ValueError('Response feature/hidden/output widths must be positive; query_width may be zero.')
        if not math.isfinite(forcing_scale) or forcing_scale <= 0:
            raise ValueError('forcing_scale must be a frozen positive finite scalar.')
        self.mode, self.spatial_dim, self.output_width = mode, spatial_dim, output_width
        self.hidden, self.background_mode = hidden, bool(background_mode)
        self.zero_offset, self.forcing_scale = bool(zero_offset), float(forcing_scale)
        self.max_sources, self.query_width = max_sources, query_width
        relative_width = spatial_dim * 9
        read_width = 2 * hidden + 3 * relative_width + query_width
        # Every common module is registered/initialized before arm-specific
        # modules. Equal RNG seeds give exactly identical common tensors.
        self.source_encoder = _mlp(source_width + context_width, hidden, hidden)
        self.environment_encoder = _mlp(environment_width + context_width, hidden, hidden)
        self.module_messages = nn.ModuleList([_mlp(2 * hidden + relative_width, message, hidden) for _ in range(2)])
        self.environment_messages = nn.ModuleList([_mlp(2 * hidden + relative_width, message, hidden) for _ in range(2)])
        self.source_updates = nn.ModuleList([_mlp(hidden + 2 * message + context_width, hidden, hidden) for _ in range(2)])
        self.environment_updates = nn.ModuleList([_mlp(2 * hidden + message + context_width, hidden, hidden) for _ in range(2)])
        self.global_encoder = _mlp(2 * hidden + context_width, hidden, hidden)
        self.near_head = _mlp(read_width, output_width, hidden)
        self.offset_head = None if zero_offset else _mlp(hidden + relative_width + query_width, output_width, hidden)
        score_hidden = max(8, hidden // 2)
        score_inputs = 2 * hidden + relative_width
        group_extra = (2 * ((score_inputs + 1) * score_hidden + score_hidden)
                       + (4 * hidden + 1) * hidden + (hidden + 1) * hidden)
        if far_hidden is None:
            # Match real far-read parameters; no dormant matching tensors.
            far_hidden = hidden if mode == 'group' else hidden + round(group_extra / (read_width + output_width + 1))
        if far_hidden < 1:
            raise ValueError('far_hidden must be positive.')
        if mode == 'group':
            self.source_membership_score = _score_mlp(score_inputs, score_hidden)
            self.environment_membership_score = _score_mlp(score_inputs, score_hidden)
            self.group_collective = _mlp(4 * hidden, hidden, hidden)
        self.far_head = _mlp(read_width, output_width, far_hidden)
        self.config = {'source_width': source_width, 'context_width': context_width,
            'environment_width': environment_width, 'mode': mode, 'spatial_dim': spatial_dim,
            'hidden': hidden, 'message': message, 'output_width': output_width, 'query_width': query_width,
            'background_mode': bool(background_mode), 'zero_offset': bool(zero_offset),
            'forcing_scale': float(forcing_scale), 'far_hidden': far_hidden, 'max_sources': max_sources}

    @staticmethod
    def _validate_shape(value, shape, name):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f'{name} shape must be {tuple(shape)}, received {tuple(value.shape)}.')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must contain finite values.')

    def prepare_context(self, sources, context, centers, present, lengths, source_lengths,
                        source_measures=None, environment_tokens=None, environment_coords=None,
                        environment_present=None, environment_measures=None, source_ids=None):
        """Explicit geometry/prescribed tensors; no current forcing argument.

        sources[B,M,F], context[B,C], centers[B,M,D], present/measures[B,M],
        lengths[B,D], source_lengths[B,M] or [B,M,1], environments[B,N,F_E].
        Missing environmental type is represented by an empty donor catalogue.
        """
        batch, modules, dimension = centers.shape
        if dimension != self.spatial_dim or (self.max_sources is not None and modules > self.max_sources):
            raise ValueError('Source coordinates/capacity disagree with the response configuration.')
        if source_lengths.ndim == 3: source_lengths = source_lengths.squeeze(-1)
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
        if environment_coords is None: raise ValueError('Environmental tokens require their physical coordinates.')
        environments = environment_tokens.shape[1]
        environment_present = sources.new_ones(batch, environments) if environment_present is None else environment_present
        environment_measures = torch.ones_like(environment_present) if environment_measures is None else environment_measures
        self._validate_shape(environment_tokens, (batch, environments, self.config['environment_width']), 'environment_tokens')
        self._validate_shape(environment_coords, (batch, environments, dimension), 'environment_coords')
        self._validate_shape(environment_present, (batch, environments), 'environment_present')
        self._validate_shape(environment_measures, (batch, environments), 'environment_measures')
        if bool((environment_measures < 0).any()) or bool(((environment_present < 0) | (environment_present > 1)).any()):
            raise ValueError('Environmental measures must be nonnegative and presence must lie in [0,1].')
        source_ids = torch.arange(modules, device=centers.device)[None].expand(batch, -1) if source_ids is None else source_ids
        self._validate_shape(source_ids, (batch, modules), 'source_ids')
        versions = _versions((sources, context, centers, present, lengths, source_lengths, source_measures,
            environment_tokens, environment_coords, environment_present, environment_measures, source_ids))
        source_mass = _normalized_measure(source_measures, present)
        environment_mass = _normalized_measure(environment_measures, environment_present)
        source_context = context[:, None].expand(-1, modules, -1)
        environment_context = context[:, None].expand(-1, environments, -1)
        state = self.source_encoder(torch.cat((sources, source_context), -1)) * present[..., None]
        environment = self.environment_encoder(torch.cat((environment_tokens, environment_context), -1)) * environment_present[..., None]
        relative_mm = _geometry((centers[:, :, None] - centers[:, None]) / lengths[:, None, None])
        relative_me = _geometry((centers[:, :, None] - environment_coords[:, None]) / lengths[:, None, None])
        relative_em = _geometry((environment_coords[:, :, None] - centers[:, None]) / lengths[:, None, None])
        self_mask = 1 - torch.eye(modules, device=centers.device, dtype=centers.dtype)[None]
        donor_mass = source_mass[:, None] * self_mask
        donor_mass = donor_mass / donor_mass.sum(-1, keepdim=True).clamp_min(torch.finfo(state.dtype).tiny)
        for module_message, environment_message, source_update, environment_update in zip(
                self.module_messages, self.environment_messages, self.source_updates, self.environment_updates):
            left = state[:, :, None].expand(-1, -1, modules, -1)
            right = state[:, None].expand(-1, modules, -1, -1)
            module_aggregate = (module_message(torch.cat((left, right, relative_mm), -1)) * donor_mass[..., None]).sum(2)
            env_aggregate = (environment_message(torch.cat((state[:, :, None].expand(-1, -1, environments, -1),
                environment[:, None].expand(-1, modules, -1, -1), relative_me), -1)) * environment_mass[:, None, :, None]).sum(2)
            source_aggregate = (environment_message(torch.cat((environment[:, :, None].expand(-1, -1, modules, -1),
                state[:, None].expand(-1, environments, -1, -1), relative_em), -1)) * source_mass[:, None, :, None]).sum(2)
            env_pool = (environment * environment_mass[..., None]).sum(1)[:, None].expand(-1, environments, -1)
            state = (state + source_update(torch.cat((state, module_aggregate, env_aggregate, source_context), -1))) * present[..., None]
            environment = (environment + environment_update(torch.cat((environment, source_aggregate, env_pool, environment_context), -1))) * environment_present[..., None]
        global_state = self.global_encoder(torch.cat(((state * source_mass[..., None]).sum(1),
            (environment * environment_mass[..., None]).sum(1), context), -1))
        prepared = PreparedResponseContext(state, environment, global_state, centers,
            source_lengths, present, source_mass, environment_coords, environment_present,
            environment_mass, lengths, source_ids, None, None, None, None, None, None, versions)
        if self.mode == 'group': prepared = self._prepare_groups(prepared)
        return prepared

    @staticmethod
    def _membership(scores, mass, group_present):
        active = mass > 0
        if mass.shape[-1] == 0: return scores
        masked = scores.masked_fill(~active[:, None], torch.finfo(scores.dtype).min)
        weight = (masked - masked.amax(-1, keepdim=True)).exp() * mass[:, None]
        return weight / weight.sum(-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny) * group_present[..., None]

    def _prepare_groups(self, context):
        state, global_state = context.source_states, context.global_state
        anchors, centers, lengths, present = state, context.centers, context.source_lengths, context.present
        if self.background_mode:
            env_center = (context.environment_coords * context.environment_measure[..., None]).sum(1)
            has_env = (context.environment_measure.sum(-1) > 0)[:, None]
            background_center = torch.where(has_env, env_center, context.lengths / 2)
            background_length = (context.source_lengths * context.source_measure).sum(1).clamp_min(1e-6)
            background_present = (context.source_measure.sum(-1) > 0).to(present.dtype)
            anchors = torch.cat((anchors, global_state[:, None]), 1)
            centers = torch.cat((centers, background_center[:, None]), 1)
            lengths = torch.cat((lengths, background_length[:, None]), 1)
            present = torch.cat((present, background_present[:, None]), 1)
        groups, modules, environments = anchors.shape[1], state.shape[1], context.environment_states.shape[1]
        rel_m = _geometry((centers[:, :, None] - context.centers[:, None]) / context.lengths[:, None, None])
        rel_e = _geometry((centers[:, :, None] - context.environment_coords[:, None]) / context.lengths[:, None, None])
        b_m = self._membership(self.source_membership_score(torch.cat((anchors[:, :, None].expand(-1, -1, modules, -1),
            state[:, None].expand(-1, groups, -1, -1), rel_m), -1)).squeeze(-1), context.source_measure, present)
        b_e = self._membership(self.environment_membership_score(torch.cat((anchors[:, :, None].expand(-1, -1, environments, -1),
            context.environment_states[:, None].expand(-1, groups, -1, -1), rel_e), -1)).squeeze(-1), context.environment_measure, present)
        group_states = self.group_collective(torch.cat((torch.bmm(b_m, state),
            torch.bmm(b_e, context.environment_states), anchors, global_state[:, None].expand(-1, groups, -1)), -1)) * present[..., None]
        return replace(context, group_states=group_states, group_centers=centers,
            group_lengths=lengths, group_present=present, source_membership=b_m, environment_membership=b_e)

    @staticmethod
    def near_weight(receivers, centers, source_lengths, present):
        distance = torch.linalg.vector_norm(receivers[:, :, None] - centers[:, None], dim=-1)
        scaled = distance / source_lengths[:, None].clamp_min(1e-12)
        transition = ((scaled - 2) / 2).clamp(0, 1)
        return (1 - transition.square() * (3 - 2 * transition)) * present[:, None]

    def _read_features(self, context, query, source_states, centers, source_lengths, features):
        count = centers.shape[1]
        domain_query = _geometry(query / context.lengths[:, None])
        delta = query[:, :, None] - centers[:, None]
        relative = torch.cat((_geometry(delta / context.lengths[:, None, None]),
            _geometry(delta / source_lengths[:, None, :, None].clamp_min(1e-12))), -1)
        return torch.cat((source_states[:, None].expand(-1, query.shape[1], -1, -1),
            context.global_state[:, None, None].expand(-1, query.shape[1], count, -1),
            domain_query[:, :, None].expand(-1, -1, count, -1), relative,
            features[:, :, None].expand(-1, -1, count, -1)), -1)

    def prepare_receivers(self, context, receivers, receiver_features=None, receiver_ids=None, chunk_size=512):
        context.assert_fresh()
        batch, queries, _dimension = receivers.shape
        self._validate_shape(receivers, (context.centers.shape[0], queries, self.spatial_dim), 'receivers')
        if chunk_size < 1: raise ValueError('chunk_size must be positive.')
        features = receivers.new_empty(batch, queries, 0) if receiver_features is None else receiver_features
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        ids = torch.arange(queries, device=receivers.device)[None].expand(batch, -1) if receiver_ids is None else receiver_ids
        self._validate_shape(ids, (batch, queries), 'receiver_ids')
        near_weights, near_indices, near_values, far_values, offsets = [], [], [], [], []
        for start in range(0, queries, chunk_size):
            query, feature = receivers[:, start:start + chunk_size], features[:, start:start + chunk_size]
            weight = self.near_weight(query, context.centers, context.source_lengths, context.present)
            selected = (weight > 0).nonzero(as_tuple=True)
            full_features = self._read_features(context, query, context.source_states, context.centers, context.source_lengths, feature)
            values = self.near_head(full_features[selected])
            near_weights.append(weight)
            near_values.append(values)
            near_indices.append(torch.stack((selected[0], selected[1] + start, selected[2])))
            if self.mode == 'direct':
                far_values.append(self.far_head(full_features) * context.present[:, None, :, None])
            else:
                group_features = self._read_features(context, query, context.group_states,
                    context.group_centers, context.group_lengths, feature)
                far_values.append(self.far_head(group_features) * context.group_present[:, None, :, None])
            if self.zero_offset:
                offsets.append(query.new_zeros(batch, query.shape[1], self.output_width))
            else:
                offsets.append(self.offset_head(torch.cat((context.global_state[:, None].expand(-1, query.shape[1], -1),
                    _geometry(query / context.lengths[:, None]), feature), -1)))
        if queries == 0:
            count = context.centers.shape[1] if self.mode == 'direct' else context.group_states.shape[1]
            far = receivers.new_empty(batch, 0, count, self.output_width)
            weight = receivers.new_empty(batch, 0, context.centers.shape[1])
            index = torch.empty(3, 0, dtype=torch.long, device=receivers.device)
            near = receivers.new_empty(0, self.output_width)
            offset = receivers.new_empty(batch, 0, self.output_width)
        else:
            far, weight = torch.cat(far_values, 1), torch.cat(near_weights, 1)
            index, near = torch.cat(near_indices, 1), torch.cat(near_values, 0)
            offset = torch.cat(offsets, 1)
        return PreparedSourceResponse(context, receivers, ids, offset, weight, index, near,
            far if self.mode == 'direct' else None, far if self.mode == 'group' else None,
            context.source_membership, context.environment_membership, context.group_present,
            self.forcing_scale, self.mode, _versions((receivers, features, ids)))

    def read_kernel(self, context, receivers, **kwargs):
        return self.prepare_receivers(context, receivers, **kwargs).dense_kernel()

    @staticmethod
    def _apply_linear(response, forcing, compression=None):
        response.assert_fresh()
        if forcing.shape != response.context.present.shape:
            raise ValueError('Forcing must be [B,M] aligned with the prepared source IDs.')
        if not bool(torch.isfinite(forcing).all()): raise ValueError('Forcing must be finite.')
        scaled = forcing / response.forcing_scale
        if response.mode == 'direct':
            if compression is not None: raise ValueError('Learned-mode compression requires the grouped response.')
            result = (response.far_kernel * (1 - response.near_weight[..., None]) * scaled[:, None, :, None]).sum(2)
            near_far = None
        else:
            a = response.receiver_functions
            if compression is not None: a = a * compression.keep[..., None]
            result = torch.einsum('bqeo,be->bqo', a, torch.bmm(response.source_membership, scaled[..., None]).squeeze(-1))
            if response.near_indices.numel():
                batch, query, source = response.near_indices
                near_far = (a[batch, query] * response.source_membership[batch, :, source, None]).sum(1)
        if response.near_indices.numel():
            batch, query, source = response.near_indices
            near = response.near_values
            if response.mode == 'group': near = near - near_far
            correction = near * response.near_weight[batch, query, source, None] * scaled[batch, source, None]
            result = result.index_put((batch, query), correction, accumulate=True)
        return result

    def apply_forcing(self, response, forcing, *, forcing_map: Callable | None = None, return_contributions=False):
        mapped = forcing if forcing_map is None else forcing_map(forcing)
        values = response.offset + self._apply_linear(response, mapped)
        if not return_contributions: return values
        contributions = response.dense_kernel() * mapped[:, None, :, None]
        return ResponseApplication(values, contributions, {'forcing_law': 'affine' if forcing_map is None else 'explicit caller nonlinear map',
            'kernel_coordinates': 'physical forcing' if forcing_map is None else 'mapped forcing', 'physical_solves': 0})

    def apply_increment(self, response, delta_forcing, *, compression=None,
                        forcing_map: Callable | None = None, baseline_forcing=None):
        if forcing_map is not None:
            if baseline_forcing is None: raise ValueError('Nonlinear forcing increments require an explicit baseline.')
            delta_forcing = forcing_map(baseline_forcing + delta_forcing) - forcing_map(baseline_forcing)
        if compression is not None:
            active_delta = delta_forcing * response.context.present
            epsilon = torch.finfo(active_delta.dtype).eps * 64
            if bool((active_delta.abs() > compression.forcing_radii + epsilon).any()):
                raise ValueError('Compressed increment exceeds its declared forcing radii.')
            if bool((active_delta.sum(-1).abs() > epsilon * active_delta.abs().sum(-1).clamp_min(1)).any()):
                raise ValueError('This response compression bound requires balanced forcing increments.')
        return self._apply_linear(response, delta_forcing, compression)

    def export_response_operator(self, response):
        return {'K': response.dense_kernel(), 'offset': response.offset,
            'source_ids': response.context.source_ids, 'receiver_ids': response.receiver_ids,
            'receivers': response.receivers, 'forcing_scale': response.forcing_scale,
            'forcing_law': 'affine in explicitly applied forcing; optional caller map is separate'}

    def export_organization(self, response):
        response.assert_fresh()
        return {'mode': response.mode, 'source_ids': response.context.source_ids,
            'source_states': response.context.source_states, 'environment_states': response.context.environment_states,
            'environment_context_measures': response.context.environment_measure,
            'b_M': response.source_membership, 'b_E': response.environment_membership,
            'a': None if response.receiver_functions is None else response.receiver_functions / response.forcing_scale,
            'near_weight': response.near_weight, 'near_indices': response.near_indices,
            'near_kernel': response.near_values / response.forcing_scale,
            'group_present': response.group_present, 'group_centers': response.context.group_centers,
            'ancestry': 'nonlinear whole-layout source/environment context; geometry-only near weights; current forcing enters only application',
            'environment_semantics': 'context donors, not independently actuated thermal sources',
            'near_work': int(response.near_indices.shape[1]), 'far_rows': int((response.far_kernel if response.mode == 'direct' else response.receiver_functions).numel() // self.output_width)}

    @staticmethod
    def geometry_matched_membership(response, iterations=256):
        """Geometry-only factor control with original row/column marginals.

        This changes the prepared readout factors, preserving learned mode
        functions and context. Bounded matrix scaling is an assignment
        normalization, not a physical solve or context recomputation.
        """
        response.assert_fresh()
        if response.mode != 'group': raise ValueError('Matched membership requires grouped coefficients.')
        if iterations < 1 or iterations > 512: raise ValueError('Assignment iterations must be in [1,512].')
        original = response.source_membership
        if original.shape[-1] == 0: return original.clone()
        dtype = original.dtype
        rows = response.group_present.to(torch.float64)
        columns = original.to(torch.float64).sum(1)
        displacement = (response.context.group_centers[:, :, None].to(torch.float64)
            - response.context.centers[:, None].to(torch.float64)) / response.context.lengths[:, None, None].to(torch.float64)
        kernel = (-displacement.square().sum(-1).clamp_max(20)).exp()
        kernel = kernel * (rows > 0)[..., None] * (columns > 0)[:, None]
        for _ in range(iterations):
            kernel = kernel * (rows / kernel.sum(-1).clamp_min(1e-300))[..., None]
            kernel = kernel * (columns / kernel.sum(1).clamp_min(1e-300))[:, None]
        if not torch.allclose(kernel.sum(-1), rows, atol=1e-6, rtol=1e-6):
            raise RuntimeError('Bounded geometry assignment did not preserve row marginals.')
        return kernel.to(dtype)

    @staticmethod
    def intervene_groups(response, *, source_membership=None, remove_mode=None, permutation=None):
        """Fixed-weight factor interventions; application recomputes subtraction."""
        if response.mode != 'group': raise ValueError('Group interventions require grouped coefficients.')
        a, b, e, present = response.receiver_functions, response.source_membership, response.environment_membership, response.group_present
        if source_membership is not None:
            if source_membership.shape != b.shape or bool((source_membership < 0).any()):
                raise ValueError('Replacement source membership must be aligned and nonnegative.')
            totals = source_membership.sum(-1)
            if not torch.allclose(totals, present, atol=1e-6, rtol=1e-6):
                raise ValueError('Replacement source memberships must retain valid-group normalization.')
            if bool((source_membership * (response.context.present[:, None] <= 0)).any()):
                raise ValueError('Replacement membership may not admit padded sources.')
            b = source_membership
        if remove_mode is not None:
            selected = torch.as_tensor(remove_mode, device=a.device).long()
            if selected.ndim == 0: selected = selected.expand(a.shape[0])
            if selected.shape != (a.shape[0],) or bool(((selected < 0) | (selected >= a.shape[2])).any()):
                raise ValueError('Removed mode must identify one allocated group per case.')
            keep = torch.arange(a.shape[2], device=a.device)[None] != selected[:, None]
            a = a * keep[:, None, :, None]
        if permutation is not None:
            permutation = torch.as_tensor(permutation, device=a.device).long()
            if permutation.ndim != 1 or not torch.equal(permutation.sort().values, torch.arange(a.shape[2], device=a.device)):
                raise ValueError('Mode permutation must cover every allocated group once.')
            a, b, e, present = a[:, :, permutation], b[:, permutation], e[:, permutation], present[:, permutation]
        return replace(response, receiver_functions=a, source_membership=b, environment_membership=e, group_present=present)

    @staticmethod
    def compress_response(response, forcing_radii, distortion_budget):
        """Conservative, target-free bounds for balanced physical increments.

        Uses effective (1-w)b weights and source-only centering. No physical
        inverse, factorization, fitted truth, or absolute-field mode omission.
        """
        response.assert_fresh()
        if response.mode != 'group': raise ValueError('Response-mode bounds require grouped factors.')
        if forcing_radii.shape != response.context.present.shape or bool((forcing_radii < 0).any()):
            raise ValueError('Forcing radii must be nonnegative [B,M] source-aligned values.')
        effective = (1 - response.near_weight[:, :, None]) * response.source_membership[:, None]
        mass = response.context.source_measure[:, None, None]
        beta = (effective * mass).sum(-1, keepdim=True)
        bound = ((effective - beta).abs() * forcing_radii[:, None, None] * response.context.present[:, None, None]).sum(-1)
        bound = bound[..., None] * response.receiver_functions.abs() / response.forcing_scale
        budget = torch.as_tensor(distortion_budget, device=bound.device, dtype=bound.dtype)
        budget = torch.broadcast_to(budget, bound.shape[:2] + (bound.shape[-1],))
        if bool((budget < 0).any()): raise ValueError('Distortion budget must be nonnegative.')
        scores = bound.amax(-1)
        order = scores.argsort(-1)
        ordered = bound.gather(2, order[..., None].expand_as(bound))
        cumulative = ordered.cumsum(2)
        omitted_ordered = (cumulative <= budget[:, :, None]).all(-1) & (budget.amax(-1)[:, :, None] > 0)
        omitted = torch.zeros_like(omitted_ordered).scatter(2, order, omitted_ordered)
        omitted_bound = (bound * omitted[..., None]).sum(2)
        return ResponseCompression(~omitted, omitted_bound, budget, forcing_radii)

    def forward(self, context, receivers, forcing, **kwargs):
        response = self.prepare_receivers(context, receivers, **kwargs)
        return self.apply_forcing(response, forcing)
