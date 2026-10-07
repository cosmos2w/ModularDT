"""Dataset-independent, source-resolved interaction computation.

Both affine Thermal and nonlinear Wind execute these context rounds, physical
measure reductions, geometry features and receiver/source feature builders.
Adapters own input meaning, scales, dependency laws and output units.
"""
from __future__ import annotations

import math
import weakref
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


def _parameter_signature(model):
    return tuple((name, id(parameter), parameter._version, parameter.data_ptr())
                 for name, parameter in model.named_parameters())


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
    model_reference: object | None = None
    parameter_signature: tuple = ()

    def assert_fresh(self):
        _check_versions(self.input_versions)
        if self.model_reference is not None:
            owner = self.model_reference()
            if owner is None or _parameter_signature(owner) != self.parameter_signature:
                raise ValueError('Prepared interaction model weights changed; rebuild this request.')


@dataclass(frozen=True)
class DependencySpec:
    """Adapter declaration; configuration controls invalidate prepared state."""

    dataset: str
    output_law: str
    configuration_inputs: tuple[str, ...]
    applicable_controls: tuple[str, ...]
    output_roles: tuple[str, ...]
    units: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()
    prepared_nodes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.output_law not in ('affine', 'nonlinear'):
            raise ValueError('Declare an affine or nonlinear output law.')
        if set(self.configuration_inputs) & set(self.applicable_controls):
            raise ValueError('A control cannot be both configuration and separately applicable.')
        if self.output_law == 'nonlinear' and self.applicable_controls:
            raise ValueError('Nonlinear configuration changes require rebuilding context.')
        if len(self.output_roles) != len(self.units):
            raise ValueError('Each declared output role requires its physical units.')
        adjacency = {}
        for donor, receiver in self.edges:
            adjacency.setdefault(donor, set()).add(receiver)
        visited, visiting = set(), set()

        def visit(node):
            if node in visiting:
                raise ValueError('Cyclic coupled dependencies require a different state update; no iterative solver is supplied.')
            if node in visited:
                return
            visiting.add(node)
            for receiver in adjacency.get(node, ()):
                visit(receiver)
            visiting.remove(node)
            visited.add(node)

        for node in adjacency:
            visit(node)
        protected = set(self.configuration_inputs) | set(self.prepared_nodes)
        for control in self.applicable_controls:
            pending, reached = list(adjacency.get(control, ())), set()
            while pending:
                node = pending.pop()
                if node in reached:
                    continue
                reached.add(node)
                pending.extend(adjacency.get(node, ()))
            if protected & reached:
                raise ValueError('Separately applicable affine control reaches prepared context; rebuild with a different capability.')


@dataclass
class InteractionScene:
    """Whitelisted prescribed inputs; observations and supervision stay outside."""

    sources: torch.Tensor
    context: torch.Tensor
    centers: torch.Tensor
    present: torch.Tensor
    lengths: torch.Tensor
    source_lengths: torch.Tensor
    dependency: DependencySpec
    source_measures: torch.Tensor | None = None
    environment_tokens: torch.Tensor | None = None
    environment_coords: torch.Tensor | None = None
    environment_present: torch.Tensor | None = None
    environment_measures: torch.Tensor | None = None
    source_ids: torch.Tensor | None = None
    environment_embedding: torch.Tensor | None = None

    def tensors(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__
                if name != 'dependency'}


class InteractionContextCore(nn.Module):
    """Two context rounds over physical sources and environment donors."""

    def __init__(self, source_width, context_width, environment_width, *,
                 spatial_dim=2, hidden=64, message=64, query_width=0, max_sources=None):
        super().__init__()
        if spatial_dim not in (2, 3) or min(source_width, context_width, environment_width, hidden, message) < 1 or query_width < 0:
            raise ValueError('Interaction widths must be positive with 2-D or 3-D geometry.')
        self.spatial_dim, self.hidden, self.query_width = spatial_dim, hidden, query_width
        self.max_sources = max_sources
        self.config = {'source_width': source_width, 'context_width': context_width,
            'environment_width': environment_width, 'spatial_dim': spatial_dim, 'hidden': hidden,
            'message': message, 'query_width': query_width, 'max_sources': max_sources}
        relative_width = spatial_dim * 9
        self.source_encoder = _mlp(source_width + context_width, hidden, hidden)
        self.environment_encoder = _mlp(environment_width + context_width, hidden, hidden)
        self.module_messages = nn.ModuleList([_mlp(2 * hidden + relative_width, message, hidden) for _ in range(2)])
        self.environment_messages = nn.ModuleList([_mlp(2 * hidden + relative_width, message, hidden) for _ in range(2)])
        self.source_updates = nn.ModuleList([_mlp(hidden + 2 * message + context_width, hidden, hidden) for _ in range(2)])
        self.environment_updates = nn.ModuleList([_mlp(2 * hidden + message + context_width, hidden, hidden) for _ in range(2)])
        self.global_encoder = _mlp(2 * hidden + context_width, hidden, hidden)

    @staticmethod
    def _validate_shape(value, shape, name):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f'{name} shape must be {tuple(shape)}, received {tuple(value.shape)}.')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must contain finite values.')

    def prepare_context(self, sources, context, centers, present, lengths, source_lengths,
                        source_measures=None, environment_tokens=None, environment_coords=None,
                        environment_present=None, environment_measures=None, source_ids=None,
                        environment_embedding=None):
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
            environment_tokens, environment_coords, environment_present, environment_measures, source_ids, environment_embedding))
        source_mass = _normalized_measure(source_measures, present)
        environment_mass = _normalized_measure(environment_measures, environment_present)
        source_context = context[:, None].expand(-1, modules, -1)
        environment_context = context[:, None].expand(-1, environments, -1)
        state = self.source_encoder(torch.cat((sources, source_context), -1)) * present[..., None]
        environment = self.environment_encoder(torch.cat((environment_tokens, environment_context), -1))
        if environment_embedding is not None:
            self._validate_shape(environment_embedding, (batch, environments, self.hidden), 'environment_embedding')
            environment = environment + environment_embedding
        environment = environment * environment_present[..., None]
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
        prepared.owner = id(self)
        prepared.model_reference = weakref.ref(self)
        prepared.parameter_signature = _parameter_signature(self)
        return prepared

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

    def _read_selected_features(self, context, query, features, selected):
        """Gather physical pairs before constructing hidden-width fine inputs."""
        batch, receiver, source = selected
        point = query[batch, receiver]
        delta = point - context.centers[batch, source]
        return torch.cat((context.source_states[batch, source],
            context.global_state[batch], _geometry(point / context.lengths[batch]),
            _geometry(delta / context.lengths[batch]),
            _geometry(delta / context.source_lengths[batch, source, None].clamp_min(1e-12)),
            features[batch, receiver]), -1)

    def prepare(self, scene):
        if getattr(self, 'output_law', scene.dependency.output_law) != scene.dependency.output_law:
            raise ValueError('Scene output capability disagrees with the selected readout.')
        prepared = self.prepare_context(**scene.tensors())
        prepared.owner = id(self)
        prepared.dependency = scene.dependency
        return prepared

    def assert_owned(self, prepared):
        prepared.assert_fresh()
        if getattr(prepared, 'owner', id(self)) != id(self):
            raise ValueError('Prepared interaction state belongs to another operator.')


class NonlinearFieldReadout(InteractionContextCore):
    """Source-wise nonlinear messages followed by a nonlinear field head.

    Environment is read through the prepared context/global state. No dense
    receiver/environment fine stack or scalar affine forcing law is executed.
    """

    output_law = 'nonlinear'

    def __init__(self, source_width, context_width, environment_width, *,
                 spatial_dim=3, hidden=64, message=64, output_width=3,
                 query_width=0, max_sources=None):
        super().__init__(source_width, context_width, environment_width,
            spatial_dim=spatial_dim, hidden=hidden, message=message,
            query_width=query_width, max_sources=max_sources)
        if output_width < 1:
            raise ValueError('Nonlinear output width must be positive.')
        self.output_width = output_width
        self.source_read = _mlp(2 * hidden + 3 * spatial_dim * 9 + query_width, message, hidden)
        self.field_head = _mlp(hidden + message + spatial_dim * 9 + query_width, output_width, hidden)
        self.config.update(output_width=output_width)

    def predict(self, prepared, receivers, receiver_features=None, *, chunk_size=512,
                return_messages=False):
        self.assert_owned(prepared)
        batch, queries, _ = receivers.shape
        self._validate_shape(receivers, (prepared.centers.shape[0], queries, self.spatial_dim), 'receivers')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive.')
        features = receivers.new_empty(batch, queries, 0) if receiver_features is None else receiver_features
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        values, messages = [], []
        for start in range(0, queries, chunk_size):
            query, feature = receivers[:, start:start + chunk_size], features[:, start:start + chunk_size]
            fine = self.source_read(self._read_features(prepared, query,
                prepared.source_states, prepared.centers, prepared.source_lengths, feature))
            reduced = (fine * prepared.source_measure[:, None, :, None]).sum(2)
            values.append(self.field_head(torch.cat((reduced,
                prepared.global_state[:, None].expand(-1, query.shape[1], -1),
                _geometry(query / prepared.lengths[:, None]), feature), -1)))
            if return_messages:
                messages.append(fine * prepared.present[:, None, :, None])
        output = torch.cat(values, 1) if values else receivers.new_empty(batch, 0, self.output_width)
        if return_messages:
            exported = torch.cat(messages, 1) if messages else receivers.new_empty(batch, 0, prepared.centers.shape[1], self.config['message'])
            return output, exported
        return output

    def forward(self, prepared, receivers, **kwargs):
        return self.predict(prepared, receivers, **kwargs)

    def apply_increment(self, *args, **kwargs):
        raise ValueError('Nonlinear fields have no exact affine finite-increment capability; use a local AD linearization.')

    def linearize(self, scene, receivers, tangent, *, wrt='centers', receiver_features=None):
        """JVP of the actual nonlinear read at a declared operating point."""
        if wrt not in ('centers', 'sources', 'context', 'receivers'):
            raise ValueError('Local linearization supports centers, sources, context or receivers.')
        value = receivers if wrt == 'receivers' else getattr(scene, wrt)
        if tangent.shape != value.shape:
            raise ValueError('Local tangent must match its declared configuration variable.')

        def evaluate(changed):
            current = scene if wrt == 'receivers' else replace(scene, **{wrt: changed})
            query = changed if wrt == 'receivers' else receivers
            return self.predict(self.prepare(current), query, receiver_features)

        output, jvp = torch.autograd.functional.jvp(evaluate, value, tangent, create_graph=torch.is_grad_enabled())
        return {'values': output, 'jvp': jvp, 'wrt': wrt,
                'semantics': 'local AD linearization at the supplied scene; no exact finite-response guarantee'}
