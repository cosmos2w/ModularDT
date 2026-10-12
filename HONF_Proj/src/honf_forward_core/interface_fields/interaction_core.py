"""Dataset-independent, source-resolved interaction computation.

Both affine Thermal and nonlinear Wind execute these context rounds, physical
measure reductions, geometry features and receiver/source feature builders.
Adapters own input meaning, scales, dependency laws and output units.
"""
from __future__ import annotations

import copy
import math
import weakref
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .native_interaction_routing import (
    NativeInteractionRouting,
    density_normalize,
    normalize_intervention,
    reassign_equal_measure_densities,
    route_endpoint_types,
)


def _mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def _project_linear_slice(linear, value, start, stop):
    """Apply a column slice of an existing Linear without copying parameters."""
    return F.linear(value, linear.weight[:, start:stop], None)


def _finish_projected_mlp(module, projected_terms):
    """Finish an MLP after its first affine map has been algebraically split."""
    first = module[0]
    preactivation = projected_terms[0]
    for term in projected_terms[1:]:
        preactivation = preactivation + term
    if first.bias is not None:
        preactivation = preactivation + first.bias
    return module[2](module[1](preactivation))


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
    # Optional payload produced by an explicitly inserted shared-core block.
    # Legacy operators leave this unset and retain their original behavior.
    intermediate_context: object | None = None
    # Runtime-only generation for execution-mode changes. It is not a model
    # parameter or buffer and therefore does not enter checkpoint weights.
    factorized_affine_generation: int = 0
    native_routing_generation: int = 0
    native_routing_enabled: bool = True
    native_aggregation_policy: str = 'measure'
    native_routing_config_snapshot: object | None = None
    # Filled only when callers explicitly request a detached diagnostic trace.
    native_routing_trace: tuple[dict, ...] | None = None
    native_route_intervention: tuple[str | None, tuple[str, ...]] = (None, ())

    def assert_fresh(self):
        _check_versions(self.input_versions)
        if self.model_reference is not None:
            owner = self.model_reference()
            if owner is None or _parameter_signature(owner) != self.parameter_signature:
                raise ValueError('Prepared interaction model weights changed; rebuild this request.')
            if (hasattr(owner, '_factorized_affine_generation')
                    and self.factorized_affine_generation != owner._factorized_affine_generation):
                raise ValueError('Prepared interaction execution mode changed; model semantics changed; rebuild this request.')
            if (hasattr(owner, '_native_routing_generation')
                    and self.native_routing_generation != owner._native_routing_generation):
                raise ValueError('Prepared native routing policy changed; rebuild this request.')
            if (hasattr(owner, 'native_routing_enabled')
                    and self.native_routing_enabled != owner.native_routing_enabled):
                raise ValueError('Prepared native routing execution changed; rebuild this request.')
            if (hasattr(owner, 'native_aggregation_policy')
                    and self.native_aggregation_policy != owner.native_aggregation_policy):
                raise ValueError('Prepared native aggregation policy changed; rebuild this request.')
            if (self.native_routing_config_snapshot is not None
                    and self.native_routing_config_snapshot != owner.config.get('native_routing')):
                raise ValueError('Prepared native routing config changed; rebuild this request.')


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
                 spatial_dim=2, hidden=64, message=64, query_width=0, max_sources=None,
                 factorized_affine_execution=False,
                 native_aggregation_policy='measure', native_control_hidden=None,
                 native_routing_seed=0):
        super().__init__()
        if spatial_dim not in (2, 3) or min(source_width, context_width, environment_width, hidden, message) < 1 or query_width < 0:
            raise ValueError('Interaction widths must be positive with 2-D or 3-D geometry.')
        self.spatial_dim, self.hidden, self.query_width = spatial_dim, hidden, query_width
        self.max_sources = max_sources
        relative_width = spatial_dim * 9
        self.config = {'source_width': source_width, 'context_width': context_width,
            'environment_width': environment_width, 'spatial_dim': spatial_dim, 'hidden': hidden,
            'message': message, 'query_width': query_width, 'max_sources': max_sources}
        if native_aggregation_policy not in ('measure', 'learned_density', 'destination_gain'):
            raise ValueError('Unsupported native aggregation policy.')
        if native_aggregation_policy == 'measure' and native_control_hidden is not None:
            raise ValueError('The legacy measure policy has no native control width.')
        if native_aggregation_policy == 'learned_density' and native_control_hidden is not None:
            raise ValueError('P-R has fixed-width score networks and no P-C control width.')
        if native_aggregation_policy == 'destination_gain' and (
                type(native_control_hidden) is not int or native_control_hidden < 1):
            raise ValueError('P-C requires a positive integer native control width.')
        self.native_aggregation_policy = native_aggregation_policy
        self.native_control_hidden = native_control_hidden
        self._native_routing_generation = 0
        self.native_routing_enabled = True
        if native_aggregation_policy != 'measure':
            pair_width = 2 * hidden + relative_width
            self.native_routing = NativeInteractionRouting(
                policy=native_aggregation_policy,
                pair_width=pair_width,
                destination_width=hidden,
                context_width=context_width,
                message_width=message,
                control_hidden=native_control_hidden,
                initialization_seed=native_routing_seed,
            )
            routing_config = {
                'schema': 'native_interaction_routing_v1',
                'policy': native_aggregation_policy,
                'initialization_seed': int(native_routing_seed),
            }
            if native_aggregation_policy == 'learned_density':
                routing_config.update(score_hidden=NativeInteractionRouting.SCORE_HIDDEN,
                                      score_bound=NativeInteractionRouting.SCORE_BOUND)
            else:
                routing_config['control_hidden'] = native_control_hidden
            self.config['native_routing'] = routing_config
        self.factorized_affine_execution = False
        self._factorized_affine_generation = 0
        self.set_factorized_affine_execution(factorized_affine_execution)
        self.source_encoder = _mlp(source_width + context_width, hidden, hidden)
        self.environment_encoder = _mlp(environment_width + context_width, hidden, hidden)
        self.module_messages = nn.ModuleList([_mlp(2 * hidden + relative_width, message, hidden) for _ in range(2)])
        self.environment_messages = nn.ModuleList([_mlp(2 * hidden + relative_width, message, hidden) for _ in range(2)])
        self.source_updates = nn.ModuleList([_mlp(hidden + 2 * message + context_width, hidden, hidden) for _ in range(2)])
        self.environment_updates = nn.ModuleList([_mlp(2 * hidden + message + context_width, hidden, hidden) for _ in range(2)])
        self.global_encoder = _mlp(2 * hidden + context_width, hidden, hidden)

    def set_factorized_affine_execution(self, enabled):
        """Select the opt-in same-function affine executor.

        This execution-only setting adds no learned state. It is recorded in
        model configuration only when enabled, preserving legacy config
        round-trips, and changing it invalidates already prepared contexts.
        Dataset adapters should construct with this option so their exported
        model configuration records the selected execution mode.
        """
        if type(enabled) is not bool:
            raise TypeError('factorized_affine_execution must be a bool.')
        if enabled != self.factorized_affine_execution:
            self._factorized_affine_generation += 1
        self.factorized_affine_execution = enabled
        if enabled:
            self.config['factorized_affine_execution'] = True
        else:
            self.config.pop('factorized_affine_execution', None)

    def set_native_routing_enabled(self, enabled: bool) -> None:
        """Toggle an inserted routing block and invalidate every prepared state.

        This setting is runtime-only. Its generation counter intentionally
        prevents an object prepared before a disable/reenable cycle from
        becoming valid again when the visible bool returns to its old value.
        """
        if type(enabled) is not bool:
            raise TypeError('native routing execution setting must be a bool.')
        if self.native_aggregation_policy == 'measure':
            raise ValueError('The legacy measure policy has no native routing block to toggle.')
        self.native_routing_enabled = enabled

    @property
    def native_routing_enabled(self) -> bool:
        return self._native_routing_enabled

    @native_routing_enabled.setter
    def native_routing_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError('native routing execution setting must be a bool.')
        if (hasattr(self, 'native_aggregation_policy')
                and self.native_aggregation_policy == 'measure' and not enabled):
            raise ValueError('The legacy measure policy has no native routing block to toggle.')
        previous = getattr(self, '_native_routing_enabled', None)
        if previous is not None and previous != enabled:
            self._native_routing_generation += 1
        self._native_routing_enabled = enabled

    @staticmethod
    def _diagnostic_copy(value):
        if value is None:
            return None
        return value.detach().to(device='cpu').clone()

    def _native_route_weights(self, route, base_weights, pair_features, measures,
                              present, donor_ids, intervention_kind,
                              intervention_routes):
        scores = None
        if (self.native_aggregation_policy == 'learned_density'
                and self.native_routing_enabled):
            scores = self.native_routing.score(route, pair_features)
            density, weights = density_normalize(base_weights, scores)
        else:
            density = torch.ones_like(base_weights)
            weights = base_weights
        receipt = {
            'changed_donors': 0,
            'skipped_singleton_buckets': 0,
            'eligible_buckets': 0,
            'eligible_donors': 0,
        }
        if intervention_kind is not None and route in intervention_routes:
            if self.native_aggregation_policy != 'learned_density' or not self.native_routing_enabled:
                raise ValueError('Native route interventions require active P-R density routing.')
            if intervention_kind == 'density_reset':
                eligible = base_weights > 0
                receipt['changed_donors'] = int(((density != 1) & eligible).sum().item())
                density = torch.ones_like(density)
            else:
                density, receipt = reassign_equal_measure_densities(
                    density, base_weights, measures, present, donor_ids)
            weights = base_weights * density
        for name, value in (('score', scores), ('density', density), ('weight', weights)):
            if value is not None and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Native {route} routing produced non-finite {name} values.')
        return scores, density, weights, receipt

    @classmethod
    def _native_route_trace(cls, *, round_index, route, destination_ids,
                            donor_ids, destination_present, donor_present,
                            destination_measure, donor_measure, self_mask,
                            base_weights, scores, density, weights,
                            values, aggregate, modulation_gain, intervention,
                            intervention_receipt):
        donor_type, destination_type = route_endpoint_types(route)
        weighted_norm = torch.linalg.vector_norm(values * weights[..., None], dim=-1)
        return {
            'round': round_index + 1,
            'legacy_route': route,
            'donor_type': donor_type,
            'destination_type': destination_type,
            'destination_ids': cls._diagnostic_copy(destination_ids),
            'donor_ids': cls._diagnostic_copy(donor_ids),
            'destination_present': cls._diagnostic_copy(destination_present),
            'donor_present': cls._diagnostic_copy(donor_present),
            'destination_present_mask': cls._diagnostic_copy(destination_present > 0),
            'donor_present_mask': cls._diagnostic_copy(donor_present > 0),
            'destination_measure': cls._diagnostic_copy(destination_measure),
            'donor_measure': cls._diagnostic_copy(donor_measure),
            'donor_active_mask': cls._diagnostic_copy(
                ((donor_present > 0) & (donor_measure > 0))[:, None, :]
                .expand_as(base_weights)),
            'donor_mask': cls._diagnostic_copy(base_weights > 0),
            'self_exclusion_mask': cls._diagnostic_copy(self_mask),
            'w0': cls._diagnostic_copy(base_weights),
            's': cls._diagnostic_copy(scores),
            'rho': cls._diagnostic_copy(density),
            'w': cls._diagnostic_copy(weights),
            'weighted_message_norm': cls._diagnostic_copy(weighted_norm),
            'aggregate': cls._diagnostic_copy(aggregate),
            'modulation_gain': cls._diagnostic_copy(modulation_gain),
            'intervention': intervention,
            'intervention_receipt': dict(intervention_receipt),
        }

    @staticmethod
    def _validate_shape(value, shape, name):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f'{name} shape must be {tuple(shape)}, received {tuple(value.shape)}.')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must contain finite values.')

    def prepare_context(self, sources, context, centers, present, lengths, source_lengths,
                        source_measures=None, environment_tokens=None, environment_coords=None,
                        environment_present=None, environment_measures=None, source_ids=None,
                        environment_embedding=None, _between_rounds=None,
                        _extra_version_tensors=(), environment_ids=None,
                        routing_trace=False, native_route_intervention=None):
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
        environment_ids = (torch.arange(environments, device=centers.device, dtype=torch.long)[None]
                           .expand(batch, -1) if environment_ids is None else environment_ids)
        self._validate_shape(environment_ids, (batch, environments), 'environment_ids')
        if type(routing_trace) is not bool:
            raise TypeError('routing_trace must be a bool.')
        intervention_kind, intervention_routes = normalize_intervention(native_route_intervention)
        if intervention_kind is not None and self.native_aggregation_policy != 'learned_density':
            raise ValueError('Native route interventions are supported only by P-R.')
        typed_native_ids = (self.native_aggregation_policy != 'measure' or routing_trace
                            or intervention_kind is not None)
        if typed_native_ids:
            for name, ids, active in (
                    ('source_ids', source_ids, present),
                    ('environment_ids', environment_ids, environment_present)):
                if ids.dtype not in (torch.int32, torch.int64):
                    raise TypeError(f'{name} must use an integer tensor dtype for native routing/traces.')
                for batch_index in range(batch):
                    active_ids = ids[batch_index][active[batch_index] > 0]
                    if (bool((active_ids < 0).any())
                            or torch.unique(active_ids).numel() != active_ids.numel()):
                        raise ValueError(f'{name} must uniquely identify every active native record.')
        versions = _versions((sources, context, centers, present, lengths, source_lengths, source_measures,
            environment_tokens, environment_coords, environment_present, environment_measures, source_ids,
            environment_ids, environment_embedding, *_extra_version_tensors))
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
        intermediate_context = None
        route_trace = [] if routing_trace else None
        for round_index, (module_message, environment_message, source_update, environment_update) in enumerate(zip(
                self.module_messages, self.environment_messages, self.source_updates, self.environment_updates)):
            me_base_weights = environment_mass[:, None, :].expand(-1, modules, -1)
            em_base_weights = source_mass[:, None, :].expand(-1, environments, -1)
            collect_routes = (self.native_aggregation_policy != 'measure'
                              or routing_trace or intervention_kind is not None)
            module_pair_features = None
            me_pair_features = None
            em_pair_features = None
            if (self.native_aggregation_policy == 'learned_density'
                    and self.native_routing_enabled):
                module_pair_features = torch.cat((
                    state[:, :, None].expand(-1, -1, modules, -1),
                    state[:, None].expand(-1, modules, -1, -1), relative_mm), -1)
                me_pair_features = torch.cat((
                    state[:, :, None].expand(-1, -1, environments, -1),
                    environment[:, None].expand(-1, modules, -1, -1), relative_me), -1)
                em_pair_features = torch.cat((
                    environment[:, :, None].expand(-1, -1, modules, -1),
                    state[:, None].expand(-1, environments, -1, -1), relative_em), -1)
            if collect_routes:
                mm_scores, mm_density, mm_weights, mm_receipt = self._native_route_weights(
                    'MM', donor_mass, module_pair_features, source_measures, present,
                    source_ids, intervention_kind, intervention_routes)
                me_scores, me_density, me_weights, me_receipt = self._native_route_weights(
                    'ME', me_base_weights, me_pair_features, environment_measures,
                    environment_present, environment_ids, intervention_kind,
                    intervention_routes)
                em_scores, em_density, em_weights, em_receipt = self._native_route_weights(
                    'EM', em_base_weights, em_pair_features, source_measures,
                    present, source_ids, intervention_kind, intervention_routes)
            if not collect_routes:
                # Keep the legacy P/P-G/P-H reduction and operation order
                # unchanged unless a caller opts into new routing or traces.
                if self.factorized_affine_execution:
                    mm_first = module_message[0]
                    mm_relative_start = 2 * self.hidden
                    mm_left = _project_linear_slice(mm_first, state, 0, self.hidden)
                    mm_right = _project_linear_slice(mm_first, state, self.hidden, mm_relative_start)
                    mm_relative = _project_linear_slice(mm_first, relative_mm,
                        mm_relative_start, mm_first.in_features)
                    module_values = _finish_projected_mlp(module_message,
                        (mm_left[:, :, None], mm_right[:, None], mm_relative))
                    module_aggregate = (module_values * donor_mass[..., None]).sum(2)

                    env_first = environment_message[0]
                    env_relative_start = 2 * self.hidden
                    state_first = _project_linear_slice(env_first, state, 0, self.hidden)
                    state_second = _project_linear_slice(env_first, state,
                        self.hidden, env_relative_start)
                    environment_first = _project_linear_slice(env_first, environment,
                        0, self.hidden)
                    environment_second = _project_linear_slice(env_first, environment,
                        self.hidden, env_relative_start)
                    me_relative = _project_linear_slice(env_first, relative_me,
                        env_relative_start, env_first.in_features)
                    me_values = _finish_projected_mlp(environment_message,
                        (state_first[:, :, None], environment_second[:, None], me_relative))
                    env_aggregate = (me_values * environment_mass[:, None, :, None]).sum(2)
                    em_relative = _project_linear_slice(env_first, relative_em,
                        env_relative_start, env_first.in_features)
                    em_values = _finish_projected_mlp(environment_message,
                        (environment_first[:, :, None], state_second[:, None], em_relative))
                    source_aggregate = (em_values * source_mass[:, None, :, None]).sum(2)
                else:
                    left = state[:, :, None].expand(-1, -1, modules, -1)
                    right = state[:, None].expand(-1, modules, -1, -1)
                    module_values = module_message(torch.cat((left, right, relative_mm), -1))
                    module_aggregate = (module_values * donor_mass[..., None]).sum(2)
                    me_values = environment_message(torch.cat((state[:, :, None].expand(-1, -1, environments, -1),
                        environment[:, None].expand(-1, modules, -1, -1), relative_me), -1))
                    env_aggregate = (me_values * environment_mass[:, None, :, None]).sum(2)
                    em_values = environment_message(torch.cat((environment[:, :, None].expand(-1, -1, modules, -1),
                        state[:, None].expand(-1, environments, -1, -1), relative_em), -1))
                    source_aggregate = (em_values * source_mass[:, None, :, None]).sum(2)
                mm_gain = me_gain = em_gain = None
            elif self.factorized_affine_execution:
                mm_first = module_message[0]
                mm_relative_start = 2 * self.hidden
                mm_left = _project_linear_slice(mm_first, state, 0, self.hidden)
                mm_right = _project_linear_slice(mm_first, state, self.hidden, mm_relative_start)
                mm_relative = _project_linear_slice(mm_first, relative_mm,
                    mm_relative_start, mm_first.in_features)
                module_values = _finish_projected_mlp(module_message,
                    (mm_left[:, :, None], mm_right[:, None], mm_relative))
                mm_gain = None
                module_aggregate = (module_values * mm_weights[..., None]).sum(2)

                env_first = environment_message[0]
                env_relative_start = 2 * self.hidden
                state_first = _project_linear_slice(env_first, state, 0, self.hidden)
                state_second = _project_linear_slice(env_first, state,
                    self.hidden, env_relative_start)
                environment_first = _project_linear_slice(env_first, environment,
                    0, self.hidden)
                environment_second = _project_linear_slice(env_first, environment,
                    self.hidden, env_relative_start)
                me_relative = _project_linear_slice(env_first, relative_me,
                    env_relative_start, env_first.in_features)
                me_values = _finish_projected_mlp(environment_message,
                    (state_first[:, :, None], environment_second[:, None], me_relative))
                me_gain = None
                env_aggregate = (me_values * me_weights[..., None]).sum(2)
                em_relative = _project_linear_slice(env_first, relative_em,
                    env_relative_start, env_first.in_features)
                em_values = _finish_projected_mlp(environment_message,
                    (environment_first[:, :, None], state_second[:, None], em_relative))
                em_gain = None
                source_aggregate = (em_values * em_weights[..., None]).sum(2)
            else:
                left = state[:, :, None].expand(-1, -1, modules, -1)
                right = state[:, None].expand(-1, modules, -1, -1)
                if module_pair_features is None:
                    module_pair_features = torch.cat((left, right, relative_mm), -1)
                module_values = module_message(module_pair_features)
                module_aggregate = (module_values * mm_weights[..., None]).sum(2)
                me_pair_features = (torch.cat((state[:, :, None].expand(-1, -1, environments, -1),
                    environment[:, None].expand(-1, modules, -1, -1), relative_me), -1)
                    if me_pair_features is None else me_pair_features)
                me_values = environment_message(me_pair_features)
                env_aggregate = (me_values * me_weights[..., None]).sum(2)
                em_pair_features = (torch.cat((environment[:, :, None].expand(-1, -1, modules, -1),
                    state[:, None].expand(-1, environments, -1, -1), relative_em), -1)
                    if em_pair_features is None else em_pair_features)
                em_values = environment_message(em_pair_features)
                source_aggregate = (em_values * em_weights[..., None]).sum(2)
                mm_gain = me_gain = em_gain = None
            if (self.native_aggregation_policy == 'destination_gain'
                    and self.native_routing_enabled):
                mm_gain = self.native_routing.gain('MM', state, context)
                me_gain = self.native_routing.gain('ME', state, context)
                em_gain = self.native_routing.gain('EM', environment, context)
                module_aggregate = module_aggregate * (1 + mm_gain)
                env_aggregate = env_aggregate * (1 + me_gain)
                source_aggregate = source_aggregate * (1 + em_gain)
            if collect_routes:
                for name, value in (('MM aggregate', module_aggregate),
                                    ('ME aggregate', env_aggregate),
                                    ('EM aggregate', source_aggregate)):
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f'Native interaction computation produced non-finite {name} values.')
            if route_trace is not None:
                route_trace.extend((
                    self._native_route_trace(
                        round_index=round_index, route='MM',
                        destination_ids=source_ids, donor_ids=source_ids,
                        destination_present=present, donor_present=present,
                        destination_measure=source_measures, donor_measure=source_measures,
                        self_mask=self_mask, base_weights=donor_mass,
                        scores=mm_scores, density=mm_density, weights=mm_weights,
                        values=module_values, aggregate=module_aggregate,
                        modulation_gain=mm_gain,
                        intervention=(intervention_kind if 'MM' in intervention_routes else None),
                        intervention_receipt=mm_receipt),
                    self._native_route_trace(
                        round_index=round_index, route='ME',
                        destination_ids=source_ids, donor_ids=environment_ids,
                        destination_present=present, donor_present=environment_present,
                        destination_measure=source_measures, donor_measure=environment_measures,
                        self_mask=None, base_weights=me_base_weights,
                        scores=me_scores, density=me_density, weights=me_weights,
                        values=me_values, aggregate=env_aggregate,
                        modulation_gain=me_gain,
                        intervention=(intervention_kind if 'ME' in intervention_routes else None),
                        intervention_receipt=me_receipt),
                    self._native_route_trace(
                        round_index=round_index, route='EM',
                        destination_ids=environment_ids, donor_ids=source_ids,
                        destination_present=environment_present, donor_present=present,
                        destination_measure=environment_measures, donor_measure=source_measures,
                        self_mask=None, base_weights=em_base_weights,
                        scores=em_scores, density=em_density, weights=em_weights,
                        values=em_values, aggregate=source_aggregate,
                        modulation_gain=em_gain,
                        intervention=(intervention_kind if 'EM' in intervention_routes else None),
                        intervention_receipt=em_receipt),
                ))
            env_pool = (environment * environment_mass[..., None]).sum(1)[:, None].expand(-1, environments, -1)
            state = (state + source_update(torch.cat((state, module_aggregate, env_aggregate, source_context), -1))) * present[..., None]
            environment = (environment + environment_update(torch.cat((environment, source_aggregate, env_pool, environment_context), -1))) * environment_present[..., None]
            if collect_routes and (not bool(torch.isfinite(state).all())
                                   or not bool(torch.isfinite(environment).all())):
                raise FloatingPointError('Native interaction update produced non-finite source or environment states.')
            if round_index == 0 and _between_rounds is not None:
                result = _between_rounds(state, environment)
                if not isinstance(result, tuple) or len(result) != 3:
                    raise TypeError('_between_rounds must return (source_states, environment_states, auxiliary_context).')
                state, environment, intermediate_context = result
                self._validate_shape(state, (batch, modules, self.hidden), 'intermediate source states')
                self._validate_shape(environment, (batch, environments, self.hidden),
                                     'intermediate environment states')
                if bool((state.masked_select((present <= 0)[..., None].expand_as(state)) != 0).any()):
                    raise ValueError('Intermediate source states must keep absent slots at zero.')
                if bool((environment.masked_select(
                        (environment_present <= 0)[..., None].expand_as(environment)) != 0).any()):
                    raise ValueError('Intermediate environment states must keep absent slots at zero.')
        global_state = self.global_encoder(torch.cat(((state * source_mass[..., None]).sum(1),
            (environment * environment_mass[..., None]).sum(1), context), -1))
        if ((self.native_aggregation_policy != 'measure' or routing_trace)
                and not bool(torch.isfinite(global_state).all())):
            raise FloatingPointError('Native interaction global context is non-finite.')
        prepared_type = getattr(self, 'prepared_context_type', PreparedResponseContext)
        prepared = prepared_type(state, environment, global_state, centers,
            source_lengths, present, source_mass, environment_coords, environment_present,
            environment_mass, lengths, source_ids, None, None, None, None, None, None, versions)
        prepared.intermediate_context = intermediate_context
        prepared.owner = id(self)
        prepared.model_reference = weakref.ref(self)
        prepared.parameter_signature = _parameter_signature(self)
        prepared.factorized_affine_execution = self.factorized_affine_execution
        prepared.factorized_affine_generation = self._factorized_affine_generation
        prepared.native_routing_generation = self._native_routing_generation
        prepared.native_routing_enabled = self.native_routing_enabled
        prepared.native_aggregation_policy = self.native_aggregation_policy
        prepared.native_routing_config_snapshot = copy.deepcopy(self.config.get('native_routing'))
        prepared.native_routing_trace = None if route_trace is None else tuple(route_trace)
        prepared.native_route_intervention = (intervention_kind, intervention_routes)
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
        if getattr(prepared, 'factorized_affine_execution', False) != self.factorized_affine_execution:
            raise ValueError('Prepared interaction execution mode changed; rebuild this request.')
        if prepared.native_routing_generation != self._native_routing_generation:
            raise ValueError('Prepared native routing policy changed; rebuild this request.')


class NonlinearFieldReadout(InteractionContextCore):
    """Source-wise nonlinear messages followed by a nonlinear field head.

    Environment is read through the prepared context/global state. No dense
    receiver/environment fine stack or scalar affine forcing law is executed.
    """

    output_law = 'nonlinear'

    def __init__(self, source_width, context_width, environment_width, *,
                 spatial_dim=3, hidden=64, message=64, output_width=3,
                 query_width=0, max_sources=None, factorized_affine_execution=False,
                 native_aggregation_policy='measure', native_control_hidden=None,
                 native_routing_seed=0):
        super().__init__(source_width, context_width, environment_width,
            spatial_dim=spatial_dim, hidden=hidden, message=message,
            query_width=query_width, max_sources=max_sources,
            factorized_affine_execution=factorized_affine_execution,
            native_aggregation_policy=native_aggregation_policy,
            native_control_hidden=native_control_hidden,
            native_routing_seed=native_routing_seed)
        if output_width < 1:
            raise ValueError('Nonlinear output width must be positive.')
        self.output_width = output_width
        self.source_read = _mlp(2 * hidden + 3 * spatial_dim * 9 + query_width, message, hidden)
        self.field_head = _mlp(hidden + message + spatial_dim * 9 + query_width, output_width, hidden)
        self.config.update(output_width=output_width)

    def _prepare_factorized_source_read(self, prepared, receivers, features, module):
        """Project receiver-independent and query-only terms once per call."""
        first = module[0]
        geometry_width = self.spatial_dim * 9
        start = 0
        source_stop = self.hidden
        global_stop = source_stop + self.hidden
        query_stop = global_stop + geometry_width
        relative1_stop = query_stop + geometry_width
        relative2_stop = relative1_stop + geometry_width
        source = _project_linear_slice(first, prepared.source_states, start, source_stop)
        global_state = _project_linear_slice(first, prepared.global_state, source_stop, global_stop)
        query_geometry = _project_linear_slice(first,
            _geometry(receivers / prepared.lengths[:, None]), global_stop, query_stop)
        receiver_features = _project_linear_slice(first, features, relative2_stop, first.in_features)
        return (source, global_state, query_geometry, receiver_features,
                query_stop, relative1_stop, relative2_stop)

    def _factorized_source_read(self, prepared, query, module, projections, start,
                                extra_receiver_projection=None):
        """Read a query chunk without building its concatenated Q-by-M inputs."""
        source, global_state, query_geometry, receiver_features, query_stop, relative1_stop, relative2_stop = projections
        delta = query[:, :, None] - prepared.centers[:, None]
        domain_relative = _geometry(delta / prepared.lengths[:, None, None])
        source_relative = _geometry(delta / prepared.source_lengths[:, None, :, None].clamp_min(1e-12))
        first = module[0]
        terms = (
            source[:, None],
            global_state[:, None, None],
            query_geometry[:, start:start + query.shape[1], None],
            _project_linear_slice(first, domain_relative, query_stop, relative1_stop),
            _project_linear_slice(first, source_relative, relative1_stop, relative2_stop),
            receiver_features[:, start:start + query.shape[1], None],
        )
        if extra_receiver_projection is not None:
            terms += (extra_receiver_projection[:, :, None],)
        return _finish_projected_mlp(module, terms)

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
        read_projections = (self._prepare_factorized_source_read(
            prepared, receivers, features, self.source_read)
            if self.factorized_affine_execution else None)
        for start in range(0, queries, chunk_size):
            query, feature = receivers[:, start:start + chunk_size], features[:, start:start + chunk_size]
            if read_projections is None:
                fine = self.source_read(self._read_features(prepared, query,
                    prepared.source_states, prepared.centers, prepared.source_lengths, feature))
            else:
                fine = self._factorized_source_read(prepared, query, self.source_read,
                    read_projections, start)
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
