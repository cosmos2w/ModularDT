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
    NATIVE_ROUTE_TYPES,
    NativeInteractionRouting,
    density_normalize,
    intervention_view,
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


def _tensor_identity_signature(tensor):
    if tensor is None:
        return None
    return (id(tensor), tensor._version, tensor.data_ptr(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.device, tensor.dtype)


def _type_sensitive_value_signature(value):
    """Encode JSON-like metadata without Python's ``True == 1`` equality."""
    value_type = type(value)
    type_name = (value_type.__module__, value_type.__qualname__)
    if value_type is dict:
        entries = [(_type_sensitive_value_signature(key),
                    _type_sensitive_value_signature(item))
                   for key, item in value.items()]
        return type_name, tuple(sorted(entries, key=repr))
    if value_type in (list, tuple):
        return type_name, tuple(_type_sensitive_value_signature(item) for item in value)
    if value_type in (str, int, float, bool, type(None)):
        return type_name, value
    return type_name, repr(value)


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
    native_routing_views: int = 1
    native_validation_generation: int = 0
    consolidated_native_validation: bool = False
    native_aggregation_policy: str = 'measure'
    native_routing_config_snapshot: object | None = None
    native_routing_semantics_snapshot: object | None = None
    # Filled only when callers explicitly request a detached diagnostic trace.
    native_routing_trace: tuple[dict, ...] | None = None
    native_routing_trace_metadata: dict | None = None
    native_route_intervention: tuple[str | None, tuple[str, ...]] = (None, ())
    native_route_intervention_view: int | None = None

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
            if (hasattr(owner, '_native_validation_generation')
                    and self.native_validation_generation != owner._native_validation_generation):
                raise ValueError('Prepared native validation mode changed; rebuild this request.')
            if (hasattr(owner, 'consolidated_native_validation')
                    and self.consolidated_native_validation != owner.consolidated_native_validation):
                raise ValueError('Prepared native validation mode changed; rebuild this request.')
            if (hasattr(owner, 'native_routing_enabled')
                    and self.native_routing_enabled != owner.native_routing_enabled):
                raise ValueError('Prepared native routing execution changed; rebuild this request.')
            if (hasattr(owner, 'native_aggregation_policy')
                    and self.native_aggregation_policy != owner.native_aggregation_policy):
                raise ValueError('Prepared native aggregation policy changed; rebuild this request.')
            if (hasattr(owner, 'native_routing_views')
                    and self.native_routing_views != owner.native_routing_views):
                raise ValueError('Prepared native routing views changed; rebuild this request.')
            if (self.native_routing_config_snapshot is not None
                    and self.native_routing_config_snapshot
                    != _type_sensitive_value_signature(owner.config.get('native_routing'))):
                raise ValueError('Prepared native routing config changed; rebuild this request.')
            if self.native_routing_semantics_snapshot is not None:
                try:
                    active_routing_semantics = owner._native_routing_semantics_signature()
                except (TypeError, ValueError, AttributeError, KeyError) as exc:
                    raise ValueError(
                        'Prepared native routing semantics changed; rebuild this request.'
                    ) from exc
                if active_routing_semantics != self.native_routing_semantics_snapshot:
                    raise ValueError(
                        'Prepared native routing semantics changed; rebuild this request.'
                    )


@dataclass(frozen=True)
class PreparedNonlinearReadPlan:
    """Bound vectorized source/receiver pairs for one nonlinear final read.

    The plan is an execution artifact, not model state. It binds the exact
    prepared context and receiver tensors plus explicit policy and receiver
    generations so a selected support cannot silently outlive its inputs.
    """

    mode: str
    policy_identity: str
    policy_generation: int
    receiver_generation: int
    owner_id: int
    prepared: object
    source_ids: torch.Tensor
    receivers: torch.Tensor
    receiver_ids: torch.Tensor | None
    receiver_features: torch.Tensor | None
    context_bindings: tuple
    source_ids_binding: tuple
    receiver_binding: tuple
    receiver_ids_binding: tuple | None
    receiver_features_binding: tuple | None
    parameter_signature: tuple
    pair_batch: torch.Tensor
    pair_receiver: torch.Tensor
    pair_source: torch.Tensor
    pair_versions: tuple
    active_pair_count: int
    logical_pair_count: int
    batch_size: int
    query_count: int
    source_capacity: int

    @property
    def evaluated_pair_count(self):
        return self.pair_batch.numel()

    @property
    def padding_pair_count(self):
        return self.logical_pair_count - self.active_pair_count

    @property
    def omitted_active_pair_count(self):
        return self.active_pair_count - self.evaluated_pair_count

    def reduction_workspace_elements(self, chunk_size, message_width):
        """Return dense reduction scratch capacity, separate from evaluated rows.

        Selected execution still allocates a source-axis reduction workspace
        for each query chunk so the legacy ``sum(source_axis)`` order is
        preserved. This capacity is therefore not a sparse-work count.
        """
        if (type(chunk_size) is not int or chunk_size < 1
                or type(message_width) is not int or message_width < 1):
            raise ValueError('Workspace accounting needs positive integer chunk and message widths.')
        return (self.batch_size * min(self.query_count, chunk_size)
                * self.source_capacity * message_width)

    def assert_compatible(self, model, prepared, receivers, receiver_ids,
                          receiver_features, *,
                          policy_identity, policy_generation, receiver_generation):
        if model is None or id(model) != self.owner_id:
            raise ValueError('Prepared nonlinear read plan belongs to another operator.')
        owner_reference = getattr(prepared, 'model_reference', None)
        if owner_reference is None or owner_reference() is not model:
            raise ValueError('Prepared nonlinear read plan owner is stale; rebuild this request.')
        if prepared is not self.prepared:
            raise ValueError('Prepared nonlinear read plan belongs to another context.')
        if self.mode not in ('active_dense', 'selected'):
            raise ValueError('Prepared nonlinear read plan mode changed; rebuild this request.')
        if receivers is not self.receivers or receiver_features is not self.receiver_features:
            raise ValueError('Prepared nonlinear read plan receiver inputs changed; rebuild this request.')
        if receiver_ids is not self.receiver_ids:
            raise ValueError('Prepared nonlinear read plan receiver IDs changed; rebuild this request.')
        if _tensor_identity_signature(receivers) != self.receiver_binding:
            raise ValueError('Prepared nonlinear read plan receiver generation changed; rebuild this request.')
        if _tensor_identity_signature(receiver_ids) != self.receiver_ids_binding:
            raise ValueError('Prepared nonlinear read plan receiver IDs changed; rebuild this request.')
        if _tensor_identity_signature(receiver_features) != self.receiver_features_binding:
            raise ValueError('Prepared nonlinear read plan receiver features changed; rebuild this request.')
        if (getattr(prepared, 'source_ids', None) is not self.source_ids
                or _tensor_identity_signature(getattr(prepared, 'source_ids', None))
                != self.source_ids_binding):
            raise ValueError('Prepared nonlinear read plan source IDs changed; rebuild this request.')
        bindings = tuple(_tensor_identity_signature(getattr(prepared, name)) for name in (
            'source_states', 'global_state', 'centers', 'source_lengths', 'present',
            'source_measure', 'lengths', 'source_ids'))
        if bindings != self.context_bindings:
            raise ValueError('Prepared nonlinear read plan context inputs changed; rebuild this request.')
        if _parameter_signature(model) != self.parameter_signature:
            raise ValueError('Prepared nonlinear read plan model weights changed; rebuild this request.')
        if (policy_identity != self.policy_identity
                or type(policy_generation) is not int
                or policy_generation != self.policy_generation):
            raise ValueError('Prepared nonlinear read plan input policy changed; rebuild this request.')
        if (getattr(model, '_read_policy_identity', None) != self.policy_identity
                or getattr(model, '_read_policy_generation', None) != self.policy_generation):
            raise ValueError('Prepared nonlinear read plan model policy generation changed; rebuild this request.')
        if (type(receiver_generation) is not int
                or receiver_generation != self.receiver_generation):
            raise ValueError('Prepared nonlinear read plan receiver generation changed; rebuild this request.')
        current_pair_versions = tuple(tensor._version for tensor in (
            self.pair_batch, self.pair_receiver, self.pair_source))
        if current_pair_versions != self.pair_versions:
            raise ValueError('Prepared nonlinear read plan pairs changed; rebuild this request.')


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
                 native_routing_seed=0, native_routing_views=1):
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
        if type(native_routing_views) is not int or native_routing_views not in (1, 2):
            raise ValueError('Native routing supports one or two feature views.')
        if native_aggregation_policy != 'learned_density' and native_routing_views != 1:
            raise ValueError('Only learned-density routing supports multiple feature views.')
        if native_routing_views == 2 and message != 128:
            raise ValueError('P-R2 requires one existing 128-coordinate message with two 64-coordinate views.')
        self.native_aggregation_policy = native_aggregation_policy
        self.native_control_hidden = native_control_hidden
        self._native_routing_generation = 0
        self.native_routing_views = native_routing_views
        self._native_validation_generation = 0
        self._consolidated_native_validation = False
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
                views=native_routing_views,
            )
            routing_config = {
                'schema': ('native_interaction_routing_v2' if native_routing_views == 2
                           else 'native_interaction_routing_v1'),
                'policy': native_aggregation_policy,
                'initialization_seed': int(native_routing_seed),
            }
            if native_aggregation_policy == 'learned_density':
                routing_config.update(score_hidden=NativeInteractionRouting.SCORE_HIDDEN,
                                      score_bound=NativeInteractionRouting.SCORE_BOUND)
                if native_routing_views == 2:
                    routing_config.update(views=2,
                                          feature_blocks=list(self.native_routing.feature_blocks))
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

    def set_consolidated_native_validation(self, enabled: bool) -> None:
        """Opt into one synchronization for route-intermediate finite checks.

        The ordinary path retains eager public score/gain checks. This runtime
        switch is intentionally absent from serialized model configuration.
        """
        if type(enabled) is not bool:
            raise TypeError('consolidated native validation setting must be a bool.')
        self.consolidated_native_validation = enabled

    @property
    def consolidated_native_validation(self) -> bool:
        return self._consolidated_native_validation

    @consolidated_native_validation.setter
    def consolidated_native_validation(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError('consolidated native validation setting must be a bool.')
        previous = getattr(self, '_consolidated_native_validation', None)
        if previous is not None and previous != enabled:
            self._native_validation_generation += 1
        self._consolidated_native_validation = enabled

    @property
    def native_routing_views(self) -> int:
        return self._native_routing_views

    @native_routing_views.setter
    def native_routing_views(self, views: int) -> None:
        if type(views) is not int or views not in (1, 2):
            raise ValueError('Native routing supports one or two feature views.')
        if getattr(self, 'native_aggregation_policy', None) != 'learned_density' and views != 1:
            raise ValueError('Only learned-density routing supports multiple feature views.')
        if getattr(self, 'config', {}).get('message') and views == 2 and self.config['message'] != 128:
            raise ValueError('P-R2 requires one existing 128-coordinate message with two 64-coordinate views.')
        previous = getattr(self, '_native_routing_views', None)
        if previous is not None and previous != views:
            raise ValueError('Native routing view count is architectural; construct a new model.')
        self._native_routing_views = views

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
                              intervention_routes, intervention_view_index,
                              finite_flags):
        scores = None
        if (self.native_aggregation_policy == 'learned_density'
                and self.native_routing_enabled):
            scores = self.native_routing.score(
                route, pair_features, finite_flags=finite_flags)
            density, weights = density_normalize(base_weights, scores)
        else:
            if self.native_routing_views == 2:
                density = torch.ones((*base_weights.shape, 2), dtype=base_weights.dtype,
                                     device=base_weights.device)
                weights = base_weights[..., None] * density
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
                raise ValueError('Native route interventions require active P-R/P-R2 density routing.')
            if intervention_kind == 'density_reset':
                eligible = (base_weights > 0 if density.ndim == 3 else
                            (base_weights > 0)[..., None])
                if intervention_view_index is None:
                    receipt['changed_donors'] = int(((density != 1) & eligible).sum().item())
                    density = torch.ones_like(density)
                else:
                    if intervention_view_index >= self.native_routing_views:
                        raise ValueError('Native route intervention view is outside the model view axis.')
                    if density.ndim == 3 and intervention_view_index != 0:
                        raise ValueError('Single-view P-R exposes only view zero.')
                    if density.ndim == 3:
                        receipt['changed_donors'] = int(
                            ((density != 1) & (base_weights > 0)).sum().item())
                        density = torch.ones_like(density)
                    else:
                        receipt['changed_donors'] = int(
                            ((density[..., intervention_view_index] != 1)
                             & (base_weights > 0)).sum().item())
                        density = density.clone()
                        density[..., intervention_view_index] = 1
            elif intervention_kind == 'average_collapse':
                if self.native_routing_views != 2 or density.ndim != 4:
                    raise ValueError('Average-collapse intervention requires two-view P-R2.')
                collapsed = density.mean(-1, keepdim=True).expand_as(density)
                eligible = (base_weights > 0)[..., None]
                receipt['changed_donors'] = int(
                    ((density != collapsed) & eligible).sum().item())
                density = collapsed
            elif intervention_kind == 'equal_measure_reassignment':
                if intervention_view_index is None:
                    density, receipt = reassign_equal_measure_densities(
                        density, base_weights, measures, present, donor_ids)
                elif density.ndim == 3:
                    if intervention_view_index != 0:
                        raise ValueError('Single-view P-R exposes only view zero.')
                    density, receipt = reassign_equal_measure_densities(
                        density, base_weights, measures, present, donor_ids)
                else:
                    if intervention_view_index >= self.native_routing_views:
                        raise ValueError('Native route intervention view is outside the model view axis.')
                    reassigned, receipt = reassign_equal_measure_densities(
                        density[..., intervention_view_index], base_weights, measures,
                        present, donor_ids)
                    density = density.clone()
                    density[..., intervention_view_index] = reassigned
            else:
                raise ValueError(f'Unsupported native route intervention {intervention_kind!r}.')
            if density.ndim == 3:
                weights = base_weights * density
            else:
                weights = base_weights[..., None] * density
        for name, value in (('score', scores), ('density', density), ('weight', weights)):
            if value is None:
                continue
            finite = torch.isfinite(value).all()
            if finite_flags is None:
                if not bool(finite):
                    raise FloatingPointError(f'Native {route} routing produced non-finite {name} values.')
            else:
                finite_flags.append((f'Native {route} routing {name}', finite))
        return scores, density, weights, receipt

    @classmethod
    def _native_route_trace(cls, *, round_index, route, destination_ids,
                            donor_ids, destination_present, donor_present,
                            destination_measure, donor_measure, self_mask,
                            base_weights, scores, density, weights,
                            values, aggregate, modulation_gain, intervention,
                            intervention_view_index, intervention_receipt):
        donor_type, destination_type = route_endpoint_types(route)
        if weights.ndim == 4:
            view_count = weights.shape[-1]
            block_width = values.shape[-1] // view_count
            value_blocks = values.reshape(*values.shape[:-1], view_count, block_width)
            weighted_content = value_blocks * weights[..., None]
            head_aggregate = weighted_content.sum(2)
            weighted_norm = torch.linalg.vector_norm(weighted_content, dim=-1)
        else:
            view_count = 1
            weighted_content = values * weights[..., None]
            head_aggregate = weighted_content.sum(2).unsqueeze(-2)
            weighted_norm = torch.linalg.vector_norm(weighted_content, dim=-1)
        return {
            'round': round_index + 1,
            'legacy_route': route,
            'view_ids': cls._diagnostic_copy(torch.arange(view_count, device=weights.device)),
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
            'row_mass': cls._diagnostic_copy(base_weights.sum(-1)),
            's': cls._diagnostic_copy(scores),
            'rho': cls._diagnostic_copy(density),
            'w': cls._diagnostic_copy(weights),
            'weighted_message_content': cls._diagnostic_copy(weighted_content),
            'head_aggregate': cls._diagnostic_copy(head_aggregate),
            'weighted_message_norm': cls._diagnostic_copy(weighted_norm),
            'aggregate': cls._diagnostic_copy(aggregate),
            'modulation_gain': cls._diagnostic_copy(modulation_gain),
            'intervention': intervention,
            'intervention_view': intervention_view_index,
            'intervention_receipt': dict(intervention_receipt),
        }

    def _native_routing_trace_metadata(self, routing_semantics=None) -> dict:
        if routing_semantics is None:
            routing_semantics = self._native_routing_semantics_signature()
        output_law = {
            key: getattr(self, key, None)
            for key in ('output_law', 'field_output_law', 'response_output_law')
        }
        return {
            'owner_id': id(self),
            'configuration': copy.deepcopy(self.config),
            'weight_freshness_binding': _parameter_signature(self),
            'routing_generation': self._native_routing_generation,
            'routing_semantics_version': 'native_interaction_routing_semantics_v1',
            'active_routing_semantics': routing_semantics,
            'native_validation_generation': self._native_validation_generation,
            'output_law_metadata': output_law,
        }

    def _native_routing_semantics_signature(self):
        """Validate and bind the active route module and its declared config.

        ``config`` is a mutable export surface. Prepared state therefore binds
        both its exact typed representation and the live module attributes
        that determine view layout and route evaluation. This is runtime-only;
        it does not alter serialized model configuration or parameter state.
        """
        policy = self.native_aggregation_policy
        if policy == 'measure':
            return None
        if type(policy) is not str or policy not in ('learned_density', 'destination_gain'):
            raise ValueError('Native routing active semantics have an unsupported core policy.')

        routing = getattr(self, 'native_routing', None)
        if routing is None:
            raise ValueError('Native routing active semantics are missing the route module.')
        semantics_generation = routing.routing_semantics_generation
        if type(semantics_generation) is not int or semantics_generation < 0:
            raise ValueError('Native routing semantic generation must be a nonnegative integer.')
        views = self.native_routing_views
        if type(views) is not int or views not in (1, 2):
            raise ValueError('Native routing active semantics have an invalid core view count.')
        if type(routing.policy) is not str or routing.policy != policy:
            raise ValueError('Native routing module policy disagrees with the core policy.')
        if type(routing.views) is not int or routing.views != views:
            raise ValueError('Native routing module views disagree with the core view count.')
        if (type(routing.feature_blocks) is not tuple
                or any(type(width) is not int for width in routing.feature_blocks)):
            raise ValueError('Native routing module feature blocks must be an integer tuple.')

        hidden = self.config.get('hidden')
        context_width = self.config.get('context_width')
        message_width = self.config.get('message')
        spatial_dim = self.config.get('spatial_dim')
        if any(type(value) is not int for value in
               (hidden, context_width, message_width, spatial_dim)):
            raise ValueError('Native routing core widths must retain their declared integer types.')
        expected_pair_width = 2 * hidden + spatial_dim * 9
        expected_feature_blocks = ((message_width,) if views == 1 else (64, 64))
        if (type(routing.pair_width) is not int or routing.pair_width != expected_pair_width
                or type(routing.destination_width) is not int
                or routing.destination_width != hidden
                or type(routing.context_width) is not int
                or routing.context_width != context_width
                or type(routing.message_width) is not int
                or routing.message_width != message_width
                or routing.feature_blocks != expected_feature_blocks):
            raise ValueError('Native routing module widths or feature partition disagree with the core.')
        if (type(routing.initialization_seed) is not int
                or routing.initialization_seed < 0):
            raise ValueError('Native routing initialization seed must retain its nonnegative integer type.')

        expected_config = {
            'schema': ('native_interaction_routing_v2' if views == 2
                       else 'native_interaction_routing_v1'),
            'policy': policy,
            'initialization_seed': routing.initialization_seed,
        }
        if policy == 'learned_density':
            if type(routing.SCORE_HIDDEN) is not int or routing.SCORE_HIDDEN < 1:
                raise ValueError('Native score hidden width must be a positive integer.')
            if (type(routing.SCORE_BOUND) is not float
                    or not math.isfinite(routing.SCORE_BOUND)
                    or routing.SCORE_BOUND <= 0):
                raise ValueError('Native score bound must be a finite positive float.')
            if routing.control_hidden is not None:
                raise ValueError('Learned-density routing cannot declare a destination-gain width.')
            expected_config.update(score_hidden=routing.SCORE_HIDDEN,
                                   score_bound=routing.SCORE_BOUND)
            if views == 2:
                expected_config.update(views=2, feature_blocks=[64, 64])
            if (tuple(routing.score_mlps.keys()) != NATIVE_ROUTE_TYPES
                    or tuple(routing.aggregate_modulators.keys())):
                raise ValueError('Native learned-density route module inventory is inconsistent.')
            network_layout = []
            output_width = views
            for route in NATIVE_ROUTE_TYPES:
                network = routing.score_mlps[route]
                if (type(network) is not nn.Sequential or len(network) != 3
                        or type(network[0]) is not nn.Linear
                        or type(network[1]) is not nn.SiLU
                        or type(network[2]) is not nn.Linear
                        or network[0].in_features != expected_pair_width
                        or network[0].out_features != routing.SCORE_HIDDEN
                        or network[1].inplace
                        or network[2].in_features != routing.SCORE_HIDDEN
                        or network[2].out_features != output_width
                        or network[2].bias is not None):
                    raise ValueError(f'Native {route} score network disagrees with its sealed widths.')
                network_layout.append((route, tuple(
                    (type(layer).__module__, type(layer).__qualname__,
                     getattr(layer, 'in_features', None),
                     getattr(layer, 'out_features', None),
                     getattr(layer, 'inplace', None),
                     getattr(layer, 'bias', None) is not None)
                    for layer in network)))
        else:
            control_hidden = routing.control_hidden
            if type(control_hidden) is not int or control_hidden < 1:
                raise ValueError('Native destination-gain width must be a positive integer.')
            if type(self.native_control_hidden) is not int or self.native_control_hidden != control_hidden:
                raise ValueError('Native destination-gain module width disagrees with the core.')
            if (tuple(routing.aggregate_modulators.keys()) != NATIVE_ROUTE_TYPES
                    or tuple(routing.score_mlps.keys())):
                raise ValueError('Native destination-gain route module inventory is inconsistent.')
            expected_config['control_hidden'] = control_hidden
            network_layout = []
            for route in NATIVE_ROUTE_TYPES:
                network = routing.aggregate_modulators[route]
                if (type(network) is not nn.Sequential or len(network) != 3
                        or type(network[0]) is not nn.Linear
                        or type(network[1]) is not nn.SiLU
                        or type(network[2]) is not nn.Linear
                        or network[0].in_features != hidden + context_width
                        or network[0].out_features != control_hidden
                        or network[1].inplace
                        or network[2].in_features != control_hidden
                        or network[2].out_features != message_width
                        or network[2].bias is None):
                    raise ValueError(f'Native {route} destination-gain network disagrees with its sealed widths.')
                network_layout.append((route, tuple(
                    (type(layer).__module__, type(layer).__qualname__,
                     getattr(layer, 'in_features', None),
                     getattr(layer, 'out_features', None),
                     getattr(layer, 'inplace', None),
                     getattr(layer, 'bias', None) is not None)
                    for layer in network)))

        if hasattr(self, 'mode'):
            mode = self.mode
            if type(mode) is not str:
                raise ValueError('Native routing mode must retain its declared string type.')
            expected_config['mode'] = mode
        declared_config = self.config.get('native_routing')
        if (type(declared_config) is not dict
                or _type_sensitive_value_signature(declared_config)
                != _type_sensitive_value_signature(expected_config)):
            raise ValueError('Native routing declared config disagrees with active module semantics.')

        active_widths = (policy, views, routing.pair_width, routing.destination_width,
                         routing.context_width, routing.message_width,
                         routing.control_hidden, routing.initialization_seed,
                         routing.SCORE_HIDDEN, routing.SCORE_BOUND,
                         routing.feature_blocks, semantics_generation)
        return (
            'native_interaction_routing_semantics_v1',
            _type_sensitive_value_signature(active_widths),
            _type_sensitive_value_signature(declared_config),
            _type_sensitive_value_signature(tuple(network_layout)),
        )

    @staticmethod
    def _check_consolidated_finite_flags(flags):
        if not flags:
            return
        combined = torch.stack([flag for _name, flag in flags]).all()
        if not bool(combined):
            checked = ', '.join(name for name, _flag in flags)
            raise FloatingPointError(
                'Native interaction preparation produced non-finite intermediate values; '
                f'checked flags: {checked}.')

    def _native_gather(self, values, weights):
        """Gather one full message or R2's fixed equal-width feature blocks."""
        if self.native_routing_views == 1:
            return (values * weights[..., None]).sum(2)
        views = self.native_routing_views
        block_width = values.shape[-1] // views
        if weights.shape[-1] != views or block_width * views != values.shape[-1]:
            raise ValueError('Native view weights and message feature blocks are incompatible.')
        blocks = values.reshape(*values.shape[:-1], views, block_width)
        return (blocks * weights[..., None]).sum(2).reshape(*values.shape[:2], values.shape[-1])

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
        native_routing_semantics = self._native_routing_semantics_signature()
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
        intervention_view_index = intervention_view(native_route_intervention)
        if intervention_kind is not None and self.native_aggregation_policy != 'learned_density':
            raise ValueError('Native route interventions are supported only by P-R/P-R2.')
        if intervention_view_index is not None:
            if intervention_view_index >= self.native_routing_views:
                raise ValueError('Native route intervention view is outside the model view axis.')
            if intervention_kind not in ('density_reset', 'equal_measure_reassignment'):
                raise ValueError('A selected view is supported only for reset or donor reassignment.')
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
        trace_metadata = (self._native_routing_trace_metadata(native_routing_semantics)
                          if routing_trace else None)
        finite_flags = [] if self.consolidated_native_validation else None
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
                    source_ids, intervention_kind, intervention_routes,
                    intervention_view_index, finite_flags)
                me_scores, me_density, me_weights, me_receipt = self._native_route_weights(
                    'ME', me_base_weights, me_pair_features, environment_measures,
                    environment_present, environment_ids, intervention_kind,
                    intervention_routes, intervention_view_index, finite_flags)
                em_scores, em_density, em_weights, em_receipt = self._native_route_weights(
                    'EM', em_base_weights, em_pair_features, source_measures,
                    present, source_ids, intervention_kind, intervention_routes,
                    intervention_view_index, finite_flags)
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
                module_aggregate = self._native_gather(module_values, mm_weights)

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
                env_aggregate = self._native_gather(me_values, me_weights)
                em_relative = _project_linear_slice(env_first, relative_em,
                    env_relative_start, env_first.in_features)
                em_values = _finish_projected_mlp(environment_message,
                    (environment_first[:, :, None], state_second[:, None], em_relative))
                em_gain = None
                source_aggregate = self._native_gather(em_values, em_weights)
            else:
                left = state[:, :, None].expand(-1, -1, modules, -1)
                right = state[:, None].expand(-1, modules, -1, -1)
                if module_pair_features is None:
                    module_pair_features = torch.cat((left, right, relative_mm), -1)
                module_values = module_message(module_pair_features)
                module_aggregate = self._native_gather(module_values, mm_weights)
                me_pair_features = (torch.cat((state[:, :, None].expand(-1, -1, environments, -1),
                    environment[:, None].expand(-1, modules, -1, -1), relative_me), -1)
                    if me_pair_features is None else me_pair_features)
                me_values = environment_message(me_pair_features)
                env_aggregate = self._native_gather(me_values, me_weights)
                em_pair_features = (torch.cat((environment[:, :, None].expand(-1, -1, modules, -1),
                    state[:, None].expand(-1, environments, -1, -1), relative_em), -1)
                    if em_pair_features is None else em_pair_features)
                em_values = environment_message(em_pair_features)
                source_aggregate = self._native_gather(em_values, em_weights)
                mm_gain = me_gain = em_gain = None
            if (self.native_aggregation_policy == 'destination_gain'
                    and self.native_routing_enabled):
                mm_gain = self.native_routing.gain(
                    'MM', state, context, finite_flags=finite_flags)
                me_gain = self.native_routing.gain(
                    'ME', state, context, finite_flags=finite_flags)
                em_gain = self.native_routing.gain(
                    'EM', environment, context, finite_flags=finite_flags)
                module_aggregate = module_aggregate * (1 + mm_gain)
                env_aggregate = env_aggregate * (1 + me_gain)
                source_aggregate = source_aggregate * (1 + em_gain)
            if collect_routes:
                for name, value in (('MM aggregate', module_aggregate),
                                    ('ME aggregate', env_aggregate),
                                    ('EM aggregate', source_aggregate)):
                    finite = torch.isfinite(value).all()
                    if finite_flags is None:
                        if not bool(finite):
                            raise FloatingPointError(
                                f'Native interaction computation produced non-finite {name} values.')
                    else:
                        finite_flags.append((name, finite))
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
                        intervention_view_index=(intervention_view_index
                                                 if 'MM' in intervention_routes else None),
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
                        intervention_view_index=(intervention_view_index
                                                 if 'ME' in intervention_routes else None),
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
                        intervention_view_index=(intervention_view_index
                                                 if 'EM' in intervention_routes else None),
                        intervention_receipt=em_receipt),
                ))
            env_pool = (environment * environment_mass[..., None]).sum(1)[:, None].expand(-1, environments, -1)
            state = (state + source_update(torch.cat((state, module_aggregate, env_aggregate, source_context), -1))) * present[..., None]
            environment = (environment + environment_update(torch.cat((environment, source_aggregate, env_pool, environment_context), -1))) * environment_present[..., None]
            if collect_routes:
                source_finite = torch.isfinite(state).all()
                environment_finite = torch.isfinite(environment).all()
                if finite_flags is None:
                    if not bool(source_finite) or not bool(environment_finite):
                        raise FloatingPointError(
                            'Native interaction update produced non-finite source or environment states.')
                else:
                    finite_flags.append((f'Round {round_index + 1} source states', source_finite))
                    finite_flags.append((f'Round {round_index + 1} environment states',
                                         environment_finite))
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
        if self.native_aggregation_policy != 'measure' or routing_trace:
            global_finite = torch.isfinite(global_state).all()
            if finite_flags is None:
                if not bool(global_finite):
                    raise FloatingPointError('Native interaction global context is non-finite.')
            else:
                finite_flags.append(('Global interaction state', global_finite))
        if finite_flags is not None:
            self._check_consolidated_finite_flags(finite_flags)
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
        prepared.native_routing_views = self.native_routing_views
        prepared.native_validation_generation = self._native_validation_generation
        prepared.consolidated_native_validation = self.consolidated_native_validation
        prepared.native_aggregation_policy = self.native_aggregation_policy
        prepared.native_routing_config_snapshot = (
            _type_sensitive_value_signature(self.config.get('native_routing'))
            if self.native_aggregation_policy != 'measure' else None)
        prepared.native_routing_semantics_snapshot = native_routing_semantics
        prepared.native_routing_trace = None if route_trace is None else tuple(route_trace)
        prepared.native_routing_trace_metadata = trace_metadata
        prepared.native_route_intervention = (intervention_kind, intervention_routes)
        prepared.native_route_intervention_view = intervention_view_index
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
                 native_routing_seed=0, native_routing_views=1):
        super().__init__(source_width, context_width, environment_width,
            spatial_dim=spatial_dim, hidden=hidden, message=message,
            query_width=query_width, max_sources=max_sources,
            factorized_affine_execution=factorized_affine_execution,
            native_aggregation_policy=native_aggregation_policy,
            native_control_hidden=native_control_hidden,
            native_routing_seed=native_routing_seed,
            native_routing_views=native_routing_views)
        if output_width < 1:
            raise ValueError('Nonlinear output width must be positive.')
        self.output_width = output_width
        self.source_read = _mlp(2 * hidden + 3 * spatial_dim * 9 + query_width, message, hidden)
        self.field_head = _mlp(hidden + message + spatial_dim * 9 + query_width, output_width, hidden)
        self._read_policy_identity = 'active_dense_v1'
        self._read_policy_generation = 0
        self.config.update(output_width=output_width)

    def bind_read_policy_generation(self, policy_identity, generation):
        """Bind the runtime read policy whose plans this model may execute.

        This execution binding adds no parameter or buffer and is deliberately
        absent from ``state_dict``. Callers advance it when a selector or its
        configuration changes; already prepared plans then fail closed.
        """
        if (not isinstance(policy_identity, str) or not policy_identity
                or type(generation) is not int or generation < 0):
            raise ValueError('Read policy binding requires a name and nonnegative integer generation.')
        if (policy_identity == self._read_policy_identity
                and generation < self._read_policy_generation):
            raise ValueError('Read policy generation cannot move backward for the same policy.')
        self._read_policy_identity = policy_identity
        self._read_policy_generation = generation

    def prepare_read_plan(self, prepared, receivers, receiver_features=None, *, mode,
                          policy_identity, policy_generation, receiver_generation,
                          selected_pairs=None, receiver_ids=None):
        """Bind active-only or selected nonlinear reads to one prepared request.

        ``selected_pairs`` is a boolean ``[B,Q,M]`` support. It must be a
        subset of active physical sources. ``mode='active_dense'`` instead
        admits every active physical source and is the padding-removal
        control. The plan keeps the original prepared source measures; it
        never renormalizes them over the admitted subset.
        """
        self.assert_owned(prepared)
        if mode not in ('active_dense', 'selected'):
            raise ValueError("Read-plan mode must be 'active_dense' or 'selected'.")
        if mode == 'active_dense' and selected_pairs is not None:
            raise ValueError('Active-dense read plans do not accept a selected support.')
        if mode == 'selected' and selected_pairs is None:
            raise ValueError('Selected read plans require an explicit boolean support.')
        if (not isinstance(policy_identity, str) or not policy_identity
                or type(policy_generation) is not int or policy_generation < 0
                or type(receiver_generation) is not int or receiver_generation < 0):
            raise ValueError('Read plans require a named policy and nonnegative integer generations.')
        if (policy_identity != self._read_policy_identity
                or policy_generation != self._read_policy_generation):
            raise ValueError('Read plan does not match the model-bound input policy generation.')
        batch, queries, _ = receivers.shape
        self._validate_shape(receivers, (prepared.centers.shape[0], queries, self.spatial_dim), 'receivers')
        features = receivers.new_empty(batch, queries, 0) if receiver_features is None else receiver_features
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        if receivers.device != prepared.centers.device or features.device != receivers.device:
            raise ValueError('Read-plan context, receivers and features must share a device.')
        source_ids = getattr(prepared, 'source_ids', None)
        if (not torch.is_tensor(source_ids)
                or tuple(source_ids.shape) != (batch, prepared.centers.shape[1])
                or source_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Read plans require aligned integer physical source IDs.')
        if (receiver_ids is not None
                and (not torch.is_tensor(receiver_ids)
                     or tuple(receiver_ids.shape) != (batch, queries)
                     or receiver_ids.dtype not in (torch.int32, torch.int64))):
            raise ValueError('Receiver IDs must be an integer tensor with shape [B,Q].')
        active = (prepared.present > 0)[:, None, :].expand(batch, queries, -1)
        if mode == 'active_dense':
            admitted = active
        else:
            if (not torch.is_tensor(selected_pairs)
                    or selected_pairs.dtype != torch.bool
                    or tuple(selected_pairs.shape) != (batch, queries, prepared.centers.shape[1])
                    or selected_pairs.device != receivers.device):
                raise ValueError('Selected support must be boolean [B,Q,M] on the receiver device.')
            if bool((selected_pairs & ~active).any()):
                raise ValueError('Selected read support includes an absent or padded source.')
            admitted = selected_pairs
        pair_batch, pair_receiver, pair_source = admitted.nonzero(as_tuple=True)
        context_bindings = tuple(_tensor_identity_signature(getattr(prepared, name)) for name in (
            'source_states', 'global_state', 'centers', 'source_lengths', 'present',
            'source_measure', 'lengths', 'source_ids'))
        pair_tensors = (pair_batch, pair_receiver, pair_source)
        return PreparedNonlinearReadPlan(
            mode=mode,
            policy_identity=policy_identity,
            policy_generation=policy_generation,
            receiver_generation=receiver_generation,
            owner_id=id(self),
            prepared=prepared,
            source_ids=source_ids,
            receivers=receivers,
            receiver_ids=receiver_ids,
            receiver_features=receiver_features,
            context_bindings=context_bindings,
            source_ids_binding=_tensor_identity_signature(source_ids),
            receiver_binding=_tensor_identity_signature(receivers),
            receiver_ids_binding=_tensor_identity_signature(receiver_ids),
            receiver_features_binding=_tensor_identity_signature(receiver_features),
            parameter_signature=_parameter_signature(self),
            pair_batch=pair_batch,
            pair_receiver=pair_receiver,
            pair_source=pair_source,
            pair_versions=tuple(tensor._version for tensor in pair_tensors),
            active_pair_count=int(active.sum().item()),
            logical_pair_count=batch * queries * prepared.centers.shape[1],
            batch_size=batch,
            query_count=queries,
            source_capacity=prepared.centers.shape[1],
        )

    def _predict_with_read_plan(self, prepared, receivers, features, read_plan, *,
                                chunk_size, return_messages):
        """Execute admitted pairs; exported messages are zero on omitted rows."""
        batch, queries, _ = receivers.shape
        values, messages = [], []
        for start in range(0, queries, chunk_size):
            stop = min(start + chunk_size, queries)
            query, feature = receivers[:, start:stop], features[:, start:stop]
            in_chunk = ((read_plan.pair_receiver >= start)
                        & (read_plan.pair_receiver < stop))
            pair_batch = read_plan.pair_batch[in_chunk]
            pair_receiver = read_plan.pair_receiver[in_chunk] - start
            pair_source = read_plan.pair_source[in_chunk]
            selected = (pair_batch, pair_receiver, pair_source)
            fine = self.source_read(self._read_selected_features(
                prepared, query, feature, selected))
            weighted = fine * prepared.source_measure[pair_batch, pair_source, None]
            weighted_messages = weighted.new_zeros(
                batch, stop - start, prepared.centers.shape[1], self.config['message'])
            weighted_messages.index_put_(selected, weighted)
            reduced = weighted_messages.sum(2)
            values.append(self.field_head(torch.cat((reduced,
                prepared.global_state[:, None].expand(-1, stop - start, -1),
                _geometry(query / prepared.lengths[:, None]), feature), -1)))
            if return_messages:
                visible = fine * prepared.present[pair_batch, pair_source, None]
                evaluated = visible.new_zeros(
                    batch, stop - start, prepared.centers.shape[1], self.config['message'])
                messages.append(evaluated.index_put_(selected, visible))
        output = torch.cat(values, 1) if values else receivers.new_empty(batch, 0, self.output_width)
        if return_messages:
            exported = (torch.cat(messages, 1) if messages else receivers.new_empty(
                batch, 0, prepared.centers.shape[1], self.config['message']))
            return output, exported
        return output

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
                return_messages=False, read_plan=None, read_policy_identity=None,
                read_policy_generation=None, receiver_generation=None,
                receiver_ids=None):
        self.assert_owned(prepared)
        batch, queries, _ = receivers.shape
        self._validate_shape(receivers, (prepared.centers.shape[0], queries, self.spatial_dim), 'receivers')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive.')
        features = receivers.new_empty(batch, queries, 0) if receiver_features is None else receiver_features
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        if read_plan is not None:
            if not isinstance(read_plan, PreparedNonlinearReadPlan):
                raise TypeError('read_plan must be a PreparedNonlinearReadPlan.')
            read_plan.assert_compatible(self, prepared, receivers, receiver_ids,
                receiver_features,
                policy_identity=read_policy_identity,
                policy_generation=read_policy_generation,
                receiver_generation=receiver_generation)
            return self._predict_with_read_plan(prepared, receivers, features, read_plan,
                chunk_size=chunk_size, return_messages=return_messages)
        if any(value is not None for value in (
                read_policy_identity, read_policy_generation, receiver_generation,
                receiver_ids)):
            raise ValueError('Read-plan identities require a prepared nonlinear read plan.')
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
