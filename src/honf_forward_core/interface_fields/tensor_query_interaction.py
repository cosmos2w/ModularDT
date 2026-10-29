"""Frozen-host QM/QE source/receiver calibration and explicit joint contrast.

H-add and H-joint register identical parameters. Their only executed difference
is inclusion of I before the residual tanh. Physical sources remain fine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .tensor_source_group_residual import (
    TensorPhaseActions,
    TensorResidualAccess,
    TensorSourceGroupResidual,
    TensorSourceGroupResidualField,
    TensorSourcePlan,
)
from .typed_hypergraph_field import TypedHypergraphField
from .typed_hypergraph_state import SOURCE_TYPE, pad_geometry

QUERY_ROUTES = ('QM', 'QE')


def stable_tanh_contrast(base_tanh, residual):
    """Same outer tanh difference, exact zero with a live residual derivative.

    Here residual is the inner tanh correction, hence bounded by one. The
    denominator is at least 1-tanh(1); compact base tanh is computed using the
    parent reader's shape rather than a second wide-pair tanh kernel.
    """
    residual_tanh = torch.tanh(residual)
    return residual_tanh * (1 - base_tanh.square()) / (1 + base_tanh * residual_tanh)


@dataclass(frozen=True)
class QueryInteractionPlan(TensorSourcePlan):
    mean_access: torch.Tensor
    mean_density: dict[str, torch.Tensor]
    reference_ids: torch.Tensor
    reference_roles: torch.Tensor
    reference_origin: str


@dataclass(frozen=True)
class QueryInteractionPhase(TensorPhaseActions):
    source_only: dict[str, torch.Tensor]
    source_moments: dict[str, torch.Tensor]
    receiver_moments: dict[str, torch.Tensor]


@dataclass(frozen=True)
class QueryInteractionAccess(TensorResidualAccess):
    query_components: dict[str, torch.Tensor] | None = None
    preactivation: torch.Tensor | None = None
    base_gain_tanh: torch.Tensor | None = None
    executed_gain_contrast: torch.Tensor | None = None


class TensorQueryInteraction(TensorSourceGroupResidual):
    """One wrapper-owned source plan and cached positive-panel access mean."""

    def __init__(self, hidden_dim, num_heads, *, mode='joint', distance_alpha=.25,
                 distance_length=.25, background_zero_bias=True, geometry_length=.25,
                 admission_mode='curriculum'):
        if mode not in ('add', 'joint'):
            raise ValueError('query_interaction_mode must be add or joint')
        if admission_mode not in ('curriculum', 'soft'):
            raise ValueError('query_admission_mode must be curriculum or soft')
        if not math.isfinite(distance_alpha) or distance_alpha < 0:
            raise ValueError('query distance alpha must be finite and nonnegative')
        if not math.isfinite(distance_length) or distance_length <= 0:
            raise ValueError('query distance length must be positive and finite')
        super().__init__(hidden_dim, num_heads, parent_epoch=0, geometry_length=geometry_length)
        # No unused physical-transport correction heads are registered.
        self.gamma = nn.ModuleDict({tau: self.gamma[tau] for tau in QUERY_ROUTES})
        self.relative_score = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.relative_score[-1].weight)
        nn.init.zeros_(self.relative_score[-1].bias)
        self.mode = mode
        self.admission_mode = admission_mode
        self.distance_alpha, self.distance_length = float(distance_alpha), float(distance_length)
        self.background_zero_bias = bool(background_zero_bias)

    def admission_sparse_fraction(self):
        return 0. if self.admission_mode == 'soft' else super().admission_sparse_fraction()

    def access(self, plan, receivers, mechanism, *, intervention=None):
        if mechanism not in QUERY_ROUTES:
            raise ValueError('Query-interaction access is QM/QE only')
        role = receivers.new_zeros((*receivers.shape[:2], 3)); role[..., 2] = 1
        key = self.receiver_key(torch.cat((pad_geometry(receivers / plan.coordinate_scale), role,
            plan.global_state[:, None].expand(-1, receivers.shape[1], -1)), -1))
        delta = (receivers[:, :, None] - plan.centres[:, None]) / plan.coordinate_scale[:, None]
        score = 4 * torch.tanh(torch.bmm(key, plan.keys.transpose(1, 2)) / math.sqrt(16))
        score = score + 4 * torch.tanh(self.relative_score(pad_geometry(delta)).squeeze(-1))
        prior = -self.distance_alpha * torch.log1p(delta.square().sum(-1) / self.distance_length ** 2)
        if self.background_zero_bias:
            prior = torch.cat((prior[..., :-1], torch.zeros_like(prior[..., -1:])), -1)
        score = score + prior
        active = plan.valid & (plan.admission > 0)
        masked = score.masked_fill(~active[:, None], torch.finfo(score.dtype).min)
        exponential = torch.exp(masked - masked.amax(-1, keepdim=True)) * active[:, None]
        value = exponential * plan.admission[:, None]
        return value / value.sum(-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)

    def build_plan(self, encoded):
        # Reuse source planning, bypassing the historical role-centroid panel.
        # Its temporary fallback is replaced before any query access is read.
        base = super().build_plan(replace(encoded, receiver_anchor_coords=None,
            receiver_anchor_weights=None, receiver_anchor_roles=None))
        coords, weights, roles = (encoded.receiver_anchor_coords, encoded.receiver_anchor_weights,
                                  encoded.receiver_anchor_roles)
        if coords is None:
            coords, weights = encoded.env_coords, encoded.env_weights
            roles = torch.full(coords.shape[:2], -1, dtype=torch.long, device=coords.device)
            origin = 'complete positive-measure environmental catalogue; no adapter receiver panel'
        else:
            origin = 'complete adapter receiver-anchor catalogue, original row ordering'
            if weights is None or roles is None:
                raise ValueError('Explicit receiver panel requires aligned weights and roles')
        if coords.ndim == 2: coords = coords[None].expand(base.centres.shape[0], -1, -1)
        if weights.ndim == 1: weights = weights[None].expand(coords.shape[:2])
        if roles.ndim == 1: roles = roles[None].expand(coords.shape[:2])
        if weights.shape != coords.shape[:2] or roles.shape != coords.shape[:2]:
            raise ValueError('Query reference coordinates/weights/roles must align')
        if (not bool(torch.isfinite(coords).all()) or not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any()) or bool((weights.sum(-1) <= 0).any())):
            raise ValueError('Query interaction requires a finite positive receiver reference measure')
        nu = self._normalize(weights)
        base = replace(base, reference_coords={**base.reference_coords, 'Q': coords},
                       reference_weights={**base.reference_weights, 'Q': nu})
        # QM/QE share the query-role access; compute this graph exactly once.
        mean_access = (self.access(base, coords, 'QM') * nu[..., None]).sum(1)
        mean_density = {kind: (density * base.measures[kind][:, None]).sum(-1)
                        for kind, density in base.density.items()}
        fields = {name: getattr(base, name) for name in TensorSourcePlan.__dataclass_fields__}
        ids = torch.arange(coords.shape[1], device=coords.device)[None].expand(coords.shape[:2])
        return QueryInteractionPlan(**fields, mean_access=mean_access, mean_density=mean_density,
            reference_ids=ids, reference_roles=roles, reference_origin=origin)

    def phase_actions(self, plan, encoded, module_states, base):
        pooled = []
        for kind, states in (('M', module_states), ('E', encoded.env_tokens)):
            mass = plan.density[kind] * plan.measures[kind][:, None]
            pooled.append(torch.bmm(mass, self.content_encoder[kind](states)))
        phase = F.one_hot(torch.tensor(base.phase, device=module_states.device), 3).to(module_states.dtype)
        content = self.collective(torch.cat((*pooled, plan.descriptors,
            encoded.global_token[:, None].expand(-1, plan.centres.shape[1], -1),
            phase.expand(*plan.valid.shape, -1)), -1))
        gamma = {tau: self.gamma[tau](content) for tau in QUERY_ROUTES}
        constant, source_only, source_moments, receiver_moments = {}, {}, {}, {}
        for tau, g in gamma.items():
            kind = SOURCE_TYPE[tau]
            centered_b = plan.density[kind] - plan.mean_density[kind][..., None]
            moments = centered_b[..., None] * g[:, :, None]
            source_moments[tau] = moments.flatten(2)
            source_only[tau] = torch.bmm(plan.mean_access[:, None], source_moments[tau]).reshape(
                g.shape[0], centered_b.shape[-1], g.shape[-1])
            receiver_moments[tau] = plan.mean_density[kind][..., None] * g
            constant[tau] = torch.bmm(plan.mean_access[:, None], receiver_moments[tau]).squeeze(1)
        return QueryInteractionPhase(plan, content, gamma, constant, {}, source_only,
                                      source_moments, receiver_moments)

    def components(self, phase, receivers, tau, *, diagnostics=True, query_access=None):
        """Compute executed terms; materialize unused I only for diagnostics.

        The public component inspection remains complete by default. Native
        reads explicitly omit diagnostic work, and pass one call-local access
        tensor shared by QM/QE for the identical receiver object.
        """
        plan, g = phase.plan, phase.gamma[tau]
        access = self.access(plan, receivers, tau) if query_access is None else query_access
        centered_a = access - plan.mean_access[:, None]
        receiver = torch.bmm(centered_a, phase.receiver_moments[tau])[:, :, None]
        source = phase.source_only[tau][:, None]
        shape = (g.shape[0], receivers.shape[1], plan.density[SOURCE_TYPE[tau]].shape[-1], g.shape[-1])
        joint = (torch.bmm(centered_a, phase.source_moments[tau]).reshape(shape)
                 if diagnostics or self.mode == 'joint' else None)
        used_joint = joint if self.mode == 'joint' else None
        if self.intervention == 'zero_joint': used_joint = None
        elif self.intervention == 'remove_joint_group' and self.mode == 'joint':
            selected = self.removed_group
            if selected is None: selected = plan.admission.argmax(-1)
            keep = ~F.one_hot(selected, plan.admission.shape[-1]).bool()
            used_joint = torch.bmm(centered_a * keep[:, None], phase.source_moments[tau]).reshape(shape)
        elif self.intervention not in ('normal', 'zero_corrections', 'remove_joint_group'):
            raise ValueError('Unknown query-interaction intervention')
        preactivation = source + receiver
        if used_joint is not None: preactivation = preactivation + used_joint
        if self.intervention == 'zero_corrections': preactivation = preactivation * 0
        # Small broadcast zero represents the absent executed interaction.
        # In ordinary Add reads no query-by-source joint tensor is allocated.
        if used_joint is None:
            used_joint = joint * 0 if diagnostics else torch.zeros_like(receiver)
        return {'C': phase.centering[tau][:, None, None], 'S': source, 'R': receiver,
                'I': joint, 'I_used': used_joint, 'access': access}, preactivation


class TensorQueryInteractionField(TensorSourceGroupResidualField):
    """G-fast physical transport with a frozen-host, query-only new interface."""

    def __init__(self, *args, options=None, **kwargs):
        settings = dict(options or {})
        settings.pop('tensor_query_interaction', None)
        mode = settings.pop('query_interaction_mode', 'joint')
        alpha = settings.pop('query_distance_alpha', .25)
        length = settings.pop('query_distance_length', .25)
        background = settings.pop('query_background_zero_bias', True)
        admission_mode = settings.pop('query_admission_mode', 'curriculum')
        self.frozen_backbone_epoch = int(settings.pop('query_interface_parent_epoch', 1000))
        geometry = settings.pop('residual_geometry_length', .25)
        TypedHypergraphField.__init__(self, *args, options=settings, **kwargs)
        if not self.live_task_controls or not self.global_fast_reader:
            raise ValueError('Query interaction requires native G-fast global control')
        self.tensor_query_interaction = TensorQueryInteraction(args[0], args[2], mode=mode,
            distance_alpha=alpha, distance_length=length, background_zero_bias=background,
            geometry_length=geometry, admission_mode=admission_mode)
        self.fit_epoch, self.fit_total_epochs = 0, None
        self.set_training_progress(epoch=0)
        self.freeze_backbone()

    @property
    def tensor_residual(self):
        # Compatibility for existing actual-plan evidence hooks, without a
        # second registered module or historical Tensor-H parameter keys.
        return self.tensor_query_interaction

    def freeze_backbone(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith('tensor_query_interaction.')
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def set_training_progress(self, *, epoch, total_epochs=None):
        self.fit_epoch = int(epoch)
        if total_epochs is not None: self.fit_total_epochs = int(total_epochs)
        TypedHypergraphField.set_training_progress(self, epoch=self.frozen_backbone_epoch,
                                                  total_epochs=self.frozen_backbone_epoch)
        self.tensor_query_interaction.set_epoch(epoch)

    def selection_state(self):
        # Checkpoint restoration schedules the new interface from its own fit
        # age, while inherited global-control progress remains frozen at e1000.
        return {'epoch': self.fit_epoch, 'total_epochs': self.fit_total_epochs}

    def _query_access_pair(self, plan, receivers, actions, *, mode, soft, diagnostics):
        phase = plan.strategy_data['tensor_phase_actions']
        query_access = self.tensor_query_interaction.access(phase.plan, receivers, 'QM')
        return tuple(self._numerical_access(plan, receivers, tau, actions,
            mode=mode, soft=soft, diagnostics=diagnostics, query_access=query_access)
            for tau in QUERY_ROUTES)

    def _numerical_access(self, plan, receivers, mechanism, actions, receiver_tokens=None,
                          pair_valid=None, *, mode=None, soft=None, diagnostics=False,
                          query_access=None):
        base = TypedHypergraphField._numerical_access(self, plan, receivers, mechanism, actions,
            receiver_tokens, pair_valid, mode=mode, soft=soft, diagnostics=diagnostics)
        if mechanism not in QUERY_ROUTES: return base
        phase = plan.strategy_data['tensor_phase_actions']
        components, preactivation = self.tensor_query_interaction.components(phase, receivers, mechanism,
            diagnostics=diagnostics, query_access=query_access)
        preactivation = preactivation * base.support[..., None]
        residual = torch.tanh(preactivation)
        base_projected = base.projected
        if mechanism == 'QE':
            heads = self.env_attention.num_heads
            base_projected = torch.cat((torch.zeros_like(base_projected[..., :heads]),
                                        base_projected[..., heads:]), -1)
        if (mode or self.plan_intervention) == 'control_identity':
            residual = residual * 0; base_projected = base_projected * 0
        if mechanism == 'QM':
            base_gain = torch.tanh(base_projected[:, 0, 0][:, None])[:, :, None]
            gain_contrast = stable_tanh_contrast(base_gain, residual)
        else:
            heads = self.env_attention.num_heads
            base_gain = torch.tanh(base_projected[..., heads:].permute(0, 3, 1, 2)[..., :1])
            gain_contrast = stable_tanh_contrast(base_gain, residual[..., heads:].permute(0, 3, 1, 2))
            base_gain, gain_contrast = base_gain.permute(0, 2, 3, 1), gain_contrast.permute(0, 2, 3, 1)
        return QueryInteractionAccess(base.density, base.weight, base_projected + residual,
            base.support, components['access'], diagnostics=base.diagnostics,
            base_projected=base_projected, residual_projected=residual,
            query_components=components, preactivation=preactivation,
            base_gain_tanh=base_gain, executed_gain_contrast=gain_contrast)

    def _reduce_messages(self, messages, access, mechanism, *, soft_access=None,
                         source_measure=None, mode=None, soft=None):
        if mechanism not in QUERY_ROUTES:
            return TypedHypergraphField._reduce_messages(self, messages, access, mechanism,
                soft_access=soft_access, source_measure=source_measure, mode=mode, soft=soft)
        if soft_access is not None:
            raise ValueError('Query interaction does not use shadow reductions')
        weighted = messages * access.weight[..., None]
        if source_measure is not None: weighted = weighted * source_measure[..., None]
        return (weighted.sum(2) * (1 + access.base_gain_tanh.squeeze(2)) +
                (weighted * access.executed_gain_contrast).sum(2))

    def _tensor_attention_context(self, attention, values, gain, access):
        base = access.base_gain_tanh.permute(0, 3, 1, 2)
        contrast = access.executed_gain_contrast.permute(0, 3, 1, 2)
        return torch.matmul(attention, values) * (1 + base) + torch.matmul(attention * contrast, values)

    def export_typed_state(self, state):
        exported = TypedHypergraphField.export_typed_state(self, state)
        base = state['hypergraph_plan']; phase = base.strategy_data['tensor_phase_actions']; plan = phase.plan
        controls = {tau: plan.density if tau in QUERY_ROUTES else base.control_memberships[tau]
                    for tau in base.memberships}
        memberships = {tau: plan.density[SOURCE_TYPE[tau]] if tau in QUERY_ROUTES else base.memberships[tau]
                       for tau in base.memberships}
        admission = {tau: plan.admission if tau in QUERY_ROUTES else base.admission for tau in base.memberships}
        centres = {tau: plan.centres if tau in QUERY_ROUTES else base.centres for tau in base.memberships}
        presence = {tau: {kind: (density * plan.measures[kind][:, None]).sum(-1) > 0
                         for kind, density in plan.density.items()} if tau in QUERY_ROUTES
                    else base.control_presence[tau] for tau in base.memberships}
        return {**exported, 'base_route_controls': base.controls,
            'tensor_residual': {'plan': plan, 'phase': phase},
            'query_interaction': {'mode': self.tensor_query_interaction.mode,
                'admission_mode': self.tensor_query_interaction.admission_mode,
                'components': lambda receivers, tau: self.tensor_query_interaction.components(phase, receivers, tau),
                'mean_access': plan.mean_access, 'mean_density': plan.mean_density,
                'reference_ids': plan.reference_ids, 'reference_roles': plan.reference_roles,
                'reference_origin': plan.reference_origin,
                'reference_id_semantics': 'original adapter receiver row index (adapter supplies no separate IDs)',
                'distance_alpha': self.tensor_query_interaction.distance_alpha,
                'distance_length': self.tensor_query_interaction.distance_length,
                'background_zero_bias': self.tensor_query_interaction.background_zero_bias,
                'gain_arithmetic': 'compact parent tanh plus stable tanh-addition contrast; exact zero attachment with live residual derivative'},
            'group_admission': plan.admission, 'group_centres': plan.centres,
            'typed_admission': admission, 'typed_centres': centres,
            'source_membership': memberships, 'control_membership': controls, 'control_presence': presence,
            'control_donor_measures': plan.measures, 'group_controls': phase.content,
            'residual_gamma': phase.gamma, 'global_centering': phase.centering,
            'reference_receiver_coords': plan.reference_coords, 'reference_receiver_weights': plan.reference_weights,
            'additional_age': plan.additional_age, 'sparse_fraction': plan.sparse_fraction,
            'value_membership': base.memberships,
            'receiver_access': lambda receivers, tau, receiver_tokens=None: self._numerical_access(
                base, receivers, tau, self._prepare_actions(base), receiver_tokens, diagnostics=True),
            'dependency_provenance': {'mode': 'tensor_query_interaction_v1',
                'planning': 'initial input-only plan and full positive-panel access mean once per wrapper',
                'direct_content': 'fixed-plan M/E density times physical measure times nonlinear source content',
                'contrast': 'C/S/R/I over query/source product measure before residual tanh; signed reference compensation is not donor membership',
                'unchanged_routes': 'MM/ME/EM use original G-fast actions and native fine reads given current states',
                'global_paths': 'planning, frozen global calibration, reference means, coarse/local context',
                'upstream_ancestry': 'query corrections can alter later phase states through ports and Stage-A',
                'physical_values': 'all original fine sources and native eligibility retained',
                'interpretation': 'separable preactivation calibration versus explicit joint contrast; no causal claim'}}


def freeze_query_interaction_backbone(model):
    """Freeze the complete host without disabling input/task autograd.

    The wrapper, adapters, Stage-A and physical core remain differentiable
    functions of their inputs. Only the explicitly registered new interface
    parameters receive optimizer updates.
    """
    if not isinstance(model.core.backend, TensorQueryInteractionField):
        raise TypeError('Full-host freeze requires the query-interaction backend')
    prefix = 'core.backend.tensor_query_interaction.'
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(prefix)
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
