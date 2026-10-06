"""Input-only source groups for residual controls; every native value stays fine.

Memberships are owned by one wrapper call, shared across its physical phases,
never cached on this module. The directed geometric donor prior uses length
0.25 in native coordinates divided by the adapter coordinate scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .routing_index.sparse_projection import masked_sparsemax, source_measure_sparsemax
from .typed_hypergraph_field import TypedHypergraphField
from .typed_hypergraph_state import MECHANISMS, SOURCE_TYPE, ProjectedSourceAccess, masked_mean, pad_geometry


@dataclass(frozen=True)
class TensorSourcePlan:
    centres: torch.Tensor
    valid: torch.Tensor
    descriptors: torch.Tensor
    keys: torch.Tensor
    admission: torch.Tensor
    density: dict[str, torch.Tensor]
    measures: dict[str, torch.Tensor]
    reference_coords: dict[str, torch.Tensor]
    reference_weights: dict[str, torch.Tensor]
    global_state: torch.Tensor
    coordinate_scale: torch.Tensor
    additional_age: int
    sparse_fraction: float


@dataclass(frozen=True)
class TensorPhaseActions:
    plan: TensorSourcePlan
    content: torch.Tensor
    gamma: dict[str, torch.Tensor]
    centering: dict[str, torch.Tensor]
    base: dict[str, torch.Tensor]


@dataclass(frozen=True)
class TensorResidualAccess(ProjectedSourceAccess):
    base_projected: torch.Tensor | None = None
    residual_projected: torch.Tensor | None = None


class TensorSourceGroupResidual(nn.Module):
    """Sparse control-information groups, not sparse physical-value retrieval."""
    def __init__(self, hidden_dim, num_heads, *, parent_epoch=500, width=64, control_dim=16,
                 geometry_length=0.25):
        super().__init__()
        self.parent_epoch, self.additional_age = int(parent_epoch), 0
        self.geometry_length = float(geometry_length)
        self.group_descriptor = nn.Sequential(nn.Linear(2 * hidden_dim + 7, width), nn.GELU())
        self.admission_head = nn.Linear(width, 1)
        self.group_key = nn.Linear(width, 16)
        self.source_descriptor = nn.ModuleDict({t: nn.Sequential(nn.Linear(hidden_dim, 16), nn.GELU())
                                               for t in ('M', 'E')})
        self.donor_scorer = nn.ModuleDict({t: nn.Sequential(nn.Linear(width + 16 + 6, 32), nn.GELU(),
                                                          nn.Linear(32, 1)) for t in ('M', 'E')})
        self.content_encoder = nn.ModuleDict({t: nn.Sequential(nn.Linear(hidden_dim, width), nn.GELU())
                                             for t in ('M', 'E')})
        self.collective = nn.Sequential(nn.Linear(3 * width + hidden_dim + 3, width), nn.GELU(),
                                        nn.Linear(width, control_dim), nn.GELU())
        self.receiver_key = nn.Sequential(nn.Linear(hidden_dim + 9, 32), nn.GELU(), nn.Linear(32, 16))
        self.gamma = nn.ModuleDict({t: nn.Linear(control_dim, 2 * num_heads if t == 'QE' else 1)
                                    for t in MECHANISMS})
        for head in self.gamma.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.intervention = 'normal'
        self.removed_group = None

    def set_epoch(self, absolute_epoch):
        self.additional_age = max(0, int(absolute_epoch) - self.parent_epoch)

    def admission_sparse_fraction(self):
        """Historical curriculum, independently overridable by an explicit mode."""
        return min(1., max(0., (self.additional_age - 100) / 100.))

    @staticmethod
    def _normalize(measure):
        return measure / measure.sum(-1, keepdim=True).clamp_min(torch.finfo(measure.dtype).tiny)

    @staticmethod
    def _masked_softmax(logit, valid):
        value = torch.softmax(logit.masked_fill(~valid, torch.finfo(logit.dtype).min), -1) * valid
        return value / value.sum(-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)

    def build_plan(self, encoded):
        m_valid, e_valid = encoded.module_present > .5, encoded.env_weights > 0
        measures = {'M': self._normalize(encoded.module_present * m_valid),
                    'E': self._normalize(encoded.env_weights * e_valid)}
        centre_e = masked_mean(encoded.env_coords, e_valid, encoded.env_weights)
        token_e = masked_mean(encoded.env_tokens, e_valid, encoded.env_weights)
        centres = torch.cat((encoded.module_centers, centre_e[:, None]), 1)
        valid = torch.cat((m_valid, torch.ones((m_valid.shape[0], 1), device=m_valid.device, dtype=torch.bool)), 1)
        initial_tokens = torch.cat((encoded.module_tokens, token_e[:, None]), 1)
        background = centres.new_zeros((*valid.shape, 1)); background[:, -1] = 1
        normalized = centres / encoded.coordinate_scale
        descriptors = self.group_descriptor(torch.cat((initial_tokens,
            encoded.global_token[:, None].expand(-1, centres.shape[1], -1),
            pad_geometry(normalized), background), -1))
        logits = self.admission_head(descriptors).squeeze(-1)
        fraction = self.admission_sparse_fraction()
        soft_admission = self._masked_softmax(logits, valid)
        admission = (soft_admission if fraction == 0 else
                     (1 - fraction) * soft_admission + fraction * masked_sparsemax(logits, valid))
        density = {}
        for kind, coords, tokens, donor_valid in (
                ('M', encoded.module_centers, encoded.module_tokens, m_valid),
                ('E', encoded.env_coords, encoded.env_tokens, e_valid)):
            if coords.shape[1] == 0:
                density[kind] = centres.new_zeros((*valid.shape, 0))
                continue
            delta = (centres[:, :, None] - coords[:, None]) / encoded.coordinate_scale[:, None]
            donor = self.source_descriptor[kind](tokens)
            scores = self.donor_scorer[kind](torch.cat((
                descriptors[:, :, None].expand(-1, -1, coords.shape[1], -1),
                donor[:, None].expand(-1, centres.shape[1], -1, -1), pad_geometry(delta)), -1)).squeeze(-1)
            scores = scores - delta.square().sum(-1) / self.geometry_length ** 2 + .1 * delta[..., 0] / self.geometry_length
            mask = donor_valid[:, None].expand_as(scores)
            # Count-softmax is converted to a measure-softmax DENSITY, so atom
            # splitting leaves density and quadrature integrals unchanged.
            safe_score = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            exponential = torch.exp(safe_score - safe_score.amax(-1, keepdim=True)) * mask
            denominator = (exponential * measures[kind][:, None]).sum(-1, keepdim=True)
            soft = exponential / denominator.clamp_min(torch.finfo(scores.dtype).tiny)
            if fraction == 0:
                density[kind] = soft * valid[..., None]
            else:
                sparse = source_measure_sparsemax(scores, measures[kind], mask).density.to(scores.dtype)
                density[kind] = ((1 - fraction) * soft + fraction * sparse) * valid[..., None]
        # A small role-aware reference quadrature is independent of requested
        # queries. Geometry-role centroids use masses, so splitting is invariant.
        coords = encoded.receiver_anchor_coords
        weights = encoded.receiver_anchor_weights
        roles = encoded.receiver_anchor_roles
        if coords is None:
            q_coords, q_weights = centre_e[:, None], centres.new_ones((centres.shape[0], 1))
        else:
            if coords.ndim == 2: coords = coords[None].expand(centres.shape[0], -1, -1)
            if weights is None: weights = coords.new_ones(coords.shape[:2])
            elif weights.ndim == 1: weights = weights[None].expand(coords.shape[:2])
            if roles is None: roles = torch.zeros_like(weights, dtype=torch.long)
            elif roles.ndim == 1: roles = roles[None].expand(weights.shape)
            role_mass = weights[:, None] * F.one_hot(roles, 8).transpose(1, 2).to(weights.dtype)
            mass = role_mass.sum(-1)
            q_coords = torch.bmm(role_mass, coords) / mass.clamp_min(torch.finfo(mass.dtype).tiny)[..., None]
            q_weights = self._normalize(mass)
        return TensorSourcePlan(centres, valid, descriptors, self.group_key(descriptors), admission, density,
            measures, {'M': encoded.module_centers, 'E': encoded.env_coords, 'Q': q_coords},
            {'M': measures['M'], 'E': measures['E'], 'Q': q_weights}, encoded.global_token,
            encoded.coordinate_scale, self.additional_age, fraction)

    def access(self, plan, receivers, mechanism, *, intervention=None):
        role = 0 if mechanism in ('MM', 'ME') else 1 if mechanism == 'EM' else 2
        onehot = F.one_hot(torch.full(receivers.shape[:2], role, device=receivers.device), 3).to(receivers.dtype)
        key = self.receiver_key(torch.cat((pad_geometry(receivers / plan.coordinate_scale), onehot,
            plan.global_state[:, None].expand(-1, receivers.shape[1], -1)), -1))
        score = 4 * torch.tanh(torch.bmm(key, plan.keys.transpose(1, 2)) / math.sqrt(16))
        value = torch.exp(score) * plan.admission[:, None]
        access = value / value.sum(-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)
        if (intervention or self.intervention) == 'mean_access':
            kind = 'M' if role == 0 else 'E' if role == 1 else 'Q'
            reference = self.access(plan, plan.reference_coords[kind], mechanism, intervention='normal')
            mean = (reference * plan.reference_weights[kind][..., None]).sum(1, keepdim=True)
            access = mean.expand_as(access)
        return access

    def raw_action(self, plan, gamma, receivers, mechanism):
        access = self.access(plan, receivers, mechanism)
        kind = SOURCE_TYPE[mechanism]
        # The intermediate is only [B,R,G,C], never a wide physical control.
        action = torch.einsum('brg,bgc->brgc', access, gamma)
        return torch.einsum('brgc,bgs->brsc', action, plan.density[kind])

    def phase_actions(self, plan, encoded, module_states, base):
        pooled = []
        for kind, states in (('M', module_states), ('E', encoded.env_tokens)):
            projected = self.content_encoder[kind](states)
            mass = plan.density[kind] * plan.measures[kind][:, None]
            pooled.append(torch.bmm(mass, projected))
        phase = F.one_hot(torch.tensor(base.phase, device=module_states.device), 3).to(module_states.dtype)
        content = self.collective(torch.cat((*pooled, plan.descriptors,
            encoded.global_token[:, None].expand(-1, plan.centres.shape[1], -1),
            phase.expand(*plan.valid.shape, -1)), -1))
        gamma, centering = {}, {}
        for tau in MECHANISMS:
            gamma[tau] = self.gamma[tau](content)
            if self.intervention == 'zero_residual': gamma[tau] = gamma[tau] * 0
            if self.intervention == 'remove_group':
                # Explicit input-selected group indices are chosen before the
                # intervention and never alter physical value inventories.
                selected = self.removed_group
                if selected is None: selected = plan.admission.argmax(-1)
                keep = ~F.one_hot(selected, plan.admission.shape[-1]).bool()
                gamma[tau] = gamma[tau] * keep[..., None]
            receiver_kind = 'M' if tau in ('MM', 'ME') else 'E' if tau == 'EM' else 'Q'
            raw = self.raw_action(plan, gamma[tau], plan.reference_coords[receiver_kind], tau)
            mass = plan.reference_weights[receiver_kind][:, :, None] * plan.measures[SOURCE_TYPE[tau]][:, None]
            if tau == 'MM':
                mass = mass * (~torch.eye(mass.shape[-1], device=mass.device, dtype=torch.bool))[None]
            denominator = mass.sum((1, 2))[:, None]
            centering[tau] = (raw * mass[..., None]).sum((1, 2)) / denominator.clamp_min(torch.finfo(raw.dtype).tiny)
        return TensorPhaseActions(plan, content, gamma, centering, {})


class TensorSourceGroupResidualField(TypedHypergraphField):
    """Global-C compatible base plus ordinary-autograd source-group residual."""
    shares_tensor_plan = True

    def __init__(self, *args, options=None, **kwargs):
        settings = dict(options or {})
        settings.pop('tensor_source_residual', None)
        parent = settings.pop('residual_parent_epoch', 500)
        length = settings.pop('residual_geometry_length', .25)
        super().__init__(*args, options=settings, **kwargs)
        if not self.live_task_controls or not self.global_fast_reader:
            raise ValueError('Tensor residual requires native_context_global_control_honf with global_fast_reader')
        self.tensor_residual = TensorSourceGroupResidual(args[0], args[2], parent_epoch=parent,
                                                        geometry_length=length)

    def set_training_progress(self, *, epoch, total_epochs=None):
        super().set_training_progress(epoch=epoch, total_epochs=total_epochs)
        self.tensor_residual.set_epoch(epoch)

    def _prepare_organizer_plan(self, encoded, module_states, *, phase, soft, phase_shared_state=None):
        if soft: raise ValueError('Tensor-H uses one continuous operator, without soft shadow')
        if phase == 0:
            if phase_shared_state is not None: raise ValueError('P0 must build a fresh input-only plan')
            shared = self.tensor_residual.build_plan(encoded)
        else:
            if not isinstance(phase_shared_state, TensorSourcePlan):
                raise ValueError('Tensor-H P1/P2 require the P0-owned tensor source plan')
            shared = phase_shared_state
        base = self.organizer.prepare(encoded, module_states, phase=phase, soft=False)
        phase_actions = self.tensor_residual.phase_actions(shared, encoded, module_states, base)
        return replace(base, strategy_data={**base.strategy_data, 'tensor_source_plan': shared,
                                           'tensor_phase_actions': phase_actions})

    def prepare(self, *args, **kwargs):
        state = super().prepare(*args, **kwargs)
        state['phase_shared_group_control'] = state['hypergraph_plan'].strategy_data['tensor_source_plan']
        return state

    def _prepare_actions(self, plan, *, detach_projection=False):
        if detach_projection: raise ValueError('Tensor-H has no second gradient program')
        return super()._prepare_actions(plan, detach_projection=False)

    def _numerical_access(self, plan, receivers, mechanism, actions, receiver_tokens=None,
                          pair_valid=None, *, mode=None, soft=None, diagnostics=False):
        # Base compact full-access state has the native physical masks; the
        # residual changes only its small-channel control action.
        access = self.organizer.numerical_access(plan, receivers, mechanism, actions[mechanism],
                                                pair_valid=pair_valid, include_diagnostics=diagnostics)
        phase = plan.strategy_data['tensor_phase_actions']
        residual = self.tensor_residual.raw_action(phase.plan, phase.gamma[mechanism], receivers, mechanism)
        residual = torch.tanh(residual - phase.centering[mechanism][:, None, None])
        residual = residual * access.support[..., None]
        base_projected = access.projected
        if mechanism == 'QE':
            # Keep diagnostic projected channels equal to executed scores.
            # Historical score tensors remain checkpoint-compatible, but
            # their source-independent action is omitted from Tensor-H.
            heads = self.env_attention.num_heads
            base_projected = torch.cat((torch.zeros_like(base_projected[..., :heads]),
                                        base_projected[..., heads:]), -1)
        if (mode or self.plan_intervention) == 'control_identity':
            projected = torch.zeros_like(residual)
        else: projected = base_projected + residual
        # Deliberately drop the case-constant subtype marker: fine reduction
        # must apply Tensor-H gains per physical source before the native sum.
        if (mode or self.plan_intervention) == 'control_identity':
            base_projected = torch.zeros_like(base_projected)
            residual = torch.zeros_like(residual)
        return TensorResidualAccess(access.density, access.weight, projected, access.support,
            self.tensor_residual.access(phase.plan, receivers, mechanism), diagnostics=access.diagnostics,
            base_projected=base_projected, residual_projected=residual)

    def _reduce_messages(self, messages, access, mechanism, *, soft_access=None,
                         source_measure=None, mode=None, soft=None):
        if soft_access is not None:
            raise ValueError('Tensor-H does not use shadow reductions')
        weighted = messages * access.weight[..., None]
        if source_measure is not None:
            weighted = weighted * source_measure[..., None]
        base = torch.tanh(access.base_projected)
        contrast = torch.tanh(access.projected) - base
        # Same operator, ordered to preserve the specialized parent exactly at
        # zero attachment while retaining live task gradients to gamma.
        return weighted.sum(2) * (1 + base.squeeze(2)) + (weighted * contrast).sum(2)

    def _tensor_attention_context(self, attention, values, gain, access):
        _, base_gain = access.base_projected.split(self.env_attention.num_heads, -1)
        base = torch.tanh(base_gain.permute(0, 3, 1, 2))
        return (torch.matmul(attention, values) * (1 + base) +
                torch.matmul(attention * (torch.tanh(gain) - base), values))

    def set_execution_mode(self, mode, *, receiver_chunk_size=128):
        if mode != 'dense_masked_reference':
            raise ValueError('Tensor-H retains the dense native physical source executor')
        super().set_execution_mode(mode, receiver_chunk_size=receiver_chunk_size)

    def export_typed_state(self, state):
        exported = super().export_typed_state(state)
        base = state['hypergraph_plan']
        phase = base.strategy_data['tensor_phase_actions']; plan = phase.plan
        return {**exported, 'base_route_controls': base.controls,
            'tensor_residual': {'plan': plan, 'phase': phase,
                'receiver_access': lambda receivers, mechanism: self.tensor_residual.access(plan, receivers, mechanism),
                'raw_action': lambda receivers, mechanism: self.tensor_residual.raw_action(
                    plan, phase.gamma[mechanism], receivers, mechanism)}, 'group_admission': plan.admission, 'group_centres': plan.centres,
            'control_membership': {tau: plan.density for tau in MECHANISMS},
            'control_donor_measures': plan.measures, 'group_controls': phase.content,
            'residual_gamma': phase.gamma, 'global_centering': phase.centering,
            'reference_receiver_coords': plan.reference_coords, 'reference_receiver_weights': plan.reference_weights,
            'additional_age': plan.additional_age, 'sparse_fraction': plan.sparse_fraction,
            'residual_receiver_access': lambda receivers, mechanism:
                self.tensor_residual.access(plan, receivers, mechanism),
            'receiver_access': lambda receivers, mechanism, receiver_tokens=None:
                self._numerical_access(base, receivers, mechanism, self._prepare_actions(base),
                                       receiver_tokens, diagnostics=True),
            'dependency_provenance': {'mode': 'tensor_source_control_residual_v1',
                'planning': 'initial input encodings, shared descriptors/admission/M-E densities once per wrapper',
                'local_control_content': 'fixed-membership measure-weighted live projected source states',
                'global_paths': 'planning, Global-C calibration, pre-tanh reference centering, coarse/local context',
                'value_donors': 'all original native fine sources with native eligibility',
                'upstream_ancestry': 'phase-current source states retain earlier MM/ME/EM, predicted ports and Stage-A',
                'interpretation': 'control-information groups; dense physical values; no physical-causality claim'},
            'value_membership': base.memberships,
            'source_membership': {tau: plan.density[SOURCE_TYPE[tau]] for tau in MECHANISMS}}

    def preparation_aux(self, state, *, include_diagnostics=False):
        result = super().preparation_aux(state, include_diagnostics=include_diagnostics)
        shared = state['phase_shared_group_control']
        result['hypergraph_admitted_groups'] = (shared.admission > 0).sum(-1).detach()
        result['hypergraph_allocated_groups'] = shared.admission.new_full((shared.admission.shape[0],),
                                                                       shared.admission.shape[-1])
        result['hypergraph_tensor_sparse_fraction'] = shared.admission.new_tensor(shared.sparse_fraction)
        result['hypergraph_tensor_additional_age'] = shared.admission.new_tensor(shared.additional_age)
        if include_diagnostics:
            result['hypergraph_group_centres'] = shared.centres.detach()
            result['hypergraph_tensor_admission'] = shared.admission.detach()
        return result
