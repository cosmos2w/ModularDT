"""Source-preserving cheap reads with receiver-specific fine corrections.

The same low-width base and organizer serve scalar affine coefficients and
vector nonlinear messages. Dataset adapters retain units, output laws and
supervision. A hard route is piecewise fixed for input derivatives. Training's
zero-valued router surrogate is an explicitly separate computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from .interaction_core import NonlinearFieldReadout, _geometry, _mlp, _versions
from .source_response_operator import PreparedSourceResponse, SourceResponseOperator


@dataclass(frozen=True)
class RefinementPolicy:
    mode: str = 'all_fine'
    phase: str = 'hard'
    threshold: float = 0.5
    temperature: float = 1.0
    training_signal: bool = False

    def __post_init__(self):
        if self.mode not in ('all_fine', 'adaptive', 'all_base', 'nearest', 'upstream', 'shuffle'):
            raise ValueError('Unknown source refinement execution mode.')
        if self.phase not in ('open', 'soft', 'hard'):
            raise ValueError('Refinement phase must be open, soft or hard.')
        if not 0 <= self.threshold <= 1 or not math.isfinite(self.threshold):
            raise ValueError('The declared gate threshold must lie in [0,1].')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Router temperature must be positive and finite.')


class SourceRefinement(nn.Module):
    """Small factorized per-source base; no hidden-width Q-by-M fine stack."""

    def __init__(self, hidden, spatial_dim, query_width, output_width, *,
                 base_width=16, router_hidden=32, residual_scale=1.0):
        super().__init__()
        if not 1 <= base_width <= 16 or router_hidden < 1 or residual_scale <= 0:
            raise ValueError('Declare a low-width base, positive router and residual scale.')
        self.spatial_dim, self.query_width = spatial_dim, query_width
        self.output_width, self.base_width = output_width, base_width
        self.source_projection = nn.Linear(hidden, base_width)
        self.global_projection = nn.Linear(hidden, base_width, bias=False)
        self.query_projection = nn.Linear(spatial_dim * 9 + query_width, base_width)
        self.relative_projection = nn.Linear(2 * spatial_dim + 1, base_width)
        self.base_output = nn.Linear(base_width, output_width)
        self.router = _mlp(base_width + 2 * spatial_dim + 1 + query_width, 1, router_hidden)
        # Begin open; the 501--600 supervised/native router period can restore
        # or reduce any unprotected route before cost pressure is introduced.
        with torch.no_grad():
            self.router[-1].weight.zero_()
            self.router[-1].bias.fill_(2.0)
        self.register_buffer('residual_scale', torch.tensor(float(residual_scale)))

    def source_features(self, context):
        return torch.tanh(self.source_projection(context.source_states)
                          + self.global_projection(context.global_state)[:, None])

    def read(self, context, query, features, source_features, temperature, *, router_detach_inputs=False):
        query_feature = torch.tanh(self.query_projection(torch.cat(
            (_geometry(query / context.lengths[:, None]), features), -1)))
        delta = query[:, :, None] - context.centers[:, None]
        relative = torch.cat((delta / context.lengths[:, None, None],
            delta / context.source_lengths[:, None, :, None].clamp_min(1e-12),
            torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            / context.source_lengths[:, None, :, None].clamp_min(1e-12)), -1)
        pair = torch.tanh(query_feature[:, :, None] * source_features[:, None]
                          + self.relative_projection(relative))
        base = self.base_output(pair) * context.present[:, None, :, None]
        router_input = torch.cat((pair, relative,
            features[:, :, None].expand(-1, -1, context.centers.shape[1], -1)), -1)
        if router_detach_inputs:
            # The restoring surrogate trains only the organizer. It must not
            # masquerade as a fine/context or physical-input derivative.
            router_input = router_input.detach()
        probability = torch.sigmoid(self.router(router_input).squeeze(-1) / temperature)
        return base, probability

    @staticmethod
    def route(context, query, probability, protected, policy, fixed_route=None):
        active = (context.present[:, None] > 0).expand_as(probability)
        if fixed_route is not None:
            if policy.mode != 'adaptive' or policy.phase != 'hard' or policy.training_signal:
                raise ValueError('A fixed route requires adaptive hard inference without a training surrogate.')
            if fixed_route.shape != active.shape or fixed_route.dtype != torch.bool:
                raise ValueError('A fixed route must be boolean [B,Q,M] in physical source order.')
            return (fixed_route | protected) & active
        learned = ((probability >= policy.threshold) | protected) & active
        if policy.mode == 'all_fine' or policy.phase == 'open':
            return active
        if policy.mode == 'all_base':
            return protected & active
        if policy.mode == 'adaptive':
            return learned
        eligible = active & ~protected
        additional = learned.sum(-1) - protected.sum(-1)
        delta = query[:, :, None] - context.centers[:, None]
        if policy.mode == 'nearest':
            priority = delta.square().sum(-1)
        elif policy.mode == 'upstream':
            # Adapters use inlet/wind-aligned x. Prefer donors upstream of q;
            # cross-stream/vertical separation breaks distance ties.
            priority = delta.square().sum(-1) + (delta[..., 0] < 0).to(delta) * 1e6
        else:
            # One deterministic physical-slot intervention, equal degree and
            # protected locality. This is not a relabeling invariance check.
            slots = torch.arange(active.shape[-1], device=query.device, dtype=query.dtype)
            # A shifted reversal is a permutation for every source count;
            # multiplying modulo M would collapse slots when gcd(a,M)>1.
            priority = ((3 - slots) % max(active.shape[-1], 1))[None, None].expand_as(probability)
        priority = priority.masked_fill(~eligible, torch.inf)
        ranks = priority.argsort(dim=-1, stable=True).argsort(dim=-1, stable=True)
        return (protected | (eligible & (ranks < additional[..., None]))) & active

    def auxiliary(self, base, fine, probability, protected, context, keep, *, fine_rows):
        active = (context.present[:, None] > 0).expand_as(probability)
        eligible = active & ~protected
        difference = base - fine.detach()
        numerator = (difference.square() * active[..., None]).sum()
        denominator = active.sum() * self.output_width
        difficulty = difference.detach().square().mean(-1).sqrt()
        scale = self.residual_scale.to(difficulty).clamp_min(torch.finfo(difficulty.dtype).tiny)
        target = difficulty / (difficulty + scale)
        importance = ((probability - target).square() * eligible).sum()
        expected = (probability * eligible).sum() + protected.sum()
        return {
            'base_numerator': numerator, 'base_denominator': denominator,
            'router_importance_numerator': importance,
            'router_importance_denominator': eligible.sum(),
            'expected_work_numerator': expected, 'expected_work_denominator': active.sum(),
            'base_loss': numerator / denominator.clamp_min(1),
            'router_importance_loss': importance / eligible.sum().clamp_min(1),
            'expected_work': expected / active.sum().clamp_min(1),
            'base': base, 'fine': fine, 'probability': probability,
            'protected': protected, 'keep': keep,
            'cheap_rows': int(active.numel()), 'gate_rows': int(active.numel()),
            'active_pairs': int(active.sum().detach()),
            'padded_fine_capacity': int(active.numel()),
            'fine_rows': int(fine_rows), 'selected_detail_rows': int(keep.sum().detach()),
        }


def _combine(base, fine, probability, keep, protected, policy):
    if policy.mode == 'all_fine' or policy.phase == 'open':
        value = fine
    elif policy.phase == 'soft' and policy.mode == 'adaptive':
        blend = torch.where(protected, torch.ones_like(probability), probability)
        value = base + blend[..., None] * (fine - base)
    else:
        value = torch.where(keep[..., None], fine, base)
    if policy.training_signal and policy.mode != 'all_fine' and policy.phase != 'soft':
        surrogate = (probability - probability.detach()) * ~protected
        value = value + surrogate[..., None] * (fine - base).detach()
    return value


class _RefinementMixin:
    def _initialize_refinement(self, output_width, base_width, router_hidden, residual_scale):
        self.refinement = SourceRefinement(self.hidden, self.spatial_dim, self.query_width,
            output_width, base_width=base_width, router_hidden=router_hidden,
            residual_scale=residual_scale)
        self.refinement_policy = RefinementPolicy()
        self._refinement_auxiliary = []
        self.refinement_config = {'base_width': base_width, 'router_hidden': router_hidden,
                                  'residual_scale': residual_scale}

    def set_execution(self, mode='all_fine', *, phase='hard', threshold=0.5,
                      temperature=1.0, training_signal=False):
        self.refinement_policy = RefinementPolicy(mode, phase, threshold, temperature, training_signal)
        return self

    def reset_auxiliary(self):
        self._refinement_auxiliary.clear()

    def auxiliary_terms(self):
        records = self._refinement_auxiliary
        if not records:
            return {}
        result = {}
        for name in ('base_numerator', 'base_denominator', 'router_importance_numerator',
                     'router_importance_denominator', 'expected_work_numerator',
                     'expected_work_denominator'):
            result[name] = sum(record[name] for record in records)
        for prefix, alias in (('base', 'base_loss'), ('router_importance', 'router_importance_loss'),
                              ('expected_work', 'expected_work')):
            result[alias] = result[prefix + '_numerator'] / result[prefix + '_denominator'].clamp_min(1)
        for name in ('cheap_rows', 'gate_rows', 'active_pairs', 'padded_fine_capacity',
                     'fine_rows', 'selected_detail_rows', 'near_rows'):
            result[name] = sum(record.get(name, 0) for record in records)
        return result


@dataclass
class PreparedRefinementResponse(PreparedSourceResponse):
    """Retain factors so precise arithmetic widens before mixed assembly."""
    base_far: torch.Tensor | None = None
    fine_far: torch.Tensor | None = None
    probability: torch.Tensor | None = None
    keep: torch.Tensor | None = None
    protected: torch.Tensor | None = None
    policy: RefinementPolicy | None = None

    def __post_init__(self):
        self._refinement_binding = self._binding()

    def _binding(self):
        return (id(self.policy), tuple((name, id(getattr(self, name))) for name in
                ('base_far', 'fine_far', 'probability', 'keep', 'protected')))

    def assert_fresh(self):
        super().assert_fresh()
        if self._binding() != self._refinement_binding:
            raise ValueError('Prepared refinement factors or fixed route were replaced; rebuild this request.')

    def dense_kernel(self, *, accumulation_dtype=None):
        if accumulation_dtype is None or self.base_far is None:
            return super().dense_kernel(accumulation_dtype=accumulation_dtype)
        self.assert_fresh()
        if accumulation_dtype not in (torch.float32, torch.float64):
            raise ValueError('Response accumulation dtype must be float32 or float64.')
        far = _combine(self.base_far.to(accumulation_dtype), self.fine_far.to(accumulation_dtype),
            self.probability.to(accumulation_dtype), self.keep, self.protected,
            self.policy)
        # The existing near blending and division then widen their factors too.
        return PreparedSourceResponse(**{name: getattr(self, name)
            for name in PreparedSourceResponse.__dataclass_fields__ if name != 'far_kernel'},
            far_kernel=far).dense_kernel(accumulation_dtype=accumulation_dtype)


class RefinedSourceResponseOperator(_RefinementMixin, SourceResponseOperator):
    """Opt-in Thermal coefficient refinement; current forcing never enters gates."""

    def __init__(self, *args, base_width=16, router_hidden=32, residual_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode != 'direct':
            raise ValueError('Physical-source refinement requires the retained direct readout.')
        self._initialize_refinement(self.output_width, base_width, router_hidden, residual_scale)

    @classmethod
    def from_fine(cls, fine, **kwargs):
        if type(fine) is not SourceResponseOperator or fine.mode != 'direct':
            raise ValueError('Initialize refinement from an unchanged direct fine operator.')
        result = cls(**fine.config, **kwargs).to(next(fine.parameters()))
        incompatible = result.load_state_dict(fine.state_dict(), strict=False)
        if incompatible.unexpected_keys or any(not key.startswith('refinement.')
                                               for key in incompatible.missing_keys):
            raise ValueError('Fine checkpoint keys do not match the opt-in refinement host.')
        result.train(fine.training)
        return result

    def prepare_receivers(self, context, receivers, receiver_features=None, receiver_ids=None,
                          chunk_size=512, *, fixed_route=None):
        policy = self.refinement_policy
        if (policy.mode == 'all_fine' or (policy.mode == 'adaptive' and policy.threshold == 0)) and not policy.training_signal and fixed_route is None:
            return super().prepare_receivers(context, receivers, receiver_features,
                                            receiver_ids, chunk_size)
        self.assert_owned(context)
        batch, queries, _ = receivers.shape
        self._validate_shape(receivers, (context.centers.shape[0], queries, self.spatial_dim), 'receivers')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive.')
        features = receivers.new_empty(batch, queries, 0) if receiver_features is None else receiver_features
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        ids = torch.arange(queries, device=receivers.device)[None].expand(batch, -1) if receiver_ids is None else receiver_ids
        self._validate_shape(ids, (batch, queries), 'receiver_ids')
        if fixed_route is not None and (fixed_route.dtype != torch.bool or fixed_route.shape != (batch, queries, context.centers.shape[1])):
            raise ValueError('Fixed route must be boolean [B,Q,M].')
        # Training pays the original full fine read exactly once, retaining its
        # operation order for replay and all-open compatibility.
        needs_full = policy.training_signal or (policy.phase == 'soft' and policy.mode == 'adaptive')
        full = super().prepare_receivers(context, receivers, features, ids, chunk_size) if needs_full else None
        source_features = self.refinement.source_features(context)
        fars, bases, fines, probabilities, keeps, protecteds = [], [], [], [], [], []
        weights, near_indices, near_values, records = [], [], [], []
        for start in range(0, queries, chunk_size):
            query = receivers[:, start:start + chunk_size]
            feature = features[:, start:start + chunk_size]
            weight = self.near_weight(query, context.centers, context.source_lengths, context.present)
            protected = weight > 0
            base, probability = self.refinement.read(context, query, feature, source_features, policy.temperature,
                router_detach_inputs=policy.training_signal)
            route = None if fixed_route is None else fixed_route[:, start:start + chunk_size]
            keep = self.refinement.route(context, query, probability, protected, policy, route)
            if full is not None:
                fine = full.far_kernel[:, start:start + chunk_size]
                fine_rows = fine.shape[0] * fine.shape[1] * fine.shape[2]
                near_selected = protected.nonzero(as_tuple=True)
                near = query.new_empty(0, self.output_width)
            else:
                selected = keep.nonzero(as_tuple=True)
                fine = base.clone()
                fine = fine.index_put(selected, self.far_head(self._read_selected_features(context, query, feature, selected)))
                fine_rows = selected[0].numel()
                near_selected = protected.nonzero(as_tuple=True)
                near = self.near_head(self._read_selected_features(context, query, feature, near_selected))
            far = _combine(base, fine, probability, keep, protected, policy)
            weights.append(weight)
            near_indices.append(torch.stack((near_selected[0], near_selected[1] + start, near_selected[2])))
            near_values.append(near)
            fars.append(far); bases.append(base); fines.append(fine)
            probabilities.append(probability); keeps.append(keep); protecteds.append(protected)
            if policy.training_signal:
                record = self.refinement.auxiliary(base, fine, probability, protected, context, keep, fine_rows=fine_rows)
                record['near_rows'] = near_selected[0].numel()
                records.append(record)
        if queries:
            far = torch.cat(fars, 1); base = torch.cat(bases, 1); fine = torch.cat(fines, 1)
            probability = torch.cat(probabilities, 1); keep = torch.cat(keeps, 1); protected = torch.cat(protecteds, 1)
            weight = torch.cat(weights, 1); near_index = torch.cat(near_indices, 1); near = torch.cat(near_values, 0)
        else:
            far = base = fine = receivers.new_empty(batch, 0, context.centers.shape[1], self.output_width)
            weight = probability = receivers.new_empty(batch, 0, context.centers.shape[1])
            keep = protected = torch.empty_like(weight, dtype=torch.bool)
            near_index = torch.empty(3, 0, device=receivers.device, dtype=torch.long)
            near = receivers.new_empty(0, self.output_width)
        offset = receivers.new_zeros(batch, queries, self.output_width)
        if not self.zero_offset:
            offset = self.offset_head(torch.cat((context.global_state[:, None].expand(-1, queries, -1),
                _geometry(receivers / context.lengths[:, None]), features), -1))
        if full is not None:
            # Native replay and main reads share the already paid original
            # near evaluation and offset, with unchanged operation order.
            weight, near_index, near, offset = (full.near_weight, full.near_indices,
                                              full.near_values, full.offset)
        response = PreparedRefinementResponse(context, receivers, ids, offset, weight, near_index, near,
            far, None, None, None, None, self.forcing_scale, self.mode,
            _versions((receivers, features, ids, fixed_route, base, fine, probability, keep, protected)),
            base, fine, probability, keep, protected, policy)
        response.full_response = full
        response.refinement_aux = {'base': base, 'fine': fine, 'probability': probability,
            'keep': keep, 'protected': protected,
            'active_pairs': int((context.present > 0).sum().detach()) * queries,
            'fine_rows': batch * queries * context.centers.shape[1] if full is not None else int(keep.sum().detach()),
            'cheap_rows': batch * queries * context.centers.shape[1],
            'gate_rows': batch * queries * context.centers.shape[1],
            'padded_fine_capacity': batch * queries * context.centers.shape[1],
            'selected_detail_rows': int(keep.sum().detach()),
            'near_rows': int(near_index.shape[1]),
            'complete_fine_values': full is not None,
            'context_ancestry': 'all physical source and environment context; every active donor retains base influence'}
        if policy.mode == 'all_fine' and full is not None and fixed_route is None:
            # Cheap fitting does not perturb even the FP32 assembly of the
            # full-detail reference. Replay can use this exact same object.
            full.full_response = full
            full.refinement_aux = response.refinement_aux
            response = full
        self._refinement_auxiliary.extend(records)
        return response


@dataclass
class RefinementPrediction:
    values: torch.Tensor
    full_values: torch.Tensor | None
    auxiliary: dict


class RefinedNonlinearFieldReadout(_RefinementMixin, NonlinearFieldReadout):
    """Refine individual Wind messages before the retained nonlinear field head."""

    def __init__(self, *args, base_width=16, router_hidden=32, residual_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_refinement(self.config['message'], base_width, router_hidden, residual_scale)

    def read_refinement(self, prepared, receivers, receiver_features=None, *, chunk_size=512,
                        training_signal=None, fixed_route=None, collect_pair_arrays=True):
        policy = self.refinement_policy
        if training_signal is not None:
            policy = replace(policy, training_signal=bool(training_signal))
        self.assert_owned(prepared)
        batch, queries, _ = receivers.shape
        self._validate_shape(receivers, (prepared.centers.shape[0], queries, self.spatial_dim), 'receivers')
        features = receivers.new_empty(batch, queries, 0) if receiver_features is None else receiver_features
        self._validate_shape(features, (batch, queries, self.query_width), 'receiver_features')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive.')
        if fixed_route is not None and (fixed_route.dtype != torch.bool or fixed_route.shape != (batch, queries, prepared.centers.shape[1])):
            raise ValueError('Fixed route must be boolean [B,Q,M].')
        if (policy.mode == 'all_fine' or (policy.mode == 'adaptive' and policy.threshold == 0)) and not policy.training_signal and fixed_route is None:
            return RefinementPrediction(super().predict(prepared, receivers, features, chunk_size=chunk_size), None,
                {'fine_rows': batch * queries * prepared.centers.shape[1],
                 'active_pairs': int((prepared.present > 0).sum().detach()) * queries,
                 'cheap_rows': 0, 'near_rows': 0, 'gate_rows': 0})
        source_features = self.refinement.source_features(prepared)
        outputs, full_outputs, records, export = [], [], [], []
        paid_rows = 0
        selected_count = receivers.new_zeros((), dtype=torch.int64)
        near_count = receivers.new_zeros((), dtype=torch.int64)
        for start in range(0, queries, chunk_size):
            query, feature = receivers[:, start:start + chunk_size], features[:, start:start + chunk_size]
            protected = self.near_weight(query, prepared.centers, prepared.source_lengths, prepared.present) > 0
            base, probability = self.refinement.read(prepared, query, feature, source_features, policy.temperature,
                router_detach_inputs=policy.training_signal)
            route = None if fixed_route is None else fixed_route[:, start:start + chunk_size]
            keep = self.refinement.route(prepared, query, probability, protected, policy, route)
            if policy.training_signal or policy.phase == 'soft':
                fine = self.source_read(self._read_features(prepared, query,
                    prepared.source_states, prepared.centers, prepared.source_lengths, feature))
                fine_rows = fine.shape[0] * fine.shape[1] * fine.shape[2]
            else:
                selected = keep.nonzero(as_tuple=True)
                fine = base.clone().index_put(selected, self.source_read(self._read_selected_features(prepared, query, feature, selected)))
                fine_rows = selected[0].numel()
            mixed = _combine(base, fine, probability, keep, protected, policy)
            common = (prepared.global_state[:, None].expand(-1, query.shape[1], -1),
                      _geometry(query / prepared.lengths[:, None]), feature)
            outputs.append(self.field_head(torch.cat(((mixed * prepared.source_measure[:, None, :, None]).sum(2), *common), -1)))
            if policy.training_signal:
                full_outputs.append(self.field_head(torch.cat(((fine * prepared.source_measure[:, None, :, None]).sum(2), *common), -1)))
                record = self.refinement.auxiliary(base, fine, probability, protected, prepared, keep, fine_rows=fine_rows)
                record['near_rows'] = int(protected.sum().detach())
                records.append(record)
            paid_rows += fine_rows
            selected_count = selected_count + keep.sum()
            near_count = near_count + protected.sum()
            if collect_pair_arrays:
                export.append({'base': base, 'fine': fine, 'probability': probability,
                               'keep': keep, 'protected': protected})
        output = torch.cat(outputs, 1) if outputs else receivers.new_empty(batch, 0, self.output_width)
        full = torch.cat(full_outputs, 1) if full_outputs else None
        self._refinement_auxiliary.extend(records)
        aux = {name: torch.cat([item[name] for item in export], 1) for name in
               ('base', 'fine', 'probability', 'keep', 'protected')} if export else {}
        aux.update(fine_rows=paid_rows,
                   active_pairs=int((prepared.present > 0).sum().detach()) * queries,
                   cheap_rows=batch * queries * prepared.centers.shape[1],
                   gate_rows=batch * queries * prepared.centers.shape[1],
                   selected_detail_rows=int(selected_count.detach()),
                   near_rows=int(near_count.detach()),
                   pair_arrays_collected=bool(collect_pair_arrays),
                   complete_fine_values=policy.training_signal or policy.phase == 'soft')
        return RefinementPrediction(output, full, aux)

    def predict(self, prepared, receivers, receiver_features=None, *, chunk_size=512,
                return_messages=False, fixed_route=None):
        policy = self.refinement_policy
        if (policy.mode == 'all_fine' or (policy.mode == 'adaptive' and policy.threshold == 0)) and not policy.training_signal and fixed_route is None:
            return super().predict(prepared, receivers, receiver_features, chunk_size=chunk_size,
                                   return_messages=return_messages)
        result = self.read_refinement(prepared, receivers, receiver_features, chunk_size=chunk_size,
                                      fixed_route=fixed_route, collect_pair_arrays=return_messages)
        if return_messages:
            aux = result.auxiliary
            messages = _combine(aux['base'], aux['fine'], aux['probability'], aux['keep'],
                                aux['protected'], policy) * prepared.present[:, None, :, None]
            return result.values, messages
        return result.values
