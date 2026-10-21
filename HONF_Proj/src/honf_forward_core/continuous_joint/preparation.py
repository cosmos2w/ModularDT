"""Applied typed contextual value exchange for the local-record capability.

Scores remain dense control-plane inputs. Only positive-incidence donor edges
participate in the contextual value reduction; token projections remain local.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class FrozenContextPlan:
    """A complete frozen reduction, including coefficients and normalizers.

    Intended for bounded value-plane interventions, never ordinary inference.
    Freezing only support would leave live score dependencies in this test.
    """
    incidence: Tensor
    coefficients: Tensor  # [B,R,D,H]
    message_mass: Tensor  # [B,R,H]


@dataclass(frozen=True)
class FrozenSDPAContextPlan:
    """Freeze the actual SDPA control operator while reading fresh donor V.

    Keeping Q/K and the additive mask preserves the original FP32 kernel.
    Explicit softmax coefficients are only a diagnostic approximation to that
    kernel, so they are not substituted into the all-open value execution.
    """
    incidence: Tensor
    query: Tensor
    key: Tensor
    additive_mask: Tensor
    row_present: Tensor
    message_mass: Tensor


def context_plan_tensors(plan):
    if isinstance(plan,FrozenSDPAContextPlan):
        return (plan.incidence,plan.query,plan.key,plan.additive_mask,
                plan.row_present,plan.message_mass)
    return (plan.incidence,plan.coefficients,plan.message_mass)


def contextual_value_update(attention, query, key, value, geometry_bias,
                            donor_measure, allowed, destination_mask, policy, *,
                            route_name, stage_identity, destination_ids, donor_ids):
    """Apply b before a guarded native-measure reduction; no pair-value bank."""
    from ..content_hypergraph.staged_attention_v2 import safe_masked_softmax

    scores = torch.matmul(query, key.transpose(-2, -1)) + geometry_bias
    selected = policy(scores, allowed, route_name, stage_identity=stage_identity,
                      destination_ids=destination_ids, donor_ids=donor_ids)
    if isinstance(selected,FrozenSDPAContextPlan):
        if selected.incidence.shape!=allowed.shape or selected.query.shape!=query.shape or selected.key.shape!=key.shape:
            raise ValueError('Frozen SDPA controls must match the native contextual axes.')
        if any(t.requires_grad for t in context_plan_tensors(selected)):
            raise ValueError('Frozen SDPA contextual controls must be detached in full.')
        attended=F.scaled_dot_product_attention(selected.query,selected.key,value,
            attn_mask=selected.additive_mask,dropout_p=0.,scale=1.)
        attended=attended.transpose(1,2).reshape(query.shape[0],query.shape[2],attention.width)
        update=attention.output(attended)*selected.row_present[...,None].to(attended.dtype)
        # This bounded trace describes the frozen controls; the actual value
        # operation above retains SDPA rather than using these explicit weights.
        weights=safe_masked_softmax(torch.matmul(selected.query,selected.key.transpose(-2,-1))
            +selected.additive_mask,selected.incidence.bool()[:,None],dim=-1)
        return update,weights,selected.incidence,selected.message_mass
    if isinstance(selected, FrozenContextPlan):
        gates, coefficients, mass = selected.incidence, selected.coefficients, selected.message_mass
        if coefficients.shape != (*allowed.shape, attention.heads):
            raise ValueError("Frozen contextual coefficients must match [B,R,D,H].")
        if mass.shape != (*allowed.shape[:2], attention.heads):
            raise ValueError("Frozen contextual mass must match [B,R,H].")
        if any(x.requires_grad for x in (gates, coefficients, mass)):
            raise ValueError("A frozen contextual execution plan must be detached in full.")
    else:
        gates = selected
        measure = donor_measure.to(scores.dtype)
        valid_measure = measure > 0
        effective_allowed = allowed & valid_measure[:, None, :]
        safe_measure = torch.where(valid_measure, measure, torch.ones_like(measure))
        logits = scores + safe_measure.log()[:, None, None, :]
        base_weights = safe_masked_softmax(logits, effective_allowed[:, None], dim=-1)
        unnormalized = base_weights * gates[:, None]
        z = unnormalized.sum(-1, keepdim=True)
        denominator = z + 1.0e-3 * (1 - z).square()
        coefficients = (unnormalized / denominator).permute(0, 2, 3, 1)
        mass = (z / denominator).squeeze(-1).transpose(1, 2)
    if gates.shape != allowed.shape:
        raise ValueError("Contextual incidence must match [B,R,D].")
    if gates.device.type == 'cpu' and (bool((gates < 0).any()) or bool((gates > 1).any())
            or bool((gates.masked_select(~allowed) != 0).any())):
        raise ValueError("Contextual incidence must lie in [0,1] on eligible pairs only.")
    if getattr(policy, "all_open", False) and not isinstance(selected, FrozenContextPlan):
        # Fair all-open contextual control retains the original SDPA kernel.
        additive = geometry_bias + safe_measure.log()[:, None, None, :]
        additive = additive.masked_fill(~effective_allowed[:, None], -torch.inf)
        row_present = effective_allowed.any(-1)
        additive = torch.where(row_present[:, None, :, None], additive, torch.zeros_like(additive))
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=additive,
            dropout_p=attention.dropout if attention.training else 0.0, scale=1.0)
        attended = attended.transpose(1, 2).reshape(query.shape[0], query.shape[2], attention.width)
        update = attention.output(attended) * row_present[..., None].to(attended.dtype)
        return update, coefficients.permute(0, 3, 1, 2), gates, mass
    rows = (gates > 0).nonzero(as_tuple=False)
    batch, _, receivers, _ = query.shape
    reduced = value.new_zeros(batch * receivers, attention.heads, attention.head_width)
    if rows.numel():
        b, r, d = rows.unbind(-1)
        messages = value.permute(0, 2, 1, 3)[b, d]
        weights = coefficients[b, r, d]
        contribution = messages * weights[..., None]
        reduced = reduced.index_add(0, b * receivers + r, contribution)
    else:
        # A dense zero-coefficient reduction still visits projected V in its
        # autograd graph. Preserve its zero gradient (and AdamW state/decay)
        # without materializing donor-edge values. Empty views also avoid
        # reading poisoned or nonfinite values on an unused value path.
        reduced = reduced + value.unsqueeze(0)[:0].sum()
    reduced = reduced.reshape(batch, receivers, attention.width)
    # A projection bias belongs to the donor path. Split it equally across
    # heads, so its reduced coefficient is mean per-head message mass.
    update = F.linear(reduced, attention.output.weight, None)
    if attention.output.bias is not None:
        update = update + attention.output.bias * mass.mean(-1, keepdim=True)
    update = update * destination_mask[..., None].to(update.dtype)
    return update, coefficients.permute(0, 3, 1, 2), gates, mass


def context_stage_manifest(cycle_trace):
    """Small source/stage metadata; full coefficients require an explicit trace."""
    rows = []
    for route in ("mm", "em", "ee", "me"):
        snapshot = cycle_trace[route + "_native_route_snapshot"]
        gates = cycle_trace[route + "_applied_incidence"]
        rows.append(dict(stage=snapshot.stage_identity, route=route,
            application="contextual_value_reduction", destination_type=route[0], donor_type=route[1],
            destination_ids=snapshot.destination_ids, donor_ids=snapshot.donor_ids,
            owner_token=snapshot.owner_token, scene_identity=snapshot.scene_identity,
            scene_tensor_versions=snapshot.scene_tensor_versions,
            source_state_version=snapshot.stage_identity.split('/')[0] + '/' + dict(
                mm="boundary_M",em="after_MM_M",ee="after_EM_E",me="after_EE_FFN_E")[route],
            eligible_edges=int(snapshot.allowed.sum().detach()),
            positive_value_edges=int((gates > 0).sum().detach())))
    return rows


def compact_context_work(cycle_trace, absolute_epoch):
    """Per-case proxy, exact and eligible totals, reduced before persistence."""
    age = max(0, int(absolute_epoch)-3000)
    delta = 10 ** (-2 - 2 * min(max(age-1,0)/999,1))
    summary = None
    for route in ("mm", "em", "ee", "me"):
        gates = cycle_trace[route + "_applied_incidence"]
        allowed = cycle_trace[route + "_native_route_snapshot"].allowed
        proxy = ((1+delta)*gates/(gates+delta))*allowed
        row = torch.stack((proxy.sum((1,2)), (gates.detach()>0).sum((1,2)),
                           allowed.sum((1,2))),-1)
        summary = row if summary is None else summary + row
    return summary


@dataclass(frozen=True)
class FrozenUnifiedContextPlan:
    """Bounded owner-bound two-cycle contextual intervention, with all coefficients."""
    owner_token: int
    parameter_versions: tuple
    buffer_versions: tuple
    coordinate_frame_key: object
    normalizer_key: object
    module_ids: Tensor
    environment_ids: Tensor
    module_mask: Tensor
    environment_mask: Tensor
    module_information_mask: Tensor
    environment_information_mask: Tensor
    stages: dict[str,FrozenContextPlan | FrozenSDPAContextPlan]
    tensor_stamps: tuple

    def assert_current(self,core,scene):
        from ..native_retrieval.contracts import tensor_versions
        if (core._owner_token!=self.owner_token or core._parameter_version_stamp()!=self.parameter_versions
                or core._buffer_version_stamp()!=self.buffer_versions):
            raise ValueError('Frozen unified context owner or weights changed.')
        if scene.coordinate_frame_key!=self.coordinate_frame_key or scene.normalizer_key!=self.normalizer_key:
            raise ValueError('Frozen unified context frame or input normalizer differs.')
        actual=(scene.module_ids,scene.environment_ids,scene.module_mask,scene.environment_mask,
                scene.module_measure>0,scene.environment_measure>0)
        expected=(self.module_ids,self.environment_ids,self.module_mask,self.environment_mask,
                  self.module_information_mask,self.environment_information_mask)
        if any(not torch.equal(a,b) for a,b in zip(actual,expected)):
            raise ValueError('Frozen unified context native IDs/presence/information eligibility differ.')
        tensors=list(expected)
        expected_stages={f'cycle{cycle}/{route}' for cycle in range(core.context_depth) for route in ('mm','em','ee','me')}
        if set(self.stages)!=expected_stages:
            raise ValueError('Frozen unified context stage inventory changed.')
        parameter=next(core.parameters())
        for stage in sorted(self.stages):
            plan=self.stages[stage]
            reference=plan.query if isinstance(plan,FrozenSDPAContextPlan) else plan.coefficients
            if reference.dtype!=parameter.dtype or reference.device!=parameter.device:
                raise ValueError('Frozen unified context precision/device differs from its owner.')
            tensors.extend(context_plan_tensors(plan))
        if tensor_versions(*tensors)!=self.tensor_stamps:
            raise ValueError('Frozen unified context coefficients changed after capture.')


class FrozenContextPolicy:
    """No trainable state; a temporary explicit scientific intervention."""
    def __init__(self,plan):self.plan=plan
    def __call__(self,*args,stage_identity=None,**metadata):return self.plan.stages[stage_identity]
