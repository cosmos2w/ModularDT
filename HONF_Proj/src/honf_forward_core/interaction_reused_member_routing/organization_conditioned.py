"""Candidate B: organization-conditioned competition over original members.

Candidate B inherits Candidate A's encoder, prepared applied relations,
proposal/member scorers, typed read strengths and joint physical-field head.
It changes only how the already-selected original members compete: each live
organization row forms a conditional read over those members, and the row
reads are mixed by their retained proposal mass.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .consistent import ConsistentJointMemberRoutingCore


_RETRIEVAL_CALIBRATIONS = ("normalized", "row_strength")
_SCHEMA = "interaction_reused_organization_conditioned_member_routing_v1"
_B_CONTRACTION_COUNT_KEYS = (
    "query_B_retained_view_mass_contraction_pairs",
    "query_B_view_likelihood_mass_contraction_pairs",
    "query_B_coalesced_member_base_contraction_pairs",
)


class OrganizationConditionedMemberRoutingCore(ConsistentJointMemberRoutingCore):
    """R13-5B with organization-conditioned reads over the same original M/E members."""

    schema = _SCHEMA
    _additional_decode_count_keys = ConsistentJointMemberRoutingCore._additional_decode_count_keys + (
        *_B_CONTRACTION_COUNT_KEYS,
    )

    def __init__(
        self,
        base_model_config: Mapping[str, object],
        *,
        address_mode: str = "ME",
        input_dataset: str = "native_records",
        input_capability: str = "joint_prescribed_records_v1",
        address_temperature: float = 1.0,
        joint_member_coverage: float = 0.95,
        query_tile_size: int = 256,
        retrieval_calibration: str = "normalized",
        score_calibration: str = "inherited",
    ):
        if address_mode != "ME":
            raise ValueError("R13-5B is the ME organization-conditioned member reader.")
        super().__init__(
            base_model_config,
            address_mode=address_mode,
            input_dataset=input_dataset,
            input_capability=input_capability,
            address_temperature=address_temperature,
            joint_member_coverage=joint_member_coverage,
            query_tile_size=query_tile_size,
            retrieval_calibration=retrieval_calibration,
            score_calibration=score_calibration,
        )

    @staticmethod
    def _consistent_arm_name(address_mode: str) -> str:
        if address_mode != "ME":
            raise ValueError("R13-5B is the ME organization-conditioned member reader.")
        return "R13-5B-ME-OrganizationConditionedJoint95"

    def set_address_mode(self, address_mode: str) -> None:
        if address_mode != "ME":
            raise ValueError("R13-5B is the ME organization-conditioned member reader.")
        super().set_address_mode(address_mode)

    @classmethod
    def validate_config(cls, config: Mapping[str, object]) -> None:
        if config.get("schema") != cls.schema or config.get("address_mode") != "ME":
            raise ValueError("R13-5B config must name the ME organization-conditioned schema.")
        super().validate_config(config)

    def _joint_member_contribution(
        self,
        member_prior: Tensor,
        ell: Tensor,
        candidate: Tensor,
        *,
        proposal: Tensor,
        retrieval_R: Tensor,
        row_mass: Tensor,
        retrieval_calibration: str,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return organization_conditioned_member_contribution(
            member_prior,
            ell,
            candidate,
            proposal=proposal,
            retrieval_R=retrieval_R,
            row_mass=row_mass,
            retrieval_calibration=retrieval_calibration,
        )

    def _typed_read_state_with_plan(
        self,
        prepared,
        query,
        qstates,
        intervention,
        *,
        return_trace: bool,
        shared_plan=None,
    ):
        fused, trace, counts, fields, plan = super()._typed_read_state_with_plan(
            prepared,
            query,
            qstates,
            intervention,
            return_trace=return_trace,
            shared_plan=shared_plan,
        )
        active_retrieval = plan["active_R"]
        contraction_pairs = int(
            query.batch_size * query.num_queries * active_retrieval.shape[1] * active_retrieval.shape[2]
        )
        counts.update({name: contraction_pairs for name in _B_CONTRACTION_COUNT_KEYS})
        return fused, trace, counts, fields, plan


def organization_conditioned_member_contribution(
    member_prior: Tensor,
    scores: Tensor,
    candidate: Tensor,
    *,
    proposal: Tensor,
    retrieval_R: Tensor,
    row_mass: Tensor,
    retrieval_calibration: str,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Return B's coalesced member contribution and view-conditioned trace.

    Inputs use ``member_prior/scores/candidate=[B,Q,N]``, ``proposal=[B,Q,A]``,
    ``retrieval_R=[B,A,N]`` and ``row_mass=[B,A]``. The address weights are
    normalized ``proposal`` for normalized retrieval. For row-strength
    retrieval, they are normalized ``proposal * row_mass`` so that the view
    mixture and retained prior use the same calibration. The implementation
    uses three batched matrix contractions and never creates a [B,Q,A,N]
    tensor. ``scores`` are A's declared bounded member scores and lie in
    [-4,4]; one common query offset is shared across M and E.
    """
    if retrieval_calibration not in _RETRIEVAL_CALIBRATIONS:
        raise ValueError(f"Unknown retrieval calibration: {retrieval_calibration!r}.")
    if member_prior.shape != scores.shape or member_prior.shape != candidate.shape or candidate.dtype != torch.bool:
        raise ValueError("Member prior, scores and boolean candidate mask must share [B,Q,N] axes.")
    if proposal.ndim != 3 or retrieval_R.ndim != 3:
        raise ValueError("Proposal and retrieval must have [B,Q,A] and [B,A,N] axes.")
    batch, queries, members = scores.shape
    if proposal.shape[0] != batch or proposal.shape[1] != queries:
        raise ValueError("Proposal batch/query axes do not match member scores.")
    addresses = proposal.shape[-1]
    if retrieval_R.shape != (batch, addresses, members) or row_mass.shape != (batch, addresses):
        raise ValueError("Retrieval rows or masses do not match proposal/member axes.")
    if not (scores.is_floating_point() and proposal.is_floating_point() and retrieval_R.is_floating_point()
            and row_mass.is_floating_point()):
        raise TypeError("Scores, proposal, retrieval and row mass must be floating point.")

    zeros_bq = scores.new_zeros((batch, queries))
    zeros_bqa = proposal.new_zeros((batch, queries, addresses))
    if members == 0 or addresses == 0:
        contribution = torch.zeros_like(scores)
        log_weight = torch.full_like(scores, -torch.inf)
        trace = {
            "organization_conditioned_retained_view_mass": zeros_bqa,
            "organization_conditioned_scaled_view_likelihood_mass": zeros_bqa,
            "organization_conditioned_retained_prior_mass": zeros_bq,
            "organization_conditioned_effective_view_weights": zeros_bqa,
            "organization_conditioned_view_mixture_mass": zeros_bqa,
            "organization_conditioned_scaled_view_member_coefficients": zeros_bqa,
            "organization_conditioned_scaled_member_base": torch.zeros_like(scores),
            "organization_conditioned_score_offset": zeros_bq,
            "organization_conditioned_pre_normalization_mass": zeros_bq,
            "organization_conditioned_pre_normalization_error": zeros_bq,
        }
        return contribution, log_weight, zeros_bq, trace

    effective_view_weight = proposal
    if retrieval_calibration == "row_strength":
        effective_view_weight = proposal * row_mass[:, None, :]
    weight_total = effective_view_weight.sum(-1, keepdim=True)
    tiny = torch.finfo(effective_view_weight.dtype).tiny
    view_weight = torch.where(
        weight_total > 0,
        effective_view_weight / weight_total.clamp_min(tiny),
        torch.zeros_like(effective_view_weight),
    )

    # One common query score offset is shared across all selected M and E
    # members. This keeps exp(scores) positive and bounded without erasing
    # cross-role score differences.
    has_candidate = candidate.any(-1, keepdim=True)
    score_offset = torch.where(
        has_candidate,
        torch.where(candidate, scores, torch.full_like(scores, -torch.inf)).amax(-1, keepdim=True),
        torch.zeros_like(scores[..., :1]),
    )
    scaled_likelihood = torch.where(
        candidate,
        torch.exp(scores - score_offset),
        torch.zeros_like(scores),
    )

    retrieval_t = retrieval_R.transpose(1, 2)
    candidate_values = candidate.to(retrieval_R.dtype)
    # d = C R^T and Z = (C * exp(ell-offset)) R^T.
    raw_retained_view_mass = torch.bmm(candidate_values, retrieval_t)
    live_view = raw_retained_view_mass > 0
    # At an exactly empty view, keep both its value and derivative at zero.
    # This makes empty-view behavior exact instead of allowing a zero row to
    # perturb D through a one-sided derivative at the boundary.
    retained_view_mass = torch.where(live_view, raw_retained_view_mass,
                                     torch.zeros_like(raw_retained_view_mass))
    raw_view_likelihood_mass = torch.bmm(scaled_likelihood.to(retrieval_R.dtype), retrieval_t)
    view_likelihood_mass = torch.where(live_view, raw_view_likelihood_mass,
                                       torch.zeros_like(raw_view_likelihood_mass))
    retained_prior_mass = (view_weight * retained_view_mass).sum(-1, keepdim=True)
    has_retained_mass = retained_prior_mass > 0
    safe_prior_mass = retained_prior_mass.clamp_min(torch.finfo(retained_prior_mass.dtype).tiny)
    view_mixture_mass = torch.where(
        has_retained_mass,
        view_weight * retained_view_mass / safe_prior_mass,
        torch.zeros_like(retained_view_mass),
    )
    has_likelihood_mass = view_likelihood_mass > 0
    safe_likelihood_mass = torch.where(
        has_likelihood_mass,
        view_likelihood_mass,
        torch.ones_like(view_likelihood_mass),
    )
    view_member_coefficients = torch.where(
        has_retained_mass & has_likelihood_mass,
        view_mixture_mass / safe_likelihood_mass,
        torch.zeros_like(view_mixture_mass),
    )
    # Coalesce all memberships onto the original member axis before values
    # are read. The candidate mask keeps excluded members at exact zero.
    member_base = torch.bmm(view_member_coefficients.to(retrieval_R.dtype), retrieval_R)
    member_base = member_base * candidate_values

    valid_weight = candidate & (member_base > 0)
    log_weight = torch.where(
        valid_weight,
        member_base.clamp_min(torch.finfo(member_base.dtype).tiny).log() + scores - score_offset,
        torch.full_like(scores, -torch.inf),
    )
    has_weight = valid_weight.any(-1, keepdim=True)
    common_offset = torch.where(
        has_weight,
        log_weight.amax(-1, keepdim=True),
        torch.zeros_like(log_weight[..., :1]),
    )
    shifted = torch.where(valid_weight, log_weight - common_offset, torch.zeros_like(log_weight))
    unnormalized = torch.exp(shifted) * valid_weight.to(scores.dtype)
    normalizer = unnormalized.sum(-1, keepdim=True)
    log_partition = common_offset + normalizer.clamp_min(torch.finfo(normalizer.dtype).tiny).log()
    pre_normalization_mass = torch.where(
        normalizer > 0,
        torch.exp(log_partition),
        torch.zeros_like(normalizer),
    )
    contribution = torch.where(
        normalizer > 0,
        unnormalized / normalizer.clamp_min(torch.finfo(unnormalized.dtype).tiny),
        torch.zeros_like(unnormalized),
    )
    pre_normalization_error = torch.where(
        normalizer > 0,
        pre_normalization_mass - torch.ones_like(normalizer),
        torch.zeros_like(normalizer),
    )

    trace = {
        "organization_conditioned_retained_view_mass": retained_view_mass,
        "organization_conditioned_scaled_view_likelihood_mass": view_likelihood_mass,
        "organization_conditioned_retained_prior_mass": retained_prior_mass.squeeze(-1),
        "organization_conditioned_effective_view_weights": view_weight,
        "organization_conditioned_view_mixture_mass": view_mixture_mass,
        "organization_conditioned_scaled_view_member_coefficients": view_member_coefficients,
        "organization_conditioned_scaled_member_base": member_base,
        "organization_conditioned_score_offset": score_offset.squeeze(-1),
        "organization_conditioned_pre_normalization_mass": pre_normalization_mass.squeeze(-1),
        "organization_conditioned_pre_normalization_error": pre_normalization_error.squeeze(-1),
    }
    return contribution, log_weight, common_offset.squeeze(-1), trace


__all__ = [
    "organization_conditioned_member_contribution",
]
