"""R13-7 U: one organization-conditioned distribution for statistics and values."""
from __future__ import annotations

import torch
from torch import Tensor

from .typed_organization_core import TypedOrganizationMemberRoutingCore


def _normalized_role_statistics(
    probabilities: Tensor,
    scores: Tensor,
    role_allowed: Tensor,
    scene_eligible_count: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use H's final probabilities directly, including its exact empty support.

    The H operator owns masking and numerical normalization. Repeating either
    here would give statistics a different tensor and a different Jacobian
    from the value read. The denominator remains the full eligible scene count.
    """
    if (probabilities.ndim != 3 or probabilities.shape != scores.shape
            or probabilities.shape != role_allowed.shape or role_allowed.dtype != torch.bool):
        raise ValueError("Role statistics require equal [B,Q,N] probability, score and boolean mask axes.")
    if not probabilities.is_floating_point() or probabilities.dtype != scores.dtype:
        raise TypeError("Role probabilities and scores must have the same floating dtype.")
    if probabilities.device != scores.device or probabilities.device != role_allowed.device:
        raise ValueError("Role probabilities, scores and masks must share a device.")
    if (scene_eligible_count.shape != probabilities.shape[:1]
            and scene_eligible_count.shape != probabilities.shape[:2]):
        raise ValueError("scene_eligible_count must have [B] or [B,Q] axes.")
    expectation = (probabilities * scores).sum(-1)
    tiny = torch.finfo(probabilities.dtype).tiny
    entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum(-1)
    count = scene_eligible_count.to(probabilities)
    if count.ndim == 1:
        count = count[:, None]
    normalized_entropy = torch.where(
        count > 1, entropy / count.clamp_min(2.).log(), torch.zeros_like(entropy),
    )
    return probabilities, expectation, normalized_entropy


class SingleDistributionMemberRoutingCore(TypedOrganizationMemberRoutingCore):
    """Keep H's member read and use that same distribution for typed strength."""

    schema = "interaction_reused_single_distribution_member_routing_v1"

    @staticmethod
    def _faithful_arm_name(address_mode: str) -> str:
        return f"R13-7-U-{address_mode}-SingleDistributionJoint95"

    def _role_read_contributions(
        self, marginal_contribution, member_prior, ell, candidate, *,
        full_candidate, proposal, retrieval_R, row_mass, module_count,
    ):
        # U does not construct the parallel O likelihood, including for traces.
        # Requested legacy A log weights are still calculated by the base decoder.
        del marginal_contribution, member_prior, full_candidate, row_mass
        probabilities, trace = self._typed_organization_distribution(
            ell, candidate, proposal, retrieval_R, module_count,
        )
        return probabilities, probabilities, trace

    @staticmethod
    def _role_statistics(probabilities, scores, candidate, scene_eligible_count):
        return _normalized_role_statistics(probabilities, scores, candidate, scene_eligible_count)

    def _typed_read_state_with_plan(
        self, prepared, query, qstates, intervention, *, return_trace, shared_plan=None,
    ):
        fused, trace, counts, fields, plan = super()._typed_read_state_with_plan(
            prepared, query, qstates, intervention,
            return_trace=return_trace, shared_plan=shared_plan,
        )
        # These counters describe O's positive-softmax operations, which U
        # never executes. H's three per-role contractions remain charged.
        counts["query_M_member_softmax_rows"] = 0
        counts["query_E_member_softmax_rows"] = 0
        if return_trace:
            for role in ("M", "E"):
                trace[f"member_strength_probabilities_{role}"] = trace[f"member_conditional_likelihood_{role}"]
                trace[f"member_value_probabilities_{role}"] = trace[f"member_contribution_pre_strength_{role}"]
                trace[f"member_context_norm_{role}"] = trace[f"context_{role}"].norm(dim=-1)
        return fused, trace, counts, fields, plan


__all__ = ["SingleDistributionMemberRoutingCore"]
