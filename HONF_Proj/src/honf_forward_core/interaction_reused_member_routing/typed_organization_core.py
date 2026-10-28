"""R13-6 H reader with typed organization competition over original members."""
from __future__ import annotations

import torch
from torch import Tensor

from .faithful import FaithfulMECore
from .typed_organization import organization_conditioned_role_contribution


_SCHEMA = "interaction_reused_typed_organization_member_routing_v1"
_H_CONTRACTION_COUNT_KEYS = (
    "query_H_retained_view_mass_contraction_pairs",
    "query_H_view_likelihood_mass_contraction_pairs",
    "query_H_coalesced_member_base_contraction_pairs",
)


class TypedOrganizationMemberRoutingCore(FaithfulMECore):
    """Keep O's typed gate statistics while using H's per-view member read."""

    schema = _SCHEMA
    _additional_decode_count_keys = FaithfulMECore._additional_decode_count_keys + _H_CONTRACTION_COUNT_KEYS

    def __init__(self, base_model_config, **kwargs):
        retained_prior_law = kwargs.pop("retained_prior_law", "none")
        if retained_prior_law != "none":
            raise ValueError("R13-6 H initially requires retained_prior_law='none'.")
        super().__init__(base_model_config, retained_prior_law="none", **kwargs)

    @staticmethod
    def _faithful_arm_name(address_mode: str) -> str:
        return f"R13-6-H-{address_mode}-TypedOrganizationJoint95"

    @classmethod
    def validate_config(cls, config):
        super().validate_config(config)
        if config.get("retained_prior_law") != "none":
            raise ValueError("R13-6 H config cannot combine typed organization with retained-prior law K.")

    @staticmethod
    def _typed_organization_distribution(
        ell: Tensor,
        candidate: Tensor,
        proposal: Tensor,
        retrieval_R: Tensor,
        module_count: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Build H's unchanged M/E distributions directly from the shared scores and plan."""
        if not 0 <= module_count <= ell.shape[-1]:
            raise ValueError("module_count must split the joint M/E member axis.")
        if proposal.shape[-1] != retrieval_R.shape[1]:
            raise ValueError("Typed organization proposals and retrieval rows must share their address axis.")
        m_slice = slice(0, module_count)
        e_slice = slice(module_count, ell.shape[-1])
        value_m = organization_conditioned_role_contribution(
            retrieval_R[..., m_slice],
            proposal,
            ell[..., m_slice],
            candidate[..., m_slice],
        )
        value_e = organization_conditioned_role_contribution(
            retrieval_R[..., e_slice],
            proposal,
            ell[..., e_slice],
            candidate[..., e_slice],
        )
        distribution = torch.cat((value_m, value_e), dim=-1)
        trace = {
            "typed_organization_member_contribution_M": value_m,
            "typed_organization_member_contribution_E": value_e,
        }
        return distribution, trace

    def _role_read_contributions(
        self,
        marginal_contribution: Tensor | None,
        member_prior: Tensor,
        ell: Tensor,
        candidate: Tensor,
        *,
        full_candidate: Tensor,
        proposal: Tensor,
        retrieval_R: Tensor,
        row_mass: Tensor,
        module_count: int,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        gate_contribution, _original_value_contribution, trace = super()._role_read_contributions(
            marginal_contribution,
            member_prior,
            ell,
            candidate,
            full_candidate=full_candidate,
            proposal=proposal,
            retrieval_R=retrieval_R,
            row_mass=row_mass,
            module_count=module_count,
        )
        del _original_value_contribution, member_prior, full_candidate, row_mass
        value_contribution, organization_trace = self._typed_organization_distribution(
            ell, candidate, proposal, retrieval_R, module_count,
        )
        trace.update(organization_trace)
        return gate_contribution, value_contribution, trace

    def _typed_read_state_with_plan(self, prepared, query, qstates, intervention, *,
                                    return_trace, shared_plan=None):
        fused, trace, counts, fields, plan = super()._typed_read_state_with_plan(
            prepared, query, qstates, intervention,
            return_trace=return_trace,
            shared_plan=shared_plan,
        )
        # The three H contractions run independently for M and E. Summing
        # their member-axis work gives 3 * B * Q * A * (M + E) pairs.
        retrieval = plan["active_R"]
        contraction_pairs = int(
            query.batch_size * query.num_queries * retrieval.shape[1] * retrieval.shape[2]
        )
        counts.update({name: contraction_pairs for name in _H_CONTRACTION_COUNT_KEYS})
        return fused, trace, counts, fields, plan


__all__ = ["TypedOrganizationMemberRoutingCore"]
