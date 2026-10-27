"""Faithful R13-6 O execution for the original joint M/E member reader.

This core keeps R13-5's parameter inventory and typed per-role read equation.
It reuses Candidate A's scene-live normalized keys, factored scorers and paired
query plan, while keeping each role's candidate-conditioned member law and
strength separate. The optional K law only scales a restricted role by the
fraction of that role's legal prior mass retained by its candidate set.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .consistent import (
    ConsistentJointMemberRoutingCore,
    _normalize_nonnegative,
)
from .core import InteractionReusedMemberRoutingCore, _positive_softmax
from .joint import JointMemberRoutingCore


_SCHEMA = "interaction_reused_faithful_joint_member_routing_v1"
_RETAINED_PRIOR_LAWS = ("none", "K")


def original_role_conditional_likelihood(prior: Tensor, ell: Tensor, candidate: Tensor) -> Tensor:
    """Return the original ME ``P exp(ell)`` law, normalized within one role.

    ``ell`` is already calibrated against this candidate set. This follows the
    original reader's conditional-prior normalization and positive softmax,
    including exact zero output for empty candidate roles.
    """
    if prior.shape != ell.shape or prior.shape != candidate.shape or candidate.dtype != torch.bool:
        raise ValueError("Role prior, likelihood and boolean candidates must have equal shapes.")
    if not prior.is_floating_point() or not ell.is_floating_point():
        raise TypeError("Role prior and likelihood must be floating point.")
    conditional_prior = _normalize_nonnegative(prior, candidate)
    tiny = torch.finfo(prior.dtype).tiny
    log_prior = torch.where(
        candidate,
        conditional_prior.clamp_min(tiny).log(),
        torch.zeros_like(conditional_prior),
    )
    return _positive_softmax(ell + log_prior, candidate)


def retained_role_prior_fraction(prior: Tensor, candidate: Tensor, full_candidate: Tensor) -> Tensor:
    """Return selected/legal prior mass by role, with exact zero empty behavior."""
    if (prior.shape != candidate.shape or prior.shape != full_candidate.shape
            or candidate.dtype != torch.bool or full_candidate.dtype != torch.bool):
        raise ValueError("Prior and boolean candidate masks must have equal shapes.")
    retained = (prior * candidate.to(prior.dtype)).sum(-1)
    legal = (prior * full_candidate.to(prior.dtype)).sum(-1)
    tiny = torch.finfo(prior.dtype).tiny
    return torch.where(legal > 0, retained / legal.clamp_min(tiny), torch.zeros_like(legal))


class FaithfulMECore(ConsistentJointMemberRoutingCore):
    """Original per-role ME equation with equivalent shared execution rewrites."""

    schema = _SCHEMA
    _additional_decode_count_keys = ConsistentJointMemberRoutingCore._additional_decode_count_keys

    def __init__(
        self,
        base_model_config: Mapping[str, Any],
        *,
        address_mode: str = "ME",
        input_dataset: str = "native_records",
        input_capability: str = "joint_prescribed_records_v1",
        address_temperature: float = 1.0,
        joint_member_coverage: float = 0.95,
        query_tile_size: int = 256,
        retained_prior_law: str = "none",
    ):
        self._validate_retained_prior_law(retained_prior_law)
        self.retained_prior_law = retained_prior_law
        super().__init__(
            base_model_config,
            address_mode=address_mode,
            input_dataset=input_dataset,
            input_capability=input_capability,
            address_temperature=address_temperature,
            joint_member_coverage=joint_member_coverage,
            query_tile_size=query_tile_size,
            retrieval_calibration="normalized",
            score_calibration="inherited",
        )
        self.architecture = self._faithful_arm_name(address_mode)

    @staticmethod
    def _faithful_arm_name(address_mode: str) -> str:
        return f"R13-6-O-{address_mode}-FaithfulJoint95"

    @staticmethod
    def _validate_retained_prior_law(law: str) -> None:
        if law not in _RETAINED_PRIOR_LAWS:
            raise ValueError(f"retained_prior_law must be one of {_RETAINED_PRIOR_LAWS}.")

    @property
    def architecture_name(self) -> str:
        return self._faithful_arm_name(self.address_mode)

    def set_address_mode(self, address_mode: str) -> None:
        super().set_address_mode(address_mode)
        self.architecture = self._faithful_arm_name(address_mode)

    def get_config(self) -> dict[str, Any]:
        base = InteractionReusedMemberRoutingCore.get_config(self)
        return {
            "schema": self.schema,
            "architecture": self.architecture_name,
            "address_mode": self.address_mode,
            "base_model_config": base["base_model_config"],
            "input_dataset": self.input_dataset,
            "input_capability": self.input_capability,
            "address_temperature": self.address_temperature,
            "joint_member_coverage": self.joint_member_coverage,
            "query_tile_size": self.query_tile_size,
            "retained_prior_law": self.retained_prior_law,
        }

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        required = {
            "schema", "architecture", "address_mode", "base_model_config", "input_dataset",
            "input_capability", "address_temperature", "joint_member_coverage", "query_tile_size",
            "retained_prior_law",
        }
        if set(config) != required or config.get("schema") != cls.schema:
            raise ValueError("R13-6 O faithful joint routing configuration inventory/schema differs.")
        mode = config["address_mode"]
        if mode not in ("M", "ME") or config["architecture"] != cls._faithful_arm_name(mode):
            raise ValueError("R13-6 O architecture and address_mode disagree.")
        cls._validate_retained_prior_law(config["retained_prior_law"])
        translated = {
            key: value for key, value in config.items()
            if key not in ("schema", "architecture", "retained_prior_law")
        }
        translated.update(
            schema=ConsistentJointMemberRoutingCore.schema,
            architecture=ConsistentJointMemberRoutingCore._consistent_arm_name(mode),
            retrieval_calibration="normalized",
            score_calibration="inherited",
        )
        ConsistentJointMemberRoutingCore.validate_config(translated)

    def _role_read_contributions(
        self,
        marginal_contribution: Tensor,
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
        """Preserve original role-local gates and optionally apply K to values."""
        del proposal, retrieval_R, row_mass
        total = member_prior.shape[-1]
        if not 0 <= module_count <= total:
            raise ValueError("module_count must split the joint M/E member axis.")
        role_slices = (("M", slice(0, module_count)), ("E", slice(module_count, total)))
        role_likelihoods = []
        role_fractions: dict[str, Tensor] = {}
        for role, role_slice in role_slices:
            prior_role = member_prior[..., role_slice]
            ell_role = ell[..., role_slice]
            candidate_role = candidate[..., role_slice]
            full_role = full_candidate[..., role_slice]
            role_likelihood = original_role_conditional_likelihood(prior_role, ell_role, candidate_role)
            role_likelihoods.append(role_likelihood)
            if self.retained_prior_law == "K":
                role_fractions[role] = retained_role_prior_fraction(prior_role, candidate_role, full_role)
        gate_contribution = torch.cat(role_likelihoods, dim=-1)
        if self.retained_prior_law == "K":
            value_contribution = torch.cat((
                role_likelihoods[0] * role_fractions["M"][..., None],
                role_likelihoods[1] * role_fractions["E"][..., None],
            ), dim=-1)
            trace = {
                "member_retained_prior_fraction_M": role_fractions["M"],
                "member_retained_prior_fraction_E": role_fractions["E"],
            }
        else:
            value_contribution = gate_contribution
            trace = {}
        # The legacy global contribution is accepted as an extension seam, but
        # O deliberately returns its original role-local distribution for both
        # gate statistics and value integration.
        del marginal_contribution
        return gate_contribution, value_contribution, trace

    def _typed_read_state_with_plan(
        self, prepared, query, qstates, intervention, *, return_trace, shared_plan=None,
    ):
        fused, trace, counts, fields, plan = super()._typed_read_state_with_plan(
            prepared, query, qstates, intervention,
            return_trace=return_trace,
            shared_plan=shared_plan,
        )
        # O performs separate candidate-conditioned M/E likelihood softmaxes
        # in addition to the inherited global contribution calculation.
        counts["query_M_member_softmax_rows"] = int(counts["query_M_score_calibration_rows"])
        counts["query_E_member_softmax_rows"] = int(counts["query_E_score_calibration_rows"])
        if trace is not None and "member_conditional_likelihood_M" in trace:
            for role in ("M", "E"):
                trace[f"member_relative_weights_{role}"] = trace[f"member_conditional_likelihood_{role}"]
                trace[f"member_weights_{role}"] = trace[f"member_contribution_{role}"]
        return fused, trace, counts, fields, plan


__all__ = [
    "FaithfulMECore",
    "original_role_conditional_likelihood",
    "retained_role_prior_fraction",
]
