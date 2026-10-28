"""Candidate A: consistent joint mass over organized original M/E members.

This module reuses the R13-5 encoder, prepared applied-H state, proposal,
member scorers, typed read-strength gates, and general joint field head. It
changes only the read composition: likelihood-weighted member mass is
normalized once across the retained M+E set, then multiplied by the existing
typed read strengths.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..content_hypergraph.query_chunks_v1 import slice_query_v1
from .core import (
    InteractionJointOutput,
    InteractionReusedMemberRoutingCore,
    _pack_work_counts,
    _tensor_version,
)
from .guided import prior_weighted_scores
from .joint import (
    JointMemberRoutingCore,
    _masked_softmax,
    _normalize_nonnegative,
    joint_member_prefix_mask,
)
from .relative import read_probability


_SCHEMA = "interaction_reused_consistent_joint_member_routing_v1"
_RETRIEVAL_CALIBRATIONS = ("normalized", "row_strength")
_SCORE_CALIBRATIONS = ("inherited", "bounded_tanh")
_PAIRED_SHARED_COUNT_KEYS = (
    "query_M_proposal_score_rows", "query_E_proposal_score_rows", "query_joint_address_score_rows",
    "query_joint_address_base_measure_rows", "query_joint_address_prefix_sort_rows",
    "query_joint_address_prefix_id_sort_rows", "query_address_softmax_rows",
    "query_address_index_scan_rows", "query_address_coefficient_pair_rows",
    "query_address_support_pair_rows", "query_address_relation_rows",
    "query_M_prior_composition_rows", "query_E_prior_composition_rows",
    "query_M_prior_normalization_rows", "query_E_prior_normalization_rows",
    "query_joint_raw_composition_pair_rows", "query_joint_uncut_composition_pair_rows",
    "query_joint_full_eligible_rows", "query_routing_feature_normalization_rows", "query_receiver_rows",
)
_PAIRED_COUNT_KEYS = (
    "query_M_proposal_score_rows", "query_E_proposal_score_rows", "query_joint_address_score_rows",
    "query_joint_address_base_measure_rows", "query_joint_address_prefix_sort_rows",
    "query_joint_address_prefix_id_sort_rows", "query_address_softmax_rows",
    "query_address_index_scan_rows", "query_address_coefficient_pair_rows",
    "query_address_support_pair_rows", "query_address_relation_rows", "query_typed_read_gate_rows",
    "query_M_prior_composition_rows", "query_E_prior_composition_rows",
    "query_M_prior_normalization_rows", "query_E_prior_normalization_rows",
    "query_M_continuous_value_rows", "query_E_continuous_value_rows",
    "query_M_member_softmax_rows", "query_E_member_softmax_rows", "query_M_sorted_rows",
    "query_E_sorted_rows", "query_M_id_sort_rows", "query_E_id_sort_rows",
    "query_E_prior_prefix_sort_rows", "query_E_prior_id_sort_rows", "query_joint_prefix_sort_rows",
    "query_joint_prefix_id_sort_rows", "query_joint_full_eligible_rows",
    "query_joint_member_coverage_rows", "query_joint_raw_composition_pair_rows",
    "query_joint_uncut_composition_pair_rows", "query_joint_likelihood_rows",
    "query_joint_contribution_normalization_rows", "query_joint_typed_gate_statistics_rows",
    "query_routing_feature_normalization_rows", "query_joint_value_reduction_rows",
    "query_joint_field_head_rows", "query_routing_normalization_rows", "query_receiver_rows",
    "query_M_cheap_score_rows", "query_E_cheap_score_rows", "query_M_candidate_rows",
    "query_E_candidate_rows", "query_M_eligible_candidate_rows", "query_E_eligible_candidate_rows",
    "query_M_selected_value_rows", "query_E_selected_value_rows", "query_M_value_execution_rows",
    "query_E_value_execution_rows", "query_M_dense_candidate_value_rows",
    "query_E_dense_candidate_value_rows", "query_M_score_calibration_rows",
    "query_E_score_calibration_rows",
)


def compose_joint_member_prior(
    proposal: Tensor,
    retrieval: Tensor,
    raw_coefficients: Tensor,
    row_mass: Tensor,
    member_valid: Tensor,
    *,
    calibration: str = "normalized",
) -> tuple[Tensor, Tensor]:
    """Return normalized member prior P and its live-support mask.

    ``normalized`` composes the proposal with whole-row-normalized R. The
    ``row_strength`` diagnostic composes with the existing Hμ coefficients,
    retaining address row mass before a final joint normalization. Both paths
    produce only [B,Q,N] results; neither materializes a [B,Q,A,N] tensor.
    """
    if calibration not in _RETRIEVAL_CALIBRATIONS:
        raise ValueError(f"Unknown retrieval calibration: {calibration!r}.")
    if proposal.ndim != 3 or retrieval.ndim != 3 or raw_coefficients.shape != retrieval.shape:
        raise ValueError("Proposal and retrieval tensors must have [B,Q,A] and [B,A,N] axes.")
    if proposal.shape[0] != retrieval.shape[0] or proposal.shape[-1] != retrieval.shape[1]:
        raise ValueError("Proposal and retrieval address axes disagree.")
    if row_mass.shape != retrieval.shape[:2] or member_valid.shape != proposal.shape[:2] + (retrieval.shape[-1],):
        raise ValueError("Row mass or member validity does not match proposal/retrieval axes.")
    if member_valid.dtype != torch.bool:
        raise TypeError("member_valid must be boolean.")
    if calibration == "normalized":
        composed = torch.bmm(proposal, retrieval)
    else:
        composed = torch.bmm(proposal, raw_coefficients)
    if retrieval.shape[1] == 0 or retrieval.shape[2] == 0:
        support = torch.zeros_like(member_valid)
    else:
        support = torch.bmm((proposal > 0).to(proposal.dtype),
                            (retrieval > 0).to(proposal.dtype)) > 0
        support = support & member_valid
    prior = _normalize_nonnegative(composed, support)
    return prior, support


def joint_likelihood_contribution(
    prior: Tensor,
    scores: Tensor,
    allowed: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize ``P_i exp(ell_i)`` jointly with one stable query offset.

    The returned values are (b, log-weights, common offset). Eligible entries
    with exactly zero prior remain exactly zero. Empty member sets return zero
    mass without NaNs.
    """
    if prior.shape != scores.shape or prior.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("Prior, score and allowed tensors must have equal shapes and a boolean mask.")
    if not prior.is_floating_point() or not scores.is_floating_point():
        raise TypeError("Prior and scores must be floating point.")
    if not prior.shape[-1]:
        return torch.zeros_like(prior), scores.clone(), prior.new_zeros(prior.shape[:-1])
    valid = allowed & (prior > 0)
    log_prior = torch.where(valid, prior.clamp_min(torch.finfo(prior.dtype).tiny).log(),
                            torch.zeros_like(prior))
    log_weight = torch.where(valid, log_prior + scores, torch.full_like(scores, -torch.inf))
    has_mass = valid.any(-1, keepdim=True)
    maximum = log_weight.amax(-1, keepdim=True) if log_weight.shape[-1] else log_weight
    offset = torch.where(has_mass, maximum, torch.zeros_like(maximum))
    shifted = torch.where(valid, log_weight - offset, torch.zeros_like(log_weight))
    unnormalized = torch.exp(shifted) * valid.to(prior.dtype)
    total = unnormalized.sum(-1, keepdim=True)
    contribution = torch.where(total > 0,
                               unnormalized / total.clamp_min(torch.finfo(prior.dtype).tiny),
                               torch.zeros_like(unnormalized))
    return contribution, log_weight, offset.squeeze(-1)


def _applied_block_mass_trace(
    prepared: Any,
    proposal: Tensor,
    active_address_count: int,
    modules: int,
    environments: int,
    retrieval_calibration: str,
) -> dict[str, Tensor]:
    """Attribute query mass to original applied-H routes and M self access.

    Address weights reflect the selected calibration, while row contributions
    always come from prepared applied exchange/self tensors. This keeps a
    derived retrieval view's provenance separate from relation-stage trace.
    """
    if retrieval_calibration == "normalized":
        address_weights = proposal
        weighting_code = 0
    elif retrieval_calibration == "row_strength":
        address_weights = proposal * prepared.retrieval_row_mass[:, None, :active_address_count]
        total = address_weights.sum(-1, keepdim=True)
        address_weights = torch.where(
            total > 0,
            address_weights / total.clamp_min(torch.finfo(address_weights.dtype).tiny),
            torch.zeros_like(address_weights),
        )
        weighting_code = 1
    else:
        raise ValueError(f"Unknown retrieval calibration: {retrieval_calibration!r}.")

    zero = proposal.new_zeros(proposal.shape[:2])
    result: dict[str, Tensor] = {}
    for route, row_start, row_count, col_start, col_count in (
        ("MM", 0, modules, 0, modules),
        ("ME", 0, modules, modules, environments),
        ("EM", modules, environments, 0, modules),
        ("EE", modules, environments, modules, environments),
    ):
        active_rows = max(0, min(row_count, active_address_count - row_start))
        if active_rows == 0 or col_count == 0:
            result[f"address_block_mass_{route}"] = zero
            continue
        row_slice = slice(row_start, row_start + active_rows)
        col_slice = slice(col_start, col_start + col_count)
        raw_block = prepared.H_exchange[:, row_slice, col_slice]
        measure = prepared.member_measure_joint[:, col_slice]
        row_mass = prepared.retrieval_row_mass[:, row_slice]
        normalized_block = raw_block * measure[:, None, :]
        normalized_block = torch.where(
            row_mass[..., None] > 0,
            normalized_block / row_mass[..., None].clamp_min(torch.finfo(raw_block.dtype).tiny),
            torch.zeros_like(normalized_block),
        )
        result[f"address_block_mass_{route}"] = (
            address_weights[..., row_slice] * normalized_block.sum(-1)[:, None, :]
        ).sum(-1)

    active_module_rows = max(0, min(modules, active_address_count))
    if active_module_rows == 0:
        result["address_block_mass_self_M"] = zero
    else:
        row_slice = slice(0, active_module_rows)
        self_block = prepared.H_self[:, row_slice, :] * prepared.member_measure_joint[:, None, :]
        row_mass = prepared.retrieval_row_mass[:, row_slice]
        self_block = torch.where(
            row_mass[..., None] > 0,
            self_block / row_mass[..., None].clamp_min(torch.finfo(self_block.dtype).tiny),
            torch.zeros_like(self_block),
        )
        result["address_block_mass_self_M"] = (
            address_weights[..., row_slice] * self_block.sum(-1)[:, None, :]
        ).sum(-1)
    result["address_block_mass_provenance_code"] = torch.tensor(
        0, dtype=torch.long, device=proposal.device,
    )
    result["address_block_mass_weighting_code"] = torch.tensor(
        weighting_code, dtype=torch.long, device=proposal.device,
    )
    return result


def joint_role_statistics(
    contribution: Tensor,
    scores: Tensor,
    role_allowed: Tensor,
    scene_eligible_count: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return conditional p, E[ell|role], and scene-count-normalized entropy."""
    if contribution.shape != scores.shape or contribution.shape != role_allowed.shape or role_allowed.dtype != torch.bool:
        raise ValueError("Role statistics require equal contribution, score and boolean role axes.")
    if scene_eligible_count.ndim not in (1, 2) or scene_eligible_count.shape[0] != contribution.shape[0]:
        raise ValueError("scene_eligible_count must have [B] or [B,Q] axes.")
    role_mass = torch.where(role_allowed, contribution, torch.zeros_like(contribution)).sum(-1, keepdim=True)
    tiny = torch.finfo(contribution.dtype).tiny
    conditional = torch.where(role_mass > 0,
                              torch.where(role_allowed, contribution, torch.zeros_like(contribution)) /
                              role_mass.clamp_min(tiny),
                              torch.zeros_like(contribution))
    expectation = (conditional * scores).sum(-1)
    entropy = -(conditional * conditional.clamp_min(tiny).log()).sum(-1)
    count = scene_eligible_count.to(contribution)
    if count.ndim == 1:
        count = count[:, None].expand(-1, contribution.shape[1])
    normalized_entropy = torch.where(count > 1,
                                     entropy / count.clamp_min(2.).log(),
                                     torch.zeros_like(entropy))
    return conditional, expectation, normalized_entropy


class ConsistentJointMemberRoutingCore(JointMemberRoutingCore):
    """R13-5A with joint likelihood contribution and unchanged state identity."""

    schema = _SCHEMA
    _additional_decode_count_keys = JointMemberRoutingCore._additional_decode_count_keys + (
        "query_joint_likelihood_rows", "query_joint_contribution_normalization_rows",
        "query_joint_typed_gate_statistics_rows",
        "query_routing_feature_normalization_rows",
        "query_joint_full_eligible_rows",
    )

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
        retrieval_calibration: str = "normalized",
        score_calibration: str = "inherited",
    ):
        self._validate_calibrations(retrieval_calibration, score_calibration)
        self.retrieval_calibration = retrieval_calibration
        self.score_calibration = score_calibration
        super().__init__(
            base_model_config,
            address_mode=address_mode,
            input_dataset=input_dataset,
            input_capability=input_capability,
            address_temperature=address_temperature,
            joint_member_coverage=joint_member_coverage,
            query_tile_size=query_tile_size,
        )
        self.architecture = self._consistent_arm_name(address_mode)

    @staticmethod
    def _consistent_arm_name(address_mode: str) -> str:
        return f"R13-5A-{address_mode}-ConsistentJoint95"

    @staticmethod
    def _validate_calibrations(retrieval_calibration: str, score_calibration: str) -> None:
        if retrieval_calibration not in _RETRIEVAL_CALIBRATIONS:
            raise ValueError(f"retrieval_calibration must be one of {_RETRIEVAL_CALIBRATIONS}.")
        if score_calibration not in _SCORE_CALIBRATIONS:
            raise ValueError(f"score_calibration must be one of {_SCORE_CALIBRATIONS}.")

    @property
    def architecture_name(self) -> str:
        return self._consistent_arm_name(self.address_mode)

    def set_address_mode(self, address_mode: str) -> None:
        super().set_address_mode(address_mode)
        self.architecture = self._consistent_arm_name(address_mode)

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
            "retrieval_calibration": self.retrieval_calibration,
            "score_calibration": self.score_calibration,
        }

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        required = {
            "schema", "architecture", "address_mode", "base_model_config", "input_dataset",
            "input_capability", "address_temperature", "joint_member_coverage", "query_tile_size",
            "retrieval_calibration", "score_calibration",
        }
        if set(config) != required or config.get("schema") != cls.schema:
            raise ValueError("R13-5A consistent joint routing configuration inventory/schema differs.")
        mode = config["address_mode"]
        if mode not in ("M", "ME") or config["architecture"] != cls._consistent_arm_name(mode):
            raise ValueError("R13-5A architecture and address_mode disagree.")
        cls._validate_calibrations(config["retrieval_calibration"], config["score_calibration"])
        translated = dict(config)
        translated.pop("retrieval_calibration")
        translated.pop("score_calibration")
        translated["schema"] = JointMemberRoutingCore.schema
        translated["architecture"] = JointMemberRoutingCore._joint_arm_name(mode)
        JointMemberRoutingCore.validate_config(translated)

    def _retrieval_view(self, prepared, controls: Mapping[str, Any]):
        """Return read tensors and provenance; subclasses may supply a derived R."""
        return (prepared.retrieval_R, prepared.retrieval_raw_coefficients,
                prepared.retrieval_row_mass, "applied_H")

    def prepare(self, scene, **options):
        """Prepare normalized keys and scorer key terms with this live scene.

        These tensors are ordinary scene-local autograd values: they are not
        detached, stored on the model, or reused across separately prepared
        scenes. Validation below ties them to the exact raw key tensors.
        """
        prepared = super().prepare(scene, **options)
        prepared._consistent_normalized_member_keys_M = self._routing_features(prepared.member_keys_M)
        prepared._consistent_normalized_member_keys_E = self._routing_features(prepared.member_keys_E)
        width = self.context_dim
        scorer_specs = (
            ("proposal_M_M", self.member_router.proposal_M, "M"),
            ("proposal_M_E", self.member_router.proposal_M, "E"),
            ("member_M_M", self.member_router.member_M, "M"),
            ("member_E_E", self.member_router.member_E, "E"),
        )
        prepared._consistent_affine_key_terms = {}
        prepared._consistent_affine_key_versions = []
        for name, scorer, role in scorer_specs:
            keys = (prepared._consistent_normalized_member_keys_M if role == "M"
                    else prepared._consistent_normalized_member_keys_E)
            term = F.linear(keys, scorer[0].weight[:, width:2 * width])
            prepared._consistent_affine_key_terms[name] = term
            prepared._consistent_affine_key_versions.append(
                (name, _tensor_version(term), _tensor_version(scorer[0].weight)),
            )
        prepared._consistent_affine_key_versions = tuple(prepared._consistent_affine_key_versions)
        prepared._consistent_normalized_key_versions = (
            _tensor_version(prepared.member_keys_M),
            _tensor_version(prepared.member_keys_E),
            _tensor_version(prepared._consistent_normalized_member_keys_M),
            _tensor_version(prepared._consistent_normalized_member_keys_E),
        )
        prepared.work_counts.update({
            "normalized_member_key_rows_M": int(scene.batch_size * scene.num_modules),
            "normalized_member_key_rows_E": int(scene.batch_size * scene.num_environment_tokens),
        })
        return prepared

    def _validate_prepared_state(self, prepared):
        super()._validate_prepared_state(prepared)
        normalized_M = getattr(prepared, "_consistent_normalized_member_keys_M", None)
        normalized_E = getattr(prepared, "_consistent_normalized_member_keys_E", None)
        if normalized_M is None or normalized_E is None:
            raise ValueError("Prepared Candidate A scene lacks normalized member-key views; rebuild it.")
        current = (
            _tensor_version(prepared.member_keys_M),
            _tensor_version(prepared.member_keys_E),
            _tensor_version(normalized_M),
            _tensor_version(normalized_E),
        )
        if current != prepared._consistent_normalized_key_versions:
            raise ValueError("Prepared Candidate A normalized member-key views changed; rebuild the scene.")

        affine_terms = getattr(prepared, "_consistent_affine_key_terms", None)
        if not isinstance(affine_terms, dict):
            raise ValueError("Prepared Candidate A scene lacks affine scorer key terms; rebuild it.")
        scorer_specs = (
            ("proposal_M_M", self.member_router.proposal_M),
            ("proposal_M_E", self.member_router.proposal_M),
            ("member_M_M", self.member_router.member_M),
            ("member_E_E", self.member_router.member_E),
        )
        affine_versions = []
        for name, scorer in scorer_specs:
            term = affine_terms.get(name)
            if term is None:
                raise ValueError("Prepared Candidate A scene lacks affine scorer key terms; rebuild it.")
            affine_versions.append((name, _tensor_version(term), _tensor_version(scorer[0].weight)))
        if tuple(affine_versions) != prepared._consistent_affine_key_versions:
            raise ValueError("Prepared Candidate A affine scorer key terms changed; rebuild the scene.")

    def _factored_scorer_rows(
        self,
        prepared,
        query,
        normalized_query: Tensor,
        role: str,
        rows: Tensor,
        scorer_name: str,
        scorer,
    ) -> Tensor:
        """Apply the existing scorer after summing its live affine feature slices."""
        batch_indices, query_indices, member_indices = rows.unbind(-1)
        width = self.context_dim
        first = scorer[0]
        query_term = F.linear(normalized_query, first.weight[:, :width])
        geometry = self._pair_geometry(prepared.scene, query, role, rows)
        geometry_term = F.linear(geometry, first.weight[:, 2 * width:], first.bias)
        key_term = prepared._consistent_affine_key_terms[f"{scorer_name}_{role}"]
        hidden = (query_term[batch_indices, query_indices]
                  + key_term[batch_indices, member_indices]
                  + geometry_term)
        return scorer[2](scorer[1](hidden)).squeeze(-1)

    def _address_logits(self, prepared, query, qstates, address_allowed, *, normalized_query=None):
        """Score M/E addresses using scene-local normalized keys."""
        scene = prepared.scene
        batch, queries, address_count = address_allowed.shape
        modules, environments = scene.num_modules, scene.num_environment_tokens
        if address_count != modules + environments:
            raise ValueError("Joint address eligibility does not match the padded M/E inventories.")
        if normalized_query is None:
            normalized_query = self._routing_features(qstates)
        scores_m = qstates.new_zeros(batch, queries, modules)
        scores_e = qstates.new_zeros(batch, queries, environments)
        rows_m = self._pair_rows(address_allowed[..., :modules])
        if rows_m.numel():
            b, q, m = rows_m.unbind(-1)
            scores_m = scores_m.index_put(
                (b, q, m), self._factored_scorer_rows(
                    prepared, query, normalized_query, "M", rows_m,
                    "proposal_M", self.member_router.proposal_M,
                ),
            )
        rows_e = self._pair_rows(address_allowed[..., modules:])
        if rows_e.numel():
            b, q, e = rows_e.unbind(-1)
            scores_e = scores_e.index_put(
                (b, q, e), self._factored_scorer_rows(
                    prepared, query, normalized_query, "E", rows_e,
                    "proposal_M", self.member_router.proposal_M,
                ),
            )
        return torch.cat((scores_m, scores_e), dim=-1), scores_m, scores_e

    def _member_base_logits(
        self, prepared, query, qstates, source, candidates, score_mode, *, normalized_query=None,
    ):
        """Score selected members using the same tile query and scene key views."""
        if score_mode != "learned":
            return super()._member_base_logits(prepared, query, qstates, source, candidates, score_mode)
        keys = (prepared._consistent_normalized_member_keys_M if source == "M"
                else prepared._consistent_normalized_member_keys_E)
        output = qstates.new_zeros(query.batch_size, query.num_queries, keys.shape[1])
        rows = self._pair_rows(candidates)
        if not rows.numel():
            return output, 0
        if normalized_query is None:
            normalized_query = self._routing_features(qstates)
        scorer = self.member_router.member_M if source == "M" else self.member_router.member_E
        output = output.index_put(
            tuple(rows.unbind(-1)), self._factored_scorer_rows(
                prepared, query, normalized_query, source, rows,
                "member_M" if source == "M" else "member_E", scorer,
            ),
        )
        return output, len(rows)

    @staticmethod
    def _slice_intervention(intervention: Mapping[str, Any] | None, full_queries: int,
                            start: int, stop: int) -> dict[str, Any] | None:
        sliced = InteractionReusedMemberRoutingCore._slice_intervention(
            intervention, full_queries, start, stop,
        )
        if sliced is None:
            return None
        value = sliced.get("fixed_candidate_mask_joint")
        if value is not None:
            if not isinstance(value, Tensor) or value.ndim != 3:
                raise ValueError("fixed_candidate_mask_joint must be a rank-three [B,Q,M+E] tensor.")
            if value.shape[1] == full_queries:
                sliced["fixed_candidate_mask_joint"] = value[:, start:stop]
            elif full_queries > stop - start:
                raise ValueError(f"fixed_candidate_mask_joint must cover all {full_queries} native queries.")
        return sliced

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
        """Private extension seam for fixed-marginal organization controls."""
        contribution, log_weight, common_offset = joint_likelihood_contribution(
            member_prior, ell, candidate,
        )
        return contribution, log_weight, common_offset, {}

    def _joint_member_contribution_required(self, *, return_trace: bool) -> bool:
        """Whether this read law needs Candidate A's global normalization."""
        del return_trace
        return True

    def _role_statistics(
        self,
        contribution: Tensor,
        scores: Tensor,
        candidate: Tensor,
        scene_eligible_count: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Preserve the historical normalization unless a private read law overrides it."""
        return joint_role_statistics(contribution, scores, candidate, scene_eligible_count)

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
        """Return gate and value distributions for each typed read role.

        Candidate A uses one M+E contribution normalizer for both statistics
        and values. Subclasses can preserve those marginal gate statistics
        while changing only how role contributions are assembled.
        """
        del member_prior, ell, candidate, full_candidate, proposal, retrieval_R, row_mass, module_count
        if marginal_contribution is None:
            raise RuntimeError("Candidate A requires its joint member contribution.")
        return marginal_contribution, marginal_contribution, {}

    @staticmethod
    def _score_values(raw_scores: Tensor, prior: Tensor, candidate: Tensor, law: str) -> Tensor:
        if law == "inherited":
            conditional_prior = _normalize_nonnegative(prior, candidate)
            return prior_weighted_scores(raw_scores, conditional_prior, candidate)
        if law == "bounded_tanh":
            return 4. * torch.tanh(raw_scores / 4.) * candidate.to(raw_scores.dtype)
        raise ValueError(f"Unknown member score calibration: {law!r}.")

    def _typed_read_state(self, prepared, query, qstates, intervention, *, return_trace):
        fused, trace, counts, fields, _ = self._typed_read_state_with_plan(
            prepared, query, qstates, intervention, return_trace=return_trace,
        )
        return fused, trace, counts, fields

    def _typed_read_state_with_plan(
        self, prepared, query, qstates, intervention, *, return_trace, shared_plan=None,
    ):
        scene = prepared.scene
        controls = {} if intervention is None else dict(intervention)
        legal = {
            "read_mode", "member_score_mode", "value_execution", "fixed_pi", "fixed_address_mask",
            "fixed_candidate_mask_joint", "fixed_member_logits_M", "fixed_member_logits_E",
            "retrieval_mode", "address_coverage", "retrieval_calibration", "score_calibration",
        }
        unknown = set(controls) - legal
        if unknown:
            raise ValueError(f"Unknown R13-5A routing interventions: {sorted(unknown)}")
        mode = controls.get("read_mode", self._read_execution_override or "full")
        if mode not in ("full", "restricted"):
            raise ValueError("R13-5A read_mode must be 'full' or 'restricted'.")
        execution = controls.get("value_execution", "selected")
        if execution not in ("selected", "dense_candidate_union"):
            raise ValueError("value_execution must be selected or dense_candidate_union.")
        member_score_mode = controls.get("member_score_mode", "learned")
        if member_score_mode not in ("learned", "geometry"):
            raise ValueError("member_score_mode must be learned or geometry.")
        retrieval_calibration = controls.get("retrieval_calibration", self.retrieval_calibration)
        if "retrieval_mode" in controls:
            legacy_mode = controls["retrieval_mode"]
            translated = {"normalized": "normalized", "raw": "row_strength"}.get(legacy_mode)
            if translated is None:
                raise ValueError("retrieval_mode must be normalized or raw.")
            if "retrieval_calibration" in controls:
                raise ValueError("Specify only one retrieval calibration override.")
            retrieval_calibration = translated
        score_calibration = controls.get("score_calibration", self.score_calibration)
        self._validate_calibrations(retrieval_calibration, score_calibration)
        address_coverage = controls.get("address_coverage")
        if address_coverage is not None and (
                isinstance(address_coverage, bool) or
                not isinstance(address_coverage, (int, float)) or
                not 0 < float(address_coverage) <= 1):
            raise ValueError("address_coverage must be None or a number in (0,1].")

        batch, queries = query.batch_size, query.num_queries
        modules, environments = scene.num_modules, scene.num_environment_tokens
        total = modules + environments
        if shared_plan is None:
            query_mask = self._query_mask(query)
            normalized_query = self._routing_features(qstates)
            eligible_roles = torch.cat((scene.module_mask,
                                        scene.environment_mask & (scene.environment_measure > 0)), dim=-1)
            address_eligible = prepared.joint_address_eligible & eligible_roles
            address_mask = address_eligible.clone()
            if self.address_mode == "M":
                address_mask[:, modules:] = False
            fixed_address_mask = controls.get("fixed_address_mask")
            if fixed_address_mask is not None:
                if fixed_address_mask.shape != address_mask.shape or fixed_address_mask.dtype != torch.bool:
                    raise ValueError("fixed_address_mask must be boolean [B,M+E].")
                if bool((fixed_address_mask & ~address_mask).any()):
                    raise ValueError("fixed_address_mask cannot activate an ineligible address.")
                address_mask = fixed_address_mask
            address_allowed = address_mask[:, None, :].expand(batch, queries, total) & query_mask[..., None]
            address_scores, address_scores_M, address_scores_E = self._address_logits(
                prepared, query, qstates, address_allowed, normalized_query=normalized_query,
            )
            address_score_rows_M = address_allowed[..., :modules].detach().sum(dtype=torch.long)
            address_score_rows_E = address_allowed[..., modules:].detach().sum(dtype=torch.long)
            address_omega, role_allowed = self._address_base_measure(scene, address_allowed, self.address_mode)
            address_logits = address_scores / self.address_temperature
            log_measure = torch.where(role_allowed,
                                      address_omega.clamp_min(torch.finfo(address_omega.dtype).tiny).log(),
                                      torch.zeros_like(address_omega))
            proposal_logits = address_logits + log_measure
            pi = _masked_softmax(proposal_logits, role_allowed)
            fixed_pi = controls.get("fixed_pi")
            if fixed_pi is not None:
                if fixed_pi.shape != proposal_logits.shape or fixed_pi.device != proposal_logits.device:
                    raise ValueError("fixed_pi must have shape [B,Q,M+E] on the query device.")
                invalid = ~torch.isfinite(fixed_pi) | (fixed_pi < 0) | ((fixed_pi > 0) & ~role_allowed)
                has_address = role_allowed.any(-1)
                fixed_total = fixed_pi.sum(-1)
                if bool(invalid.any()) or bool(((has_address & ((fixed_total - 1.).abs() > 1.e-4)) |
                                                 (~has_address & (fixed_total > 1.e-6))).any()):
                    raise ValueError("fixed_pi must be finite, eligible, and sum to one when nonempty.")
                pi = fixed_pi.to(qstates)

            retrieval_R, retrieval_raw, row_mass, retrieval_provenance = self._retrieval_view(prepared, controls)
            if retrieval_R.shape != (batch, total, total) or retrieval_raw.shape != retrieval_R.shape:
                raise ValueError("The selected retrieval view must preserve the prepared [B,A,N] axes.")
            if row_mass.shape != (batch, total):
                raise ValueError("The selected retrieval row mass must preserve the prepared [B,A] axes.")
            pi_uncut = pi
            address_cut_mask = role_allowed
            address_transport_mass = None
            address_omitted_mass = None
            address_omitted_fraction = None
            if address_coverage is not None:
                address_transport_mass = (pi_uncut if retrieval_calibration == "normalized"
                                          else pi_uncut * row_mass[:, None, :])
                address_cut_mask = joint_member_prefix_mask(
                    address_transport_mass, role_allowed, scene.module_ids, scene.environment_ids,
                    float(address_coverage),
                )
                pi = _normalize_nonnegative(pi_uncut, address_cut_mask)
                total_transport = (address_transport_mass * role_allowed.to(address_transport_mass.dtype)).sum(-1)
                kept_transport = (address_transport_mass * address_cut_mask.to(address_transport_mass.dtype)).sum(-1)
                address_omitted_mass = (total_transport - kept_transport).clamp_min(0.)
                address_omitted_fraction = torch.where(
                    total_transport > 0,
                    address_omitted_mass / total_transport.clamp_min(torch.finfo(total_transport.dtype).tiny),
                    torch.zeros_like(total_transport),
                )

            active_address_count = modules if self.address_mode == "M" else total
            active_pi = pi[..., :active_address_count]
            active_pi_uncut = pi_uncut[..., :active_address_count]
            active_R = retrieval_R[:, :active_address_count]
            active_raw = retrieval_raw[:, :active_address_count]
            member_valid = eligible_roles[:, None, :] & query_mask[..., None]
            member_prior, member_support = compose_joint_member_prior(
                active_pi, active_R, active_raw, row_mass[:, :active_address_count], member_valid,
                calibration=retrieval_calibration,
            )
            full_candidate = member_support
            if address_coverage is not None:
                _, full_candidate = compose_joint_member_prior(
                    active_pi_uncut, active_R, active_raw, row_mass[:, :active_address_count], member_valid,
                    calibration=retrieval_calibration,
                )
            member_prior_uncut = member_prior
            if return_trace and address_coverage is not None:
                member_prior_uncut, _ = compose_joint_member_prior(
                    active_pi_uncut, active_R, active_raw, row_mass[:, :active_address_count], member_valid,
                    calibration=retrieval_calibration,
                )
            shared_plan = {
                "prepared_id": id(prepared), "query_id": id(query), "qstates_id": id(qstates),
                "query_mask": query_mask, "normalized_query": normalized_query,
                "eligible_roles": eligible_roles, "address_allowed": address_allowed,
                "address_scores": address_scores, "address_scores_M": address_scores_M,
                "address_scores_E": address_scores_E, "address_score_rows_M": address_score_rows_M,
                "address_score_rows_E": address_score_rows_E, "address_omega": address_omega,
                "role_allowed": role_allowed, "address_logits": address_logits,
                "proposal_logits": proposal_logits, "pi": pi, "pi_uncut": pi_uncut,
                "address_cut_mask": address_cut_mask, "address_transport_mass": address_transport_mass,
                "address_omitted_mass": address_omitted_mass,
                "address_omitted_fraction": address_omitted_fraction,
                "retrieval_R": retrieval_R, "retrieval_raw": retrieval_raw, "row_mass": row_mass,
                "retrieval_provenance": retrieval_provenance,
                "active_address_count": active_address_count, "active_pi": active_pi,
                "active_pi_uncut": active_pi_uncut, "active_R": active_R, "active_raw": active_raw,
                "member_valid": member_valid, "member_prior": member_prior,
                "member_prior_uncut": member_prior_uncut, "member_support": member_support,
                "full_candidate": full_candidate,
                "routing_normalization_shared_rows": (
                    role_allowed.detach().sum(dtype=torch.long) + batch * queries * total
                ),
            }
        else:
            if (shared_plan["prepared_id"] != id(prepared) or shared_plan["query_id"] != id(query) or
                    shared_plan["qstates_id"] != id(qstates)):
                raise ValueError("A shared Candidate A query plan is valid only for its originating live tile.")
        query_mask = shared_plan["query_mask"]
        normalized_query = shared_plan["normalized_query"]
        eligible_roles = shared_plan["eligible_roles"]
        address_allowed = shared_plan["address_allowed"]
        address_scores = shared_plan["address_scores"]
        address_scores_M = shared_plan["address_scores_M"]
        address_scores_E = shared_plan["address_scores_E"]
        address_score_rows_M = shared_plan["address_score_rows_M"]
        address_score_rows_E = shared_plan["address_score_rows_E"]
        address_omega = shared_plan["address_omega"]
        role_allowed = shared_plan["role_allowed"]
        address_logits = shared_plan["address_logits"]
        proposal_logits = shared_plan["proposal_logits"]
        pi = shared_plan["pi"]
        pi_uncut = shared_plan["pi_uncut"]
        address_cut_mask = shared_plan["address_cut_mask"]
        address_transport_mass = shared_plan["address_transport_mass"]
        address_omitted_mass = shared_plan["address_omitted_mass"]
        address_omitted_fraction = shared_plan["address_omitted_fraction"]
        retrieval_R = shared_plan["retrieval_R"]
        retrieval_raw = shared_plan["retrieval_raw"]
        row_mass = shared_plan["row_mass"]
        retrieval_provenance = shared_plan["retrieval_provenance"]
        active_address_count = shared_plan["active_address_count"]
        active_pi = shared_plan["active_pi"]
        active_pi_uncut = shared_plan["active_pi_uncut"]
        active_R = shared_plan["active_R"]
        active_raw = shared_plan["active_raw"]
        member_valid = shared_plan["member_valid"]
        member_prior = shared_plan["member_prior"]
        member_prior_uncut = shared_plan["member_prior_uncut"]
        member_support = shared_plan["member_support"]
        full_candidate = shared_plan["full_candidate"]
        if mode == "restricted":
            joint_candidate = joint_member_prefix_mask(
                member_prior, full_candidate, scene.module_ids, scene.environment_ids,
                self.joint_member_coverage,
            )
        else:
            joint_candidate = full_candidate
        fixed_candidate = controls.get("fixed_candidate_mask_joint")
        if fixed_candidate is not None:
            if fixed_candidate.shape != joint_candidate.shape or fixed_candidate.dtype != torch.bool:
                raise ValueError("fixed_candidate_mask_joint must be boolean [B,Q,M+E].")
            if bool((fixed_candidate & ~full_candidate).any()):
                raise ValueError("Fixed joint candidates must remain reachable through a live retrieval path.")
            joint_candidate = fixed_candidate
        candidate_M = joint_candidate[..., :modules]
        candidate_E = joint_candidate[..., modules:]

        member_logits: dict[str, Tensor] = {}
        ell_by_role: dict[str, Tensor] = {}
        scored_rows_by_role: dict[str, int] = {}
        for role, candidate in (("M", candidate_M), ("E", candidate_E)):
            raw_scores, scored_rows = self._member_base_logits(
                prepared, query, qstates, role, candidate, member_score_mode,
                normalized_query=normalized_query,
            )
            fixed_scores = controls.get(f"fixed_member_logits_{role}")
            if fixed_scores is not None:
                if (fixed_scores.shape != raw_scores.shape or fixed_scores.device != raw_scores.device or
                        bool((~torch.isfinite(fixed_scores)).any())):
                    raise ValueError(f"fixed_member_logits_{role} must be finite and preserve [B,Q,N] axes.")
                raw_scores = fixed_scores.to(qstates)
            member_logits[role] = raw_scores
            prior_role = member_prior[..., :modules] if role == "M" else member_prior[..., modules:]
            ell_by_role[role] = self._score_values(raw_scores, prior_role, candidate, score_calibration)
            scored_rows_by_role[role] = scored_rows

        # The inherited diagnostic exactly reproduces R13-5's candidate-
        # dependent per-role calibration. The candidate-independent law is
        # applied to the same raw feature-normalized scores without centering.
        # Candidate A uses their common M+E normalizer; O/H request it only
        # for the legacy diagnostic trace.
        ell = torch.cat((ell_by_role["M"], ell_by_role["E"]), dim=-1)
        compute_joint_contribution = self._joint_member_contribution_required(return_trace=return_trace)
        if compute_joint_contribution:
            contribution, log_weight, common_offset, law_trace = self._joint_member_contribution(
                member_prior, ell, joint_candidate,
                proposal=active_pi,
                retrieval_R=active_R,
                row_mass=row_mass[:, :active_address_count],
                retrieval_calibration=retrieval_calibration,
            )
        else:
            contribution = None
            log_weight = None
            common_offset = None
            law_trace = {}
        gate_contribution, value_contribution, read_law_trace = self._role_read_contributions(
            contribution,
            member_prior,
            ell,
            joint_candidate,
            full_candidate=full_candidate,
            proposal=active_pi,
            retrieval_R=active_R,
            row_mass=row_mass[:, :active_address_count],
            module_count=modules,
        )
        if (gate_contribution.shape != ell.shape or value_contribution.shape != ell.shape
                or gate_contribution.device != ell.device or value_contribution.device != ell.device
                or gate_contribution.dtype != ell.dtype or value_contribution.dtype != ell.dtype):
            raise ValueError("Role read contributions must preserve member-score shape, device and dtype.")
        role_candidates = {"M": candidate_M, "E": candidate_E}
        role_offsets = {"M": (0, modules), "E": (modules, total)}
        contexts: dict[str, Tensor] = {}
        role_states: dict[str, dict[str, Tensor]] = {}
        for role, values, eligible_scene in (
            ("M", prepared.member_values_M, scene.module_mask),
            ("E", prepared.member_values_E, scene.environment_mask & (scene.environment_measure > 0)),
        ):
            start, stop = role_offsets[role]
            candidate = role_candidates[role]
            b_role = value_contribution[..., start:stop]
            gate_role = gate_contribution[..., start:stop]
            ell_role = ell[..., start:stop]
            p_role, expectation, entropy = self._role_statistics(
                gate_role, ell_role, candidate,
                eligible_scene.sum(-1),
            )
            gate_inputs = torch.cat((normalized_query, expectation[..., None], entropy[..., None]), -1)
            gate_logits = self.typed_read_gates[role](gate_inputs)
            contrast = gate_logits[..., 0] - gate_logits[..., 1]
            has_members = candidate.any(-1)
            strength = (read_probability(contrast) * has_members.to(qstates) * query_mask.to(qstates))
            beta = strength[..., None] * b_role
            execute = candidate if execution == "dense_candidate_union" else (candidate & (beta > 0))
            context = self._reduce_selected(values, beta, execute)
            contexts[role] = context
            role_state = {
                "candidate": candidate,
                "raw_scores": member_logits[role],
                "ell": ell_role,
                "gate_contribution": gate_role,
                "pre_strength": b_role,
                "conditional": p_role,
                "expectation": expectation,
                "entropy": entropy,
                "gate_logits": gate_logits,
                "contrast": contrast,
                "strength": strength,
                "post_strength": beta,
                "context": context,
                "role_mass": b_role.sum(-1),
            }
            if return_trace:
                prior_role = member_prior[..., start:stop]
                role_state["conditional_prior"] = _normalize_nonnegative(prior_role, candidate)
            role_states[role] = role_state

        fused = torch.cat((qstates, contexts["M"], contexts["E"]), dim=-1)
        fields = self.member_router.joint_head(self.member_router.joint_trunk(fused))
        fields = fields * query_mask[..., None].to(fields.dtype)

        address_pairs = int(batch * queries * active_address_count * total)
        prefix_sort_rows = int(batch * queries * total) if mode == "restricted" else 0
        address_prefix_rows = int(batch * total) if address_coverage is not None else 0
        candidate_counts = {
            "M": candidate_M.detach().sum(dtype=torch.long),
            "E": candidate_E.detach().sum(dtype=torch.long),
        }
        eligible_counts = {
            "M": full_candidate[..., :modules].detach().sum(dtype=torch.long),
            "E": full_candidate[..., modules:].detach().sum(dtype=torch.long),
        }
        selected_counts = {
            role: (role_states[role]["post_strength"].detach() > 0).sum(dtype=torch.long)
            for role in ("M", "E")
        }
        value_execution_counts = {
            role: (candidate_counts[role] if execution == "dense_candidate_union" else
                   (role_candidates[role] & (role_states[role]["post_strength"].detach() > 0)).sum(dtype=torch.long))
            for role in ("M", "E")
        }
        joint_candidate_count = candidate_counts["M"] + candidate_counts["E"]
        full_candidate_count = eligible_counts["M"] + eligible_counts["E"]
        role_allowed_count = role_allowed.detach().sum(dtype=torch.long)
        address_relation_count = (
            (active_pi.detach() > 0).to(torch.long) *
            (active_R.detach() > 0).sum(-1).to(torch.long)[:, None, :]
        ).sum(dtype=torch.long)
        counts: dict[str, int | Tensor] = {
            "query_M_proposal_score_rows": address_score_rows_M,
            "query_E_proposal_score_rows": address_score_rows_E,
            "query_joint_address_score_rows": address_score_rows_M + address_score_rows_E,
            "query_joint_address_base_measure_rows": role_allowed_count,
            "query_joint_address_prefix_sort_rows": address_prefix_rows,
            "query_joint_address_prefix_id_sort_rows": int(batch * total) if address_coverage is not None else 0,
            "query_address_softmax_rows": role_allowed_count,
            "query_address_index_scan_rows": 0,
            "query_address_coefficient_pair_rows": address_pairs,
            "query_address_support_pair_rows": address_pairs,
            "query_address_relation_rows": address_relation_count,
            "query_typed_read_gate_rows": 2 * batch * queries,
            "query_M_prior_composition_rows": batch * queries * active_address_count * modules,
            "query_E_prior_composition_rows": batch * queries * active_address_count * environments,
            "query_M_prior_normalization_rows": batch * queries * modules,
            "query_E_prior_normalization_rows": batch * queries * environments,
            "query_M_continuous_value_rows": candidate_counts["M"],
            "query_E_continuous_value_rows": candidate_counts["E"],
            # Candidate A uses one M+E normalization rather than two role
            # softmaxes. Keep the legacy per-role operation counters at zero.
            "query_M_member_softmax_rows": 0,
            "query_E_member_softmax_rows": 0,
            "query_M_sorted_rows": 0,
            "query_E_sorted_rows": 0,
            "query_M_id_sort_rows": 0,
            "query_E_id_sort_rows": 0,
            "query_E_prior_prefix_sort_rows": 0,
            "query_E_prior_id_sort_rows": 0,
            "query_joint_prefix_sort_rows": prefix_sort_rows,
            "query_joint_prefix_id_sort_rows": int(batch * total) if mode == "restricted" else 0,
            "query_joint_full_eligible_rows": full_candidate_count,
            "query_joint_member_coverage_rows": joint_candidate_count,
            "query_joint_raw_composition_pair_rows": address_pairs if retrieval_calibration == "row_strength" else 0,
            "query_joint_uncut_composition_pair_rows": address_pairs if address_coverage is not None else 0,
            "query_joint_likelihood_rows": joint_candidate_count if compute_joint_contribution else 0,
            "query_joint_contribution_normalization_rows": batch * queries * total if compute_joint_contribution else 0,
            "query_joint_typed_gate_statistics_rows": joint_candidate_count,
            "query_routing_feature_normalization_rows": batch * queries,
            "query_joint_value_reduction_rows": joint_candidate_count,
            "query_joint_field_head_rows": batch * queries,
            "query_routing_normalization_rows": (
                role_allowed_count + batch * queries * total +
                (joint_candidate_count if compute_joint_contribution else 0)
            ),
            "query_receiver_rows": batch * queries,
        }
        for role in ("M", "E"):
            state = role_states[role]
            candidate = state["candidate"]
            beta = state["post_strength"]
            counts.update({
                f"query_{role}_cheap_score_rows": scored_rows_by_role[role],
                f"query_{role}_candidate_rows": candidate_counts[role],
                f"query_{role}_eligible_candidate_rows": eligible_counts[role],
                f"query_{role}_selected_value_rows": selected_counts[role],
                f"query_{role}_value_execution_rows": value_execution_counts[role],
                f"query_{role}_dense_candidate_value_rows": candidate_counts[role] if execution == "dense_candidate_union" else 0,
                f"query_{role}_score_calibration_rows": candidate_counts[role],
                f"query_{role}_continuous_value_rows": candidate_counts[role],
            })

        trace: dict[str, Tensor] = {}
        if return_trace:
            if address_coverage is None:
                transport_mass = pi_uncut if retrieval_calibration == "normalized" else pi_uncut * row_mass[:, None, :]
                omitted_mass = torch.zeros_like(pi.sum(-1))
                omitted_fraction = torch.zeros_like(pi.sum(-1))
            else:
                transport_mass = address_transport_mass
                omitted_mass = address_omitted_mass
                omitted_fraction = address_omitted_fraction
            role_codes = torch.cat((torch.zeros(modules, dtype=torch.long, device=qstates.device),
                                    torch.ones(environments, dtype=torch.long, device=qstates.device)))
            trace.update({
                "proposal_logits_joint": proposal_logits,
                "proposal_scores_joint": address_scores,
                "proposal_raw_scores_joint": address_scores,
                "proposal_logits_M": address_scores_M,
                "proposal_logits_E": address_scores_E,
                "proposal_temperature_scaled_scores_joint": address_logits,
                "proposal_base_measure": address_omega,
                "proposal_weights_joint": pi,
                "pi_joint": pi,
                "address_weights": pi,
                "proposal_weights_joint_uncut": pi_uncut,
                "pi_joint_uncut": pi_uncut,
                "proposal_weights_M": pi[..., :modules],
                "proposal_weights_E": pi[..., modules:],
                "proposal_base_measure_M": address_omega[..., :modules],
                "proposal_base_measure_E": address_omega[..., modules:],
                "address_role_codes": role_codes,
                "address_original_ids": torch.cat((scene.module_ids, scene.environment_ids), dim=-1),
                "address_eligible_mask": role_allowed,
                "address_candidate_mask": address_cut_mask,
                "address_transport_mass": transport_mass,
                "address_omitted_transport_mass": omitted_mass,
                "address_omitted_transport_fraction": omitted_fraction,
                "address_coverage": qstates.new_tensor(-1. if address_coverage is None else float(address_coverage)),
                "prior_P": member_prior,
                "prior_P_joint": member_prior,
                "prior_P_uncut": member_prior_uncut,
                "full_eligible_mask": full_candidate,
                "joint_candidate_mask": joint_candidate,
                "candidate_mask_joint": joint_candidate,
                "full_eligible_mask_M": full_candidate[..., :modules],
                "full_eligible_mask_E": full_candidate[..., modules:],
                "candidate_mask_M": candidate_M,
                "candidate_mask_E": candidate_E,
                "prior_P_M": member_prior[..., :modules],
                "prior_P_E": member_prior[..., modules:],
                "original_module_ids": scene.module_ids,
                "original_environment_ids": scene.environment_ids,
                "member_measure_joint": prepared.member_measure_joint,
                "retrieval_row_mass": prepared.retrieval_row_mass,
                "retrieval_row_mass_read": row_mass,
                "retrieval_R": prepared.retrieval_R,
                "retrieval_R_read": retrieval_R,
                "retrieval_raw_coefficients": prepared.retrieval_raw_coefficients,
                "retrieval_raw_coefficients_read": retrieval_raw,
                "retrieval_provenance": torch.tensor(0 if retrieval_provenance == "applied_H" else 1,
                                                     device=qstates.device),
                "H": prepared.H,
                "H_exchange": prepared.H_exchange,
                "H_self": prepared.H_self,
                "H_self_M": prepared.H_self_M,
                "H_MM_applied": prepared.H_MM_applied,
                "H_EM_applied": prepared.H_EM_applied,
                "H_EE_applied": prepared.H_EE_applied,
                "H_ME_applied": prepared.H_ME_applied,
                "H_route_cycle_indices": prepared.H_route_cycle_indices,
                "H_route_provenance_codes": torch.tensor([0, 1, 2, 3], dtype=torch.long, device=qstates.device),
                "member_prior_calibration_code": torch.tensor(0 if retrieval_calibration == "normalized" else 1,
                                                              dtype=torch.long, device=qstates.device),
                "member_score_calibration_code": torch.tensor(0 if score_calibration == "inherited" else 1,
                                                              dtype=torch.long, device=qstates.device),
                "member_joint_log_weight": log_weight,
                "member_joint_log_weight_offset": common_offset,
                "member_joint_likelihood": gate_contribution,
                "member_joint_likelihood_M": gate_contribution[..., :modules],
                "member_joint_likelihood_E": gate_contribution[..., modules:],
                "member_contribution_pre_strength_M": value_contribution[..., :modules],
                "member_contribution_pre_strength_E": value_contribution[..., modules:],
                "member_contribution_M": role_states["M"]["post_strength"],
                "member_contribution_E": role_states["E"]["post_strength"],
                "candidate_ids_M": self._candidate_id_rows(candidate_M, scene.module_ids),
                "candidate_ids_E": self._candidate_id_rows(candidate_E, scene.environment_ids),
                "prior_role_mass_M": member_prior[..., :modules].sum(-1),
                "prior_role_mass_E": member_prior[..., modules:].sum(-1),
                "retained_joint_mass_M": (member_prior[..., :modules] * candidate_M).sum(-1),
                "retained_joint_mass_E": (member_prior[..., modules:] * candidate_E).sum(-1),
                "context_M": contexts["M"],
                "context_E": contexts["E"],
            })
            trace.update(law_trace)
            trace.update(read_law_trace)
            trace.update(_applied_block_mass_trace(
                prepared, active_pi, active_address_count, modules, environments, retrieval_calibration,
            ))
            for role, state in role_states.items():
                trace.update({
                    f"member_logits_{role}": state["raw_scores"],
                    f"member_normalized_scores_{role}": state["ell"],
                    f"member_ell_{role}": state["ell"],
                    f"member_conditional_likelihood_{role}": state["conditional"],
                    f"member_conditional_expectation_{role}": state["expectation"],
                    f"member_conditional_entropy_{role}": state["entropy"],
                    f"member_conditional_prior_{role}": state["conditional_prior"],
                    f"member_relative_weights_{role}": state["conditional"],
                    f"member_read_strength_{role}": state["strength"],
                    f"member_read_logits_{role}": state["gate_logits"],
                    f"member_read_contrast_{role}": state["contrast"],
                    f"member_gate_derivative_{role}": 0.5 / torch.hypot(
                        state["contrast"], torch.ones_like(state["contrast"]),
                    ).pow(3),
                    f"member_contribution_pre_strength_{role}": state["pre_strength"],
                    f"member_contribution_{role}": state["post_strength"],
                })
            for block, provenance in prepared.H_route_provenance.items():
                trace[f"H_{block}_destination_ids"] = provenance["destination_ids"]
                trace[f"H_{block}_donor_ids"] = provenance["donor_ids"]
        return fused, trace, counts, fields, shared_plan

    @staticmethod
    def _paired_count_names(extra_count_names=()) -> tuple[str, ...]:
        """Keep Candidate A and derived-subclass work keys in paired outputs."""
        return tuple(dict.fromkeys((*_PAIRED_COUNT_KEYS, *extra_count_names)))

    @staticmethod
    def _split_paired_counts(
        full_counts: Mapping[str, int | Tensor],
        restricted_counts: Mapping[str, int | Tensor],
        shared_plan: Mapping[str, Any],
        count_names: tuple[str, ...],
    ) -> tuple[dict[str, int | Tensor], dict[str, int | Tensor], dict[str, int | Tensor]]:
        shared = {name: full_counts.get(name, 0) for name in _PAIRED_SHARED_COUNT_KEYS}
        shared_normalization = shared_plan["routing_normalization_shared_rows"]
        shared["query_routing_normalization_rows"] = shared_normalization

        def branch_counts(counts: Mapping[str, int | Tensor]) -> dict[str, int | Tensor]:
            branch = {name: counts.get(name, 0) for name in count_names}
            for name in _PAIRED_SHARED_COUNT_KEYS:
                branch.pop(name, None)
            branch["query_routing_normalization_rows"] = (
                counts.get("query_routing_normalization_rows", 0) - shared_normalization
            )
            return branch

        return branch_counts(full_counts), branch_counts(restricted_counts), shared

    def decode_paired(
        self,
        organization,
        query,
        controls=None,
        *,
        intervention: Mapping[str, Any] | None = None,
        return_trace: bool = False,
        trace_request=None,
        **kwargs,
    ) -> tuple[InteractionJointOutput, InteractionJointOutput, dict[str, int]]:
        """Decode full and restricted readers from one live common tile plan.

        Returns ``(full, restricted, shared_counts)``. Each output owns only its
        branch-exclusive work; common address/support work is reported once in
        ``shared_counts``. The checkpoint closure captures both read modes and
        returns both field tensors, so backward traverses the common plan once.
        """
        if controls is not None:
            raise ValueError("R13-5A paired decoding accepts no affine controls.")
        if kwargs:
            allowed = {"return_basis", "accumulation_dtype", "edge_gates", "uniform_access", "local_gates"}
            unknown = sorted(set(kwargs) - allowed)
            if unknown:
                raise TypeError(f"Unsupported R13-5A paired decode options: {unknown}.")
            if any(kwargs.get(name) is not None and name != "uniform_access" for name in kwargs):
                raise ValueError("Historical affine, edge and local gates are not R13-5A paired controls.")
        if intervention is not None and "read_mode" in intervention:
            raise ValueError("decode_paired fixes its two read modes; do not supply read_mode.")
        shared_intervention = {} if intervention is None else dict(intervention)
        full_intervention = {**shared_intervention, "read_mode": "full"}
        restricted_intervention = {**shared_intervention, "read_mode": "restricted"}
        self.validate_prepared_state(organization, query)
        if trace_request is not None:
            return_trace = True
        if return_trace and query.num_queries > self.query_tile_size:
            raise ValueError("Paired routing traces require an explicit query slice no larger than query_tile_size.")

        full_fields_by_tile: list[Tensor] = []
        restricted_fields_by_tile: list[Tensor] = []
        full_trace = None
        restricted_trace = None
        full_counts: dict[str, int] = {}
        restricted_counts: dict[str, int] = {}
        shared_counts: dict[str, int] = {}
        packed_count_total = None
        count_names = self._paired_count_names(getattr(self, "_additional_decode_count_keys", ()))
        shared_names = (*_PAIRED_SHARED_COUNT_KEYS, "query_routing_normalization_rows")
        for start in range(0, max(query.num_queries, 1), self.query_tile_size):
            stop = min(start + self.query_tile_size, query.num_queries)
            if query.num_queries == 0:
                break
            tile = query if query.num_queries <= self.query_tile_size else slice_query_v1(query, start, stop)
            full_controls = self._slice_intervention(full_intervention, query.num_queries, start, stop)
            restricted_controls = self._slice_intervention(
                restricted_intervention, query.num_queries, start, stop,
            )

            def run_tile(coordinates, *, tile=tile, full_controls=full_controls,
                         restricted_controls=restricted_controls):
                local_query = replace(tile, coordinates=coordinates)
                local_qstates = self._query_states(organization, local_query)
                _, full_local_trace, full_local_counts, full_local_fields, plan = (
                    self._typed_read_state_with_plan(
                        organization, local_query, local_qstates, full_controls,
                        return_trace=return_trace,
                    )
                )
                _, restricted_local_trace, restricted_local_counts, restricted_local_fields, _ = (
                    self._typed_read_state_with_plan(
                        organization, local_query, local_qstates, restricted_controls,
                        return_trace=return_trace, shared_plan=plan,
                    )
                )
                local_full_counts, local_restricted_counts, local_shared_counts = self._split_paired_counts(
                    full_local_counts, restricted_local_counts, plan, count_names,
                )
                shared_differences = []
                for name in _PAIRED_SHARED_COUNT_KEYS:
                    full_value = full_local_counts.get(name, 0)
                    restricted_value = restricted_local_counts.get(name, 0)
                    if isinstance(full_value, Tensor) != isinstance(restricted_value, Tensor):
                        raise RuntimeError(f"Paired Candidate A count type disagrees for {name}.")
                    if isinstance(full_value, Tensor):
                        shared_differences.append((full_value.detach() - restricted_value.detach()).abs())
                    elif full_value != restricted_value:
                        raise RuntimeError(f"Paired Candidate A shared work disagrees for {name}.")
                shared_normalization = plan["routing_normalization_shared_rows"]
                normalization_underflows = [
                    (shared_normalization - local_counts["query_routing_normalization_rows"]).clamp_min(0)
                    for local_counts in (full_local_counts, restricted_local_counts)
                ]
                paired_invariant_counts = torch.stack((*shared_differences, *normalization_underflows))
                local_count_values = (
                    tuple(local_full_counts.get(name, 0) for name in count_names),
                    tuple(local_restricted_counts.get(name, 0) for name in count_names),
                    tuple(local_shared_counts.get(name, 0) for name in shared_names),
                )
                return (full_local_fields, restricted_local_fields, local_count_values, paired_invariant_counts,
                        full_local_trace, restricted_local_trace)

            if self.training and torch.is_grad_enabled() and not return_trace:
                tile_runner = run_tile
                (full_tile_fields, restricted_tile_fields, local_count_values,
                 paired_invariant_counts) = checkpoint(
                    lambda coordinates, tile_runner=tile_runner: tile_runner(coordinates)[:4],
                    tile.coordinates, use_reentrant=False,
                )
            else:
                (full_tile_fields, restricted_tile_fields, local_count_values, paired_invariant_counts,
                 full_trace, restricted_trace) = run_tile(tile.coordinates)

            local_full_counts = dict(zip(count_names, local_count_values[0]))
            local_restricted_counts = dict(zip(count_names, local_count_values[1]))
            local_shared_counts = dict(zip(shared_names, local_count_values[2]))
            packed_counts = torch.cat((
                _pack_work_counts(local_full_counts, count_names, full_tile_fields),
                _pack_work_counts(local_restricted_counts, count_names, full_tile_fields),
                _pack_work_counts(local_shared_counts, shared_names, full_tile_fields),
            ))

            full_fields_by_tile.append(full_tile_fields)
            restricted_fields_by_tile.append(restricted_tile_fields)
            tile_report = torch.cat((packed_counts.detach(), paired_invariant_counts.detach()))
            packed_count_total = (tile_report if packed_count_total is None else
                                  packed_count_total + tile_report)

        parameter = next(self.parameters())
        full_fields = torch.cat(full_fields_by_tile, dim=1) if full_fields_by_tile else parameter.new_zeros(
            query.batch_size, 0, self.output_dim,
        )
        restricted_fields = (
            torch.cat(restricted_fields_by_tile, dim=1) if restricted_fields_by_tile else
            parameter.new_zeros(query.batch_size, 0, self.output_dim)
        )
        if packed_count_total is not None:
            report_values = packed_count_total.detach().cpu().tolist()
            branch_count_width = 2 * len(count_names)
            shared_count_width = len(shared_names)
            count_values = report_values[:branch_count_width + shared_count_width]
            invariant_values = report_values[branch_count_width + shared_count_width:]
            if any(invariant_values):
                raise RuntimeError("Paired Candidate A shared work or normalization disagrees across branches.")
            full_counts = dict(zip(count_names, count_values[:len(count_names)]))
            restricted_counts = dict(zip(
                count_names, count_values[len(count_names):2 * len(count_names)],
            ))
            shared_counts = dict(zip(shared_names, count_values[2 * len(count_names):]))
        if return_trace:
            if query.num_queries == 0:
                full_trace = {"joint_fields": full_fields}
                restricted_trace = {"joint_fields": restricted_fields}
            else:
                full_trace["joint_fields"] = full_fields
                restricted_trace["joint_fields"] = restricted_fields
        return (
            InteractionJointOutput(full_fields, full_trace if return_trace else None, full_counts),
            InteractionJointOutput(restricted_fields, restricted_trace if return_trace else None, restricted_counts),
            shared_counts,
        )


__all__ = [
    "ConsistentJointMemberRoutingCore",
    "compose_joint_member_prior",
    "joint_likelihood_contribution",
    "joint_role_statistics",
]
