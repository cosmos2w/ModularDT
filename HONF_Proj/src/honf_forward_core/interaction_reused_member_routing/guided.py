"""R13-4 relation-guided access to original members.

This version keeps the R13-3 parameterization and applied-relation path. It
uses the relation-derived member prior both for weighted score calibration
and, in restricted execution, for selecting an E prefix before the member
scorer runs. M reads stay all-eligible and read strength remains separate
from the conditional member distribution.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import torch
from torch import Tensor

from .core import InteractionReusedMemberRoutingCore, _positive_softmax
from .relative import RelativeMemberRoutingCore, mass_prefix_mask, read_probability, relative_scores


def prior_weighted_scores(scores: Tensor, prior: Tensor, allowed: Tensor) -> Tensor:
    """Center and bound scores using the candidate-normalized relation prior.

    A unit variance floor keeps the bounded score scale stable. A member with
    vanishing prior mass therefore has vanishing influence on the calibration
    statistics, even while it remains relation-eligible.
    """
    if scores.shape != prior.shape or scores.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("Member scores, candidate prior and mask must have equal shapes.")
    weights = prior * allowed.to(prior.dtype)
    total = weights.sum(-1, keepdim=True)
    weights = torch.where(total > 0, weights / total.clamp_min(torch.finfo(weights.dtype).tiny),
                          torch.zeros_like(weights))
    mean = (weights * scores).sum(-1, keepdim=True)
    centered = torch.where(allowed, scores - mean, torch.zeros_like(scores))
    variance = (weights * centered.square()).sum(-1, keepdim=True)
    scaled = centered / (4. * torch.sqrt(variance + 1.))
    return 4. * torch.tanh(scaled) * allowed.to(scores.dtype)


def _normalize_nonnegative(value: Tensor, allowed: Tensor) -> Tensor:
    """Normalize a nonnegative tensor, returning zeros for an empty row."""
    masked = value * allowed.to(value.dtype)
    total = masked.sum(-1, keepdim=True)
    return torch.where(total > 0, masked / total.clamp_min(torch.finfo(value.dtype).tiny),
                       torch.zeros_like(masked))


class GuidedMemberRoutingCore(RelativeMemberRoutingCore):
    """R13-4 Full/Guide comparison core with unchanged R13-3 parameters."""

    schema = "interaction_reused_member_routing_v3"
    _additional_decode_count_keys = RelativeMemberRoutingCore._additional_decode_count_keys + (
        "query_M_eligible_candidate_rows", "query_E_eligible_candidate_rows",
        "query_M_prior_composition_rows", "query_E_prior_composition_rows",
        "query_M_prior_normalization_rows", "query_E_prior_normalization_rows",
        "query_M_score_calibration_rows", "query_E_score_calibration_rows",
        "query_E_prior_prefix_sort_rows", "query_E_prior_id_sort_rows",
    )

    def __init__(self, base_model_config: Mapping[str, Any], *, member_routing_mode="guide",
                 input_dataset="native_records", input_capability="joint_prescribed_records_v1",
                 entmax_temperature=1., query_tile_size=256, prior_mass_tolerance=0.01):
        if member_routing_mode not in ("full", "guide"):
            raise ValueError("member_routing_mode must be 'full' or 'guide'.")
        if (isinstance(prior_mass_tolerance, bool) or
                not isinstance(prior_mass_tolerance, (float, int)) or
                not 0 <= float(prior_mass_tolerance) < 1):
            raise ValueError("Prior omitted-mass tolerance must be finite and in [0,1).")
        # RelativeMemberRoutingCore owns the exact historical parameter
        # construction. Its Route arm is used only during initialization;
        # R13-4's Full/Guide name does not freeze the proposal parameters.
        super().__init__(base_model_config, member_routing_mode="route",
                         input_dataset=input_dataset, input_capability=input_capability,
                         entmax_temperature=entmax_temperature, query_tile_size=query_tile_size,
                         sparse_mass_tolerance=float(prior_mass_tolerance))
        self.member_routing_mode = member_routing_mode
        self.prior_mass_tolerance = float(prior_mass_tolerance)
        self.sparse_mass_tolerance = self.prior_mass_tolerance
        self._read_execution_override: str | None = None
        self.architecture = self._arm_name(member_routing_mode)
        self._set_mode_trainability()

    @staticmethod
    def _arm_name(mode):
        if mode not in ("full", "guide", "route"):
            raise ValueError("R13-4 member-routing mode must be 'full' or 'guide'.")
        return "R13-4-Full" if mode == "full" else "R13-4-Guide"

    def set_member_routing_mode(self, mode: str) -> None:
        if mode not in ("full", "guide"):
            raise ValueError("member-routing mode must be 'full' or 'guide'.")
        self.member_routing_mode = mode
        self.architecture = self._arm_name(mode)
        self._set_mode_trainability()

    def _set_mode_trainability(self) -> None:
        # Both branches use the same live proposal and applied B to form P.
        for parameter in self.member_router.proposal_M.parameters():
            parameter.requires_grad_(True)

    def get_config(self) -> dict[str, Any]:
        base = InteractionReusedMemberRoutingCore.get_config(self)
        return {**base, "schema": self.schema, "architecture": self.architecture_name,
                "member_routing_mode": self.member_routing_mode,
                "prior_mass_tolerance": self.prior_mass_tolerance}

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        required = {"schema", "architecture", "member_routing_mode", "base_model_config", "input_dataset",
                    "input_capability", "entmax_temperature", "query_tile_size", "prior_mass_tolerance"}
        if set(config) != required or config.get("schema") != cls.schema:
            raise ValueError("R13-4 guided member-routing configuration inventory/schema differs.")
        mode = config["member_routing_mode"]
        if mode not in ("full", "guide") or config["architecture"] != cls._arm_name(mode):
            raise ValueError("R13-4 architecture and member_routing_mode disagree.")
        tolerance = config["prior_mass_tolerance"]
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or
                not 0 <= float(tolerance) < 1):
            raise ValueError("Prior omitted-mass tolerance must be finite and in [0,1).")
        translated = dict(config)
        translated.pop("prior_mass_tolerance")
        translated.update(
            schema=InteractionReusedMemberRoutingCore.schema,
            architecture="R13-3-Full" if mode == "full" else "R13-3-Route",
            member_routing_mode="full" if mode == "full" else "route")
        InteractionReusedMemberRoutingCore.validate_config(translated)

    @contextmanager
    def read_execution(self, mode: str):
        """Select all-eligible or prior-guided execution, independent of train/eval."""
        if mode not in ("full", "restricted"):
            raise ValueError("Read execution must be 'full' or 'restricted'.")
        previous = self._read_execution_override
        self._read_execution_override = mode
        try:
            yield self
        finally:
            self._read_execution_override = previous

    def decode(self, organization, query, controls=None, *, intervention=None,
               return_trace=False, trace_request=None, **kwargs):
        # The inherited query-tile checkpoint recomputes its closure during
        # backward. Capture the resolved mode in the intervention now so the
        # recomputation cannot observe a later context-manager state.
        routed_controls = {} if intervention is None else dict(intervention)
        if "read_mode" not in routed_controls:
            routed_controls["read_mode"] = self._read_execution_override or "full"
        return super().decode(organization, query, controls, intervention=routed_controls,
                              return_trace=return_trace, trace_request=trace_request, **kwargs)

    @staticmethod
    def _row_mass_adjusted_proposals(pi: Tensor, matrix: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]:
        row_mass = (matrix * nu[:, None, :]).sum(-1)
        contribution = pi * row_mass[:, None, :]
        return _normalize_nonnegative(contribution, torch.ones_like(contribution, dtype=torch.bool)), row_mass

    def _typed_read_state(self, prepared, query, qstates, intervention, *, return_trace):
        scene = prepared.scene
        controls = {} if intervention is None else dict(intervention)
        legal = {"read_mode", "member_score_mode", "value_execution", "fixed_B_M", "fixed_B_E", "fixed_pi",
                 "fixed_candidate_mask_M", "fixed_candidate_mask_E", "fixed_member_logits_M",
                 "fixed_member_logits_E"}
        if set(controls) - legal:
            raise ValueError(f"Unknown R13-4 routing intervention: {sorted(set(controls) - legal)}")
        mode = controls.get("read_mode", "full")
        if mode not in ("full", "restricted"):
            raise ValueError("R13-4 read mode must be 'full' or 'restricted'.")
        execution = controls.get("value_execution", "selected")
        if execution not in ("selected", "dense_candidate_union"):
            raise ValueError("Unknown R13-4 value executor.")
        score_mode = controls.get("member_score_mode", "learned")
        if score_mode not in ("learned", "geometry"):
            raise ValueError("member_score_mode must be 'learned' or 'geometry'.")

        Q, M, E = query.num_queries, scene.num_modules, scene.num_environment_tokens
        qmask = self._query_mask(query)
        B_M = self._validate_dense_control(controls.get("fixed_B_M"), tuple(prepared.B_M.shape),
                                           "fixed_B_M", prepared.B_M)
        B_E = self._validate_dense_control(controls.get("fixed_B_E"), tuple(prepared.B_E.shape),
                                           "fixed_B_E", prepared.B_E)
        B_M = prepared.B_M if B_M is None else B_M
        B_E = prepared.B_E if B_E is None else B_E
        if controls.get("fixed_B_M") is not None:
            if not torch.equal(B_M.diagonal(dim1=1, dim2=2), scene.module_mask.to(B_M)):
                raise ValueError("B_M must preserve present M identity access.")
            invalid = (B_M > 0) & ~(scene.module_mask[:, :, None] & scene.module_mask[:, None, :])
            if bool(invalid.any()):
                raise ValueError("B_M cannot address absent original members.")
        if controls.get("fixed_B_E") is not None:
            invalid = (B_E > 0) & ~(scene.module_mask[:, :, None] & scene.environment_mask[:, None, :])
            if bool(invalid.any()):
                raise ValueError("B_E cannot address absent original members.")

        raw_proposal, proposal_rows = self._proposal_logits(prepared, query, qstates)
        proposal_allowed = scene.module_mask[:, None, :].expand_as(raw_proposal) & qmask[..., None]
        normalized_proposal = self._proposal_normalization(raw_proposal, proposal_allowed)
        pi = _positive_softmax(normalized_proposal / self.entmax_temperature, proposal_allowed)
        fixed_pi = controls.get("fixed_pi")
        if fixed_pi is not None:
            if fixed_pi.shape != raw_proposal.shape or fixed_pi.device != raw_proposal.device:
                raise ValueError("Fixed proposals have incorrect shape/device.")
            invalid = (~torch.isfinite(fixed_pi) | (fixed_pi < 0) |
                       ((fixed_pi > 0) & ~proposal_allowed))
            if bool(invalid.any()) or bool((fixed_pi.sum(-1) > 1.00001).any()):
                raise ValueError("Fixed proposals must be finite eligible probabilities.")
            pi = fixed_pi.to(qstates)

        coefficients: dict[str, Tensor] = {}
        eligible_masks: dict[str, Tensor] = {}
        priors: dict[str, Tensor] = {}
        row_masses: dict[str, Tensor] = {}
        mixtures: dict[str, Tensor] = {}
        relation_rows = 0
        index_scan_rows = 0
        for role, matrix, count, nu in (("M", B_M, M, prepared.nu_M),
                                        ("E", B_E, E, prepared.nu_E)):
            cached = matrix is getattr(prepared, f"B_{role}")
            a, possible, _, _, visits = self._expand_live_coefficients(
                pi, matrix, Q, count,
                cached_index=getattr(prepared, f"B_{role}_edge_index") if cached else None,
                cached_values=getattr(prepared, f"B_{role}_edge_values") if cached else None)
            if not cached:
                index_scan_rows += matrix.numel()
            role_eligible = scene.module_mask if role == "M" else scene.environment_mask & (scene.environment_measure > 0)
            eligible = possible & role_eligible[:, None, :] & qmask[..., None]
            mass = a * nu[:, None, :]
            prior = _normalize_nonnegative(mass, eligible)
            mixture, row_mass = self._row_mass_adjusted_proposals(pi, matrix, nu)
            coefficients[role] = a
            eligible_masks[role] = eligible
            priors[role] = prior
            row_masses[role] = row_mass
            mixtures[role] = mixture
            relation_rows += visits

        chosen_masks: dict[str, Tensor] = {"M": eligible_masks["M"]}
        if mode == "restricted":
            ids_E = scene.environment_ids
            chosen_masks["E"] = mass_prefix_mask(priors["E"], eligible_masks["E"], ids_E,
                                                  self.prior_mass_tolerance)
            prior_sort_rows = int(scene.batch_size * Q * E)
            prior_id_sort_rows = int(scene.batch_size * E)
        else:
            chosen_masks["E"] = eligible_masks["E"]
            prior_sort_rows = 0
            prior_id_sort_rows = 0

        for role in ("M", "E"):
            fixed = controls.get(f"fixed_candidate_mask_{role}")
            if fixed is None:
                continue
            eligible = eligible_masks[role]
            if fixed.shape != eligible.shape or fixed.dtype != torch.bool or bool((fixed & ~eligible).any()):
                raise ValueError("Fixed candidates must retain valid actual B paths.")
            chosen_masks[role] = fixed

        counts = {
            "query_M_proposal_score_rows": proposal_rows,
            "query_address_relation_rows": relation_rows,
            "query_address_coefficient_pair_rows": int(scene.batch_size * Q * M * (M + E)),
            "query_address_support_pair_rows": int(scene.batch_size * Q * M * (M + E)),
            "query_address_index_scan_rows": index_scan_rows,
            "query_address_softmax_rows": int(proposal_allowed.sum()),
            "query_typed_read_gate_rows": int(2 * scene.batch_size * Q),
            "query_receiver_rows": int(scene.batch_size * Q),
            # Address coefficient/support pair products are charged above;
            # these are the separate per-member physical-measure multiply and
            # normalization reductions that turn a into P.
            "query_M_prior_composition_rows": int(scene.batch_size * Q * M),
            "query_E_prior_composition_rows": int(scene.batch_size * Q * E),
            "query_M_prior_normalization_rows": int(scene.batch_size * Q * M),
            "query_E_prior_normalization_rows": int(scene.batch_size * Q * E),
            "query_E_prior_prefix_sort_rows": prior_sort_rows,
            "query_E_prior_id_sort_rows": prior_id_sort_rows,
            "query_routing_normalization_rows": int(
                proposal_allowed.sum() + scene.batch_size * Q * (M + E)),
        }
        trace: dict[str, Tensor] = {
            "proposal_logits_M": raw_proposal,
            "proposal_effective_scores_M": normalized_proposal,
            "proposal_weights_M": pi,
            "pi_M": pi,
            "proposal_null": (1. - pi.sum(-1)).clamp_min(0.),
            "original_module_ids": scene.module_ids,
            "original_environment_ids": scene.environment_ids,
            "B_M": B_M,
            "B_E": B_E,
        }
        contexts = []
        scene_eligible_by_role = {
            "M": scene.module_mask,
            "E": scene.environment_mask & (scene.environment_measure > 0),
        }
        for role in ("M", "E"):
            eligible = eligible_masks[role]
            chosen = chosen_masks[role]
            prior = priors[role]
            candidate_prior = _normalize_nonnegative(prior, chosen)
            raw, scored = self._member_base_logits(prepared, query, qstates, role, chosen, score_mode)
            if scored and score_mode == "learned":
                counts["query_routing_normalization_rows"] += int(scene.batch_size * (Q + chosen.shape[-1]))
            fixed_raw = controls.get(f"fixed_member_logits_{role}")
            if fixed_raw is not None:
                if fixed_raw.shape != raw.shape or fixed_raw.device != raw.device or not bool(torch.isfinite(fixed_raw).all()):
                    raise ValueError("Fixed member scores must be finite with the original query/member axes.")
                raw = fixed_raw
            normalized = prior_weighted_scores(raw, candidate_prior, chosen)
            log_prior = candidate_prior.clamp_min(torch.finfo(candidate_prior.dtype).tiny).log()
            log_prior = torch.where(chosen, log_prior, torch.zeros_like(log_prior))
            effective = (normalized + log_prior) / self.entmax_temperature
            relative = _positive_softmax(effective, chosen)
            expectation = (relative * normalized).sum(-1)
            entropy = -(relative * relative.clamp_min(torch.finfo(relative.dtype).tiny).log()).sum(-1)
            scene_count = scene_eligible_by_role[role].sum(-1)[:, None].expand(-1, Q).to(qstates)
            entropy_scale = scene_count.clamp_min(2.).log()
            entropy = torch.where(scene_count > 1, entropy / entropy_scale, torch.zeros_like(entropy))
            gate_inputs = torch.cat((self._routing_features(qstates), expectation[..., None], entropy[..., None]), -1)
            gate_logits = self.typed_read_gates[role](gate_inputs)
            contrast = gate_logits[..., 0] - gate_logits[..., 1]
            has_members = chosen.any(-1)
            strength = read_probability(contrast) * has_members.to(qstates) * qmask.to(qstates)
            continuous = strength[..., None] * relative
            beta = continuous * chosen.to(continuous.dtype)
            executed = chosen if execution == "dense_candidate_union" else (chosen & (beta > 0))
            values = prepared.member_values_M if role == "M" else prepared.member_values_E
            contexts.append(self._reduce_selected(values, beta, executed))

            eligible_count = eligible.sum(-1)
            chosen_count = chosen.sum(-1)
            prior_captured = (prior * chosen.to(prior.dtype)).sum(-1)
            counts.update({
                f"query_{role}_cheap_score_rows": scored,
                f"query_{role}_candidate_rows": int(chosen.sum()),
                f"query_{role}_eligible_candidate_rows": int(eligible.sum()),
                f"query_{role}_selected_value_rows": int((beta > 0).sum()),
                f"query_{role}_value_execution_rows": int(executed.sum()),
                f"query_{role}_dense_candidate_value_rows": int(chosen.sum()) if execution == "dense_candidate_union" else 0,
                f"query_{role}_continuous_value_rows": (
                    int(eligible.sum()) if role == "M" or mode == "full" else 0),
                f"query_{role}_member_softmax_rows": int(chosen.sum()),
                f"query_{role}_sorted_rows": int(eligible.numel()) if role == "E" and mode == "restricted" else 0,
                f"query_{role}_id_sort_rows": int(scene.environment_ids.numel()) if role == "E" and mode == "restricted" else 0,
                f"query_{role}_score_calibration_rows": int(chosen.sum()),
            })
            if return_trace:
                trace.update({
                    f"eligible_candidate_mask_{role}": eligible,
                    f"full_eligible_mask_{role}": eligible,
                    f"chosen_candidate_mask_{role}": chosen,
                    f"candidate_mask_{role}": chosen,
                    f"eligible_candidate_count_{role}": eligible_count,
                    f"chosen_candidate_count_{role}": chosen_count,
                    f"candidate_count_{role}": chosen_count,
                    f"prior_P_{role}": prior,
                    f"candidate_normalized_prior_P_{role}": candidate_prior,
                    f"address_coefficients_{role}": coefficients[role],
                    f"address_row_mass_{role}": row_masses[role],
                    f"row_mass_adjusted_proposals_{role}": mixtures[role],
                    f"tilde_pi_{role}": mixtures[role],
                    f"member_logits_{role}": raw,
                    f"member_normalized_scores_{role}": normalized,
                    f"member_prior_{role}": log_prior,
                    f"member_effective_scores_{role}": effective,
                    f"member_relative_weights_{role}": relative,
                    f"conditional_preference_{role}": relative,
                    f"member_weights_{role}": beta,
                    f"member_continuous_weights_{role}": continuous,
                    f"member_null_{role}": (1. - beta.sum(-1)).clamp_min(0.),
                    f"member_read_strength_{role}": strength,
                    f"member_read_logits_{role}": gate_logits,
                    f"member_read_contrast_{role}": contrast,
                    f"member_gate_derivative_{role}": 0.5 / torch.hypot(contrast, torch.ones_like(contrast)).pow(3),
                    f"prior_captured_mass_{role}": prior_captured,
                    f"prior_omitted_mass_{role}": (prior.sum(-1) - prior_captured).clamp_min(0.),
                    f"scene_eligible_count_{role}": scene_count,
                    f"selected_value_indices_{role}": (beta > 0).nonzero(as_tuple=False),
                })

        fused = torch.cat((qstates, *contexts), -1)
        fields = self.member_router.joint_head(self.member_router.joint_trunk(fused)) * qmask[..., None]
        return fused, trace, counts, fields

    @staticmethod
    def _proposal_normalization(scores: Tensor, allowed: Tensor) -> Tensor:
        """Keep the historical v2 proposal normalization unchanged."""
        return relative_scores(scores, allowed)


__all__ = ["GuidedMemberRoutingCore", "prior_weighted_scores"]
