"""Relative original-member selection with continuous native learning.

V1 remains loadable with its original equations. This version separates a
conditional distribution over related records from learned read/skip strength.
Training reads every eligible original member; evaluation may omit a declared
small absolute weight mass before gathering values.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import InteractionReusedMemberRoutingCore, _positive_softmax


def relative_scores(scores: Tensor, allowed: Tensor) -> Tensor:
    """Population centering with unit RMS floor and bounded score spread."""
    count = allowed.sum(-1, keepdim=True).clamp_min(1).to(scores)
    mean = (scores * allowed).sum(-1, keepdim=True) / count
    centered = torch.where(allowed, scores - mean, torch.zeros_like(scores))
    variance = centered.square().sum(-1, keepdim=True) / count
    return 4. * torch.tanh(centered / (4. * torch.sqrt(variance + 1.))) * allowed


def read_probability(contrast: Tensor) -> Tensor:
    """Softsign Bernoulli probability, using stable rational tail identities.

    Its real-arithmetic derivative is 1/(2*(1+d*d)**1.5), rather than an
    exact-zero support derivative. Native loss sensitivity is still required.
    """
    radius = torch.hypot(contrast, torch.ones_like(contrast))
    # Both branches remain finite, including the unused one. Signed clamps
    # preserve the correct derivative 1/2 at zero (abs would give zero there).
    negative_tail = 0.5 / radius / (radius - contrast.clamp_max(0.))
    positive_tail = 0.5 / radius / (radius + contrast.clamp_min(0.))
    return torch.where(contrast < 0, negative_tail, 1. - positive_tail)


def mass_prefix_mask(weights: Tensor, allowed: Tensor, original_ids: Tensor,
                     tolerance: float) -> Tensor:
    """Keep a variable prefix; omitted absolute real mass is <= tolerance.

    First sort by original ID, then stably by decreasing weight. Equal-weight
    ties therefore survive source permutations with the same physical IDs.
    Retained weights are not renormalized. No minimum count or fixed K exists.
    """
    if weights.shape != allowed.shape or original_ids.shape != (weights.shape[0], weights.shape[-1]):
        raise ValueError("Sparse member weights, masks and original IDs have inconsistent axes.")
    if not weights.shape[-1]:
        return allowed.clone()
    by_id = torch.argsort(original_ids, dim=-1, stable=True)[:, None, :].expand_as(weights)
    id_weights = weights.gather(-1, by_id)
    order = torch.argsort(id_weights, dim=-1, descending=True, stable=True)
    indices = by_id.gather(-1, order)
    sorted_weights = weights.gather(-1, indices)
    before = sorted_weights.cumsum(-1) - sorted_weights
    required = (weights.sum(-1, keepdim=True) - float(tolerance)).clamp_min(0.)
    keep = (before < required) & allowed.gather(-1, indices)
    return torch.zeros_like(allowed).scatter(-1, indices, keep)


class RelativeMemberRoutingCore(InteractionReusedMemberRoutingCore):
    """R13-3 corrective version: relative selection and a learned typed skip."""

    schema = "interaction_reused_member_routing_v2"
    _additional_decode_count_keys = (
        "query_address_softmax_rows", "query_M_continuous_value_rows", "query_E_continuous_value_rows",
        "query_M_member_softmax_rows", "query_E_member_softmax_rows", "query_M_sorted_rows", "query_E_sorted_rows",
        "query_address_support_pair_rows", "query_address_index_scan_rows", "query_typed_read_gate_rows",
        "query_routing_normalization_rows", "query_M_id_sort_rows", "query_E_id_sort_rows",
    )

    def __init__(self, base_model_config: Mapping[str, Any], *, member_routing_mode="route",
                 input_dataset="native_records", input_capability="joint_prescribed_records_v1",
                 entmax_temperature=1., query_tile_size=256, sparse_mass_tolerance=0.01):
        if member_routing_mode != "route":
            raise ValueError("The corrective version defines one Route candidate; reuse the retained Full reference.")
        if not isinstance(sparse_mass_tolerance, (float, int)) or not 0 <= sparse_mass_tolerance < 1:
            raise ValueError("Sparse omitted-mass tolerance must be finite and in [0,1).")
        self.sparse_mass_tolerance = float(sparse_mass_tolerance)
        super().__init__(base_model_config, member_routing_mode=member_routing_mode,
                         input_dataset=input_dataset, input_capability=input_capability,
                         entmax_temperature=entmax_temperature, query_tile_size=query_tile_size)
        self.sparse_mass_tolerance = float(sparse_mass_tolerance)
        self._read_execution_override: str | None = None
        # Only query features and relative selection statistics enter these
        # gates. They cannot supply an independent original-record value path.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._stable_seed("relative_typed_read_skip"))
            self.typed_read_gates = nn.ModuleDict({role: nn.Sequential(
                nn.Linear(self.context_dim + 2, max(8, self.context_dim // 4)),
                nn.SiLU(), nn.Linear(max(8, self.context_dim // 4), 2),
            ) for role in ("M", "E")})
            for module in self.typed_read_gates.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _arm_name(mode):
        if mode != "route":
            raise ValueError("The corrective model has only the relative Route arm.")
        return "R13-3-Relative"

    def get_config(self):
        return {**super().get_config(), "sparse_mass_tolerance": self.sparse_mass_tolerance}

    @classmethod
    def validate_config(cls, config):
        required = {"schema", "architecture", "member_routing_mode", "base_model_config", "input_dataset",
                    "input_capability", "entmax_temperature", "query_tile_size", "sparse_mass_tolerance"}
        if set(config) != required or config.get("schema") != cls.schema:
            raise ValueError("Relative member-routing configuration inventory/schema differs.")
        if config["architecture"] != "R13-3-Relative" or config["member_routing_mode"] != "route":
            raise ValueError("Relative member-routing requires the separately named corrective arm.")
        # Validate the unchanged encoder/context configuration through v1's
        # ordinary constructor; no old checkpoint is reinterpreted as v2.
        translated = dict(config)
        translated.pop("sparse_mass_tolerance")
        translated.update(schema=InteractionReusedMemberRoutingCore.schema, architecture="R13-3-Route")
        InteractionReusedMemberRoutingCore.validate_config(translated)
        if not isinstance(config["sparse_mass_tolerance"], (float, int)) or not 0 <= config["sparse_mass_tolerance"] < 1:
            raise ValueError("Sparse omitted-mass tolerance must be finite and in [0,1).")

    @contextmanager
    def read_execution(self, mode: str):
        """Explicit same-checkpoint continuous reference versus sparse executor."""
        if mode not in ("continuous", "sparse"):
            raise ValueError("Read execution must be continuous or sparse.")
        previous = self._read_execution_override
        self._read_execution_override = mode
        try:
            yield self
        finally:
            self._read_execution_override = previous

    @staticmethod
    def _routing_features(features):
        return F.layer_norm(features, (features.shape[-1],))

    def _proposal_logits(self, prepared, query, qstates):
        allowed = prepared.scene.module_mask[:, None, :].expand(-1, query.num_queries, -1)
        allowed = allowed & self._query_mask(query)[..., None]
        rows = self._pair_rows(allowed)
        output = qstates.new_zeros(query.batch_size, query.num_queries, prepared.scene.num_modules)
        if not rows.numel():
            return output, 0
        b, q, source = rows.unbind(-1)
        features = torch.cat((self._routing_features(qstates)[b, q],
                              self._routing_features(prepared.member_keys_M)[b, source],
                              self._pair_geometry(prepared.scene, query, "M", rows)), -1)
        return output.index_put((b, q, source), self.member_router.proposal_M(features).squeeze(-1)), len(rows)

    def _member_base_logits(self, prepared, query, qstates, source, candidates, score_mode):
        if score_mode != "learned":
            return super()._member_base_logits(prepared, query, qstates, source, candidates, score_mode)
        keys = prepared.member_keys_M if source == "M" else prepared.member_keys_E
        output = qstates.new_zeros(query.batch_size, query.num_queries, keys.shape[1])
        rows = self._pair_rows(candidates)
        if not rows.numel():
            return output, 0
        b, q, source_index = rows.unbind(-1)
        features = torch.cat((self._routing_features(qstates)[b, q],
                              self._routing_features(keys)[b, source_index],
                              self._pair_geometry(prepared.scene, query, source, rows)), -1)
        scorer = self.member_router.member_M if source == "M" else self.member_router.member_E
        return output.index_put((b, q, source_index), scorer(features).squeeze(-1)), len(rows)

    def _typed_read_state(self, prepared, query, qstates, intervention, *, return_trace):
        scene = prepared.scene
        controls = {} if intervention is None else dict(intervention)
        legal = {"member_score_mode", "value_execution", "read_mode", "fixed_B_M", "fixed_B_E", "fixed_pi",
                 "fixed_candidate_mask_M", "fixed_candidate_mask_E", "fixed_member_logits_M", "fixed_member_logits_E"}
        if set(controls) - legal:
            raise ValueError(f"Unknown corrective routing intervention: {sorted(set(controls) - legal)}")
        mode = controls.get("read_mode", self._read_execution_override or ("continuous" if self.training else "sparse"))
        if mode not in ("continuous", "sparse"):
            raise ValueError("Corrective read mode must be continuous or sparse.")
        execution = controls.get("value_execution", "selected")
        if execution not in ("selected", "dense_candidate_union"):
            raise ValueError("Unknown corrective value executor.")
        B_M = self._validate_dense_control(controls.get("fixed_B_M"), tuple(prepared.B_M.shape), "fixed_B_M", prepared.B_M)
        B_E = self._validate_dense_control(controls.get("fixed_B_E"), tuple(prepared.B_E.shape), "fixed_B_E", prepared.B_E)
        B_M = prepared.B_M if B_M is None else B_M
        B_E = prepared.B_E if B_E is None else B_E
        if controls.get("fixed_B_M") is not None:
            if not torch.equal(B_M.diagonal(dim1=1, dim2=2), scene.module_mask.to(B_M)):
                raise ValueError("B_M must preserve present M identity access.")
            if bool(((B_M > 0) & ~(scene.module_mask[:, :, None] & scene.module_mask[:, None, :])).any()):
                raise ValueError("B_M cannot address absent original members.")
        if controls.get("fixed_B_E") is not None and bool(((B_E > 0) & ~(scene.module_mask[:, :, None] & scene.environment_mask[:, None, :])).any()):
            raise ValueError("B_E cannot address absent original members.")
        raw_proposal, proposal_rows = self._proposal_logits(prepared, query, qstates)
        qmask = self._query_mask(query)
        allowed = scene.module_mask[:, None, :].expand_as(raw_proposal) & qmask[..., None]
        normalized_proposal = relative_scores(raw_proposal, allowed)
        pi = _positive_softmax(normalized_proposal / self.entmax_temperature, allowed)
        if controls.get("fixed_pi") is not None:
            pi = controls["fixed_pi"]
            if pi.shape != raw_proposal.shape or pi.device != raw_proposal.device:
                raise ValueError("Fixed proposals have incorrect shape/device.")
            if bool((~torch.isfinite(pi) | (pi < 0) | ((pi > 0) & ~allowed)).any()) or bool((pi.sum(-1) > 1.00001).any()):
                raise ValueError("Fixed proposals must be finite eligible probabilities.")
        Q, M, E = query.num_queries, scene.num_modules, scene.num_environment_tokens
        coefficients = {}
        candidates = {}
        relation_rows = 0
        index_scan_rows = 0
        for role, matrix, n in (("M", B_M, M), ("E", B_E, E)):
            cached = matrix is getattr(prepared, f"B_{role}")
            a, possible, _, _, visits = self._expand_live_coefficients(
                pi, matrix, Q, n,
                cached_index=getattr(prepared, f"B_{role}_edge_index") if cached else None,
                cached_values=getattr(prepared, f"B_{role}_edge_values") if cached else None,
            )
            if not cached:
                index_scan_rows += matrix.numel()
            eligible = scene.module_mask if role == "M" else scene.environment_mask & (scene.environment_measure > 0)
            candidate = possible & eligible[:, None, :] & qmask[..., None]
            fixed = controls.get(f"fixed_candidate_mask_{role}")
            if fixed is not None:
                if fixed.shape != candidate.shape or fixed.dtype != torch.bool or bool((fixed & ~candidate).any()):
                    raise ValueError("Fixed candidates must retain valid actual B paths.")
                candidate = fixed
            coefficients[role], candidates[role] = a, candidate
            relation_rows += visits
        counts = {"query_M_proposal_score_rows": proposal_rows, "query_address_relation_rows": relation_rows,
                  "query_address_coefficient_pair_rows": scene.batch_size * Q * M * (M + E),
                  "query_address_support_pair_rows": scene.batch_size * Q * M * (M + E),
                  "query_address_index_scan_rows": index_scan_rows,
                  "query_typed_read_gate_rows": 2 * scene.batch_size * Q,
                  "query_routing_normalization_rows": 2 * scene.batch_size * Q + (
                      scene.batch_size * (Q + M) if proposal_rows else 0),
                  "query_receiver_rows": scene.batch_size * Q, "query_address_softmax_rows": int(allowed.sum())}
        trace = {"proposal_logits_M": raw_proposal, "proposal_effective_scores_M": normalized_proposal,
                 "proposal_weights_M": pi, "proposal_null": (1. - pi.sum(-1)).clamp_min(0.),
                 "original_module_ids": scene.module_ids, "original_environment_ids": scene.environment_ids,
                 "B_M": B_M, "B_E": B_E}
        contexts = []
        for role in ("M", "E"):
            candidate, a = candidates[role], coefficients[role]
            raw, scored = self._member_base_logits(prepared, query, qstates, role, candidate,
                                                   controls.get("member_score_mode", "learned"))
            if scored and controls.get("member_score_mode", "learned") == "learned":
                counts["query_routing_normalization_rows"] += scene.batch_size * (Q + candidate.shape[-1])
            fixed_raw = controls.get(f"fixed_member_logits_{role}")
            if fixed_raw is not None:
                if fixed_raw.shape != raw.shape or fixed_raw.device != raw.device or not bool(torch.isfinite(fixed_raw).all()):
                    raise ValueError("Fixed member scores must be finite with the original query/member axes.")
                raw = fixed_raw
            normalized = relative_scores(raw, candidate)
            nu = prepared.nu_M if role == "M" else prepared.nu_E
            log_a = a.clamp_min(torch.finfo(a.dtype).tiny).log()
            log_nu = nu.clamp_min(torch.finfo(nu.dtype).tiny).log()[:, None, :].expand_as(a)
            log_weight = (log_a + log_nu).masked_fill(~candidate, -torch.inf)
            has_members = candidate.any(-1)
            if a.shape[-1]:
                # An empty row must not backpropagate through logsumexp(-inf).
                safe_log_weight = torch.where(has_members[..., None], log_weight, torch.zeros_like(log_weight))
                normalizer = torch.logsumexp(safe_log_weight, -1, keepdim=True)
                normalizer = torch.where(has_members[..., None], normalizer, torch.zeros_like(normalizer))
                prior = torch.where(candidate, log_weight - normalizer, torch.zeros_like(log_weight))
            else:
                prior = log_weight
            effective = (normalized + prior) / self.entmax_temperature
            relative = _positive_softmax(effective, candidate)
            expectation = (relative * normalized).sum(-1)
            entropy = -(relative * relative.clamp_min(torch.finfo(relative.dtype).tiny).log()).sum(-1)
            n = candidate.sum(-1).to(qstates)
            entropy = torch.where(n > 1, entropy / n.clamp_min(2).log(), torch.zeros_like(entropy))
            gate_inputs = torch.cat((self._routing_features(qstates), expectation[..., None], entropy[..., None]), -1)
            gate_logits = self.typed_read_gates[role](gate_inputs)
            contrast = gate_logits[..., 0] - gate_logits[..., 1]
            strength = read_probability(contrast) * has_members.to(qstates) * qmask.to(qstates)
            continuous = strength[..., None] * relative
            ids = scene.module_ids if role == "M" else scene.environment_ids
            support = (mass_prefix_mask(continuous, candidate, ids, self.sparse_mass_tolerance)
                       if mode == "sparse" else candidate)
            beta = continuous * support.to(continuous)
            executed = candidate if execution == "dense_candidate_union" else support & (beta > 0)
            values = prepared.member_values_M if role == "M" else prepared.member_values_E
            contexts.append(self._reduce_selected(values, beta, executed))
            counts.update({f"query_{role}_cheap_score_rows": scored,
                           f"query_{role}_candidate_rows": int(candidate.sum()),
                           f"query_{role}_selected_value_rows": int((beta > 0).sum()),
                           f"query_{role}_value_execution_rows": int(executed.sum()),
                           f"query_{role}_dense_candidate_value_rows": int(candidate.sum()) if execution == "dense_candidate_union" else 0,
                           f"query_{role}_continuous_value_rows": int(candidate.sum()) if mode == "continuous" else 0,
                           f"query_{role}_member_softmax_rows": int(candidate.sum()),
                           f"query_{role}_id_sort_rows": ids.numel() if mode == "sparse" else 0,
                           f"query_{role}_sorted_rows": int(candidate.numel()) if mode == "sparse" else 0})
            if return_trace:
                trace.update({f"candidate_mask_{role}": candidate, f"address_coefficients_{role}": a,
                              f"member_logits_{role}": raw, f"member_normalized_scores_{role}": normalized,
                              f"member_prior_{role}": prior, f"member_log_address_{role}": log_a,
                              f"member_log_measure_{role}": log_nu, f"member_effective_scores_{role}": effective,
                              f"member_relative_weights_{role}": relative, f"member_weights_{role}": beta,
                              f"member_continuous_weights_{role}": continuous,
                              f"member_null_{role}": (1. - beta.sum(-1)).clamp_min(0.),
                              f"member_read_strength_{role}": strength, f"member_read_logits_{role}": gate_logits,
                              f"member_read_contrast_{role}": contrast,
                              f"member_gate_derivative_{role}": 0.5 / torch.hypot(contrast, torch.ones_like(contrast)).pow(3),
                              f"member_omitted_mass_{role}": continuous.sum(-1) - beta.sum(-1),
                              f"selected_value_indices_{role}": (beta > 0).nonzero(as_tuple=False)})
        fused = torch.cat((qstates, *contexts), -1)
        fields = self.member_router.joint_head(self.member_router.joint_trunk(fused)) * qmask[..., None]
        return fused, trace, counts, fields


__all__ = ["RelativeMemberRoutingCore", "relative_scores", "read_probability", "mass_prefix_mask"]
