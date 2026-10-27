"""Joint M/E organizational views and original-member routing (R13-5).

This capability keeps the R13-4 parameter inventory and joint physical head,
while adding live E-address rows and one joint M/E member-coverage prefix.
The prepared H tensor contains the applied, stage-labelled contextual gates
plus the explicit M identity access ledger. Its row-normalized retrieval view
is derived from H and the typed member measure; it is not a physical influence
matrix or a replacement for native physical integration measures.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .core import InteractionReusedMemberRoutingCore, _positive_softmax
from .guided import GuidedMemberRoutingCore, prior_weighted_scores
from .relative import read_probability


_SCHEMA = "interaction_reused_joint_member_routing_v1"
_ROUTE_ORDER = ("mm", "em", "ee", "me")
_BLOCK_NAMES = {"mm": "MM", "em": "EM", "ee": "EE", "me": "ME"}
_ROLE_M = 0
_ROLE_E = 1


def _normalize_nonnegative(value: Tensor, allowed: Tensor) -> Tensor:
    """Normalize a nonnegative tensor along its final axis, preserving empties."""
    if value.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("Values and boolean eligibility must have equal shapes.")
    masked = torch.where(allowed, value, torch.zeros_like(value))
    total = masked.sum(-1, keepdim=True)
    tiny = torch.finfo(value.dtype).tiny
    return torch.where(total > 0, masked / total.clamp_min(tiny), torch.zeros_like(masked))


def _masked_softmax(logits: Tensor, allowed: Tensor) -> Tensor:
    """Stable softmax over eligible entries, with an exact zero for empty rows."""
    if logits.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("Proposal logits and boolean address mask must have equal shapes.")
    if not logits.shape[-1]:
        return logits.clone()
    any_allowed = allowed.any(-1, keepdim=True)
    masked = logits.masked_fill(~allowed, -torch.inf)
    maximum = masked.amax(-1, keepdim=True)
    maximum = torch.where(any_allowed, maximum, torch.zeros_like(maximum))
    shifted = torch.where(allowed, logits - maximum, torch.zeros_like(logits))
    exponentials = torch.exp(shifted).clamp_min(torch.finfo(logits.dtype).tiny) * allowed.to(logits.dtype)
    total = exponentials.sum(-1, keepdim=True)
    probabilities = exponentials / total.clamp_min(torch.finfo(logits.dtype).tiny)
    return torch.where(any_allowed, probabilities, torch.zeros_like(probabilities))


def build_joint_member_measure(
    module_mask: Tensor,
    environment_mask: Tensor,
    environment_measure: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return μ with unit total measure for each present M/E role.

    M records use a discrete equal measure. Eligible E records use their
    positive physical cell measures normalized within E. Empty roles remain
    all zero and do not borrow mass from the other role.
    """
    if module_mask.ndim != 2 or environment_mask.ndim != 2 or environment_measure.shape != environment_mask.shape:
        raise ValueError("M/E masks and E measure must have compatible [B,N] shapes.")
    if module_mask.shape[0] != environment_mask.shape[0]:
        raise ValueError("M and E member batches must agree.")
    if module_mask.dtype != torch.bool or environment_mask.dtype != torch.bool:
        raise TypeError("M/E member masks must be boolean.")
    dtype = dtype or environment_measure.dtype
    m_mask = module_mask
    e_measure = environment_measure.to(dtype)
    e_mask = environment_mask & (e_measure > 0)
    m_count = m_mask.sum(-1, keepdim=True).to(dtype).clamp_min(1.)
    e_total = torch.where(e_mask, e_measure, torch.zeros_like(e_measure)).sum(-1, keepdim=True)
    mu_m = m_mask.to(dtype) / m_count
    mu_e = torch.where(e_mask, e_measure / e_total.clamp_min(torch.finfo(dtype).tiny),
                       torch.zeros_like(e_measure))
    return torch.cat((mu_m, mu_e), dim=-1)


def build_joint_retrieval(
    H: Tensor,
    member_measure: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the whole-row normalized R and return (R, row mass, raw Hμ).

    μ is multiplied exactly once. Each complete M+E address row is normalized
    together; columns are never normalized separately.
    """
    if H.ndim != 3 or member_measure.ndim != 2 or H.shape[0] != member_measure.shape[0] or H.shape[2] != member_measure.shape[1]:
        raise ValueError("H and member measure must have shapes [B,A,N] and [B,N].")
    if not H.is_floating_point() or not member_measure.is_floating_point():
        raise TypeError("H and member measure must be floating point.")
    weighted = H * member_measure.to(H)[:, None, :]
    row_mass = weighted.sum(-1)
    retrieval = torch.where(row_mass[..., None] > 0,
                            weighted / row_mass[..., None].clamp_min(torch.finfo(H.dtype).tiny),
                            torch.zeros_like(weighted))
    return retrieval, row_mass, weighted


def assemble_joint_incidence(
    gate_mm: Tensor,
    gate_em: Tensor,
    gate_ee: Tensor,
    gate_me: Tensor,
    module_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Assemble stage-labelled four-block H with M identity as a separate ledger.

    Route labels are receiver-first. The inputs are the actual applied gates
    from the final contextual cycle: MM/M×M, EM/E×M, EE/E×E and ME/M×E.
    EE's captured diagonal is retained as-is; no second E self term is added.
    """
    if module_mask.ndim != 2 or module_mask.dtype != torch.bool:
        raise ValueError("module_mask must be boolean with shape [B,M].")
    if any(value.ndim != 3 for value in (gate_mm, gate_em, gate_ee, gate_me)):
        raise ValueError("All four applied relation blocks must have rank three.")
    batch, modules = module_mask.shape
    expected = {
        "MM": (batch, modules, modules),
        "EM": (batch, gate_em.shape[1], modules),
        "EE": (batch, gate_ee.shape[1], gate_ee.shape[1]),
        "ME": (batch, modules, gate_me.shape[2]),
    }
    blocks = {"MM": gate_mm, "EM": gate_em, "EE": gate_ee, "ME": gate_me}
    for name, value in blocks.items():
        if tuple(value.shape) != expected[name]:
            raise ValueError(f"Applied {name} gate has shape {tuple(value.shape)}, expected {expected[name]}.")
        if value.device != module_mask.device or not value.is_floating_point():
            raise ValueError(f"Applied {name} gate must be floating point on the scene device.")
        if bool((~torch.isfinite(value)).any()) or bool((value < 0).any()):
            raise ValueError(f"Applied {name} gate must be finite and nonnegative.")
    environments = gate_ee.shape[1]
    if gate_em.shape[1] != environments or gate_me.shape[2] != environments:
        raise ValueError("EM, EE and ME blocks must share the padded E inventory size.")

    H_exchange = torch.cat((torch.cat((gate_mm, gate_me), dim=-1),
                            torch.cat((gate_em, gate_ee), dim=-1)), dim=-2)
    H_self_M = torch.diag_embed(module_mask.to(gate_mm.dtype))
    H_self = torch.cat((torch.cat((H_self_M, gate_mm.new_zeros(batch, modules, environments)), dim=-1),
                        gate_mm.new_zeros(batch, environments, modules + environments)), dim=-2)
    H = H_exchange + H_self
    return H_exchange, H_self, H, blocks


def joint_member_prefix_mask(
    weights: Tensor,
    allowed: Tensor,
    module_ids: Tensor,
    environment_ids: Tensor,
    retained_mass: float = 0.95,
) -> Tensor:
    """Select one shortest M/E prefix, tied by (role, original ID).

    Role order is M then E; original integer IDs break ties within a role.
    Padding position never resolves a tie. The output is one union mask, to be
    split only after this selection is complete.
    """
    if weights.ndim != 3 or allowed.shape != weights.shape or allowed.dtype != torch.bool:
        raise ValueError("Joint weights and eligibility must share [B,Q,M+E] axes.")
    batch, queries, total = weights.shape
    modules = module_ids.shape[-1] if module_ids.ndim == 2 else -1
    environments = environment_ids.shape[-1] if environment_ids.ndim == 2 else -1
    if module_ids.shape != (batch, modules) or environment_ids.shape != (batch, environments) or modules + environments != total:
        raise ValueError("Typed original IDs do not match the joint member inventory.")
    if not 0 < float(retained_mass) <= 1:
        raise ValueError("retained_mass must be in (0,1].")
    if not total:
        return allowed.clone()

    order_m = torch.argsort(module_ids, dim=-1, stable=True)
    order_e = torch.argsort(environment_ids, dim=-1, stable=True) + modules
    by_identity = torch.cat((order_m, order_e), dim=-1)
    by_identity = by_identity[:, None, :].expand(batch, queries, total)
    identity_allowed = allowed.gather(-1, by_identity)
    identity_weights = weights.gather(-1, by_identity)
    sort_values = identity_weights.masked_fill(~identity_allowed, -torch.inf)
    sorted_order = torch.argsort(sort_values, dim=-1, descending=True, stable=True)
    ordered_indices = by_identity.gather(-1, sorted_order)
    sorted_weights = weights.gather(-1, ordered_indices)
    sorted_allowed = allowed.gather(-1, ordered_indices)
    total_mass = torch.where(allowed, weights, torch.zeros_like(weights)).sum(-1, keepdim=True)
    target = float(retained_mass) * total_mass
    mass_before = sorted_weights.cumsum(-1) - sorted_weights
    keep_sorted = sorted_allowed & (sorted_weights > 0) & (mass_before < target)
    return torch.zeros_like(allowed).scatter(-1, ordered_indices, keep_sorted)


class JointMemberRoutingCore(GuidedMemberRoutingCore):
    """R13-5 core with M-only or joint M/E addresses and joint95 access."""

    schema = _SCHEMA
    _additional_decode_count_keys = GuidedMemberRoutingCore._additional_decode_count_keys + (
        "query_E_proposal_score_rows", "query_joint_address_score_rows",
        "query_joint_address_base_measure_rows", "query_joint_address_prefix_sort_rows",
        "query_joint_address_prefix_id_sort_rows", "query_joint_prefix_sort_rows",
        "query_joint_prefix_id_sort_rows", "query_joint_member_coverage_rows",
        "query_joint_raw_composition_pair_rows", "query_joint_uncut_composition_pair_rows",
        "query_joint_value_reduction_rows", "query_joint_field_head_rows",
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
    ):
        if address_mode not in ("M", "ME"):
            raise ValueError("address_mode must be 'M' or 'ME'.")
        if isinstance(address_temperature, bool) or not isinstance(address_temperature, (int, float)) or not 0.05 <= float(address_temperature) <= 20:
            raise ValueError("address_temperature must be finite and in [0.05,20].")
        if isinstance(joint_member_coverage, bool) or not isinstance(joint_member_coverage, (int, float)) or not 0 < float(joint_member_coverage) <= 1:
            raise ValueError("joint_member_coverage must be finite and in (0,1].")
        self.address_mode = address_mode
        self.address_temperature = float(address_temperature)
        self.joint_member_coverage = float(joint_member_coverage)
        super().__init__(
            base_model_config,
            member_routing_mode="guide",
            input_dataset=input_dataset,
            input_capability=input_capability,
            # Parent construction remains state-compatible; proposal and
            # member temperature laws are implemented explicitly below.
            entmax_temperature=1.0,
            query_tile_size=query_tile_size,
            prior_mass_tolerance=1.0 - float(joint_member_coverage),
        )
        self.architecture = self._joint_arm_name(self.address_mode)

    @staticmethod
    def _joint_arm_name(address_mode: str) -> str:
        return f"R13-5-{address_mode}-Joint95"

    @property
    def architecture_name(self) -> str:
        return self._joint_arm_name(self.address_mode)

    def set_address_mode(self, address_mode: str) -> None:
        if address_mode not in ("M", "ME"):
            raise ValueError("address_mode must be 'M' or 'ME'.")
        self.address_mode = address_mode
        self.architecture = self._joint_arm_name(address_mode)

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
        }

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        required = {
            "schema", "architecture", "address_mode", "base_model_config", "input_dataset",
            "input_capability", "address_temperature", "joint_member_coverage", "query_tile_size",
        }
        if set(config) != required or config.get("schema") != cls.schema:
            raise ValueError("R13-5 joint M/E routing configuration inventory/schema differs.")
        mode = config["address_mode"]
        if mode not in ("M", "ME") or config["architecture"] != cls._joint_arm_name(mode):
            raise ValueError("R13-5 architecture and address_mode disagree.")
        temperature = config["address_temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0.05 <= float(temperature) <= 20:
            raise ValueError("R13-5 address temperature is outside its valid range.")
        coverage = config["joint_member_coverage"]
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 < float(coverage) <= 1:
            raise ValueError("R13-5 joint member coverage is outside its valid range.")
        if type(config["query_tile_size"]) is not int or config["query_tile_size"] < 1:
            raise ValueError("R13-5 query tile size must be positive.")
        translated = {
            "schema": GuidedMemberRoutingCore.schema,
            "architecture": "R13-4-Guide",
            "member_routing_mode": "guide",
            "base_model_config": config["base_model_config"],
            "input_dataset": config["input_dataset"],
            "input_capability": config["input_capability"],
            "entmax_temperature": 1.0,
            "query_tile_size": config["query_tile_size"],
            "prior_mass_tolerance": 1.0 - float(coverage),
        }
        GuidedMemberRoutingCore.validate_config(translated)

    def _finalize_native_preparation(self, organization, cycle_trace):
        """Capture the four final-cycle applied blocks and prepare live H/R."""
        organization = super()._finalize_native_preparation(organization, cycle_trace)
        cycle_trace = {} if cycle_trace is None else cycle_trace
        scene = organization.scene
        gates = {
            "mm": self._safe_snapshot_gate(cycle_trace, "mm", scene),
            "em": self._safe_snapshot_gate(cycle_trace, "em", scene),
            "ee": self._safe_snapshot_gate(cycle_trace, "ee", scene),
            "me": self._safe_snapshot_gate(cycle_trace, "me", scene),
        }
        expected_ids = {
            "mm": (scene.module_ids, scene.module_ids),
            "em": (scene.environment_ids, scene.module_ids),
            "ee": (scene.environment_ids, scene.environment_ids),
            "me": (scene.module_ids, scene.environment_ids),
        }
        snapshots = {}
        cycles = set()
        for route in _ROUTE_ORDER:
            snapshot = cycle_trace.get(route + "_native_route_snapshot")
            if snapshot is None:
                raise ValueError(f"Final contextual cycle lacks the actual {route.upper()} route snapshot.")
            if getattr(snapshot, "route", None) != route or not str(getattr(snapshot, "stage_identity", "")).endswith("/" + route):
                raise ValueError(f"Captured {route.upper()} provenance does not match its route/stage.")
            destination_ids, donor_ids = expected_ids[route]
            if not torch.equal(snapshot.destination_ids, destination_ids) or not torch.equal(snapshot.donor_ids, donor_ids):
                raise ValueError(f"Captured {route.upper()} IDs do not preserve the original typed inventory.")
            if snapshot.applied_incidence is not None and snapshot.applied_incidence.shape != gates[route].shape:
                raise ValueError(f"Captured {route.upper()} snapshot and applied gate axes disagree.")
            cycles.add(str(snapshot.stage_identity).rsplit("/", 1)[0])
            snapshots[route] = snapshot
        if len(cycles) != 1:
            raise ValueError("The four R13-5 relation blocks must come from one final contextual cycle.")

        H_exchange, H_self, H, blocks = assemble_joint_incidence(
            gates["mm"], gates["em"], gates["ee"], gates["me"], scene.module_mask,
        )
        mu = build_joint_member_measure(
            scene.module_mask, scene.environment_mask, scene.environment_measure, dtype=H.dtype,
        )
        retrieval_R, row_mass, raw_coefficients = build_joint_retrieval(H, mu)

        organization.H_MM_applied = blocks["MM"]
        organization.H_EM_applied = blocks["EM"]
        organization.H_EE_applied = blocks["EE"]
        organization.H_ME_applied = blocks["ME"]
        organization.H_exchange = H_exchange
        organization.H_self_M = H_self[:, :scene.num_modules, :scene.num_modules]
        organization.H_self = H_self
        organization.H = H
        organization.member_measure_joint = mu
        organization.retrieval_row_mass = row_mass
        organization.retrieval_raw_coefficients = raw_coefficients
        organization.retrieval_R = retrieval_R
        organization.joint_address_eligible = row_mass > 0
        organization.H_route_provenance = {
            _BLOCK_NAMES[route]: {
                "stage_identity": snapshots[route].stage_identity,
                "route": route,
                "receiver_role": "M" if route in ("mm", "me") else "E",
                "donor_role": "M" if route in ("mm", "em") else "E",
                "destination_ids": snapshots[route].destination_ids,
                "donor_ids": snapshots[route].donor_ids,
            }
            for route in _ROUTE_ORDER
        }
        organization.H_route_cycle_indices = torch.tensor(
            [int(snapshots[route].stage_identity.split("/", 1)[0].removeprefix("cycle"))
             for route in _ROUTE_ORDER], dtype=torch.long, device=H.device,
        )
        organization.work_counts.update({
            "prepared_joint_H_block_rows": int(sum(value.numel() for value in blocks.values())),
            "prepared_joint_H_retrieval_normalization_rows": int(H.shape[0] * H.shape[1]),
            "prepared_joint_H_positive_edges": int((H > 0).sum()),
        })
        return organization

    @staticmethod
    def _slice_intervention(intervention: Mapping[str, Any] | None, full_queries: int,
                            start: int, stop: int) -> dict[str, Any] | None:
        """Slice joint-only query controls alongside the inherited controls."""
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

    def _address_logits(self, prepared, query, qstates, address_allowed):
        """Score M/E addresses with one shared proposal_M scorer."""
        scene = prepared.scene
        batch, queries, address_count = address_allowed.shape
        modules, environments = scene.num_modules, scene.num_environment_tokens
        if address_count != modules + environments:
            raise ValueError("Joint address eligibility does not match the padded M/E inventories.")
        scores_m = qstates.new_zeros(batch, queries, modules)
        scores_e = qstates.new_zeros(batch, queries, environments)
        normalized_query = self._routing_features(qstates)
        rows_m = self._pair_rows(address_allowed[..., :modules])
        if rows_m.numel():
            b, q, m = rows_m.unbind(-1)
            geometry = self._pair_geometry(scene, query, "M", rows_m)
            features = torch.cat((normalized_query[b, q],
                                  self._routing_features(prepared.member_keys_M[b, m]), geometry), -1)
            scores = self.member_router.proposal_M(features).squeeze(-1)
            scores_m = scores_m.index_put((b, q, m), scores)
        rows_e = self._pair_rows(address_allowed[..., modules:])
        if rows_e.numel():
            b, q, e = rows_e.unbind(-1)
            geometry = self._pair_geometry(scene, query, "E", rows_e)
            features = torch.cat((normalized_query[b, q],
                                  self._routing_features(prepared.member_keys_E[b, e]), geometry), -1)
            scores = self.member_router.proposal_M(features).squeeze(-1)
            scores_e = scores_e.index_put((b, q, e), scores)
        return torch.cat((scores_m, scores_e), dim=-1), scores_m, scores_e

    @staticmethod
    def _address_base_measure(scene, address_eligible, address_mode):
        modules, environments = scene.num_modules, scene.num_environment_tokens
        allowed = address_eligible.clone()
        if address_mode == "M":
            allowed[..., modules:] = False
        m_allowed = allowed[..., :modules]
        e_allowed = allowed[..., modules:]
        m_count = m_allowed.sum(-1, keepdim=True).to(scene.module_features.dtype).clamp_min(1.)
        omega_m = m_allowed.to(scene.module_features.dtype) / m_count
        raw_e_measure = scene.environment_measure.to(scene.module_features)
        e_weights = raw_e_measure[:, None, :].expand_as(e_allowed) * e_allowed.to(raw_e_measure.dtype)
        e_total = e_weights.sum(-1, keepdim=True)
        omega_e = torch.where(e_total > 0,
                              e_weights / e_total.clamp_min(torch.finfo(e_weights.dtype).tiny),
                              torch.zeros_like(e_weights))
        return torch.cat((omega_m, omega_e), dim=-1), allowed

    @staticmethod
    def _member_support(pi, retrieval):
        if retrieval.shape[1] == 0 or retrieval.shape[2] == 0:
            return pi.new_zeros(pi.shape[0], pi.shape[1], retrieval.shape[2], dtype=torch.bool)
        return torch.bmm((pi > 0).to(pi.dtype), (retrieval > 0).to(pi.dtype)) > 0

    def _typed_read_state(self, prepared, query, qstates, intervention, *, return_trace):
        scene = prepared.scene
        controls = {} if intervention is None else dict(intervention)
        legal = {
            "read_mode", "member_score_mode", "value_execution", "fixed_pi", "fixed_address_mask",
            "fixed_candidate_mask_joint", "fixed_member_logits_M", "fixed_member_logits_E",
            "retrieval_mode", "address_coverage",
        }
        unknown = set(controls) - legal
        if unknown:
            raise ValueError(f"Unknown R13-5 routing interventions: {sorted(unknown)}")
        mode = controls.get("read_mode", self._read_execution_override or "full")
        if mode not in ("full", "restricted"):
            raise ValueError("R13-5 read_mode must be 'full' or 'restricted'.")
        score_mode = controls.get("member_score_mode", "learned")
        if score_mode not in ("learned", "geometry"):
            raise ValueError("member_score_mode must be 'learned' or 'geometry'.")
        execution = controls.get("value_execution", "selected")
        if execution not in ("selected", "dense_candidate_union"):
            raise ValueError("value_execution must be selected or dense_candidate_union.")
        retrieval_mode = controls.get("retrieval_mode", "normalized")
        if retrieval_mode not in ("normalized", "raw"):
            raise ValueError("retrieval_mode must be 'normalized' or 'raw'.")
        address_coverage = controls.get("address_coverage")
        if address_coverage is not None and (
                isinstance(address_coverage, bool)
                or not isinstance(address_coverage, (int, float))
                or not 0 < float(address_coverage) <= 1):
            raise ValueError("address_coverage must be None or a number in (0,1].")

        batch, queries = query.batch_size, query.num_queries
        modules, environments = scene.num_modules, scene.num_environment_tokens
        total = modules + environments
        query_mask = self._query_mask(query)
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
            prepared, query, qstates, address_allowed,
        )
        address_score_rows_M = int((address_allowed[..., :modules]).sum())
        address_score_rows_E = int((address_allowed[..., modules:]).sum())
        address_omega, role_allowed = self._address_base_measure(scene, address_allowed, self.address_mode)
        address_logits = address_scores / self.address_temperature
        address_log_measure = torch.where(role_allowed,
                                          address_omega.clamp_min(torch.finfo(address_omega.dtype).tiny).log(),
                                          torch.zeros_like(address_omega))
        proposal_logits = address_logits + address_log_measure
        pi = _masked_softmax(proposal_logits, role_allowed)
        fixed_pi = controls.get("fixed_pi")
        if fixed_pi is not None:
            if fixed_pi.shape != proposal_logits.shape or fixed_pi.device != proposal_logits.device:
                raise ValueError("fixed_pi must have shape [B,Q,M+E] on the query device.")
            if bool((~torch.isfinite(fixed_pi) | (fixed_pi < 0) | ((fixed_pi > 0) & ~role_allowed)).any()):
                raise ValueError("fixed_pi must be finite, nonnegative and restricted to eligible addresses.")
            fixed_total = fixed_pi.sum(-1)
            has_address = role_allowed.any(-1)
            if bool(((has_address & ((fixed_total - 1.).abs() > 1.e-4)) | (~has_address & (fixed_total > 1.e-6))).any()):
                raise ValueError("fixed_pi must sum to one when addresses are eligible and zero otherwise.")
            pi = fixed_pi.to(qstates)

        active_address_count = modules if self.address_mode == "M" else total
        pi_uncut = pi
        address_cut_mask = role_allowed
        address_transport_mass = None
        address_omitted_mass = None
        address_omitted_fraction = None
        if address_coverage is not None:
            if retrieval_mode == "normalized":
                address_transport_mass = pi_uncut
            else:
                address_transport_mass = pi_uncut * prepared.retrieval_row_mass[:, None, :]
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

        active_pi = pi[..., :active_address_count]
        active_pi_uncut = pi_uncut[..., :active_address_count]
        active_retrieval = prepared.retrieval_R[:, :active_address_count]
        active_raw_coefficients = prepared.retrieval_raw_coefficients[:, :active_address_count]
        raw_prior = None
        if retrieval_mode == "raw" or return_trace:
            raw_prior = torch.bmm(active_pi, active_raw_coefficients)
        if retrieval_mode == "normalized":
            member_prior = torch.bmm(active_pi, active_retrieval)
        else:
            member_prior = raw_prior
        member_valid = eligible_roles[:, None, :] & query_mask[..., None]
        member_support = self._member_support(active_pi, active_retrieval) & member_valid
        member_prior = _normalize_nonnegative(member_prior, member_valid & member_support)
        uncut_member_prior = None
        uncut_full_candidate = None
        if return_trace:
            uncut_full_candidate = self._member_support(active_pi_uncut, active_retrieval) & member_valid
            if address_coverage is None:
                uncut_member_prior = member_prior
            else:
                if retrieval_mode == "normalized":
                    uncut_composed = torch.bmm(active_pi_uncut, active_retrieval)
                else:
                    uncut_composed = torch.bmm(active_pi_uncut, active_raw_coefficients)
                uncut_member_prior = _normalize_nonnegative(
                    uncut_composed, member_valid & uncut_full_candidate,
                )
        # Use the support product to retain every actually reachable original
        # ID. Coalescing repeated address contributions happens in πR's member
        # columns, before any query-conditioned member scorer is evaluated.
        full_candidate = member_support
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
                raise ValueError("Fixed joint candidates must remain reachable through a live H path.")
            joint_candidate = fixed_candidate

        candidate_M = joint_candidate[..., :modules]
        candidate_E = joint_candidate[..., modules:]
        prior_M = member_prior[..., :modules]
        prior_E = member_prior[..., modules:]
        address_pairs = int(batch * queries * active_address_count * total)
        prefix_sort_rows = int(batch * queries * total) if mode == "restricted" else 0
        address_prefix_sort_rows = int(batch * queries * total) if address_coverage is not None else 0
        counts: dict[str, int] = {
            "query_M_proposal_score_rows": address_score_rows_M,
            "query_E_proposal_score_rows": address_score_rows_E,
            "query_joint_address_score_rows": address_score_rows_M + address_score_rows_E,
            "query_joint_address_base_measure_rows": int(role_allowed.sum()),
            "query_joint_address_prefix_sort_rows": address_prefix_sort_rows,
            "query_joint_address_prefix_id_sort_rows": int(batch * total) if address_coverage is not None else 0,
            "query_address_softmax_rows": int(role_allowed.sum()),
            "query_address_index_scan_rows": 0,
            "query_address_coefficient_pair_rows": address_pairs,
            "query_address_support_pair_rows": address_pairs,
            "query_address_relation_rows": int((
                (active_pi > 0).to(active_retrieval.dtype)
                * (active_retrieval > 0).sum(-1).to(active_retrieval.dtype)[:, None, :]
            ).sum()),
            "query_typed_read_gate_rows": 2 * batch * queries,
            "query_M_prior_composition_rows": batch * queries * active_address_count * modules,
            "query_E_prior_composition_rows": batch * queries * active_address_count * environments,
            # These count the member entries in the one joint P normalization,
            # partitioned by typed column for compatibility with older reports.
            "query_M_prior_normalization_rows": batch * queries * modules,
            "query_E_prior_normalization_rows": batch * queries * environments,
            "query_M_continuous_value_rows": int(candidate_M.sum()),
            "query_E_continuous_value_rows": int(candidate_E.sum()),
            "query_M_member_softmax_rows": int(candidate_M.sum()),
            "query_E_member_softmax_rows": int(candidate_E.sum()),
            "query_M_sorted_rows": 0,
            "query_E_sorted_rows": 0,
            "query_M_id_sort_rows": 0,
            "query_E_id_sort_rows": 0,
            "query_E_prior_prefix_sort_rows": 0,
            "query_E_prior_id_sort_rows": 0,
            "query_joint_prefix_sort_rows": prefix_sort_rows,
            "query_joint_prefix_id_sort_rows": int(batch * total) if mode == "restricted" else 0,
            "query_joint_member_coverage_rows": int(joint_candidate.sum()),
            "query_joint_raw_composition_pair_rows": address_pairs if raw_prior is not None else 0,
            "query_joint_uncut_composition_pair_rows": (
                address_pairs * (2 if retrieval_mode == "normalized" else 1)
                if return_trace and address_coverage is not None else 0
            ),
            "query_joint_value_reduction_rows": 0,
            "query_joint_field_head_rows": batch * queries,
            "query_routing_normalization_rows": int(
                role_allowed.sum() + batch * queries * total + joint_candidate.sum()
            ),
        }
        trace: dict[str, Tensor] = {}
        contexts: list[Tensor] = []
        role_states: dict[str, dict[str, Tensor]] = {}
        role_specs = (
            ("M", candidate_M, prior_M, prepared.member_keys_M, prepared.member_values_M, scene.module_ids),
            ("E", candidate_E, prior_E, prepared.member_keys_E, prepared.member_values_E, scene.environment_ids),
        )
        for role, candidate, prior, keys, values, original_ids in role_specs:
            raw_scores, scored_rows = self._member_base_logits(
                prepared, query, qstates, role, candidate, score_mode,
            )
            fixed_scores = controls.get(f"fixed_member_logits_{role}")
            if fixed_scores is not None:
                if fixed_scores.shape != raw_scores.shape or fixed_scores.device != raw_scores.device or bool((~torch.isfinite(fixed_scores)).any()):
                    raise ValueError(f"fixed_member_logits_{role} must be finite and preserve [B,Q,N] axes.")
                raw_scores = fixed_scores.to(qstates)
            conditional_prior = _normalize_nonnegative(prior, candidate)
            normalized_scores = prior_weighted_scores(raw_scores, conditional_prior, candidate)
            log_prior = torch.where(candidate,
                                    conditional_prior.clamp_min(torch.finfo(conditional_prior.dtype).tiny).log(),
                                    torch.zeros_like(conditional_prior))
            # R13-5 retains the R13-4 typed calibration and read-strength law;
            # member temperature is one in the declared equation.
            effective_scores = normalized_scores + log_prior
            relative = _positive_softmax(effective_scores, candidate)
            expectation = (relative * normalized_scores).sum(-1)
            entropy = -(relative * relative.clamp_min(torch.finfo(relative.dtype).tiny).log()).sum(-1)
            scene_count = (scene.module_mask if role == "M" else
                           scene.environment_mask & (scene.environment_measure > 0)).sum(-1)[:, None]
            entropy = torch.where(scene_count > 1,
                                  entropy / scene_count.to(qstates).clamp_min(2.).log(),
                                  torch.zeros_like(entropy))
            gate_inputs = torch.cat((self._routing_features(qstates), expectation[..., None], entropy[..., None]), -1)
            gate_logits = self.typed_read_gates[role](gate_inputs)
            contrast = gate_logits[..., 0] - gate_logits[..., 1]
            has_members = candidate.any(-1)
            strength = read_probability(contrast) * has_members.to(qstates) * query_mask.to(qstates)
            beta = strength[..., None] * relative
            execute = candidate if execution == "dense_candidate_union" else (candidate & (beta > 0))
            context = self._reduce_selected(values, beta, execute)
            contexts.append(context)

            retained_mass = (prior * candidate.to(prior.dtype)).sum(-1)
            full_role_mass = (member_prior[..., :modules] if role == "M" else member_prior[..., modules:]).sum(-1)
            role_states[role] = {
                "candidate": candidate,
                "raw_scores": raw_scores,
                "normalized_scores": normalized_scores,
                "conditional_prior": conditional_prior,
                "effective_scores": effective_scores,
                "relative": relative,
                "strength": strength,
                "beta": beta,
                "context": context,
                "retained_mass": retained_mass,
                "full_role_mass": full_role_mass,
                "gate_logits": gate_logits,
                "read_contrast": contrast,
                "gate_derivative": 0.5 / torch.hypot(contrast, torch.ones_like(contrast)).pow(3),
            }
            counts.update({
                f"query_{role}_cheap_score_rows": scored_rows,
                f"query_{role}_candidate_rows": int(candidate.sum()),
                f"query_{role}_eligible_candidate_rows": int(full_candidate[..., :modules].sum()) if role == "M" else int(full_candidate[..., modules:].sum()),
                f"query_{role}_selected_value_rows": int((beta > 0).sum()),
                f"query_{role}_value_execution_rows": int(execute.sum()),
                f"query_{role}_dense_candidate_value_rows": int(candidate.sum()) if execution == "dense_candidate_union" else 0,
                f"query_{role}_member_softmax_rows": int(candidate.sum()),
                f"query_{role}_score_calibration_rows": int(candidate.sum()),
                f"query_{role}_continuous_value_rows": int(candidate.sum()),
            })

        fused = torch.cat((qstates, *contexts), dim=-1)
        fields = self.member_router.joint_head(self.member_router.joint_trunk(fused))
        fields = fields * query_mask[..., None].to(fields.dtype)
        counts["query_joint_value_reduction_rows"] = (
            counts["query_M_value_execution_rows"] + counts["query_E_value_execution_rows"]
        )
        if return_trace:
            if address_coverage is None:
                trace_transport_mass = (
                    pi_uncut if retrieval_mode == "normalized"
                    else pi_uncut * prepared.retrieval_row_mass[:, None, :]
                )
                omitted_transport_mass = torch.zeros_like(pi.sum(-1))
                omitted_transport_fraction = torch.zeros_like(pi.sum(-1))
            else:
                trace_transport_mass = address_transport_mass
                omitted_transport_mass = address_omitted_mass
                omitted_transport_fraction = address_omitted_fraction
            if address_coverage is not None and retrieval_mode == "normalized":
                uncut_raw_prior = torch.bmm(active_pi_uncut, active_raw_coefficients)
            elif retrieval_mode == "raw":
                uncut_raw_prior = uncut_composed if address_coverage is not None else raw_prior
            else:
                uncut_raw_prior = raw_prior
            address_roles = torch.cat((torch.full((modules,), _ROLE_M, dtype=torch.long, device=qstates.device),
                                       torch.full((environments,), _ROLE_E, dtype=torch.long, device=qstates.device)))
            trace.update({
                "proposal_logits_joint": proposal_logits,
                "proposal_scores_joint": address_scores,
                "proposal_raw_scores_joint": address_scores,
                "proposal_logits_M": address_scores_M,
                "proposal_logits_E": address_scores_E,
                "proposal_scores_M": address_scores_M,
                "proposal_scores_E": address_scores_E,
                "proposal_temperature_scaled_scores_joint": address_logits,
                "proposal_base_measure": address_omega,
                "proposal_weights_joint": pi,
                "pi_joint": pi,
                "address_weights": pi,
                "proposal_weights_joint_uncut": pi_uncut,
                "pi_joint_uncut": pi_uncut,
                "proposal_weights_M": pi[..., :modules],
                "proposal_weights_E": pi[..., modules:],
                "pi_M": pi[..., :modules],
                "pi_E": pi[..., modules:],
                "proposal_weights_M_uncut": pi_uncut[..., :modules],
                "proposal_weights_E_uncut": pi_uncut[..., modules:],
                "proposal_base_measure_M": address_omega[..., :modules],
                "proposal_base_measure_E": address_omega[..., modules:],
                "address_role_codes": address_roles,
                "address_original_ids": torch.cat((scene.module_ids, scene.environment_ids), dim=-1),
                "address_eligible_mask": role_allowed,
                "address_candidate_mask": address_cut_mask,
                "address_transport_mass": trace_transport_mass,
                "address_omitted_transport_mass": omitted_transport_mass,
                "address_omitted_path_mass": omitted_transport_mass,
                "address_omitted_transport_fraction": omitted_transport_fraction,
                "address_coverage": qstates.new_tensor(-1.0 if address_coverage is None else float(address_coverage)),
                "prior_P": member_prior,
                "prior_P_joint": member_prior,
                "prior_P_uncut": uncut_member_prior,
                "uncut_prior_P": uncut_member_prior,
                "raw_prior_P_joint": _normalize_nonnegative(raw_prior, member_valid & member_support),
                "raw_prior_P_joint_uncut": _normalize_nonnegative(
                    uncut_raw_prior, member_valid & uncut_full_candidate,
                ),
                "retrieval_raw_prior": raw_prior,
                "retrieval_raw_prior_uncut": uncut_raw_prior,
                "retrieval_mode_code": torch.tensor(0 if retrieval_mode == "normalized" else 1,
                                                     dtype=torch.long, device=qstates.device),
                "full_eligible_mask": full_candidate,
                "full_eligible_mask_uncut": uncut_full_candidate,
                "joint_candidate_mask": joint_candidate,
                "candidate_mask_joint": joint_candidate,
                "full_eligible_mask_M": full_candidate[..., :modules],
                "full_eligible_mask_E": full_candidate[..., modules:],
                "candidate_mask_M": candidate_M,
                "candidate_mask_E": candidate_E,
                "chosen_candidate_mask_M": candidate_M,
                "chosen_candidate_mask_E": candidate_E,
                "prior_P_M": member_prior[..., :modules],
                "prior_P_E": member_prior[..., modules:],
                "member_prior_M": member_prior[..., :modules],
                "member_prior_E": member_prior[..., modules:],
                "original_module_ids": scene.module_ids,
                "original_environment_ids": scene.environment_ids,
                "member_measure_joint": prepared.member_measure_joint,
                "retrieval_row_mass": prepared.retrieval_row_mass,
                "retrieval_R": prepared.retrieval_R,
                "retrieval_raw_coefficients": prepared.retrieval_raw_coefficients,
                "H": prepared.H,
                "H_exchange": prepared.H_exchange,
                "H_self": prepared.H_self,
                "H_self_M": prepared.H_self_M,
                "H_MM_applied": prepared.H_MM_applied,
                "H_EM_applied": prepared.H_EM_applied,
                "H_EE_applied": prepared.H_EE_applied,
                "H_ME_applied": prepared.H_ME_applied,
                "H_route_cycle_indices": prepared.H_route_cycle_indices,
                "H_route_provenance_codes": torch.tensor(
                    [0, 1, 2, 3], dtype=torch.long, device=qstates.device,
                ),
                "candidate_ids_M": self._candidate_id_rows(candidate_M, scene.module_ids),
                "candidate_ids_E": self._candidate_id_rows(candidate_E, scene.environment_ids),
                "prior_role_mass_M": member_prior[..., :modules].sum(-1),
                "prior_role_mass_E": member_prior[..., modules:].sum(-1),
                "retained_joint_mass_M": (member_prior[..., :modules] * candidate_M).sum(-1),
                "retained_joint_mass_E": (member_prior[..., modules:] * candidate_E).sum(-1),
                "context_M": contexts[0],
                "context_E": contexts[1],
            })
            for route, row_start, row_count, col_start, col_count in (
                ("MM", 0, modules, 0, modules),
                ("ME", 0, modules, modules, environments),
                ("EM", modules, environments, 0, modules),
                ("EE", modules, environments, modules, environments),
            ):
                if route == "MM":
                    row_pi = pi[..., row_start:row_start + row_count]
                elif route == "ME":
                    row_pi = pi[..., row_start:row_start + row_count]
                else:
                    row_pi = pi[..., row_start:row_start + row_count]
                raw_block = prepared.H_exchange[:, row_start:row_start + row_count,
                                                col_start:col_start + col_count]
                block_measure = prepared.member_measure_joint[:, col_start:col_start + col_count]
                row_mass = prepared.retrieval_row_mass[:, row_start:row_start + row_count]
                block = torch.where(
                    row_mass[..., None] > 0,
                    raw_block * block_measure[:, None, :] / row_mass[..., None].clamp_min(
                        torch.finfo(raw_block.dtype).tiny),
                    torch.zeros_like(raw_block),
                )
                route_mass = (row_pi * block.sum(-1)[:, None, :]).sum(-1)
                trace[f"address_block_mass_{route}"] = route_mass
            self_block = prepared.H_self[:, :modules, :]
            self_block = self_block * prepared.member_measure_joint[:, None, :]
            self_block = torch.where(
                prepared.retrieval_row_mass[:, :modules, None] > 0,
                self_block / prepared.retrieval_row_mass[:, :modules, None].clamp_min(
                    torch.finfo(self_block.dtype).tiny),
                torch.zeros_like(self_block),
            )
            self_mass = (pi[..., :modules] * self_block.sum(-1)[:, None, :]).sum(-1)
            trace["address_block_mass_self_M"] = self_mass
            for block, provenance in prepared.H_route_provenance.items():
                trace[f"H_{block}_destination_ids"] = provenance["destination_ids"]
                trace[f"H_{block}_donor_ids"] = provenance["donor_ids"]
            for role, state in role_states.items():
                trace.update({
                    f"member_logits_{role}": state["raw_scores"],
                    f"member_normalized_scores_{role}": state["normalized_scores"],
                    f"member_conditional_prior_{role}": state["conditional_prior"],
                    f"member_effective_scores_{role}": state["effective_scores"],
                    f"conditional_preference_{role}": state["relative"],
                    f"member_relative_weights_{role}": state["relative"],
                    f"member_weights_{role}": state["beta"],
                    f"member_read_strength_{role}": state["strength"],
                    f"member_read_logits_{role}": state["gate_logits"],
                    f"member_read_contrast_{role}": state["read_contrast"],
                    f"member_gate_derivative_{role}": state["gate_derivative"],
                    f"prior_role_mass_{role}": state["full_role_mass"],
                    f"retained_joint_mass_{role}": state["retained_mass"],
                    f"prior_omitted_mass_{role}": (
                        state["full_role_mass"] - state["retained_mass"]
                    ).clamp_min(0.),
                    f"selected_value_indices_{role}": (state["beta"] > 0).nonzero(as_tuple=False),
                })

        counts["query_receiver_rows"] = int(batch * queries)
        return fused, trace, counts, fields

    @staticmethod
    def _candidate_id_rows(mask: Tensor, ids: Tensor) -> Tensor:
        rows = mask.nonzero(as_tuple=False)
        if not rows.numel():
            return torch.empty((0, 4), dtype=ids.dtype, device=ids.device)
        batch, query, position = rows.unbind(-1)
        original = ids[batch, position]
        return torch.cat((rows.to(ids.dtype), original[:, None]), dim=-1)

__all__ = [
    "JointMemberRoutingCore",
    "assemble_joint_incidence",
    "build_joint_member_measure",
    "build_joint_retrieval",
    "joint_member_prefix_mask",
]
