"""Target-free typed organization and source-preserving access algebra.

Group paths are reduced to one density and one control moment per physical
receiver/source pair *before* a fine kernel or a source-union attention read.
This module never replaces physical source values with group summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .types import EncodedInterfaceCase

MECHANISMS = ("MM", "ME", "EM", "QM", "QE")
SOURCE_TYPE = {"MM": "M", "ME": "E", "EM": "M", "QM": "M", "QE": "E"}


def pad_geometry(coordinates: torch.Tensor) -> torch.Tensor:
    """Signed xyz plus explicit axis-presence flags; y never becomes z."""
    dimension = int(coordinates.shape[-1])
    if dimension not in (2, 3):
        raise ValueError("Typed hypergraphs support spatial dimensions 2 and 3.")
    padded = torch.nn.functional.pad(coordinates, (0, 3 - dimension))
    axes = coordinates.new_tensor([1.0, 1.0, float(dimension == 3)])
    return torch.cat((padded, axes.expand_as(padded)), dim=-1)


def masked_mean(values: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted source mean, returning a masked zero for an empty population."""
    valid = valid.to(dtype=torch.bool)
    mass = valid.to(values.dtype) if weights is None else torch.where(valid, weights, torch.zeros_like(weights))
    selected = torch.where(valid[..., None], values, torch.zeros_like(values))
    total = mass.sum(dim=-1, keepdim=True)
    return (selected * mass[..., None]).sum(dim=-2) / torch.where(total > 0, total, torch.ones_like(total))


def masked_max(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A negative-valued valid population is never dominated by fake zeros."""
    present = valid.to(torch.bool).any(dim=-1)
    if values.shape[-2] == 0:
        return values.new_zeros((*values.shape[:-2], values.shape[-1])), present
    maximum = torch.where(valid[..., None], values, values.new_full((), -torch.inf)).amax(dim=-2)
    return torch.where(present[..., None], maximum, torch.zeros_like(maximum)), present


@dataclass(frozen=True)
class TypedSourceAccess:
    density: torch.Tensor
    weight: torch.Tensor
    control: torch.Tensor
    support: torch.Tensor
    edge_access: torch.Tensor
    near: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedProjectedAction:
    """Phase-current linear actions, before receiver expansion and bias.

    This is an internal numerical record. It contains no physical field values
    and must be rebuilt after changes to weights or phase-current states.
    """
    source_moment: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class ProjectedSourceAccess:
    """Reader action with small projected channels; not a graph export."""
    density: torch.Tensor
    weight: torch.Tensor
    projected: torch.Tensor
    support: torch.Tensor
    edge_access: torch.Tensor
    near: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


def prepare_projected_action(membership, controls, weight, bias, *, detach_projection=False):
    """Project each group once, retaining the wide permission moment chain."""
    if detach_projection:
        weight, bias = weight.detach(), bias.detach()
    linear = torch.nn.functional.linear(controls.to(weight.dtype), weight)
    dtype = torch.promote_types(membership.dtype, linear.dtype)
    moment = membership.to(dtype)[..., None] * linear.to(dtype)[:, :, None, :]
    return PreparedProjectedAction(moment, bias)


@dataclass(frozen=True)
class TypedHypergraphState:
    memberships: dict[str, torch.Tensor]
    controls: dict[str, torch.Tensor]
    centres: torch.Tensor
    admission: torch.Tensor
    source_coords: dict[str, torch.Tensor]
    source_measures: dict[str, torch.Tensor]
    source_valid: dict[str, torch.Tensor]
    source_ids: dict[str, torch.Tensor]
    phase: int = 0
    source_lengths: dict[str, torch.Tensor] | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    strategy_data: dict[str, Any] = field(default_factory=dict)
    # Value donors alone do not disclose collective control information.
    control_memberships: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    control_presence: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    dependency_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def group_count(self) -> int:
        return int(self.admission.shape[-1])

    def export(self) -> dict[str, Any]:
        """Live differentiable input-only state for frozen-forward inverse reuse.

        Connectivity is explicit; callers can freeze it for a local proposal
        and rebuild after that proposal. Continuous geometry remains live.
        Receiver access is supplied by the owning organizer, not a target or
        a case-ID lookup.
        """
        return {
            "source_ids": self.source_ids,
            "source_types": tuple(self.source_ids),
            "source_coords": self.source_coords,
            "source_measures": self.source_measures,
            "source_valid": self.source_valid,
            "source_lengths": self.source_lengths,
            "group_admission": self.admission,
            "source_membership": self.memberships,
            "value_membership": self.memberships,
            "control_membership": self.control_memberships,
            "control_presence": self.control_presence,
            "group_controls": self.controls,
            "group_centres": self.centres,
            "phase": self.phase,
            "structural_measure_policy_version": self.strategy_data.get("structural_measure_policy_version"),
            "topology_semantics": "input-recomputed; freeze connectivity for a local inverse proposal",
            "control_input_provenance": ", ".join(self.dependency_provenance["control_content"])
                if self.dependency_provenance else "phase-current source states, signed geometry, declared global context, phase",
            "diagnostics": self.diagnostics,
            "dependency_provenance": self.dependency_provenance,
        }


def build_source_catalogue(encoded: EncodedInterfaceCase, module_states: torch.Tensor | None = None) -> dict[str, Any]:
    """Adapter-owned coordinates, quadrature, lengths and physical slot IDs."""
    del module_states
    valid_m = encoded.module_present > 0.5
    valid_e = encoded.env_weights > 0
    ids = {}
    for kind, valid in (("M", valid_m), ("E", valid_e)):
        supplied = getattr(encoded, "module_source_ids" if kind == "M" else "env_source_ids", None)
        physical = torch.arange(valid.shape[-1], device=valid.device).expand(valid.shape[0], -1) if supplied is None else supplied
        ids[kind] = torch.where(valid, physical, torch.full_like(physical, -1))
    lengths = {}
    for kind, attr in (("M", "module_characteristic_lengths"), ("E", "env_characteristic_lengths")):
        value = getattr(encoded, attr, None)
        if value is not None:
            lengths[kind] = value
    return {
        "source_coords": {"M": encoded.module_centers, "E": encoded.env_coords},
        "source_measures": {"M": torch.where(valid_m, encoded.module_present, torch.zeros_like(encoded.module_present)),
                            "E": torch.where(valid_e, encoded.env_weights, torch.zeros_like(encoded.env_weights))},
        "source_valid": {"M": valid_m, "E": valid_e},
        "source_ids": ids,
        "source_lengths": lengths or None,
    }


def source_moments(
    edge_access: torch.Tensor,
    membership: torch.Tensor,
    controls: torch.Tensor,
    source_measures: torch.Tensor,
    source_valid: torch.Tensor,
    *,
    pair_valid: torch.Tensor | None = None,
    near: torch.Tensor | None = None,
    prepared_action: PreparedProjectedAction | None = None,
    include_diagnostics: bool = True,
) -> TypedSourceAccess | ProjectedSourceAccess:
    """Deduplicate group paths and normalize access density on eligible mass.

    Uniform positive density gives weight one independently of K. A local
    envelope uses ``near + (1-near)*far_weight`` and attenuates far controls.
    Pair validity can exclude MM self pairs before normalization.
    """
    if edge_access.ndim != 3 or membership.ndim != 3 or controls.ndim != 3:
        raise ValueError("access, membership and controls must be [B,R,K], [B,K,S], [B,K,C].")
    if edge_access.shape[0] != membership.shape[0] or edge_access.shape[-1] != membership.shape[1] or controls.shape[:2] != membership.shape[:2]:
        raise ValueError("Typed group axes must align.")
    if source_measures.shape != membership.shape[:1] + membership.shape[2:] or source_valid.shape != source_measures.shape:
        raise ValueError("Source measures and validity must align on [B,S].")
    # Soft organizers supply FP64 permissions. Keep the complete union,
    # quotient and normalization chain in that dtype, including controls and
    # physical measures; casting before a quotient can overflow its VJP even
    # when the normalized forward value and final score gradient are finite.
    permission_dtype = torch.promote_types(edge_access.dtype, membership.dtype)
    if permission_dtype == torch.float64:
        edge_access = edge_access.to(permission_dtype)
        membership = membership.to(permission_dtype)
        controls = controls.to(permission_dtype)
        source_measures = source_measures.to(permission_dtype)
        if near is not None:
            near = near.to(permission_dtype)
    valid = source_valid[:, None, :].to(torch.bool)
    if pair_valid is not None:
        valid = valid & torch.broadcast_to(pair_valid.to(torch.bool), (edge_access.shape[0], edge_access.shape[1], membership.shape[2]))
    membership = torch.where(source_valid[:, None, :], membership, torch.zeros_like(membership))
    density = torch.bmm(edge_access, membership)
    density = torch.where(valid, density, torch.zeros_like(density))
    if prepared_action is None:
        moment = torch.einsum("brk,bks,bkc->brsc", edge_access, membership, controls)
    else:
        # The prepared [B,K,S,P] tensor has P=1 or 2*heads, never C=16.
        source_moment = prepared_action.source_moment.to(permission_dtype)
        moment = torch.bmm(edge_access, source_moment.flatten(2)).reshape(
            *density.shape, source_moment.shape[-1])
    positive = density > 0
    safe_density = torch.where(positive, density, torch.ones_like(density))
    control = torch.where(positive[..., None], moment / safe_density[..., None], torch.zeros_like(moment))
    measure = torch.where(valid, source_measures[:, None, :], torch.zeros_like(density))
    total = measure.sum(dim=-1, keepdim=True)
    mean = (measure * density).sum(dim=-1, keepdim=True) / torch.where(total > 0, total, torch.ones_like(total))
    weight = torch.where(mean > 0, density / torch.where(mean > 0, mean, torch.ones_like(mean)), torch.zeros_like(density))
    if near is not None:
        near = torch.where(valid, torch.broadcast_to(near, density.shape), torch.zeros_like(density))
        weight = near + (1.0 - near) * weight
        control = control * (1.0 - near[..., None])
    weight = torch.where(valid, weight, torch.zeros_like(weight))
    support = valid & (weight > 0)
    diagnostics = {
        "eligible_pairs": valid.expand_as(density).sum(),
        "pair_valid": valid.expand_as(density),
    }
    if include_diagnostics:
        paths = torch.bmm((edge_access > 0).to(density.dtype), (membership > 0).to(density.dtype))
        diagnostics["unique_pairs"] = support.sum()
        diagnostics["far_unique_pairs"] = (valid & positive).sum()
        diagnostics["repeated_paths_removed"] = (torch.where(valid, paths, torch.zeros_like(paths)) - (valid & positive).to(paths.dtype)).sum()
    if near is not None and include_diagnostics:
        diagnostics["near_mandatory_pairs"] = (near > 0).sum()
        diagnostics["near_full_pairs"] = (near == 1).sum()
    if prepared_action is not None:
        # Original zero controls project to bias even on newly admitted pairs.
        projected = control.to(prepared_action.bias.dtype) + prepared_action.bias
        return ProjectedSourceAccess(density, weight, projected, support, edge_access, near, diagnostics)
    return TypedSourceAccess(density, weight, control, support, edge_access, near, diagnostics)


def smooth_near_envelope(receivers: torch.Tensor, sources: torch.Tensor, source_lengths: torch.Tensor, *, inner: float = 1.0, outer: float = 2.0) -> torch.Tensor:
    """One physical length is fully admitted, two lengths have zero access."""
    if not 0 <= float(inner) < float(outer):
        raise ValueError("Near radii must satisfy 0 <= inner < outer.")
    if bool((source_lengths <= 0).any()) or not bool(torch.isfinite(source_lengths).all()):
        raise ValueError("Source characteristic lengths must be positive and finite.")
    distance = torch.linalg.vector_norm(receivers[:, :, None] - sources[:, None], dim=-1)
    fraction = ((distance / source_lengths[:, None] - inner) / (outer - inner)).clamp(0, 1)
    return 1.0 - fraction.square() * (3.0 - 2.0 * fraction)


def structural_pressure(epoch: int) -> float:
    """Absolute-epoch ramp; resuming at 100 never resets the curriculum."""
    return min(1.0, max(0.0, (int(epoch) - 25) / 75.0))


def _case_measure_fraction(values, measure, valid):
    """Normalize within each eligible case before averaging cases."""
    measure = torch.where(valid, torch.broadcast_to(measure, values.shape), torch.zeros_like(values))
    axes = tuple(range(1, values.ndim))
    total = measure.sum(axes)
    integral = (torch.where(valid, values, torch.zeros_like(values)) * measure).sum(axes)
    present = total > 0
    fraction = integral / torch.where(present, total, torch.ones_like(total))
    return fraction.sum() / present.to(fraction.dtype).sum().clamp_min(1)


def _measure_structural_cost(accesses, state, incidence_coefficient, group_coefficient, preparation=None):
    """Physical pair integral and dual-donor pooling occupancy, case balanced."""
    version = state.strategy_data.get("structural_measure_policy_version", 2)
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError("Structural measure policy version must be 1 or 2")

    def eligible_mean(terms, eligible):
        if not terms:
            return state.admission.sum() * 0
        if version == 1:
            # Preserve the exact e1-100 Tree-F3204 reduction. Version two is
            # an explicit objective amendment, never an implicit reload fix.
            return torch.stack(terms).mean()
        flags = torch.stack(eligible).to(terms[0].dtype)
        return (torch.stack(terms) * flags).sum() / flags.sum().clamp_min(1)

    terms, pair_eligible, metrics = [], [], {}
    if preparation is not None:
        terms = list(preparation["pair_terms"])
        pair_eligible = list(preparation["pair_eligible"])
        metrics = dict(preparation["metrics"])
    for tau, access in accesses.items():
        if preparation is not None and tau in preparation["mechanisms"]:
            continue
        kind = SOURCE_TYPE[tau]
        valid = access.diagnostics.get("pair_valid", state.source_valid[kind][:, None].expand_as(access.density))
        receiver_mass = access.diagnostics.get("receiver_measures", state.strategy_data.get("receiver_measures", {}).get(tau))
        if receiver_mass is None:
            receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else None
            # Query counting measure is declared separately from environmental
            # quadrature. Native random-query training uses equal query mass.
            receiver_mass = (state.source_measures[receiver_kind] if receiver_kind is not None
                             else access.density.new_ones(access.density.shape[:2]))
        if receiver_mass.shape != access.density.shape[:2]:
            raise ValueError("Structural receiver measures must align on [B,R]")
        product_measure = receiver_mass[..., None] * state.source_measures[kind][:, None]
        occupancy = -torch.expm1(-3.0 * access.density)
        if access.near is not None:
            occupancy = access.near + (1.0 - access.near) * occupancy
        term = _case_measure_fraction(occupancy, product_measure, valid)
        terms.append(term)
        pair_eligible.append((valid & (product_measure > 0)).any())
        metrics[f"{tau}_smooth_pair_fraction"] = term
        metrics[f"{tau}_unique_pairs"] = access.support.sum()
        metrics[f"{tau}_control_pool_rows"] = access.density.new_zeros((), dtype=torch.long)
    anchor = state.admission
    pair_cost = eligible_mean(terms, pair_eligible)
    if preparation is not None:
        metrics["smooth_pair_cost"] = pair_cost
        constant = (float(incidence_coefficient) * metrics["smooth_incidence_cost"]
                    + float(group_coefficient) * metrics["smooth_group_cost"])
        return pair_cost + constant, metrics

    def incidence_fraction(tau, kind, member):
        active = state.strategy_data.get("typed_admission", {}).get(tau, anchor)
        valid_nodes = state.strategy_data.get("node_valid", {}).get(tau)
        if valid_nodes is None:
            trees = state.strategy_data.get("trees", {}).get(tau)
            if trees is not None:
                lengths = member.new_tensor([len(tree.nodes) for tree in trees], dtype=torch.long)
                valid_nodes = torch.arange(member.shape[1], device=member.device)[None] < lengths[:, None]
            else:
                valid_nodes = torch.ones_like(active, dtype=torch.bool)
        valid = state.source_valid[kind][:, None] & valid_nodes[..., None]
        occupancy = -torch.expm1(-3.0 * member) * active[..., None]
        mass = state.source_measures[kind][:, None]
        # The organizer pools ALL candidate nodes before frontier selection.
        # Count padded source slots that the packed reduction actually visits;
        # padded node slots are appended only after these computations.
        rows = valid_nodes.sum() * member.shape[-1]
        return _case_measure_fraction(occupancy, mass, valid), rows, (valid & (mass > 0)).any()

    value_fractions = [incidence_fraction(tau, SOURCE_TYPE[tau], member)
                       for tau, member in state.memberships.items()]
    donors = getattr(state, "control_memberships", {}) or state.strategy_data.get("control_memberships", {})
    control_terms, control_eligible = [], []
    for tau, typed_members in donors.items():
        typed_terms, typed_eligible, rows = [], [], anchor.new_zeros((), dtype=torch.long)
        for kind, member in typed_members.items():
            term, count, present = incidence_fraction(tau, kind, member)
            typed_terms.append(term)
            typed_eligible.append(present)
            rows = rows + count
        if typed_terms:
            control_terms.append(eligible_mean(typed_terms, typed_eligible))
            control_eligible.append(torch.stack(typed_eligible).any())
        metrics[f"{tau}_control_pool_rows"] = rows
    value_incidence = eligible_mean([item[0] for item in value_fractions], [item[2] for item in value_fractions])
    control_incidence = eligible_mean(control_terms, control_eligible) if control_terms else value_incidence
    # The active-frontier penalty remains separate from physical access and
    # control-pooling occupancy. Capacity is declared, never source row count.
    groups = anchor.mean()
    cost = pair_cost + float(incidence_coefficient) * control_incidence + float(group_coefficient) * groups
    metrics.update(smooth_pair_cost=pair_cost, smooth_incidence_cost=control_incidence,
                   smooth_value_incidence_cost=value_incidence, smooth_control_incidence_cost=control_incidence,
                   smooth_group_cost=groups)
    return cost, metrics


def structural_cost(accesses: dict[str, TypedSourceAccess], state: TypedHypergraphState, *, incidence_coefficient: float = 0.05, group_coefficient: float = 0.01, preparation=None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced smooth work proxy; exact support counts stay separate.

    A single dense edge pays for dense pair occupancy, rather than earning
    sparsity merely by being called K=1. This is not an executor time model.
    """
    if state.strategy_data.get("measure_consistent", False):
        return _measure_structural_cost(accesses, state, incidence_coefficient, group_coefficient, preparation)
    terms = [] if preparation is None else list(preparation["pair_terms"])
    metrics = {} if preparation is None else dict(preparation["metrics"])
    for mechanism, access in accesses.items():
        if preparation is not None and mechanism in preparation["mechanisms"]:
            continue
        kind = SOURCE_TYPE[mechanism]
        valid = access.diagnostics.get("pair_valid", state.source_valid[kind][:, None].expand_as(access.density))
        count = access.diagnostics["eligible_pairs"].clamp_min(1)
        # Memberships are physical-measure densities with all-source mean
        # one, not source probabilities of order 1/S. Multiplying by S would
        # saturate the QE192 proxy and eliminate its structural gradients.
        # Three is an initial training-surrogate scale, not a calibrated
        # physical work or latency constant.
        occupancy = 1.0 - torch.exp(-3.0 * access.density)
        if access.near is not None:
            occupancy = access.near + (1.0 - access.near) * occupancy
        term = torch.where(valid, occupancy, torch.zeros_like(occupancy)).sum() / count
        terms.append(term)
        metrics[f"{mechanism}_smooth_pair_fraction"] = term
        metrics[f"{mechanism}_unique_pairs"] = access.support.sum()
    anchor = state.admission
    pair_cost = torch.stack(terms).mean() if terms else anchor.sum() * 0.0
    if preparation is not None:
        metrics["smooth_pair_cost"] = pair_cost
        constant = (float(incidence_coefficient) * metrics["smooth_incidence_cost"]
                    + float(group_coefficient) * metrics["smooth_group_cost"])
        return pair_cost + constant, metrics
    incidence_terms = []
    for mechanism, member in state.memberships.items():
        valid_sources = state.source_valid[SOURCE_TYPE[mechanism]]
        eligible = valid_sources[:, None].expand_as(member)
        occupancy = 1.0 - torch.exp(-3.0 * member)
        active = state.strategy_data.get("typed_admission", {}).get(mechanism, state.admission)
        weighted = torch.where(eligible, occupancy, torch.zeros_like(occupancy)) * active[..., None]
        incidence_terms.append(weighted.sum() / eligible.sum().clamp_min(1))
    incidence = torch.stack(incidence_terms).mean()
    groups = anchor.mean()
    cost = pair_cost + float(incidence_coefficient) * incidence + float(group_coefficient) * groups
    metrics.update(smooth_pair_cost=pair_cost, smooth_incidence_cost=incidence, smooth_group_cost=groups)
    return cost, metrics


def prepare_structural_cost(accesses, state, *, incidence_coefficient=0.05, group_coefficient=0.01):
    """Cache live phase-constant terms; preserve the original per-chunk loss.

    Read chunks still contribute cost*query_count to the existing aggregation.
    Candidate incidence and preparation pair penalties are computed once and
    receive the identical aggregate weighting, rather than one extra copy per
    receiver chunk.
    """
    _cost, metrics = structural_cost(accesses, state,
        incidence_coefficient=incidence_coefficient, group_coefficient=group_coefficient)
    terms, eligible = [], []
    for tau, access in accesses.items():
        terms.append(metrics[f"{tau}_smooth_pair_fraction"])
        kind = SOURCE_TYPE[tau]
        valid = access.diagnostics["pair_valid"]
        receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else None
        receiver_mass = access.diagnostics.get("receiver_measures",
            state.strategy_data.get("receiver_measures", {}).get(tau))
        if receiver_mass is None:
            receiver_mass = (state.source_measures[receiver_kind] if receiver_kind is not None
                             else access.density.new_ones(access.density.shape[:2]))
        measure = receiver_mass[..., None] * state.source_measures[kind][:, None]
        eligible.append((valid & (measure > 0)).any())
    return {"mechanisms": tuple(accesses), "pair_terms": tuple(terms),
            "pair_eligible": tuple(eligible), "metrics": metrics}


__all__ = ["MECHANISMS", "SOURCE_TYPE", "PreparedProjectedAction", "ProjectedSourceAccess", "TypedHypergraphState", "TypedSourceAccess", "build_source_catalogue", "masked_max", "masked_mean", "pad_geometry", "prepare_projected_action", "prepare_structural_cost", "smooth_near_envelope", "source_moments", "structural_cost", "structural_pressure"]
