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
            "group_controls": self.controls,
            "group_centres": self.centres,
            "phase": self.phase,
            "topology_semantics": "input-recomputed; freeze connectivity for a local inverse proposal",
            "control_input_provenance": "pre-interaction source states, signed geometry, prescribed global context, phase",
            "diagnostics": self.diagnostics,
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
) -> TypedSourceAccess:
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
    valid = source_valid[:, None, :].to(torch.bool)
    if pair_valid is not None:
        valid = valid & torch.broadcast_to(pair_valid.to(torch.bool), (edge_access.shape[0], edge_access.shape[1], membership.shape[2]))
    membership = torch.where(source_valid[:, None, :], membership, torch.zeros_like(membership))
    density = torch.einsum("brk,bks->brs", edge_access, membership)
    density = torch.where(valid, density, torch.zeros_like(density))
    moment = torch.einsum("brk,bks,bkc->brsc", edge_access, membership, controls)
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
    paths = torch.einsum("brk,bks->brs", (edge_access > 0).to(density.dtype), (membership > 0).to(density.dtype))
    diagnostics = {
        "unique_pairs": support.sum(),
        "far_unique_pairs": (valid & positive).sum(),
        "repeated_paths_removed": (torch.where(valid, paths, torch.zeros_like(paths)) - (valid & positive).to(paths.dtype)).sum(),
        "eligible_pairs": valid.expand_as(density).sum(),
        "pair_valid": valid.expand_as(density),
    }
    if near is not None:
        diagnostics["near_mandatory_pairs"] = (near > 0).sum()
        diagnostics["near_full_pairs"] = (near == 1).sum()
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


def structural_cost(accesses: dict[str, TypedSourceAccess], state: TypedHypergraphState, *, incidence_coefficient: float = 0.05, group_coefficient: float = 0.01) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced smooth work proxy; exact support counts stay separate.

    A single dense edge pays for dense pair occupancy, rather than earning
    sparsity merely by being called K=1. This is not an executor time model.
    """
    terms = []
    metrics = {}
    for mechanism, access in accesses.items():
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


__all__ = ["MECHANISMS", "SOURCE_TYPE", "TypedHypergraphState", "TypedSourceAccess", "build_source_catalogue", "masked_max", "masked_mean", "pad_geometry", "smooth_near_envelope", "source_moments", "structural_cost", "structural_pressure"]
