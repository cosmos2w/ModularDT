"""Input-conditioned overlapping source-control hyperedges and local access.

The organizer is shared by physical adapters. It consumes pre-interaction
encoded inputs only, stores a case/phase plan independently of query batches,
and exports source-resolved permissions to the native fine-reader kernels.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch import nn

from honf_forward_core.nn import MLP

from .case_group_budget import CaseGroupGate
from .routing_index.sparse_projection import masked_sparsemax, source_measure_sparsemax
from .topology_probe import FixedTopologyInvalid, fixed_active_projection, validate_catalogue
from .typed_hypergraph_state import (
    MECHANISMS,
    SOURCE_TYPE,
    TypedHypergraphState,
    TypedSourceAccess,
    build_source_catalogue,
    masked_mean,
    pad_geometry,
    smooth_near_envelope,
    source_moments,
    structural_pressure,
)
from .types import EncodedInterfaceCase


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor, temperature: float) -> torch.Tensor:
    """Strictly positive eligible shadow permissions; empty rows stay zero."""
    valid = torch.broadcast_to(valid, logits.shape)
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    selected = (logits / temperature).masked_fill(~valid, -torch.inf)
    occupied = valid.any(dim=-1, keepdim=True)
    selected = torch.where(occupied, selected, torch.zeros_like(selected))
    # Prevent dtype underflow from recreating exact hard zeros in the shadow.
    probability = torch.softmax(selected, dim=-1)
    probability = torch.where(valid, probability.clamp_min(torch.finfo(logits.dtype).tiny), torch.zeros_like(probability))
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


def _phase_index(phase: int | str) -> int:
    value = int(phase[1:] if isinstance(phase, str) and phase.startswith("P") else phase)
    if value not in (0, 1, 2):
        raise ValueError("Physical phase must be 0/1/2 or P0/P1/P2.")
    return value


def _source_density(logits: torch.Tensor, measures: torch.Tensor, *, soft: bool, temperature: float) -> torch.Tensor:
    """Physical-measure density preserves an identical quadrature atom split."""
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    normalized = measures / measures.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(measures.dtype).tiny)
    valid = (measures > 0)[:, None].expand_as(logits)
    if not soft:
        return source_measure_sparsemax(logits, normalized, valid).density.to(logits.dtype)
    probability = _masked_softmax(logits, valid, temperature)
    denominator = (probability * normalized[:, None]).sum(dim=-1, keepdim=True)
    return probability / torch.where(denominator > 0, denominator, torch.ones_like(denominator))


class OverlapControlHypergraph(nn.Module):
    """Eight provisional edges, typed controls, case/phase admission gates.

    Local lengths are physical adapter inputs when supplied. The explicit
    fallback is the configured module length and environment bounding-box
    cell spacing; neither fallback is fitted from targets or development
    errors. They are exported so local protection can be audited.
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        *,
        group_count: int = 8,
        control_dim: int = 16,
        local_access: bool = False,
        module_characteristic_length: float = 1.0,
        near_inner: float = 1.0,
        near_outer: float = 2.0,
        gate_temperature: float = 2.0 / 3.0,
        soft_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if spatial_dim not in (2, 3):
            raise ValueError("Overlap hypergraphs support spatial dimensions 2 and 3.")
        if min(hidden_dim, group_count, control_dim) <= 0 or soft_temperature <= 0:
            raise ValueError("Hypergraph widths, capacity and temperature must be positive.")
        if module_characteristic_length <= 0 or not 0 <= near_inner < near_outer:
            raise ValueError("Local physical length and envelope radii must be valid.")
        self.hidden_dim = int(hidden_dim)
        self.spatial_dim = int(spatial_dim)
        self.group_count = int(group_count)
        self.control_dim = int(control_dim)
        self.local_access = bool(local_access)
        self.module_characteristic_length = float(module_characteristic_length)
        self.near_inner, self.near_outer = float(near_inner), float(near_outer)
        self.soft_temperature = float(soft_temperature)
        self.register_buffer("training_epoch", torch.tensor(0, dtype=torch.long))
        self.source_projection = nn.ModuleDict({kind: nn.Linear(hidden_dim, control_dim) for kind in ("M", "E")})
        self.context_projection = nn.Linear(hidden_dim, control_dim)
        self.proposals = nn.Parameter(torch.randn(group_count, control_dim) / math.sqrt(control_dim))
        # Proposal codes are exchangeable feature queries, never source IDs.
        # Input-relative centroids/extents give them a physical reference.
        offsets = torch.tensor(list(itertools.product((-0.35, 0.35), repeat=3)))
        if group_count != 8:
            positions = torch.arange(group_count, dtype=torch.float32)
            offsets = torch.stack((torch.cos(positions * 2.39996), torch.sin(positions * 2.39996), torch.cos(positions * 1.61803)), dim=-1) * 0.35
        self.register_buffer("proposal_offsets", offsets)
        self.source_scores = nn.ModuleDict({tau: MLP(3 * control_dim + 9, control_dim, 1, num_layers=2) for tau in MECHANISMS})
        self.receiver_scores = nn.ModuleDict({tau: MLP(3 * control_dim + 9, control_dim, 1, num_layers=2) for tau in MECHANISMS})
        self.control_heads = nn.ModuleDict({tau: MLP(4 * control_dim + 12, control_dim, control_dim, num_layers=2) for tau in MECHANISMS})
        for head in self.control_heads.values():
            nn.init.zeros_(head.net[-1].weight)
            with torch.no_grad():
                head.net[-1].bias.copy_(torch.linspace(-0.1, 0.1, control_dim))
        self.admission_gate = CaseGroupGate(control_dim=control_dim, group_count=group_count, hidden_dim=control_dim, temperature=gate_temperature, initial_optional_open_probability=0.75, rescue_mode=True)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative.")
        self.training_epoch.fill_(int(epoch))

    def _propose_centres(self, encoded: EncodedInterfaceCase, catalogue: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        means, lows, highs, occupied = [], [], [], []
        for kind in ("M", "E"):
            coords, valid, measure = catalogue["source_coords"][kind], catalogue["source_valid"][kind], catalogue["source_measures"][kind]
            means.append(masked_mean(coords, valid, measure))
            presence = valid.any(dim=-1)
            occupied.append(presence)
            if coords.shape[1] == 0:
                low = high = coords.new_zeros((coords.shape[0], self.spatial_dim))
            else:
                low = torch.where(valid[..., None], coords, coords.new_full((), torch.inf)).amin(dim=1)
                high = torch.where(valid[..., None], coords, coords.new_full((), -torch.inf)).amax(dim=1)
                low = torch.where(presence[:, None], low, torch.zeros_like(low))
                high = torch.where(presence[:, None], high, torch.zeros_like(high))
            lows.append(low)
            highs.append(high)
        count = sum(p.to(encoded.global_token.dtype) for p in occupied).clamp_min(1)
        centre = sum(mean * present[:, None] for mean, present in zip(means, occupied, strict=True)) / count[:, None]
        low = torch.minimum(torch.where(occupied[0][:, None], lows[0], lows[1]), torch.where(occupied[1][:, None], lows[1], lows[0]))
        high = torch.maximum(torch.where(occupied[0][:, None], highs[0], highs[1]), torch.where(occupied[1][:, None], highs[1], highs[0]))
        extent = (high - low).clamp_min(self.module_characteristic_length)
        centres = centre[:, None] + self.proposal_offsets[None, :, :self.spatial_dim].to(centre) * extent[:, None]
        return centres, extent

    def _lengths(self, catalogue: dict[str, Any], extent: torch.Tensor) -> dict[str, torch.Tensor]:
        lengths = dict(catalogue["source_lengths"] or {})
        if "M" not in lengths:
            lengths["M"] = torch.full_like(catalogue["source_measures"]["M"], self.module_characteristic_length)
        if "E" not in lengths:
            valid = catalogue["source_valid"]["E"]
            cells = valid.sum(dim=-1).clamp_min(1).to(extent.dtype)
            spacing = (extent.prod(dim=-1) / cells).pow(1.0 / self.spatial_dim)
            lengths["E"] = spacing[:, None].expand_as(catalogue["source_measures"]["E"])
        return lengths

    def prepare(
        self,
        encoded: EncodedInterfaceCase,
        module_states: torch.Tensor,
        *,
        phase: int | str = 0,
        soft: bool = False,
        deterministic: bool | None = None,
        gate_noise: torch.Tensor | None = None,
        capture_topology: bool = False,
        fixed_topology: TypedHypergraphState | None = None,
    ) -> TypedHypergraphState:
        phase = _phase_index(phase)
        if encoded.module_centers.shape[-1] != self.spatial_dim:
            raise ValueError("Encoded geometry does not match organizer spatial dimension.")
        catalogue = build_source_catalogue(encoded, module_states)
        if fixed_topology is not None:
            if self.training or soft or fixed_topology.phase != phase:
                raise ValueError("Fixed topology is evaluation-only and requires a matching hard physical phase")
            validate_catalogue(catalogue, fixed_topology)
        centres, extent = self._propose_centres(encoded, catalogue)
        batch = module_states.shape[0]
        context = self.context_projection(encoded.global_token)
        phase_features = torch.nn.functional.one_hot(torch.full((batch,), phase, device=module_states.device), 3).to(module_states.dtype)
        states = {"M": self.source_projection["M"](module_states), "E": self.source_projection["E"](encoded.env_tokens)}
        memberships, controls, source_logits = {}, {}, {}
        summaries, masses = {}, {}
        for tau in MECHANISMS:
            kind = SOURCE_TYPE[tau]
            source = states[kind]
            relative = (catalogue["source_coords"][kind][:, None] - centres[:, :, None]) / extent[:, None, None]
            feature = torch.cat((source[:, None].expand(-1, self.group_count, -1, -1), self.proposals[None, :, None].expand(batch, -1, source.shape[1], -1), context[:, None, None].expand(-1, self.group_count, source.shape[1], -1), pad_geometry(relative), phase_features[:, None, None].expand(-1, self.group_count, source.shape[1], -1)), dim=-1)
            score = self.source_scores[tau](feature).squeeze(-1)
            score = score + torch.einsum("bsc,kc->bks", source, self.proposals) / math.sqrt(self.control_dim) - relative.square().sum(dim=-1)
            source_logits[tau] = score
            if fixed_topology is None:
                memberships[tau] = _source_density(score, catalogue["source_measures"][kind], soft=soft, temperature=self.soft_temperature)
            else:
                mu = catalogue["source_measures"][kind]
                mu = mu / mu.sum(-1, keepdim=True).clamp_min(torch.finfo(mu.dtype).tiny)
                memberships[tau] = fixed_active_projection(score, fixed_topology.strategy_data["source_logits"][tau],
                    fixed_topology.memberships[tau], mu)
            # All controls summarize both source types separately. A group's
            # opposite-type population uses the corresponding typed route.
        for tau in MECHANISMS:
            module_route = tau if SOURCE_TYPE[tau] == "M" else ("MM" if tau == "ME" else "QM")
            env_route = tau if SOURCE_TYPE[tau] == "E" else ("ME" if tau in ("MM", "EM") else "QE")
            typed_summary, typed_mass = {}, {}
            for kind, route in (("M", module_route), ("E", env_route)):
                mass = memberships[route] * catalogue["source_measures"][kind][:, None]
                total = mass.sum(dim=-1)
                safe_source = torch.where(catalogue["source_valid"][kind][..., None], states[kind], torch.zeros_like(states[kind]))
                summary = torch.einsum("bks,bsc->bkc", mass, safe_source) / torch.where(total > 0, total, torch.ones_like(total))[..., None]
                typed_summary[kind], typed_mass[kind] = summary, total
            receiver = typed_summary["M"] if tau in ("MM", "ME") else typed_summary["E"] if tau == "EM" else 0.5 * (typed_summary["M"] + typed_summary["E"])
            mass_features = torch.stack((typed_mass["M"].log1p(), typed_mass["E"].log1p(), (typed_mass["M"] > 0).to(module_states.dtype), (typed_mass["E"] > 0).to(module_states.dtype)), dim=-1)
            mechanism = torch.nn.functional.one_hot(torch.full((batch, self.group_count), MECHANISMS.index(tau), device=module_states.device), 5).to(module_states.dtype)
            control_input = torch.cat((typed_summary["M"], typed_summary["E"], receiver, context[:, None].expand(-1, self.group_count, -1), mass_features, phase_features[:, None].expand(-1, self.group_count, -1), mechanism), dim=-1)
            controls[tau] = self.control_heads[tau](control_input)
            summaries[tau], masses[tau] = typed_summary, typed_mass
        gate_input = torch.cat((summaries["MM"]["M"], summaries["ME"]["E"], context[:, None].expand(-1, self.group_count, -1), module_states.new_full((batch, self.group_count, 1), phase / 2), self.proposals[None].expand(batch, -1, -1)), dim=-1)
        gate_logits = self.admission_gate.network(gate_input).squeeze(-1)
        budget = self.admission_gate.build_budget(gate_logits, noise=gate_noise, deterministic=soft or (not self.training if deterministic is None else deterministic))
        admission = torch.sigmoid(gate_logits).clamp_min(torch.finfo(gate_logits.dtype).tiny) if soft else budget.z
        if fixed_topology is not None:
            reference = fixed_topology.admission.to(admission)
            rescue = fixed_topology.diagnostics["admission_rescue"][:, None]
            interior = (reference > 0) & (reference < 1) & ~rescue
            live_gate = torch.sigmoid(gate_logits) * (self.admission_gate.stretch_upper - self.admission_gate.stretch_lower) + self.admission_gate.stretch_lower
            if bool((interior & ((live_gate < 0) | (live_gate > 1))).any()):
                raise FixedTopologyInvalid("Admission continuation left its recorded clipping region")
            admission = torch.where(interior, live_gate, (reference > 0).to(admission.dtype))
        return TypedHypergraphState(
            memberships=memberships, controls=controls, centres=centres,
            admission=admission, phase=phase,
            **{**catalogue, "source_lengths": self._lengths(catalogue, extent)},
            diagnostics={"allocated_groups": admission.new_full((batch,), self.group_count), "admitted_groups": (admission > 0).sum(dim=-1), "admission_rescue": budget.fallback_used, "admission_probability": budget.positive_probability, "structural_pressure": admission.new_tensor(structural_pressure(int(self.training_epoch)))},
            strategy_data={"context": context, "phase_features": phase_features, "extent": extent, "source_logits": source_logits, "gate_logits": gate_logits, "soft": soft, "gate_noise": budget.noise, "local_access": self.local_access},
        )

    def access(
        self,
        state: TypedHypergraphState,
        receivers: torch.Tensor,
        mechanism: str,
        receiver_tokens: torch.Tensor | None = None,
        *,
        soft: bool = False,
        pair_valid: torch.Tensor | None = None,
        capture_topology: bool = False,
        fixed_receiver_access: dict | None = None,
    ) -> TypedSourceAccess:
        if mechanism not in SOURCE_TYPE:
            raise ValueError(f"Unknown typed mechanism {mechanism!r}.")
        if bool(state.strategy_data["soft"]) != bool(soft):
            raise ValueError("Access must use the hard/soft mode of its prepared phase state.")
        batch, receiver_count, _ = receivers.shape
        context = state.strategy_data["context"]
        relative = (receivers[:, :, None] - state.centres[:, None]) / state.strategy_data["extent"][:, None, None]
        kind = "E" if mechanism == "EM" else "M"
        receiver = torch.zeros((batch, receiver_count, self.control_dim), device=receivers.device, dtype=context.dtype) if receiver_tokens is None else self.source_projection[kind](receiver_tokens)
        features = torch.cat((receiver[:, :, None].expand(-1, -1, self.group_count, -1), self.proposals[None, None].expand(batch, receiver_count, -1, -1), context[:, None, None].expand(-1, receiver_count, self.group_count, -1), pad_geometry(relative), state.strategy_data["phase_features"][:, None, None].expand(-1, receiver_count, self.group_count, -1)), dim=-1)
        logits = self.receiver_scores[mechanism](features).squeeze(-1) - relative.square().sum(dim=-1)
        gates = state.admission[:, None].expand_as(logits)
        live = gates > 0
        safe_gates = torch.where(live, gates, torch.ones_like(gates))
        logits = logits + safe_gates.log()
        if fixed_receiver_access is None:
            edge_access = _masked_softmax(logits, live, self.soft_temperature) if soft else masked_sparsemax(logits, live)
        else:
            edge_access = fixed_active_projection(logits, fixed_receiver_access["receiver_logits"],
                fixed_receiver_access["edge_access"], torch.ones_like(state.admission))
        source_type = SOURCE_TYPE[mechanism]
        lengths = torch.where(state.source_valid[source_type], state.source_lengths[source_type], torch.ones_like(state.source_lengths[source_type]))
        near = smooth_near_envelope(receivers, state.source_coords[source_type], lengths, inner=self.near_inner, outer=self.near_outer) if self.local_access else None
        result = source_moments(edge_access, state.memberships[mechanism], state.controls[mechanism], state.source_measures[source_type], state.source_valid[source_type], pair_valid=pair_valid, near=near)
        if capture_topology:
            result.diagnostics["receiver_logits"] = logits
        return result

    def export(self, state: TypedHypergraphState) -> dict[str, Any]:
        exported = state.export()
        exported["receiver_access"] = lambda receivers, mechanism, receiver_tokens=None: self.access(state, receivers, mechanism, receiver_tokens, soft=bool(state.strategy_data["soft"]))
        exported["local_access"] = {"enabled": self.local_access, "inner": self.near_inner, "outer": self.near_outer, "fallback_module_length": self.module_characteristic_length, "fallback_environment_length": "input bounding-box cell spacing"}
        return exported


__all__ = ["OverlapControlHypergraph"]
