"""One collective action per typed route, with unchanged full fine access.

Source summaries condition controls only. Every physical value remains in the
fine reader; this organizer has no partition, sparse selector or frontier loss.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .topology_probe import FixedTopologyInvalid, validate_catalogue
from .typed_hypergraph_state import (
    MECHANISMS,
    SOURCE_TYPE,
    ProjectedSourceAccess,
    TypedHypergraphState,
    build_source_catalogue,
    masked_mean,
    pad_geometry,
    source_moments,
)


class GlobalSourceAccess(ProjectedSourceAccess):
    """Native eligibility with one case action, never an expanded pair action."""

    case_constant_action = True


class GlobalControlHypergraph(nn.Module):
    """Measure-weighted, task-trained controls without learned source omission."""

    faithful_controls = True
    measure_consistent = True

    def __init__(self, hidden_dim: int, *, spatial_dim: int = 2,
                 organizer_dim: int = 64, control_dim: int = 16):
        super().__init__()
        if spatial_dim not in (2, 3) or min(hidden_dim, organizer_dim, control_dim) < 1:
            raise ValueError("Global controls require positive dimensions and 2D or 3D geometry")
        self.hidden_dim = hidden_dim
        self.control_geometry_encoder = nn.Sequential(nn.Linear(21, organizer_dim), nn.GELU())
        self.control_heads = nn.ModuleDict({tau: nn.Sequential(
            nn.Linear(3 * hidden_dim + organizer_dim + 7, organizer_dim), nn.GELU(),
            nn.Linear(organizer_dim, control_dim)) for tau in MECHANISMS})
        for head in self.control_heads.values():
            nn.init.zeros_(head[-1].weight)
            with torch.no_grad():
                head[-1].bias.copy_(torch.linspace(-0.1, 0.1, control_dim))

    @staticmethod
    def _receiver_geometry(encoded, tau):
        if tau in ("MM", "ME"):
            coords, mass, roles = encoded.module_centers, encoded.module_present, None
            role = 0
        elif tau == "EM" or encoded.receiver_anchor_coords is None:
            coords, mass, roles = encoded.env_coords, encoded.env_weights, None
            role = 1 if tau == "EM" else 2
        else:
            coords, mass, roles = (encoded.receiver_anchor_coords,
                                  encoded.receiver_anchor_weights, encoded.receiver_anchor_roles)
            if coords.ndim == 2:
                coords = coords[None].expand(encoded.module_tokens.shape[0], -1, -1)
            if mass is None:
                mass = coords.new_ones(coords.shape[:2])
            elif mass.ndim == 1:
                mass = mass[None].expand(coords.shape[0], -1)
            role = 2
        valid = mass > (0.5 if tau in ("MM", "ME") else 0)
        mass = torch.where(valid, mass, torch.zeros_like(mass))
        centre = masked_mean(coords, valid, mass)
        if coords.shape[1]:
            low = torch.where(valid[..., None], coords, torch.full_like(coords, torch.inf)).amin(1)
            high = torch.where(valid[..., None], coords, torch.full_like(coords, -torch.inf)).amax(1)
            # Pull an equal-coordinate extremum back by physical mass, so
            # splitting one quadrature atom does not change its tied VJP.
            def measure_extremum(extreme):
                ties = valid[..., None] & (coords == extreme[:, None])
                selected_mass = torch.where(ties, mass[..., None], torch.zeros_like(coords))
                total = selected_mass.sum(1)
                mean = (torch.where(ties, coords, torch.zeros_like(coords)) * selected_mass).sum(1)
                mean = mean / torch.where(total > 0, total, torch.ones_like(total))
                finite = torch.where(valid.any(-1)[:, None], extreme, torch.zeros_like(extreme))
                return finite.detach() + (mean - mean.detach())
            low, high = measure_extremum(low), measure_extremum(high)
            extent = torch.where(valid.any(-1)[:, None], high - low, torch.zeros_like(high))
        else:
            extent = torch.zeros_like(centre)
        if roles is None:
            roles = torch.full(valid.shape, role, device=coords.device, dtype=torch.long)
        elif roles.ndim == 1:
            roles = roles[None].expand_as(valid)
        if bool(((roles[valid] < 0) | (roles[valid] >= 8)).any()):
            raise ValueError("receiver role codes must lie in [0,7]")
        roles = torch.where(valid, roles, torch.zeros_like(roles))
        role_summary = masked_mean(F.one_hot(roles, 8).to(coords.dtype), valid, mass)
        scale = encoded.coordinate_scale.reshape(-1, coords.shape[-1])
        descriptor = torch.cat((pad_geometry(centre / scale), pad_geometry(extent / scale),
                                centre.new_zeros((centre.shape[0], 1)), role_summary), -1)
        return centre, descriptor

    def recompute_controls(self, state, encoded, module_states):
        sources = {"M": module_states, "E": encoded.env_tokens}
        phase = F.one_hot(torch.tensor(state.phase, device=module_states.device), 3).to(module_states.dtype)
        result = {}
        for tau in MECHANISMS:
            summaries, masses = {}, {}
            for kind in ("M", "E"):
                mass = state.control_memberships[tau][kind] * state.source_measures[kind][:, None]
                summaries[kind] = masked_mean(sources[kind][:, None], mass > 0, mass)
                masses[kind] = mass.sum(-1)
            geometry = state.strategy_data["control_geometry"][tau]
            statistics = torch.stack((masses["M"].log1p(), masses["E"].log1p(),
                (masses["M"] > 0).to(geometry.dtype), (masses["E"] > 0).to(geometry.dtype)), -1)
            result[tau] = self.control_heads[tau](torch.cat((summaries["M"], summaries["E"],
                geometry, statistics, encoded.global_token[:, None],
                phase.expand(*geometry.shape[:-1], -1)), -1))
        return result

    def prepare(self, encoded, module_states, *, phase=0, soft=False,
                capture_topology=False, fixed_topology=None,
                topology_mode="fixed_active_set"):
        del capture_topology  # The sole group's full source support is fixed.
        if topology_mode != "fixed_active_set":
            raise ValueError("Global full-access controls do not support receiver-frontier continuation")
        if phase not in (0, 1, 2):
            raise ValueError("Global control phase must be P0, P1 or P2")
        catalogue = build_source_catalogue(encoded)
        if fixed_topology is not None:
            if self.training or soft:
                raise ValueError("Fixed global topology is evaluation-only and requires hard access")
            validate_catalogue(catalogue, fixed_topology)
            if fixed_topology.phase != phase:
                raise FixedTopologyInvalid("Global control phase differs from recorded topology")
        measures, valid = catalogue["source_measures"], catalogue["source_valid"]
        members = {kind: value[:, None].to(module_states.dtype) for kind, value in valid.items()}
        centres, geometry = {}, {}
        for tau in MECHANISMS:
            shared = "MM" if tau == "ME" else "QM" if tau == "QE" else None
            if shared is not None:
                centres[tau], geometry[tau] = centres[shared], geometry[shared]
            else:
                centres[tau], descriptor = self._receiver_geometry(encoded, tau)
                geometry[tau] = self.control_geometry_encoder(descriptor)[:, None]
        state = TypedHypergraphState(
            memberships={tau: members[SOURCE_TYPE[tau]] for tau in MECHANISMS}, controls={},
            centres=centres["QM"][:, None], admission=module_states.new_ones((module_states.shape[0], 1)),
            **catalogue, phase=phase,
            strategy_data={"soft": bool(soft), "control_geometry": geometry,
                           "typed_centres": {tau: value[:, None] for tau, value in centres.items()},
                           "typed_admission": {tau: module_states.new_ones((module_states.shape[0], 1))
                                               for tau in MECHANISMS}},
            control_memberships={tau: dict(members) for tau in MECHANISMS},
            control_presence={tau: {kind: (measures[kind].sum(-1) > 0)[:, None]
                                   for kind in ("M", "E")} for tau in MECHANISMS},
            dependency_provenance={
                "mode": "global_measure_weighted_v1",
                "planning": ("one fixed full-access group; no learned selector or partition",),
                "value_donors": dict(SOURCE_TYPE),
                "control_donors": {tau: ("M", "E") for tau in MECHANISMS},
                "control_content": ("measure-weighted module state", "measure-weighted environmental state",
                                    "receiver geometry/role", "explicit donor mass/presence",
                                    "prescribed global context", "physical phase"),
                "upstream_ancestry": "P0 input encodings" if phase == 0 else
                    "phase-current states retain earlier transport, predicted ports and Stage-A ancestry",
                "interpretation": "conditional computational dependencies; full access is not physical causality"})
        return replace(state, controls=self.recompute_controls(state, encoded, module_states))

    def numerical_access(self, state, receivers, mechanism, projected_action, *,
                         pair_valid=None, include_diagnostics=False):
        """Exact full-access one-group algebra, without density/moment bmm.

        Semantic exports still use :meth:`access`. The compact projected action
        includes the affine projection bias and is shared by every eligible
        pair; zero/invalid pairs remain masked by the native reader.
        """
        tau = str(mechanism).upper()
        if tau not in MECHANISMS:
            raise ValueError(f"unknown typed mechanism {mechanism!r}")
        kind = SOURCE_TYPE[tau]
        receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else None
        edge = receivers.new_ones((*receivers.shape[:2], 1))
        valid = state.source_valid[kind][:, None].expand(-1, receivers.shape[1], -1)
        if receiver_kind is not None:
            receiver_valid = state.source_valid[receiver_kind]
            if receivers.shape[:2] != receiver_valid.shape:
                raise ValueError("Global native receiver axis must match its physical catalogue")
            edge = edge * receiver_valid[..., None]
            valid = valid & receiver_valid[..., None]
            if tau == "MM":
                valid = valid & ~torch.eye(receivers.shape[1], device=receivers.device, dtype=torch.bool)[None]
        if pair_valid is not None:
            valid = valid & torch.broadcast_to(pair_valid.to(torch.bool), valid.shape)
        weight = valid.to(receivers.dtype)
        diagnostics = {"eligible_pairs": valid.sum(), "pair_valid": valid,
                       "receiver_measures": state.source_measures[receiver_kind]
                       if receiver_kind is not None else receivers.new_ones(receivers.shape[:2])}
        if include_diagnostics:
            diagnostics.update(unique_pairs=valid.sum(), far_unique_pairs=valid.sum(),
                               repeated_paths_removed=valid.new_zeros((), dtype=torch.long))
        return GlobalSourceAccess(weight, weight, projected_action[:, :, None], valid,
                                  edge, diagnostics=diagnostics)

    def access(self, state, receivers, mechanism, receiver_tokens=None, *, soft=False,
               pair_valid=None, prepared_action=None, include_diagnostics=True, detach_permissions=False,
               capture_topology=False, fixed_receiver_access=None):
        del receiver_tokens, soft, detach_permissions, capture_topology
        tau = str(mechanism).upper()
        if tau not in MECHANISMS:
            raise ValueError(f"unknown typed mechanism {mechanism!r}")
        receiver_kind = "M" if tau in ("MM", "ME") else "E" if tau == "EM" else None
        edge = receivers.new_ones((*receivers.shape[:2], 1))
        if receiver_kind is not None:
            receiver_valid = state.source_valid[receiver_kind]
            if receivers.shape[:2] != receiver_valid.shape:
                raise ValueError("Global native receiver axis must match its physical catalogue")
            edge = edge * receiver_valid[..., None]
            if pair_valid is None:
                pair_valid = receiver_valid[:, :, None]
                if tau == "MM":
                    pair_valid = pair_valid & ~torch.eye(receivers.shape[1], device=receivers.device, dtype=torch.bool)[None]
        kind = SOURCE_TYPE[tau]
        access = source_moments(edge, state.memberships[tau], state.controls[tau],
            state.source_measures[kind], state.source_valid[kind], pair_valid=pair_valid,
            prepared_action=prepared_action, include_diagnostics=include_diagnostics)
        if fixed_receiver_access is not None:
            if not torch.equal(access.edge_access, fixed_receiver_access["edge_access"].to(access.edge_access)):
                raise FixedTopologyInvalid("Global receiver eligibility differs from recorded topology")
            if "support" not in fixed_receiver_access:
                raise FixedTopologyInvalid("Global fixed access requires recorded fine source support")
            if not torch.equal(access.support, fixed_receiver_access["support"].to(access.support)):
                raise FixedTopologyInvalid("Global fine source support differs from recorded topology")
        access.diagnostics["receiver_measures"] = (state.source_measures[receiver_kind]
            if receiver_kind is not None else receivers.new_ones(receivers.shape[:2]))
        return access


__all__ = ["GlobalControlHypergraph", "GlobalSourceAccess"]
