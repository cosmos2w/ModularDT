"""Query-relative values within the existing anchored member reader."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .anchored import AnchoredMemberRoutingCore
from .primitives import build_joint_member_measure


class DetailAnchoredMemberRoutingCore(AnchoredMemberRoutingCore):
    """Continuous E-row coverage and selected original-member value messages."""

    schema = "interaction_reused_detail_anchored_member_routing_v1"
    _share_paired_selected_values = True
    _ANCHOR_COUNT_KEYS = AnchoredMemberRoutingCore._ANCHOR_COUNT_KEYS + (
        "query_relative_value_pairs_M", "query_relative_value_pairs_E",
        "query_value_projection_rows_M", "query_value_projection_rows_E",
    )
    _additional_decode_count_keys = AnchoredMemberRoutingCore._additional_decode_count_keys + (
        "query_relative_value_pairs_M", "query_relative_value_pairs_E",
        "query_value_projection_rows_M", "query_value_projection_rows_E",
    )

    def __init__(self, base_model_config: Mapping[str, Any], *, locator_mode: str = "local_e_cover",
                 reader_capability: str = "anchored", **kwargs):
        if locator_mode != "local_e_cover":
            raise ValueError("R14 detail requires local_e_cover.")
        if reader_capability not in ("anchored", "evidence_strength", "all_native"):
            raise ValueError("Unsupported detail reader capability.")
        self.reader_capability = reader_capability
        super().__init__(base_model_config, locator_mode=locator_mode, **kwargs)
        # Do not change any inherited initialization or the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._stable_seed("r14_query_relative_values"))
            self.query_value = nn.ModuleDict({role: nn.Linear(self.context_dim, self.context_dim)
                                              for role in ("M", "E")})
            self.relative_value = nn.ModuleDict({role: nn.Linear(7 * self.coordinate_dim, self.context_dim, bias=False)
                                                 for role in ("M", "E")})
            self.local_value = nn.ModuleDict({role: nn.Linear(self.coordinate_dim, self.context_dim, bias=False)
                                              for role in ("M", "E")})
        if reader_capability == "evidence_strength":
            self.evidence_strength = nn.ParameterDict({role: nn.Parameter(torch.zeros(3)) for role in ("M", "E")})

    @staticmethod
    def _anchored_arm_name(locator_mode: str) -> str:
        return "R14-Detail-LocalECover"

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        config = dict(config)
        capability = config.pop("reader_capability", "anchored")
        if capability not in ("anchored", "evidence_strength", "all_native"):
            raise ValueError("Unsupported detail reader capability.")
        super().validate_config(config)
        if config["locator_mode"] != "local_e_cover":
            raise ValueError("R14 detail requires local_e_cover.")

    def get_config(self):
        config = super().get_config()
        if self.reader_capability != "anchored":
            config["reader_capability"] = self.reader_capability
        return config

    @classmethod
    def from_config(cls, config):
        cls.validate_config(config)
        return cls(config["base_model_config"], locator_mode=config["locator_mode"],
                   input_dataset=config["input_dataset"], query_tile_size=config["query_tile_size"],
                   retained_mass=config["retained_mass"], score_calibration=config["score_calibration"],
                   access_measure=config.get("access_measure", "inventory_unit_total"),
                   ee_incidence_aggregation=config.get("ee_incidence_aggregation", "pre_gate"),
                   ee_gate_mode=config.get("ee_gate_mode", "legacy"),
                   ee_initial_tau=config.get("ee_initial_tau"),
                   reader_capability=config.get("reader_capability", "anchored"))

    def _packed_memberships(self, prepared, query, table, intervention):
        if self.reader_capability != "all_native":
            return super()._packed_memberships(prepared, query, table, intervention)
        if set(intervention) - {"read_mode"}:
            raise ValueError("All-native reads do not accept directory interventions.")
        scene = prepared.scene
        measure = build_joint_member_measure(scene.module_mask, scene.environment_mask,
                                             scene.environment_measure, dtype=query.coordinates.dtype)
        b, q = self._query_mask(query).nonzero(as_tuple=True)
        reads = torch.stack((b, q, torch.zeros_like(b)), -1)
        eligible = measure[b] > 0
        rid, member = eligible.nonzero(as_tuple=True)
        prior = measure[b[rid], member]
        is_module = member < scene.num_modules
        ids = torch.cat((scene.module_ids, scene.environment_ids), -1)
        total = measure.shape[-1]
        return {"reads": reads, "anchors": query.anchor_indices[b, q, 0],
                "weights": prior.new_ones(b.shape), "read_id": rid,
                "entries": member, "members": member, "prior": prior,
                "original": ids[b[rid], member], "is_module": is_module,
                "pair_key": ((b[rid] * query.num_queries + q[rid]) * total + member)}

    def decode_paired(self, organization, query, *, intervention=None, return_trace=False, trace_request=None):
        if self.reader_capability != "all_native":
            return super().decode_paired(organization, query, intervention=intervention,
                                         return_trace=return_trace, trace_request=trace_request)
        output = self.decode(organization, query, intervention=intervention,
                             return_trace=return_trace, trace_request=trace_request)
        return output, output, {}

    def _selected_member_values(self, prepared, query, qstates, role, batch, query_index, slot):
        values, _ = super()._selected_member_values(prepared, query, qstates, role, batch, query_index, slot)
        rows = torch.stack((batch, query_index, slot), dim=-1)
        geometry = self._pair_geometry(prepared.scene, query, role, rows)
        relative = geometry[:, :self.coordinate_dim]
        gamma = [relative]
        for frequency in (1.0, 2.0, 4.0):
            phase = (2.0 * math.pi * frequency) * relative
            gamma.extend((phase.sin(), phase.cos()))
        local = geometry[:, -self.coordinate_dim:]
        query_value = self.query_value[role](qstates)
        message = F.silu(values + query_value[batch, query_index]
                         + self.relative_value[role](torch.cat(gamma, dim=-1))
                         + self.local_value[role](local))
        return message, {f"query_relative_value_pairs_{role}": slot.numel(),
                         f"query_value_projection_rows_{role}": query.batch_size * query.num_queries}


__all__ = ["DetailAnchoredMemberRoutingCore"]
