"""Query-relative values within the existing anchored member reader."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .anchored import AnchoredMemberRoutingCore


class DetailAnchoredMemberRoutingCore(AnchoredMemberRoutingCore):
    """Continuous E-row coverage and selected original-member value messages."""

    schema = "interaction_reused_detail_anchored_member_routing_v1"
    _ANCHOR_COUNT_KEYS = AnchoredMemberRoutingCore._ANCHOR_COUNT_KEYS + (
        "query_relative_value_pairs_M", "query_relative_value_pairs_E",
    )
    _additional_decode_count_keys = AnchoredMemberRoutingCore._additional_decode_count_keys + (
        "query_relative_value_pairs_M", "query_relative_value_pairs_E",
    )

    def __init__(self, base_model_config: Mapping[str, Any], *, locator_mode: str = "local_e_cover", **kwargs):
        if locator_mode != "local_e_cover":
            raise ValueError("R14 detail requires local_e_cover.")
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

    @staticmethod
    def _anchored_arm_name(locator_mode: str) -> str:
        return "R14-Detail-LocalECover"

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        super().validate_config(config)
        if config["locator_mode"] != "local_e_cover":
            raise ValueError("R14 detail requires local_e_cover.")

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
        message = F.silu(values + self.query_value[role](qstates[batch, query_index])
                         + self.relative_value[role](torch.cat(gamma, dim=-1))
                         + self.local_value[role](local))
        return message, {f"query_relative_value_pairs_{role}": slot.numel()}


__all__ = ["DetailAnchoredMemberRoutingCore"]
