"""Experimental rank-eight contextual pair content for the R14 detail core."""
from __future__ import annotations

import torch
from torch import nn

from .detail import DetailAnchoredMemberRoutingCore


class PairContent(nn.Module):
    """Project positive eligible edge content only, then reduce before U."""

    def __init__(self, width: int, heads: int, coordinate_dim: int, seed: int):
        super().__init__()
        self.heads, self.rank = heads, 8
        self.last_positive_edges = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.P = nn.Linear(width, heads * self.rank, bias=False)
            self.Q = nn.Linear(width, heads * self.rank, bias=False)
            self.R = nn.Linear(2 * coordinate_dim + 1, heads * self.rank, bias=False)
            self.U = nn.Parameter(torch.zeros(heads, self.rank, width // heads))

    def forward(self, destination, donor, geometry, coefficients, gates, allowed, donor_measure):
        batch, receivers = destination.shape[:2]
        eligible = ((gates > 0) & allowed & (donor_measure > 0)[:, None, :]
                    & (coefficients > 0).any(dim=-1))
        edges = eligible.nonzero(as_tuple=False)
        self.last_positive_edges = edges.shape[0]
        result = destination.new_zeros(batch * receivers, self.heads, self.rank)
        if edges.numel():
            b, r, d = edges.unbind(-1)
            p = self.P(destination[b, r]).view(-1, self.heads, self.rank)
            q = self.Q(donor[b, d]).view(-1, self.heads, self.rank)
            g = self.R(geometry[b, r, d]).view(-1, self.heads, self.rank).tanh()
            result = result.index_add(0, b * receivers + r, (p + q) * g * coefficients[b, r, d, :, None])
        return torch.einsum('brhk,hkv->brhv', result.view(batch, receivers, self.heads, self.rank), self.U)


class PairContentDetailAnchoredCore(DetailAnchoredMemberRoutingCore):
    """Explicit experimental model with an independent versioned config."""

    schema = "interaction_reused_detail_anchored_pair_content_v1"

    def __init__(self, base_model_config, *, pair_seed=15808, **kwargs):
        if type(pair_seed) is not int or pair_seed < 0:
            raise ValueError("pair_seed must be a nonnegative integer.")
        super().__init__(base_model_config, **kwargs)
        self.pair_seed = pair_seed
        for i, cycle in enumerate(self.context_blocks):
            for j, name in enumerate(("mm", "em", "ee", "me")):
                route = getattr(cycle, name)
                route.pair_content = PairContent(self.context_dim, self.num_heads, self.coordinate_dim,
                                                 pair_seed + 4 * i + j)

    def prepare(self, *args, **kwargs):
        # Each route is executed once in each ordered cycle; resetting also
        # accounts for the route's zero-donor early return without stale work.
        for cycle in self.context_blocks:
            for name in ("mm", "em", "ee", "me"):
                getattr(cycle, name).pair_content.last_positive_edges = 0
        prepared = super().prepare(*args, **kwargs)
        total = 0
        for name in ("mm", "em", "ee", "me"):
            count = sum(getattr(cycle, name).pair_content.last_positive_edges for cycle in self.context_blocks)
            total += count
            prepared.work_counts[f"context_pair_content_{name}_positive_edges"] = count
            prepared.work_counts[f"context_pair_content_{name}_head_pairs"] = count * self.num_heads
            prepared.work_counts[f"context_pair_content_{name}_rank_scalars"] = count * self.num_heads * 8
        prepared.work_counts["context_pair_content_positive_edges"] = total
        prepared.work_counts["context_pair_content_head_pairs"] = total * self.num_heads
        prepared.work_counts["context_pair_content_rank_scalars"] = total * self.num_heads * 8
        return prepared

    def get_config(self):
        return {**super().get_config(), "pair_seed": self.pair_seed}

    @classmethod
    def validate_config(cls, config):
        values = dict(config)
        seed = values.pop("pair_seed", None)
        if type(seed) is not int or seed < 0:
            raise ValueError("Experimental pair_seed must be a nonnegative integer.")
        super().validate_config(values)

    @classmethod
    def from_config(cls, config):
        cls.validate_config(config)
        values = dict(config)
        seed = values.pop("pair_seed")
        # Parent parser dispatches through cls and retains all inherited settings.
        values["pair_seed"] = seed
        return cls(values["base_model_config"], locator_mode=values["locator_mode"],
                   input_dataset=values["input_dataset"], query_tile_size=values["query_tile_size"],
                   retained_mass=values["retained_mass"], score_calibration=values["score_calibration"],
                   access_measure=values.get("access_measure", "inventory_unit_total"),
                   ee_incidence_aggregation=values.get("ee_incidence_aggregation", "pre_gate"),
                   ee_gate_mode=values.get("ee_gate_mode", "legacy"),
                   ee_initial_tau=values.get("ee_initial_tau"), pair_seed=seed)

    @classmethod
    def from_legacy_config(cls, config, pair_seed=15808):
        DetailAnchoredMemberRoutingCore.validate_config(config)
        return cls.from_config({**config, "schema": cls.schema, "pair_seed": pair_seed})

    def load_legacy_state_dict(self, state_dict):
        expected = {name for name in self.state_dict() if ".pair_content." in name}
        incoming = set(state_dict)
        if incoming != set(self.state_dict()) - expected:
            raise RuntimeError("Legacy state inventory differs from inherited model parameters.")
        return self.load_state_dict(state_dict, strict=False)
