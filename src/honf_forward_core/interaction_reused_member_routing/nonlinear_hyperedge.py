"""Optional nonlinear reads of the actual query-address M/E contexts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .detail import DetailAnchoredMemberRoutingCore


class _HyperedgeRead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight_M = nn.Parameter(torch.empty(32, width))
        self.weight_E = nn.Parameter(torch.empty(32, width))
        self.bias = nn.Parameter(torch.zeros(32))
        self.output = nn.Linear(32, width, bias=False)
        nn.init.xavier_uniform_(self.weight_M)
        nn.init.xavier_uniform_(self.weight_E)
        nn.init.zeros_(self.output.weight)

    def hidden(self, module, environment):
        return F.silu(F.linear(module, self.weight_M) + F.linear(environment, self.weight_E)
                      + self.bias) - F.silu(self.bias)


class HyperedgeNonlinearReadCore(DetailAnchoredMemberRoutingCore):
    """C applies a shared nonlinear read after blending; G applies it per row."""

    schema = "interaction_reused_nonlinear_hyperedge_member_routing_v1"
    NEW_STATE_KEYS = frozenset({
        "hyperedge_read.weight_M", "hyperedge_read.weight_E",
        "hyperedge_read.bias", "hyperedge_read.output.weight",
    })
    _extra_read_interventions = {"collapse_hyperedge_rows"}
    _HYPEREDGE_COUNT_KEYS = (
        "query_hyperedge_active_address_rows", "query_hyperedge_row_reduction_pairs_M",
        "query_hyperedge_row_reduction_pairs_E", "query_hyperedge_row_reduction_channels_M",
        "query_hyperedge_row_reduction_channels_E", "query_hyperedge_hidden_rows",
        "query_hyperedge_hidden_activations", "query_hyperedge_input_projection_macs",
        "query_hyperedge_output_projection_rows", "query_hyperedge_output_projection_macs",
    )
    _ANCHOR_COUNT_KEYS = DetailAnchoredMemberRoutingCore._ANCHOR_COUNT_KEYS + _HYPEREDGE_COUNT_KEYS
    _additional_decode_count_keys = (DetailAnchoredMemberRoutingCore._additional_decode_count_keys
                                     + _HYPEREDGE_COUNT_KEYS)
    # Bounds the temporary incidence-by-context tensor within each query tile.
    _row_reduction_chunk_size = 4096

    def __init__(self, base_model_config: Mapping[str, Any], *, hyperedge_mode: str, **kwargs):
        if hyperedge_mode not in ("C", "G"):
            raise ValueError("hyperedge_mode must be 'C' or 'G'.")
        self.hyperedge_mode = hyperedge_mode
        super().__init__(base_model_config, **kwargs)
        if self.reader_capability != "anchored":
            raise ValueError("Nonlinear hyperedge reads require actual anchored directory rows.")
        # C and G get identical tensors without advancing the training-data RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self._stable_seed("r16_4_hyperedge_read"))
            self.hyperedge_read = _HyperedgeRead(self.context_dim)

    @staticmethod
    def _anchored_arm_name(locator_mode: str) -> str:
        return "R16-4-Hyperedge-LocalECover"

    @property
    def _uses_hyperedge_rows(self):
        return self.hyperedge_mode == "G"

    def get_config(self):
        return {**super().get_config(), "hyperedge_mode": self.hyperedge_mode}

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        config = dict(config)
        if config.pop("hyperedge_mode", None) not in ("C", "G"):
            raise ValueError("A saved nonlinear hyperedge mode C or G is required.")
        if config.get("reader_capability", "anchored") != "anchored":
            raise ValueError("Nonlinear hyperedge reads require actual anchored directory rows.")
        super().validate_config(config)

    @classmethod
    def from_config(cls, config):
        cls.validate_config(config)
        return cls(config["base_model_config"], hyperedge_mode=config["hyperedge_mode"],
                   locator_mode=config["locator_mode"], input_dataset=config["input_dataset"],
                   query_tile_size=config["query_tile_size"], retained_mass=config["retained_mass"],
                   score_calibration=config["score_calibration"],
                   access_measure=config.get("access_measure", "inventory_unit_total"),
                   ee_incidence_aggregation=config.get("ee_incidence_aggregation", "pre_gate"),
                   ee_gate_mode=config.get("ee_gate_mode", "legacy"),
                   ee_initial_tau=config.get("ee_initial_tau"),
                   reader_capability=config.get("reader_capability", "anchored"))

    @classmethod
    def from_detail_config(cls, config, mode):
        """Create an explicit C/G configuration from a validated historical H0 config."""
        DetailAnchoredMemberRoutingCore.validate_config(config)
        if config.get("reader_capability", "anchored") != "anchored":
            raise ValueError("The source must use actual anchored directory rows.")
        result = {**config, "schema": cls.schema,
                  "architecture": cls._anchored_arm_name(config["locator_mode"]),
                  "hyperedge_mode": mode}
        cls.validate_config(result)
        return result

    def initialize_detail_weights(self, source_state):
        """Strictly copy the complete H0 inventory, retaining the four new tensors."""
        current = self.state_dict()
        source = dict(source_state)
        expected = set(current) - self.NEW_STATE_KEYS
        if set(source) != expected:
            raise ValueError("Detail source and nonlinear-read shared state inventories differ.")
        merged = {**current, **source}
        self.load_state_dict(merged, strict=True)

    def _hyperedge_role_rows(self, role, packed, union_index, positions, values, probabilities, strength):
        """Reduce each role's actual incidences, without a scene-sized incidence/value tensor."""
        rows = values.new_zeros((len(packed["reads"]), self.context_dim))
        member_positions = (packed["is_module"] if role == "M" else ~packed["is_module"]).nonzero(
            as_tuple=False).flatten()
        if positions.numel() and member_positions.numel():
            lookup = torch.searchsorted(positions, union_index[member_positions])
            safe = lookup.clamp_max(positions.numel() - 1)
            found = (lookup < positions.numel()) & (positions[safe] == union_index[member_positions])
            member_positions, lookup = member_positions[found], safe[found]
            rid = packed["read_id"]
            role_index = 0 if role == "M" else 1
            for start in range(0, member_positions.numel(), self._row_reduction_chunk_size):
                subset = member_positions[start:start + self._row_reduction_chunk_size]
                value_index = lookup[start:start + self._row_reduction_chunk_size]
                coefficient = probabilities[subset] * strength[rid[subset], role_index]
                # The row buffer starts at zero and is not consumed until all
                # chunks are accumulated. Mutating it avoids copying every
                # row-by-width buffer once for each incidence chunk.
                rows.index_add_(0, rid[subset], values[value_index] * coefficient[:, None])
        else:
            member_positions = member_positions[:0]
        pairs = member_positions.numel()
        return rows, {f"query_hyperedge_row_reduction_pairs_{role}": pairs,
                      f"query_hyperedge_row_reduction_channels_{role}": pairs * self.context_dim}

    def _hyperedge_correction(self, query, packed, contexts, row_contexts, intervention, *, return_trace):
        collapse = intervention.get("collapse_hyperedge_rows", False)
        if type(collapse) is not bool or (collapse and self.hyperedge_mode != "G"):
            raise ValueError("collapse_hyperedge_rows is a boolean intervention for G only.")
        batch_queries = query.batch_size * query.num_queries
        reads, weights = packed["reads"], packed["weights"]
        flat_query = reads[:, 0] * query.num_queries + reads[:, 1]
        if self.hyperedge_mode == "C":
            hidden = self.hyperedge_read.hidden(contexts["M"], contexts["E"])
            hidden_rows = batch_queries
        elif collapse:
            collapsed = {role: self._segment_sum(row_contexts[role] * weights[:, None], flat_query,
                                                  batch_queries).reshape(query.batch_size, query.num_queries, -1)
                         for role in ("M", "E")}
            hidden = self.hyperedge_read.hidden(collapsed["M"], collapsed["E"])
            hidden_rows = batch_queries
        else:
            row_hidden = self.hyperedge_read.hidden(row_contexts["M"], row_contexts["E"])
            hidden = self._segment_sum(row_hidden * weights[:, None], flat_query, batch_queries)
            hidden = hidden.reshape(query.batch_size, query.num_queries, 32)
            hidden_rows = len(reads)
        correction = self.hyperedge_read.output(hidden)
        counts = {
            "query_hyperedge_active_address_rows": len(reads),
            "query_hyperedge_hidden_rows": hidden_rows,
            "query_hyperedge_hidden_activations": hidden_rows * 32,
            "query_hyperedge_input_projection_macs": hidden_rows * 2 * self.context_dim * 32,
            "query_hyperedge_output_projection_rows": batch_queries,
            "query_hyperedge_output_projection_macs": batch_queries * self.context_dim * 32,
        }
        trace = None
        if return_trace:
            trace = {"hyperedge_read_mode": self.hyperedge_mode,
                     "hyperedge_rows_collapsed": collapse, "hyperedge_correction": correction}
            if row_contexts is not None:
                trace.update({f"hyperedge_row_context_{role}": row_contexts[role] for role in ("M", "E")})
        return correction, counts, trace


__all__ = ["HyperedgeNonlinearReadCore"]
