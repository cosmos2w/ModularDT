"""Bounded, input-identified evidence for contextual frontier organization.

These helpers inspect the maintained native adapter/core objects. Their
access summaries describe executed retrieval and do not establish physical
causality or end-to-end speed.
"""
from __future__ import annotations

import io
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "contextual_frontier_organization_evidence_v1"
_FIELD_CHANNELS = (("u", 0), ("v", 1), ("p", 2), ("omega", 3), ("temperature", 4))


def _regional_channel_map(prediction_channels: int, target_channels: int,
                          channel_map: Sequence[tuple[str, int, int]] | None = None):
    if channel_map is not None:
        mapping = tuple((str(name), int(pred), int(target)) for name, pred, target in channel_map)
    elif prediction_channels == 5 and target_channels == 5:
        mapping = tuple((name, index, index) for name, index in _FIELD_CHANNELS)
    else:
        raise ValueError("Native Thermal regional evaluation requires [u,v,p,omega,T] predictions and targets.")
    if not mapping or any(pred < 0 or pred >= prediction_channels or target < 0 or target >= target_channels
                          for _name, pred, target in mapping):
        raise ValueError("Regional field channel map does not match prediction and target widths.")
    return mapping


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _plain(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _id_strings(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([json.dumps(_plain(value), sort_keys=True, ensure_ascii=False) for value in values], dtype=str)


def _ragged_indices(rows: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(rows) + 1,), dtype=np.int64)
    if rows:
        offsets[1:] = np.cumsum([len(row) for row in rows], dtype=np.int64)
        values = np.asarray([value for row in rows for value in row], dtype=np.int64)
    else:
        values = np.empty((0,), dtype=np.int64)
    return offsets, values


def _depths(forest: Any) -> np.ndarray:
    output = np.zeros((len(forest.nodes),), dtype=np.int64)
    for node in forest.nodes:
        depth, parent = 0, node.parent
        while parent is not None:
            depth += 1
            parent = forest.nodes[parent].parent
        output[node.index] = depth
    return output


def _json_id_rows(rows: Sequence[Sequence[Any]]) -> np.ndarray:
    return np.asarray([
        json.dumps([_plain(value) for value in row], sort_keys=True, ensure_ascii=False)
        for row in rows
    ], dtype=str)


def _bank_arrays(
    *, bank_name: str, bank: Any, core: Any, query_count: int,
    trace: Mapping[str, Any], arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    forest = bank.forest
    frontier = bank.frontier
    prefix = bank_name
    n = int(bank.present_count)
    k = int(bank.live_count)
    expected_k = None if frontier is None else float(frontier.expected_K.detach().cpu())
    if forest is None:
        arrays[f"{prefix}_record_ids"] = np.empty((0,), dtype=str)
        arrays[f"{prefix}_record_scene_rows"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_candidate_node_children"] = np.empty((0, 2), dtype=np.int64)
        arrays[f"{prefix}_candidate_node_parent"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_candidate_node_is_leaf"] = np.empty((0,), dtype=np.bool_)
        arrays[f"{prefix}_candidate_node_depth"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_candidate_ancestry_offsets"] = np.zeros((1,), dtype=np.int64)
        arrays[f"{prefix}_candidate_ancestry_local_rows"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_candidate_ancestry_ids"] = np.empty((0,), dtype=str)
        arrays[f"{prefix}_selected_node_indices"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_selected_ancestry_offsets"] = np.zeros((1,), dtype=np.int64)
        arrays[f"{prefix}_selected_ancestry_local_rows"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_selected_ancestry_ids"] = np.empty((0,), dtype=str)
        arrays[f"{prefix}_candidate_stop_logit"] = np.empty((0,), dtype=np.float32)
        arrays[f"{prefix}_candidate_stop_probability"] = np.empty((0,), dtype=np.float32)
        arrays[f"{prefix}_candidate_decision"] = np.empty((0,), dtype=np.int8)
        arrays[f"{prefix}_risk_node_indices"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_candidate_risk_logits"] = np.empty((0, 1), dtype=np.float32)
        arrays[f"{prefix}_candidate_edge_rows"] = np.empty((0, 2), dtype=np.int64)
        for suffix in ("geometry_distance", "context_distance", "context_cosine_similarity", "cost"):
            arrays[f"{prefix}_candidate_edge_{suffix}"] = np.empty((0,), dtype=np.float64)
        arrays[f"{prefix}_candidate_edge_flags"] = np.empty((0, 3), dtype=np.bool_)
        arrays[f"{prefix}_group_access"] = np.zeros((query_count, 0), dtype=np.float32)
        arrays[f"{prefix}_c_row_mass"] = np.empty((0,), dtype=np.float32)
        arrays[f"{prefix}_c_support_count"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_c_group_index"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_c_record_local_index"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_c_value"] = np.empty((0,), dtype=np.float32)
        arrays[f"{prefix}_donor_indices"] = np.empty((0,), dtype=str)
        query_edges = trace[f"{bank_name}_query_gates"][0]
        source_reads = trace[f"{bank_name}_source_read_attention"][0]
        read_queries = source_reads.get("edge_query_rows")
        read_sources = source_reads.get("edge_source_rows")
        has_source = torch.zeros((query_count,), dtype=torch.bool, device=query_edges["query_indices"].device)
        if read_queries is not None and read_queries.numel():
            has_source[read_queries.long()] = True
        arrays[f"{prefix}_query_has_source"] = _numpy(has_source).astype(np.bool_, copy=False)
        arrays[f"{prefix}_group_record_route_query_index"] = _numpy(query_edges["query_indices"]).astype(np.int64, copy=False)
        arrays[f"{prefix}_group_record_route_local_index"] = _numpy(query_edges["record_local_indices"]).astype(np.int64, copy=False)
        arrays[f"{prefix}_group_record_route_gate"] = _numpy(query_edges["gate_values"]).astype(np.float32, copy=False)
        arrays[f"{prefix}_query_source_index"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_query_source_local_index"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_query_source_record_id"] = np.empty((0,), dtype=str)
        arrays[f"{prefix}_selected_record_ids"] = np.empty((0,), dtype=str)
        arrays[f"{prefix}_route_selected_scene_rows"] = _numpy(
            trace[f"{bank_name}_selected_scene_rows"][0]
        ).astype(np.int64, copy=False)
        arrays[f"{prefix}_selected_scene_rows"] = np.empty((0,), dtype=np.int64)
        arrays[f"{prefix}_read_query_index"] = (
            np.empty((0,), dtype=np.int64) if source_reads.get("edge_query_rows") is None
            else _numpy(source_reads["edge_query_rows"]).astype(np.int64, copy=False)
        )
        arrays[f"{prefix}_read_source_local_index"] = (
            np.empty((0,), dtype=np.int64) if source_reads.get("edge_source_rows") is None
            else _numpy(source_reads["edge_source_rows"]).astype(np.int64, copy=False)
        )
        arrays[f"{prefix}_read_head_weights"] = (
            np.empty((0, 0), dtype=np.float32) if source_reads.get("weights") is None
            else _numpy(source_reads["weights"]).astype(np.float32, copy=False)
        )
        source_count = 0 if read_sources is None else int(read_sources.numel())
        return {
            "N": n, "K": 0, "allocated_K": k,
            "present_records": n, "actual_K": 0, "allocated_groups": k,
            "expected_K": expected_k, "expected_K_fraction": None,
            "candidate_nodes": 0, "internal_candidates": 0,
            "candidate_edges": 0, "frontier_node_indices": [], "frontier_ancestry_ids": [],
            "frontier_ancestry_sizes": [], "frontier_depths": [],
            "stop_probability_temperature": None, "frontier_log_probability": None,
            "frontier_mode": None,
            "C_allocated_cells": 0, "C_positive_entries": 0, "C_positive_row_mass": [],
            "actual_trace_source_edges": source_count, "empty_source_read": source_count == 0,
            "actual_trace_value_read_rows": source_count,
            "actual_trace_group_record_route_edges": int(query_edges["query_indices"].numel()),
            "actual_trace_group_access_rows": 0, "empty_group_access": True,
            "unique_unordered_candidate_pairs": 0,
            "candidate_matrix_slots": 0, "geometry_distance_matrix_cells_allocated": 0,
            "context_similarity_matrix_cells_allocated": 0,
        }

    ids = tuple(forest.original_ids)
    arrays[f"{prefix}_record_ids"] = _id_strings(ids)
    arrays[f"{prefix}_record_scene_rows"] = _numpy(bank.scene_rows).astype(np.int64, copy=False)

    node_count = len(forest.nodes)
    children = np.full((node_count, 2), -1, dtype=np.int64)
    parent = np.full((node_count,), -1, dtype=np.int64)
    leaf = np.zeros((node_count,), dtype=np.bool_)
    candidate_ancestry = []
    candidate_ancestry_ids = []
    for node in forest.nodes:
        if node.children is None:
            leaf[node.index] = True
        else:
            children[node.index] = np.asarray(node.children, dtype=np.int64)
        if node.parent is not None:
            parent[node.index] = int(node.parent)
        candidate_ancestry.append(tuple(int(value) for value in node.ancestry_local_indices))
        candidate_ancestry_ids.append(tuple(node.ancestry_ids))
    candidate_offsets, candidate_rows = _ragged_indices(candidate_ancestry)
    selected_ancestry = () if frontier is None else frontier.ancestry_local_indices
    selected_ids = () if frontier is None else frontier.ancestry_ids
    selected_offsets, selected_rows = _ragged_indices(selected_ancestry)
    selected_node_indices = np.asarray(() if frontier is None else frontier.node_indices, dtype=np.int64)
    depths = _depths(forest)
    arrays.update({
        f"{prefix}_candidate_node_children": children,
        f"{prefix}_candidate_node_parent": parent,
        f"{prefix}_candidate_node_is_leaf": leaf,
        f"{prefix}_candidate_node_depth": depths,
        f"{prefix}_candidate_ancestry_offsets": candidate_offsets,
        f"{prefix}_candidate_ancestry_local_rows": candidate_rows,
        f"{prefix}_candidate_ancestry_ids": _json_id_rows(candidate_ancestry_ids),
        f"{prefix}_selected_node_indices": selected_node_indices,
        f"{prefix}_selected_ancestry_offsets": selected_offsets,
        f"{prefix}_selected_ancestry_local_rows": selected_rows,
        f"{prefix}_selected_ancestry_ids": _json_id_rows(selected_ids),
    })

    internal = tuple(forest.internal_indices)
    logits = np.full((node_count,), np.nan, dtype=np.float32)
    probabilities = np.full((node_count,), np.nan, dtype=np.float32)
    decisions = np.full((node_count,), -1, dtype=np.int8)
    risk_logits: list[np.ndarray] = []
    risk_nodes: list[int] = []
    if frontier is not None:
        policy = core.frontier_policies[bank_name]
        epoch = int(core.organization_epoch)
        temperature = float(policy.temperature_for_epoch(epoch))
        for index in internal:
            logit = frontier.stop_logits[index].detach().to(device="cpu", dtype=torch.float32)
            logits[index] = float(logit)
            probabilities[index] = float(torch.sigmoid(logit / temperature))
            if index in frontier.decisions:
                decisions[index] = int(bool(frontier.decisions[index]))
            risk = frontier.risk_logits[index].detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy()
            risk_nodes.append(int(index))
            risk_logits.append(risk)
    arrays[f"{prefix}_candidate_stop_logit"] = logits
    arrays[f"{prefix}_candidate_stop_probability"] = probabilities
    arrays[f"{prefix}_candidate_decision"] = decisions
    arrays[f"{prefix}_risk_node_indices"] = np.asarray(risk_nodes, dtype=np.int64)
    arrays[f"{prefix}_candidate_risk_logits"] = (
        np.stack(risk_logits) if risk_logits else np.empty((0, 1), dtype=np.float32)
    )

    edges = tuple(forest.candidate_edges)
    edge_pairs = np.asarray([(edge.left, edge.right) for edge in edges], dtype=np.int64).reshape(-1, 2)
    arrays[f"{prefix}_candidate_edge_rows"] = edge_pairs
    arrays[f"{prefix}_candidate_edge_geometry_distance"] = np.asarray(
        [edge.geometry_distance for edge in edges], dtype=np.float64
    )
    arrays[f"{prefix}_candidate_edge_context_distance"] = np.asarray(
        [edge.context_distance for edge in edges], dtype=np.float64
    )
    arrays[f"{prefix}_candidate_edge_context_cosine_similarity"] = 1.0 - arrays[
        f"{prefix}_candidate_edge_context_distance"
    ]
    arrays[f"{prefix}_candidate_edge_cost"] = np.asarray([edge.cost for edge in edges], dtype=np.float64)
    arrays[f"{prefix}_candidate_edge_flags"] = np.asarray([
        (edge.spatial_knn, edge.context_knn, edge.spatial_spanning) for edge in edges
    ], dtype=np.bool_).reshape(-1, 3)

    membership = bank.membership.detach()
    c_positive = membership > 0
    c_group, c_record = torch.nonzero(c_positive, as_tuple=True)
    c_values = membership[c_group, c_record]
    arrays[f"{prefix}_c_row_mass"] = _numpy(membership.sum(dim=-1)).astype(np.float32, copy=False)
    arrays[f"{prefix}_c_support_count"] = _numpy(c_positive.sum(dim=-1)).astype(np.int64, copy=False)
    arrays[f"{prefix}_c_group_index"] = _numpy(c_group).astype(np.int64, copy=False)
    arrays[f"{prefix}_c_record_local_index"] = _numpy(c_record).astype(np.int64, copy=False)
    arrays[f"{prefix}_c_value"] = _numpy(c_values).astype(np.float32, copy=False)
    arrays[f"{prefix}_donor_indices"] = _json_id_rows(bank.donor_local_indices)

    group_access = trace[f"{bank_name}_group_access"][0]
    query_edges = trace[f"{bank_name}_query_gates"][0]
    source_count = int(query_edges["query_indices"].numel())
    source_reads = trace[f"{bank_name}_source_read_attention"][0]
    read_query_rows = source_reads.get("edge_query_rows")
    read_source_rows = source_reads.get("edge_source_rows")
    value_read_count = 0 if read_source_rows is None else int(read_source_rows.numel())
    query_has_source = torch.zeros((query_count,), dtype=torch.bool, device=group_access.device)
    if read_query_rows is not None and read_query_rows.numel():
        query_has_source[read_query_rows.long()] = True
    arrays[f"{prefix}_group_access"] = _numpy(group_access).astype(np.float32, copy=False)
    arrays[f"{prefix}_query_has_source"] = _numpy(query_has_source).astype(np.bool_, copy=False)
    gate_local = _numpy(query_edges["record_local_indices"]).astype(np.int64, copy=False)
    read_local = np.empty((0,), dtype=np.int64) if read_source_rows is None else _numpy(read_source_rows).astype(np.int64, copy=False)
    read_query = np.empty((0,), dtype=np.int64) if read_query_rows is None else _numpy(read_query_rows).astype(np.int64, copy=False)
    arrays[f"{prefix}_group_record_route_query_index"] = _numpy(query_edges["query_indices"]).astype(np.int64, copy=False)
    arrays[f"{prefix}_group_record_route_local_index"] = gate_local
    arrays[f"{prefix}_group_record_route_gate"] = _numpy(query_edges["gate_values"]).astype(np.float32, copy=False)
    arrays[f"{prefix}_query_source_index"] = read_query
    arrays[f"{prefix}_query_source_local_index"] = read_local
    arrays[f"{prefix}_query_source_record_id"] = _id_strings([ids[int(index)] for index in read_local])
    arrays[f"{prefix}_selected_record_ids"] = _id_strings([ids[int(index)] for index in np.unique(read_local)])
    selected_scene_rows = trace[f"{bank_name}_selected_scene_rows"][0]
    arrays[f"{prefix}_route_selected_scene_rows"] = _numpy(selected_scene_rows).astype(np.int64, copy=False)
    arrays[f"{prefix}_selected_scene_rows"] = _numpy(
        bank.scene_rows.index_select(0, torch.as_tensor(read_local, device=bank.scene_rows.device, dtype=torch.long))
    ).astype(np.int64, copy=False)

    for source_name, trace_name, dtype in (
        ("read_query_index", "edge_query_rows", np.int64),
        ("read_source_local_index", "edge_source_rows", np.int64),
        ("read_head_weights", "weights", np.float32),
    ):
        value = source_reads.get(trace_name)
        if value is None:
            value = np.empty((0,), dtype=dtype)
        arrays[f"{prefix}_{source_name}"] = _numpy(value).astype(dtype, copy=False)

    return {
        "present_records": n,
        "N": n,
        "actual_K": 0 if frontier is None else int(frontier.K),
        "K": 0 if frontier is None else int(frontier.K),
        "allocated_groups": int(bank.group_states.shape[0]),
        "allocated_K": int(bank.group_states.shape[0]),
        "expected_K": expected_k,
        "expected_K_fraction": None if expected_k is None or n == 0 else expected_k / n,
        "candidate_nodes": node_count,
        "internal_candidates": len(internal),
        "candidate_edges": len(edges),
        "frontier_node_indices": [] if frontier is None else list(map(int, frontier.node_indices)),
        "frontier_ancestry_ids": [[_plain(value) for value in row] for row in selected_ids],
        "frontier_ancestry_sizes": [len(row) for row in selected_ancestry],
        "frontier_depths": [] if frontier is None else [int(depths[index]) for index in frontier.node_indices],
        "stop_probability_temperature": None if frontier is None else temperature,
        "frontier_log_probability": None if frontier is None else float(frontier.log_prob.detach().cpu()),
        "frontier_mode": None if frontier is None else str(frontier.mode),
        "C_allocated_cells": int(membership.numel()),
        "C_positive_entries": int(c_positive.sum().item()),
        "C_positive_row_mass": [float(value) for value in arrays[f"{prefix}_c_row_mass"]],
        "actual_trace_source_edges": value_read_count,
        "actual_trace_value_read_rows": value_read_count,
        "actual_trace_group_record_route_edges": source_count,
        "actual_trace_group_access_rows": int((group_access > 0).sum().item()),
        "empty_source_read": value_read_count == 0,
        "empty_group_access": not bool((group_access > 0).any().item()),
        "unique_unordered_candidate_pairs": n * max(0, n - 1) // 2,
        "candidate_matrix_slots": 2 * n * n,
        "geometry_distance_matrix_cells_allocated": n * n,
        "context_similarity_matrix_cells_allocated": n * n,
        "context_similarity_ordered_dot_products": n * n,
    }

def inspect_prepared(model: Any, prepared_native: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return a JSON-safe receipt and compact arrays for one real native prepare.

    The core's real ``decode(..., return_trace=True)`` is used to capture the
    executed group and source edges. No synthetic support is substituted.
    """
    core = getattr(model, "core", model)
    organization = getattr(prepared_native, "organization", None)
    query = getattr(prepared_native, "query_batch", None)
    if organization is None or query is None or not callable(getattr(core, "decode", None)):
        raise TypeError("inspect_prepared requires a maintained native preparation and core decode API.")
    if organization.scene.batch_size != 1:
        raise ValueError("inspect_prepared currently requires a single-case native preparation.")
    with torch.no_grad():
        output = core.decode(organization, query, return_trace=True)
    if not isinstance(output.trace, Mapping):
        raise RuntimeError("The maintained core did not return its requested sparse execution trace.")
    trace = output.trace
    query_count = int(query.coordinates.shape[1])
    arrays: dict[str, np.ndarray] = {}
    banks = {}
    for index, (module_bank, environment_bank) in enumerate(zip(
        organization.module_banks, organization.environment_banks
    )):
        if index != 0:
            raise ValueError("inspect_prepared currently requires one prepared native scene at a time.")
        banks["module"] = _bank_arrays(
            bank_name="module", bank=module_bank, core=core, query_count=query_count,
            trace=trace, arrays=arrays,
        )
        banks["environment"] = _bank_arrays(
            bank_name="environment", bank=environment_bank, core=core, query_count=query_count,
            trace=trace, arrays=arrays,
        )
    scene = organization.scene
    decode_counters = {str(key): int(value) for key, value in output.work_counts.items()}
    module_has_source = arrays["module_query_has_source"]
    environment_has_source = arrays["environment_query_has_source"]
    e_only = (~module_has_source) & environment_has_source
    local_pair_input_rows = int(query.local_weights.numel()) if query.local_weights is not None else 0
    local_pair_count = decode_counters.get("module_local_packed_pair_rows", 0)
    local_query_counts = np.zeros((query_count,), dtype=np.int64)
    local_exceptions: list[int] = []
    if query.local_batch_indices is not None:
        keep = query.local_batch_indices == 0
        q_index = _numpy(query.local_query_indices[keep]).astype(np.int64, copy=False)
        local_module_index = _numpy(query.local_module_indices[keep]).astype(np.int64, copy=False)
        local_weights = _numpy(query.local_weights[keep]).astype(np.float32, copy=False)
        if q_index.size:
            np.add.at(local_query_counts, q_index, 1)
        local_exceptions = np.flatnonzero((local_query_counts > 0) & ~module_has_source).astype(int).tolist()
        local_ids = _id_strings([_numpy(scene.module_ids[0, index]).item() for index in local_module_index])
    else:
        q_index = np.empty((0,), dtype=np.int64)
        local_module_index = np.empty((0,), dtype=np.int64)
        local_weights = np.empty((0,), dtype=np.float32)
        local_ids = np.empty((0,), dtype=str)
    arrays["query_local_pair_count"] = local_query_counts
    arrays["query_e_only"] = e_only.astype(np.bool_, copy=False)
    arrays["local_pair_query_index"] = q_index
    arrays["local_pair_module_local_index"] = local_module_index
    arrays["local_pair_module_id"] = local_ids
    arrays["local_pair_weight"] = local_weights

    padded_m = int(scene.num_modules)
    padded_e = int(scene.num_environment_tokens)
    present_m = int(scene.module_mask[0].sum().item())
    present_e = int(scene.environment_mask[0].sum().item())
    context_depth = int(core.context_depth)
    contextual_pair_slots_per_cycle = (padded_m * padded_m + padded_m * padded_e
                                        + padded_e * padded_e + padded_e * padded_m)
    prepare_counters = {str(key): int(value) for key, value in organization.work_counts.items()}
    candidate_cost = {}
    for bank_name, bank in (("module", organization.module_banks[0]),
                            ("environment", organization.environment_banks[0])):
        count = int(bank.present_count)
        unique_pairs = count * max(0, count - 1) // 2
        candidate_cost[bank_name] = {
            "unique_unordered_geometry_distance_evaluations": unique_pairs,
            "unique_unordered_context_similarity_values": unique_pairs,
            "context_cosine_ordered_dot_products": count * count,
            "context_cosine_scalar_multiply_terms": count * count * int(bank.context_states.shape[-1]),
            "geometry_distance_matrix_cells_allocated": count * count,
            "context_similarity_matrix_cells_allocated": count * count,
            "reported_model_candidate_distance_pairs": int(
                organization.work_counts.get(f"{bank_name}_candidate_distance_pairs", 0)
            ),
        }
    cost = {
        "context_encoder_input_rows": {"module": padded_m, "environment": padded_e,
                                        "total": padded_m + padded_e},
        "present_context_records": {"module": present_m, "environment": present_e,
                                     "total": present_m + present_e},
        "contextual_attention_pair_slots_per_cycle": {
            "mm": padded_m * padded_m, "em": padded_e * padded_m,
            "ee": padded_e * padded_e, "me": padded_m * padded_e,
            "total": contextual_pair_slots_per_cycle,
        },
        "contextual_attention_head_pair_rows_per_cycle": contextual_pair_slots_per_cycle * int(core.num_heads),
        "contextual_geometry_bias_elements_per_cycle": contextual_pair_slots_per_cycle * int(core.num_heads),
        "context_depth": context_depth,
        "contextual_attention_pair_slots_all_cycles": contextual_pair_slots_per_cycle * context_depth,
        "contextual_attention_head_pair_rows_all_cycles": (
            contextual_pair_slots_per_cycle * int(core.num_heads) * context_depth
        ),
        "contextual_geometry_bias_elements_all_cycles": (
            contextual_pair_slots_per_cycle * int(core.num_heads) * context_depth
        ),
        "candidate_distance_accounting_by_bank": candidate_cost,
        "candidate_distance_counter_semantics": (
            "Contextual encoder counts use padded M/E widths. Per cycle, MM/EM/EE/ME score positions are Mpad^2 + "
            "2*Mpad*Epad + Epad^2; head rows and geometry-bias elements multiply this by head count and depth. "
            "Forest geometry computes choose(N,2) off-diagonal distances but allocates an N-by-N symmetric matrix; "
            "context cosine executes an ordered N-by-N Gram product (including diagonal), while its symmetric result "
            "also occupies N-by-N cells. Model candidate counters report two unique-unordered records per pair."
        ),
    }
    receipt = {
        "schema": SCHEMA,
        "scene_key": _plain(organization.scene_key),
        "case_count": 1,
        "query_rows": query_count,
        "valid_query_rows": int(query.query_mask[0].sum().item()) if query.query_mask is not None else query_count,
        "native_union_rows": int(prepared_native.union.unique_counts.sum().item())
            if getattr(prepared_native, "union", None) is not None else query_count,
        "local_packed_pair_rows": local_pair_count,
        "local_pair_input_rows": local_pair_input_rows,
        "local_pair_queries_without_module_source": local_exceptions,
        "module_source_read_empty": banks["module"]["empty_source_read"],
        "environment_source_read_empty": banks["environment"]["empty_source_read"],
        "module_group_access_empty": banks["module"]["empty_group_access"],
        "environment_group_access_empty": banks["environment"]["empty_group_access"],
        "e_only_query_rows": int(e_only.sum()),
        "banks": banks,
        "cost_inventory": cost,
        "prepare_work_counts": prepare_counters,
        "decode_work_counts": decode_counters,
        "interpretation": "Observed computation and sparse access only; no physical causal or end-to-end speed claim.",
    }
    json.dumps(receipt, sort_keys=True, allow_nan=False)
    return receipt, arrays


def decode_receiver_roles(
    model: Any,
    prepared_native: Any,
    receiver_queries: Mapping[str, Any],
    *,
    return_trace: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode multiple fixed receiver roles from the same prepared physical scene."""
    core = getattr(model, "core", model)
    organization = getattr(prepared_native, "organization", prepared_native)
    if not receiver_queries or not callable(getattr(core, "decode", None)):
        raise ValueError("Provide a prepared organization and at least one named receiver QueryBatch.")
    source_key = getattr(prepared_native, "scene_key", organization.scene_key)
    if _plain(source_key) != _plain(organization.scene_key):
        raise ValueError("Native preparation and contextual scene identities differ; reprepare the inputs.")
    outputs: dict[str, Any] = {}
    role_rows = {}
    prepared_identity = id(organization)
    for role_name, query in receiver_queries.items():
        if not isinstance(role_name, str) or not role_name:
            raise ValueError("Receiver role names must be non-empty strings.")
        if _plain(getattr(query, "scene_key", None)) != _plain(source_key):
            raise ValueError(f"Receiver role {role_name!r} does not belong to the prepared physical scene.")
        output = core.decode(organization, query, return_trace=return_trace)
        outputs[role_name] = output
        role_rows[role_name] = {
            "query_rows": int(query.coordinates.shape[1]),
            "work_counts": {str(key): int(value) for key, value in output.work_counts.items()},
        }
    if id(organization) != prepared_identity:
        raise RuntimeError("Receiver role decoding changed the shared query-independent preparation.")
    return {
        "schema": "contextual_frontier_receiver_reuse_v1",
        "scene_key": _plain(source_key),
        "receiver_roles": role_rows,
        "same_prepared_organization_for_all_roles": True,
        "prepare_calls": 0,
        "interpretation": "Repeated fixed receiver roles reuse this exact prepared input context; changed source controls require a fresh prepare.",
    }, outputs


def fixed_train_panel(
    provider: Any,
    *,
    max_cases: int = 24,
    max_per_module_count: int = 4,
    review_epochs: Sequence[int] = (100, 500),
) -> dict[str, Any]:
    """Select a deterministic input-only TRAIN panel covering every M stratum."""
    if type(max_cases) is not int or max_cases < 1:
        raise ValueError("max_cases must be a positive integer.")
    if type(max_per_module_count) is not int or max_per_module_count < 1:
        raise ValueError("max_per_module_count must be a positive integer.")
    epochs = tuple(int(value) for value in review_epochs)
    if not epochs or any(value < 1 for value in epochs) or tuple(sorted(set(epochs))) != epochs:
        raise ValueError("review_epochs must be a sorted, unique sequence of positive absolute epochs.")
    cases = tuple(getattr(provider, "training_cases", ()))
    case_ids = tuple(map(str, getattr(provider, "training_case_ids", ())))
    if not cases or len(cases) != len(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("The maintained provider must expose aligned unique TRAIN IDs and cases.")
    buckets: dict[int, list[str]] = defaultdict(list)
    for case_id, case in zip(case_ids, cases):
        if str(case.get("case_id", case_id)) != case_id:
            raise ValueError("TRAIN case objects and provider case IDs are not aligned in source membership order.")
        structure = case.get("structure")
        if not isinstance(structure, Mapping) or "module_present" not in structure:
            raise ValueError(f"TRAIN case {case_id!r} has no input-only module_present inventory.")
        present = np.asarray(structure["module_present"]) > 0.5
        buckets[int(present.sum())].append(case_id)
    if max_cases < len(buckets):
        raise ValueError("max_cases is too small to cover every present-module-count stratum.")
    selected: list[str] = []
    target_per_stratum = min(max_per_module_count, max(1, math.ceil(max_cases / len(buckets))))
    while len(selected) < max_cases:
        made_progress = False
        for count in sorted(buckets):
            stratum_rows = buckets[count]
            already = sum(1 for case_id in selected if case_id in stratum_rows)
            if already < min(target_per_stratum, len(stratum_rows)):
                selected.append(stratum_rows[already])
                made_progress = True
                if len(selected) == max_cases:
                    break
        if not made_progress:
            break
    selected_counts = {
        case_id: next(count for count, rows in buckets.items() if case_id in rows)
        for case_id in selected
    }
    strata = {str(count): [case_id for case_id in selected if selected_counts[case_id] == count]
              for count in sorted(buckets)}
    return {
        "schema": "contextual_frontier_fixed_train_panel_v1",
        "case_ids": selected,
        "module_count_by_case": selected_counts,
        "by_module_count": strata,
        "available_train_cases": len(cases),
        "max_cases": max_cases,
        "selection_rule": "first cases in fixed TRAIN membership order, round-robin by present module count; input-only",
        "review_epochs": list(epochs),
        "subset_manifest_sha256": (getattr(provider, "manifest", None) or {}).get("manifest_sha256"),
    }


def _query_source_flags(trace: Mapping[str, Any], bank: str, query_count: int, device: torch.device) -> torch.Tensor:
    flags = torch.zeros((query_count,), dtype=torch.bool, device=device)
    reads = trace[f"{bank}_source_read_attention"][0]
    query_rows = reads.get("edge_query_rows")
    if query_rows is not None and query_rows.numel():
        flags[query_rows.long()] = True
    return flags


def _stencil_e_only_mask(stencil: Any, e_only_query: torch.Tensor) -> torch.Tensor:
    indices = stencil.indices.long()
    weights = stencil.weights
    use = weights > 0
    if e_only_query.ndim == 1:
        gathered = e_only_query[indices]
    else:
        batch = torch.arange(indices.shape[0], device=indices.device).reshape(-1, *([1] * (indices.ndim - 1)))
        gathered = e_only_query[batch, indices]
    all_contributors_e_only = ((~use) | gathered).all(dim=-1)
    return stencil.valid & use.any(dim=-1) & all_contributors_e_only


def _region_metrics(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,
                    common_region: torch.Tensor,
                    channel_map: Sequence[tuple[str, int, int]] | None = None
                    ) -> dict[str, dict[str, float | int | None]]:
    pred = prediction.detach().cpu().double().numpy()
    truth = target.detach().cpu().double().numpy()
    query_weights = weights.detach().cpu().double().numpy()
    region = common_region.detach().cpu().numpy().astype(bool)
    if pred.ndim != 3 or truth.ndim != 3 or pred.shape[:2] != truth.shape[:2] or region.shape != pred.shape[:2]:
        raise ValueError("Regional native predictions, targets, and shared mask have incompatible shapes.")
    mapping = _regional_channel_map(pred.shape[-1], truth.shape[-1], channel_map)
    rows = {}
    for name, predicted_channel, target_channel in mapping:
        finite = np.isfinite(truth[..., target_channel]) & np.isfinite(pred[..., predicted_channel])
        mask = region & finite & (query_weights > 0)
        local_weights = query_weights * mask
        denominator = float(local_weights.sum())
        error = np.nan_to_num(pred[..., predicted_channel] - truth[..., target_channel])
        target_values = np.nan_to_num(truth[..., target_channel])
        error_l2 = math.sqrt(float(np.sum(error * error * local_weights)))
        target_l2 = math.sqrt(float(np.sum(target_values * target_values * local_weights)))
        rows[name] = {
            "valid_rows": int(mask.sum()),
            "rmse": math.sqrt(float(np.sum(error * error * local_weights)) / denominator) if denominator else None,
            "relative_l2": error_l2 / max(target_l2, 1e-12) if denominator else None,
        }
    return rows


def _evaluate_common_e_only_quality(
    arms: Mapping[str, tuple[Any, Any]],
    case_ids: Sequence[str],
    *,
    epoch: int,
    split: str = "validation",
    channel_map: Sequence[tuple[str, int, int]] | None = None,
) -> dict[str, Any]:
    """Compare exposed provider field errors on the shared actual E-only mask.

    A fluid target contributes only when every positive-weight native-union
    stencil row is E-read and M-unread in every arm. The same provider target,
    point weight, and geometric validity mask is then used for all arms.
    """
    if split != "validation":
        raise ValueError("Regional E-only quality is restricted to the exposed validation provider.")
    if not arms or not case_ids or len(set(map(str, case_ids))) != len(case_ids):
        raise ValueError("Provide unique case IDs and at least one contextual arm.")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive absolute organization epoch.")
    ids = tuple(map(str, case_ids))
    predictions_by_case: dict[str, dict[str, tuple[Any, Any, Any, torch.Tensor, Mapping[str, Any]]]] = {}
    for arm_name, (model, provider) in arms.items():
        exposed = set(map(str, getattr(provider, "validation_case_ids", ())))
        if not set(ids).issubset(exposed):
            raise ValueError(f"The case panel is outside {arm_name!r}'s exposed validation membership.")
        model.eval()
        model.core.set_organization_clock(epoch, 0)
        batches = provider.native_evaluation_batches(split="validation", case_keys=ids, microbatch_size=1)
        observed = []
        for batch in batches:
            case_batch = tuple(map(str, batch.case_keys))
            if len(case_batch) != 1 or case_batch[0] != ids[len(observed)]:
                raise ValueError("Validation provider changed the fixed regional case order.")
            with torch.no_grad():
                predictions, _aux = provider.predict_native(
                    model, batch.scene_inputs, batch.receivers, model.architecture, "validation", epoch=epoch,
                )
                prepared = predictions.native_prepared
                traced = model.core.decode(prepared.organization, prepared.query_batch, return_trace=True)
            trace = traced.trace
            if not isinstance(trace, Mapping):
                raise RuntimeError("The core returned no actual sparse read trace for E-only evaluation.")
            query_count = int(prepared.query_batch.coordinates.shape[1])
            module_read = _query_source_flags(trace, "module", query_count, prepared.query_batch.coordinates.device)
            environment_read = _query_source_flags(trace, "environment", query_count, prepared.query_batch.coordinates.device)
            e_only = (~module_read) & environment_read
            fluid_region = _stencil_e_only_mask(prepared.stencils["fluid"], e_only)
            predictions_by_case.setdefault(case_batch[0], {})[arm_name] = (
                batch, predictions, prepared, fluid_region, traced.work_counts,
            )
            observed.append(case_batch[0])
        if tuple(observed) != ids:
            raise ValueError("Validation provider did not return the exact fixed regional panel.")

    case_rows = []
    aggregate: dict[str, dict[str, list[float]]] = {arm: {} for arm in arms}
    for case_id in ids:
        per_arm = predictions_by_case[case_id]
        common_region = None
        physical_mask = None
        for arm_name in arms:
            batch, predictions, prepared, region, _counts = per_arm[arm_name]
            common_region = region if common_region is None else common_region & region
            valid = prepared.stencils["fluid"].valid & (batch.targets.point_weights > 0)
            physical_mask = valid if physical_mask is None else physical_mask & valid
        common_region = common_region & physical_mask
        reference_batch = per_arm[next(iter(arms))][0]
        target = reference_batch.targets.field_targets
        point_weights = reference_batch.targets.point_weights
        for arm_name in arms:
            batch, _predictions, _prepared, _region, _counts = per_arm[arm_name]
            if not torch.equal(batch.targets.point_weights, point_weights):
                raise ValueError(f"Provider point weights differ across arms for validation case {case_id!r}.")
            if not torch.allclose(batch.targets.field_targets, target, rtol=0.0, atol=0.0, equal_nan=True):
                raise ValueError(f"Provider field targets differ across arms for validation case {case_id!r}.")
        row: dict[str, Any] = {
            "case_id": case_id,
            "common_e_only_physical_queries": int(common_region.sum().item()),
            "e_only_queries_by_arm": {
                name: int(per_arm[name][3].sum().item()) for name in arms
            },
            "actual_read_edges_by_arm": {
                name: {
                "module": int(per_arm[name][4].get("module_value_read_rows", 0)),
                "environment": int(per_arm[name][4].get("environment_value_read_rows", 0)),
                } for name in arms
            },
            "empty_source_reads_by_arm": {
                name: {
                    "module": int(per_arm[name][4].get("module_value_read_rows", 0)) == 0,
                    "environment": int(per_arm[name][4].get("environment_value_read_rows", 0)) == 0,
                } for name in arms
            },
            "common_physical_mask": "fluid stencil valid AND positive provider point weight; per-channel finite targets are applied within each metric",
            "interpretation": "Observed access subset; not a physical noninterference or causal claim.",
            "arms": {},
        }
        for arm_name in arms:
            batch, predictions, _prepared, _region, _counts = per_arm[arm_name]
            metric = _region_metrics(
                predictions.native_main["pred_field"], target,
                batch.targets.point_weights, common_region, channel_map,
            )
            row["arms"][arm_name] = metric
            for channel in metric:
                value = metric[channel]["relative_l2"]
                if value is not None and math.isfinite(float(value)):
                    aggregate[arm_name].setdefault(channel, []).append(float(value))
        case_rows.append(row)
    return {
        "schema": "contextual_frontier_common_e_only_quality_v1",
        "epoch": epoch,
        "split": split,
        "case_ids": list(ids),
        "per_case": case_rows,
        "equal_case_mean_relative_l2_on_common_mask": {
            arm: {channel: (float(np.mean(values)) if values else None) for channel, values in channels.items()}
            for arm, channels in aggregate.items()
        },
        "channel_map": None if channel_map is None else [list(row) for row in channel_map],
        "interpretation": "Regional accuracy uses the intersection of observed E-only reads and common physical target masks; it does not establish causality.",
    }


def evaluate_common_e_only_quality(
    arms: Mapping[str, tuple[Any, Any]],
    case_ids: Sequence[str],
    *,
    epoch: int,
    split: str = "validation",
    channel_map: Sequence[tuple[str, int, int]] | None = None,
) -> dict[str, Any]:
    """Evaluate shared E-only validation queries and restore model clocks/modes."""
    if not arms:
        raise ValueError("Provide at least one contextual arm.")
    old_state = {
        name: (bool(model.training), int(model.core.organization_epoch), int(model.core.organization_update))
        for name, (model, _provider) in arms.items()
    }
    try:
        return _evaluate_common_e_only_quality(
            arms, case_ids, epoch=epoch, split=split, channel_map=channel_map
        )
    finally:
        for name, (model, _provider) in arms.items():
            was_training, old_epoch, old_update = old_state[name]
            model.core.set_organization_clock(old_epoch, old_update)
            model.train(was_training)


def compare_reprepared_source_change(
    model: Any,
    prepared_native: Any,
    changed_structure: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reprepare actual changed source inputs and report contextual/output deltas."""
    adapter = model
    old_scene = prepared_native.organization.scene
    new_scene = adapter.scene_batch(changed_structure, scene_key=prepared_native.scene_key)
    compared = ("module_features", "environment_features", "operating_features")
    changed = {
        name: not torch.equal(getattr(old_scene, name), getattr(new_scene, name))
        for name in compared
        if getattr(old_scene, name) is not None and getattr(new_scene, name) is not None
    }
    if not any(changed.values()):
        raise ValueError("Changed structure did not alter module, environment, or operating context inputs.")
    preserved_scene_fields = (
        "module_coordinates", "environment_coordinates", "module_ids", "environment_ids",
        "module_mask", "environment_mask", "module_measure", "environment_measure", "local_radius",
        "coordinate_origin", "coordinate_scale", "module_measure_reference", "environment_measure_reference",
        "module_characteristic_lengths", "environment_characteristic_lengths",
    )
    same_physical_records = all(
        (getattr(old_scene, name) is None and getattr(new_scene, name) is None)
        or (getattr(old_scene, name) is not None and getattr(new_scene, name) is not None
            and torch.equal(getattr(old_scene, name), getattr(new_scene, name)))
        for name in preserved_scene_fields
    )
    same_physical_records = same_physical_records and (
        _plain(old_scene.normalizer_key) == _plain(new_scene.normalizer_key)
        and _plain(old_scene.coordinate_frame_key) == _plain(new_scene.coordinate_frame_key)
    )
    if not same_physical_records:
        raise ValueError("Source-change probe must preserve record IDs, geometry, physical measures, and masks.")
    rebuilt = adapter.prepare_native(
        changed_structure,
        prepared_native.fluid_xy,
        local_query_points=prepared_native.local_query_points,
        ntheta=prepared_native.ntheta,
        scene_key=prepared_native.scene_key,
    )
    if rebuilt is prepared_native or rebuilt.organization is prepared_native.organization:
        raise RuntimeError("Changed source inputs reused stale native preparation instead of rebuilding context.")
    for name in compared:
        expected = getattr(new_scene, name)
        actual = getattr(rebuilt.organization.scene, name)
        if (expected is None) != (actual is None) or (
            expected is not None and not torch.equal(expected, actual)
        ):
            raise RuntimeError("Fresh native preparation did not consume the changed prescribed context inputs.")
    if not torch.equal(prepared_native.query_batch.coordinates, rebuilt.query_batch.coordinates):
        raise ValueError("Source-change reprepare changed the fixed native receiver coordinates.")
    for field in ("query_mask", "local_batch_indices", "local_query_indices", "local_module_indices", "local_weights"):
        old_value = getattr(prepared_native.query_batch, field, None)
        new_value = getattr(rebuilt.query_batch, field, None)
        if (old_value is None) != (new_value is None) or (
            old_value is not None and not torch.equal(old_value, new_value)
        ):
            raise ValueError("Source-change reprepare changed the fixed native query/local receiver map.")
    with torch.no_grad():
        before = model.core.decode(prepared_native.organization, prepared_native.query_batch)
        after = model.core.decode(rebuilt.organization, rebuilt.query_batch)
    if before.joint_fields.shape != after.joint_fields.shape:
        raise ValueError("The unchanged physical geometry produced a different receiver-union shape.")
    arrays = {
        "module_context_delta": _numpy(rebuilt.organization.module_states - prepared_native.organization.module_states),
        "environment_context_delta": _numpy(rebuilt.organization.environment_states - prepared_native.organization.environment_states),
        "joint_field_delta": _numpy(after.joint_fields - before.joint_fields),
    }
    native_deltas: dict[str, float] = {}
    apply_native = getattr(adapter, "apply_native", None)
    if callable(apply_native):
        with torch.no_grad():
            before_native = apply_native(prepared_native)
            after_native = apply_native(rebuilt)
        if isinstance(before_native, Mapping) and isinstance(after_native, Mapping):
            for name in ("pred_field", "pred_interface", "pred_internal_temperature"):
                old_value, new_value = before_native.get(name), after_native.get(name)
                if torch.is_tensor(old_value) and torch.is_tensor(new_value):
                    if old_value.shape != new_value.shape:
                        raise ValueError(f"Native output {name!r} changed shape under a fixed physical receiver map.")
                    delta = _numpy(new_value - old_value)
                    arrays[f"native_{name}_delta"] = delta
                    native_deltas[name] = float(np.linalg.norm(delta.reshape(-1)))
    receipt = {
        "schema": "contextual_frontier_source_reprepare_v1",
        "scene_key": _plain(prepared_native.scene_key),
        "changed_input_tensors": changed,
        "reprepared_new_native_object": rebuilt is not prepared_native,
        "reprepared_new_context_object": rebuilt.organization is not prepared_native.organization,
        "same_physical_records_geometry_and_masks": same_physical_records,
        "module_context_delta_l2": float(np.linalg.norm(arrays["module_context_delta"].reshape(-1))),
        "environment_context_delta_l2": float(np.linalg.norm(arrays["environment_context_delta"].reshape(-1))),
        "joint_field_delta_l2": float(np.linalg.norm(arrays["joint_field_delta"].reshape(-1))),
        "native_output_delta_l2": native_deltas,
        "interpretation": "Actual fresh-prepare response to changed supplied source/context inputs; model-space and available native-output deltas are observational, not causal attribution to an individual source.",
    }
    return receipt, arrays


def _resolve_device(device: str | torch.device | None, module: Any | None) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
    else:
        parameter = next(iter(module.parameters()), None) if module is not None and callable(
            getattr(module, "parameters", None)
        ) else None
        resolved = parameter.device if parameter is not None else torch.device("cpu")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the cost measurement but is unavailable.")
    return resolved


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(function: Callable[[], Any], device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        _synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        _synchronize(device)
        return result, float(start.elapsed_time(end))
    start_time = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start_time) * 1000.0


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "samples_ms": [float(value) for value in values],
        "mean_ms": float(statistics.mean(values)) if values else None,
        "median_ms": float(statistics.median(values)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
    }


def measure_native_cost(
    prepare: Callable[[], Any],
    decode: Callable[[Any], Any],
    *,
    reference: Callable[[Any], Any] | None = None,
    loss: Callable[[Any], torch.Tensor] | None = None,
    module: Any | None = None,
    device: str | torch.device | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    """Measure prepare, decode, optional reference decode, and backward separately.

    This is a measurement helper only: it never steps an optimizer. On CUDA,
    each stage uses synchronized events; CPU uses ``perf_counter``. Supplying
    ``loss`` enables a real autograd backward over the prepared+decoded graph.
    """
    if not callable(prepare) or not callable(decode) or (reference is not None and not callable(reference)):
        raise TypeError("prepare/decode/reference must be callables.")
    if loss is not None and not callable(loss):
        raise TypeError("loss must be callable when supplied.")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    resolved_device = _resolve_device(device, module)
    records: dict[str, list[float]] = {"prepare": [], "decode": [], "backward": [], "reference": []}
    peak_absolute: list[int] = []
    peak_delta: list[int] = []
    allocated_baseline: list[int] = []
    for _ in range(repeats):
        if module is not None and callable(getattr(module, "zero_grad", None)):
            module.zero_grad(set_to_none=True)
        if resolved_device.type == "cuda":
            _synchronize(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
            baseline = int(torch.cuda.memory_allocated(resolved_device))
            allocated_baseline.append(baseline)
        prepared, prepare_ms = _time_call(prepare, resolved_device)
        records["prepare"].append(prepare_ms)
        output, decode_ms = _time_call(lambda: decode(prepared), resolved_device)
        records["decode"].append(decode_ms)
        if reference is not None:
            reference_output, reference_ms = _time_call(lambda: reference(prepared), resolved_device)
            records["reference"].append(reference_ms)
            del reference_output
        if loss is not None:
            objective = loss(output)
            if not torch.is_tensor(objective) or objective.numel() != 1 or not objective.requires_grad:
                raise ValueError("loss(output) must be one differentiable scalar.")
            _, backward_ms = _time_call(lambda: objective.backward(), resolved_device)
            records["backward"].append(backward_ms)
            del objective
        if resolved_device.type == "cuda":
            _synchronize(resolved_device)
            peak = int(torch.cuda.max_memory_allocated(resolved_device))
            peak_absolute.append(peak)
            peak_delta.append(max(0, peak - baseline))
        del output, prepared
    if module is not None and callable(getattr(module, "zero_grad", None)):
        module.zero_grad(set_to_none=True)
    _synchronize(resolved_device)
    return {
        "schema": "contextual_frontier_native_cost_v1",
        "device": str(resolved_device),
        "timing_source": "synchronized_cuda_events" if resolved_device.type == "cuda" else "perf_counter",
        "optimizer_steps": 0,
        "repeats": repeats,
        "prepare": _timing_summary(records["prepare"]),
        "decode": _timing_summary(records["decode"]),
        "backward": _timing_summary(records["backward"]),
        "reference_decode": _timing_summary(records["reference"]),
        "cuda_memory": None if resolved_device.type != "cuda" else {
            "allocated_baseline_bytes": allocated_baseline,
            "peak_allocated_bytes": peak_absolute,
            "peak_increment_bytes": peak_delta,
            "peak_increment_max_bytes": max(peak_delta, default=0),
        },
        "interpretation": "Measured callable stage costs only; not a training-throughput or end-to-end speed claim.",
    }


def write_evidence_bundle(
    output_dir: str | Path,
    receipt: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    max_output_bytes: int,
    minimum_free_headroom_bytes: int,
) -> dict[str, Any]:
    """Write a caller-selected ignored evidence directory beneath mounted DATA2.

    No output root is chosen here. The caller must pass an already-existing
    ignored parent, a hard payload budget, and the required filesystem headroom.
    """
    if type(max_output_bytes) is not int or max_output_bytes < 1:
        raise ValueError("max_output_bytes must be supplied as a positive hard budget.")
    if type(minimum_free_headroom_bytes) is not int or minimum_free_headroom_bytes < 0:
        raise ValueError("minimum_free_headroom_bytes must be supplied as a nonnegative budget.")
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to write organization evidence.")
    out = Path(output_dir).expanduser().resolve(strict=False)
    parent = out.parent.resolve(strict=True)
    storage_root = (data2 / "wanglz/ModularDT").resolve(strict=True)
    if not parent.is_relative_to(storage_root) or parent.stat().st_dev != data2.stat().st_dev:
        raise ValueError("Organization evidence must be written under the mounted /data2 workspace.")
    if out.exists():
        raise FileExistsError(out)
    json_payload = json.dumps(dict(receipt), sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{str(name): np.asarray(value) for name, value in arrays.items()})
    npz_payload = buffer.getvalue()
    payload_bytes = len(json_payload) + len(npz_payload)
    if payload_bytes > max_output_bytes:
        raise OSError(f"Organization evidence is {payload_bytes} bytes, over budget {max_output_bytes}.")
    free_before = int(shutil.disk_usage(parent).free)
    if free_before - payload_bytes < minimum_free_headroom_bytes:
        raise OSError("Organization evidence would breach the required filesystem headroom.")
    out.mkdir()
    try:
        for name, payload in (("receipt.json", json_payload), ("arrays.npz", npz_payload)):
            fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=out)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, out / name)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
    except Exception:
        for child in out.iterdir():
            child.unlink()
        out.rmdir()
        raise
    free_after = int(shutil.disk_usage(parent).free)
    return {
        "output_dir": str(out), "receipt_bytes": len(json_payload), "arrays_bytes": len(npz_payload),
        "total_bytes": payload_bytes, "maximum_output_bytes": max_output_bytes,
        "free_bytes_before": free_before, "free_bytes_after": free_after,
        "minimum_free_headroom_bytes": minimum_free_headroom_bytes,
    }
