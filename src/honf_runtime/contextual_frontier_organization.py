"""Bounded, input-identified evidence for contextual frontier organization.

These helpers inspect the maintained native adapter/core objects. Their
access summaries describe executed retrieval and do not establish physical
causality or end-to-end speed.
"""
from __future__ import annotations

import copy
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "contextual_frontier_organization_evidence_v1"
_ORGANIZATION_EVALUATION_MAX_CASES = 24
_ORGANIZATION_EVALUATION_DETAIL_SCENES = 4
_ORGANIZATION_TRACE_QUERY_LIMIT = 256
_ORGANIZATION_EVALUATION_MAX_BYTES = 128 * 1024**2
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


def _native_e_only_region(prepared: Any, e_only_query: torch.Tensor) -> torch.Tensor:
    region = _stencil_e_only_mask(prepared.stencils["fluid"], e_only_query)
    curl = getattr(prepared, "curl_stencil", None)
    if curl is not None:
        # Conservative common mask for all five native fluid outputs: a curl
        # query also consumes its neighbor nodes, even when its center is E-only.
        contributors = e_only_query[curl.gather_indices.long()]
        region = region & contributors.all(dim=-1)
    return region


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
        mask = region & np.isfinite(truth[..., target_channel]) & (query_weights > 0)
        nonfinite_predictions = int((mask & ~np.isfinite(pred[..., predicted_channel])).sum())
        local_weights = query_weights * mask
        denominator = float(local_weights.sum())
        error = np.nan_to_num(pred[..., predicted_channel] - truth[..., target_channel],
                              nan=0.0, posinf=0.0, neginf=0.0)
        target_values = np.nan_to_num(truth[..., target_channel], nan=0.0, posinf=0.0, neginf=0.0)
        error_l2 = math.sqrt(float(np.sum(error * error * local_weights)))
        target_l2 = math.sqrt(float(np.sum(target_values * target_values * local_weights)))
        rows[name] = {
            "valid_rows": int(mask.sum()),
            "nonfinite_prediction_rows": nonfinite_predictions,
            "status": "invalid_nonfinite_prediction" if nonfinite_predictions else "valid" if denominator else "empty",
            "rmse": math.sqrt(float(np.sum(error * error * local_weights)) / denominator)
                if denominator and not nonfinite_predictions else None,
            "relative_l2": error_l2 / target_l2
                if denominator and target_l2 > 0 and not nonfinite_predictions else None,
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
            fluid_region = _native_e_only_region(prepared, e_only)
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
        output, decode_ms = _time_call(lambda prepared=prepared: decode(prepared), resolved_device)
        records["decode"].append(decode_ms)
        if reference is not None:
            reference_output, reference_ms = _time_call(lambda prepared=prepared: reference(prepared), resolved_device)
            records["reference"].append(reference_ms)
            del reference_output
        if loss is not None:
            objective = loss(output)
            if not torch.is_tensor(objective) or objective.numel() != 1 or not objective.requires_grad:
                raise ValueError("loss(output) must be one differentiable scalar.")
            _, backward_ms = _time_call(lambda objective=objective: objective.backward(), resolved_device)
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


def _detail_train_cases(panel: Mapping[str, Any], *, limit: int) -> tuple[list[str], list[int]]:
    """Choose a small input-only representative set and expose omitted strata."""
    case_ids = [str(value) for value in panel["case_ids"]]
    counts = {str(key): int(value) for key, value in panel["module_count_by_case"].items()}
    strata = sorted({counts[case_id] for case_id in case_ids})
    if len(strata) > limit:
        selected_strata = [strata[int(index)] for index in np.linspace(0, len(strata) - 1, num=limit).round()]
    else:
        selected_strata = strata
    chosen = []
    for count in selected_strata:
        chosen.append(next(case_id for case_id in case_ids if counts[case_id] == count))
    for case_id in case_ids:
        if len(chosen) >= limit:
            break
        if case_id not in chosen:
            chosen.append(case_id)
    chosen_counts = [counts[case_id] for case_id in chosen]
    return chosen, chosen_counts


def _bounded_query_subset(query: Any, *, limit: int) -> tuple[Any, np.ndarray]:
    """Take an input-order-spread valid receiver subset and remap packed local pairs."""
    if query.coordinates.ndim != 3 or query.coordinates.shape[0] != 1:
        raise ValueError("Bounded contextual trace selection requires a single native scene.")
    query_count = int(query.coordinates.shape[1])
    if query.query_mask is None:
        valid = torch.arange(query_count, device=query.coordinates.device, dtype=torch.long)
    else:
        valid = torch.nonzero(query.query_mask[0], as_tuple=False).reshape(-1)
    if not valid.numel():
        raise ValueError("Cannot trace a native receiver union with no valid queries.")
    if valid.numel() > limit:
        positions = np.linspace(0, int(valid.numel()) - 1, num=limit).round().astype(np.int64)
        selected = valid.index_select(0, torch.as_tensor(positions, device=valid.device, dtype=torch.long))
    else:
        selected = valid
    old_to_new = torch.full((query_count,), -1, device=query.coordinates.device, dtype=torch.long)
    old_to_new[selected] = torch.arange(selected.numel(), device=selected.device, dtype=torch.long)

    def sliced(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.index_select(1, selected)

    packed = (query.local_batch_indices, query.local_query_indices,
              query.local_module_indices, query.local_weights)
    if any(value is not None for value in packed):
        if not all(value is not None for value in packed):
            raise ValueError("Native query has incomplete packed local-pair metadata.")
        keep = (query.local_batch_indices == 0) & (old_to_new[query.local_query_indices.long()] >= 0)
        local_batch = query.local_batch_indices[keep]
        local_query = old_to_new[query.local_query_indices[keep].long()]
        local_module = query.local_module_indices[keep]
        local_weights = query.local_weights[keep]
    else:
        local_batch = local_query = local_module = local_weights = None
    subset = replace(
        query,
        coordinates=sliced(query.coordinates),
        features=sliced(query.features),
        query_mask=sliced(query.query_mask),
        query_ids=sliced(query.query_ids),
        local_batch_indices=local_batch,
        local_query_indices=local_query,
        local_module_indices=local_module,
        local_weights=local_weights,
    )
    return subset, _numpy(selected).astype(np.int64, copy=False)


def _remap_inspection_query_rows(arrays: dict[str, np.ndarray], original_rows: np.ndarray) -> None:
    for name, values in tuple(arrays.items()):
        if name.endswith((
            "_group_record_route_query_index", "_query_source_index", "_read_query_index",
        )) or name == "local_pair_query_index":
            if values.size:
                arrays[name] = original_rows[values.astype(np.int64, copy=False)]


def _native_with_query(prepared_native: Any, query: Any) -> Any:
    if hasattr(prepared_native, "__dataclass_fields__"):
        return replace(prepared_native, query_batch=query)
    result = copy.copy(prepared_native)
    result.query_batch = query
    return result


def _masked_rms(
    values: np.ndarray,
    reference_mask: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    equal_modules: bool = False,
) -> tuple[float | None, int, bool]:
    """Use a fixed physical denominator; any nonfinite prediction invalidates it."""
    count = int(reference_mask.sum())
    if count == 0:
        return None, 0, True
    if not np.isfinite(values[reference_mask]).all():
        return None, count, False
    if weights is not None:
        local_weights = weights * reference_mask
        denominator = float(local_weights.sum())
        return (math.sqrt(float(np.sum(np.square(values) * local_weights) / denominator))
                if denominator > 0 else None), count, True
    if equal_modules:
        module_mse = [float(np.mean(np.square(values[index, reference_mask[index]], dtype=np.float64)))
                      for index in range(values.shape[0]) if reference_mask[index].any()]
        return (math.sqrt(float(np.mean(module_mse))) if module_mse else None), count, True
    return math.sqrt(float(np.mean(np.square(values[reference_mask], dtype=np.float64)))), count, True


def _native_output_changes(
    baseline: Any,
    parent_stop: Any,
    descendants: Any,
    batch: Any,
    metric_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure full native output deltas on the provider's shared physical masks."""
    base_output = baseline.native_main
    coarse_output = parent_stop.native_main
    fine_output = descendants.native_main
    base_prepared = baseline.native_prepared
    target = batch.targets

    def metric_rmse(row: Mapping[str, Any], role: str) -> float | None:
        if role in ("u", "v", "p", "omega"):
            result = row["flow_rmse"][role]
        elif role.startswith("temperature_"):
            result = row["roles"][role]["rmse"]
        elif role == "q_proxy":
            result = row["q_proxy_rmse"]
        else:
            raise KeyError(role)
        value = float(result)
        return value if math.isfinite(value) else None

    def ratio(change: float | None, baseline_rmse: float | None) -> float | None:
        return change / max(baseline_rmse, 1e-12) if change is not None and baseline_rmse is not None else None

    changes: dict[str, Any] = {}
    field_names = (("u", 0), ("v", 1), ("p", 2), ("omega", 3), ("temperature_fluid", 4))
    base_field = _numpy(base_output["pred_field"])[0].astype(np.float64, copy=False)
    coarse_field = _numpy(coarse_output["pred_field"])[0].astype(np.float64, copy=False)
    fine_field = _numpy(fine_output["pred_field"])[0].astype(np.float64, copy=False)
    target_field = _numpy(target.field_targets)[0].astype(np.float64, copy=False)
    fluid_valid = _numpy(base_prepared.stencils["fluid"].valid)[0].astype(bool, copy=False)
    point_weights = _numpy(target.point_weights)[0].astype(np.float64, copy=False)
    for role, channel in field_names:
        mask = fluid_valid & (point_weights > 0) & np.isfinite(target_field[:, channel])
        role_row = {
            "baseline_native_rmse": metric_rmse(metric_rows["baseline"], role),
            "parent_stop_native_rmse": metric_rmse(metric_rows["parent_stop"], role),
            "descendant_native_rmse": metric_rmse(metric_rows["descendant"], role),
            "valid_rows": int(mask.sum()),
            "mask": "shared fluid stencil validity, positive provider point weight, and finite target; prediction nonfinites invalidate the component",
        }
        finite_all = all(np.isfinite(values[mask]).all() for values in (
            base_field[:, channel], coarse_field[:, channel], fine_field[:, channel]
        ))
        role_row["all_predictions_finite_on_reference_mask"] = bool(finite_all)
        for label, left, right in (
            ("parent_stop_vs_baseline", coarse_field[:, channel], base_field[:, channel]),
            ("descendant_vs_baseline", fine_field[:, channel], base_field[:, channel]),
            ("parent_stop_vs_descendant", coarse_field[:, channel], fine_field[:, channel]),
        ):
            baseline_rmse = role_row["baseline_native_rmse"]
            rms, _count, finite = _masked_rms(left - right, mask, weights=point_weights)
            if not finite:
                rms = None
            role_row[f"{label}_change_rms"] = rms
            role_row[f"{label}_change_over_baseline_rmse"] = ratio(rms, baseline_rmse)
        changes[role] = role_row

    base_interface = _numpy(base_output["pred_interface"])[0].astype(np.float64, copy=False)
    coarse_interface = _numpy(coarse_output["pred_interface"])[0].astype(np.float64, copy=False)
    fine_interface = _numpy(fine_output["pred_interface"])[0].astype(np.float64, copy=False)
    interface_target = _numpy(target.interface_target)[0].astype(np.float64, copy=False)
    present = _numpy(base_prepared.source_present)[0].astype(bool, copy=False)
    interface_valid = _numpy(target.interface_valid_mask)[0] > 0.5
    surface_valid = _numpy(base_prepared.stencils["surface"].valid).reshape(base_interface.shape[:-1]).astype(bool)
    outside_valid = _numpy(base_prepared.stencils["outside"].valid).reshape(base_interface.shape[:-1]).astype(bool)
    for role, channel, geometric_mask in (
        ("temperature_surface", 0, surface_valid),
        ("q_proxy", 1, surface_valid & outside_valid),
    ):
        mask = (
            present[:, None] & geometric_mask & interface_valid
            & np.isfinite(interface_target[..., channel])
        )
        role_row = {
            "baseline_native_rmse": metric_rmse(metric_rows["baseline"], role),
            "parent_stop_native_rmse": metric_rmse(metric_rows["parent_stop"], role),
            "descendant_native_rmse": metric_rmse(metric_rows["descendant"], role),
            "valid_rows": int(mask.sum()),
            "active_modules": int(np.any(mask, axis=1).sum()),
            "mask": "shared native surface/outside geometry, source-present slots, and finite target; prediction nonfinites invalidate the component",
            "reduction": "equal mean over active modules, then square root; matches maintained native role reduction",
        }
        finite_all = all(np.isfinite(values[..., channel][mask]).all() for values in (
            base_interface, coarse_interface, fine_interface
        ))
        role_row["all_predictions_finite_on_reference_mask"] = bool(finite_all)
        for label, left, right in (
            ("parent_stop_vs_baseline", coarse_interface[..., channel], base_interface[..., channel]),
            ("descendant_vs_baseline", fine_interface[..., channel], base_interface[..., channel]),
            ("parent_stop_vs_descendant", coarse_interface[..., channel], fine_interface[..., channel]),
        ):
            rms, _count, finite = _masked_rms(left - right, mask, equal_modules=True)
            if not finite:
                rms = None
            baseline_rmse = role_row["baseline_native_rmse"]
            role_row[f"{label}_change_rms"] = rms
            role_row[f"{label}_change_over_baseline_rmse"] = ratio(rms, baseline_rmse)
        changes[role] = role_row

    base_material = _numpy(base_output["pred_internal_temperature"])[0, ..., 0].astype(np.float64, copy=False)
    coarse_material = _numpy(coarse_output["pred_internal_temperature"])[0, ..., 0].astype(np.float64, copy=False)
    fine_material = _numpy(descendants.native_main["pred_internal_temperature"])[0, ..., 0].astype(np.float64, copy=False)
    material_target = _numpy(target.material_targets)[0].astype(np.float64, copy=False)
    material_valid = _numpy(base_prepared.stencils["material"].valid).reshape(material_target.shape).astype(bool)
    material_mask = (
        present[:, None] & material_valid & np.isfinite(material_target)
    )
    role = "temperature_material"
    role_row = {
        "baseline_native_rmse": metric_rmse(metric_rows["baseline"], role),
        "parent_stop_native_rmse": metric_rmse(metric_rows["parent_stop"], role),
        "descendant_native_rmse": metric_rmse(metric_rows["descendant"], role),
        "valid_rows": int(material_mask.sum()),
        "active_modules": int(np.any(material_mask, axis=1).sum()),
        "mask": "shared native material geometry, source-present slots, and finite target; prediction nonfinites invalidate the component",
        "reduction": "equal mean over active modules, then square root; matches maintained native role reduction",
    }
    finite_all = all(np.isfinite(values[material_mask]).all() for values in (
        base_material, coarse_material, fine_material
    ))
    role_row["all_predictions_finite_on_reference_mask"] = bool(finite_all)
    for label, left, right in (
        ("parent_stop_vs_baseline", coarse_material, base_material),
        ("descendant_vs_baseline", fine_material, base_material),
        ("parent_stop_vs_descendant", coarse_material, fine_material),
    ):
        rms, _count, finite = _masked_rms(left - right, material_mask, equal_modules=True)
        if not finite:
            rms = None
        baseline_rmse = role_row["baseline_native_rmse"]
        role_row[f"{label}_change_rms"] = rms
        role_row[f"{label}_change_over_baseline_rmse"] = ratio(rms, baseline_rmse)
    changes[role] = role_row
    endpoint_rows = []
    endpoint_reference_errors = []
    endpoint_deltas: dict[str, list[float]] = {
        "parent_stop_vs_baseline": [], "descendant_vs_baseline": [], "parent_stop_vs_descendant": [],
    }
    endpoint_nonfinite_modules = []
    for module in range(material_mask.shape[0]):
        active = material_mask[module]
        if not active.any():
            continue
        reference_values = material_target[module, active]
        source_values = base_material[module, active]
        parent_values = coarse_material[module, active]
        descendant_values = fine_material[module, active]
        all_finite = all(np.isfinite(values).all() for values in (
            reference_values, source_values, parent_values, descendant_values
        ))
        if not all_finite:
            endpoint_nonfinite_modules.append(module)
            endpoint_rows.append({"module_index": module, "valid_rows": int(active.sum()), "finite": False})
            continue
        target_max = float(np.max(reference_values))
        baseline_max = float(np.max(source_values))
        parent_max = float(np.max(parent_values))
        descendant_max = float(np.max(descendant_values))
        baseline_error = baseline_max - target_max
        parent_error = parent_max - target_max
        descendant_error = descendant_max - target_max
        endpoint_reference_errors.append(baseline_error)
        deltas = {
            "parent_stop_vs_baseline": parent_max - baseline_max,
            "descendant_vs_baseline": descendant_max - baseline_max,
            "parent_stop_vs_descendant": parent_max - descendant_max,
        }
        for name, value in deltas.items():
            endpoint_deltas[name].append(value)
        endpoint_rows.append({
            "module_index": module,
            "valid_rows": int(active.sum()),
            "finite": True,
            "target_max": target_max,
            "baseline_predicted_max": baseline_max,
            "parent_stop_predicted_max": parent_max,
            "descendant_predicted_max": descendant_max,
            "baseline_endpoint_error": baseline_error,
            "parent_stop_endpoint_error": parent_error,
            "descendant_endpoint_error": descendant_error,
            **{f"{name}_maximum_change": value for name, value in deltas.items()},
        })
    endpoint_rmse = (
        math.sqrt(float(np.mean(np.square(endpoint_reference_errors))))
        if endpoint_reference_errors and not endpoint_nonfinite_modules else None
    )
    endpoint_row: dict[str, Any] = {
        "baseline_native_endpoint_rmse": endpoint_rmse,
        "baseline_native_material_rmse": metric_rmse(metric_rows["baseline"], "temperature_material"),
        "active_modules": len(endpoint_rows),
        "finite_modules": int(len(endpoint_rows) - len(endpoint_nonfinite_modules)),
        "nonfinite_modules": endpoint_nonfinite_modules,
        "per_module": endpoint_rows,
        "mask": "all source-present material receivers with valid native material geometry and finite targets; any nonfinite prediction invalidates endpoint RMS",
        "maximum_definition": "separate prediction and target maxima over every valid full-native material receiver for each module",
    }
    for name, values in endpoint_deltas.items():
        change = None if endpoint_nonfinite_modules or not values else math.sqrt(float(np.mean(np.square(values))))
        endpoint_row[f"{name}_change_rms"] = change
        endpoint_row[f"{name}_change_over_baseline_endpoint_rmse"] = ratio(change, endpoint_rmse)
        endpoint_row[f"{name}_change_over_baseline_material_rmse"] = ratio(
            change, endpoint_row["baseline_native_material_rmse"]
        )
    changes["temperature_material_endpoint_maxima"] = endpoint_row
    return changes


def _compact_actual_support(
    core: Any,
    organization: Any,
    query: Any,
    *,
    array_prefix: str,
    original_query_rows: np.ndarray,
    full_query_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Capture sparse query-to-group/source support from an executed trace."""
    with torch.no_grad():
        output = core.decode(organization, query, return_trace=True)
    if not isinstance(output.trace, Mapping):
        raise RuntimeError("The core returned no actual trace for an intervention decode.")
    arrays: dict[str, np.ndarray] = {}
    bank_receipts = {}
    query_count = int(query.coordinates.shape[1])
    for bank_name, banks in (("module", organization.module_banks), ("environment", organization.environment_banks)):
        bank = banks[0]
        trace = output.trace
        source = trace[f"{bank_name}_source_read_attention"][0]
        source_queries = source.get("edge_query_rows")
        source_rows = source.get("edge_source_rows")
        if source_queries is None:
            source_queries = torch.empty((0,), dtype=torch.long, device=query.coordinates.device)
        if source_rows is None:
            source_rows = torch.empty((0,), dtype=torch.long, device=query.coordinates.device)
        if source_queries.numel() != source_rows.numel():
            raise RuntimeError("Executed source-read trace has misaligned query/source rows.")
        group_access = trace[f"{bank_name}_group_access"][0]
        group_query, group_index = torch.nonzero(group_access > 0, as_tuple=True)
        routes = trace[f"{bank_name}_query_gates"][0]
        local_ids = tuple(bank.forest.original_ids) if bank.forest is not None else ()
        prefix = f"{array_prefix}__{bank_name}"
        query_rows_subset = _numpy(source_queries).astype(np.int64, copy=False)
        query_rows_np = original_query_rows[query_rows_subset]
        source_rows_np = _numpy(source_rows).astype(np.int64, copy=False)
        arrays[f"{prefix}_source_query_rows"] = query_rows_np
        arrays[f"{prefix}_source_local_rows"] = source_rows_np
        group_query_rows = original_query_rows[_numpy(group_query).astype(np.int64, copy=False)]
        arrays[f"{prefix}_group_access_query_rows"] = group_query_rows
        arrays[f"{prefix}_group_access_group_rows"] = _numpy(group_index).astype(np.int64, copy=False)
        arrays[f"{prefix}_group_access_values"] = _numpy(group_access[group_query, group_index]).astype(np.float32, copy=False)
        route_query_subset = _numpy(routes["query_indices"]).astype(np.int64, copy=False)
        arrays[f"{prefix}_group_record_route_query_rows"] = original_query_rows[route_query_subset]
        route_local = _numpy(routes["record_local_indices"]).astype(np.int64, copy=False)
        arrays[f"{prefix}_group_record_route_local_rows"] = route_local
        arrays[f"{prefix}_group_record_route_gates"] = _numpy(routes["gate_values"]).astype(np.float32, copy=False)
        read_weight = source.get("weights")
        arrays[f"{prefix}_source_read_head_weights"] = (
            np.empty((0, 0), dtype=np.float32) if read_weight is None
            else _numpy(read_weight).astype(np.float32, copy=False)
        )
        read_ids = [local_ids[int(row)] for row in source_rows_np]
        id_by_json = {json.dumps(_plain(value), sort_keys=True, ensure_ascii=False): _plain(value)
                      for value in read_ids}
        selected_ids = [id_by_json[key] for key in sorted(id_by_json)]
        bank_receipts[bank_name] = {
            "present_records": int(bank.present_count),
            "live_groups": int(bank.live_count),
            "source_read_edges": int(source_rows_np.size),
            "unique_source_records_read": int(np.unique(source_rows_np).size),
            "source_read_query_rows": int(np.unique(query_rows_np).size),
            "source_read_empty": bool(source_rows_np.size == 0),
            "group_access_edges": int(group_query.numel()),
            "group_access_query_rows": int(torch.unique(group_query).numel()),
            "group_record_route_edges": int(routes["query_indices"].numel()),
            "group_access_empty": bool(group_query.numel() == 0),
            "selected_source_ids": selected_ids,
            "trace_query_rows": query_count,
            "full_native_query_rows": int(full_query_count),
            "trace_coverage_fraction": query_count / max(1, int(full_query_count)),
            "decode_work_counts": {str(key): int(value) for key, value in output.work_counts.items()},
        }
    return {"banks": bank_receipts, "decode_work_counts": {str(k): int(v) for k, v in output.work_counts.items()}}, arrays


def _original_record_payload_arrays(core: Any, organization: Any) -> dict[str, np.ndarray]:
    """Retain every source slot, immutable ID, encoded key, and projected value."""
    scene = organization.scene
    arrays: dict[str, np.ndarray] = {}
    for bank_name, states, keys, ids, mask, value_layer in (
        ("module", organization.module_states[0], organization.module_keys[0], scene.module_ids[0],
         scene.module_mask[0], core.module_value),
        ("environment", organization.environment_states[0], organization.environment_keys[0],
         scene.environment_ids[0], scene.environment_mask[0], core.environment_value),
    ):
        arrays[f"{bank_name}_source_slot_ids"] = _id_strings([_numpy(ids[index]).item() for index in range(ids.shape[0])])
        arrays[f"{bank_name}_source_slot_present"] = _numpy(mask).astype(np.bool_, copy=False)
        arrays[f"{bank_name}_source_context_states_all_slots"] = _numpy(states)
        arrays[f"{bank_name}_source_keys_all_slots"] = _numpy(keys)
        with torch.no_grad():
            arrays[f"{bank_name}_source_value_slots_all_slots"] = _numpy(value_layer(states))
    return arrays


def _native_metric_row(provider: Any, predictions: Any, batch: Any, auxiliary: Any) -> dict[str, Any]:
    metrics = provider.validation_metrics(predictions, batch.targets, auxiliary)
    rows = list(metrics.get("case_rows", ()))
    case_id = str(batch.case_keys[0])
    if len(rows) != 1 or str(rows[0].get("case_id")) != case_id:
        raise ValueError("Maintained native metric provider must return one aligned row for each full native scene.")
    return dict(rows[0])


def _native_with_organization(prepared_native: Any, organization: Any) -> Any:
    if hasattr(prepared_native, "__dataclass_fields__"):
        return replace(prepared_native, organization=organization)
    result = copy.copy(prepared_native)
    result.organization = organization
    return result


def _prediction_with_native(predictions: Any, native_prepared: Any, native_main: Any, work: Any) -> Any:
    if hasattr(predictions, "__dataclass_fields__"):
        values = {"native_prepared": native_prepared, "native_main": native_main}
        if "work" in getattr(predictions, "__dataclass_fields__", {}):
            values["work"] = work
        return replace(predictions, **values)
    result = copy.copy(predictions)
    result.native_prepared = native_prepared
    result.native_main = native_main
    result.work = work
    return result


def _candidate_nodes(bank: Any, *, limit: int) -> tuple[list[int], np.ndarray]:
    if bank.forest is None:
        return [], np.empty((0,), dtype=np.int64)
    depths = _depths(bank.forest)
    internal = [int(index) for index in bank.forest.internal_indices]
    if not internal:
        return [], depths
    ordered = sorted(internal, key=lambda index: (
        int(depths[index]), json.dumps(_plain(bank.forest.nodes[index].ancestry_ids), sort_keys=True), index
    ))
    selected = [ordered[0]]
    if limit > 1 and len(ordered) > 1 and ordered[-1] != ordered[0]:
        selected.append(ordered[-1])
    return selected[:limit], depths


def evaluate_organization(
    checkpoint: str | Path,
    device: str | torch.device,
    output: str | Path,
    max_cases: int = 24,
) -> dict[str, Any]:
    """Evaluate executed contextual organization interventions on a fixed TRAIN panel.

    The complete selected-TRAIN panel is retained in the receipt for every M
    stratum. Full native predictions, traces, and at most two legal internal
    node interventions per bank are detailed for up to four input-selected
    scenes. The baseline forest and encoded records are reused for both sides
    of each parent-stop/descendant comparison; no optimization is performed.
    """
    if type(max_cases) is not int or not 1 <= max_cases <= _ORGANIZATION_EVALUATION_MAX_CASES:
        raise ValueError(f"max_cases must be an integer from 1 to {_ORGANIZATION_EVALUATION_MAX_CASES}.")
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to run organization evaluation.")
    output_path = Path(output).expanduser().resolve(strict=False)
    output_parent = output_path.parent.resolve(strict=True)
    workspace_root = (data2 / "wanglz/ModularDT").resolve(strict=True)
    if not output_parent.is_relative_to(workspace_root) or output_parent.stat().st_dev != data2.stat().st_dev:
        raise ValueError("Organization evaluation output must be beneath the mounted /data2 workspace.")
    if output_path.exists():
        raise FileExistsError(output_path)
    preflight_free_bytes = int(shutil.disk_usage(output_parent).free)
    from .contextual_frontier_evaluation import load_contextual_checkpoint

    resolved_device = torch.device(device)
    model, provider, payload = load_contextual_checkpoint(checkpoint, resolved_device)
    identity = payload.get("experiment_identity", {})
    provider_identity = identity.get("provider_identity", {}) if isinstance(identity, Mapping) else {}
    recipe = provider_identity.get("recipe", {}) if isinstance(provider_identity, Mapping) else {}
    recipe_storage = recipe.get("storage", {}) if isinstance(recipe, Mapping) else {}
    minimum_headroom = int(recipe_storage.get("minimum_free_headroom_bytes", 1024**3))
    if preflight_free_bytes - _ORGANIZATION_EVALUATION_MAX_BYTES < minimum_headroom:
        raise OSError("Organization evaluation output budget would breach /data2 minimum free-space headroom.")
    checkpoint_epoch = int(payload.get("epoch", payload.get("current_epoch", 0)))
    arm = str(payload.get("arm", getattr(model, "architecture", "")))
    if arm not in {"J-D", "J-H"} or getattr(model, "architecture", arm) != arm:
        raise ValueError("Maintained organization evaluation requires a matching J-D or J-H checkpoint.")
    if not hasattr(model, "core") or not callable(getattr(model, "apply_native_with_work_counts", None)):
        raise TypeError("Loaded checkpoint must expose the maintained joint native adapter and contextual core.")
    core = model.core
    if checkpoint_epoch < 1:
        raise ValueError("Maintained contextual checkpoint must identify a positive absolute epoch.")
    panel = fixed_train_panel(provider, max_cases=max_cases)
    detail_ids, detail_counts = _detail_train_cases(panel, limit=_ORGANIZATION_EVALUATION_DETAIL_SCENES)
    old_clock = (int(core.organization_epoch), int(core.organization_update))
    old_modes = {module: bool(module.training) for module in model.modules()}
    arrays: dict[str, np.ndarray] = {}
    detailed_scenes = []
    native_batch_ids = []
    try:
        model.eval()
        core.set_organization_clock(checkpoint_epoch, 0)
        batches = provider.native_evaluation_batches(
            split="train", case_keys=detail_ids, microbatch_size=1
        )
        for scene_index, batch in enumerate(batches):
            case_keys = tuple(map(str, batch.case_keys))
            if len(case_keys) != 1 or case_keys[0] != detail_ids[scene_index]:
                raise ValueError("Full native TRAIN provider changed the fixed input-selected scene order.")
            case_id = case_keys[0]
            if tuple(map(str, batch.targets.case_ids)) != (case_id,):
                raise ValueError("Full native TRAIN receivers and targets have different source IDs.")
            with torch.no_grad():
                predictions, auxiliary = provider.predict_native(
                    model, batch.scene_inputs, batch.receivers, arm, "joint", epoch=checkpoint_epoch,
                )
            native = predictions.native_prepared
            organization = native.organization
            if organization.scene.batch_size != 1 or _plain(organization.scene_key) != _plain(native.scene_key):
                raise ValueError("Full native prepared context must be the exact one-case receiver scene.")
            if _plain(native.query_batch.scene_key) != _plain(native.scene_key):
                raise ValueError("Full native query union does not share the selected TRAIN source ID.")
            baseline_row = _native_metric_row(provider, predictions, batch, auxiliary)
            trace_query, trace_query_rows = _bounded_query_subset(
                native.query_batch, limit=_ORGANIZATION_TRACE_QUERY_LIMIT
            )
            trace_native = _native_with_query(native, trace_query)
            trace_receipt, trace_arrays = inspect_prepared(model, trace_native)
            _remap_inspection_query_rows(trace_arrays, trace_query_rows)
            trace_arrays["trace_query_original_rows"] = trace_query_rows
            trace_arrays["trace_query_original_coordinates"] = _numpy(
                native.query_batch.coordinates[0, trace_query_rows]
            ).astype(np.float32, copy=False)
            if native.query_batch.query_ids is None:
                trace_query_ids = trace_query_rows
            else:
                trace_query_ids = _numpy(native.query_batch.query_ids[0, trace_query_rows])
            trace_arrays["trace_query_original_ids"] = _id_strings(
                [_plain(value) for value in np.asarray(trace_query_ids).reshape(-1)]
            )
            trace_receipt["trace_scope"] = "input-order-spread subset of at most 256 valid native-union query rows; full native output and work counters remain complete"
            trace_receipt["trace_query_count"] = int(trace_query_rows.size)
            trace_receipt["full_native_query_rows"] = int(native.query_batch.coordinates.shape[1])
            trace_receipt["trace_query_coverage_fraction"] = (
                int(trace_query_rows.size) / max(1, int(native.query_batch.coordinates.shape[1]))
            )
            # Edge-wise repeated string IDs are recoverable by joining local rows
            # to the retained original source-slot ID arrays below.
            for key in ("module_query_source_record_id", "environment_query_source_record_id"):
                trace_arrays.pop(key, None)
            prefix = f"scene_{scene_index:02d}"
            arrays.update({f"{prefix}__{key}": value for key, value in trace_arrays.items()})
            source_arrays = _original_record_payload_arrays(core, organization)
            arrays.update({f"{prefix}__{key}": value for key, value in source_arrays.items()})
            scene_interventions = []
            bank_details = {}
            for bank_name, banks in (("module", organization.module_banks), ("environment", organization.environment_banks)):
                bank = banks[0]
                chosen_nodes, candidate_depths = _candidate_nodes(bank, limit=2)
                candidate_total = 0 if bank.forest is None else len(bank.forest.internal_indices)
                candidate_rows = []
                for node_id in chosen_nodes:
                    node = bank.forest.nodes[node_id]
                    frontier = bank.frontier
                    policy = core.frontier_policies[bank_name]
                    temperature = float(policy.temperature_for_epoch(checkpoint_epoch))
                    stop_probability = float(torch.sigmoid(frontier.stop_logits[node_id] / temperature).detach().cpu())
                    candidate = {
                        "node_id": int(node_id),
                        "identity": _plain(node.identity),
                        "ancestry_ids": [_plain(value) for value in node.ancestry_ids],
                        "ancestry_size": len(node.ancestry_ids),
                        "depth": int(candidate_depths[node_id]),
                        "stop_probability": stop_probability,
                        "baseline_frontier_selected": int(node_id) in set(map(int, frontier.node_indices)),
                    }
                    try:
                        with torch.no_grad():
                            parent_organization = core.intervene_frontier(
                                organization, bank_name, 0, int(node_id), stop=True
                            )
                            descendant_organization = core.intervene_frontier(
                                organization, bank_name, 0, int(node_id), stop=False
                            )
                        parent_bank = (parent_organization.module_banks if bank_name == "module"
                                       else parent_organization.environment_banks)[0]
                        descendant_bank = (descendant_organization.module_banks if bank_name == "module"
                                           else descendant_organization.environment_banks)[0]
                        source_payload_unchanged = (
                            torch.equal(parent_bank.context_states, bank.context_states)
                            and torch.equal(descendant_bank.context_states, bank.context_states)
                            and torch.equal(parent_bank.keys, bank.keys)
                            and torch.equal(descendant_bank.keys, bank.keys)
                            and torch.equal(
                                getattr(core, f"{bank_name}_value")(parent_bank.context_states),
                                getattr(core, f"{bank_name}_value")(bank.context_states),
                            )
                            and torch.equal(
                                getattr(core, f"{bank_name}_value")(descendant_bank.context_states),
                                getattr(core, f"{bank_name}_value")(bank.context_states),
                            )
                            and torch.equal(parent_bank.scene_rows, bank.scene_rows)
                            and torch.equal(descendant_bank.scene_rows, bank.scene_rows)
                            and tuple(parent_bank.forest.original_ids) == tuple(bank.forest.original_ids)
                            and tuple(descendant_bank.forest.original_ids) == tuple(bank.forest.original_ids)
                        )
                        if not source_payload_unchanged:
                            raise RuntimeError("Frontier intervention changed original encoded source records or slots.")
                        parent_native = _native_with_organization(native, parent_organization)
                        descendant_native = _native_with_organization(native, descendant_organization)
                        if parent_native.query_batch is not native.query_batch or descendant_native.query_batch is not native.query_batch:
                            raise RuntimeError("Intervention changed the complete native query/receiver union.")
                        with torch.no_grad():
                            parent_main, parent_work = model.apply_native_with_work_counts(
                                parent_native, batch.receivers.heat
                            )
                            descendant_main, descendant_work = model.apply_native_with_work_counts(
                                descendant_native, batch.receivers.heat
                            )
                        parent_prediction = _prediction_with_native(
                            predictions, parent_native, parent_main, parent_work
                        )
                        descendant_prediction = _prediction_with_native(
                            predictions, descendant_native, descendant_main, descendant_work
                        )
                        parent_auxiliary = {"work": parent_work}
                        descendant_auxiliary = {"work": descendant_work}
                        parent_metric = _native_metric_row(provider, parent_prediction, batch, parent_auxiliary)
                        descendant_metric = _native_metric_row(
                            provider, descendant_prediction, batch, descendant_auxiliary
                        )
                        per_field = _native_output_changes(
                            predictions,
                            parent_prediction,
                            descendant_prediction,
                            batch,
                            {"baseline": baseline_row, "parent_stop": parent_metric,
                             "descendant": descendant_metric},
                        )
                        parent_trace, parent_trace_arrays = _compact_actual_support(
                            core, parent_organization, trace_query,
                            array_prefix=f"{prefix}__{bank_name}__node_{node_id}__parent_stop",
                            original_query_rows=trace_query_rows,
                            full_query_count=int(native.query_batch.coordinates.shape[1]),
                        )
                        descendant_trace, descendant_trace_arrays = _compact_actual_support(
                            core, descendant_organization, trace_query,
                            array_prefix=f"{prefix}__{bank_name}__node_{node_id}__descendant",
                            original_query_rows=trace_query_rows,
                            full_query_count=int(native.query_batch.coordinates.shape[1]),
                        )
                        arrays.update(parent_trace_arrays)
                        arrays.update(descendant_trace_arrays)
                        candidate.update({
                            "status": "completed",
                            "parent_stop_K": int(parent_bank.live_count),
                            "descendant_K": int(descendant_bank.live_count),
                            "parent_stop_frontier_ids": [[_plain(value) for value in row]
                                                          for row in parent_bank.frontier.ancestry_ids],
                            "descendant_frontier_ids": [[_plain(value) for value in row]
                                                         for row in descendant_bank.frontier.ancestry_ids],
                            "original_source_ids_and_encoded_slots_preserved": True,
                            "complete_native_query_union_reused": True,
                            "parent_stop_native_metrics": parent_metric,
                            "descendant_native_metrics": descendant_metric,
                            "per_field_output_changes": per_field,
                            "parent_stop_actual_support": parent_trace,
                            "descendant_actual_support": descendant_trace,
                            "native_work_counts": {
                                "parent_stop": {str(key): int(value) for key, value in parent_work.items()},
                                "descendant": {str(key): int(value) for key, value in descendant_work.items()},
                            },
                        })
                    except Exception as exc:
                        candidate["status"] = "failed"
                        candidate["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                    candidate_rows.append(candidate)
                depth_histogram = {}
                if bank.forest is not None:
                    for node_id in bank.forest.internal_indices:
                        depth = int(candidate_depths[int(node_id)])
                        depth_histogram[str(depth)] = depth_histogram.get(str(depth), 0) + 1
                bank_details[bank_name] = {
                    "present_records_N": int(bank.present_count),
                    "baseline_live_K": int(bank.live_count),
                    "expected_K": None if bank.frontier is None else float(bank.frontier.expected_K.detach().cpu()),
                    "expected_K_fraction": (
                        None if bank.frontier is None or not bank.present_count else
                        float(bank.frontier.expected_K.detach().cpu()) / int(bank.present_count)
                    ),
                    "frontier_log_probability": (
                        None if bank.frontier is None else float(bank.frontier.log_prob.detach().cpu())
                    ),
                    "baseline_frontier_node_ids": [] if bank.frontier is None else list(map(int, bank.frontier.node_indices)),
                    "candidate_internal_node_count": candidate_total,
                    "candidate_depth_histogram": depth_histogram,
                    "selected_candidate_count": len(chosen_nodes),
                    "selected_candidate_coverage": (len(chosen_nodes) / candidate_total if candidate_total else None),
                    "selected_candidates": candidate_rows,
                }
                scene_interventions.extend({"bank": bank_name, **row} for row in candidate_rows)
            prepared_work = {str(key): int(value) for key, value in organization.work_counts.items()}
            native_work = getattr(predictions, "work", None) or {}
            detailed_scenes.append({
                "case_id": case_id,
                "module_count": int(detail_counts[scene_index]),
                "scene_key": _plain(organization.scene_key),
                "native_query_rows": int(native.query_batch.coordinates.shape[1]),
                "trace_query_selection": {
                    "limit": _ORGANIZATION_TRACE_QUERY_LIMIT,
                    "selected_rows": int(trace_query_rows.size),
                    "coverage_fraction": trace_receipt["trace_query_coverage_fraction"],
                    "selection_rule": "valid receiver rows spread evenly in immutable native-union order; input-only",
                    "query_support_arrays_indexed_by_original_native_union_row": True,
                },
                "native_union_rows": int(native.union.unique_counts.sum().item()),
                "native_prepare_work_counts": prepared_work,
                "native_prediction_work_counts": {str(key): int(value) for key, value in native_work.items()},
                "instrumented_trace_decode_replay": trace_receipt["decode_work_counts"],
                "baseline_native_metrics": baseline_row,
                "original_source_payloads": {
                    bank_name: {
                        "all_source_slots": int(source_arrays[f"{bank_name}_source_slot_ids"].size),
                        "present_source_records": int(source_arrays[f"{bank_name}_source_slot_present"].sum()),
                        "record_ids_and_context_key_value_arrays_retained": True,
                        "value_slots_shape": list(source_arrays[f"{bank_name}_source_value_slots_all_slots"].shape),
                    }
                    for bank_name in ("module", "environment")
                },
                "baseline_trace": trace_receipt,
                "banks": bank_details,
                "interventions": scene_interventions,
            })
            native_batch_ids.append(case_id)
        if tuple(native_batch_ids) != tuple(detail_ids):
            raise ValueError("Full native TRAIN evaluator did not return every selected scene exactly once.")
    finally:
        core.set_organization_clock(*old_clock)
        for module, was_training in old_modes.items():
            module.training = was_training

    detailed_strata = sorted({int(value) for value in detail_counts})
    panel_strata = sorted(int(value) for value in panel["by_module_count"])
    failure_count = sum(
        row.get("status") != "completed"
        for scene in detailed_scenes for row in scene["interventions"]
    )
    receipt = {
        "schema": "contextual_frontier_organization_native_evaluation_v1",
        "status": "completed_with_failures" if failure_count else "completed",
        "intervention_failure_count": int(failure_count),
        "checkpoint": str(payload.get("checkpoint_path", Path(checkpoint).expanduser())),
        "checkpoint_arm": arm,
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(resolved_device),
        "train_panel": panel,
        "detail_case_ids": detail_ids,
        "detail_scene_limit": _ORGANIZATION_EVALUATION_DETAIL_SCENES,
        "detail_module_count_strata": detailed_strata,
        "panel_module_count_strata_not_detailed": [value for value in panel_strata if value not in detailed_strata],
        "selection_boundary": "The fixed TRAIN panel is input-only and stratum-covering; full native work is limited to up to four representatives selected from that panel by module count. No uncovered detailed stratum is implied to have been evaluated.",
        "scenes": detailed_scenes,
        "intervention_contract": {
            "maximum_internal_nodes_per_bank_per_scene": 2,
            "comparison": "same encoded scene and forest; force parent node stop versus force its descendant frontier; rerun the complete native adapter for identical receiver roles",
            "intervention_node_selection": "shallowest and deepest distinct legal internal nodes, deterministic ties by immutable ancestry IDs and node index; input-only",
            "candidate_probabilities_are_descriptive_only": True,
            "K_is_not_an_accuracy_or_quality_measure": True,
            "source_record_ids_keys_and_projected_value_slots_are_retained": True,
            "causal_claim": False,
            "memory_reduction_claim": False,
            "native_role_changes": "Per-field output RMS is measured for parent-stop versus baseline, descendant versus baseline, and parent-stop versus descendant on shared complete native physical masks, with each normalized by baseline native RMSE.",
        },
        "output_budget": {
            "maximum_payload_bytes": _ORGANIZATION_EVALUATION_MAX_BYTES,
            "minimum_free_headroom_bytes": minimum_headroom,
            "pre_evaluation_free_bytes": preflight_free_bytes,
            "estimated_peak_new_disk_bytes": _ORGANIZATION_EVALUATION_MAX_BYTES,
            "filesystem": str(data2),
        },
        "interpretation": "Measured full-native sensitivity to executed contextual frontier interventions on this bounded fixed TRAIN panel; support changes are observed model information flow, not physical causality or standalone success.",
    }
    json.dumps(receipt, sort_keys=True, allow_nan=False)
    written = write_evidence_bundle(
        output_path,
        receipt,
        arrays,
        max_output_bytes=_ORGANIZATION_EVALUATION_MAX_BYTES,
        minimum_free_headroom_bytes=minimum_headroom,
    )
    return {**receipt, "write": written}
