from __future__ import annotations

import numpy as np
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCore, QueryBatch, SceneBatch
from tools.content_hypergraph_inspect import (
    _condition_trace_arrays,
    local_branch_zero_hook,
    rank_thermal_edge,
    rank_wind_edge,
    removed_edge_gate,
    select_thermal_probes,
    select_wind_probes,
    slice_query,
)


def _scene() -> SceneBatch:
    return SceneBatch(
        module_features=torch.tensor([[[0.2, 0.4], [0.7, -0.3]]]),
        environment_features=torch.tensor([[[0.1, 0.2, 0.3], [0.4, -0.2, 0.5], [-0.5, 0.7, 0.1]]]),
        module_coordinates=torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
        environment_coordinates=torch.tensor([[[-0.1, 0.3], [0.5, 0.7], [1.1, 0.2]]]),
        module_mask=torch.ones((1, 2), dtype=torch.bool),
        environment_mask=torch.ones((1, 3), dtype=torch.bool),
        module_measure=torch.tensor([[0.7, 1.1]]),
        environment_measure=torch.tensor([[0.2, 0.4, 0.3]]),
        module_ids=torch.tensor([[13, 29]]),
        environment_ids=torch.tensor([[101, 107, 131]]),
        local_radius=torch.full((1, 2), 0.8),
        operating_features=torch.tensor([[0.3, -0.2]]),
        module_measure_reference=torch.tensor([1.0]),
        environment_measure_reference=torch.tensor([0.3]),
        scene_key="inspection-scene",
        normalizer_key="inspection-normalizer",
        coordinate_frame_key="inspection-frame",
    )


def _query() -> QueryBatch:
    return QueryBatch(
        coordinates=torch.tensor([[[0.1, 0.0], [0.5, 0.08], [1.1, 0.1]]]),
        features=torch.tensor([[[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1]]]),
        query_ids=torch.tensor([[3, 5, 8]]),
        scene_key="inspection-scene",
        coordinate_frame_key="inspection-frame",
    )


def _core(thermal_rank: int) -> ContentHypergraphCore:
    return ContentHypergraphCore(
        module_input_dim=2,
        environment_input_dim=3,
        coordinate_dim=2,
        query_input_dim=2,
        operating_input_dim=2,
        output_dim=3,
        architecture="A-H6",
        context_dim=16,
        num_heads=4,
        context_depth=1,
        num_hyperedges=6,
        thermal_response_rank=thermal_rank,
        local_hidden_dim=8,
        init_seed=42,
    ).eval()


def test_thermal_edge_rank_uses_signed_field_contributions_not_attention_mass():
    contributions = np.zeros((3, 6, 2), dtype=np.float64)
    contributions[:, 0, :] = 0.1
    contributions[:, 2, :] = -1.5
    selected, scores = rank_thermal_edge(contributions)

    assert selected == 2
    assert scores[2] > scores[0]


def test_wind_edge_rank_uses_weighted_messages_not_receiver_occupancy():
    access = np.asarray([[0.95, 0.05, 0.0], [0.95, 0.05, 0.0]])
    messages = np.zeros((2, 3, 4), dtype=np.float64)
    messages[:, 0] = 0.01
    messages[:, 1] = 8.0
    selected, scores = rank_wind_edge(access, messages)

    assert selected == 1
    assert scores[1] > scores[0]


def test_local_zero_keeps_pair_work_and_edge_access_but_removes_exact_local_heat():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    query = _query()
    controls = torch.tensor([[0.7, -0.4]])
    full = core.decode(organization, query, controls, return_trace=True)

    with local_branch_zero_hook(core):
        local_zero = core.decode(organization, query, controls, return_trace=True)

    trace = full.trace
    pair_batch = trace["local_pair_batch_indices"]
    pair_query = trace["local_pair_query_indices"]
    pair_module = trace["local_pair_module_indices"]
    local_temperature = trace["local_pair_support_weights"] * trace["local_temperature_coefficients"] * controls[
        pair_batch, pair_module
    ]
    expected = torch.zeros_like(full.thermal_temperature)
    expected.index_put_((pair_batch, pair_query), local_temperature, accumulate=True)

    assert full.work_counts["local_rows"] > 0
    assert local_zero.work_counts["local_rows"] == full.work_counts["local_rows"]
    torch.testing.assert_close(
        local_zero.trace["local_pair_query_indices"], full.trace["local_pair_query_indices"]
    )
    torch.testing.assert_close(
        local_zero.trace["local_pair_module_indices"], full.trace["local_pair_module_indices"]
    )
    torch.testing.assert_close(full.trace["edge_used_access"], local_zero.trace["edge_used_access"])
    torch.testing.assert_close(full.thermal_temperature - local_zero.thermal_temperature, expected)
    assert not torch.allclose(full.nonlinear_field, local_zero.nonlinear_field)


def test_removed_edge_keeps_other_access_weights_and_thermal_delta_is_factor_exact():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    query = _query()
    controls = torch.tensor([[0.7, -0.4]])
    full = core.decode(organization, query, controls, return_trace=True)
    edge_index, _ = rank_thermal_edge(full.trace["edge_temperature_contributions"][0].sum(dim=-1)[:, :, None])
    gates = removed_edge_gate(1, organization.num_hyperedges, edge_index, device="cpu", dtype=torch.float32)
    removed = core.decode(organization, query, controls, edge_gates=gates, return_trace=True)

    expected_access = full.trace["edge_base_access"].clone()
    expected_access[:, :, edge_index] = 0.0
    expected_delta = full.trace["edge_temperature_contributions"][:, :, edge_index].sum(dim=-1)
    torch.testing.assert_close(removed.trace["edge_used_access"], expected_access)
    torch.testing.assert_close(full.thermal_temperature - removed.thermal_temperature, expected_delta)


def test_wind_edge_gate_recomputes_nonlinear_field_with_unrenormalized_access():
    core = _core(thermal_rank=0)
    organization = core.prepare(_scene())
    query = _query()
    full = core.decode(organization, query, return_trace=True)
    edge_index, _ = rank_wind_edge(
        full.trace["edge_used_access"][0],
        full.trace["edge_messages"][0],
    )
    gates = removed_edge_gate(1, organization.num_hyperedges, edge_index, device="cpu", dtype=torch.float32)
    removed = core.decode(organization, query, edge_gates=gates, return_trace=True)
    expected_access = full.trace["edge_base_access"].clone()
    expected_access[:, :, edge_index] = 0.0

    torch.testing.assert_close(removed.trace["edge_used_access"], expected_access)
    assert removed.work_counts["edge_rows"] == full.work_counts["edge_rows"]
    assert not torch.allclose(removed.nonlinear_field, full.nonlinear_field)


def test_query_subset_remaps_packed_pairs_and_preserves_source_kernel_rows():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    original = QueryBatch(
        coordinates=torch.tensor([[[0.1, 0.0], [0.5, 0.08], [1.1, 0.1]]]),
        features=torch.tensor([[[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1]]]),
        query_ids=torch.tensor([[3, 5, 8]]),
        local_batch_indices=torch.tensor([0, 0, 0]),
        local_query_indices=torch.tensor([0, 1, 2]),
        local_module_indices=torch.tensor([0, 0, 1]),
        local_weights=torch.tensor([0.7, 0.4, 0.8]),
        scene_key="inspection-scene",
        coordinate_frame_key="inspection-frame",
    )
    full_kernels = core.export_source_kernels(organization, original)
    subset = slice_query(original, [2, 0])
    subset_kernels = core.export_source_kernels(organization, subset)

    assert subset.query_ids.tolist() == [[8, 3]]
    assert subset.local_query_indices.tolist() == [1, 0]
    torch.testing.assert_close(subset_kernels, full_kernels[:, [2, 0]])


def test_trace_export_keeps_edge_gate_axis_and_slices_per_query_rows():
    core = _core(thermal_rank=2)
    organization = core.prepare(_scene())
    output = core.decode(organization, _query(), torch.tensor([[0.7, -0.4]]), return_trace=True)
    arrays = {}
    _condition_trace_arrays(arrays, "fixture", output, np.asarray([0, 2]))

    assert arrays["fixture__edge_gates"].shape == (1, 6)
    assert arrays["fixture__edge_access"].shape == (1, 2, 6)
    assert arrays["fixture__edge_messages"].shape == (1, 2, 6, 8)


def test_probe_selection_uses_only_native_geometry_and_role_membership():
    thermal_points = np.asarray([[0.2, 0.0], [0.8, 0.0], [3.0, 0.0], [-0.2, 0.0]])
    thermal = select_thermal_probes(
        thermal_points,
        np.arange(4),
        np.ones(4, dtype=bool),
        np.asarray([[0.0, 0.0], [0.7, 0.0]]),
        np.asarray([0.2, 0.2]),
        np.asarray([0.9, 0.9]),
    )
    assert thermal == {"near_body": 0, "downstream_overlap": 1, "background": 2}

    wind_points = np.asarray(
        [[0.0, 0.49, 0.875], [1.0, 0.0, 0.875], [1.1, 0.0, 0.875], [20.0, 10.0, 2.0]]
    )
    wind = select_wind_probes(
        wind_points,
        np.asarray([10, 20, 21, 99]),
        np.asarray([3, 2, 2, 4]),
        np.asarray([[0.0, 0.0, 0.875], [1.0, 0.0, 0.875]]),
        np.asarray([0.5, 0.5]),
    )
    assert wind == {"near_body": 0, "downstream_overlap": 2, "background": 3}
