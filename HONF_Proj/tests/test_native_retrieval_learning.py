from __future__ import annotations

from dataclasses import replace

import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.staged_attention_v2 import SpatialRouteAttentionV2
from honf_forward_core.native_retrieval.contracts import NativeQueryRetrievalPlan
from honf_forward_core.native_retrieval.hypergraph import build_native_hypergraph
from honf_forward_core.native_retrieval.planning import build_query_retrieval_plan
from honf_forward_core.native_retrieval.reader import NativeSelectedReader
from honf_forward_core.native_retrieval.route_capture import NativeRouteSnapshot


def _empty_me_snapshot(module_ids: torch.Tensor, *, owner: int, parameter_versions) -> NativeRouteSnapshot:
    batch, modules = module_ids.shape
    empty_ids = torch.empty(batch, 0, dtype=torch.long)
    empty_scores = torch.empty(batch, 1, modules, 0, dtype=torch.float64)
    return NativeRouteSnapshot(
        route="me",
        query_scaled=torch.empty(batch, 1, modules, 4, dtype=torch.float64),
        key_normalized=torch.empty(batch, 1, 0, 4, dtype=torch.float64),
        geometry_bias=empty_scores,
        raw_scores=empty_scores,
        allowed=torch.zeros(batch, modules, 0, dtype=torch.bool),
        destination_mask=torch.ones(batch, modules, dtype=torch.bool),
        donor_mask=torch.zeros(batch, 0, dtype=torch.bool),
        destination_ids=module_ids,
        donor_ids=empty_ids,
        donor_measure=torch.empty(batch, 0, dtype=torch.float64),
        measure_convention="unnormalized_native_donor_measure_in_attention_softmax",
        stage_identity="cycle0/me",
        owner_token=owner,
        parameter_versions=parameter_versions,
        buffer_versions=(),
        scene_identity=("fixture", "norm", "frame"),
    )


def test_task_gradient_flows_through_live_mm_permission_to_existing_route_parameter():
    torch.manual_seed(73)
    attention = SpatialRouteAttentionV2(width=4, heads=1, coordinate_dim=1).double().eval()
    # Make content scores exactly neutral and route geometry deterministic.
    with torch.no_grad():
        attention.query.weight.zero_()
        attention.key.weight.zero_()
        attention.geometry_score[0].weight.zero_()
        attention.geometry_score[0].bias.zero_()
        attention.geometry_score[0].weight[0, 0] = 1.0
        attention.geometry_score[2].weight.zero_()
        attention.geometry_score[2].weight[0, 0] = 12.0

    core = ContentHypergraphCoreV2(
        module_input_dim=2,
        environment_input_dim=2,
        coordinate_dim=1,
        query_input_dim=1,
        operating_input_dim=0,
        output_dim=2,
        architecture="A2-D",
        context_dim=8,
        num_heads=1,
        context_depth=1,
        num_hyperedges=0,
        thermal_response_rank=2,
        local_hidden_dim=8,
        dropout=0.0,
        init_seed=73,
    ).eval()
    scene, query = _small_scene_and_query()
    query = replace(
        query,
        coordinates=torch.tensor([[[0.5923]]]),
        features=torch.tensor([[[-0.0581]]]),
    )
    organization = core.prepare(scene)
    query_states = core._query_states(organization, query)
    reader = core.direct_reader
    # Use the unmodified A2-D scorer and native projections. The fixed receiver
    # lies closest to seed1, while raw decoder scores suppress seed2's self path.
    query_scores = NativeSelectedReader.score_queries(reader, organization, query, query_states)
    module_ids = scene.module_ids
    states = torch.randn(1, 3, 4, dtype=torch.float64)
    coordinates = torch.tensor([[[0.0], [0.5], [1.0]]], dtype=torch.float64)
    lengths = torch.ones(1, 3, 1, dtype=torch.float64)
    measures = torch.ones(1, 3, dtype=torch.float64)
    geometry_features = torch.zeros(1, 3, 3, 3, dtype=torch.float64)
    geometry_features[0, 0, 2, 0] = -1.0  # seed0 cannot reach module2
    geometry_features[0, 1, 2, 0] = -0.2  # seed1 -> module2 lies in S5 transition
    geometry_features[0, 2, 1, 0] = -1.0
    parameter_versions = tuple((id(parameter), int(parameter._version)) for parameter in attention.parameters())
    _, _, snapshot = attention(
        states,
        states,
        coordinates,
        coordinates,
        lengths,
        lengths,
        measures,
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.float64),
        remove_self=True,
        route_name="mm",
        destination_ids=module_ids,
        donor_ids=module_ids,
        geometry_features=geometry_features,
        capture_native_route_snapshot=True,
        stage_identity="cycle0/mm",
        owner_token=12,
        parameter_versions=parameter_versions,
        scene_identity=("fixture", "norm", "frame"),
    )
    graph = build_native_hypergraph(
        snapshot,
        _empty_me_snapshot(module_ids, owner=12, parameter_versions=parameter_versions),
        tau_mm=0.2,
        tau_me=0.2,
    )
    plan = build_query_retrieval_plan(query_scores.raw_logits, graph, tau_q=0.3)
    assert plan.module_permission[0, :, 0, 2].min() > 0
    assert plan.module_permission[0, :, 0, 2].max() < 1
    assert plan.module_permission[0, :, 0, 2].grad_fn is not None
    # The task loss actually reads selected native values and evaluates a
    # signed Thermal response at nonzero controls. The MM route is independent
    # of those values, so this gradient can only travel through permission.
    result = NativeSelectedReader.evaluate(
        reader,
        organization,
        query,
        query_states,
        plan,
        backend="selected",
        scores=query_scores,
        measure_reference=scene.module_measure_reference,
    )
    response = result.response.apply(torch.ones(1, scene.num_modules))
    objective = result.context.square().sum() + response.square().sum()
    raw_score_gradient, geometry_gradient = torch.autograd.grad(
        objective,
        (snapshot.raw_scores, attention.geometry_score[2].weight),
    )
    assert result.counts["native_unique_nonlinear_rows"] == 3
    assert raw_score_gradient.abs().sum() > 0
    assert geometry_gradient.abs().sum() > 0


def _small_scene_and_query():
    scene = SceneBatch(
        module_features=torch.tensor([[[0.2, 0.4], [0.7, -0.3], [-0.1, 0.8]]]),
        environment_features=torch.tensor([[[0.1, 0.2], [0.4, -0.2]]]),
        module_coordinates=torch.tensor([[[0.0], [0.5], [1.0]]]),
        environment_coordinates=torch.tensor([[[-0.2], [0.7]]]),
        module_mask=torch.ones(1, 3, dtype=torch.bool),
        environment_mask=torch.ones(1, 2, dtype=torch.bool),
        module_measure=torch.tensor([[0.2, 0.3, 0.5]]),
        environment_measure=torch.tensor([[0.4, 0.6]]),
        module_ids=torch.tensor([[13, 29, 47]]),
        environment_ids=torch.tensor([[101, 107]]),
        local_radius=torch.full((1, 3), 0.2),
        coordinate_origin=torch.tensor([[0.0]]),
        coordinate_scale=torch.tensor([[1.0]]),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        module_characteristic_lengths=torch.full((1, 3, 1), 0.1),
        environment_characteristic_lengths=torch.full((1, 2, 1), 0.1),
        scene_key="selected-value-scene",
        normalizer_key="train-only-transform",
        coordinate_frame_key="native-1d-frame",
    )
    query = QueryBatch(
        coordinates=torch.tensor([[[0.2]]]),
        features=torch.tensor([[[0.3]]]),
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        query_ids=torch.tensor([[9001]]),
        scene_key="selected-value-scene",
        coordinate_frame_key="native-1d-frame",
        receiver_key="one-query",
    )
    return scene, query


def test_zero_permission_native_rows_have_no_direct_value_gradient():
    torch.manual_seed(83)
    core = ContentHypergraphCoreV2(
        module_input_dim=2,
        environment_input_dim=2,
        coordinate_dim=1,
        query_input_dim=1,
        operating_input_dim=0,
        output_dim=2,
        architecture="A2-D",
        context_dim=8,
        num_heads=2,
        context_depth=1,
        num_hyperedges=0,
        thermal_response_rank=2,
        local_hidden_dim=8,
        dropout=0.0,
        init_seed=83,
    ).eval()
    scene, query = _small_scene_and_query()
    organization = core.prepare(scene)
    query_states = core._query_states(organization, query)
    scores = NativeSelectedReader.score_queries(core.direct_reader, organization, query, query_states)
    permission = torch.zeros(1, core.num_heads, 1, scene.num_modules)
    permission[..., 0] = 1.0
    support = permission.amax(dim=1) > 0
    pair_indices = torch.nonzero(support, as_tuple=False)
    plan = NativeQueryRetrievalPlan(
        seed_permission=permission,
        module_permission=permission,
        module_support=support,
        unique_module_indices=pair_indices,
        module_ids=scene.module_ids,
        seed_ids=scene.module_ids,
        query_ids=query.query_ids,
        query_mask=query.query_mask,
        eligible_module_rows=scene.module_mask[:, None, :] & query.query_mask[:, :, None],
    )
    result = NativeSelectedReader.evaluate(
        core.direct_reader,
        organization,
        query,
        query_states,
        plan,
        backend="selected",
        scores=scores,
        return_messages=True,
        measure_reference=scene.module_measure_reference,
    )
    response = result.response.apply(torch.ones(1, scene.num_modules))
    objective = result.context.sum() + response.sum()
    value_gradient = torch.autograd.grad(objective, organization.direct_source_values)[0]
    assert value_gradient[:, 0].abs().sum() > 0
    assert torch.count_nonzero(value_gradient[:, 1:]) == 0
    assert result.counts["native_unique_nonlinear_rows"] == 1
    assert result.trace["unique_module_indices"].shape == (1, 3)
