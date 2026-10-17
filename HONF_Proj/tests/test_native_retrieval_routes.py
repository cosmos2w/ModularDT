from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.content_hypergraph.contracts import QueryBatch
from honf_forward_core.content_hypergraph.staged_attention_v2 import StagedContextCycleV2
from honf_forward_core.native_retrieval.contracts import NativeTokenRecords, NativeTokenStore
from honf_forward_core.native_retrieval.hypergraph import (
    build_native_hypergraph,
    build_native_retrieval_hypergraph,
    density_permission,
)
from honf_forward_core.native_retrieval.planning import (
    build_query_retrieval_plan,
    max_product_union,
)
from honf_forward_core.native_retrieval.route_capture import NativeRouteSnapshot


def _snapshot(
    route: str,
    scores: torch.Tensor,
    allowed: torch.Tensor,
    destination_ids: torch.Tensor,
    donor_ids: torch.Tensor,
    donor_measure: torch.Tensor,
    *,
    stage: str | None = None,
    destination_mask: torch.Tensor | None = None,
    donor_mask: torch.Tensor | None = None,
    scene_tensor_versions: tuple[tuple[int, int, int], ...] = (),
) -> NativeRouteSnapshot:
    batch, heads, destinations, donors = scores.shape
    destination_mask = (
        torch.ones(batch, destinations, dtype=torch.bool) if destination_mask is None else destination_mask
    )
    donor_mask = torch.ones(batch, donors, dtype=torch.bool) if donor_mask is None else donor_mask
    return NativeRouteSnapshot(
        route=route,
        query_scaled=scores.new_zeros(batch, heads, destinations, 1),
        key_normalized=scores.new_zeros(batch, heads, donors, 1),
        geometry_bias=scores,
        raw_scores=scores,
        allowed=allowed,
        destination_mask=destination_mask,
        donor_mask=donor_mask,
        destination_ids=destination_ids,
        donor_ids=donor_ids,
        donor_measure=donor_measure,
        measure_convention="unnormalized_native_donor_measure_in_attention_softmax",
        stage_identity=stage or f"cycle1/{route}",
        owner_token=17,
        parameter_versions=((1, 2),),
        buffer_versions=((3, 4),),
        scene_identity=("scene-A", "train-norm-v1", "native-frame-v2"),
        scene_tensor_versions=scene_tensor_versions,
        model_config_key=(("family", "native-test"),),
    )


def _graph_fixture(*, dtype=torch.float64, modules: int = 3, environments: int = 2):
    module_ids = torch.arange(10, 10 + modules, dtype=torch.long)[None, :]
    environment_ids = torch.arange(70, 70 + environments, dtype=torch.long)[None, :]
    batch, heads = 1, 2
    mm_allowed = ~torch.eye(modules, dtype=torch.bool)[None, :, :]
    mm_allowed = mm_allowed.expand(batch, modules, modules).clone()
    mm_scores = torch.zeros(batch, heads, modules, modules, dtype=dtype)
    for seed in range(modules):
        for donor in range(modules):
            if seed != donor:
                mm_scores[:, :, seed, donor] = -2.0 * abs(seed - donor)
    me_allowed = torch.ones(batch, modules, environments, dtype=torch.bool)
    me_scores = torch.zeros(batch, heads, modules, environments, dtype=dtype)
    if environments > 1:
        me_scores[..., 1] = -2.0
    mm = _snapshot(
        "mm",
        mm_scores,
        mm_allowed,
        module_ids,
        module_ids,
        torch.full((batch, modules), 1.0 / max(1, modules), dtype=dtype),
    )
    me = _snapshot(
        "me",
        me_scores,
        me_allowed,
        module_ids,
        environment_ids,
        torch.full((batch, environments), 1.0 / max(1, environments), dtype=dtype),
    )
    return mm, me


def test_staged_capture_preserves_sdpa_and_adds_live_final_mm_me_snapshots():
    torch.manual_seed(12)
    batch, modules, environments, width = 1, 3, 2, 8
    scene = SimpleNamespace(
        module_characteristic_lengths=torch.ones(batch, modules, 2),
        environment_characteristic_lengths=torch.ones(batch, environments, 2),
        # Presence and scalar source measure are separate: the first native
        # module remains a retrieval self-member despite measure zero.
        module_measure=torch.tensor([[0.0, 0.3, 0.5]]),
        environment_measure=torch.tensor([[0.4, 0.6]]),
        module_measure_reference=torch.ones(batch),
        environment_measure_reference=torch.ones(batch),
        module_coordinates=torch.rand(batch, modules, 2),
        environment_coordinates=torch.rand(batch, environments, 2),
        module_mask=torch.ones(batch, modules, dtype=torch.bool),
        environment_mask=torch.ones(batch, environments, dtype=torch.bool),
        module_ids=torch.tensor([[4, 2, 8]]),
        environment_ids=torch.tensor([[19, 3]]),
    )
    block = StagedContextCycleV2(width, heads=2, coordinate_dim=2).eval()
    module_states = torch.randn(batch, modules, width, requires_grad=True)
    environment_states = torch.randn(batch, environments, width, requires_grad=True)
    frame_scale = torch.ones(batch, 2)
    ordinary = block(module_states, environment_states, scene, frame_scale)
    captured = block(
        module_states,
        environment_states,
        scene,
        frame_scale,
        capture_native_routes=True,
        cycle_index=1,
        owner_token=44,
        parameter_versions=((5, 6),),
        buffer_versions=((7, 8),),
        scene_identity=("s", "n", "f"),
    )
    torch.testing.assert_close(ordinary[0], captured[0])
    torch.testing.assert_close(ordinary[1], captured[1])
    mm = captured[2]["mm_native_route_snapshot"]
    me = captured[2]["me_native_route_snapshot"]
    assert mm.route == "mm" and me.route == "me"
    assert mm.stage_identity == "cycle1/mm" and me.stage_identity == "cycle1/me"
    assert mm.owner_token == 44 and mm.parameter_versions == ((5, 6),)
    assert torch.equal(mm.destination_ids, scene.module_ids)
    assert torch.equal(me.donor_ids, scene.environment_ids)
    assert mm.donor_mask[0, 0] and not mm.allowed[0, :, 0].any()
    assert torch.equal(mm.allowed[:, torch.arange(modules), torch.arange(modules)], torch.zeros(batch, modules, dtype=torch.bool))
    assert mm.raw_scores.shape == (batch, 2, modules, modules)
    assert me.raw_scores.shape == (batch, 2, modules, environments)
    assert mm.raw_scores.requires_grad and me.raw_scores.requires_grad
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    assert graph.module_valid[0, 0]
    assert graph.module_member_permissions[0, :, 0, 0].eq(1).all()
    grad = torch.autograd.grad(mm.raw_scores.sum() + me.raw_scores.sum(), tuple(block.parameters()), allow_unused=True)
    assert any(value is not None and value.abs().sum() > 0 for value in grad)


def test_hypergraph_keeps_self_only_for_retrieval_and_tau_zero_is_all_open():
    mm, me = _graph_fixture()
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    assert torch.equal(graph.module_context_allowed, mm.allowed)
    assert not graph.module_context_allowed[0, 0, 0]
    assert graph.module_member_permissions[0, :, 0, 0].eq(1).all()
    assert graph.module_retrieval_allowed[0, 0, 0]
    assert graph.environment_member_permissions[0, :, 0, 0].eq(1).all()
    assert not hasattr(graph, "edge_values")
    all_open = build_native_retrieval_hypergraph(mm, me, tau_mm=0.0, tau_me=0.0)
    assert torch.equal(
        all_open.module_member_permissions > 0,
        all_open.module_retrieval_allowed[:, None].expand_as(all_open.module_member_permissions),
    )
    assert torch.equal(
        all_open.environment_member_permissions > 0,
        me.allowed[:, None].expand_as(all_open.environment_member_permissions),
    )


def test_density_relative_membership_is_measure_invariant_but_attention_probability_is_not():
    # One destination has two exactly identical records; splitting one donor's
    # quadrature measure changes attention mass but leaves relative density.
    score_a = torch.tensor([[[[0.0, -1.2]]]], dtype=torch.float64)
    allowed_a = torch.ones(1, 1, 2, dtype=torch.bool)
    density_a, permission_a = density_permission(score_a, allowed_a, 0.2)
    score_split = torch.tensor([[[[0.0, -1.2, -1.2]]]], dtype=torch.float64)
    allowed_split = torch.ones(1, 1, 3, dtype=torch.bool)
    density_split, permission_split = density_permission(score_split, allowed_split, 0.2)
    torch.testing.assert_close(density_a[..., 1], density_split[..., 1])
    torch.testing.assert_close(density_split[..., 1], density_split[..., 2])
    torch.testing.assert_close(permission_a[..., 1], permission_split[..., 1])
    torch.testing.assert_close(permission_split[..., 1], permission_split[..., 2])
    measure_a = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    measure_split = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float64)
    attention_a = torch.softmax(score_a[0, 0, 0] + measure_a[0].log(), dim=-1)
    attention_split = torch.softmax(score_split[0, 0, 0] + measure_split[0].log(), dim=-1)
    assert attention_a[1] > attention_split[1]
    assert attention_split[1] == pytest.approx(float(attention_split[2]))


def test_empty_environment_and_zero_module_axes_are_well_defined():
    mm, me = _graph_fixture(environments=0)
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    assert graph.environment_member_permissions.shape == (1, 2, 3, 0)
    scores = torch.zeros(1, 2, 4, 3, dtype=torch.float64)
    plan = build_query_retrieval_plan(
        scores, graph, tau_q=0.2, include_environment=True,
        query_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    assert plan.environment_permission.shape == (1, 2, 4, 0)
    assert plan.unique_environment_indices.shape == (0, 3)
    assert plan.environment_value_consumer is False

    empty_mm = _snapshot(
        "mm", torch.zeros(1, 2, 0, 0), torch.zeros(1, 0, 0, dtype=torch.bool),
        torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0),
    )
    empty_me = _snapshot(
        "me", torch.zeros(1, 2, 0, 2), torch.zeros(1, 0, 2, dtype=torch.bool),
        torch.empty(1, 0, dtype=torch.long), torch.tensor([[70, 71]]), torch.tensor([[0.5, 0.5]]),
    )
    empty_graph = build_native_hypergraph(empty_mm, empty_me, tau_mm=0.0, tau_me=0.0)
    empty_plan = build_query_retrieval_plan(
        torch.empty(1, 3, 5, 0), empty_graph, tau_q=0.0,
        query_mask=torch.ones(1, 5, dtype=torch.bool), include_environment=True,
    )
    assert empty_plan.module_permission.shape == (1, 3, 5, 0)
    assert empty_plan.unique_module_indices.shape == (0, 3)


def test_zero_measure_present_module_keeps_its_seed_self_retrieval_route():
    mm, me = _graph_fixture(environments=1)
    mm.donor_measure[0, 0] = 0.0
    mm_allowed = mm.allowed.clone()
    mm_allowed[..., 0] = False
    mm_snapshot = _snapshot(
        "mm", mm.raw_scores, mm_allowed, mm.destination_ids, mm.donor_ids,
        mm.donor_measure, donor_mask=mm.donor_mask,
    )
    me_snapshot = _snapshot(
        "me", me.raw_scores, me.allowed, me.destination_ids, me.donor_ids, me.donor_measure,
    )
    graph = build_native_hypergraph(mm_snapshot, me_snapshot, tau_mm=0.0, tau_me=0.0)
    assert graph.module_valid[0, 0]  # presence is independent from scalar measure
    assert not graph.module_context_allowed[0, :, 0].any()  # zero measure is not a contextual donor
    assert graph.module_member_permissions[0, :, 0, 0].eq(1).all()  # but self remains retrievable
    plan = build_query_retrieval_plan(
        torch.zeros(1, 2, 1, 3), graph, tau_q=0.0,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    assert plan.module_permission[0, :, 0, 0].eq(1).all()
    assert plan.module_support[0, 0, 0]


def test_query_union_is_max_product_duplicate_invariant_and_optional_for_e():
    mm, me = _graph_fixture()
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    scores = torch.tensor([[[[0.0, -0.3, -4.0]], [[-0.2, 0.0, -3.0]]]], dtype=torch.float64)
    protected = torch.zeros(1, 1, 3, dtype=torch.float64)
    protected[0, 0, 2] = 0.35
    plan = build_query_retrieval_plan(
        scores, graph, tau_q=0.2, protected_module=protected,
        query_ids=torch.tensor([[900]]),
    )
    assert plan.module_permission.shape == (1, 2, 1, 3)
    assert plan.module_permission[..., 2].amin() >= 0.35
    assert plan.unique_module_indices.ndim == 2
    assert plan.unique_module_indices.shape[-1] == 3
    assert plan.environment_permission is None  # no QxE allocation in M-only mode
    assert plan.retrieved_module_pairs.item() == int(plan.module_support.sum())

    seed = torch.tensor([[[[0.3, 0.8]]]], dtype=torch.float64)
    members = torch.tensor([[[1.0, 0.2], [0.4, 1.0]]], dtype=torch.float64)
    original = max_product_union(seed, members)
    duplicated = max_product_union(
        torch.cat((seed, seed[..., 1:2]), dim=-1),
        torch.cat((members, members[:, 1:2]), dim=1),
    )
    torch.testing.assert_close(original, duplicated)


def test_all_open_query_permissions_preserve_multiple_decoder_heads():
    mm, me = _graph_fixture()
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    scores = torch.zeros(1, 3, 2, 3, dtype=torch.float64)
    query_mask = torch.tensor([[True, False]])
    plan = build_query_retrieval_plan(
        scores,
        graph,
        tau_q=0.0,
        query_mask=query_mask,
        include_environment=True,
    )
    assert plan.seed_permission.shape == scores.shape
    assert plan.module_permission.shape == (1, 3, 2, 3)
    assert plan.environment_permission.shape == (1, 3, 2, 2)
    assert torch.equal(plan.seed_permission[:, :, 0], torch.ones_like(plan.seed_permission[:, :, 0]))
    assert torch.equal(plan.seed_permission[:, :, 1], torch.zeros_like(plan.seed_permission[:, :, 1]))
    assert torch.equal(plan.module_support[:, 0], torch.ones_like(plan.module_support[:, 0]))
    assert torch.equal(plan.module_support[:, 1], torch.zeros_like(plan.module_support[:, 1]))


def test_query_and_native_order_permutations_preserve_id_aligned_support():
    mm, me = _graph_fixture()
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    scores = torch.tensor([[[[0.0, -0.6, -2.0]], [[-0.5, 0.0, -1.0]]]], dtype=torch.float64)
    plan = build_query_retrieval_plan(
        scores, graph, tau_q=0.2, query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    mm_permuted = _snapshot(
        "mm",
        mm.raw_scores[:, :, permutation][:, :, :, permutation],
        mm.allowed[:, permutation][:, :, permutation],
        mm.destination_ids[:, permutation], mm.donor_ids[:, permutation],
        mm.donor_measure[:, permutation],
    )
    me_permuted = _snapshot(
        "me",
        me.raw_scores[:, :, permutation], me.allowed[:, permutation],
        me.destination_ids[:, permutation], me.donor_ids, me.donor_measure,
    )
    graph_permuted = build_native_hypergraph(mm_permuted, me_permuted, tau_mm=0.2, tau_me=0.2)
    permuted_plan = build_query_retrieval_plan(
        scores[:, :, :, permutation], graph_permuted, tau_q=0.2,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(plan.module_permission, permuted_plan.module_permission[..., inverse])
    torch.testing.assert_close(plan.module_support, permuted_plan.module_support[..., inverse])

    query_order = torch.tensor([2, 0, 1])
    repeated = build_query_retrieval_plan(
        scores.expand(1, 2, 3, 3), graph, tau_q=0.2,
        query_mask=torch.ones(1, 3, dtype=torch.bool),
        query_ids=torch.tensor([[10, 11, 12]]),
    )
    query_permuted = build_query_retrieval_plan(
        scores.expand(1, 2, 3, 3)[:, :, query_order], graph, tau_q=0.2,
        query_mask=torch.ones(1, 3, dtype=torch.bool),
        query_ids=torch.tensor([[12, 10, 11]]),
    )
    torch.testing.assert_close(repeated.module_permission[:, :, query_order], query_permuted.module_permission)


def test_live_query_and_mm_permissions_keep_autograd_path_and_all_open_support():
    module_ids = torch.tensor([[0, 1, 2]])
    allowed = (~torch.eye(3, dtype=torch.bool))[None]
    raw = torch.full((1, 1, 3, 3), -4.0, dtype=torch.float64)
    raw[0, 0, 1, 0] = 0.0
    raw[0, 0, 1, 2] = torch.log(torch.tensor(0.3, dtype=torch.float64))
    raw[0, 0, 0, 1] = 0.0
    raw[0, 0, 0, 2] = torch.log(torch.tensor(0.05, dtype=torch.float64))
    raw.requires_grad_()
    mm = _snapshot("mm", raw, allowed, module_ids, module_ids, torch.ones(1, 3, dtype=torch.float64))
    me = _snapshot(
        "me", torch.zeros(1, 1, 3, 0, dtype=torch.float64), torch.zeros(1, 3, 0, dtype=torch.bool),
        module_ids, torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.float64),
    )
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    q_scores = torch.stack(
        (
            torch.tensor(0.0, dtype=torch.float64),
            torch.log(torch.tensor(0.3, dtype=torch.float64)),
            torch.log(torch.tensor(0.05, dtype=torch.float64)),
        )
    ).reshape(1, 1, 1, 3).requires_grad_()
    plan = build_query_retrieval_plan(q_scores, graph, tau_q=0.2)
    target_permission = plan.module_permission[:, :, :, 2].sum()
    grad_query, grad_route = torch.autograd.grad(target_permission, (q_scores, raw))
    assert grad_query.abs().sum() > 0
    assert grad_route.abs().sum() > 0
    assert plan.module_permission[..., 2].max() < 1
    all_open = build_query_retrieval_plan(q_scores, graph, tau_q=0.0)
    assert torch.equal(all_open.module_support, torch.ones_like(all_open.module_support))


def _records(kind: str, *, zero_measure: bool = False) -> NativeTokenRecords:
    measure = torch.tensor([[0.0, 0.4]]) if zero_measure else torch.tensor([[0.2, 0.4]])
    return NativeTokenRecords(
        kind,
        torch.tensor([[10, 11]]),
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, dtype=torch.bool),
        measure,
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 4, requires_grad=True),
        torch.zeros(1, 2, 2, 2),
        torch.zeros(1, 2, 2, 2),
    )


def test_token_store_binds_versions_and_expires_after_backward_including_zero_measure_present_ids():
    modules = _records("module", zero_measure=True)
    environments = _records("environment")
    store = NativeTokenStore(
        modules,
        environments,
        5,
        ((1, 2),),
        ((3, 4),),
        ((5, 6, 7),),
        ("scene", "normalizer", "frame"),
        (("threshold", 0.2),),
        frame_key="frame",
        normalizer_key="normalizer",
        live_autograd_graph=True,
    )
    assert store.modules.measure[0, 0] == 0
    store.assert_current(
        model_owner=5,
        model_parameter_versions=((1, 2),),
        model_buffer_versions=((3, 4),),
        scene_tensor_versions=((5, 6, 7),),
        scene_identity=("scene", "normalizer", "frame"),
        model_config_key=(("threshold", 0.2),),
    )
    with torch.no_grad():
        modules.native_ids[0, 0] = 99
    with pytest.raises(ValueError, match="module token records changed in place"):
        store.assert_current(
            model_owner=5,
            model_parameter_versions=((1, 2),),
            model_buffer_versions=((3, 4),),
            scene_tensor_versions=((5, 6, 7),),
            scene_identity=("scene", "normalizer", "frame"),
            model_config_key=(("threshold", 0.2),),
        )
    store.mark_backward_consumed()
    with pytest.raises(ValueError, match="autograd graph was consumed"):
        store.assert_current(
            model_owner=5,
            model_parameter_versions=((1, 2),),
            model_buffer_versions=((3, 4),),
            scene_tensor_versions=((5, 6, 7),),
            scene_identity=("scene", "normalizer", "frame"),
            model_config_key=(("threshold", 0.2),),
        )


def test_hypergraph_rejects_mutated_route_sources_and_derived_memberships():
    mm, me = _graph_fixture()
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    graph.assert_current(
        owner_token=17,
        parameter_versions=((1, 2),),
        buffer_versions=((3, 4),),
        scene_identity=("scene-A", "train-norm-v1", "native-frame-v2"),
        model_config_key=(("family", "native-test"),),
    )
    with torch.no_grad():
        graph.module_member_permissions[0, 0, 0, 1].add_(0.01)
    with pytest.raises(ValueError, match="memberships, probabilities, or masks changed"):
        graph.assert_current()

    mm2, me2 = _graph_fixture()
    graph2 = build_native_hypergraph(mm2, me2, tau_mm=0.2, tau_me=0.2)
    with torch.no_grad():
        mm2.raw_scores[0, 0, 0, 1].add_(0.1)
    with pytest.raises(ValueError, match="Captured native route scores"):
        build_query_retrieval_plan(torch.zeros(1, 1, 2, 3), graph2, tau_q=0.2)


def test_scene_tensor_binding_rejects_same_ids_and_key_graph_swap():
    mm, me = _graph_fixture()
    source_versions = ((101, 201, 1), (102, 202, 4))
    replacement_versions = ((101, 201, 2), (102, 202, 4))
    source_mm = _snapshot(
        "mm", mm.raw_scores, mm.allowed, mm.destination_ids, mm.donor_ids, mm.donor_measure,
        scene_tensor_versions=source_versions,
    )
    source_me = _snapshot(
        "me", me.raw_scores, me.allowed, me.destination_ids, me.donor_ids, me.donor_measure,
        scene_tensor_versions=source_versions,
    )
    source_graph = build_native_hypergraph(source_mm, source_me, tau_mm=0.2, tau_me=0.2)

    replacement_mm = _snapshot(
        "mm", mm.raw_scores + 0.25, mm.allowed, mm.destination_ids, mm.donor_ids, mm.donor_measure,
        scene_tensor_versions=replacement_versions,
    )
    replacement_me = _snapshot(
        "me", me.raw_scores + 0.1, me.allowed, me.destination_ids, me.donor_ids, me.donor_measure,
        scene_tensor_versions=replacement_versions,
    )
    replacement_graph = build_native_hypergraph(
        replacement_mm, replacement_me, tau_mm=0.2, tau_me=0.2
    )
    assert torch.equal(source_graph.module_ids, replacement_graph.module_ids)
    assert source_graph.mm_snapshot.scene_identity == replacement_graph.mm_snapshot.scene_identity
    assert source_graph.mm_snapshot.parameter_versions == replacement_graph.mm_snapshot.parameter_versions
    source_graph.assert_current(scene_tensor_versions=source_versions)
    with pytest.raises(ValueError, match="scene tensor versions binding is stale"):
        replacement_graph.assert_current(scene_tensor_versions=source_versions)


def _query_fixture(*, receiver="receiver-a"):
    return QueryBatch(
        coordinates=torch.tensor([[[0.15, 0.25], [0.45, 0.55]]], dtype=torch.float64),
        features=torch.tensor([[[0.2], [0.4]]], dtype=torch.float64),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
        query_ids=torch.tensor([[901, 902]]),
        local_batch_indices=torch.tensor([0, 0]),
        local_query_indices=torch.tensor([0, 1]),
        local_module_indices=torch.tensor([1, 2]),
        local_weights=torch.tensor([0.7, 0.8], dtype=torch.float64),
        scene_key="scene-A",
        coordinate_frame_key="native-frame-v2",
        receiver_key=receiver,
    )


def _bound_plan(graph, query, *, token_store=None):
    return build_query_retrieval_plan(
        torch.zeros(1, 2, 2, 3, dtype=torch.float64),
        graph,
        tau_q=0.2,
        query_mask=query.query_mask,
        query_ids=query.query_ids,
        query_batch=query,
        token_store=token_store,
    )


def _store_bound_to_graph(graph):
    def records(kind, ids):
        batch, count = ids.shape
        return NativeTokenRecords(
            kind,
            ids,
            torch.zeros(batch, count, 2, dtype=torch.float64),
            torch.ones(batch, count, 2, dtype=torch.float64),
            torch.ones(batch, count, dtype=torch.bool),
            torch.full((batch, count), 1.0 / max(count, 1), dtype=torch.float64),
            torch.zeros(batch, count, 3, dtype=torch.float64),
            torch.zeros(batch, count, 4, dtype=torch.float64),
        )

    modules = records("module", graph.module_ids)
    environments = records("environment", graph.environment_ids)
    snapshot = graph.mm_snapshot
    return NativeTokenStore(
        modules,
        environments,
        snapshot.owner_token,
        snapshot.parameter_versions,
        snapshot.buffer_versions,
        snapshot.scene_tensor_versions,
        snapshot.scene_identity,
        snapshot.model_config_key,
        live_autograd_graph=True,
    )


def test_bound_query_plan_rejects_mutated_geometry_ids_identity_and_model_owner():
    mm, me = _graph_fixture()
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)

    query = _query_fixture()
    plan = _bound_plan(graph, query)
    assert plan.query_tensor_versions is not None
    assert plan.query_identity == ("scene-A", "native-frame-v2", "receiver-a")
    plan.assert_query_current(query, organization_model_owner=17)
    with pytest.raises(ValueError, match="different model instance"):
        plan.assert_query_current(query, organization_model_owner=18)

    swapped = replace(query, coordinates=query.coordinates.clone() + 0.1)
    assert torch.equal(swapped.query_ids, query.query_ids)
    with pytest.raises(ValueError, match="geometry, features, masks, IDs, or local support changed"):
        plan.assert_query_current(swapped)

    changed_receiver = replace(query, receiver_key="receiver-b")
    with pytest.raises(ValueError, match="scene, frame, or receiver identity changed"):
        plan.assert_query_current(changed_receiver)

    changed_coordinates = _query_fixture()
    coordinate_plan = _bound_plan(graph, changed_coordinates)
    with torch.no_grad():
        changed_coordinates.coordinates[0, 0, 0].add_(0.01)
    with pytest.raises(ValueError, match="geometry, features, masks, IDs, or local support changed"):
        coordinate_plan.assert_query_current(changed_coordinates)

    changed_ids = _query_fixture()
    id_plan = _bound_plan(graph, changed_ids)
    with torch.no_grad():
        changed_ids.query_ids[0, 0].add_(100)
    with pytest.raises(ValueError, match="geometry, features, masks, IDs, or local support changed"):
        id_plan.assert_query_current(changed_ids)


def test_query_plan_checks_owned_tensors_rebinds_on_replace_and_tracks_store_lifetime():
    mm, me = _graph_fixture()
    graph = build_native_hypergraph(mm, me, tau_mm=0.2, tau_me=0.2)
    low_level_plan = build_query_retrieval_plan(
        torch.zeros(1, 2, 2, 3, dtype=torch.float64), graph, tau_q=0.2
    )
    low_level_plan.assert_query_current()
    with torch.no_grad():
        low_level_plan.module_permission[0, 0, 0, 0].add_(0.1)
    with pytest.raises(ValueError, match="plan permissions, support, IDs, or indices changed"):
        low_level_plan.assert_query_current()

    fresh = build_query_retrieval_plan(
        torch.zeros(1, 2, 2, 3, dtype=torch.float64), graph, tau_q=0.2
    )
    restamped = replace(fresh, module_permission=fresh.module_permission.clone())
    restamped.assert_query_current()

    query = _query_fixture()
    store = _store_bound_to_graph(graph)
    bound = _bound_plan(graph, query, token_store=store)
    bound.assert_query_current(query, organization_model_owner=17)
    same_metadata_store = _store_bound_to_graph(graph)
    assert same_metadata_store.model_owner == store.model_owner
    assert same_metadata_store.scene_identity == store.scene_identity
    assert torch.equal(same_metadata_store.modules.native_ids, store.modules.native_ids)
    with pytest.raises(ValueError, match="different prepared token store"):
        bound.assert_query_current(
            query,
            organization_model_owner=17,
            organization_token_store=same_metadata_store,
        )
    bound.assert_query_current(
        query,
        organization_model_owner=17,
        organization_token_store=store,
    )
    store.mark_backward_consumed()
    with pytest.raises(ValueError, match="autograd graph was consumed"):
        bound.assert_query_current(query, organization_model_owner=17)
