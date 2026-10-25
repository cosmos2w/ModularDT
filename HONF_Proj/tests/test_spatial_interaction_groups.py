from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_forward_core.content_hypergraph import QueryBatch, SceneBatch
from honf_forward_core.interaction_groups.model import nullable
from honf_forward_core.spatial_interaction_groups import (
    SpatialInteractionGroupCore,
    build_balanced_grid_template,
    extend_spatial_coordinate,
    map_compatible_parent_tensors,
)
from honf_forward_core.spatial_interaction_groups.frontier import geometry_key
from honf_forward_core.spatial_interaction_groups.model import (
    _SpatialGroupReader,
    _node_statistics,
    _utility_feature_rows,
)


def _grid(nx: int = 5, ny: int = 3) -> np.ndarray:
    x = np.linspace(0.0, 8.0, nx)
    y = np.linspace(-0.5, 0.5, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)


def _scene(batch: int = 2) -> SceneBatch:
    nx, ny = 4, 2
    x = torch.linspace(0.0, 6.0, nx)
    y = torch.tensor([-0.4, 0.4])
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    environment_coordinates = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    modules = torch.tensor([[0.3, -0.2], [2.1, 0.1], [4.9, 0.3]])
    return SceneBatch(
        module_features=torch.randn(batch, 3, 9),
        environment_features=torch.randn(batch, nx * ny, 6),
        module_coordinates=modules[None].expand(batch, -1, -1).clone(),
        environment_coordinates=environment_coordinates[None].expand(batch, -1, -1).clone(),
        module_mask=torch.tensor([[True, True, False]]).expand(batch, -1).clone(),
        environment_mask=torch.tensor([[True] * (nx * ny)]).expand(batch, -1).clone(),
        module_measure=torch.tensor([[0.7, 1.3, 0.0]]).expand(batch, -1).clone(),
        environment_measure=torch.tensor([[0.2, 0.5, 0.8, 0.4, 1.1, 0.3, 0.9, 0.6]]).expand(batch, -1).clone(),
        module_ids=torch.tensor([[17, 31, 43]]).expand(batch, -1).clone(),
        environment_ids=torch.arange(101, 101 + nx * ny)[None].expand(batch, -1).clone(),
        local_radius=torch.full((batch, 3), 0.4),
        operating_features=torch.randn(batch, 12),
        coordinate_origin=torch.tensor([[0.0, -0.4]]).expand(batch, -1).clone(),
        coordinate_scale=torch.tensor([[6.0, 0.8]]).expand(batch, -1).clone(),
        module_measure_reference=torch.ones(batch),
        environment_measure_reference=torch.ones(batch),
        scene_key=("spatial-case-a", "spatial-case-b")[:batch],
        normalizer_key="fixed-test-input-transform",
        coordinate_frame_key="spatial-test-frame",
        module_characteristic_lengths=torch.full((batch, 3, 2), 0.1),
        environment_characteristic_lengths=torch.full((batch, nx * ny, 2), 0.08),
    )


def _query(batch: int = 2, count: int = 5) -> QueryBatch:
    points = torch.tensor([[0.1, -0.2], [1.4, 0.15], [3.0, 0.0], [5.1, 0.25], [6.5, -0.1]])
    return QueryBatch(
        coordinates=points[None].expand(batch, -1, -1)[:, :count].clone(),
        query_mask=torch.ones(batch, count, dtype=torch.bool),
        scene_key=("spatial-case-a", "spatial-case-b")[:batch],
        coordinate_frame_key="spatial-test-frame",
    )


def _core(architecture: str = "F13-1", **overrides) -> SpatialInteractionGroupCore:
    values = dict(
        module_input_dim=9,
        environment_input_dim=6,
        coordinate_dim=2,
        query_input_dim=0,
        operating_input_dim=12,
        output_dim=4,
        architecture=architecture,
        context_dim=16,
        num_heads=4,
        context_depth=2,
        local_hidden_dim=16,
        dropout=0.0,
        init_seed=731,
        query_tile_size=2,
        environment_nx=4,
        environment_ny=2,
    )
    values.update(overrides)
    return SpatialInteractionGroupCore(**values)


def test_balanced_geometry_is_permutation_invariant_and_uses_physical_extents() -> None:
    coordinates = _grid(nx=5, ny=3)
    template = build_balanced_grid_template(coordinates, 5, 3)
    permutation = np.random.default_rng(12).permutation(len(coordinates))
    assert geometry_key(coordinates) == geometry_key(coordinates[permutation])
    assert template == build_balanced_grid_template(coordinates[permutation], 5, 3)

    extreme_x, extreme_y = np.meshgrid(
        [0.0, 100.0, 200.0], [0.0, 0.01, 0.02, 0.03], indexing="xy"
    )
    extreme = np.stack((extreme_x, extreme_y), axis=-1).reshape(-1, 2)
    long_x = build_balanced_grid_template(extreme, 3, 4)
    left, right = long_x.children[long_x.root]
    assert long_x.box_max[left][0] < long_x.box_min[right][0]

    tree = template.bind(coordinates)
    all_leaves = tree.cover_at_depth(tree.max_depth)
    labels = tree.labels_for_cut(all_leaves)
    assert sorted(labels) == list(range(len(coordinates)))
    assert len(tree.cover_at_depth(0)) == 1
    assert len(tree.cover_at_depth(tree.max_depth)) == len(coordinates)


def test_weighted_all_node_moments_exactly_recover_affine_fields_and_zero_rank_deficient_slopes() -> None:
    coordinates = _grid(nx=5, ny=3)
    template = build_balanced_grid_template(coordinates, 5, 3)
    tree = template.bind(coordinates)
    xy = torch.as_tensor(
        (coordinates - np.array([0.0, -0.5])) / np.array([8.0, 1.0]),
        dtype=torch.float64,
    )
    mass = torch.linspace(0.2, 2.0, len(coordinates), dtype=torch.float64)
    mass[4] = 0.0
    coefficients = torch.tensor([[0.7, -0.2], [-0.3, 0.6], [0.5, 0.4]], dtype=torch.float64)
    intercept = torch.tensor([1.2, -0.5, 2.1], dtype=torch.float64)
    latent = intercept + xy @ coefficients.T
    relation = torch.arange(2 * len(coordinates) * 3, dtype=torch.float64).reshape(2, len(coordinates), 3)
    stats = _node_statistics(
        latent,
        relation,
        xy,
        mass,
        tree,
        torch.tensor([0.0, -0.5], dtype=torch.float64),
        torch.tensor([8.0, 1.0], dtype=torch.float64),
    )
    root = tree.root
    expected = coefficients.T * stats.scales[root, :, None]
    assert torch.allclose(stats.slopes[root], expected, atol=1e-5, rtol=1e-5)
    assert torch.isclose(stats.mass[root], mass.sum(), atol=1e-12, rtol=1e-12)
    assert stats.latent_constant_residual[root] > stats.latent_residual[root]
    assert stats.latent_residual[root] < 1e-5

    singleton = next(node for node in range(tree.leaf_count) if tree.sizes[node] == 1)
    assert torch.equal(stats.slopes[singleton], torch.zeros_like(stats.slopes[singleton]))
    line_nodes = []
    for node in tree.internal_nodes:
        start, end = tree.intervals[node]
        member_ids = tree.leaf_order[start:end]
        positive_ids = [index for index in member_ids if mass[index] > 0]
        if len(positive_ids) < 2:
            continue
        support = xy[positive_ids]
        span = support.amax(0) - support.amin(0)
        if (span[0] > 0) ^ (span[1] > 0):
            line_nodes.append((node, span))
    assert line_nodes
    line_node, line_span = line_nodes[0]
    live_axis = int(torch.nonzero(line_span > 0, as_tuple=False)[0])
    missing_axis = 1 - live_axis
    expected_live = coefficients.T[live_axis] * stats.scales[line_node, live_axis]
    assert torch.allclose(stats.slopes[line_node, live_axis], expected_live, atol=1e-5, rtol=1e-5)
    assert torch.equal(stats.slopes[line_node, missing_axis], torch.zeros_like(stats.slopes[line_node, missing_axis]))

    diagonal_mass = torch.zeros_like(mass)
    diagonal_mass[0], diagonal_mass[-1] = 0.7, 1.3
    diagonal = _node_statistics(
        latent,
        relation,
        xy,
        diagonal_mass,
        tree,
        torch.tensor([0.0, -0.5], dtype=torch.float64),
        torch.tensor([8.0, 1.0], dtype=torch.float64),
    )
    root_mass = diagonal.mass[root]
    root_mean = (latent * diagonal_mass[:, None]).sum(0) / root_mass
    root_center = (xy * diagonal_mass[:, None]).sum(0) / root_mass
    relative = (xy - root_center) / diagonal.scales[root]
    design_covariance = (
        relative[:, :, None] * relative[:, None, :] * diagonal_mass[:, None, None] / root_mass
    ).sum(0)
    design_cross = (
        relative[:, :, None]
        * (latent - root_mean)[:, None, :]
        * diagonal_mass[:, None, None]
        / root_mass
    ).sum(0)
    declared_ridge_solution = torch.linalg.solve(
        design_covariance + 1.0e-6 * torch.eye(2, dtype=torch.float64), design_cross
    )
    assert torch.allclose(diagonal.slopes[root], declared_ridge_solution, atol=1e-10, rtol=1e-10)

    permutation = torch.randperm(len(coordinates), generator=torch.Generator().manual_seed(44))
    perm_tree = template.bind(coordinates[permutation.numpy()])
    permuted = _node_statistics(
        latent[permutation],
        relation[:, permutation],
        xy[permutation],
        mass[permutation],
        perm_tree,
        torch.tensor([0.0, -0.5], dtype=torch.float64),
        torch.tensor([8.0, 1.0], dtype=torch.float64),
    )
    for name in ("mass", "means", "slopes", "latent_residual", "relation_mean", "relation_scatter"):
        assert torch.allclose(getattr(stats, name), getattr(permuted, name), atol=1e-10, rtol=1e-10)

    all_zero_latent = torch.randn_like(latent, requires_grad=True)
    empty = _node_statistics(
        all_zero_latent,
        relation,
        xy,
        torch.zeros_like(mass),
        tree,
        torch.tensor([0.0, -0.5], dtype=torch.float64),
        torch.tensor([8.0, 1.0], dtype=torch.float64),
    )
    empty_loss = empty.means.square().sum() + empty.slopes.square().sum() + empty.latent_residual.sum()
    empty_loss.backward()
    assert torch.isfinite(all_zero_latent.grad).all()
    assert torch.equal(all_zero_latent.grad, torch.zeros_like(all_zero_latent.grad))


def test_position_state_reassignment_preserves_marginals_but_reverses_group_slope() -> None:
    coordinates = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    tree = build_balanced_grid_template(coordinates, 2, 1).bind(coordinates)
    xy = torch.as_tensor((coordinates - np.asarray([-1.0, 0.0])) / np.asarray([2.0, 1.0]))
    xy = xy.to(torch.float64)
    mass = torch.ones(2, dtype=torch.float64)
    relation = torch.zeros(2, 2, 1, dtype=torch.float64)
    first_assignment = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    reassigned = torch.flip(first_assignment, dims=(0,))
    frame_origin = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    frame_scale = torch.tensor([2.0, 1.0], dtype=torch.float64)

    first = _node_statistics(
        first_assignment, relation, xy, mass, tree, frame_origin, frame_scale
    )
    second = _node_statistics(
        reassigned, relation, xy, mass, tree, frame_origin, frame_scale
    )
    root = tree.root

    # Both joint assignments have the same coordinate and latent-state marginals,
    # and hence the same root mean, while their coordinate/state association flips.
    first_coordinate_marginal = (xy * mass[:, None]).sum(0) / mass.sum()
    second_coordinate_marginal = (xy * mass.flip(0)[:, None]).sum(0) / mass.sum()
    assert torch.equal(first_coordinate_marginal, second_coordinate_marginal)
    assert torch.equal(first_assignment.sort(dim=0).values, reassigned.sort(dim=0).values)
    assert torch.equal(first.means[root], second.means[root])
    assert torch.allclose(
        second.slopes[root], -first.slopes[root], atol=1.0e-12, rtol=1.0e-12
    )

    relative = (xy - first.centers[root]) / first.scales[root]
    first_prediction = first.means[root] + relative @ first.slopes[root]
    second_prediction = second.means[root] + relative @ second.slopes[root]
    assert torch.allclose(second_prediction, -first_prediction, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(
        first_prediction[:, 0], first_assignment[:, 0], atol=1.0e-5, rtol=1.0e-5
    )


def test_module_axis_relation_permutation_leaves_organizer_features_invariant() -> None:
    coordinates = _grid(nx=5, ny=3)
    template = build_balanced_grid_template(coordinates, 5, 3)
    tree = template.bind(coordinates)
    xy = torch.as_tensor(
        (coordinates - np.asarray([0.0, -0.5])) / np.asarray([8.0, 1.0]),
        dtype=torch.float64,
    )
    mass = torch.linspace(0.2, 2.0, len(coordinates), dtype=torch.float64)
    latent = torch.randn(
        len(coordinates),
        7,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(61),
    )
    relation = torch.randn(
        4,
        len(coordinates),
        5,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(62),
    )
    module_permutation = torch.tensor([2, 0, 4, 1, 3])
    frame_origin = torch.tensor([0.0, -0.5], dtype=torch.float64)
    frame_scale = torch.tensor([8.0, 1.0], dtype=torch.float64)

    original = _node_statistics(
        latent, relation, xy, mass, tree, frame_origin, frame_scale
    )
    permuted = _node_statistics(
        latent, relation[..., module_permutation], xy, mass, tree, frame_origin, frame_scale
    )
    nodes, original_features = _utility_feature_rows(original, relation_features=True)
    permuted_nodes, permuted_features = _utility_feature_rows(permuted, relation_features=True)

    assert nodes == permuted_nodes == tree.internal_nodes
    assert torch.allclose(
        permuted.relation_mean,
        original.relation_mean[..., module_permutation],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.allclose(
        original.relation_scatter, permuted.relation_scatter, atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.allclose(
        original.relation_magnitude,
        permuted.relation_magnitude,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    # Relation-derived utility columns are the final magnitude, scatter and
    # left/right difference features; all organizer decisions therefore ignore
    # the arbitrary ordering of the module axis.
    assert torch.allclose(
        original_features[:, -3:],
        permuted_features[:, -3:],
        atol=1.0e-12,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize("zero_slopes", [False, True])
def test_factorized_spatial_read_matches_literal_value_expansion_and_gradients(zero_slopes: bool) -> None:
    torch.manual_seed(43)
    reader = _SpatialGroupReader(width=8, heads=2).double()
    query = torch.randn(4, 2, 4, dtype=torch.float64, requires_grad=True)
    receivers = torch.tensor([[0.2, -0.4], [1.3, 0.5], [2.0, -1.5], [0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    masses = torch.tensor([0.8, 0.4, 0.0], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[0.5, 0.1], [1.2, -0.3], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[0.7, 0.5], [0.3, 0.8], [1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    keys = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    coefficients = torch.randn(3, 3, 2, 4, dtype=torch.float64, requires_grad=True)
    group = SimpleNamespace(
        mass=masses,
        centers=centers,
        scales=scales,
        keys=keys,
        value_coefficients=coefficients,
    )
    actual, actual_weights, _ = reader.read(query, receivers, group, zero_slopes=zero_slopes)

    relative = (receivers[:, None, :] - centers[None, :, :]) / scales[None, :, :].clamp_min(1.0e-12)
    log_extent = scales.clamp_min(1.0e-12).log().mean(-1)
    geo_features = torch.cat((relative, log_extent[None, :, None].expand(len(query), -1, -1)), dim=-1)
    geo = reader.geo(geo_features).movedim(-1, -3)
    logits = torch.einsum("qhd,khd->hqk", query, keys) / 2.0
    logits = logits + geo + masses.clamp_min(1.0e-30).log()[None, None, :]
    valid = (masses > 0)[None, None, :].expand(2, len(query), -1)
    weights = nullable(logits, valid)
    basis = extend_spatial_coordinate(relative)
    literal_values = coefficients[:, 0]
    if not zero_slopes:
        literal_values = literal_values[None] + torch.einsum(
            "qka,kahd->qkhd", basis, coefficients[:, 1:]
        )
        context = torch.einsum("hqk,qkhd->qhd", weights, literal_values)
    else:
        context = torch.einsum("hqk,khd->qhd", weights, literal_values)
    expected = reader.out(context.flatten(-2))
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)
    assert torch.allclose(actual_weights, weights, atol=1e-11, rtol=1e-11)

    parameters = tuple(reader.geo.parameters()) + tuple(reader.out.parameters())
    inputs = (query, receivers, masses, centers, scales, keys, coefficients)
    actual_loss = actual.square().sum() + actual_weights.square().sum()
    expected_loss = expected.square().sum() + weights.square().sum()
    actual_grad = torch.autograd.grad(actual_loss, inputs + parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected_loss, inputs + parameters)
    for left, right in zip(actual_grad, expected_grad):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-10)


def test_forced_parent_descendants_preserve_off_path_cover_and_expose_exploration_metadata() -> None:
    coordinates = _grid(nx=8, ny=4)
    tree = build_balanced_grid_template(coordinates, 8, 4).bind(coordinates)
    decisions = [True] * tree.node_count
    # Stop the root's left child, then probe one internal node below that stop.
    root_left, root_right = tree.children[tree.root]
    decisions[root_left] = False
    target = next(
        node
        for node in tree.internal_nodes
        if tree.parents[node] == root_left
    )
    baseline = tree.cut_from_decisions(decisions)
    parent = tree.cut_from_decisions(decisions, (target, "parent"))
    children = tree.cut_from_decisions(decisions, (target, "children"))
    grandchildren = tree.cut_from_decisions(decisions, (target, "grandchildren"))
    target_span = tree.intervals[target]
    assert target in parent
    assert set(tree.children[target]).issubset(children)
    grandchildren_active = set(grandchildren)
    for child in tree.children[target]:
        grandchild_pair = tree.children[child]
        if grandchild_pair is None:
            assert child in grandchildren_active
        else:
            assert set(grandchild_pair).issubset(grandchildren_active)
    outside_nodes = lambda cover: {
        node for node in cover if tree.intervals[node][1] <= target_span[0] or tree.intervals[node][0] >= target_span[1]
    }
    assert outside_nodes(parent) == outside_nodes(children) == outside_nodes(grandchildren)
    for cover in (baseline, parent, children, grandchildren):
        assert len(tree.labels_for_cut(cover)) == len(coordinates)

    core = _core()
    core.train()
    core.set_frontier_context(epoch=4, update=7, seed=13, exploration=True)
    prepared = core.prepare(_scene(batch=1), exploration=True)
    frontier = prepared.layers[0][0].frontier
    assert frontier.selection_mode in {"learned", "scale_exploration"}
    if frontier.exploration_selected:
        assert frontier.exploration_depth is not None
        assert frontier.nodes == frontier.tree.cover_at_depth(frontier.exploration_depth)


def test_cached_live_group_count_only_changes_decoder_work_counters() -> None:
    core = _core()
    core.eval()
    scene, query = _scene(batch=1), _query(batch=1, count=3)
    core.set_frontier_context(epoch=2, update=0, seed=91, exploration=False)
    prepared = core.prepare(scene, exploration=False)
    group = prepared.final_groups[0]
    assert group.live_group_count == int((group.mass > 0).sum())
    parameters = (core.layers[1].group_projection.v.weight, core.field_head.weight)

    first = core.decode(prepared, query)
    first_gradients = torch.autograd.grad(
        first.joint_fields.square().sum(), parameters, retain_graph=True
    )
    original_tile_size = core.query_tile_size
    core.query_tile_size = 1
    retiled = core.decode(prepared, query)
    core.query_tile_size = original_tile_size
    assert torch.allclose(first.joint_fields, retiled.joint_fields, atol=1.0e-7, rtol=1.0e-7)
    assert first.work_counts["group_scores"] == retiled.work_counts["group_scores"]
    assert first.work_counts["group_value_modes_per_group"] == core.num_modes
    assert retiled.work_counts["group_value_modes_per_group"] == core.num_modes
    cached_count = group.live_group_count
    group.live_group_count = 0
    second = core.decode(prepared, query)
    second_gradients = torch.autograd.grad(second.joint_fields.square().sum(), parameters)
    group.live_group_count = cached_count

    assert torch.equal(first.joint_fields, second.joint_fields)
    assert first.work_counts["group_live_scores"] > second.work_counts["group_live_scores"]
    assert second.work_counts["empty_queries"] == query.coordinates.shape[1]
    for first_gradient, second_gradient in zip(first_gradients, second_gradients):
        assert torch.equal(first_gradient, second_gradient)


def test_two_arms_share_parameterization_and_native_prepare_decode_relay_gradients() -> None:
    torch.manual_seed(31)
    scene = _scene()
    query = _query()
    arm0, arm1 = _core("F13-0"), _core("F13-1")
    assert arm0.num_modes == 1 and arm1.num_modes == 3
    assert arm0.state_dict().keys() == arm1.state_dict().keys()
    assert all(torch.equal(value, arm1.state_dict()[name]) for name, value in arm0.state_dict().items())
    assert arm0.utility_feature_dim == arm1.utility_feature_dim

    arm0.train()
    arm0.set_frontier_context(epoch=3, update=9, seed=117, exploration=True)
    arm0_prepared = arm0.prepare(scene, exploration=True)
    arm1.train()
    arm1.set_frontier_context(epoch=3, update=9, seed=117, exploration=True)
    prepared = arm1.prepare(scene, exploration=True)
    assert [group.frontier.nodes for group in arm0_prepared.layers[0]] == [
        group.frontier.nodes for group in prepared.layers[0]
    ]
    assert [group.frontier.selection_mode for group in arm0_prepared.layers[0]] == [
        group.frontier.selection_mode for group in prepared.layers[0]
    ]
    assert prepared.layers[0][0].frontier.nodes == prepared.final_groups[0].frontier.nodes
    assert prepared.layers[0][0].keys.numel() > 0  # Training preserves the relay graph.
    assert prepared.final_groups[0].keys.shape[0] == len(prepared.final_groups[0].cut)
    output = arm1.decode(prepared, query)
    assert output.joint_fields is not None and output.joint_fields.shape == (2, 5, 4)
    output.joint_fields.square().mean().backward()
    for parameter in (
        arm1.layers[0].group_projection.k.weight,
        arm1.layers[0].group_projection.v.weight,
        arm1.layers[1].group_projection.k.weight,
        arm1.layers[1].group_projection.v.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    restored = SpatialInteractionGroupCore.from_config(arm1.get_config())
    restored.load_state_dict(arm1.state_dict(), strict=True)
    assert restored.schema == "spatial_interaction_group_fields_v1"
    fixed = arm1.prepare(scene, forced_frontier={"depth": 1}, exploration=True)
    assert all(
        group.frontier.nodes == group.frontier.tree.cover_at_depth(1)
        for group in fixed.layers[0]
    )
    assert fixed.layers[0][0].frontier.selection_mode == "fixed_cover"
    bypass = arm1.prepare(scene, clamp_layer=0, exploration=False)
    assert bypass.layers[0][0].work_counts["relay_bypassed"] == 1
    assert bypass.layers[0][0].frontier.nodes == bypass.final_groups[0].frontier.nodes
    arm1.eval()
    with torch.no_grad():
        compact = arm1.prepare(scene, exploration=False)
        traced = arm1.prepare(scene, exploration=False, return_trace=True)
    assert compact.layers[0][0].keys.numel() == 0
    assert compact.layers[0][0].value_coefficients.numel() == 0
    assert compact.layers[0][0].environment_states.numel() == 0
    assert compact.final_groups[0].environment_states.numel() == 0
    assert compact.environment_states.numel() == 0
    assert compact.final_groups[0].latent_constant_residual.shape == compact.final_groups[0].mass.shape
    assert compact.final_groups[0].all_node_constant_residual.shape == compact.final_groups[0].all_node_mass.shape
    assert compact.final_groups[0].keys.numel() > 0
    assert compact.relay_trace is None
    assert traced.relay_trace is not None and traced.layers[0][0].keys.numel() > 0
    assert traced.layers[0][0].environment_states.numel() > 0
    assert traced.final_groups[0].environment_states.numel() > 0
    assert traced.environment_states.numel() > 0


def test_parent_mapping_copies_each_stage_projection_and_resets_control_clocks() -> None:
    model = _core("F13-1")
    target = model.state_dict()
    source: dict[str, torch.Tensor] = {}
    clocks = {"organization_epoch", "organization_update", "organization_seed", "exploration_override"}
    remapped_targets = {name for name in target if name.startswith("group_reader.")}
    remapped_targets |= {
        name for name in target if name.startswith("layers.") and ".group_projection." in name
    }
    for name, value in target.items():
        if name in clocks or name in remapped_targets:
            continue
        source[name] = value.detach().clone()
    for name in clocks:
        source[name] = torch.tensor(100 if name == "organization_epoch" else 3, dtype=target[name].dtype)
    for stage in (0, 1):
        for projection in ("k", "v"):
            source[f"layers.{stage}.gm.{projection}.weight"] = torch.full_like(
                target[f"layers.{stage}.group_projection.{projection}.weight"], float(stage + 1)
            )
    for suffix in (
        "norm.weight",
        "norm.bias",
        "q.weight",
        "out.weight",
        "geo.0.weight",
        "geo.0.bias",
        "geo.2.weight",
        "ff.0.weight",
        "ff.0.bias",
        "ff.1.weight",
        "ff.1.bias",
        "ff.3.weight",
        "ff.3.bias",
    ):
        source[f"layers.0.gm.{suffix}"] = target[f"group_reader.{suffix}"].detach().clone()

    mapped, report = map_compatible_parent_tensors(source, model)
    for stage in (0, 1):
        for projection in ("k", "v"):
            name = f"layers.{stage}.group_projection.{projection}.weight"
            assert torch.equal(mapped[name], torch.full_like(mapped[name], float(stage + 1)))
            assert {"source": f"layers.{stage}.gm.{projection}.weight", "target": name} in report["copied"]
    assert mapped["organization_epoch"].item() == 1
    assert mapped["organization_update"].item() == 0
    assert mapped["organization_seed"].item() == model.init_seed
    assert mapped["exploration_override"].item() == -1
    assert set(report["control_clock_reset"]) == clocks
    assert all(name in {entry["target"] for entry in report["new"]} for name in clocks)


def test_prepared_state_rejects_model_mutation_and_query_batching_does_not_change_cut() -> None:
    core = _core()
    core.eval()
    scene = _scene(batch=1)
    core.set_frontier_context(epoch=2, update=5, seed=93, exploration=False)
    first = core.prepare(scene, query_chunk_size=1, exploration=False)
    second = core.prepare(scene, query_chunk_size=7, exploration=False)
    assert first.layers[0][0].frontier.nodes == second.layers[0][0].frontier.nodes
    assert torch.equal(first.layers[0][0].utility_features, second.layers[0][0].utility_features)
    hook = core.module_encoder.register_forward_hook(
        lambda *_args: (_ for _ in ()).throw(AssertionError("candidate metadata ran the encoder"))
    )
    metadata = core.utility_candidate_metadata_for_scene(scene)
    hook.remove()
    assert metadata and all(set(item) == {"node_id", "depth", "size"} for item in metadata)
    query = _query(batch=1, count=3)
    with torch.no_grad():
        core.field_head.bias.add_(0.1)
    with pytest.raises(ValueError, match="different model owner/state"):
        core.decode(first, query)
