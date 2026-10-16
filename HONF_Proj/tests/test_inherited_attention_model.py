from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.model_v2 import ContentHypergraphCoreV2
from honf_forward_core.inherited_attention import (
    InheritedAttentionCoreV1,
    InheritedQueryBatch,
    ReceiverCover,
    build_typed_atlas,
    smoothstep_grid_receiver_cover,
    with_receiver_cover,
)


def _scene(*, batch: int = 2, empty_second_module_set: bool = False) -> SceneBatch:
    module_coordinates = torch.tensor(
        [[0.18, 0.20], [0.79, 0.24], [0.48, 0.78]], dtype=torch.float32
    )
    environment_coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    module_features = torch.tensor([[0.2, -0.1], [0.7, 0.3], [-0.4, 0.6]], dtype=torch.float32)
    environment_features = torch.tensor(
        [[0.1, 0.5], [0.6, -0.3], [-0.2, 0.8], [0.4, 0.2]], dtype=torch.float32
    )
    module_mask = torch.ones(batch, 3, dtype=torch.bool)
    module_measure = torch.tensor([[0.3, 0.4, 0.3]], dtype=torch.float32).expand(batch, -1).clone()
    if empty_second_module_set and batch > 1:
        module_mask[1] = False
        module_measure[1] = 0.0
    return SceneBatch(
        module_features=module_features[None].expand(batch, -1, -1).clone(),
        environment_features=environment_features[None].expand(batch, -1, -1).clone(),
        module_coordinates=module_coordinates[None].expand(batch, -1, -1).clone(),
        environment_coordinates=environment_coordinates[None].expand(batch, -1, -1).clone(),
        module_mask=module_mask,
        environment_mask=torch.ones(batch, 4, dtype=torch.bool),
        module_measure=module_measure,
        environment_measure=torch.tensor([[0.2, 0.3, 0.1, 0.4]], dtype=torch.float32).expand(batch, -1).clone(),
        module_ids=torch.tensor([[41, 7, 29]], dtype=torch.long).expand(batch, -1).clone(),
        environment_ids=torch.tensor([[83, 19, 61, 37]], dtype=torch.long).expand(batch, -1).clone(),
        local_radius=torch.full((batch, 3), 0.48, dtype=torch.float32),
        operating_features=torch.tensor([[0.2], [-0.1]], dtype=torch.float32)[:batch].clone(),
        coordinate_origin=torch.zeros(batch, 2, dtype=torch.float32),
        coordinate_scale=torch.ones(batch, 2, dtype=torch.float32),
        module_measure_reference=torch.ones(batch, dtype=torch.float32),
        environment_measure_reference=torch.ones(batch, dtype=torch.float32),
        scene_key="iag-scene-fixture",
        normalizer_key="fixture-input-transform",
        coordinate_frame_key="fixture-frame-2d",
        module_characteristic_lengths=torch.full((batch, 3, 2), 0.12, dtype=torch.float32),
        environment_characteristic_lengths=torch.full((batch, 4, 2), 0.10, dtype=torch.float32),
    )


def _many_anchor_scene() -> SceneBatch:
    scene = _scene()
    axis = torch.linspace(0.0, 1.0, 4)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).reshape(1, 16, 2)
    order = torch.tensor([7, 0, 13, 2, 8, 5, 15, 1, 11, 4, 10, 3, 14, 6, 9, 12])
    coordinates = coordinates[:, order]
    features = torch.stack((coordinates[..., 0], coordinates[..., 1]), dim=-1)
    ids = torch.tensor([[701, 103, 509, 211, 907, 307, 809, 401, 607, 101, 503, 203, 709, 311, 811, 409]])
    return replace(
        scene,
        environment_features=features.expand(scene.batch_size, -1, -1).clone(),
        environment_coordinates=coordinates.expand(scene.batch_size, -1, -1).clone(),
        environment_mask=torch.ones(scene.batch_size, 16, dtype=torch.bool),
        environment_measure=torch.full((scene.batch_size, 16), 1.0 / 16.0),
        environment_ids=ids.expand(scene.batch_size, -1).clone(),
        environment_characteristic_lengths=torch.full((scene.batch_size, 16, 2), 0.10),
    )


def _base_query(
    scene: SceneBatch,
    *,
    coordinates: torch.Tensor | None = None,
    requires_grad: bool = False,
    query_mask: torch.Tensor | None = None,
) -> QueryBatch:
    if coordinates is None:
        coordinates = torch.tensor(
            [[0.12, 0.16], [0.43, 0.27], [0.77, 0.32], [0.37, 0.82], [0.88, 0.76]],
            dtype=torch.float32,
        )[None].expand(scene.batch_size, -1, -1).clone()
    if requires_grad:
        coordinates.requires_grad_()
    batch, queries = coordinates.shape[:2]
    features = torch.linspace(-0.3, 0.4, queries, dtype=coordinates.dtype)[None, :, None]
    features = features.expand(batch, -1, -1).clone()
    query_ids = torch.arange(500, 500 + queries, dtype=torch.long)[None].expand(batch, -1).clone()
    if query_mask is None:
        query_mask = torch.ones(batch, queries, dtype=torch.bool)
    return QueryBatch(
        coordinates=coordinates,
        features=features,
        query_mask=query_mask,
        query_ids=query_ids,
        scene_key=scene.scene_key,
        coordinate_frame_key=scene.coordinate_frame_key,
        receiver_key="fixture-receiver-panel",
    )


def _query(
    scene: SceneBatch,
    *,
    coordinates: torch.Tensor | None = None,
    requires_grad: bool = False,
    query_mask: torch.Tensor | None = None,
) -> InheritedQueryBatch:
    base = _base_query(
        scene, coordinates=coordinates, requires_grad=requires_grad, query_mask=query_mask
    )
    bounds = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=base.coordinates.dtype)
    cover = smoothstep_grid_receiver_cover(
        base.coordinates,
        scene.environment_coordinates,
        scene.environment_ids,
        scene.environment_mask,
        base.query_ids,
        frame_key=scene.coordinate_frame_key,
        query_mask=base.query_mask,
        legal_bounds=bounds.expand(scene.batch_size, -1, -1),
    )
    return with_receiver_cover(base, cover)


def _core(**overrides) -> InheritedAttentionCoreV1:
    options = {
        "module_input_dim": 2,
        "environment_input_dim": 2,
        "coordinate_dim": 2,
        "query_input_dim": 1,
        "operating_input_dim": 1,
        "output_dim": 2,
        "context_dim": 8,
        "num_heads": 2,
        "context_depth": 1,
        "local_hidden_dim": 8,
        "dropout": 0.0,
        "init_seed": 29,
        "thermal_response": True,
    }
    options.update(overrides)
    return InheritedAttentionCoreV1(**options)


def _slice_scene(scene: SceneBatch, index: int) -> SceneBatch:
    values = {}
    for item in fields(scene):
        value = getattr(scene, item.name)
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == scene.batch_size:
            values[item.name] = value[index : index + 1]
        else:
            values[item.name] = value
    return SceneBatch(**values)


def _select_query(query: InheritedQueryBatch, indices: torch.Tensor, *, axis: int = 1) -> InheritedQueryBatch:
    base = QueryBatch(
        coordinates=query.coordinates.index_select(axis, indices),
        features=query.features.index_select(axis, indices),
        query_mask=query.query_mask.index_select(axis, indices),
        query_ids=query.query_ids.index_select(axis, indices),
        scene_key=query.scene_key,
        coordinate_frame_key=query.coordinate_frame_key,
        receiver_key=query.receiver_key,
    )
    cover = query.receiver_cover
    assert cover is not None
    selected_cover = ReceiverCover(
        anchor_indices=cover.anchor_indices.index_select(axis, indices),
        anchor_ids=cover.anchor_ids.index_select(axis, indices),
        weights=cover.weights.index_select(axis, indices),
        validity=cover.validity.index_select(axis, indices),
        query_ids=cover.query_ids.index_select(axis, indices),
        frame_key=cover.frame_key,
        query_mask=cover.query_mask.index_select(axis, indices) if cover.query_mask is not None else None,
    )
    return with_receiver_cover(base, selected_cover)


def _common_parameters(left, right) -> tuple[str, ...]:
    left_parameters, right_parameters = dict(left.named_parameters()), dict(right.named_parameters())
    common = tuple(sorted(set(left_parameters) & set(right_parameters)))
    assert common
    for name in common:
        assert left_parameters[name].shape == right_parameters[name].shape, name
        torch.testing.assert_close(left_parameters[name], right_parameters[name], rtol=0, atol=0)
    return common


def test_fresh_soft_matches_d0_common_named_init_and_sparse_epoch_law():
    options = {
        "module_input_dim": 2,
        "environment_input_dim": 2,
        "coordinate_dim": 2,
        "query_input_dim": 1,
        "operating_input_dim": 1,
        "output_dim": 2,
        "context_dim": 8,
        "num_heads": 2,
        "context_depth": 1,
        "local_hidden_dim": 8,
        "dropout": 0.0,
        "init_seed": 29,
    }
    d0 = ContentHypergraphCoreV2(
        **options, architecture="A2-D", num_hyperedges=0, thermal_response_rank=4
    )
    soft = _core()
    common = _common_parameters(d0, soft)
    assert "module_encoder.0.weight" in common
    assert "context_blocks.0.em.query.weight" in common
    assert "query_encoder.net.0.weight" in common
    assert "local_reader.relative_projection.weight" in common

    sparse = _core(normalization="measure_entmax15", scene_executor="selected")
    sparse.load_state_dict(soft.state_dict())
    scene = _scene()
    sparse.eval()
    soft.eval()

    sparse.set_inherited_attention_training_epoch(500)
    soft_org = soft.prepare(scene)
    sparse_500 = sparse.prepare(scene)
    torch.testing.assert_close(sparse_500.module_atlas.probabilities, soft_org.module_atlas.probabilities, rtol=0, atol=0)
    torch.testing.assert_close(sparse_500.environment_atlas.probabilities, soft_org.environment_atlas.probabilities, rtol=0, atol=0)
    assert sparse_500.normalization_blend == 0.0

    sparse.set_inherited_attention_training_epoch(501)
    sparse_501 = sparse.prepare(scene)
    blend = 1.0 / 300.0
    sparse_module = build_typed_atlas(sparse_501.module_score_snapshot, normalization="entmax15")
    soft_module = build_typed_atlas(sparse_501.module_score_snapshot, normalization="softmax")
    sparse_environment = build_typed_atlas(sparse_501.environment_score_snapshot, normalization="entmax15")
    soft_environment = build_typed_atlas(sparse_501.environment_score_snapshot, normalization="softmax")
    torch.testing.assert_close(
        sparse_501.module_atlas.probabilities,
        (1.0 - blend) * soft_module.probabilities + blend * sparse_module.probabilities,
        rtol=2e-6, atol=2e-7,
    )
    torch.testing.assert_close(
        sparse_501.environment_atlas.probabilities,
        (1.0 - blend) * soft_environment.probabilities + blend * sparse_environment.probabilities,
        rtol=2e-6, atol=2e-7,
    )
    assert sparse_501.module_atlas.normalization == "softmax_entmax15_blend"
    assert sparse_501.normalization_blend == blend

    sparse.set_inherited_attention_training_epoch(800)
    sparse_800 = sparse.prepare(scene)
    exact_module = build_typed_atlas(sparse_800.module_score_snapshot, normalization="entmax15")
    exact_environment = build_typed_atlas(sparse_800.environment_score_snapshot, normalization="entmax15")
    torch.testing.assert_close(sparse_800.module_atlas.probabilities, exact_module.probabilities, rtol=0, atol=0)
    torch.testing.assert_close(sparse_800.environment_atlas.probabilities, exact_environment.probabilities, rtol=0, atol=0)
    assert sparse_800.normalization_blend == 1.0
    assert sparse_800.work_counts["scene_value_executed_associations"] == int(
        (sparse_800.module_atlas.probabilities > 0).sum()
        + (sparse_800.environment_atlas.probabilities > 0).sum()
    )


def test_only_final_em_ee_snapshots_and_live_score_value_gradients():
    scene = _scene()
    core = _core(context_depth=2).eval()
    organization = core.prepare(scene, return_trace=True)
    assert organization.scene_key == scene.scene_key
    assert organization.trace["score_stage"] == "last_em_ee"
    assert organization.trace["payload_stage"] == "final_tokens"
    cycles = organization.trace["context_cycles"]
    assert len(cycles) == 2
    assert "em_score_snapshot" not in cycles[0]
    assert "ee_score_snapshot" not in cycles[0]
    assert cycles[1]["em_score_snapshot"] is organization.module_score_snapshot
    assert cycles[1]["ee_score_snapshot"] is organization.environment_score_snapshot
    assert organization.module_score_snapshot.score_origin == "last_em"
    assert organization.environment_score_snapshot.score_origin == "last_ee"
    assert organization.module_score_snapshot.raw_logits.shape == (2, 2, 4, 3)
    assert organization.environment_score_snapshot.raw_logits.shape == (2, 2, 4, 4)
    assert organization.module_values.values.shape == (2, 2, 3, 4)
    assert organization.environment_values.values.shape == (2, 2, 4, 4)
    assert organization.module_score_snapshot.raw_logits.requires_grad
    assert organization.environment_score_snapshot.raw_logits.requires_grad
    assert organization.module_values.values.requires_grad
    assert organization.environment_values.values.requires_grad

    query = _query(scene)
    output = core.decode(organization, query).nonlinear_field
    leaves = (
        organization.module_score_snapshot.raw_logits,
        organization.environment_score_snapshot.raw_logits,
        organization.module_values.values,
        organization.environment_values.values,
    )
    gradients = torch.autograd.grad(output.square().sum(), leaves)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_dense_reference_and_indexed_query_read_match_outputs_parameter_and_geometry_grads():
    scene_a, scene_b = _many_anchor_scene(), _many_anchor_scene()
    scene_a.environment_coordinates.requires_grad_()
    scene_b.environment_coordinates.requires_grad_()
    query_a = _query(scene_a, requires_grad=True)
    query_b = _query(scene_b, requires_grad=True)
    indexed = _core(query_executor="indexed").eval()
    dense = _core(query_executor="dense_reference").eval()
    dense.load_state_dict(indexed.state_dict())
    org_a, org_b = indexed.prepare(scene_a), dense.prepare(scene_b)
    out_a = indexed.decode(org_a, query_a)
    out_b = dense.decode(org_b, query_b)
    torch.testing.assert_close(out_a.nonlinear_field, out_b.nonlinear_field, rtol=3e-5, atol=3e-6)
    assert out_a.work_counts["cover_executed_anchor_reads"] < out_b.work_counts["cover_executed_anchor_reads"]
    params_a, params_b = dict(indexed.named_parameters()), dict(dense.named_parameters())
    names = ("module_value_projection.weight", "context_blocks.0.em.geometry_score.0.weight")
    grads_a = torch.autograd.grad(
        out_a.nonlinear_field.square().sum(),
        (params_a[names[0]], params_a[names[1]], query_a.coordinates, scene_a.environment_coordinates),
    )
    grads_b = torch.autograd.grad(
        out_b.nonlinear_field.square().sum(),
        (params_b[names[0]], params_b[names[1]], query_b.coordinates, scene_b.environment_coordinates),
    )
    for actual, expected in zip(grads_a, grads_b):
        torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-5)


def test_selected_scene_edges_match_dense_values_gradients_thermal_and_vjp():
    scene_dense, scene_selected = _scene(batch=1), _scene(batch=1)
    for scene in (scene_dense, scene_selected):
        scene.module_features.requires_grad_()
        scene.environment_features.requires_grad_()
        scene.module_coordinates.requires_grad_()
        scene.environment_coordinates.requires_grad_()
    query_dense = _query(scene_dense, requires_grad=True)
    query_selected = _query(scene_selected, requires_grad=True)
    dense = _core(normalization="measure_entmax15", scene_executor="dense").eval()
    selected = _core(normalization="measure_entmax15", scene_executor="selected").eval()
    selected.load_state_dict(dense.state_dict())
    dense.set_inherited_attention_training_epoch(801)
    selected.set_inherited_attention_training_epoch(801)
    organization_dense = dense.prepare(scene_dense)
    organization_selected = selected.prepare(scene_selected)

    torch.testing.assert_close(
        organization_dense.module_atlas.probabilities,
        organization_selected.module_atlas.probabilities,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        organization_dense.environment_atlas.probabilities,
        organization_selected.environment_atlas.probabilities,
        rtol=0,
        atol=0,
    )
    for dense_values, selected_values in (
        (organization_dense.module_anchor_values.values, organization_selected.module_anchor_values.values),
        (organization_dense.environment_anchor_values.values, organization_selected.environment_anchor_values.values),
    ):
        torch.testing.assert_close(dense_values, selected_values, rtol=3.0e-5, atol=3.0e-6)
    selected_edges = int(
        (organization_selected.module_atlas.probabilities > 0).sum()
        + (organization_selected.environment_atlas.probabilities > 0).sum()
    )
    assert organization_selected.work_counts["scene_value_executed_associations"] == selected_edges
    assert organization_dense.work_counts["scene_value_executed_associations"] == int(
        organization_dense.module_atlas.probabilities.numel()
        + organization_dense.environment_atlas.probabilities.numel()
    )

    controls_dense = torch.tensor([[0.7, -0.4, 1.1]], requires_grad=True)
    controls_selected = controls_dense.detach().clone().requires_grad_()
    output_dense = dense.decode(organization_dense, query_dense, controls_dense)
    output_selected = selected.decode(organization_selected, query_selected, controls_selected)
    assert output_dense.thermal_temperature is not None
    assert output_selected.thermal_temperature is not None
    torch.testing.assert_close(output_dense.nonlinear_field, output_selected.nonlinear_field, rtol=3.0e-5, atol=3.0e-6)
    torch.testing.assert_close(
        output_dense.thermal_temperature,
        output_selected.thermal_temperature,
        rtol=3.0e-5,
        atol=3.0e-6,
    )

    cotangent = torch.tensor([[0.2, -0.3, 0.7, -0.5, 0.9]])
    vjp_dense = dense.heat_vjp(organization_dense, query_dense, cotangent)
    vjp_selected = selected.heat_vjp(organization_selected, query_selected, cotangent)
    torch.testing.assert_close(vjp_dense, vjp_selected, rtol=1.0e-4, atol=1.0e-5)
    autograd_vjp_dense = torch.autograd.grad(
        (output_dense.thermal_temperature * cotangent).sum(), controls_dense, retain_graph=True
    )[0]
    autograd_vjp_selected = torch.autograd.grad(
        (output_selected.thermal_temperature * cotangent).sum(), controls_selected, retain_graph=True
    )[0]
    torch.testing.assert_close(autograd_vjp_dense, vjp_dense, rtol=1.0e-4, atol=1.0e-5)
    torch.testing.assert_close(autograd_vjp_selected, vjp_selected, rtol=1.0e-4, atol=1.0e-5)

    parameters_dense = dict(dense.named_parameters())
    parameters_selected = dict(selected.named_parameters())
    parameter_names = (
        "context_blocks.0.em.geometry_score.0.weight",
        "module_value_projection.weight",
        "source_carrier.1.weight",
        "response_phi.0.weight",
    )
    inputs_dense = (
        *(parameters_dense[name] for name in parameter_names),
        scene_dense.module_features,
        scene_dense.environment_features,
        scene_dense.module_coordinates,
        scene_dense.environment_coordinates,
        query_dense.coordinates,
        controls_dense,
    )
    inputs_selected = (
        *(parameters_selected[name] for name in parameter_names),
        scene_selected.module_features,
        scene_selected.environment_features,
        scene_selected.module_coordinates,
        scene_selected.environment_coordinates,
        query_selected.coordinates,
        controls_selected,
    )
    loss_dense = output_dense.nonlinear_field.square().mean() + output_dense.thermal_temperature.square().mean()
    loss_selected = (
        output_selected.nonlinear_field.square().mean() + output_selected.thermal_temperature.square().mean()
    )
    gradients_dense = torch.autograd.grad(loss_dense, inputs_dense)
    gradients_selected = torch.autograd.grad(loss_selected, inputs_selected)
    for actual, expected in zip(gradients_selected, gradients_dense):
        torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-5)


def test_masks_empty_module_set_and_query_subset_order_batch_invariance():
    scene = _scene(empty_second_module_set=True)
    query_mask = torch.tensor([[True, True, False, True, True], [True, True, True, False, True]])
    query = _query(scene, query_mask=query_mask)
    core = _core(scene_executor="selected").eval()
    organization = core.prepare(scene)
    assert torch.equal(
        organization.module_atlas.probabilities[1],
        torch.zeros_like(organization.module_atlas.probabilities[1]),
    )
    assert torch.equal(
        organization.module_anchor_values.values[1],
        torch.zeros_like(organization.module_anchor_values.values[1]),
    )
    full = core.decode(organization, query)
    assert torch.isfinite(full.nonlinear_field).all()
    assert torch.equal(full.nonlinear_field[~query_mask], torch.zeros_like(full.nonlinear_field[~query_mask]))

    order = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
    reordered = core.decode(organization, _select_query(query, order))
    torch.testing.assert_close(reordered.nonlinear_field, full.nonlinear_field.index_select(1, order))
    subset_indices = torch.tensor([1, 4], dtype=torch.long)
    subset = core.decode(organization, _select_query(query, subset_indices))
    torch.testing.assert_close(subset.nonlinear_field, full.nonlinear_field.index_select(1, subset_indices))

    for batch_index in range(scene.batch_size):
        scene_one = _slice_scene(scene, batch_index)
        query_one = _select_query(query, torch.tensor([batch_index]), axis=0)
        output_one = core.decode(core.prepare(scene_one), query_one)
        torch.testing.assert_close(output_one.nonlinear_field, full.nonlinear_field[batch_index : batch_index + 1])


def test_nonlattice_c1_bump_cover_is_a_generic_native_interface():
    scene = _scene(batch=1)
    irregular_positions = torch.tensor(
        [[[0.05, 0.12], [0.91, 0.08], [0.28, 0.87], [1.13, 0.64]]], dtype=torch.float32
    )
    scene = replace(scene, environment_coordinates=irregular_positions)
    coordinates = torch.tensor([[[0.22, 0.25], [0.74, 0.52]]], dtype=torch.float32, requires_grad=True)
    base = _base_query(scene, coordinates=coordinates)
    delta = coordinates[:, :, None, :] - irregular_positions[:, None, :, :]
    squared_distance = delta.square().sum(dim=-1)
    radius_squared = 1.8**2
    bump = torch.relu(1.0 - squared_distance / radius_squared).square()
    dense_weights = bump / bump.sum(dim=-1, keepdim=True)
    permutation = torch.tensor([[[2, 0, 3, 1], [1, 3, 0, 2]]])
    weights = torch.gather(dense_weights, 2, permutation)
    anchor_ids = torch.gather(
        scene.environment_ids[:, None, :].expand(1, 2, 4), 2, permutation
    )
    cover = ReceiverCover(
        anchor_indices=permutation,
        anchor_ids=anchor_ids,
        weights=weights,
        validity=torch.ones_like(weights, dtype=torch.bool),
        query_ids=base.query_ids,
        frame_key=scene.coordinate_frame_key,
        query_mask=base.query_mask,
    )
    cover.validate_partition()
    wrapped = with_receiver_cover(base, cover)
    core = _core().eval()
    output = core(scene, wrapped).nonlinear_field
    grad = torch.autograd.grad(output.square().sum(), coordinates)[0]
    assert torch.isfinite(output).all() and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    assert torch.equal(cover.anchor_ids[0, 0], scene.environment_ids[0, permutation[0, 0]])

    distance = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    edge_bump = torch.relu(1.0 - distance.square()).square()
    edge_derivative = torch.autograd.grad(edge_bump, distance)[0]
    torch.testing.assert_close(edge_derivative, torch.zeros_like(edge_derivative), atol=0, rtol=0)

    bad_ids = cover.anchor_ids.clone()
    bad_ids[0, 0, 0] += 1000
    bad_cover = replace(cover, anchor_ids=bad_ids)
    with pytest.raises(ValueError, match="anchor IDs"):
        core.decode(core.prepare(scene), replace(wrapped, receiver_cover=bad_cover))


def test_grid_cover_masks_outside_padding_but_rejects_outside_live_queries():
    scene = _scene(batch=1)
    coordinates = torch.tensor([[[0.25, 0.25], [1.5, -0.2]]], dtype=torch.float32)
    query = _base_query(
        scene,
        coordinates=coordinates,
        query_mask=torch.tensor([[True, False]]),
    )
    cover = smoothstep_grid_receiver_cover(
        query.coordinates,
        scene.environment_coordinates,
        scene.environment_ids,
        scene.environment_mask,
        query.query_ids,
        frame_key=scene.coordinate_frame_key,
        query_mask=query.query_mask,
        legal_bounds=torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
    )
    assert cover.weights[0, 1].sum() == 0
    assert torch.isfinite(cover.weights).all()
    outside_live = replace(query, query_mask=torch.ones_like(query.query_mask))
    with pytest.raises(ValueError, match="legal domain"):
        smoothstep_grid_receiver_cover(
            outside_live.coordinates,
            scene.environment_coordinates,
            scene.environment_ids,
            scene.environment_mask,
            outside_live.query_ids,
            frame_key=scene.coordinate_frame_key,
            query_mask=outside_live.query_mask,
            legal_bounds=torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        )


def test_affine_heat_reuse_explicit_k_jvp_vjp_and_conditioning_diagnostic():
    scene = _scene(batch=1)
    query = _query(scene)
    core = _core().eval()
    organization = core.prepare(scene)
    prepared = core.prepare_affine_query(organization, query)
    controls = torch.tensor([[0.7, -0.4, 1.1]], dtype=torch.float64)
    kernels = core.export_source_kernels(
        organization, query, accumulation_dtype=torch.float64
    )
    expected = torch.einsum("bqm,bm->bq", kernels, controls)
    direct = core.apply_increment(
        organization, query, controls, accumulation_dtype=torch.float64
    )
    retained = core.apply_prepared_affine(
        prepared, controls, accumulation_dtype=torch.float64
    )
    torch.testing.assert_close(direct, expected, rtol=1.0e-7, atol=1.0e-9)
    torch.testing.assert_close(retained.thermal_temperature, expected, rtol=1.0e-7, atol=1.0e-9)
    zero = core.apply_prepared_affine(prepared, torch.zeros_like(controls), accumulation_dtype=torch.float64)
    assert torch.equal(zero.thermal_temperature, torch.zeros_like(zero.thermal_temperature))
    first = core.apply_prepared_affine(prepared, controls, accumulation_dtype=torch.float64).thermal_temperature
    second_controls = torch.tensor([[-0.2, 0.6, 0.4]], dtype=torch.float64)
    second = core.apply_prepared_affine(prepared, second_controls, accumulation_dtype=torch.float64).thermal_temperature
    summed = core.apply_prepared_affine(
        prepared, controls + second_controls, accumulation_dtype=torch.float64
    ).thermal_temperature
    torch.testing.assert_close(summed, first + second, rtol=1.0e-7, atol=1.0e-9)

    cancellation = torch.zeros_like(controls)
    cancellation[0, 0] = 1.0e8 * kernels[0, 0, 1]
    cancellation[0, 1] = -1.0e8 * kernels[0, 0, 0]
    cancellation_expected = torch.einsum("bqm,bm->bq", kernels, cancellation)
    cancellation_actual = core.apply_prepared_affine(
        prepared, cancellation, accumulation_dtype=torch.float64
    ).thermal_temperature
    cancellation_error = (cancellation_actual - cancellation_expected).abs().max()
    cancellation_scale = (kernels.abs() * cancellation[:, None, :].abs()).sum(dim=-1).max()
    # This extreme signed cancellation is only a conditioning diagnostic. The
    # large contributions are around 1e8 times a kernel coefficient, so its
    # absolute discrepancy is bounded by a few FP64 epsilons of that scale;
    # it is not a pass under the physical-range 1e-9 / 1e-7 comparison above.
    assert cancellation_error <= 5.0e-15 * cancellation_scale + 1.0e-12
    row_scale = (kernels[0, 0, :2].abs() * cancellation[0, :2].abs()).sum()
    assert abs(float(cancellation_actual[0, 0])) < 1.0e-8 * max(float(row_scale), 1.0)

    tangent = torch.randn_like(controls)
    jvp = core.heat_jvp(organization, query, tangent, accumulation_dtype=torch.float64)
    torch.testing.assert_close(jvp, torch.einsum("bqm,bm->bq", kernels, tangent), rtol=1.0e-7, atol=1.0e-9)
    cotangent = torch.randn(1, query.num_queries, dtype=torch.float64)
    vjp = core.heat_vjp_prepared_affine(prepared, cotangent, accumulation_dtype=torch.float64)
    rhs = (tangent * vjp).sum()
    lhs = (jvp * cotangent).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1.0e-7, atol=1.0e-9)

    differentiable_controls = controls.clone().requires_grad_()
    temperature = core.apply_prepared_affine(
        prepared, differentiable_controls, accumulation_dtype=torch.float64
    ).thermal_temperature
    autograd_vjp = torch.autograd.grad((temperature * cotangent).sum(), differentiable_controls)[0]
    torch.testing.assert_close(autograd_vjp, vjp, rtol=1.0e-4, atol=1.0e-5)


def test_prepared_ownership_mutation_and_consumed_graph_guards():
    scene, query = _scene(batch=1), None
    query = _query(scene)

    mode_core = _core().train()
    training_mode_organization = mode_core.prepare(scene)
    mode_core.eval()
    with pytest.raises(ValueError, match="configuration"):
        mode_core.decode(training_mode_organization, query)
    evaluation_mode_organization = mode_core.prepare(scene)
    mode_core.train()
    with pytest.raises(ValueError, match="configuration"):
        mode_core.decode(evaluation_mode_organization, query)

    owner = _core().eval()
    other = _core().eval()
    other.load_state_dict(owner.state_dict())
    organization = owner.prepare(scene)
    with pytest.raises(ValueError, match="another model"):
        other.decode(organization, query)

    scene_stale = owner.prepare(scene)
    scene.module_features[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="scene/atlas/value factors"):
        owner.decode(scene_stale, query)
    scene.module_features[0, 0, 0] -= 0.01

    model_stale = owner.prepare(scene)
    with torch.no_grad():
        next(owner.parameters()).add_(0.001)
    with pytest.raises(ValueError, match="weights or normalization buffers"):
        owner.decode(model_stale, query)
    with torch.no_grad():
        next(owner.parameters()).sub_(0.001)

    buffer_stale = owner.prepare(scene)
    owner.set_inherited_attention_training_epoch(1)
    with pytest.raises(ValueError, match="weights or normalization buffers"):
        owner.decode(buffer_stale, query)
    owner.set_inherited_attention_training_epoch(0)

    cover_stale_org = owner.prepare(scene)
    affine = owner.prepare_affine_query(cover_stale_org, query)
    assert query.receiver_cover is not None
    query.receiver_cover.weights[0, 0, 0] *= 0.9
    with pytest.raises((RuntimeError, ValueError), match="cover|Receiver-cover"):
        owner.apply_prepared_affine(affine, torch.ones(1, 3))

    fresh_scene, fresh_query = _scene(batch=1), None
    fresh_query = _query(fresh_scene)
    training_core = _core().train()
    graph_org = training_core.prepare(fresh_scene)
    training_output = training_core.decode(graph_org, fresh_query)
    training_output.nonlinear_field.square().sum().backward()
    assert graph_org.graph_lifetime.consumed
    with pytest.raises(ValueError, match="graph was consumed"):
        training_core.decode(graph_org, fresh_query)


def test_inference_tensors_cannot_bypass_prepared_mutation_guards():
    core = _core().eval()
    scene = _scene(batch=1)
    with torch.inference_mode(), pytest.raises(ValueError, match="versioned tensors"):
        core.prepare(scene)
    with torch.no_grad():
        organization = core.prepare(scene)
    with torch.inference_mode():
        unversioned_query = _query(scene)
    with pytest.raises(ValueError, match="versioned tensors"):
        core.decode(organization, unversioned_query)
    with torch.no_grad():
        output = core.decode(organization, _query(scene))
    assert torch.isfinite(output.nonlinear_field).all()


def test_prepared_heating_reuses_query_local_and_field_features():
    core = _core().eval()
    scene = _scene(batch=1)
    query = _query(scene)
    calls = {"query": 0, "local": 0, "field": 0}

    def count(name):
        def hook(_module, _inputs, _output):
            calls[name] += 1
        return hook

    handles = [module.register_forward_hook(count(name)) for name, module in (
        ("query", core.query_encoder), ("local", core.local_reader), ("field", core.field_head),
    )]
    try:
        with torch.no_grad():
            organization = core.prepare(scene)
            prepared = core.prepare_affine_query(organization, query)
            first = core.apply_prepared_affine(prepared, torch.ones(1, 3))
            core.apply_prepared_affine(prepared, torch.full((1, 3), 0.5))
        assert calls == {"query": 1, "local": 1, "field": 1}
        assert first.work_counts["local_rows"] == 0
        assert first.work_counts["field_head_rows"] == 0
        assert first.work_counts["cover_executed_anchor_reads"] == (
            query.num_queries * query.receiver_cover.slots * core.num_heads
        )
        with torch.no_grad():
            increment = core.apply_increment(organization, query, torch.ones(1, 3))
        torch.testing.assert_close(increment, first.thermal_temperature)
        assert calls == {"query": 2, "local": 2, "field": 1}
    finally:
        for handle in handles:
            handle.remove()


def test_read_only_association_interventions_rebuild_factors_and_restore_original():
    core = _core().eval()
    scene = _scene(batch=1)
    query = _query(scene)
    with torch.no_grad():
        organization = core.prepare(scene)
        original = core.decode(organization, query, torch.ones(1, 3))
        module_p = organization.module_atlas.normalized_measure[:, None, None, :].expand_as(
            organization.module_atlas.probabilities).clone()
        environment_p = organization.environment_atlas.normalized_measure[:, None, None, :].expand_as(
            organization.environment_atlas.probabilities).clone()
        changed = core.intervene_associations(organization, module_p, environment_p, label="typed_measure_only")
        control = core.decode(changed, query, torch.ones(1, 3))
        restored = core.decode(organization, query, torch.ones(1, 3))
    assert changed.module_values is organization.module_values
    assert changed.source_carriers is organization.source_carriers
    assert not torch.equal(control.nonlinear_field, original.nonlinear_field)
    torch.testing.assert_close(restored.nonlinear_field, original.nonlinear_field, atol=0, rtol=0)
    torch.testing.assert_close(restored.thermal_temperature, original.thermal_temperature, atol=0, rtol=0)
    with pytest.raises(ValueError, match="not a training arm"):
        core.intervene_associations(organization, module_p, environment_p, label="invalid_grad_enabled")
    module_p[0, 0, 0, 0] *= 0.5
    with torch.no_grad(), pytest.raises(ValueError, match="scene/atlas/value factors"):
        core.decode(changed, query)


def _optimizer_step(model, optimizer, scene: SceneBatch, query: InheritedQueryBatch) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(scene, query)
    loss = output.nonlinear_field.square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _assert_optimizer_states_equal(left, right) -> None:
    assert len(left.param_groups) == len(right.param_groups)
    for group_left, group_right in zip(left.param_groups, right.param_groups):
        assert group_left.keys() == group_right.keys()
        for key in group_left.keys() - {"params"}:
            assert group_left[key] == group_right[key]
        assert len(group_left["params"]) == len(group_right["params"])
    for group_left, group_right in zip(left.param_groups, right.param_groups):
        for parameter_left, parameter_right in zip(group_left["params"], group_right["params"]):
            state_left = left.state.get(parameter_left, {})
            state_right = right.state.get(parameter_right, {})
            assert state_left.keys() == state_right.keys()
            for key, left_value in state_left.items():
                right_value = state_right[key]
                if isinstance(left_value, torch.Tensor):
                    torch.testing.assert_close(left_value, right_value, rtol=0, atol=0)
                else:
                    assert left_value == right_value


def test_state_dict_roundtrip_and_same_backend_one_step_resume():
    scene, query = _scene(batch=1), None
    query = _query(scene)
    original = _core().train()
    config = original.get_config()
    restored = InheritedAttentionCoreV1.from_config(config).train()
    restored.load_state_dict(deepcopy(original.state_dict()))
    assert restored.get_config() == config
    original.eval()
    restored.eval()
    output_a = original(scene, query).nonlinear_field
    output_b = restored(scene, query).nonlinear_field
    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)

    original.train()
    optimizer_a = torch.optim.Adam(original.parameters(), lr=1.0e-3)
    _optimizer_step(original, optimizer_a, scene, query)
    saved_model_state = deepcopy(original.state_dict())
    saved_optimizer_state = deepcopy(optimizer_a.state_dict())

    resumed = InheritedAttentionCoreV1.from_config(config).train()
    resumed.load_state_dict(saved_model_state)
    optimizer_b = torch.optim.Adam(resumed.parameters(), lr=1.0e-3)
    optimizer_b.load_state_dict(saved_optimizer_state)
    uninterrupted_loss = _optimizer_step(original, optimizer_a, scene, query)
    resumed_loss = _optimizer_step(resumed, optimizer_b, scene, query)
    assert uninterrupted_loss == resumed_loss
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        original.named_parameters(), resumed.named_parameters()
    ):
        assert name_a == name_b
        torch.testing.assert_close(parameter_a, parameter_b, rtol=0, atol=0)
    _assert_optimizer_states_equal(optimizer_a, optimizer_b)
