from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.readers_v2 import DirectSourceReaderV2
from honf_forward_core.native_retrieval import execution as native_execution
from honf_forward_core.native_retrieval.reader import NativeSelectedReader, _selected_value_rows


def _fixture(seed=4102, *, batch=2, queries=3, modules=5, width=16, heads=4):
    torch.manual_seed(seed)
    dtype = torch.float64
    coordinate_dim = 2
    module_states = torch.randn(batch, modules, width, dtype=dtype, requires_grad=True)
    query_states = torch.randn(batch, queries, width, dtype=dtype, requires_grad=True)
    module_coordinates = torch.randn(batch, modules, coordinate_dim, dtype=dtype, requires_grad=True)
    query_coordinates = torch.randn(batch, queries, coordinate_dim, dtype=dtype, requires_grad=True)
    module_lengths = (torch.rand(batch, modules, coordinate_dim, dtype=dtype) + 0.4).requires_grad_(True)
    measure = (torch.rand(batch, modules, dtype=dtype) + 0.2).requires_grad_(True)
    measure_reference = torch.ones(batch, dtype=dtype, requires_grad=True)
    scene = SceneBatch(
        module_features=torch.zeros(batch, modules, 1, dtype=dtype),
        environment_features=torch.zeros(batch, 0, 1, dtype=dtype),
        module_coordinates=module_coordinates,
        environment_coordinates=torch.zeros(batch, 0, coordinate_dim, dtype=dtype),
        module_mask=torch.ones(batch, modules, dtype=torch.bool),
        environment_mask=torch.zeros(batch, 0, dtype=torch.bool),
        module_measure=measure,
        environment_measure=torch.zeros(batch, 0, dtype=dtype),
        module_ids=torch.arange(modules).expand(batch, -1),
        environment_ids=torch.zeros(batch, 0, dtype=torch.long),
        local_radius=torch.ones(batch, modules, dtype=dtype),
        module_measure_reference=measure_reference,
        module_characteristic_lengths=module_lengths,
        environment_characteristic_lengths=torch.ones(batch, 0, coordinate_dim, dtype=dtype),
    )
    query = QueryBatch(
        coordinates=query_coordinates,
        query_mask=torch.ones(batch, queries, dtype=torch.bool),
        query_ids=torch.arange(queries).expand(batch, -1),
    )
    reader = DirectSourceReaderV2(
        width, coordinate_dim, heads=heads, thermal=True, source_value_conditioning="source_scale"
    ).double()
    keys, values = reader.prepare_sources(module_states)
    organization = SimpleNamespace(
        scene=scene,
        frame_scale=torch.tensor([[2.0, 3.0]] * batch, dtype=dtype, requires_grad=True),
        direct_source_keys=keys,
        direct_source_values=values,
    )
    return reader, organization, query, query_states, module_states


def _plan(permission):
    support = (permission > 0).any(dim=1)
    indices = torch.nonzero(support, as_tuple=False)
    return SimpleNamespace(module_permission=permission, unique_module_indices=indices)


def _native_loss(result):
    response = result.response.explicit_dense_kernel()
    return result.context.square().sum() + 0.03 * result.alpha.square().sum() + 0.07 * response.square().sum()


def test_all_open_recovers_original_a2d_outputs_gradients_and_optimizer_step():
    reader, organization, query, query_states, module_states = _fixture()
    batch, queries, _ = query_states.shape
    modules = organization.scene.num_modules
    permission = torch.ones(batch, reader.heads, queries, modules, dtype=query_states.dtype)
    plan = _plan(permission)
    scores = NativeSelectedReader.score_queries(reader, organization, query, query_states)
    actual = NativeSelectedReader.evaluate(
        reader, organization, query, query_states, plan, scores=scores, return_messages=True,
        measure_reference=organization.scene.module_measure_reference,
    )
    expected_context, expected_alpha, expected_coefficients, expected_messages = reader(
        organization.scene, query, query_states, organization.frame_scale,
        organization.scene.module_measure_reference, organization.direct_source_keys,
        organization.direct_source_values, return_messages=True,
    )
    torch.testing.assert_close(actual.context, expected_context, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(actual.alpha, expected_alpha, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(actual.response.explicit_dense_kernel(), expected_coefficients,
                               atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(actual.messages, expected_messages, atol=1e-10, rtol=1e-9)
    assert actual.counts["fine_selected_rows"] == batch * queries * modules
    assert actual.counts["fine_executed_rows"] == batch * queries * modules
    assert actual.counts["source_full_fastpath"] == 0

    expected_loss = (
        expected_context.square().sum()
        + 0.03 * expected_alpha.square().sum()
        + 0.07 * expected_coefficients.square().sum()
    )
    variables = (
        query_states, module_states, query.coordinates, organization.scene.module_coordinates,
        organization.scene.module_characteristic_lengths, organization.scene.module_measure,
        organization.frame_scale, organization.scene.module_measure_reference,
        *tuple(reader.parameters()),
    )
    expected_gradients = torch.autograd.grad(expected_loss, variables, retain_graph=True, allow_unused=True)
    actual_gradients = torch.autograd.grad(_native_loss(actual), variables, retain_graph=True, allow_unused=True)
    for left, right in zip(actual_gradients, expected_gradients, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=2e-9, rtol=2e-8)

    reference_reader = deepcopy(reader)
    selected_reader = deepcopy(reader)
    reference_states = module_states.detach().clone().requires_grad_(True)
    selected_states = module_states.detach().clone().requires_grad_(True)
    reference_queries = query_states.detach().clone().requires_grad_(True)
    selected_queries = query_states.detach().clone().requires_grad_(True)
    reference_keys, reference_values = reference_reader.prepare_sources(reference_states)
    selected_keys, selected_values = selected_reader.prepare_sources(selected_states)
    reference_context, reference_alpha, reference_response, _ = reference_reader(
        organization.scene, query, reference_queries, organization.frame_scale,
        organization.scene.module_measure_reference, reference_keys, reference_values,
    )
    selected_organization = SimpleNamespace(
        scene=organization.scene, frame_scale=organization.frame_scale,
        direct_source_keys=selected_keys, direct_source_values=selected_values,
    )
    selected_result = NativeSelectedReader.evaluate(
        selected_reader, selected_organization, query, selected_queries, plan,
        measure_reference=organization.scene.module_measure_reference,
    )
    reference_optimizer = torch.optim.SGD(reference_reader.parameters(), lr=1e-3)
    selected_optimizer = torch.optim.SGD(selected_reader.parameters(), lr=1e-3)
    reference_optimizer.zero_grad(set_to_none=True)
    selected_optimizer.zero_grad(set_to_none=True)
    reference_loss = (
        reference_context.square().sum()
        + 0.03 * reference_alpha.square().sum()
        + 0.07 * reference_response.square().sum()
    )
    reference_loss.backward()
    _native_loss(selected_result).backward()
    reference_optimizer.step()
    selected_optimizer.step()
    for left, right in zip(reference_reader.parameters(), selected_reader.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-8)


def test_partial_selected_and_dense_masked_share_outputs_and_live_gate_gradients():
    reader, organization, query, query_states, module_states = _fixture(seed=4141)
    batch, queries, _ = query_states.shape
    modules = organization.scene.num_modules
    gate_values = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0, 0.35, 0.0], [0.0, 0.0, 0.8, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0, 0.7]],
                [[0.3, 0.0, 0.0, 0.0, 0.0], [0.0, 0.6, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.25, 0.0]],
                [[0.0, 0.0, 0.0, 0.9, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5]],
                [[0.0, 0.0, 0.45, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.6], [0.0, 0.3, 0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.7, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.35, 0.0]],
                [[0.0, 0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.7, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.4], [0.0, 0.45, 0.0, 0.0, 0.0], [0.0, 0.0, 0.25, 0.0, 0.0]],
                [[0.6, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 0.0]],
            ],
        ], dtype=query_states.dtype, requires_grad=True,
    )
    plan = _plan(gate_values)
    scores = NativeSelectedReader.score_queries(reader, organization, query, query_states)
    calls = []
    call_shapes = []

    def capture_relative_rows(_module, inputs):
        calls.append(inputs[0].numel() // (7 * reader.coordinate_dim))
        call_shapes.append(tuple(inputs[0].shape))

    handle = reader.relative_value.register_forward_pre_hook(capture_relative_rows)
    try:
        selected = NativeSelectedReader.evaluate(
            reader, organization, query, query_states, plan, scores=scores, backend="selected",
            return_messages=True, measure_reference=organization.scene.module_measure_reference,
        )
        dense = NativeSelectedReader.evaluate(
            reader, organization, query, query_states, plan, scores=scores, backend="dense_masked",
            return_messages=True, measure_reference=organization.scene.module_measure_reference,
        )
    finally:
        handle.remove()
    assert calls == [plan.unique_module_indices.shape[0], batch * queries * modules]
    assert call_shapes[0] == (plan.unique_module_indices.shape[0], 7 * reader.coordinate_dim)
    assert call_shapes[1] == (batch, queries, modules, 7 * reader.coordinate_dim)
    assert selected.counts["fine_executed_rows"] == plan.unique_module_indices.shape[0]
    assert dense.counts["fine_executed_rows"] == batch * queries * modules
    assert dense.counts["native_dense_masked_rows"] == batch * queries * modules
    assert torch.count_nonzero(selected.response.residual_coefficients).item() > 0
    torch.testing.assert_close(selected.context, dense.context, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(selected.alpha, dense.alpha, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(selected.messages, dense.messages, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(selected.response.explicit_dense_kernel(),
                               dense.response.explicit_dense_kernel(), atol=1e-10, rtol=1e-9)
    selected_indices = plan.unique_module_indices
    row_batch, row_query, row_module = selected_indices.unbind(dim=1)
    expected_signed = reader.thermal_output(
        selected.messages[row_batch, row_query, row_module]
    ).squeeze(-1) * gate_values.amax(dim=1)[row_batch, row_query, row_module]
    torch.testing.assert_close(selected.response.residual_coefficients, expected_signed,
                               atol=1e-10, rtol=1e-9)
    assert (expected_signed < 0).any() and (expected_signed > 0).any()
    variables = (
        gate_values, query_states, module_states, query.coordinates,
        organization.scene.module_coordinates, organization.scene.module_characteristic_lengths,
        organization.scene.module_measure, organization.frame_scale,
        *tuple(reader.parameters()),
    )
    selected_grad = torch.autograd.grad(_native_loss(selected), variables, retain_graph=True, allow_unused=True)
    dense_grad = torch.autograd.grad(_native_loss(dense), variables, retain_graph=True, allow_unused=True)
    for left, right in zip(selected_grad, dense_grad, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=2e-9, rtol=2e-8)
    assert selected_grad[0].abs().sum() > 0


def test_selected_native_rows_preserve_distinctions_hidden_by_equal_mean():
    reader, organization, query, query_states, _module_states = _fixture(
        seed=4177, batch=1, queries=1, modules=2
    )
    scene = organization.scene
    indices = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
    scores = NativeSelectedReader.score_queries(reader, organization, query, query_states)
    zero = torch.zeros(reader.width, dtype=query_states.dtype)
    axis_a = torch.linspace(-0.8, 0.8, reader.width, dtype=query_states.dtype)
    axis_b = torch.cos(torch.arange(reader.width, dtype=query_states.dtype)) * 0.8
    values_a = torch.stack((axis_a, -axis_a))[None]
    values_b = torch.stack((axis_b, -axis_b))[None]
    torch.testing.assert_close(values_a.mean(dim=1), values_b.mean(dim=1), atol=0.0, rtol=0.0)
    hidden_a = _selected_value_rows(
        reader, scene, query, query_states, organization.frame_scale, values_a,
        scores.score_descriptor, indices,
    )
    hidden_b = _selected_value_rows(
        reader, scene, query, query_states, organization.frame_scale, values_b,
        scores.score_descriptor, indices,
    )
    assert not torch.allclose(hidden_a, hidden_b)
    assert not torch.allclose(reader.message_output(hidden_a).sum(dim=0),
                              reader.message_output(hidden_b).sum(dim=0))
    # A pre-query group mean sees the same zero source vector in both cases.
    mean_hidden_a = _selected_value_rows(
        reader, scene, query, query_states, organization.frame_scale, zero.reshape(1, 1, -1),
        scores.score_descriptor[:, :, :1], torch.tensor([[0, 0, 0]], dtype=torch.long),
    )
    mean_hidden_b = _selected_value_rows(
        reader, scene, query, query_states, organization.frame_scale, zero.reshape(1, 1, -1),
        scores.score_descriptor[:, :, :1], torch.tensor([[0, 0, 0]], dtype=torch.long),
    )
    torch.testing.assert_close(mean_hidden_a, mean_hidden_b, atol=0.0, rtol=0.0)


def test_empty_axes_and_one_source_have_finite_zero_or_normalized_results():
    for queries, modules in ((2, 0), (0, 3), (2, 1)):
        reader, organization, query, query_states, _module_states = _fixture(
            seed=4200 + queries + modules, batch=2, queries=queries, modules=modules,
            width=8, heads=2,
        )
        permission = torch.ones(2, reader.heads, queries, modules, dtype=query_states.dtype)
        if queries == 0 or modules == 0:
            permission = permission[..., :modules]
        plan = _plan(permission)
        scores = NativeSelectedReader.score_queries(reader, organization, query, query_states)
        result = NativeSelectedReader.evaluate(
            reader, organization, query, query_states, plan, scores=scores, return_messages=True
        )
        assert torch.isfinite(result.context).all()
        assert torch.isfinite(result.alpha).all()
        assert result.alpha.shape == (2, queries, modules, reader.heads)
        assert result.messages.shape == (2, queries, modules, reader.width)
        assert result.response.explicit_dense_kernel().shape == (2, queries, modules)
        if modules == 0 or queries == 0:
            assert torch.count_nonzero(result.context) == 0
            assert result.counts["fine_executed_rows"] == 0
        else:
            torch.testing.assert_close(
                result.alpha.sum(dim=2), torch.ones(2, queries, reader.heads, dtype=query_states.dtype)
            )
            assert result.counts["fine_executed_rows"] == 2 * queries


def test_zero_measure_present_module_keeps_native_signed_response_but_no_information_weight():
    reader, organization, query, query_states, _module_states = _fixture(
        seed=4209, batch=1, queries=2, modules=3, width=8, heads=2
    )
    measure = organization.scene.module_measure.clone()
    measure[:, 0] = 0.0
    scene = replace(organization.scene, module_measure=measure)
    organization = SimpleNamespace(
        scene=scene,
        frame_scale=organization.frame_scale,
        direct_source_keys=organization.direct_source_keys,
        direct_source_values=organization.direct_source_values,
    )
    permission = torch.ones(1, reader.heads, 2, 3, dtype=query_states.dtype)
    plan = _plan(permission)
    result = NativeSelectedReader.evaluate(
        reader, organization, query, query_states, plan,
        scores=NativeSelectedReader.score_queries(reader, organization, query, query_states),
        measure_reference=scene.module_measure_reference,
    )
    expected_context, expected_alpha, expected_coefficients, _ = reader(
        scene, query, query_states, organization.frame_scale, scene.module_measure_reference,
        organization.direct_source_keys, organization.direct_source_values,
    )
    torch.testing.assert_close(result.context, expected_context, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(result.alpha, expected_alpha, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(result.response.explicit_dense_kernel(), expected_coefficients,
                               atol=1e-10, rtol=1e-9)
    assert torch.count_nonzero(result.alpha[:, :, 0, :]) == 0
    assert torch.count_nonzero(result.response.explicit_dense_kernel()[:, :, 0]) > 0
    assert result.counts["fine_selected_rows"] == 6


def test_heterogeneous_padded_modules_and_batch_companion_do_not_change_a_scene():
    reader, organization, query, query_states, _module_states = _fixture(seed=4211)
    batch, queries, _ = query_states.shape
    modules = organization.scene.num_modules
    mask = torch.tensor(
        [[True, True, True, False, False], [True, False, False, False, False]],
        dtype=torch.bool,
    )
    scene = replace(organization.scene, module_mask=mask)
    organization = SimpleNamespace(
        scene=scene, frame_scale=organization.frame_scale,
        direct_source_keys=organization.direct_source_keys,
        direct_source_values=organization.direct_source_values,
    )
    permission = mask[:, None, None, :].expand(batch, reader.heads, queries, modules).to(query_states.dtype)
    plan = _plan(permission)
    result = NativeSelectedReader.evaluate(
        reader, organization, query, query_states, plan,
        scores=NativeSelectedReader.score_queries(reader, organization, query, query_states),
    )
    assert result.counts["fine_selected_rows"] == queries * (3 + 1)
    torch.testing.assert_close(
        result.alpha.sum(dim=2), torch.ones(batch, queries, reader.heads, dtype=query_states.dtype)
    )

    scene_values = {}
    for field in fields(scene):
        value = getattr(scene, field.name)
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
            scene_values[field.name] = value[:1]
    single_scene = replace(scene, **scene_values)
    single_query = replace(
        query, coordinates=query.coordinates[:1], query_mask=query.query_mask[:1], query_ids=query.query_ids[:1]
    )
    single_organization = SimpleNamespace(
        scene=single_scene,
        frame_scale=organization.frame_scale[:1],
        direct_source_keys=organization.direct_source_keys[:1],
        direct_source_values=organization.direct_source_values[:1],
    )
    single_permission = permission[:1]
    single_plan = _plan(single_permission)
    single = NativeSelectedReader.evaluate(
        reader, single_organization, single_query, query_states[:1], single_plan,
        scores=NativeSelectedReader.score_queries(
            reader, single_organization, single_query, query_states[:1]
        ),
    )
    torch.testing.assert_close(single.context, result.context[:1], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(single.alpha, result.alpha[:1], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        single.response.explicit_dense_kernel(), result.response.explicit_dense_kernel()[:1],
        atol=1e-10, rtol=1e-9,
    )


def test_repeated_query_ids_permutation_and_chunking_preserve_reader_outputs():
    reader, organization, query, query_states, _module_states = _fixture(seed=4227)
    batch, queries, _ = query_states.shape
    modules = organization.scene.num_modules
    query_ids = torch.tensor([[17, 17, 42], [8, 8, 8]], dtype=torch.long)
    query = replace(query, query_ids=query_ids)
    permission = torch.zeros(batch, reader.heads, queries, modules, dtype=query_states.dtype)
    permission[:, 0, :, 0] = torch.tensor([[0.4, 0.7, 0.5], [0.3, 0.45, 0.8]], dtype=query_states.dtype)
    permission[:, 0, :, 2] = torch.tensor([[0.8, 0.0, 0.35], [0.0, 0.65, 0.0]], dtype=query_states.dtype)
    permission[:, 1, :, 1] = torch.tensor([[0.7, 0.5, 0.0], [0.6, 0.35, 0.0]], dtype=query_states.dtype)
    permission[:, 1, :, 3] = torch.tensor([[0.0, 0.4, 0.75], [0.0, 0.0, 0.45]], dtype=query_states.dtype)
    full = NativeSelectedReader.evaluate(
        reader, organization, query, query_states, _plan(permission),
        scores=NativeSelectedReader.score_queries(reader, organization, query, query_states),
        return_messages=True,
    )

    permutation = torch.tensor([2, 0, 1])
    inverse = torch.tensor([1, 2, 0])
    permuted_query = replace(
        query, coordinates=query.coordinates[:, permutation],
        query_mask=query.query_mask[:, permutation], query_ids=query.query_ids[:, permutation],
    )
    permuted_states = query_states[:, permutation]
    permuted_permission = permission[:, :, permutation]
    permuted = NativeSelectedReader.evaluate(
        reader, organization, permuted_query, permuted_states, _plan(permuted_permission),
        scores=NativeSelectedReader.score_queries(reader, organization, permuted_query, permuted_states),
        return_messages=True,
    )
    torch.testing.assert_close(permuted.context[:, inverse], full.context, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(permuted.alpha[:, inverse], full.alpha, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(permuted.messages[:, inverse], full.messages, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        permuted.response.explicit_dense_kernel()[:, inverse],
        full.response.explicit_dense_kernel(), atol=1e-10, rtol=1e-9,
    )

    chunk_results = []
    for start, stop in ((0, 1), (1, 3)):
        chunk_query = replace(
            query, coordinates=query.coordinates[:, start:stop],
            query_mask=query.query_mask[:, start:stop], query_ids=query.query_ids[:, start:stop],
        )
        chunk_states = query_states[:, start:stop]
        chunk_permission = permission[:, :, start:stop]
        chunk_results.append(NativeSelectedReader.evaluate(
            reader, organization, chunk_query, chunk_states, _plan(chunk_permission),
            scores=NativeSelectedReader.score_queries(reader, organization, chunk_query, chunk_states),
            return_messages=True,
        ))
    torch.testing.assert_close(torch.cat([part.context for part in chunk_results], dim=1),
                               full.context, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(torch.cat([part.alpha for part in chunk_results], dim=1),
                               full.alpha, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(torch.cat([part.messages for part in chunk_results], dim=1),
                               full.messages, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(torch.cat([
        part.response.explicit_dense_kernel() for part in chunk_results
    ], dim=1), full.response.explicit_dense_kernel(), atol=1e-10, rtol=1e-9)


def test_fp64_gradcheck_and_gradgradcheck_for_fractional_gates_with_fixed_support():
    reader, organization, query, query_states, _module_states = _fixture(
        seed=4242, batch=1, queries=2, modules=3, width=8, heads=2
    )
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    # Treat the prepared original A2-D projections and inputs as fixed; only
    # admitted fractional permission values are differentiated here.
    detached_scene_values = {}
    for field in fields(organization.scene):
        value = getattr(organization.scene, field.name)
        if isinstance(value, torch.Tensor):
            detached_scene_values[field.name] = value.detach()
    scene = replace(organization.scene, **detached_scene_values)
    detached_query = replace(query, coordinates=query.coordinates.detach())
    detached_states = query_states.detach()
    fixed_organization = SimpleNamespace(
        scene=scene,
        frame_scale=organization.frame_scale.detach(),
        direct_source_keys=organization.direct_source_keys.detach(),
        direct_source_values=organization.direct_source_values.detach(),
    )
    scores = NativeSelectedReader.score_queries(reader, fixed_organization, detached_query, detached_states)
    support = torch.tensor(
        [[[[True, True, False], [False, True, True]],
          [[True, False, True], [True, True, False]]]], dtype=torch.bool,
    )
    positions = torch.nonzero(support, as_tuple=False)
    unique = torch.nonzero(support.any(dim=1), as_tuple=False)
    template = torch.zeros_like(support, dtype=torch.float64)
    initial = torch.tensor(
        [0.31, 0.63, 0.42, 0.27, 0.54, 0.78, 0.46, 0.36],
        dtype=torch.float64, requires_grad=True,
    )
    assert initial.numel() == positions.shape[0]

    def gate_outputs(values):
        full_permission = template.index_put(tuple(positions.unbind(dim=1)), values)
        plan = SimpleNamespace(module_permission=full_permission, unique_module_indices=unique)
        result = NativeSelectedReader.evaluate(
            reader, fixed_organization, detached_query, detached_states, plan, scores=scores,
        )
        return torch.cat((result.context.reshape(-1), result.alpha.reshape(-1),
                          result.response.residual_coefficients))

    assert torch.autograd.gradcheck(gate_outputs, (initial,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(gate_outputs, (initial,), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_tensor_value_reduction_matches_selected_reader_and_compiled_dispatch(monkeypatch):
    reader, organization, query, query_states, _module_states = _fixture(seed=4253)
    batch, queries, _ = query_states.shape
    modules = organization.scene.num_modules
    permission = torch.zeros(batch, reader.heads, queries, modules, dtype=query_states.dtype)
    permission[:, 0, :, 0] = 0.6
    permission[:, 1, :, 2] = 0.25
    permission[:, 2, :, 3] = 0.8
    plan = _plan(permission)
    scores = NativeSelectedReader.score_queries(reader, organization, query, query_states)

    with torch.no_grad():
        eager = NativeSelectedReader.evaluate(
            reader, organization, query, query_states, plan, scores=scores,
            backend="selected", return_messages=True,
        )
        b, q, m = plan.unique_module_indices.unbind(dim=1)
        reduced_context, packed_messages, coefficients = native_execution.selected_value_reduction(
            query.coordinates.to(query_states.dtype),
            organization.scene.module_coordinates.to(query_states.dtype),
            organization.frame_scale,
            reader.query_value(query_states),
            organization.direct_source_values,
            scores.score_descriptor,
            plan.unique_module_indices,
            eager.alpha,
            permission.amax(dim=1)[b, q, m],
            reader.relative_value.weight,
            reader.message_output.weight,
            reader.message_output.bias,
            reader.thermal_output.weight,
            reader.thermal_output.bias,
            reader.source_scale_projection.weight,
        )
        torch.testing.assert_close(reduced_context, eager.context, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(packed_messages, eager.messages[b, q, m], atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(
            coefficients, eager.response.residual_coefficients, atol=1e-10, rtol=1e-9
        )

        dispatch_calls = []

        def eager_compiled_stub(*args):
            dispatch_calls.append(tuple(args[6].shape))
            return native_execution.selected_value_reduction(*args)

        monkeypatch.setattr(native_execution, "_compiled_selected_value_reduction", eager_compiled_stub)
        dispatched = NativeSelectedReader.evaluate(
            reader, organization, query, query_states, plan, scores=scores,
            backend="compiled_selected", return_messages=True,
        )
    assert dispatch_calls == [(plan.unique_module_indices.shape[0], 3)]
    assert dispatched.backend == "compiled_selected"
    assert dispatched.counts["native_compiled_selected_backend"] == 1
    assert dispatched.counts["native_selected_backend"] == 0
    torch.testing.assert_close(dispatched.context, eager.context, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(dispatched.alpha, eager.alpha, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(dispatched.messages, eager.messages, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        dispatched.response.explicit_dense_kernel(), eager.response.explicit_dense_kernel(),
        atol=1e-10, rtol=1e-9,
    )

    # A configured compiled backend stays differentiable by selecting the eager
    # native implementation whenever autograd is enabled.
    gradient_path = NativeSelectedReader.evaluate(
        reader, organization, query, query_states, plan, scores=scores,
        backend="compiled_selected",
    )
    assert gradient_path.backend == "selected"
    assert gradient_path.counts["native_compiled_selected_backend"] == 0
    assert gradient_path.counts["native_selected_backend"] == 1
    assert torch.autograd.grad(gradient_path.context.square().sum(), query_states)[0] is not None
