from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.readers_v2 import DirectSourceReaderV2
from honf_forward_core.native_retrieval.budgeted_reader import BudgetedNativeReader, dense_hard_router_gate


def _fixture(seed=7221, *, batch=2, queries=5, modules=4, width=16, heads=4):
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


def _literal_a2d(reader, organization, query, query_states, support):
    """Small independent FP64 token/head loop for the original native function."""
    scene = organization.scene
    batch, queries, _ = query_states.shape
    modules = scene.num_modules
    contexts_by_batch = []
    alphas_by_batch = []
    coefficients_by_batch = []
    for b in range(batch):
        contexts_by_query = []
        alphas_by_query = []
        coefficients_by_query = []
        for q in range(queries):
            query_key = reader.query_key(reader.query_norm(query_states[b, q])).reshape(reader.heads, reader.head_width)
            query_key = F.normalize(query_key, dim=-1, eps=1.0e-6)
            scores_by_module = []
            for m in range(modules):
                source_key = organization.direct_source_keys[b, m].reshape(reader.heads, reader.head_width)
                source_key = F.normalize(source_key, dim=-1, eps=1.0e-6)
                delta = query.coordinates[b, q] - scene.module_coordinates[b, m]
                normalized = delta / organization.frame_scale[b]
                distance = torch.linalg.vector_norm(normalized).reshape(1)
                local = torch.asinh(delta / scene.module_characteristic_lengths[b, m])
                descriptor = torch.cat((normalized, distance, local), dim=-1)
                geometry_score = reader.geometry_score(descriptor).reshape(reader.heads)
                scores_by_module.append((query_key * source_key).sum(dim=-1) * reader.temperature + geometry_score)
            scores = torch.stack(scores_by_module, dim=-1)
            info_valid = scene.module_mask[b] & (scene.module_measure[b] > 0)
            alpha_by_head = []
            for h in range(reader.heads):
                eligible_terms = [
                    (scene.module_measure[b, m] / scene.module_measure_reference[b]) * torch.exp(scores[h, m])
                    if bool(support[b, q, m] and info_valid[m] and query.query_mask[b, q])
                    else query_states.new_zeros(())
                    for m in range(modules)
                ]
                denominator = torch.stack(eligible_terms).sum()
                alpha_by_head.append(torch.stack([
                    term / denominator if bool(denominator > 0) else query_states.new_zeros(())
                    for term in eligible_terms
                ]))
            alpha_q = torch.stack(alpha_by_head, dim=-1)
            alphas_by_query.append(alpha_q)
            message_rows = []
            coefficient_rows = []
            for m in range(modules):
                delta = query.coordinates[b, q] - scene.module_coordinates[b, m]
                normalized = delta / organization.frame_scale[b]
                pieces = [normalized]
                for frequency in (1.0, 2.0, 4.0):
                    phase = (2.0 * torch.pi * frequency) * normalized
                    pieces.extend((torch.sin(phase), torch.cos(phase)))
                gamma = torch.cat(pieces, dim=-1)
                relative = reader.relative_value(gamma)
                source_scale = torch.asinh(delta / scene.module_characteristic_lengths[b, m])
                relative = relative + reader.source_scale_projection(source_scale)
                hidden = F.silu(
                    reader.query_value(query_states[b, q])
                    + organization.direct_source_values[b, m]
                    + relative
                )
                message = reader.message_output(hidden)
                message_rows.append(message)
                coefficient_rows.append(
                    reader.thermal_output(message).squeeze()
                    * support[b, q, m].to(query_states.dtype)
                    * query.query_mask[b, q].to(query_states.dtype)
                    * scene.module_mask[b, m].to(query_states.dtype)
                )
            message_matrix = torch.stack(message_rows)
            weighted_heads = []
            for h in range(reader.heads):
                weighted_heads.append(
                    torch.stack([
                        message_matrix[m].reshape(reader.heads, reader.head_width)[h] * alpha_q[m, h]
                        for m in range(modules)
                    ]).sum(dim=0)
                )
            contexts_by_query.append(torch.cat(weighted_heads))
            coefficients_by_query.append(torch.stack(coefficient_rows))
        contexts_by_batch.append(torch.stack(contexts_by_query))
        alphas_by_batch.append(torch.stack(alphas_by_query))
        coefficients_by_batch.append(torch.stack(coefficients_by_query))
    return torch.stack(contexts_by_batch), torch.stack(alphas_by_batch), torch.stack(coefficients_by_batch)


def _as_dense_response(result):
    return result.response.explicit_dense_kernel()


def test_dense_hard_reference_matches_independent_fp64_literal_and_fixed_support_grads():
    reader, organization, query, query_states, module_states = _fixture()
    support = torch.tensor(
        [
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
            [[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
        ],
        dtype=torch.bool,
    )
    actual = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, support, backend="dense_hard_reference", return_messages=True
    )
    expected_context, expected_alpha, expected_coefficients = _literal_a2d(
        reader, organization, query, query_states, support
    )
    torch.testing.assert_close(actual.context, expected_context, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(actual.alpha, expected_alpha, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(_as_dense_response(actual), expected_coefficients, atol=2e-12, rtol=2e-11)
    assert actual.counts["native_useful_value_rows"] == int(support.sum())
    assert actual.counts["native_value_rows"] == support.numel()
    assert actual.counts["native_dense_value_rows"] == support.numel()
    assert actual.messages.shape == (*support.shape, reader.width)

    # Fixed-support VJPs must preserve the original nonlinear function for
    # every reader parameter and every differentiable physical/model input.
    torch.manual_seed(7222)
    context_cotangent = torch.randn_like(actual.context)
    alpha_cotangent = torch.randn_like(actual.alpha)
    response_cotangent = torch.randn_like(expected_coefficients)
    actual_loss = (
        (actual.context * context_cotangent).sum()
        + (actual.alpha * alpha_cotangent).sum()
        + (_as_dense_response(actual) * response_cotangent).sum()
    )
    literal_loss = (
        (expected_context * context_cotangent).sum()
        + (expected_alpha * alpha_cotangent).sum()
        + (expected_coefficients * response_cotangent).sum()
    )
    differentiable_inputs = (
        query_states,
        query.coordinates,
        organization.scene.module_coordinates,
        organization.scene.module_characteristic_lengths,
        organization.frame_scale,
        organization.scene.module_measure,
        organization.scene.module_measure_reference,
        module_states,
        *tuple(reader.parameters()),
    )
    actual_grads = torch.autograd.grad(actual_loss, differentiable_inputs, retain_graph=True, allow_unused=True)
    literal_grads = torch.autograd.grad(literal_loss, differentiable_inputs, allow_unused=True)
    assert len(actual_grads) == len(literal_grads)
    for actual_grad, literal_grad in zip(actual_grads, literal_grads, strict=True):
        assert (actual_grad is None) == (literal_grad is None)
        if actual_grad is not None:
            assert torch.isfinite(actual_grad).all() and torch.isfinite(literal_grad).all()
            torch.testing.assert_close(actual_grad, literal_grad, atol=2e-10, rtol=2e-9)

    # Only compare at all-open support here; this is an independent check that
    # the original full reader boundary remains exact.
    all_open = torch.ones_like(support)
    all_open_actual = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, all_open, backend="dense_hard_reference"
    )
    expected_context_native, expected_alpha_native, expected_coefficients_native, _ = reader(
        organization.scene, query, query_states, organization.frame_scale,
        organization.scene.module_measure_reference, organization.direct_source_keys,
        organization.direct_source_values, return_messages=True,
    )
    torch.testing.assert_close(all_open_actual.context, expected_context_native, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(all_open_actual.alpha, expected_alpha_native, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(_as_dense_response(all_open_actual), expected_coefficients_native,
                               atol=2e-12, rtol=2e-11)


def test_dense_router_estimator_keeps_hard_zero_and_restorative_derivative():
    reader, organization, query, query_states, _ = _fixture(batch=1, queries=2, modules=3)
    protected = torch.zeros((1, 2, 3), dtype=torch.bool)
    hard_membership = torch.tensor([[[1, 0, 1], [1, 1, 0], [0, 1, 1]]], dtype=torch.float64)
    membership_soft = torch.full_like(hard_membership, 0.35, requires_grad=True)
    membership_st = hard_membership + (membership_soft - membership_soft.detach())
    hard_dispatch = F.one_hot(torch.tensor([[0, 1]]), num_classes=3).to(torch.float64)
    dispatch_soft = torch.tensor(
        [[[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]]], dtype=torch.float64, requires_grad=True
    )
    dispatch_st = hard_dispatch + (dispatch_soft - dispatch_soft.detach())
    gate = dense_hard_router_gate(protected, dispatch_st, membership_st)
    expected_support = torch.tensor([[[1, 0, 1], [1, 1, 0]]], dtype=torch.float64)
    torch.testing.assert_close(gate.detach(), expected_support, atol=0, rtol=0)
    result = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, gate, backend="dense_hard_reference"
    )
    # The target is the dense all-open result, so at least one omitted member
    # has a signed restoration direction under the hard-forward estimator.
    all_open = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states,
        torch.ones((1, 2, 3), dtype=torch.bool), backend="dense_hard_reference",
    )
    loss = F.mse_loss(result.context, all_open.context.detach())
    gradient = torch.autograd.grad(loss, membership_soft)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    omitted = hard_membership == 0
    assert torch.count_nonzero(gradient[omitted]) > 0
    restorative = omitted & (gradient < 0)
    assert torch.count_nonzero(restorative) > 0
    # Turning on the most useful omitted native member moves the actual hard
    # field toward the all-open target, matching the estimator's local sign.
    batch_index, seed_index, module_index = torch.nonzero(restorative, as_tuple=True)
    restored_membership = hard_membership.clone()
    restored_membership[batch_index[0], seed_index[0], module_index[0]] = 1.0
    restored_st = restored_membership + (membership_soft - membership_soft.detach())
    restored_gate = dense_hard_router_gate(protected, dispatch_st, restored_st)
    restored_result = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, restored_gate, backend="dense_hard_reference"
    )
    restored_loss = F.mse_loss(restored_result.context, all_open.context.detach())
    assert restored_loss < loss.detach()


def test_dense_hard_reference_rejects_fractional_forward_gates():
    reader, organization, query, query_states, _ = _fixture(batch=1, queries=2, modules=3)
    gate = torch.full((1, 2, 3), 0.5, dtype=query_states.dtype)
    with pytest.raises(ValueError, match="binary hard forward"):
        BudgetedNativeReader.evaluate(
            reader, organization, query, query_states, gate, backend="dense_hard_reference"
        )


def test_rectangular_and_grouped_padding_do_not_leak_invalid_nan_values():
    reader, organization, query, query_states, _ = _fixture(batch=1, queries=2, modules=3)
    module_mask = torch.tensor([[True, True, False]])
    scene = replace(organization.scene, module_mask=module_mask)
    query_mask = torch.tensor([[True, False]])
    query = replace(query, query_mask=query_mask)
    query_states = query_states.clone()
    query_states[:, 1] = torch.nan
    source_values = organization.direct_source_values.clone()
    source_values[:, 2] = torch.nan
    source_keys = organization.direct_source_keys.clone()
    source_keys[:, 2] = torch.nan
    organization = SimpleNamespace(
        scene=scene,
        frame_scale=organization.frame_scale,
        direct_source_keys=source_keys,
        direct_source_values=source_values,
    )
    support = torch.tensor([[[True, True, False], [False, False, False]]])

    graph = SimpleNamespace(
        member_indices=torch.tensor([[[0, 1, 2], [0, 1, 2], [2, 0, 1]]]),
        member_valid=torch.tensor([[[True, True, False], [True, True, False], [True, False, False]]]),
        seed_valid=torch.tensor([[True, True, False]]),
    )
    dispatch_plan = SimpleNamespace(
        dispatch_seed=torch.tensor([[0, -1]]),
        tie_indeterminate=torch.zeros((1, 2), dtype=torch.bool),
    )
    protected = torch.zeros_like(support)
    results = [
        BudgetedNativeReader.evaluate(
            reader, organization, query, query_states, support, backend="dense_hard_reference"
        ),
        BudgetedNativeReader.evaluate(
            reader, organization, query, query_states, support, backend="flat_selected_reference"
        ),
        BudgetedNativeReader.evaluate(
            reader, organization, query, query_states, support,
            backend="grouped_native", graph=graph, dispatch_plan=dispatch_plan,
            protected_mask=protected, query_tile_size=1,
        ),
    ]
    for result in results:
        assert torch.isfinite(result.context).all()
        assert torch.isfinite(result.alpha).all()
        assert torch.isfinite(result.response.explicit_dense_kernel()).all()
    torch.testing.assert_close(results[1].context, results[0].context)
    torch.testing.assert_close(results[2].context, results[0].context)


def test_empty_query_and_empty_native_module_axes_return_empty_reads():
    for reader, organization, query, query_states, _ in (
        _fixture(batch=1, queries=0, modules=3),
        _fixture(batch=1, queries=2, modules=0),
    ):
        batch, queries = query_states.shape[:2]
        modules = organization.scene.num_modules
        support = torch.zeros((batch, queries, modules), dtype=torch.bool)
        graph = SimpleNamespace(
            member_indices=torch.zeros((batch, modules, modules), dtype=torch.long),
            member_valid=torch.zeros((batch, modules, modules), dtype=torch.bool),
            seed_valid=torch.zeros((batch, modules), dtype=torch.bool),
        )
        dispatch_plan = SimpleNamespace(
            dispatch_seed=torch.zeros((batch, queries), dtype=torch.long),
            tie_indeterminate=torch.zeros((batch, queries), dtype=torch.bool),
        )
        for backend in ("dense_hard_reference", "flat_selected_reference", "grouped_native"):
            result = BudgetedNativeReader.evaluate(
                reader, organization, query, query_states, support,
                backend=backend, graph=graph, dispatch_plan=dispatch_plan,
                protected_mask=support, return_messages=True,
            )
            assert result.context.shape == (batch, queries, reader.width)
            assert result.alpha.shape == (batch, queries, modules, reader.heads)
            assert result.messages.shape == (batch, queries, modules, reader.width)
            assert result.response.explicit_dense_kernel().shape == (batch, queries, modules)
            assert torch.count_nonzero(result.context) == 0


def test_flat_and_grouped_references_match_with_padding_protection_and_source_order():
    reader, organization, query, query_states, module_states = _fixture(batch=2, queries=5, modules=4)
    batch, queries, modules = 2, 5, 4
    seeds = modules
    member_mask = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
        ],
        dtype=torch.bool,
    )
    member_ids = torch.arange(modules).reshape(1, 1, modules).expand(batch, seeds, modules).clone()
    member_valid = member_mask.clone()
    dispatch = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]], dtype=torch.long)
    protected = torch.zeros((batch, queries, modules), dtype=torch.bool)
    protected[0, 0, 2] = True  # protected exception extends seed 0's union list
    protected[1, 3, 3] = True
    selected_members = member_mask.gather(
        1, dispatch[:, :, None].expand(batch, queries, modules)
    )
    support = (selected_members | protected) & organization.scene.module_mask[:, None, :]
    graph = SimpleNamespace(
        member_indices=member_ids,
        member_valid=member_valid,
        seed_valid=torch.ones((batch, seeds), dtype=torch.bool),
    )
    dispatch_plan = SimpleNamespace(dispatch_seed=dispatch, tie_indeterminate=torch.zeros((batch, queries), dtype=torch.bool))

    flat = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, support,
        backend="flat_selected_reference", return_messages=True,
    )
    grouped = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, support,
        backend="grouped_native", graph=graph, dispatch_plan=dispatch_plan,
        protected_mask=protected, return_messages=True, query_tile_size=2,
    )
    dense = BudgetedNativeReader.evaluate(
        reader, organization, query, query_states, support,
        backend="dense_hard_reference", return_messages=True,
    )
    torch.testing.assert_close(grouped.context, flat.context, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(grouped.context, dense.context, atol=2e-11, rtol=2e-10)
    torch.testing.assert_close(grouped.alpha, flat.alpha, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(_as_dense_response(grouped), _as_dense_response(flat), atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(grouped.messages, flat.messages, atol=2e-12, rtol=2e-11)
    assert grouped.counts["native_useful_value_rows"] == int(support.sum())
    assert grouped.counts["native_value_rows"] >= grouped.counts["native_useful_value_rows"]
    assert grouped.counts["native_padded_rows"] == (
        grouped.counts["native_value_rows"] - grouped.counts["native_useful_value_rows"]
    )
    assert grouped.counts["native_dispatch_rows"] == batch * queries
    assert grouped.counts["native_dispatch_groups"] == batch * seeds
    assert grouped.counts["native_protection_exception_rows"] == 2
    assert grouped.counts["unique_nonlinear_rows"] == grouped.counts["native_value_rows"]
    assert grouped.counts["native_unique_nonlinear_rows"] == int(support.sum())
    assert flat.counts["native_value_rows"] == int(support.sum())
    assert dense.counts["native_value_rows"] == batch * queries * modules
    assert dense.counts["native_valid_value_rows"] == batch * queries * modules
    assert dense.counts["fine_scalar_projection_rows"] == batch * queries * modules
    for b in range(batch):
        for seed in range(seeds):
            ids = grouped.trace["native_group_member_ids"][b, seed][
                grouped.trace["native_group_member_valid"][b, seed]
            ]
            assert torch.equal(ids, torch.sort(ids).values)

    # The grouped and flat executors retain the fixed-support value VJP, even
    # though their forward row packing and reduction order differ.
    torch.manual_seed(7223)
    context_cotangent = torch.randn_like(dense.context)
    alpha_cotangent = torch.randn_like(dense.alpha)
    response_cotangent = torch.randn_like(_as_dense_response(dense))

    def scalar_loss(result):
        return (
            (result.context * context_cotangent).sum()
            + (result.alpha * alpha_cotangent).sum()
            + (_as_dense_response(result) * response_cotangent).sum()
        )

    differentiable_inputs = (
        query_states,
        query.coordinates,
        organization.scene.module_coordinates,
        organization.scene.module_characteristic_lengths,
        organization.frame_scale,
        organization.scene.module_measure,
        module_states,
        *tuple(reader.parameters()),
    )
    dense_grads = torch.autograd.grad(scalar_loss(dense), differentiable_inputs, retain_graph=True, allow_unused=True)
    flat_grads = torch.autograd.grad(scalar_loss(flat), differentiable_inputs, retain_graph=True, allow_unused=True)
    grouped_grads = torch.autograd.grad(scalar_loss(grouped), differentiable_inputs, retain_graph=True, allow_unused=True)
    for expected_grads in (flat_grads, grouped_grads):
        for expected_grad, dense_grad in zip(expected_grads, dense_grads, strict=True):
            assert (expected_grad is None) == (dense_grad is None)
            if dense_grad is not None:
                assert torch.isfinite(expected_grad).all() and torch.isfinite(dense_grad).all()
                torch.testing.assert_close(expected_grad, dense_grad, atol=3e-10, rtol=3e-9)

    permutation = torch.tensor([4, 0, 3, 1, 2])
    permuted_query = QueryBatch(
        coordinates=query.coordinates[:, permutation],
        query_mask=query.query_mask[:, permutation],
        query_ids=query.query_ids[:, permutation],
    )
    permuted = BudgetedNativeReader.evaluate(
        reader, organization, permuted_query, query_states[:, permutation], support[:, permutation],
        backend="grouped_native", graph=graph,
        dispatch_plan=SimpleNamespace(
            dispatch_seed=dispatch[:, permutation], tie_indeterminate=torch.zeros((batch, queries), dtype=torch.bool)
        ),
        protected_mask=protected[:, permutation], query_tile_size=2,
    )
    torch.testing.assert_close(permuted.context[:, torch.argsort(permutation)], grouped.context, atol=2e-11, rtol=2e-10)
