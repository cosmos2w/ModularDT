from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2, QueryBatch, SceneBatch
from honf_forward_core.content_hypergraph.source_read_execution import (
    compose_scalar_projection,
    reduce_headwise_linear,
    reduce_indexed_linear,
)


def _scene(*, requires_grad: bool = False) -> SceneBatch:
    module_coordinates = torch.tensor([[[0.0, 0.0], [0.45, 0.2], [0.9, 0.4]]])
    if requires_grad:
        module_coordinates.requires_grad_()
    return SceneBatch(
        module_features=torch.tensor([[[0.2, 0.4], [0.7, -0.3], [-0.1, 0.8]]]),
        environment_features=torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, -0.2, 0.5], [-0.5, 0.7, 0.1], [0.8, 0.1, -0.4]]]
        ),
        module_coordinates=module_coordinates,
        environment_coordinates=torch.tensor([[[-0.1, 0.3], [0.35, 0.1], [0.7, 0.5], [1.1, 0.2]]]),
        module_mask=torch.tensor([[True, False, True]]),
        environment_mask=torch.ones((1, 4), dtype=torch.bool),
        module_measure=torch.tensor([[0.7, 0.0, 1.1]]),
        environment_measure=torch.tensor([[0.2, 0.4, 0.3, 0.5]]),
        module_ids=torch.tensor([[13, 29, 47]]),
        environment_ids=torch.tensor([[101, 107, 131, 149]]),
        local_radius=torch.full((1, 3), 0.7),
        operating_features=torch.tensor([[0.3, -0.2]]),
        coordinate_origin=torch.tensor([[-0.1, 0.0]]),
        coordinate_scale=torch.tensor([[1.2, 0.6]]),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        module_characteristic_lengths=torch.tensor([[[0.2, 0.12], [0.18, 0.10], [0.25, 0.14]]]),
        environment_characteristic_lengths=torch.full((1, 4, 2), 0.1),
        scene_key="source-read-scene",
        normalizer_key="source-read-transform",
        coordinate_frame_key="source-read-frame",
    )


def _query(*, requires_grad: bool = False) -> QueryBatch:
    coordinates = torch.tensor([[[0.06, 0.03], [0.49, 0.19], [0.91, 0.37], [0.52, 0.22]]])
    if requires_grad:
        coordinates.requires_grad_()
    return QueryBatch(
        coordinates=coordinates,
        features=torch.tensor([[[0.1, 0.5], [-0.3, 0.7], [0.4, -0.1], [0.2, 0.6]]]),
        query_mask=torch.tensor([[True, True, True, False]]),
        query_ids=torch.arange(4)[None],
        scene_key="source-read-scene",
        coordinate_frame_key="source-read-frame",
        receiver_key="source-read-panel",
    )


def _core(**overrides) -> ContentHypergraphCoreV2:
    config = dict(
        module_input_dim=2,
        environment_input_dim=3,
        coordinate_dim=2,
        query_input_dim=2,
        operating_input_dim=2,
        output_dim=3,
        architecture="A2-D",
        context_dim=128,
        num_heads=4,
        context_depth=1,
        num_hyperedges=6,
        thermal_response_rank=4,
        local_hidden_dim=128,
        dropout=0.0,
        init_seed=1947,
    )
    config.update(overrides)
    return ContentHypergraphCoreV2(**config)


def _close_gradients(actual, expected, *, atol: float = 1.0e-11, rtol: float = 1.0e-10) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_headwise_reduction_fp64_vjps_bias_masks_chunks_permutations_and_empty_sources():
    torch.manual_seed(301)
    batch, query_count, sources, width, heads = 2, 5, 7, 128, 4
    projection = torch.nn.Linear(width, width).double()
    hidden = torch.randn(batch, query_count, sources, width, dtype=torch.float64, requires_grad=True)
    raw_weights = torch.rand(batch, query_count, sources, heads, dtype=torch.float64)
    allowed = torch.rand(batch, query_count, sources, heads) > 0.35
    weights = (raw_weights * allowed).mul_(0.37).requires_grad_()

    actual = reduce_headwise_linear(hidden, weights, projection, heads)
    legacy_messages = F.linear(hidden, projection.weight, projection.bias)
    legacy_blocks = legacy_messages.reshape(batch, query_count, sources, heads, width // heads)
    expected = torch.einsum("bqmh,bqmhd->bqhd", weights, legacy_blocks).reshape(batch, query_count, width)
    torch.testing.assert_close(actual, expected, atol=2.0e-13, rtol=2.0e-13)

    cotangent = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        (actual * cotangent).sum(), (hidden, weights, projection.weight, projection.bias)
    )
    expected_grads = torch.autograd.grad(
        (expected * cotangent).sum(), (hidden, weights, projection.weight, projection.bias)
    )
    _close_gradients(actual_grads, expected_grads)

    chunked = torch.cat(
        [reduce_headwise_linear(hidden[:, start : start + 2], weights[:, start : start + 2], projection, heads)
         for start in range(0, query_count, 2)],
        dim=1,
    )
    torch.testing.assert_close(chunked, actual, atol=1.0e-14, rtol=1.0e-14)
    permutation = torch.randperm(sources)
    permuted = reduce_headwise_linear(
        hidden.index_select(2, permutation), weights.index_select(2, permutation), projection, heads
    )
    torch.testing.assert_close(permuted, actual, atol=3.0e-13, rtol=3.0e-13)

    empty_hidden = hidden[:, :, :0]
    empty_weights = weights[:, :, :0]
    empty = reduce_headwise_linear(empty_hidden, empty_weights, projection, heads)
    assert torch.equal(empty, torch.zeros_like(empty))


def test_composed_scalar_projection_and_local_reduction_match_pairwise_fp64_vjps():
    torch.manual_seed(302)
    width = 128
    message_output = torch.nn.Linear(width, width).double()
    scalar_output = torch.nn.Linear(width, 1).double()
    hidden = torch.randn(2, 3, 5, width, dtype=torch.float64, requires_grad=True)
    composed_weight, composed_bias = compose_scalar_projection(message_output, scalar_output)
    actual = F.linear(hidden, composed_weight, composed_bias)
    expected = scalar_output(message_output(hidden))
    torch.testing.assert_close(actual, expected, atol=2.0e-13, rtol=2.0e-13)
    cotangent = torch.randn_like(actual)
    parameters = (
        hidden,
        message_output.weight,
        message_output.bias,
        scalar_output.weight,
        scalar_output.bias,
    )
    actual_grads = torch.autograd.grad((actual * cotangent).sum(), parameters)
    expected_grads = torch.autograd.grad((expected * cotangent).sum(), parameters)
    _close_gradients(actual_grads, expected_grads)

    pair_count, destinations = 13, 8
    local_projection = torch.nn.Linear(width, width).double()
    pair_hidden = torch.randn(pair_count, width, dtype=torch.float64, requires_grad=True)
    pair_weights = (torch.rand(pair_count, dtype=torch.float64) * 0.4).requires_grad_()
    destination_index = torch.tensor([0, 1, 1, 4, 0, 6, 6, 6, 2, 4, 1, 7, 0])
    reduced = reduce_indexed_linear(
        pair_hidden, pair_weights, destination_index, destinations, local_projection
    )
    pair_values = local_projection(pair_hidden) * pair_weights[:, None]
    pairwise = torch.zeros(destinations, width, dtype=torch.float64).index_add(
        0, destination_index, pair_values
    )
    torch.testing.assert_close(reduced, pairwise, atol=2.0e-13, rtol=2.0e-13)
    local_cotangent = torch.randn_like(reduced)
    local_parameters = (pair_hidden, pair_weights, local_projection.weight, local_projection.bias)
    actual_local_grads = torch.autograd.grad((reduced * local_cotangent).sum(), local_parameters)
    expected_local_grads = torch.autograd.grad((pairwise * local_cotangent).sum(), local_parameters)
    _close_gradients(actual_local_grads, expected_local_grads)

    empty_local = reduce_indexed_linear(
        pair_hidden[:0], pair_weights[:0], destination_index[:0], destinations, local_projection
    )
    assert torch.equal(empty_local, torch.zeros_like(empty_local))
    with pytest.raises(ValueError, match="outside the destination axis"):
        reduce_indexed_linear(
            pair_hidden[:1], pair_weights[:1], torch.tensor([destinations]), destinations, local_projection
        )


def test_reduced_projection_core_fp32_gradients_and_one_adamw_step_match_legacy():
    torch.manual_seed(303)
    legacy = _core(reader_execution_backend="legacy").train()
    reduced = _core(reader_execution_backend="reduced_projection").train()
    assert set(dict(legacy.named_parameters())) == set(dict(reduced.named_parameters()))
    for name, parameter in legacy.named_parameters():
        torch.testing.assert_close(parameter, dict(reduced.named_parameters())[name], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        legacy.state_dict()["direct_reader.message_output.weight"],
        reduced.state_dict()["direct_reader.message_output.weight"],
        atol=0.0,
        rtol=0.0,
    )

    scene_legacy = _scene(requires_grad=True)
    scene_reduced = _scene(requires_grad=True)
    query_legacy = _query(requires_grad=True)
    query_reduced = _query(requires_grad=True)
    controls = torch.tensor([[0.5, -0.2, 0.7]])
    optimizers = [
        torch.optim.AdamW(legacy.parameters(), lr=1.0e-4, weight_decay=1.0e-5),
        torch.optim.AdamW(reduced.parameters(), lr=1.0e-4, weight_decay=1.0e-5),
    ]

    outputs = []
    for model, scene, query in zip((legacy, reduced), (scene_legacy, scene_reduced),
                                   (query_legacy, query_reduced), strict=True):
        organization = model.prepare(scene)
        output = model.decode(organization, query, controls=controls)
        loss = output.nonlinear_field.square().mean() + 0.3 * output.thermal_temperature.square().mean()
        loss.backward()
        outputs.append(output)
    torch.testing.assert_close(
        outputs[0].nonlinear_field, outputs[1].nonlinear_field, atol=3.0e-6, rtol=3.0e-5
    )
    torch.testing.assert_close(
        outputs[0].thermal_temperature, outputs[1].thermal_temperature, atol=3.0e-6, rtol=3.0e-5
    )
    _close_gradients((query_legacy.coordinates.grad, scene_legacy.module_coordinates.grad),
                     (query_reduced.coordinates.grad, scene_reduced.module_coordinates.grad),
                     atol=1.0e-5, rtol=1.0e-4)
    reduced_parameters = dict(reduced.named_parameters())
    for name, parameter in legacy.named_parameters():
        other = reduced_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1.0e-5, rtol=1.0e-4, msg=name)

    for optimizer in optimizers:
        optimizer.step()
    for name, parameter in legacy.named_parameters():
        other = reduced_parameters[name]
        torch.testing.assert_close(parameter, other, atol=2.0e-6, rtol=1.0e-4, msg=name)
        left_state = optimizers[0].state[parameter]
        right_state = optimizers[1].state[other]
        for state_name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                left_state[state_name], right_state[state_name], atol=1.0e-7, rtol=1.0e-4,
                msg=f"{name}.{state_name}",
            )

    with torch.no_grad():
        for model, scene, query in zip((legacy, reduced), (scene_legacy, scene_reduced),
                                       (query_legacy, query_reduced), strict=True):
            refreshed = model.decode(model.prepare(scene), query, controls=controls)
            assert refreshed.work_counts["direct_temperature_coefficient_entries"] == 4 * 3
            assert refreshed.work_counts["direct_dense_reference_entries"] == 4 * 3
    torch.testing.assert_close(
        legacy.decode(legacy.prepare(scene_legacy), query_legacy, controls=controls).nonlinear_field,
        reduced.decode(reduced.prepare(scene_reduced), query_reduced, controls=controls).nonlinear_field,
        atol=3.0e-6,
        rtol=3.0e-5,
    )


def test_dscale_zero_initialization_named_tensor_identity_and_config_roundtrip():
    baseline = _core().eval()
    dscale = _core(source_value_conditioning="source_scale").eval()
    baseline_config = baseline.get_config()
    assert "reader_execution_backend" not in baseline_config
    assert "source_value_conditioning" not in baseline_config
    assert ContentHypergraphCoreV2.from_config(baseline_config).get_config() == baseline_config

    scale_config = dscale.get_config()
    assert scale_config["source_value_conditioning"] == "source_scale"
    restored = ContentHypergraphCoreV2.from_config(scale_config).eval()
    assert restored.get_config() == scale_config
    assert torch.equal(dscale.direct_reader.source_scale_projection.weight,
                       torch.zeros_like(dscale.direct_reader.source_scale_projection.weight))
    baseline_names = set(dict(baseline.named_parameters()))
    dscale_names = set(dict(dscale.named_parameters()))
    assert baseline_names.issubset(dscale_names)
    assert dscale_names - baseline_names == {"direct_reader.source_scale_projection.weight"}
    assert dscale.direct_reader.source_scale_projection.weight.numel() == 2 * 128
    scale_parameters = dict(dscale.named_parameters())
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(parameter, scale_parameters[name], atol=0.0, rtol=0.0)
    for name, value in dscale.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0.0, rtol=0.0)

    scene, query = _scene(), _query()
    baseline_output = baseline.decode(baseline.prepare(scene), query, controls=torch.ones(1, 3))
    dscale_output = dscale.decode(dscale.prepare(scene), query, controls=torch.ones(1, 3))
    torch.testing.assert_close(baseline_output.nonlinear_field, dscale_output.nonlinear_field, atol=0.0, rtol=0.0)
    torch.testing.assert_close(baseline_output.thermal_temperature, dscale_output.thermal_temperature,
                               atol=0.0, rtol=0.0)

    dscale.train()
    initialized_projection = dscale.direct_reader.source_scale_projection.weight
    organization = dscale.prepare(scene)
    output = dscale.decode(organization, query, controls=torch.ones(1, 3))
    (output.nonlinear_field.square().mean() + output.thermal_temperature.square().mean()).backward()
    assert initialized_projection.grad is not None
    assert torch.isfinite(initialized_projection.grad).all()
    assert initialized_projection.grad.abs().sum() > 0

    with torch.no_grad():
        initialized_projection.fill_(0.01)
        changed = dscale.decode(dscale.prepare(scene), query, controls=torch.ones(1, 3))
    assert not torch.equal(changed.nonlinear_field, baseline_output.nonlinear_field)


def test_dense_backend_config_is_explicit_and_rejects_h6_changes():
    reduced = _core(reader_execution_backend="reduced_projection")
    payload = reduced.get_config()
    assert payload["reader_execution_backend"] == "reduced_projection"
    assert "source_value_conditioning" not in payload
    assert ContentHypergraphCoreV2.from_config(payload).get_config() == payload
    combined = _core(
        reader_execution_backend="reduced_projection", source_value_conditioning="source_scale"
    )
    combined_payload = combined.get_config()
    assert combined_payload["reader_execution_backend"] == "reduced_projection"
    assert combined_payload["source_value_conditioning"] == "source_scale"
    assert ContentHypergraphCoreV2.from_config(combined_payload).get_config() == combined_payload
    with pytest.raises(ValueError, match="only by A2-D"):
        ContentHypergraphCoreV2(
            module_input_dim=2,
            environment_input_dim=3,
            coordinate_dim=2,
            architecture="A2-H6",
            reader_execution_backend="reduced_projection",
        )
    with pytest.raises(ValueError, match="only by A2-D"):
        ContentHypergraphCoreV2(
            module_input_dim=2,
            environment_input_dim=3,
            coordinate_dim=2,
            architecture="A2-H6",
            source_value_conditioning="source_scale",
        )


def test_reduced_explicit_source_message_trace_projects_only_selected_queries():
    torch.manual_seed(304)
    legacy = _core(reader_execution_backend="legacy").eval()
    reduced = _core(reader_execution_backend="reduced_projection").eval()
    reduced.load_state_dict(legacy.state_dict())
    scene, query = _scene(), _query()

    def read_messages(model, request):
        projected_shapes = []
        handle = model.direct_reader.message_output.register_forward_pre_hook(
            lambda _module, args: projected_shapes.append(tuple(args[0].shape))
        )
        try:
            result = model.decode(model.prepare(scene), query, trace_request=request)
        finally:
            handle.remove()
        return result.trace["direct_source_messages"], projected_shapes

    full_request = {"include": ["direct_source_messages"]}
    full_messages, full_projection_shapes = read_messages(reduced, full_request)
    assert full_projection_shapes == [(1, 4, 3, 128)]

    partial_indices = torch.tensor([2, 0])
    partial_request = {"include": ["direct_source_messages"], "query_indices": partial_indices}
    partial_messages, partial_projection_shapes = read_messages(reduced, partial_request)
    assert partial_projection_shapes == [(1, 2, 3, 128)]
    torch.testing.assert_close(partial_messages, full_messages.index_select(1, partial_indices))

    full_permutation = torch.tensor([3, 1, 2, 0])
    permutation_request = {"include": ["direct_source_messages"], "query_indices": full_permutation}
    permuted_messages, permutation_projection_shapes = read_messages(reduced, permutation_request)
    assert permutation_projection_shapes == [(1, 4, 3, 128)]
    torch.testing.assert_close(permuted_messages, full_messages.index_select(1, full_permutation))

    legacy_messages, legacy_projection_shapes = read_messages(legacy, partial_request)
    assert legacy_projection_shapes == [(1, 4, 3, 128)]
    torch.testing.assert_close(legacy_messages, partial_messages, atol=3.0e-6, rtol=3.0e-5)


@pytest.mark.parametrize("packed_local", [False, True])
def test_explicit_query_tiles_preserve_live_native_pairs_outputs_and_gradients(packed_local):
    from dataclasses import replace
    from honf_forward_core.content_hypergraph.query_chunks_v1 import slice_query_v1

    whole = _core(reader_execution_backend="reduced_projection").double()
    tiled = _core(reader_execution_backend="reduced_projection", reader_query_chunk_size=2).double()
    tiled.load_state_dict(whole.state_dict())
    scene = _scene(requires_grad=True)
    query = _query(requires_grad=True)
    # Keep floating coordinates/projections in FP64 while preserving IDs/masks.
    scene = replace(scene, **{name: value.double() for name, value in vars(scene).items()
                              if torch.is_tensor(value) and value.is_floating_point()})
    query = replace(query, coordinates=query.coordinates.double(), features=query.features.double())
    if packed_local:
        query = replace(query, local_batch_indices=torch.tensor([0, 0, 0, 0, 0]),
                        local_query_indices=torch.tensor([0, 1, 2, 2, 3]),
                        local_module_indices=torch.tensor([0, 0, 0, 2, 2]),
                        local_weights=query.coordinates[0, [0, 1, 2, 2, 3], 0].sigmoid())
        tail = slice_query_v1(query, 2, 4)
        assert tail.local_query_indices.tolist() == [0, 0, 1]
        assert tail.local_module_indices.tolist() == [0, 2, 2]
        assert tail.local_weights.grad_fn is not None
    controls = torch.tensor([[0.5, -0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    outputs = [model.decode(model.prepare(scene), query, controls) for model in (whole, tiled)]
    for name in ("nonlinear_field", "thermal_temperature"):
        torch.testing.assert_close(getattr(outputs[0], name), getattr(outputs[1], name), atol=1e-9, rtol=1e-7)
    assert outputs[1].work_counts["reader_query_tiles"] == 2
    for name in ("receiver_rows", "direct_source_rows", "local_rows", "direct_response_contraction_rows"):
        assert outputs[0].work_counts[name] == outputs[1].work_counts[name]
    variables = (query.coordinates, scene.module_coordinates, controls)
    gradients = [torch.autograd.grad(output.nonlinear_field.square().sum()
                                    + output.thermal_temperature.square().sum(), variables, retain_graph=True)
                 for output in outputs]
    _close_gradients(*gradients, atol=1e-9, rtol=1e-7)
    assert "reader_query_chunk_size" not in whole.get_config()
    assert ContentHypergraphCoreV2.from_config(tiled.get_config()).reader_query_chunk_size == 2
    # Explicit diagnostic traces use the full-request path and retain its
    # documented global query indexing and cap semantics.
    trace = tiled.decode(tiled.prepare(scene), query,
                         trace_request={"include": ["direct_source_messages"], "query_indices": [3, 0]})
    assert trace.trace["trace_query_indices"].tolist() == [3, 0]
