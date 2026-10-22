"""Independent numerical checks for continuous source-read executors."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.continuous_joint.reader import read_continuous
from tests.test_budgeted_native_core import _double
from tests.test_content_hypergraph_v2_core import _query, _scene
from tests.test_continuous_joint import _model


def _fixture():
    _, model = _model("direct", tile=8)
    model.eval()
    scene, query = _double(_scene(1)), _double(_query(1))

    # A present, zero-information heater still contributes to signed K.
    measure = scene.module_measure.clone()
    measure[:, 1] = 0
    module_mask = torch.ones_like(scene.module_mask)
    scene = replace(
        scene,
        module_measure=measure,
        module_mask=module_mask,
        module_coordinates=scene.module_coordinates.clone().requires_grad_(),
        module_characteristic_lengths=scene.module_characteristic_lengths.clone().requires_grad_(),
    )

    # Repeated receivers remain distinct rows in the returned query order.
    coordinates = query.coordinates[:, :4].clone()
    features = query.features[:, :4].clone()
    coordinates[:, 3] = coordinates[:, 1]
    features[:, 3] = features[:, 1]
    coordinates.requires_grad_()
    query = replace(query, coordinates=coordinates, features=features,
                    query_ids=query.query_ids[:, :4],
                    query_mask=torch.ones((coordinates.shape[0], coordinates.shape[1]), dtype=torch.bool))
    organization = model.prepare(scene)
    qstates = model._query_states(organization, query)
    reader = model.direct_reader

    batch, queries = qstates.shape[:2]
    modules = scene.num_modules
    gate = torch.tensor(
        [[[0.7, 0.4, 0.2], [0.2, 0.6, 0.8], [0.9, 0.5, 0.1], [0.2, 0.6, 0.8]]],
        dtype=torch.float64,
    ).expand(batch, queries, modules).clone().requires_grad_()
    alpha = torch.tensor(
        [[[[0.11, 0.23, 0.37, 0.19], [0.0, 0.0, 0.0, 0.0], [0.17, 0.29, 0.41, 0.31]],
          [[0.31, 0.18, 0.09, 0.21], [0.0, 0.0, 0.0, 0.0], [0.12, 0.26, 0.33, 0.24]],
          [[0.22, 0.35, 0.13, 0.28], [0.0, 0.0, 0.0, 0.0], [0.19, 0.08, 0.27, 0.16]],
          [[0.31, 0.18, 0.09, 0.21], [0.0, 0.0, 0.0, 0.0], [0.12, 0.26, 0.33, 0.24]]]],
        dtype=torch.float64,
    ).requires_grad_()
    args = dict(
        coordinates=scene.module_coordinates,
        lengths=scene.module_characteristic_lengths,
        valid=scene.module_mask,
        measure=scene.module_measure,
        frame_scale=organization.frame_scale,
        keys=organization.direct_source_keys,
        values=organization.direct_source_values,
        gate=gate,
        frozen_coefficients=alpha,
    )
    return model, reader, scene, query, qstates, args


def _literal_reference(reader, query, qstates, *, coordinates, lengths, valid,
                       measure, frame_scale, keys, values, gate, frozen_coefficients):
    """Nested FP64 record/head sum, independent of executor reduction helpers."""
    batch, queries, _ = qstates.shape
    modules = coordinates.shape[1]
    heads = reader.heads
    head_width = reader.head_width
    contexts = []
    responses = []
    qmask = query.query_mask
    effective_gate = gate * valid[:, None] * qmask[..., None]
    for b in range(batch):
        query_contexts = []
        query_responses = []
        for q in range(queries):
            head_context = [qstates.new_zeros((head_width,)) for _ in range(heads)]
            source_coefficients = []
            for n in range(modules):
                support_open = effective_gate[b, q, n].detach() > 0
                delta = query.coordinates[b, q] - coordinates[b, n]
                relative = delta / frame_scale[b]
                features = [relative]
                for frequency in (1.0, 2.0, 4.0):
                    phase = (2.0 * torch.pi * frequency) * relative
                    features.extend((torch.sin(phase), torch.cos(phase)))
                relative_value = reader.relative_value(torch.cat(features, dim=-1))
                if reader.source_scale_projection is not None:
                    relative_value = relative_value + reader.source_scale_projection(
                        torch.asinh(delta / lengths[b, n])
                    )
                hidden = F.silu(reader.query_value(qstates[b, q]) + values[b, n] + relative_value)
                message = reader.message_output(hidden)
                message_heads = message.reshape(heads, head_width)
                for h in range(heads):
                    head_context[h] = (
                        head_context[h]
                        + frozen_coefficients[b, q, n, h] * message_heads[h] * support_open
                    )
                if reader.thermal_output is not None:
                    coefficient = reader.thermal_output(message).squeeze(-1)
                    source_coefficients.append(coefficient * effective_gate[b, q, n])
            query_contexts.append(torch.stack(head_context).reshape(-1))
            if reader.thermal_output is not None:
                query_responses.append(torch.stack(source_coefficients))
        contexts.append(torch.stack(query_contexts))
        if reader.thermal_output is not None:
            responses.append(torch.stack(query_responses))
    context = torch.stack(contexts)
    response = torch.stack(responses) if responses else None
    return context, response


def _read(reader, query, qstates, args, backend, *, checkpoint_values=False):
    return read_continuous(
        reader, query, qstates, **args, backend=backend,
        checkpoint_values=checkpoint_values,
    )


def _loss(read):
    query_weights = torch.arange(1, read.context.shape[1] + 1, dtype=read.context.dtype)
    weighted_context = (read.context * query_weights[None, :, None]).square().sum()
    source_weights = torch.arange(1, read.response.shape[-1] + 1, dtype=read.response.dtype)
    weighted_response = (read.response * source_weights[None, None, :]).square().sum()
    return weighted_context + 0.7 * weighted_response


def _gradient_inputs(reader, query, qstates, args):
    parameters = tuple(reader.parameters())
    tensors = (
        qstates,
        query.coordinates,
        args["coordinates"],
        args["lengths"],
        args["values"],
        args["gate"],
        args["frozen_coefficients"],
    )
    return parameters, tensors


def _compare_gradients(actual, expected, parameters, tensors, *, atol=2e-11, rtol=2e-10):
    actual_inputs = (*parameters, *tensors)
    actual_gradients = torch.autograd.grad(
        _loss(actual), actual_inputs, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        _loss(expected), actual_inputs, allow_unused=True
    )
    assert len(actual_gradients) == len(expected_gradients)
    for index, (left, right) in enumerate(zip(actual_gradients, expected_gradients, strict=True)):
        assert (left is None) == (right is None), index
        if left is not None:
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=str(index))


@pytest.mark.parametrize("backend", ["segment", "dense_weighted"])
def test_hidden_first_backend_matches_independent_fp64_values_and_vjps(backend):
    _, reader, _, query, qstates, args = _fixture()
    expected_context, expected_response = _literal_reference(
        reader, query, qstates, **args
    )
    actual = _read(reader, query, qstates, args, backend)
    torch.testing.assert_close(actual.context, expected_context, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(actual.response, expected_response, atol=3e-13, rtol=3e-13)
    assert actual.response[:, :, 1].abs().sum() > 0  # zero-measure heater remains a physical source
    torch.testing.assert_close(actual.context[:, 1], actual.context[:, 3], atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(actual.response[:, 1], actual.response[:, 3], atol=2e-14, rtol=2e-14)

    parameters, tensors = _gradient_inputs(reader, query, qstates, args)
    expected = type(actual)(expected_context, actual.alpha, expected_response, None, {})
    _compare_gradients(actual, expected, parameters, tensors)


def test_segment_empty_support_zero_rows_and_device_resident_work_counts():
    _, reader, _, query, qstates, args = _fixture()
    empty_args = dict(args)
    empty_args["gate"] = torch.zeros_like(args["gate"], requires_grad=True)
    empty_args["frozen_coefficients"] = torch.zeros_like(args["frozen_coefficients"], requires_grad=True)
    empty = _read(reader, query, qstates, empty_args, "segment")
    assert empty.context.count_nonzero() == 0
    assert empty.response.count_nonzero() == 0
    assert int(empty.counts["value_rows"]) == 0
    assert int(empty.counts["message_projection_rows"]) == 0
    assert int(empty.counts["head_reduction_projection_rows"]) == 0
    assert all(value.ndim == 0 and value.device == qstates.device for value in empty.counts.values())

    (empty.context.sum() + empty.response.sum()).backward()
    for name in ("query_value", "relative_value", "message_output", "thermal_output"):
        layer = getattr(reader, name)
        if layer is not None:
            assert all(parameter.grad is not None for parameter in layer.parameters()), name


def test_dense_weighted_executes_zero_rows_and_segment_preserves_source_permutation():
    _, reader, scene, query, qstates, args = _fixture()
    zero_args = dict(args)
    zero_args["gate"] = torch.zeros_like(args["gate"])
    zero_args["frozen_coefficients"] = torch.zeros_like(args["frozen_coefficients"])
    dense_zero = _read(reader, query, qstates, zero_args, "dense_weighted")
    assert int(dense_zero.counts["value_rows"]) == qstates.shape[0] * qstates.shape[1] * scene.num_modules
    assert int(dense_zero.counts["selected_rows"]) == 0
    assert int(dense_zero.counts["message_projection_rows"]) == 0
    assert int(dense_zero.counts["head_reduction_projection_rows"]) == qstates.shape[0] * qstates.shape[1] * reader.heads

    # A single exactly closed row has a zero contribution under both executors.
    partial_args = dict(args)
    partial_gate = args["gate"].detach().clone()
    partial_alpha = args["frozen_coefficients"].detach().clone()
    partial_gate[0, 0, 2] = 0
    partial_alpha[0, 0, 2] = 0
    partial_args["gate"] = partial_gate
    partial_args["frozen_coefficients"] = partial_alpha
    partial_segment = _read(reader, query, qstates, partial_args, "segment")
    partial_dense = _read(reader, query, qstates, partial_args, "dense_weighted")
    torch.testing.assert_close(partial_segment.context, partial_dense.context, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(partial_segment.response, partial_dense.response, atol=3e-13, rtol=3e-13)

    original = _read(reader, query, qstates, args, "segment")
    permutation = torch.tensor([2, 0, 1])
    permuted_args = dict(args)
    for key in ("coordinates", "lengths", "valid", "measure", "keys", "values"):
        permuted_args[key] = args[key].index_select(1, permutation)
    permuted_args["gate"] = args["gate"].index_select(2, permutation)
    permuted_args["frozen_coefficients"] = args["frozen_coefficients"].index_select(2, permutation)
    permuted = _read(reader, query, qstates, permuted_args, "segment")
    torch.testing.assert_close(permuted.context, original.context, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(
        permuted.response, original.response.index_select(2, permutation), atol=3e-13, rtol=3e-13
    )


def test_segment_query_chunking_duplicate_receivers_and_checkpoint_values_match():
    _, reader, _, query, qstates, args = _fixture()
    whole = _read(reader, query, qstates, args, "segment")
    chunk_contexts = []
    chunk_responses = []
    for start in range(query.num_queries):
        end = start + 1
        query_chunk = replace(
            query,
            coordinates=query.coordinates[:, start:end],
            features=query.features[:, start:end],
            query_mask=query.query_mask[:, start:end],
            query_ids=query.query_ids[:, start:end],
        )
        args_chunk = dict(args)
        args_chunk["gate"] = args["gate"][:, start:end]
        args_chunk["frozen_coefficients"] = args["frozen_coefficients"][:, start:end]
        chunk = _read(reader, query_chunk, qstates[:, start:end], args_chunk, "segment")
        chunk_contexts.append(chunk.context)
        chunk_responses.append(chunk.response)
    torch.testing.assert_close(torch.cat(chunk_contexts, dim=1), whole.context, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(torch.cat(chunk_responses, dim=1), whole.response, atol=2e-13, rtol=2e-13)

    checkpointed = _read(reader, query, qstates, args, "segment", checkpoint_values=True)
    torch.testing.assert_close(checkpointed.context, whole.context, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(checkpointed.response, whole.response, atol=2e-13, rtol=2e-13)
    parameters, tensors = _gradient_inputs(reader, query, qstates, args)
    _compare_gradients(checkpointed, whole, parameters, tensors)
