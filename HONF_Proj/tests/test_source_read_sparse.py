import pytest
import torch

from honf_forward_core.content_hypergraph.readers_v2 import DirectSourceReaderV2
from honf_forward_core.content_hypergraph.source_read_plans import build_source_read_plan_v1
from honf_forward_core.content_hypergraph.source_read_sparse import (
    evaluate_fine_source_rows_v1, reduce_selected_headwise_v1, source_score_descriptor_v1,
)
from honf_forward_core.content_hypergraph.spatial_features_v2 import query_source_features


@pytest.mark.parametrize("completion", [False, True])
def test_selected_and_packet_fine_execution_match_masked_function_and_gradients(completion):
    torch.manual_seed(918)
    reader = DirectSourceReaderV2(128, 2, source_value_conditioning="source_scale").double()
    query = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    source = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([[2., 3.], [4., 1.]], dtype=torch.float64)
    lengths = torch.rand(2, 7, 2, dtype=torch.float64) + .2
    descriptor = source_score_descriptor_v1(query, source, scale, lengths)
    legacy_descriptor, _ = query_source_features(query, source, scale, lengths)
    torch.testing.assert_close(descriptor, legacy_descriptor, atol=0., rtol=0.)
    qv = torch.randn(2, 5, 128, dtype=torch.float64, requires_grad=True)
    sv = torch.randn(2, 7, 128, dtype=torch.float64, requires_grad=True)
    alpha = torch.rand(2, 5, 7, 4, dtype=torch.float64, requires_grad=True)
    gates = torch.tensor([[[1., .3, 0., 0., .6, 0., 0.], [0.]*7, [.5]*7, [1., 0., 0., 0., 0., 0., 0.], [1.]*7]]*2,
                         dtype=torch.float64, requires_grad=True)
    present = torch.tensor([[True]*6+[False], [True]*7])
    plan = build_source_read_plan_v1(gates, alpha, present)
    # Any completion contributes at all original alpha weights; the selected
    # residual and its projection bias then obey precisely the same function.
    base = torch.randn(2, 5, 4, 128, dtype=torch.float64, requires_grad=True) if completion else None
    outputs, counts = [], []
    for executor in ("dense_masked", "packed", "packet"):
        calls = []
        handle = reader.relative_value.register_forward_pre_hook(lambda _m, args: calls.append(args[0].numel() // 14))
        try:
            rows = evaluate_fine_source_rows_v1(reader, plan, descriptor, qv, sv, executor=executor)
        finally:
            handle.remove()
        assert sum(calls) == rows.counts["fine_executed_rows"]
        outputs.append(reduce_selected_headwise_v1(rows.hidden, plan, reader.message_output, base_head_hidden=base))
        counts.append(rows.counts)
    assert counts[0]["fine_executed_rows"] == 2*5*7
    assert counts[1]["fine_executed_rows"] == plan.pair_count
    assert counts[2]["fine_executed_rows"] == plan.pair_count + counts[2]["fine_padding_rows"]
    assert counts[2]["fine_padding_rows"] > 0
    variables = (query, source, qv, sv, gates, alpha, *tuple(reader.parameters()))
    gradients = [torch.autograd.grad(value.square().sum(), variables, retain_graph=True, allow_unused=True)
                 for value in outputs]
    for actual, grad in zip(outputs[1:], gradients[1:]):
        torch.testing.assert_close(actual, outputs[0], atol=1e-9, rtol=1e-7)
        for left, right in zip(grad, gradients[0], strict=True):
            assert (left is None) == (right is None)
            if left is not None:
                torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)


def test_empty_selected_support_has_no_fine_kernel_rows_and_zero_mass_bias():
    reader = DirectSourceReaderV2(128, 2).double()
    gates = torch.zeros(1, 3, 5, dtype=torch.float64)
    alpha = torch.rand(1, 3, 5, 4, dtype=torch.float64)
    plan = build_source_read_plan_v1(gates, alpha, torch.ones(1, 5, dtype=torch.bool))
    for executor in ("packed", "packet"):
        rows = evaluate_fine_source_rows_v1(reader, plan, torch.zeros(1, 3, 5, 5, dtype=torch.float64),
                                           torch.zeros(1, 3, 128, dtype=torch.float64),
                                           torch.zeros(1, 5, 128, dtype=torch.float64), executor=executor)
        assert rows.hidden.shape == (0, 128)
        assert rows.counts["fine_executed_rows"] == 0
        assert torch.equal(reduce_selected_headwise_v1(rows.hidden, plan, reader.message_output),
                           torch.zeros(1, 3, 128, dtype=torch.float64))
