"""Independent literal sums and finite-difference checks for all Round6 readers."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from honf_forward_core.inherited_attention.contracts import ReceiverCover
from honf_forward_core.inherited_attention.hyperedge_readout import HyperedgeReceiverReadout


class ReadoutFixture(nn.Module):
    def __init__(self, reduction, access, dtype):
        super().__init__()
        self.reader = HyperedgeReceiverReadout(
            coordinate_dim=2, reduction=reduction, access=access,
            geometry_mode="scene_frame_normalized", init_seed=17,
        ).to(dtype=dtype).eval()
        with torch.no_grad():
            weight = self.reader.output_projection.weight
            weight.copy_(.02 * torch.cos(torch.arange(weight.numel(), dtype=dtype).reshape_as(weight)))
            self.reader.hidden_bias.copy_(torch.linspace(-.1, .2, 64, dtype=dtype))

    def forward(self, states, queries, query_coordinates, anchors, weights):
        ids = torch.tensor([[11, 29]])
        prepared = self.reader.prepare_scene(states, ids, frame_key="oracle", scene_version=(), owner_token=7)
        common = {
            "prepared": prepared, "query_states": queries, "query_coordinates": query_coordinates,
            "anchor_coordinates": anchors, "frame_scale": states.new_tensor([[2., 3.]]),
            "query_mask": torch.ones(1, 1, dtype=torch.bool), "collect_exact_work_counts": False,
        }
        if self.reader.access == "full_bank_diagnostic":
            return self.reader.full_bank_correction(anchor_mask=torch.ones(1, 2, dtype=torch.bool),
                                                    anchor_measure=weights[:, 0], **common)[0]
        cover = ReceiverCover(
            anchor_indices=torch.tensor([[[0, 1]]]), anchor_ids=ids[:, None], weights=weights,
            validity=torch.ones(1, 1, 2, dtype=torch.bool), query_ids=torch.tensor([[43]]),
            frame_key="oracle", query_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        return self.reader.local_correction(cover=cover, **common)[0]


def inputs(dtype):
    generator = torch.Generator().manual_seed(238)
    return (
        torch.randn(1, 2, 128, generator=generator, dtype=dtype) * .2,
        torch.randn(1, 1, 128, generator=generator, dtype=dtype) * .2,
        torch.tensor([[[.37, .58]]], dtype=dtype),
        torch.tensor([[[.1, .2], [.8, .9]]], dtype=dtype),
        torch.tensor([[[.3, .6]]], dtype=dtype),
    )


def literal(fixture, values):
    """Scalar query/anchor loops, independently ordered geometry and sums."""
    states, queries, query_coordinates, anchors, weights = values
    reader = fixture.reader
    weights = weights[0, 0]
    if reader.access == "full_bank_diagnostic":
        weights = weights / weights.sum()
    query_term = F.linear(queries[0, 0], reader.query_projection.weight) + reader.hidden_bias
    reduced = states.new_zeros(64)
    for anchor in range(2):
        relative = (query_coordinates[0, 0] - anchors[0, anchor]) / states.new_tensor([2., 3.])
        features = []
        for coordinate in relative:
            features.extend((torch.asinh(coordinate), torch.sin(torch.pi * coordinate), torch.cos(torch.pi * coordinate),
                             torch.sin(torch.pi * coordinate / 2), torch.cos(torch.pi * coordinate / 2),
                             torch.sin(torch.pi * coordinate / 4), torch.cos(torch.pi * coordinate / 4)))
        edge_term = F.linear(states[0, anchor], reader.anchor_projection.weight)
        edge_term = edge_term + F.linear(torch.stack(features), reader.geometry_projection.weight)
        term = query_term + edge_term
        reduced = reduced + weights[anchor] * (F.silu(term) if reader.reduction == "edge_first" else term)
    if reader.reduction == "pool_first":
        reduced = F.silu(reduced)
    return F.linear(reduced, reader.output_projection.weight)[None, None]


MODES = [("pool_first", "local_cover"), ("edge_first", "local_cover"), ("edge_first", "full_bank_diagnostic")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("reduction", "access"), MODES)
def test_indexed_reader_matches_independent_literal_values_and_all_gradients(dtype, reduction, access):
    fixture = ReadoutFixture(reduction, access, dtype)
    values = tuple(value.requires_grad_() for value in inputs(dtype))
    actual = fixture(*values)
    expected = literal(fixture, values)
    tolerance = {"rtol": 3e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 2e-10, "atol": 2e-11}
    torch.testing.assert_close(actual, expected, **tolerance)
    cotangent = torch.linspace(-.7, .5, 128, dtype=dtype)[None, None]
    watched = (*values, *fixture.parameters())
    first = torch.autograd.grad((actual * cotangent).sum(), watched)
    reference = torch.autograd.grad((expected * cotangent).sum(), watched)
    for candidate, baseline in zip(first, reference):
        torch.testing.assert_close(candidate, baseline, **tolerance)


@pytest.mark.parametrize(("reduction", "access"), MODES)
def test_smooth_reader_passes_independent_gradcheck_and_gradgradcheck(reduction, access):
    fixture = ReadoutFixture(reduction, access, torch.float64)
    values = tuple(value.requires_grad_() for value in inputs(torch.float64))
    names = tuple(dict(fixture.named_parameters()))
    parameters = tuple(value.detach().clone().requires_grad_() for value in fixture.parameters())

    def function(*arguments):
        supplied_values, supplied_parameters = arguments[:5], arguments[5:]
        return torch.func.functional_call(fixture, dict(zip(names, supplied_parameters)), supplied_values, strict=True)

    # Fast mode checks independent random Jacobian projections for the complete
    # input and parameter tensors, avoiding a quadratic width-128 Jacobian.
    arguments = (*values, *parameters)
    assert torch.autograd.gradcheck(function, arguments, eps=1e-6, atol=2e-6, rtol=2e-4, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, arguments, eps=1e-6, atol=3e-6, rtol=3e-4, fast_mode=True)
