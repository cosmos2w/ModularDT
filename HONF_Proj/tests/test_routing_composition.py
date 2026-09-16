"""Independent references for source-measure routing composition."""
import torch

from honf_forward_core.interface_fields.routing_index.pair_join import compile_two_hop_pairs
from honf_forward_core.interface_fields.routing_index.sparse_projection import (
    build_typed_source_incidence,
    masked_sparsemax,
    source_measure_sparsemax,
)


def effective(membership, omega, logits):
    source = build_typed_source_incidence(omega, membership)
    density = source_measure_sparsemax(logits, source.hub_measure).density
    pairs = compile_two_hop_pairs(density, source)
    dense = density.new_zeros((logits.shape[0], logits.shape[1], membership.shape[1]))
    dense[pairs.batch_index, pairs.receiver_index, pairs.source_index] = pairs.prior
    return dense


def test_composed_source_query_pair_gradient_check():
    x = torch.tensor([[[.2, -.1, .5], [.3, .1, -.2]]], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([[[.2, -.1, .4], [1.3, -.5, .2]]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[.2, .8]], dtype=torch.float64, requires_grad=True)
    fn = lambda x, z, w: effective(masked_sparsemax(x), w, z)
    assert torch.autograd.gradcheck(fn, (x, z, w))
    result = fn(x, z, w)
    torch.testing.assert_close(result.sum(-1), torch.ones_like(result.sum(-1)))


def test_source_and_conditional_hub_splitting_preserve_prior():
    a = torch.tensor([[[.6, .4, 0.], [.1, .2, .7], [0., .3, .7]]], dtype=torch.float64)
    w = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    z = torch.tensor([[[1.4, -.3, .2], [-.4, .3, .1]]], dtype=torch.float64)
    base = effective(a, w, z)
    # Identical copies of source 0 split its physical quadrature measure.
    source_split = effective(a[:, [0, 0, 1, 2]], torch.tensor([[.07, .13, .3, .5]], dtype=torch.float64), z)
    reconstructed = torch.stack((source_split[..., 0] + source_split[..., 1], source_split[..., 2], source_split[..., 3]), -1)
    torch.testing.assert_close(reconstructed, base)
    # This conditional identity supplies split incidences; it does not rerun
    # ordinary sparsemax on a duplicated raw candidate bank.
    split_a = torch.cat((a[..., :1] * .35, a[..., :1] * .65, a[..., 1:]), -1)
    split_z = torch.cat((z[..., :1], z), -1)
    torch.testing.assert_close(effective(split_a, w, split_z), base)


def test_vanishing_hub_has_no_finite_last_member_jump():
    w = torch.tensor([[.4, .6]], dtype=torch.float64)
    z = torch.tensor([[[.3, .8]]], dtype=torch.float64)
    def value(eps):
        a = torch.tensor([[[1.-eps, eps], [1., 0.]]], dtype=torch.float64)
        return effective(a, w, z)
    endpoint = value(0.)
    errors = [(value(eps) - endpoint).abs().max().item() for eps in (1e-2, 1e-4, 1e-6)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-6


def test_support_entry_is_continuous_with_one_sided_derivatives():
    def projected(delta):
        z = torch.tensor([[1. + delta, 0.]], dtype=torch.float64)
        return masked_sparsemax(z)
    h = 1e-6
    center, left, right = projected(0.), projected(-h), projected(h)
    assert torch.equal(left > 0, torch.tensor([[True, True]]))
    assert torch.equal(right > 0, torch.tensor([[True, False]]))
    torch.testing.assert_close((center-left)/h, torch.tensor([[.5,-.5]], dtype=torch.float64))
    torch.testing.assert_close((right-center)/h, torch.zeros_like(center))
    assert (left-right).abs().max() <= h
