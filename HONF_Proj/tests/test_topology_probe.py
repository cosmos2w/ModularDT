"""Hard-value anchor identity and valid-region continuous physics gradients."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.topology_probe import (
    FixedTopologyInvalid,
    OrganizerTopologyProbe,
    fixed_active_projection,
)
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _case():
    torch.manual_seed(193)
    return EncodedInterfaceCase(torch.randn(1, 4, 8), torch.randn(1, 7, 8), torch.randn(1, 8),
        torch.rand(1, 4, 2), torch.rand(1, 7, 2), torch.ones(1, 4), torch.randn(1, 4, 3),
        None, torch.ones(1, 7), torch.ones(2))


def test_affine_projection_is_exact_at_anchor_and_rejects_negative_permissions():
    logits = torch.tensor([[[2., 1., 0.]]], dtype=torch.float64)
    density = torch.tensor([[[1., 0., 0.]]], dtype=torch.float64)
    result = fixed_active_projection(logits.requires_grad_(), logits.detach(), density, torch.ones(1, 3))
    assert torch.equal(result, density)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), logits)[0], torch.zeros_like(logits))
    reference = torch.tensor([[[.5, .5, 0.]]])
    with pytest.raises(FixedTopologyInvalid, match="negative"):
        fixed_active_projection(torch.tensor([[[0., 3., 0.]]]), torch.zeros(1, 1, 3), reference, torch.ones(1, 3))


@pytest.mark.parametrize("architecture", ["adaptive_receiver_hypergraph_honf", "overlap_control_hypergraph_honf", "local_overlap_hypergraph_honf"])
def test_recorded_topology_preserves_anchor_values_gradients_and_live_geometry(architecture):
    encoded = _case()
    backend = TypedHypergraphField(8, 12, 2, 2, architecture=architecture, spatial_dim=2,
        module_characteristic_length=.1).eval()
    query, features = torch.rand(1, 5, 2), torch.rand(1, 5, 6)
    with torch.no_grad():
        backend.read(backend.prepare(encoded, encoded.module_tokens), encoded, query, features)
        for head in backend.organizer.control_heads.values():
            final = head[-1] if architecture.startswith("adaptive") else head.net[-1]
            final.weight.normal_(0, .1)
        for gain in backend.control_gain.values():
            gain.weight.normal_(0, .1)
    backend.requires_grad_(False)

    def forward(value, centre):
        case = replace(encoded, module_tokens=value, module_centers=centre)
        prepared = backend.prepare(case, value)
        return backend.read(prepared, case, query, features)[0]

    with OrganizerTopologyProbe(backend.organizer) as recorded:
        normal = forward(encoded.module_tokens, encoded.module_centers)
    for snapshot in recorded.record.states.values():
        assert snapshot.phase == 0
    source, centres = encoded.module_tokens.clone().requires_grad_(), encoded.module_centers.clone().requires_grad_()
    with OrganizerTopologyProbe(backend.organizer, recorded.record) as fixed:
        anchored = forward(source, centres)
    torch.testing.assert_close(anchored, normal, rtol=0, atol=0)
    fixed_gradients = torch.autograd.grad(anchored.square().sum(), (source, centres))
    ordinary = forward(source, centres)
    gradients = torch.autograd.grad(ordinary.square().sum(), (source, centres))
    for actual, expected in zip(fixed_gradients, gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    if architecture.startswith("adaptive"):
        trees = fixed.record.states[0].strategy_data["trees"]
        baseline = recorded.record.states[0].strategy_data["trees"]
        for tau in trees:
            assert trees[tau][0].nodes == baseline[tau][0].nodes
    assert "prepare" not in backend.organizer.__dict__ and "access" not in backend.organizer.__dict__


def test_probe_requires_eval_and_restores_methods_on_invalid_case():
    backend = TypedHypergraphField(8, 12, 2, 2, architecture="adaptive_receiver_hypergraph_honf",
        spatial_dim=2, module_characteristic_length=.1)
    with pytest.raises(ValueError, match="evaluation"):
        OrganizerTopologyProbe(backend.organizer)
    backend.eval()
    encoded = _case()
    with OrganizerTopologyProbe(backend.organizer) as recorded:
        backend.prepare(encoded, encoded.module_tokens)
    changed = replace(encoded, module_present=torch.tensor([[1., 1., 1., 0.]]))
    with pytest.raises(FixedTopologyInvalid, match="source_valid"), OrganizerTopologyProbe(
        backend.organizer, recorded.record
    ):
        backend.prepare(changed, changed.module_tokens)
    assert "prepare" not in backend.organizer.__dict__ and "access" not in backend.organizer.__dict__
