"""Conditional content locality and disclosed donors of the faithful Tree."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph
from .test_adaptive_receiver_hypergraph import _case


def _trained_like(dimension=2):
    model = AdaptiveReceiverHypergraph(8, spatial_dim=dimension, organizer_dim=16,
        control_dim=4, faithful_controls=True, measure_consistent=True).double().eval()
    with torch.no_grad():
        for head in (*model.source_scores.values(), *model.control_heads.values()):
            head[-1].weight.normal_(0, .2)
        model.split_head.weight.normal_(0, .2)
        for value in model.geometry_strength.values():
            value.fill_(.4)
    return model


def _double(encoded):
    return replace(encoded, **{name: value.double() for name, value in vars(encoded).items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()})


def test_fixed_membership_content_exclusion_and_admitted_gradient():
    encoded = _double(_case())
    model = _trained_like()
    state = model.prepare(encoded, encoded.module_tokens, capture_topology=True)
    recomputed = model.recompute_controls(state, encoded, encoded.module_tokens)
    for tau in state.controls:
        torch.testing.assert_close(recomputed[tau], state.controls[tau], atol=1e-12, rtol=1e-12)
    memberships = {tau: {kind: value.clone() for kind, value in donors.items()}
                   for tau, donors in state.control_memberships.items()}
    # Source zero is excluded from BOTH donor types for the chosen route.
    for kind in ("M", "E"):
        memberships["QE"][kind][..., 0] = 0
    fixed = replace(state, control_memberships=memberships)
    modules = encoded.module_tokens.clone().requires_grad_()
    environment = encoded.env_tokens.clone().requires_grad_()
    case = replace(encoded, module_tokens=modules, env_tokens=environment)
    first = model.recompute_controls(fixed, case, modules)["QE"]
    grad_m, grad_e = torch.autograd.grad(first.square().sum(), (modules, environment))
    assert torch.equal(grad_m[:, 0], torch.zeros_like(grad_m[:, 0]))
    assert torch.equal(grad_e[:, 0], torch.zeros_like(grad_e[:, 0]))
    assert grad_m[:, 1:].abs().sum() > 0 and grad_e[:, 1:].abs().sum() > 0
    changed = replace(case, module_tokens=modules.clone(), env_tokens=environment.clone())
    changed.module_tokens[:, 0] = 1000
    changed.env_tokens[:, 0] = -1000
    actual = model.recompute_controls(fixed, changed, changed.module_tokens)["QE"]
    torch.testing.assert_close(actual, first, atol=0, rtol=0)
    export = state.export()
    assert set(export["control_membership"]["QE"]) == {"M", "E"}
    assert export["dependency_provenance"]["mode"] == "membership_local_v1"
    assert "all current module states" in export["dependency_provenance"]["planning"]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("soft", [False, True])
def test_unequal_atom_refinement_rebuilds_all_receiver_and_control_states(dimension, soft):
    encoded = _double(_case(dimension))
    # Let query receivers derive from the same environmental support. This
    # exercises whole-index rebuilding rather than a fixed prepared reader.
    encoded = replace(encoded, receiver_anchor_coords=None, receiver_anchor_weights=None,
        env_weights=torch.linspace(.2, 2.4, 12, dtype=torch.float64)[None])
    model = _trained_like(dimension)
    mass = encoded.env_weights.clone()
    parent_mass = mass[:, :1].clone()
    mass[:, 0] *= .3
    refined = replace(encoded,
        env_coords=torch.cat((encoded.env_coords, encoded.env_coords[:, :1]), 1),
        env_tokens=torch.cat((encoded.env_tokens, encoded.env_tokens[:, :1]), 1),
        env_weights=torch.cat((mass, parent_mass * .7), 1))
    query = torch.rand(1, 7, dimension, dtype=torch.float64)
    for phase in (0, 1, 2):
        original = model.prepare(encoded, encoded.module_tokens, phase=phase, soft=soft)
        split = model.prepare(refined, refined.module_tokens, phase=phase, soft=soft)
        for tau in original.controls:
            torch.testing.assert_close(split.controls[tau], original.controls[tau], rtol=1e-10, atol=1e-10)
            for kind in ("M", "E"):
                actual = split.control_memberships[tau][kind]
                expected = original.control_memberships[tau][kind]
                if kind == "E":
                    torch.testing.assert_close(actual[..., :-1], expected, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(actual[..., -1], expected[..., 0], rtol=1e-10, atol=1e-10)
                else:
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        for tau in ("MM", "ME", "EM", "QM", "QE"):
            receiver = encoded.module_centers if tau in ("MM", "ME") else encoded.env_coords if tau == "EM" else query
            base = model.access(original, receiver, tau)
            actual = model.access(split, receiver, tau)
            if tau in ("ME", "QE"):
                torch.testing.assert_close(actual.weight[..., :-1], base.weight, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(actual.control[..., :-1, :], base.control, rtol=1e-10, atol=1e-10)
            else:
                torch.testing.assert_close(actual.weight, base.weight, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(actual.control, base.control, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("modules,environment", [(0, 0), (1, 0), (0, 3)])
def test_empty_control_types_export_absence_and_zero_summary(modules, environment):
    encoded = _double(_case(modules=modules, environment=environment))
    model = _trained_like()
    state = model.prepare(encoded, encoded.module_tokens)
    for tau in state.controls:
        assert torch.isfinite(state.controls[tau]).all()
        for kind, count in (("M", modules), ("E", environment)):
            if count == 0:
                assert not state.control_presence[tau][kind].any()
    rebuilt = model.recompute_controls(state, encoded, encoded.module_tokens)
    for tau in state.controls:
        torch.testing.assert_close(rebuilt[tau], state.controls[tau])
