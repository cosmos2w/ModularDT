"""An explicit live source projection with fixed receiver organization."""

from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph
from honf_forward_core.interface_fields.topology_probe import OrganizerTopologyProbe
from .test_adaptive_receiver_hypergraph import _case


def _organizer():
    case = _case(modules=4, environment=12)
    organizer = AdaptiveReceiverHypergraph(8, organizer_dim=16, control_dim=4,
        max_depth=2, faithful_controls=True, measure_consistent=True).eval()
    with torch.no_grad():
        for head in organizer.source_scores.values():
            head[-1].weight.normal_(0, .2)
        for head in organizer.control_heads.values():
            head[-1].weight.normal_(0, .1)
    return organizer, case


def test_live_membership_keeps_frontier_and_receiver_connections_but_allows_source_support_change():
    organizer, case = _organizer()
    reference = organizer.prepare(case, case.module_tokens, capture_topology=True)
    same = organizer.prepare(case, case.module_tokens, fixed_topology=reference,
        topology_mode="fixed_frontier_live_membership")
    for tau in reference.memberships:
        torch.testing.assert_close(same.memberships[tau], reference.memberships[tau], rtol=0, atol=0)
        torch.testing.assert_close(same.controls[tau], reference.controls[tau], rtol=0, atol=0)
    changed = replace(case, module_tokens=case.module_tokens * 20,
        env_tokens=case.env_tokens * 20)
    live = organizer.prepare(changed, changed.module_tokens, fixed_topology=reference,
        topology_mode="fixed_frontier_live_membership")
    assert any(not torch.equal(reference.memberships[tau] > 0, live.memberships[tau] > 0)
               for tau in reference.memberships)
    for tau in reference.memberships:
        query = (case.module_centers if tau in {"MM", "ME"} else
                 case.env_coords if tau == "EM" else case.receiver_anchor_coords)
        torch.testing.assert_close(live.strategy_data["gates"][tau], reference.strategy_data["gates"][tau], rtol=0, atol=0)
        before = organizer.access(reference, query, tau)
        after = organizer.access(live, query, tau,
            fixed_receiver_access={"edge_access": before.edge_access})
        torch.testing.assert_close(before.edge_access, after.edge_access, rtol=0, atol=0)
        assert torch.isfinite(after.weight).all() and torch.isfinite(after.control).all()
    assert live.strategy_data["topology_continuation_mode"] == "fixed_frontier_live_membership"


def test_live_projection_same_support_ad_matches_finite_difference():
    organizer, case = _organizer()
    organizer = organizer.double().requires_grad_(False)
    case = replace(case, **{f.name: getattr(case, f.name).double()
        for f in fields(case) if torch.is_tensor(getattr(case, f.name))})
    reference = organizer.prepare(case, case.module_tokens, capture_topology=True)
    direction = torch.linspace(-.1, .1, case.module_tokens.numel()).reshape_as(case.module_tokens)

    def evaluate(alpha):
        states = case.module_tokens + alpha * direction
        state = organizer.prepare(case, states, fixed_topology=reference,
            topology_mode="fixed_frontier_live_membership")
        access = organizer.access(state, case.receiver_anchor_coords, "QE")
        return access.control.square().mean() + access.weight.square().mean(), state

    alpha = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    value, base = evaluate(alpha)
    ad, = torch.autograd.grad(value, alpha)
    plus, high = evaluate(alpha.detach() + 1e-5)
    minus, low = evaluate(alpha.detach() - 1e-5)
    for tau in reference.memberships:
        assert torch.equal(base.memberships[tau] > 0, high.memberships[tau] > 0)
        assert torch.equal(base.memberships[tau] > 0, low.memberships[tau] > 0)
    assert abs(float(ad)) > 1e-8
    torch.testing.assert_close(ad, (plus-minus)/2e-5, rtol=2e-4, atol=1e-8)


def test_unknown_or_unbound_frontier_mode_rejected_and_legacy_default_preserved():
    organizer, case = _organizer()
    with pytest.raises(ValueError, match="Unknown topology"):
        organizer.prepare(case, case.module_tokens, topology_mode="clamp_negative")
    with pytest.raises(ValueError, match="requires a reference"):
        organizer.prepare(case, case.module_tokens, topology_mode="fixed_frontier_live_membership")
    reference = organizer.prepare(case, case.module_tokens, capture_topology=True)
    default = organizer.prepare(case, case.module_tokens, fixed_topology=reference)
    explicit = organizer.prepare(case, case.module_tokens, fixed_topology=reference,
        topology_mode="fixed_active_set")
    for tau in reference.memberships:
        torch.testing.assert_close(default.memberships[tau], explicit.memberships[tau], rtol=0, atol=0)
        torch.testing.assert_close(default.controls[tau], explicit.controls[tau], rtol=0, atol=0)


def test_public_probe_threads_live_mode_and_restores_only_its_methods_on_exception():
    organizer, case = _organizer()
    with OrganizerTopologyProbe(organizer) as baseline:
        state = organizer.prepare(case, case.module_tokens)
        organizer.access(state, case.receiver_anchor_coords, "QE")
    with pytest.raises(ValueError, match="Unknown topology"):
        OrganizerTopologyProbe(organizer, baseline.record, topology_mode="clamp_negative")
    with pytest.raises(ValueError, match="requires a reference"):
        OrganizerTopologyProbe(organizer, topology_mode="fixed_frontier_live_membership")
    with pytest.raises(RuntimeError, match="intentional"), OrganizerTopologyProbe(
        organizer, baseline.record, topology_mode="fixed_frontier_live_membership"
    ) as replay:
        state = organizer.prepare(case, case.module_tokens)
        access = organizer.access(state, case.receiver_anchor_coords, "QE")
        torch.testing.assert_close(access.edge_access,
            baseline.record.accesses[0, "QE", 0]["edge_access"], rtol=0, atol=0)
        assert state.strategy_data["topology_continuation_mode"] == "fixed_frontier_live_membership"
        raise RuntimeError("intentional continuation exit")
    assert "prepare" not in organizer.__dict__ and "access" not in organizer.__dict__
