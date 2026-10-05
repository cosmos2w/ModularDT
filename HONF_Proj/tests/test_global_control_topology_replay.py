"""Full-access Global controls interoperate with ordinary topology diagnostics."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from thermal_campaign_topology import _NativeRebuildCapture

from honf_forward_core.interface_fields.global_control_hypergraph import GlobalControlHypergraph
from honf_forward_core.interface_fields.topology_probe import FixedTopologyInvalid, OrganizerTopologyProbe

from .test_native_context_organizers import _encoded


def test_native_rebuild_capture_supplies_full_global_receiver_and_support_record():
    encoded = _encoded()
    organizer = GlobalControlHypergraph(8, organizer_dim=16).eval()
    with torch.no_grad():
        for head in organizer.control_heads.values():
            head[-1].weight.normal_(0, .1)
    model = SimpleNamespace(core=SimpleNamespace(
        backend=SimpleNamespace(organizer=organizer), decode_queries=lambda *_args, **_kwargs: None))
    query = torch.rand(2, 5, 2)

    def run(states):
        plan = organizer.prepare(encoded, states)
        values = {}
        for tau in plan.memberships:
            receiver = (encoded.module_centers if tau in {"MM", "ME"} else
                        encoded.env_coords if tau == "EM" else query)
            values[tau] = organizer.access(plan, receiver, tau)
        return plan, values

    with _NativeRebuildCapture(model) as capture:
        original, ordinary = run(encoded.module_tokens)
    with _NativeRebuildCapture(model, fixed=capture):
        current, replay = run(encoded.module_tokens + .2)
    assert any(not torch.equal(current.controls[tau], original.controls[tau]) for tau in current.controls)
    for tau in ordinary:
        torch.testing.assert_close(replay[tau].edge_access, ordinary[tau].edge_access, rtol=0, atol=0)
        torch.testing.assert_close(replay[tau].support, ordinary[tau].support, rtol=0, atol=0)
        torch.testing.assert_close(replay[tau].weight, ordinary[tau].weight, rtol=0, atol=0)
    assert "prepare" not in organizer.__dict__ and "access" not in organizer.__dict__


def test_global_replay_rejects_soft_training_catalogue_phase_and_frontier_modes():
    encoded = _encoded()
    organizer = GlobalControlHypergraph(8, organizer_dim=16).eval()
    reference = organizer.prepare(encoded, encoded.module_tokens)
    organizer.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        organizer.prepare(encoded, encoded.module_tokens, fixed_topology=reference)
    organizer.eval()
    with pytest.raises(ValueError, match="hard access"):
        organizer.prepare(encoded, encoded.module_tokens, fixed_topology=reference, soft=True)
    with pytest.raises(FixedTopologyInvalid, match="phase"):
        organizer.prepare(encoded, encoded.module_tokens, fixed_topology=reference, phase=1)
    damaged = replace(encoded, module_present=encoded.module_present.clone())
    damaged.module_present[0, 0] = 0
    with pytest.raises(FixedTopologyInvalid, match="source_valid"):
        organizer.prepare(damaged, damaged.module_tokens, fixed_topology=reference)
    with pytest.raises(ValueError, match="receiver-frontier"):
        organizer.prepare(encoded, encoded.module_tokens, fixed_topology=reference,
            topology_mode="fixed_frontier_live_membership")
    with OrganizerTopologyProbe(organizer) as probe:
        organizer.prepare(encoded, encoded.module_tokens)
    with pytest.raises(ValueError, match="receiver-tree"):
        OrganizerTopologyProbe(organizer, probe.record, topology_mode="fixed_frontier_live_membership")
