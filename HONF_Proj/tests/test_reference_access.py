"""Fixed normal permission replay leaves physical values live and rejects drift."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from honf_forward_core.evaluation.reference_access import FixedReferenceAccessReplay
from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder
from honf_forward_core.interface_fields.typed_hypergraph_state import TypedHypergraphState, source_moments


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.live_value = nn.Parameter(torch.tensor(3.0))
        self.plan_intervention = "normal"
        self.permission_mode = "hard"
        self.flip = False
        self.calls = 0

    def set_plan_intervention(self, value):
        self.plan_intervention = value

    def _access(self, plan, receivers, mechanism):
        self.calls += 1
        edge = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        if self.flip:
            edge = edge.flip(-1)
        controls = plan.controls[mechanism]
        if self.plan_intervention == "control_identity":
            controls = torch.zeros_like(controls)
        return source_moments(
            edge,
            plan.memberships[mechanism],
            controls,
            plan.source_measures["M"],
            plan.source_valid["M"],
            pair_valid=plan.strategy_data["pair_valid"],
            near=torch.tensor([[[0.25, 1.0], [0.0, 0.0]]]),
        )

    def prepare(self, phase):
        coords = {kind: torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]) for kind in ("M", "E")}
        return {
            "hypergraph_plan": TypedHypergraphState(
                {"MM": torch.eye(2)[None]},
                {"MM": torch.ones(1, 2, 16)},
                torch.zeros(1, 2, 2),
                torch.ones(1, 2),
                coords,
                {kind: torch.tensor([[1.0, 2.0]]) for kind in coords},
                {kind: torch.ones(1, 2, dtype=torch.bool) for kind in coords},
                {kind: torch.tensor([[0, 1]]) for kind in coords},
                phase=phase,
                strategy_data={"pair_valid": torch.ones(1, 2, 2, dtype=torch.bool)},
            )
        }


def reference(backend, phases=3):
    with TypedWorkEvidenceRecorder(backend) as recorded:
        for phase in range(phases):
            plan = backend.prepare(phase)["hypergraph_plan"]
            backend._access(plan, plan.source_coords["M"], "MM")
    return recorded.arrays


@pytest.mark.parametrize("historical", [False, True])
def test_allphase_permissions_bitexact_controls_zero_and_live_values_differ(historical):
    backend = Backend().eval()
    normal = reference(backend)
    if historical:
        normal = {key: value for key, value in normal.items() if not key.endswith("/density")}
    backend.flip = True
    with FixedReferenceAccessReplay(backend, normal) as replay, TypedWorkEvidenceRecorder(backend) as recorded:
        outputs = []
        for phase in range(3):
            plan = backend.prepare(phase)["hypergraph_plan"]
            access = backend._access(plan, plan.source_coords["M"], "MM")
            assert backend.plan_intervention == "control_identity"
            assert not torch.count_nonzero(access.control)
            outputs.append((access.weight * backend.live_value).sum())
        sum(outputs).backward()
    assert backend.live_value.grad.abs() > 0
    assert backend.plan_intervention == "normal" and "_access" not in backend.__dict__
    assert len(replay.seen) == 3 and replay.density_reconstruction_max_weight_error == 0
    for key in normal:
        if key.startswith("access/") and key.rsplit("/", 1)[-1] in {
            "density",
            "weight",
            "support",
            "edge_access",
            "near",
        }:
            np.testing.assert_array_equal(normal[key], recorded.arrays[key])
    assert backend.calls == 6  # One current native access per replay, no physical cache replay.


@pytest.mark.parametrize(
    "field",
    [
        "receivers",
        "source_ids",
        "source_coords",
        "source_measures",
        "source_valid",
        "diagnostics/pair_valid",
        "weight",
        "support",
        "near",
    ],
)
def test_reference_mismatches_fail_and_restore_mode_and_methods(field):
    backend = Backend().eval()
    normal = reference(backend, phases=1)
    key = f"access/P0/MM/00000/{field}"
    normal[key] = normal[key].copy()
    normal[key].flat[0] = not normal[key].flat[0] if normal[key].dtype == np.bool_ else normal[key].flat[0] + 1
    with pytest.raises(ValueError, match="mismatch"), FixedReferenceAccessReplay(backend, normal):
        plan = backend.prepare(0)["hypergraph_plan"]
        backend._access(plan, plan.source_coords["M"], "MM")
    assert backend.plan_intervention == "normal" and "_access" not in backend.__dict__


def test_missing_extra_and_soft_streams_rejected_and_restored():
    backend = Backend().eval()
    normal = reference(backend, phases=1)
    with pytest.raises(ValueError, match="Missing native"), FixedReferenceAccessReplay(backend, normal):
        pass
    assert backend.plan_intervention == "normal" and "_access" not in backend.__dict__
    with pytest.raises(ValueError, match="Extra or mismatched"), FixedReferenceAccessReplay(backend, normal):
        plan = backend.prepare(0)["hypergraph_plan"]
        backend._access(replace(plan, phase=1), plan.source_coords["M"], "MM")
    assert backend.plan_intervention == "normal" and "_access" not in backend.__dict__
    backend.permission_mode = "soft"
    with pytest.raises(ValueError, match="hard frozen"), FixedReferenceAccessReplay(backend, normal):
        backend._access(plan, plan.source_coords["M"], "MM")
    assert backend.plan_intervention == "normal" and "_access" not in backend.__dict__
    backend.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        FixedReferenceAccessReplay(backend, normal)
