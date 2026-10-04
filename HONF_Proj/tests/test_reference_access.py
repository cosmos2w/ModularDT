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
        "density",
        "weight",
        "support",
        "near",
    ],
)
@pytest.mark.parametrize("mode", ["control_identity", "normal_fixed_actions"])
def test_reference_mismatches_fail_and_restore_mode_and_methods(field, mode):
    backend = Backend().eval()
    normal = reference(backend, phases=1)
    key = f"access/P0/MM/00000/{field}"
    normal[key] = normal[key].copy()
    normal[key].flat[0] = not normal[key].flat[0] if normal[key].dtype == np.bool_ else normal[key].flat[0] + 1
    with pytest.raises(ValueError, match="mismatch"), FixedReferenceAccessReplay(backend, normal, mode=mode):
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


@pytest.mark.parametrize("mode", ["normal_fixed_actions", "full_access_fixed_controls", "geometry_reference_actions"])
def test_reference_controls_kept_normal_bias_mode_and_physical_gradient(mode):
    backend = Backend().eval()
    normal = reference(backend)
    backend.flip = True
    with FixedReferenceAccessReplay(backend, normal, mode=mode), TypedWorkEvidenceRecorder(backend) as recorded:
        for phase in range(3):
            plan = backend.prepare(phase)["hypergraph_plan"]
            plan = replace(plan, controls={name: value * 7 for name, value in plan.controls.items()})
            value = backend._access(plan, plan.source_coords["M"], "MM")
            assert backend.plan_intervention == "normal"
            (value.weight * backend.live_value).sum().backward()
            if mode == "full_access_fixed_controls":
                assert torch.equal(value.weight, value.diagnostics["pair_valid"].float())
                assert torch.equal(value.support, value.diagnostics["pair_valid"])
    assert backend.live_value.grad.abs() > 0 and backend.plan_intervention == "normal"
    for phase in range(3):
        prefix = f"access/P{phase}/MM/00000"
        if mode in {"normal_fixed_actions", "full_access_fixed_controls"}:
            for key in (
                "control_probe",
                "control_receiver_max_abs",
                "control_receiver_nonzero_count",
                "control_source_channel_mean",
            ):
                np.testing.assert_array_equal(normal[f"{prefix}/{key}"], recorded.arrays[f"{prefix}/{key}"])
        if mode == "normal_fixed_actions":
            for key in ("density", "weight", "support", "edge_access", "near"):
                np.testing.assert_array_equal(normal[f"{prefix}/{key}"], recorded.arrays[f"{prefix}/{key}"])


@pytest.mark.parametrize(
    "name",
    ["control_probe", "control_receiver_max_abs", "control_receiver_nonzero_count", "control_source_channel_mean"],
)
@pytest.mark.parametrize("mode", ["normal_fixed_actions", "full_access_fixed_controls"])
def test_reference_control_reconstruction_guards(name, mode):
    backend = Backend().eval()
    normal = reference(backend, phases=1)
    normal[f"access/P0/MM/00000/{name}"].flat[0] += 1
    with (
        pytest.raises(ValueError, match="reconstructed control"),
        FixedReferenceAccessReplay(backend, normal, mode=mode),
    ):
        plan = backend.prepare(0)["hypergraph_plan"]
        backend._access(plan, plan.source_coords["M"], "MM")
    assert backend.plan_intervention == "normal" and "_access" not in backend.__dict__


class EmptyBackend(Backend):
    def __init__(self, kind, source_count):
        super().__init__()
        self.kind, self.source_count = kind, source_count
        self.route = "QM" if kind == "M" else "QE"

    def prepare(self, phase):
        counts = {kind: self.source_count if kind == self.kind else 2 for kind in ("M", "E")}
        return {
            "hypergraph_plan": TypedHypergraphState(
                {self.route: torch.ones(1, 1, self.source_count)},
                {self.route: torch.ones(1, 1, 16)},
                torch.zeros(1, 1, 2),
                torch.ones(1, 1),
                {kind: torch.zeros(1, count, 2) for kind, count in counts.items()},
                {kind: torch.ones(1, count) for kind, count in counts.items()},
                {kind: torch.ones(1, count, dtype=torch.bool) for kind, count in counts.items()},
                {kind: torch.arange(count)[None] for kind, count in counts.items()},
                phase=phase,
            )
        }

    def _access(self, plan, receivers, mechanism):
        self.calls += 1
        return source_moments(
            torch.ones(1, receivers.shape[1], 1),
            plan.memberships[mechanism],
            plan.controls[mechanism],
            plan.source_measures[self.kind],
            plan.source_valid[self.kind],
            pair_valid=torch.ones(1, receivers.shape[1], self.source_count, dtype=torch.bool),
        )


@pytest.mark.parametrize("kind,receivers,sources", [("M", 2, 0), ("E", 2, 0), ("M", 0, 2), ("E", 0, 2), ("M", 0, 0)])
@pytest.mark.parametrize("mode", ["control_identity", "normal_fixed_actions", "full_access_fixed_controls", "geometry_reference_actions"])
def test_empty_source_types_and_receiver_axes_record_and_reconstruct(kind, receivers, sources, mode):
    backend = EmptyBackend(kind, sources).eval()
    queries = torch.zeros(1, receivers, 2)
    with TypedWorkEvidenceRecorder(backend) as normal:
        plan = backend.prepare(0)["hypergraph_plan"]
        backend._access(plan, queries, backend.route)
    prefix = f"access/P0/{backend.route}/00000"
    assert normal.arrays[f"{prefix}/control_receiver_max_abs"].shape == (1, receivers)
    assert normal.arrays[f"{prefix}/control_source_channel_mean"].shape == (1, sources, 16)
    for name in ("control_probe", "control_receiver_max_abs", "control_source_channel_mean"):
        assert np.isfinite(normal.arrays[f"{prefix}/{name}"]).all()
    with FixedReferenceAccessReplay(backend, normal.arrays, mode=mode), TypedWorkEvidenceRecorder(backend) as result:
        plan = backend.prepare(0)["hypergraph_plan"]
        access = backend._access(plan, queries, backend.route)
        assert access.weight.shape == (1, receivers, sources)
    assert backend.calls == 2 and "_access" not in backend.__dict__ and "prepare" not in backend.__dict__
    assert backend.plan_intervention == "normal"
    for name in (
        "control_probe",
        "control_receiver_max_abs",
        "control_receiver_nonzero_count",
        "control_source_channel_mean",
    ):
        np.testing.assert_array_equal(normal.arrays[f"{prefix}/{name}"], result.arrays[f"{prefix}/{name}"])
    with (
        pytest.raises(ValueError, match="receivers mismatch"),
        FixedReferenceAccessReplay(backend, normal.arrays, mode=mode),
    ):
        backend._access(plan, torch.zeros(1, receivers + 1, 2), backend.route)
    assert "_access" not in backend.__dict__ and backend.plan_intervention == "normal"
