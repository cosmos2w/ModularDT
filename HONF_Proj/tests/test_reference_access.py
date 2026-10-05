"""Fixed normal permission replay leaves physical values live and rejects drift."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from honf_forward_core.evaluation.reference_access import FixedReferenceAccessReplay, ProjectedRouteUniformReplay
from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder
from honf_forward_core.interface_fields.typed_hypergraph_state import (
    ProjectedSourceAccess,
    TypedHypergraphState,
    prepare_projected_action,
    source_moments,
)


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


def test_materialized_reference_context_preserves_unrelated_hooks_and_outputs_on_exception():
    class LazyBackend(Backend):
        def __init__(self):
            super().__init__()
            self.head = nn.LazyLinear(1)

        def forward(self):
            plan = self.prepare(0)["hypergraph_plan"]
            access = self._access(plan, plan.source_coords["M"], "MM")
            return self.head(access.weight * self.live_value)

    backend = LazyBackend().eval()
    lazy_hooks = dict(backend.head._forward_pre_hooks)
    assert lazy_hooks
    backend()  # Ordinary materialization owns removal of its lazy hook.
    assert not backend.head._forward_pre_hooks
    external_handle = backend.register_forward_pre_hook(lambda *_: None)
    try:
        inventory = {name: dict(child._forward_pre_hooks) for name, child in backend.named_modules()}
        normal = reference(backend, phases=1)
        original = backend().detach().clone()
        with FixedReferenceAccessReplay(backend, normal, mode="normal_fixed_actions"):
            backend()
        assert inventory == {name: dict(child._forward_pre_hooks) for name, child in backend.named_modules()}
        torch.testing.assert_close(backend(), original, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="intentional"), FixedReferenceAccessReplay(
            backend, normal, mode="normal_fixed_actions"
        ):
            backend()
            raise RuntimeError("intentional context exit")
        assert inventory == {name: dict(child._forward_pre_hooks) for name, child in backend.named_modules()}
        assert "_access" not in backend.__dict__ and backend.plan_intervention == "normal"
        torch.testing.assert_close(backend(), original, rtol=0, atol=0)
    finally:
        external_handle.remove()


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


class ProjectedBackend(Backend):
    def __init__(self, tau="QM", empty=False):
        super().__init__()
        self.tau, self.empty = tau, empty
        self.control_execution = "projected"
        self.control_gain = nn.ModuleDict({tau: nn.Linear(16, 2 if tau == "QE" else 1)})
        self.control_score = nn.Linear(16, 2)
        self.near_offset = 0.0
        with torch.no_grad():
            for projection in (self.control_gain[tau], self.control_score):
                projection.weight.fill_(0.03)
                projection.bias.fill_(0.1)

    def prepare(self, phase):
        plan = super().prepare(phase)["hypergraph_plan"]
        return {"hypergraph_plan": replace(plan,
            memberships={self.tau: torch.tensor([[[1., .2], [.1, 1.]]])},
            controls={self.tau: torch.arange(32).reshape(1, 2, 16).float() / 16 + phase},
            source_valid={kind: torch.zeros_like(value) if self.empty else value
                          for kind, value in plan.source_valid.items()})}

    def actions(self, plan):
        tau = self.tau
        gain = self.control_gain[tau]
        weight, bias = gain.weight, gain.bias
        if tau == "QE":
            weight = torch.cat((self.control_score.weight, weight), 0)
            bias = torch.cat((self.control_score.bias, bias), 0)
        return {tau: prepare_projected_action(plan.memberships[tau], plan.controls[tau], weight, bias)}

    def _numerical_access(self, plan, receivers, mechanism, actions, **kwargs):
        self.calls += 1
        x = receivers[..., 0]
        edge = torch.stack((1 - x, x), -1)
        kind = "E" if mechanism == "QE" else "M"
        near = x[..., None].expand(-1, -1, plan.source_measures[kind].shape[-1]) * 0.25 + self.near_offset
        access = source_moments(edge, plan.memberships[mechanism], plan.controls[mechanism],
            plan.source_measures[kind], plan.source_valid[kind], near=near,
            prepared_action=actions[mechanism], include_diagnostics=True)
        access.diagnostics["receiver_measures"] = (plan.source_measures["M"] if mechanism == "MM"
                                                  else torch.ones_like(x))
        return access


def projected_reference(backend):
    chunks = [torch.tensor([[[0., 0.], [.2, 0.]]]), torch.tensor([[[1., 0.]]])]
    accesses = []
    with TypedWorkEvidenceRecorder(backend) as recorder:
        for phase in (0, 1):
            plan = backend.prepare(phase)["hypergraph_plan"]
            actions = backend.actions(plan)
            for receivers in chunks:
                accesses.append(backend._numerical_access(plan, receivers, backend.tau, actions))
    return recorder.arrays, chunks, accesses


@pytest.mark.parametrize("tau", ["QM", "QE"])
def test_projected_route_uniform_uses_full_stream_measure_mean_and_preserves_permissions(tau, monkeypatch):
    import honf_forward_core.evaluation.reference_access as module
    backend = ProjectedBackend(tau).eval()
    reference_arrays, chunks, normal = projected_reference(backend)
    original = module.source_moments

    def projected_only(*args, **kwargs):
        assert kwargs.get("prepared_action") is not None
        value = original(*args, **kwargs)
        assert value.projected.shape[-1] < 16
        return value

    monkeypatch.setattr(module, "source_moments", projected_only)
    with ProjectedRouteUniformReplay(backend, reference_arrays) as replay:
        changed = []
        for phase in (0, 1):
            plan = backend.prepare(phase)["hypergraph_plan"]
            actions = backend.actions(plan)
            source_mass = plan.source_measures["E" if tau == "QE" else "M"]
            masses = [access.weight.double() * source_mass.double()[:, None]
                      for access in normal[phase * 2:phase * 2 + 2]]
            denominator = sum(mass.sum((1, 2)) for mass in masses)
            expected = sum((mass[..., None] * access.projected.double()).sum((1, 2))
                           for mass, access in zip(masses, normal[phase * 2:phase * 2 + 2])) / denominator[:, None]
            chunk_mean = (masses[0][..., None] * normal[phase * 2].projected.double()).sum((1, 2)) / masses[0].sum((1, 2))[:, None]
            assert not torch.allclose(expected, chunk_mean)
            torch.testing.assert_close(replay.means[phase, tau], expected.float())
            torch.testing.assert_close(replay.denominators[phase, tau], denominator)
            for ordinal, points in enumerate(chunks):
                access = backend._numerical_access(plan, points, tau, actions)
                assert isinstance(access, ProjectedSourceAccess)
                for name in ("density", "weight", "support", "edge_access", "near"):
                    assert torch.equal(getattr(access, name), getattr(normal[phase * 2 + ordinal], name))
                torch.testing.assert_close(access.projected, expected.float()[:, None, None].expand_as(access.projected))
                changed.append(access)
    assert "_numerical_access" not in backend.__dict__
    assert backend.control_execution == "projected" and backend.plan_intervention == "normal"
    coefficients = torch.tensor([1., 4.])[None, None, :, None]
    normal_forward = sum((value.weight[..., None] * (1 + value.projected.tanh()) * coefficients).sum() for value in normal)
    changed_forward = sum((value.weight[..., None] * (1 + value.projected.tanh()) * coefficients).sum() for value in changed)
    assert not torch.isclose(normal_forward, changed_forward)
    assert replay.action_statistics[f"P0/{tau}"]["repeated_physical_reads_counted"]
    assert replay.action_statistics[f"P0/{tau}"]["weighted_affine_change_rms_by_case_channel"][0][0] > 0


def test_projected_uniform_zero_support_and_ownership_exception_restoration():
    backend = ProjectedBackend(empty=True).eval()
    reference_arrays, chunks, _ = projected_reference(backend)
    original = backend._numerical_access
    backend._numerical_access = original  # Existing instance ownership must survive.
    with pytest.raises(RuntimeError, match="intentional"), ProjectedRouteUniformReplay(backend, reference_arrays) as replay:
        plan = backend.prepare(0)["hypergraph_plan"]
        value = backend._numerical_access(plan, chunks[0], backend.tau, backend.actions(plan))
        assert not value.support.any() and not value.weight.any() and not value.projected.any()
        assert not replay.denominators[0, backend.tau].any()
        raise RuntimeError("intentional")
    assert backend.__dict__["_numerical_access"] is original
    assert backend.control_execution == "projected"


def test_projected_uniform_unequal_source_atom_refinement_preserves_route_means():
    original_backend = ProjectedBackend().eval()
    original_reference, chunks, _ = projected_reference(original_backend)
    backend = ProjectedBackend().eval()
    backend.load_state_dict(original_backend.state_dict())
    original_prepare = backend.prepare

    def refined_prepare(phase):
        plan = original_prepare(phase)["hypergraph_plan"]
        return {"hypergraph_plan": replace(plan,
            memberships={tau: value.repeat_interleave(2, -1) for tau, value in plan.memberships.items()},
            source_coords={kind: value.repeat_interleave(2, 1) for kind, value in plan.source_coords.items()},
            source_measures={kind: value.repeat_interleave(2, 1) * torch.tensor([.3, .7, .3, .7])
                             for kind, value in plan.source_measures.items()},
            source_valid={kind: value.repeat_interleave(2, 1) for kind, value in plan.source_valid.items()},
            source_ids={kind: torch.arange(4)[None] for kind in plan.source_ids})}

    backend.prepare = refined_prepare
    refined_reference, _, _ = projected_reference(backend)
    with ProjectedRouteUniformReplay(original_backend, original_reference) as original:
        original_means = {key: value.clone() for key, value in original.means.items()}
        for phase in (0, 1):
            plan = original_backend.prepare(phase)["hypergraph_plan"]
            for points in chunks:
                original_backend._numerical_access(plan, points, original_backend.tau, original_backend.actions(plan))
    with ProjectedRouteUniformReplay(backend, refined_reference) as refined:
        for route, mean in original_means.items():
            torch.testing.assert_close(mean, refined.means[route], atol=1e-6, rtol=1e-6)
        for phase in (0, 1):
            plan = backend.prepare(phase)["hypergraph_plan"]
            for points in chunks:
                backend._numerical_access(plan, points, backend.tau, backend.actions(plan))


def test_projected_uniform_legacy_native_receiver_measure_fallback():
    backend = ProjectedBackend("MM").eval()
    with TypedWorkEvidenceRecorder(backend) as recorder:
        plan = backend.prepare(0)["hypergraph_plan"]
        normal = backend._numerical_access(plan, plan.source_coords["M"], "MM", backend.actions(plan))
    reference_arrays = {key: value for key, value in recorder.arrays.items()
                        if not key.endswith("/receiver_measures")}
    mass = plan.source_measures["M"][:, :, None] * plan.source_measures["M"][:, None] * normal.weight
    expected = (mass[..., None] * normal.projected).sum((1, 2)) / mass.sum((1, 2))[:, None]
    with ProjectedRouteUniformReplay(backend, reference_arrays) as replay:
        value = backend._numerical_access(plan, plan.source_coords["M"], "MM", backend.actions(plan))
        torch.testing.assert_close(value.projected, expected[:, None, None].expand_as(value.projected))
    assert replay.action_statistics["P0/MM"]["receiver_measure_sources"] == ["legacy_physical_receiver_measure"]


@pytest.mark.parametrize("drift", ["projection", "source_ids", "source_measures", "receivers", "near", "missing", "extra"])
def test_projected_uniform_strict_reference_and_call_guards(drift):
    backend = ProjectedBackend().eval()
    reference_arrays, chunks, _ = projected_reference(backend)
    if drift == "projection":
        with torch.no_grad():
            backend.control_gain[backend.tau].bias.add_(0.1)
    with pytest.raises(ValueError), ProjectedRouteUniformReplay(backend, reference_arrays):
        for phase in (0, 1):
            plan = backend.prepare(phase)["hypergraph_plan"]
            if drift in ("source_ids", "source_measures"):
                changed = {kind: value + 1 for kind, value in getattr(plan, drift).items()}
                plan = replace(plan, **{drift: changed})
            if drift == "near":
                backend.near_offset = 0.1
            points = chunks if drift != "missing" else chunks[:1]
            if drift == "extra":
                points = chunks + chunks[:1]
            for receivers in points:
                if drift == "receivers":
                    receivers = receivers + 0.01
                backend._numerical_access(plan, receivers, backend.tau, backend.actions(plan))
    assert "_numerical_access" not in backend.__dict__
