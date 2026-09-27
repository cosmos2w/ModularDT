from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from channelthermal.inverse.interaction_guided import DecisionEstimate, DecisionObservation
from channelthermal.inverse.local_interface_contract import LocalInterfacePlan
from channelthermal.inverse.native_corrected import (
    NativeBaselineCorrection,
    ThermalNativeQuantityPredictor,
    baseline_correct_quantities,
    baseline_correct_tensor_quantities,
)
from channelthermal.inverse.native_matched import NativeCandidate, run_matched_native_round
from channelthermal.response_control.contracts import AbsolutePrediction, DesignInput, RoleQuery

from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    CandidateNode,
    CaseLocalReceiverTree,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_inverse_core.contracts import NamedContext, PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints

IDS = ("module-a", "module-b")


def _design(*, x: float = 1.0, heat: tuple[float, float] = (10.0, 10.0)) -> PhysicalDesign:
    return PhysicalDesign(
        module_centers=np.asarray([[x, 1.0], [3.0, 1.0]], dtype=np.float32),
        module_present=np.ones(2, dtype=np.float32),
        heat_powers=np.asarray(heat, dtype=np.float32),
    )


def _context() -> NamedContext:
    return NamedContext(
        feature_names=("domain_length_x", "domain_length_y", "module_radius"),
        vector=np.asarray([5.0, 2.0, 0.2], dtype=np.float32),
        schema_name="native-correction-test",
    )


def _reference(a: float = 100.0, b: float = 200.0, p: float = 5.0) -> DecisionObservation:
    return DecisionObservation(
        module_temperature_by_id={IDS[0]: a, IDS[1]: b},
        pressure_drop=p,
        pressure_drop_units="Pa",
        evidence_source="reference_solver",
    )


def _estimate(a: float, b: float, p: float) -> DecisionEstimate:
    return DecisionEstimate(
        module_temperature_by_id={IDS[0]: a, IDS[1]: b},
        pressure_drop=p,
        pressure_drop_units="Pa",
    )


def test_constant_absolute_bias_cancels_per_module_and_pressure() -> None:
    corrected = baseline_correct_quantities(
        _reference(), _estimate(110.0, 220.0, 8.0), _estimate(111.0, 219.0, 8.02)
    )
    assert corrected.module_temperature_by_id == pytest.approx({IDS[0]: 101.0, IDS[1]: 199.0})
    assert corrected.pressure_drop == pytest.approx(5.02)


def test_correction_rejects_identity_units_and_functional_mismatch() -> None:
    baseline = _reference()
    predicted = _estimate(110.0, 220.0, 8.0)
    with pytest.raises(ValueError, match="module IDs"):
        baseline_correct_quantities(
            baseline, predicted,
            DecisionEstimate({IDS[0]: 111.0}, 8.02, "Pa"),
        )
    with pytest.raises(ValueError, match="units"):
        baseline_correct_quantities(
            baseline, predicted, DecisionEstimate({IDS[0]: 111.0, IDS[1]: 219.0}, 8.02, "bar")
        )


class _NativePredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[float, tuple[float, float], object | None]] = []

    def __call__(self, design, context, module_ids_by_slot, *, frozen_topology):
        assert context.schema_name == "native-correction-test"
        assert tuple(module_ids_by_slot) == IDS
        x = float(design.module_centers[0, 0])
        heat = tuple(float(value) for value in design.heat_powers)
        self.calls.append((x, heat, frozen_topology))
        topology_effect = 0.5 if frozen_topology == ("right",) else 0.0
        return _estimate(
            110.0 + x + 0.2 * heat[0] + topology_effect,
            220.0 - 0.5 * x + 0.2 * heat[1] + topology_effect,
            8.0 + 0.01 * x,
        )


def test_trial_refreshes_continuous_state_then_reanchors_after_reference() -> None:
    predictor = _NativePredictor()
    topology_calls: list[float] = []

    def topology_factory(design, context, module_ids_by_slot):
        assert context.schema_name == "native-correction-test"
        assert tuple(module_ids_by_slot) == IDS
        x = float(design.module_centers[0, 0])
        topology_calls.append(x)
        return ("right",) if x >= 1.5 else ("left",)

    anchor = NativeBaselineCorrection(
        predictor=predictor,
        baseline_design=_design(),
        reference_baseline=_reference(),
        context=_context(),
        module_ids_by_slot=IDS,
        topology_factory=topology_factory,
    )
    trial_design = _design(x=2.0, heat=(11.0, 9.0))
    trial = anchor.evaluate(trial_design)
    assert topology_calls == [1.0]
    assert predictor.calls == [
        (1.0, (10.0, 10.0), ("left",)),
        (2.0, (11.0, 9.0), ("left",)),
    ]
    assert trial.baseline_corrected.module_temperature_by_id[IDS[0]] == pytest.approx(101.2)
    assert trial.baseline_corrected.pressure_drop == pytest.approx(5.01)

    switched = anchor.recomputed_topology_discrepancy(trial_design, trial)
    assert topology_calls == [1.0, 2.0]
    assert switched.module_temperature_by_id[IDS[0]] - trial.native_absolute.module_temperature_by_id[IDS[0]] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="new reference observation"):
        anchor.reanchor(trial_design, anchor.reference_baseline)

    accepted_reference = _reference(a=101.5, b=199.4, p=5.01)
    next_anchor = anchor.reanchor(trial_design, accepted_reference)
    assert next_anchor is not anchor
    assert topology_calls == [1.0, 2.0, 2.0]
    assert predictor.calls[-1] == (2.0, (11.0, 9.0), ("right",))
    assert next_anchor.evaluate(trial_design).baseline_corrected.module_temperature_by_id == pytest.approx(
        accepted_reference.module_temperature_by_id
    )


def test_teacher_output_cannot_be_used_as_measured_anchor() -> None:
    predictor = _NativePredictor()
    with pytest.raises(ValueError, match="independent reference"):
        NativeBaselineCorrection(
            predictor=predictor,
            baseline_design=_design(),
            reference_baseline=replace(_reference(), evidence_source="surrogate_teacher"),
            context=_context(),
            module_ids_by_slot=IDS,
        )
    assert predictor.calls == []


def test_frozen_plan_rejects_a_hash_only_policy_substitute() -> None:
    class HashOnlyPlan:
        def canonical_hash(self) -> str:
            return "f" * 64

    queries = {
        "fluid_fields": RoleQuery(
            "fluid_fields", torch.tensor([[0.0, 0.0]]),
            ("p",), ("Pa",), None, "eulerian",
        ),
        "solid_temperature": RoleQuery(
            "solid_temperature", torch.zeros((2, 2)),
            ("temperature",), ("degC",), (0, 1), "solid_material_normalized_xy",
        ),
    }
    with pytest.raises(ValueError, match="typed mechanism cover plan"):
        LocalInterfacePlan.from_anchor(
            baseline=_design(), module_ids_by_slot=IDS, context=_context(),
            role_queries=queries, checkpoint_hash="a" * 64,
            cover_plans=(HashOnlyPlan(),), max_position_delta=0.2,
        )


def test_native_bridge_uses_fresh_design_and_keeps_tensor_gradient() -> None:
    class AbsoluteProbe:
        def __init__(self) -> None:
            self.positions: list[float] = []
            self.last_positions: torch.Tensor | None = None

        def __call__(self, design, context, role_queries):
            assert context["domain_length_x"] == 5.0
            assert set(role_queries) == {"fluid_fields", "solid_temperature"}
            x = design.module_positions[0, 0]
            self.positions.append(float(x.detach()))
            self.last_positions = design.module_positions
            pressure = torch.stack([10.0 + 0.1 * x, x.new_tensor(2.0)])[:, None]
            solid = (
                design.module_heating + design.module_positions[:, 0]
            )[:, None]
            return AbsolutePrediction({"fluid_fields": pressure, "solid_temperature": solid})

    queries = {
        "fluid_fields": RoleQuery(
            "fluid_fields", torch.tensor([[0.0, 0.0], [5.0, 0.0]]),
            ("p",), ("Pa",), None, "eulerian",
        ),
        "solid_temperature": RoleQuery(
            "solid_temperature", torch.zeros((2, 2)),
            ("temperature",), ("degC",), (0, 1), "solid_material_normalized_xy",
        ),
    }
    operator = AbsoluteProbe()
    predictor = ThermalNativeQuantityPredictor(operator, queries, device="cpu")
    first = predictor(_design(), _context(), IDS, frozen_topology=None)
    second = predictor(_design(x=2.0), _context(), IDS, frozen_topology=None)
    assert operator.positions == [1.0, 2.0]
    assert second.module_temperature_by_id[IDS[0]] - first.module_temperature_by_id[IDS[0]] == pytest.approx(1.0)
    assert second.pressure_drop - first.pressure_drop == pytest.approx(0.1, abs=1.0e-6)
    tensors = predictor.tensor_quantities(
        _design(x=2.0), _context(), IDS, requires_grad=True
    )
    assert operator.last_positions is not None
    derivative = torch.autograd.grad(
        tensors.module_peak_temperature[IDS[0]] + tensors.pressure_drop,
        operator.last_positions,
    )[0]
    assert tensors.module_peak_temperature[IDS[0]].requires_grad
    assert tensors.pressure_drop.requires_grad
    assert derivative[0, 0].item() == pytest.approx(1.1, abs=1.0e-5)
    with pytest.raises(TypeError, match="LocalInterfacePlan"):
        predictor(_design(), _context(), IDS, frozen_topology=("unconnected",))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native inverse gradient test requires CUDA")
def test_frozen_native_cover_uses_live_caller_tensors_and_rejects_invalid_trial() -> None:
    device = torch.device("cuda:0")  # The test runner exposes only physical GPU2.

    class MaskedProbe:
        def __init__(self) -> None:
            self.masked_calls = 0

        def __call__(self, design, context, role_queries):
            x = design.module_positions[0, 0]
            p = torch.stack((10.0 + 0.1 * x, x.new_tensor(2.0)))[:, None]
            solid = (design.module_heating + design.module_positions[:, 0])[:, None]
            return AbsolutePrediction({"fluid_fields": p, "solid_temperature": solid})

        def predict_with_frozen_topology(self, design, context, role_queries, *, fixed_cover_plans):
            assert len(fixed_cover_plans) == 1 and isinstance(fixed_cover_plans[0], MechanismPlan)
            self.masked_calls += 1
            x = design.module_positions[0, 0]
            p = torch.stack((10.0 + 0.3 * x, x.new_tensor(2.0)))[:, None]
            solid = (design.module_heating + 2.0 * design.module_positions[:, 0])[:, None]
            return AbsolutePrediction({"fluid_fields": p, "solid_temperature": solid})

    queries = {
        "fluid_fields": RoleQuery(
            "fluid_fields", torch.tensor([[0.0, 0.0], [5.0, 0.0]], device=device),
            ("p",), ("Pa",), None, "eulerian",
        ),
        "solid_temperature": RoleQuery(
            "solid_temperature", torch.zeros((2, 2), device=device),
            ("temperature",), ("degC",), (0, 1), "solid_material_normalized_xy",
        ),
    }
    checkpoint_hash = "a" * 64
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor([[1.0, 1.0], [3.0, 1.0]], device=device),
        weights=torch.ones(2, device=device),
        roles=torch.ones(2, dtype=torch.int64, device=device),
        coordinate_scale=torch.ones(2, device=device),
    )
    tree = CaseLocalReceiverTree(universe, (CandidateNode((0, 1)),), 0.06, False)
    cover = MechanismPlan.full_access(tree, torch.ones(2, device=device), 1)
    topology = LocalInterfacePlan.from_anchor(
        baseline=_design(), module_ids_by_slot=IDS, context=_context(), role_queries=queries,
        checkpoint_hash=checkpoint_hash, cover_plans=(cover,), max_position_delta=0.2,
    )
    operator = MaskedProbe()
    predictor = ThermalNativeQuantityPredictor(
        operator, queries, device=device, checkpoint_hash=checkpoint_hash,
    )
    anchor = NativeBaselineCorrection(
        predictor=predictor, baseline_design=_design(), reference_baseline=_reference(),
        context=_context(), module_ids_by_slot=IDS, topology_factory=lambda *_: topology,
    )
    result = anchor.evaluate(_design(x=1.1))
    assert operator.masked_calls == 2
    assert result.baseline_corrected.module_temperature_by_id[IDS[0]] == pytest.approx(100.2)
    assert result.baseline_corrected.pressure_drop == pytest.approx(5.03)

    positions = torch.tensor(_design(x=1.1).module_centers, device=device, requires_grad=True)
    heating = torch.tensor(_design().heat_powers, device=device, requires_grad=True)
    live = DesignInput(positions, heating, torch.ones(2, device=device, dtype=torch.bool))
    model_baseline = predictor.tensor_quantities(_design(), _context(), IDS, frozen_topology=topology)
    model_trial = predictor.tensor_quantities_from_input(
        live, _context(), IDS, module_family_id="thermal_disk", frozen_topology=topology,
    )
    corrected = baseline_correct_tensor_quantities(_reference(), model_baseline, model_trial)
    gradients = torch.autograd.grad(
        corrected.module_peak_temperature[IDS[0]] + corrected.pressure_drop,
        (positions, heating),
    )
    assert gradients[0][0, 0].item() == pytest.approx(2.3, abs=1e-5)
    assert gradients[1][0].item() == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="trust region"):
        anchor.evaluate(_design(x=1.3))
    with pytest.raises(ValueError, match="heating"):
        anchor.evaluate(_design(x=1.1, heat=(11.0, 10.0)))
    changed_queries = dict(queries)
    changed_queries["fluid_fields"] = RoleQuery(
        "fluid_fields", queries["fluid_fields"].query_features.clone() + 0.01,
        ("p",), ("Pa",), None, "eulerian",
    )
    with pytest.raises(ValueError, match="receiver coordinates"):
        topology.validate_trial(
            _design(), _context(), changed_queries, checkpoint_hash=checkpoint_hash,
            module_ids_by_slot=IDS,
        )
    with pytest.raises(ValueError, match="module-to-slot"):
        topology.validate_trial(
            _design(), _context(), queries, checkpoint_hash=checkpoint_hash,
            module_ids_by_slot=tuple(reversed(IDS)),
        )


def _position_candidates(first: tuple[int, int], second: tuple[int, int]):
    candidates = []
    for sign_a, sign_b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        centers = np.asarray(_design().module_centers, dtype=np.float32).copy()
        centers[first] += 0.1 * sign_a
        centers[second] += 0.1 * sign_b
        candidates.append(NativeCandidate(
            f"{sign_a:+d}{sign_b:+d}",
            PhysicalDesign(centers, np.ones(2, np.float32), np.asarray([10, 10], np.float32)),
        ))
    return tuple(candidates)


class _MatchedPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, design, context, module_ids_by_slot, *, frozen_topology):
        assert frozen_topology is None
        assert tuple(module_ids_by_slot) == IDS
        assert context.schema_name == "native-correction-test"
        self.calls += 1
        centers = design.module_centers
        delta = centers - _design().module_centers
        return _estimate(
            110.0,
            220.0 - 2.0 * delta[0, 0] - delta[1, 1]
            - 0.5 * delta[1, 0] - 0.25 * delta[0, 1],
            8.0 + 0.05 * delta[0, 0],
        )


def _constraints() -> GeometryConstraints:
    return GeometryConstraints(2, 2, 1.0, 0.0, 0.0, 0.0)


def test_matched_native_round_charges_one_reference_and_rejects_false_feasible() -> None:
    predictor = _MatchedPredictor()
    candidates = {
        "graph_guided": _position_candidates((0, 0), (1, 1)),
        "size_matched_random": _position_candidates((0, 1), (1, 0)),
    }
    attempted = []
    selected_before_reference = []

    def selection_callback(selections):
        assert attempted == []
        assert set(selections) == set(candidates)
        selected_before_reference.extend(selections)

    def reference_trial(policy, candidate):
        attempted.append((policy, candidate.label))
        if policy == "graph_guided":
            return _reference(b=199.0, p=5.2)
        return _reference(b=199.5, p=5.0)

    results = run_matched_native_round(
        predictor=predictor,
        baseline_design=_design(), reference_baseline=_reference(),
        context=_context(), module_ids_by_slot=IDS,
        candidates_by_policy=candidates, reference_trial=reference_trial,
        constraints=_constraints(), pressure_limit=5.1, trust_radius=0.1,
        selection_callback=selection_callback,
    )
    assert set(selected_before_reference) == set(candidates)
    assert predictor.calls == 10  # one fresh baseline plus four trials per arm
    assert len(attempted) == 2
    assert all(result.reference_attempts == 1 for result in results.values())
    assert results["graph_guided"].selected_predicted_feasible
    assert not results["graph_guided"].accepted
    assert results["graph_guided"].suggested_next_radius == pytest.approx(0.05)
    assert results["size_matched_random"].accepted


def test_matched_native_round_rejects_unequal_budgets_before_model_calls() -> None:
    predictor = _MatchedPredictor()
    graph = _position_candidates((0, 0), (1, 1))
    with pytest.raises(ValueError, match="same nonzero candidate count"):
        run_matched_native_round(
            predictor=predictor,
            baseline_design=_design(), reference_baseline=_reference(),
            context=_context(), module_ids_by_slot=IDS,
            candidates_by_policy={"graph_guided": graph, "random": graph[:2]},
            reference_trial=lambda _policy, _candidate: _reference(),
            constraints=_constraints(), pressure_limit=5.1, trust_radius=0.1,
        )
    assert predictor.calls == 0
