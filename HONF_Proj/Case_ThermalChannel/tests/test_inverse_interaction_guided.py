from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
import channelthermal.inverse.interaction_guided as interaction_guided_module

from honf_inverse_core.contracts import NamedContext, PhysicalDesign
from honf_inverse_core.request_schema import GeometryConstraints
from channelthermal.inverse.interaction_guided import (
    DecisionObservation,
    InverseStudyConfig,
    PRESSURE_DROP_DEFINITION,
    PreparedResponseBaseline,
    _build_candidate_pool_with_feasibility_redraw,
    _matched_size_schedule,
    decode_thermal_field_response,
    relative_pressure_limit,
    run_matched_inverse_design_study,
    smooth_peak_temperature,
)


MODULE_IDS = ("layout-0001:module:0", "layout-0001:module:1", "layout-0001:module:2")


@dataclass(frozen=True)
class _Factor:
    factor_id: str
    donor_ids: tuple[str, ...]


class _ToyFactorOracle:
    """Small factor-only operator fixture; dense trial calls are forbidden."""

    def __init__(self) -> None:
        self.factors = (
            _Factor("pair_ab", MODULE_IDS[:2]),
            _Factor("pair_ac", (MODULE_IDS[0], MODULE_IDS[2])),
            _Factor("pair_bc", MODULE_IDS[1:]),
        )
        self.factor_calls: list[tuple[str, tuple[str, ...]]] = []

    def prepare_baseline(self, design, context, module_ids_by_slot) -> PreparedResponseBaseline:
        return PreparedResponseBaseline(cache=None, queries=None, factors=self.factors)

    def rank_factors(self, prepared: PreparedResponseBaseline) -> Sequence[_Factor]:
        return tuple(sorted(prepared.factors, key=lambda factor: factor.factor_id))

    def predict_factor_response(
        self,
        prepared: PreparedResponseBaseline,
        factor: _Factor,
        delta_by_module_id: Mapping[str, np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        assert set(delta_by_module_id) == set(factor.donor_ids)
        self.factor_calls.append((factor.factor_id, tuple(sorted(delta_by_module_id))))
        solid_delta = np.zeros((6, 1), dtype=np.float64)
        if len(factor.donor_ids) == 1:
            module_id = factor.donor_ids[0]
            a = np.asarray(delta_by_module_id[module_id], dtype=np.float64) / np.asarray([10.0, 4.0])
            mixed = -0.5 * float(np.sum(a))
            for slot, active_id in enumerate(MODULE_IDS):
                if active_id == module_id:
                    solid_delta[2 * slot + 1, 0] = mixed
        else:
            first, second = factor.donor_ids
            a = np.asarray(delta_by_module_id[first], dtype=np.float64) / np.asarray([10.0, 4.0])
            b = np.asarray(delta_by_module_id[second], dtype=np.float64) / np.asarray([10.0, 4.0])
            mixed = -12.0 * float(np.dot(a, b))
            for slot, module_id in enumerate(MODULE_IDS):
                if module_id in factor.donor_ids:
                    solid_delta[2 * slot + 1, 0] = mixed
        return {
            "fluid_fields": np.zeros((3, 11, 5), dtype=np.float64),
            "solid_temperature": solid_delta,
            "interface": np.zeros((6, 2), dtype=np.float64),
        }

    def predict_trial(self, *args, **kwargs):
        raise AssertionError("A dense perturbed-design bypass was called.")


class _MixedOrderToyOracle(_ToyFactorOracle):
    def __init__(self) -> None:
        super().__init__()
        self.factors = (
            _Factor("unary_a", (MODULE_IDS[0],)),
            _Factor("unary_b", (MODULE_IDS[1],)),
            _Factor("unary_c", (MODULE_IDS[2],)),
            *self.factors,
        )


class _ReanchoringToyOracle(_ToyFactorOracle):
    def __init__(self, reanchor_history: list[tuple[float, ...]] | None = None) -> None:
        super().__init__()
        self.reanchor_history = reanchor_history if reanchor_history is not None else []

    def with_new_baseline(self, design, observation, context, module_ids_by_slot):
        del observation, context, module_ids_by_slot
        self.reanchor_history.append(tuple(np.asarray(design.module_centers).reshape(-1).tolist()))
        return _ReanchoringToyOracle(self.reanchor_history)


class _OutOfEvidenceToyOracle(_ToyFactorOracle):
    def factor_validity_status(self, prepared, factor, delta_by_module_id):
        del prepared, factor, delta_by_module_id
        return False


class _NoFactorToyOracle(_ToyFactorOracle):
    def __init__(self) -> None:
        super().__init__()
        self.factors = ()


def _context() -> NamedContext:
    return NamedContext(
        feature_names=("module_radius", "domain_length_x", "domain_length_y"),
        vector=np.asarray([0.25, 10.0, 4.0], dtype=np.float32),
        schema_name="test_context",
    )


def _design() -> PhysicalDesign:
    return PhysicalDesign(
        module_centers=np.asarray([[2.0, 1.0], [5.0, 1.0], [8.0, 1.0]], dtype=np.float32),
        module_present=np.ones(3, dtype=np.float32),
        heat_powers=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )


def _geometry_constraints() -> GeometryConstraints:
    return GeometryConstraints(
        module_count_min=3,
        module_count_max=3,
        minimum_center_distance=1.0,
        wall_clearance=0.2,
        inlet_clearance=0.5,
        outlet_clearance=0.5,
        total_heat_range=(30.0, 30.0),
    )


def _baseline_observation(source: str = "stored_reference") -> DecisionObservation:
    x_line = np.arange(11, dtype=np.float64)
    y_line = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    y_grid, x_grid = np.meshgrid(y_line, x_line, indexing="ij")
    grid_xy = np.stack([x_grid, y_grid], axis=-1)
    fields = np.zeros((3, 11, 5), dtype=np.float64)
    fields[..., 2] = -x_grid
    fields[..., 4] = 0.0
    solid = np.asarray([[9.0], [10.0], [11.0], [12.0], [14.0], [15.0]], dtype=np.float64)
    return DecisionObservation(
        module_temperature_by_id={MODULE_IDS[0]: 10.0, MODULE_IDS[1]: 12.0, MODULE_IDS[2]: 15.0},
        pressure_drop=10.0,
        pressure_drop_units="Pa",
        evidence_source=source,  # type: ignore[arg-type]
        output_state={
            "fluid_fields": fields,
            "grid_xy": grid_xy,
            "grid_valid_mask": np.ones((3, 11), dtype=bool),
            "solid_temperature": solid,
            "solid_module_ids": np.repeat(np.asarray(MODULE_IDS, dtype=object), 2),
            "active_module_ids": MODULE_IDS,
            "interface": np.zeros((6, 2), dtype=np.float64),
            "channel_order": ("u", "v", "p", "omega", "temperature"),
        },
    )


def _config(**overrides: Any) -> InverseStudyConfig:
    values: dict[str, Any] = {
        "tau_temperature": 0.5,
        "tau_source": "preselected fixture scale; production tau must come from training-only temperatures",
        "pressure_drop_limit": 10.5,
        "pressure_limit_basis": "relative benchmark, baseline plus five percent",
        "pressure_drop_units": "Pa",
        "trust_radius_normalized": 0.04,
        "candidate_budget": 3,
        "max_reference_trials_per_policy": 3,
        "max_updates_per_policy": 3,
        "validation_mode": "exhaustive_pool",
        "random_seed": 93,
    }
    values.update(overrides)
    return InverseStudyConfig(**values)


def _synthetic_evaluator(
    design: PhysicalDesign,
    context: NamedContext,
    module_ids_by_slot: Sequence[str | None],
) -> DecisionObservation:
    del context
    moved = {
        module_id: float(design.module_centers[slot, 0])
        for slot, module_id in enumerate(module_ids_by_slot)
        if module_id is not None
    }
    temperatures = {
        module_id: value + 0.08 * (moved[module_id] - baseline_x)
        for module_id, value, baseline_x in zip(MODULE_IDS, (10.0, 12.0, 15.0), (2.0, 5.0, 8.0))
    }
    return DecisionObservation(
        module_temperature_by_id=temperatures,
        pressure_drop=10.0 + 0.01 * sum(abs(moved[key] - baseline_x) for key, baseline_x in zip(MODULE_IDS, (2.0, 5.0, 8.0))),
        pressure_drop_units="Pa",
        evidence_source="analytic_synthetic",
        pressure_drop_definition=PRESSURE_DROP_DEFINITION,
        provenance={"fixture": "analytic quadratic response"},
    )


def _run(*, oracle=None, evaluator=None, config=None, decision_decoder=decode_thermal_field_response):
    return run_matched_inverse_design_study(
        initial_design=_design(),
        initial_observation=_baseline_observation(),
        context=_context(),
        module_ids_by_slot=MODULE_IDS,
        geometry_constraints=_geometry_constraints(),
        config=config or _config(),
        response_oracle=oracle or _ToyFactorOracle(),
        evaluator=evaluator or _synthetic_evaluator,
        evaluator_source="analytic_synthetic",
        decision_decoder=decision_decoder,
    )


def test_smooth_peak_is_stable_and_approaches_true_max() -> None:
    temperatures = np.asarray([10.0, 12.0, 15.0])
    assert smooth_peak_temperature(temperatures, tau=0.5) == pytest.approx(14.4517, abs=2.0e-3)
    assert smooth_peak_temperature([1.0e5, 1.0e5 - 1.0], tau=0.25) == pytest.approx(99999.8315, abs=2.0e-3)


def test_relative_pressure_limit_is_explicit_and_positive_baseline_only() -> None:
    assert relative_pressure_limit(10.0, allowance_fraction=0.05) == pytest.approx(10.5)
    with pytest.raises(ValueError, match="positive baseline"):
        relative_pressure_limit(-10.0)


def test_thermal_decoder_adds_only_factor_outputs_and_uses_fixed_pressure_sections() -> None:
    baseline = _baseline_observation()
    increments = {
        "fluid_fields": np.zeros((3, 11, 5), dtype=np.float64),
        "solid_temperature": np.zeros((6, 1), dtype=np.float64),
        "interface": np.zeros((6, 2), dtype=np.float64),
    }
    increments["solid_temperature"][1, 0] = 2.0
    estimate = decode_thermal_field_response(baseline, increments, _design(), _context())
    assert estimate.module_temperature_by_id[MODULE_IDS[0]] == pytest.approx(12.0)
    assert estimate.pressure_drop == pytest.approx(10.0)
    assert estimate.pressure_drop_definition == PRESSURE_DROP_DEFINITION
    assert "fluid_fields" in estimate.output_state and "interface" in estimate.output_state


def test_zero_factor_response_keeps_pressure_fixed_when_trial_disk_crosses_section() -> None:
    baseline = _baseline_observation()
    original = _design()
    centers = original.module_centers.copy()
    centers[0] = np.asarray([0.1, 0.0], dtype=np.float32)
    boundary_crossing_design = PhysicalDesign(
        module_centers=centers,
        module_present=original.module_present.copy(),
        heat_powers=original.heat_powers.copy(),
    )
    zero_factor_response = {
        "fluid_fields": np.zeros_like(baseline.output_state["fluid_fields"]),
        "solid_temperature": np.zeros_like(baseline.output_state["solid_temperature"]),
        "interface": np.zeros_like(baseline.output_state["interface"]),
    }

    estimate = decode_thermal_field_response(
        baseline, zero_factor_response, boundary_crossing_design, _context()
    )

    assert estimate.pressure_drop == pytest.approx(baseline.pressure_drop)
    assert estimate.pressure_section_mask_changed is True


def test_section_mask_change_is_unscored_and_never_spends_a_reference_call() -> None:
    def unsupported_decoder(baseline, increments, design, context):
        estimate = decode_thermal_field_response(baseline, increments, design, context)
        return replace(estimate, pressure_section_mask_changed=True)

    config = _config(
        candidate_budget=1,
        max_reference_trials_per_policy=1,
        max_updates_per_policy=1,
        validation_mode="selected_only",
    )
    result = _run(config=config, decision_decoder=unsupported_decoder)

    for policy in result.policies.values():
        assert policy.evaluator_calls == 0
        assert policy.status == "no_supported_pressure_candidates"
        candidate = policy.candidates[0]
        assert candidate.pressure_prediction_status == "candidate_section_mask_changed"
        assert candidate.score_status == "unsupported_trial_section_geometry"
        assert candidate.predicted_feasible is None
        assert candidate.predicted_pressure_drop is None


def test_feasibility_redraw_keeps_slot_group_size_and_radius_fixed(monkeypatch) -> None:
    oracle = _MixedOrderToyOracle()
    prepared = oracle.prepare_baseline(_design(), _context(), MODULE_IDS)
    schedule = _matched_size_schedule(prepared.factors, candidate_budget=3, active_module_count=3)
    config = _config(candidate_redraw_max_seed_offsets=4, candidate_redraw_seed_offset_stride=104729)
    original_evaluate = interaction_guided_module.evaluate_geometry
    seen = 0

    def invalidate_first_proposal(design, context, constraints):
        nonlocal seen
        seen += 1
        if seen == 1:
            return SimpleNamespace(valid=False, total_violation=1.0)
        return original_evaluate(design, context, constraints)

    monkeypatch.setattr(interaction_guided_module, "evaluate_geometry", invalidate_first_proposal)
    candidates, factor_calls, geometry_rejections, audit = _build_candidate_pool_with_feasibility_redraw(
        policy="graph_guided",
        policy_index=0,
        update_index=0,
        design=_design(),
        baseline=_baseline_observation(),
        context=_context(),
        module_ids_by_slot=MODULE_IDS,
        geometry_constraints=_geometry_constraints(),
        config=config,
        prepared=prepared,
        size_schedule=schedule,
        active_ids=MODULE_IDS,
        domain_lengths=(10.0, 4.0),
        pressure_scale=10.0,
        response_oracle=oracle,
        decision_decoder=decode_thermal_field_response,
    )

    assert len(candidates) == 3
    assert factor_calls > 0 and geometry_rejections == 1
    assert audit[0]["selected_seed_offset"] == 1
    assert audit[0]["status"] == "selected"
    assert [attempt["seed_offset"] for attempt in audit[0]["attempts"]] == [0, 1]
    assert len({tuple(attempt["source_module_ids"]) for attempt in audit[0]["attempts"]}) == 1
    assert len({attempt["normalized_dimension"] for attempt in audit[0]["attempts"]}) == 1
    assert all(attempt["normalized_radius"] == pytest.approx(config.trust_radius_normalized) for attempt in audit[0]["attempts"])
    assert all(item.estimate is not None for item in candidates)


def test_thermal_decoder_ignores_unresolved_solid_queries() -> None:
    baseline = _baseline_observation()
    output = dict(baseline.output_state)
    solid = np.asarray(output["solid_temperature"]).copy()
    solid[4, 0] = np.nan
    output["solid_temperature"] = solid
    output["solid_valid_mask"] = np.asarray([[1], [1], [1], [1], [0], [1]], dtype=bool)
    fluid = np.asarray(output["fluid_fields"]).copy()
    grid_valid = np.asarray(output["grid_valid_mask"]).copy()
    fluid[1, 5, 2] = np.nan
    grid_valid[1, 5] = False
    output["fluid_fields"] = fluid
    output["grid_valid_mask"] = grid_valid
    baseline = DecisionObservation(
        module_temperature_by_id=baseline.module_temperature_by_id,
        pressure_drop=baseline.pressure_drop,
        pressure_drop_units=baseline.pressure_drop_units,
        evidence_source=baseline.evidence_source,
        output_state=output,
    )
    estimate = decode_thermal_field_response(
        baseline,
        {"solid_temperature": np.zeros_like(solid)},
        _design(),
        _context(),
    )
    assert estimate.module_temperature_by_id[MODULE_IDS[2]] == pytest.approx(15.0)


def test_matched_policies_have_identical_dimensions_radius_and_no_trial_bypass() -> None:
    oracle = _ToyFactorOracle()
    result = _run(oracle=oracle)
    dimensions = {
        policy: tuple(candidate.normalized_dimension for candidate in outcome.candidates)
        for policy, outcome in result.policies.items()
    }
    assert dimensions["graph_guided"] == dimensions["size_matched_random"] == dimensions["ungrouped_local"]
    radii = {
        policy: [candidate.normalized_radius for candidate in outcome.candidates]
        for policy, outcome in result.policies.items()
    }
    for policy_radii in radii.values():
        np.testing.assert_allclose(policy_radii, 0.04, rtol=1.0e-6, atol=1.0e-6)
    assert all(candidate.normalized_dimension == 4 for outcome in result.policies.values() for candidate in outcome.candidates)
    assert result.proposal_dimension_by_index == (4, 4, 4)
    assert result.policy_pools_are_common is False
    assert "not physical solver validation" in result.evidence_interpretation
    assert all(outcome.evaluator_calls == 3 for outcome in result.policies.values())
    assert all(outcome.reference_calls == 0 and outcome.synthetic_evaluations == 3 for outcome in result.policies.values())
    assert oracle.factor_calls
    assert all(
        candidate.evidence_validity_status == "unknown"
        for outcome in result.policies.values()
        for candidate in outcome.candidates
    )


def test_matched_mixed_order_schedule_includes_unary_and_pair_candidates() -> None:
    config = _config(
        candidate_budget=6,
        max_reference_trials_per_policy=1,
        max_updates_per_policy=1,
        validation_mode="selected_only",
    )
    result = _run(oracle=_MixedOrderToyOracle(), config=config)
    assert result.proposal_order_counts == {1: 3, 2: 3}
    assert "baseline input-only support score" in result.factor_source_selection_basis
    assert "unmaterialized source sets are not objective-pre-screened" in result.factor_source_selection_basis
    expected = (4, 2, 4, 2, 4, 2)
    assert result.proposal_dimension_by_index == expected
    for policy in result.policies.values():
        assert tuple(candidate.normalized_dimension for candidate in policy.candidates) == expected
        assert sum(len(candidate.source_module_ids) == 1 for candidate in policy.candidates) == 3
        assert sum(len(candidate.source_module_ids) == 2 for candidate in policy.candidates) == 3
        np.testing.assert_allclose(
            [candidate.normalized_radius for candidate in policy.candidates], 0.04, rtol=1.0e-6, atol=1.0e-6
        )


def test_inverse_trace_marks_proposals_outside_measured_neighborhood() -> None:
    result = _run(oracle=_OutOfEvidenceToyOracle())
    assert all(
        candidate.evidence_validity_status == "outside_measured_neighborhood"
        for outcome in result.policies.values()
        for candidate in outcome.candidates
    )


def test_empty_graph_is_reported_as_unavailable_with_fallbacks() -> None:
    result = _run(oracle=_NoFactorToyOracle())
    assert result.graph_guidance_status == "unavailable"
    assert result.graph_factor_fallback_count == 3
    graph_candidates = result.policies["graph_guided"].candidates
    assert all(candidate.graph_factor_status == "missing_order_fallback" for candidate in graph_candidates)


def test_initial_feasible_reference_remains_incumbent_when_trials_are_worse() -> None:
    baseline = _baseline_observation()

    def worse_evaluator(design, context, module_ids_by_slot):
        del design, context, module_ids_by_slot
        return DecisionObservation(
            module_temperature_by_id={key: value + 1.0 for key, value in baseline.module_temperature_by_id.items()},
            pressure_drop=10.0,
            pressure_drop_units="Pa",
            evidence_source="analytic_synthetic",
            output_state=baseline.output_state,
        )

    config = _config(
        candidate_budget=1,
        max_reference_trials_per_policy=1,
        max_updates_per_policy=1,
        validation_mode="selected_only",
    )
    result = _run(evaluator=worse_evaluator, config=config)
    for policy in result.policies.values():
        assert policy.accepted_updates == 0
        assert policy.best_evaluated_feasible_observation is baseline or (
            policy.best_evaluated_feasible_observation is not None
            and policy.best_evaluated_feasible_observation.module_temperature_by_id
            == baseline.module_temperature_by_id
        )
        assert policy.best_physical_feasible_observation is not None
        assert policy.best_physical_feasible_observation.module_temperature_by_id == baseline.module_temperature_by_id


def test_accepted_updates_reanchor_policy_locally() -> None:
    oracle = _ReanchoringToyOracle()
    baseline = _baseline_observation()
    evaluated = 0

    def improving_evaluator(design, context, module_ids_by_slot):
        nonlocal evaluated
        del design, context, module_ids_by_slot
        evaluated += 1
        return DecisionObservation(
            module_temperature_by_id={key: value - 0.5 * evaluated for key, value in baseline.module_temperature_by_id.items()},
            pressure_drop=10.0,
            pressure_drop_units="Pa",
            evidence_source="analytic_synthetic",
            output_state=baseline.output_state,
        )

    config = _config(
        candidate_budget=1,
        max_reference_trials_per_policy=2,
        max_updates_per_policy=2,
        validation_mode="selected_only",
    )
    result = _run(oracle=oracle, evaluator=improving_evaluator, config=config)
    assert all(policy.accepted_updates == 2 for policy in result.policies.values()), {
        name: (policy.accepted_updates, policy.status, [(trial.status, trial.exception, trial.accepted) for trial in policy.trials])
        for name, policy in result.policies.items()
    }
    assert len(oracle.reanchor_history) == 2 * len(result.policies)
    assert all(policy.candidate_rank_spearman is None for policy in result.policies.values())


def test_factor_ranker_is_required_when_candidate_budget_truncates_graph() -> None:
    oracle = _ToyFactorOracle()
    oracle.rank_factors = None  # type: ignore[assignment]
    config = _config(candidate_budget=2, max_reference_trials_per_policy=2, max_updates_per_policy=2)
    with pytest.raises(ValueError, match="input-only baseline factor ranker"):
        _run(oracle=oracle, config=config)


def test_failed_reference_trials_consume_the_declared_budget() -> None:
    def failing_evaluator(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    baseline = _baseline_observation()
    config = _config(
        candidate_budget=2,
        max_reference_trials_per_policy=2,
        max_updates_per_policy=2,
        validation_mode="exhaustive_pool",
    )
    result = run_matched_inverse_design_study(
        initial_design=_design(),
        initial_observation=baseline,
        context=_context(),
        module_ids_by_slot=MODULE_IDS,
        geometry_constraints=_geometry_constraints(),
        config=config,
        response_oracle=_ToyFactorOracle(),
        evaluator=failing_evaluator,
        evaluator_source="reference_solver",
    )
    for outcome in result.policies.values():
        assert outcome.evaluator_calls == outcome.reference_calls == outcome.physical_solver_calls == 2
        assert len(outcome.trials) == 2
        assert all(trial.status == "failed" for trial in outcome.trials)
        assert outcome.accepted_updates == 0


def test_policy_aware_evaluator_receives_identity_without_call_count_guessing() -> None:
    class PolicyAwareEvaluator:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def __call__(self, *args, **kwargs):  # pragma: no cover - fallback is forbidden by this fixture.
            raise AssertionError("The policy-aware callback should be used.")

        def evaluate_for_policy(self, policy, design, context, module_ids_by_slot):
            self.seen.append(policy)
            return _synthetic_evaluator(design, context, module_ids_by_slot)

    evaluator = PolicyAwareEvaluator()
    config = _config(
        candidate_budget=1,
        max_reference_trials_per_policy=1,
        max_updates_per_policy=1,
        validation_mode="selected_only",
    )
    result = _run(oracle=_MixedOrderToyOracle(), evaluator=evaluator, config=config)
    assert evaluator.seen == ["graph_guided", "size_matched_random", "ungrouped_local"]
    assert all(policy.evaluator_calls == 1 for policy in result.policies.values())


def test_new_design_evaluator_cannot_be_labeled_stored_reference() -> None:
    with pytest.raises(ValueError, match="not a newly perturbed-design evaluator"):
        run_matched_inverse_design_study(
            initial_design=_design(),
            initial_observation=_baseline_observation(),
            context=_context(),
            module_ids_by_slot=MODULE_IDS,
            geometry_constraints=_geometry_constraints(),
            config=_config(),
            response_oracle=_ToyFactorOracle(),
            evaluator=_synthetic_evaluator,
            evaluator_source="stored_reference",
        )
