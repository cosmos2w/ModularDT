"""The ThermalChannel inverse bridge keeps trial design out of the baseline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from channelthermal.inverse.context import CONTEXT_FEATURE_NAMES
from channelthermal.inverse.response_factor_oracle import (
    HonfThermalResponseOracle,
    ResponseFeatureSpec,
    build_response_model_inputs,
)

from honf_forward_core.interaction_response.factor_operator import AnchoredResponseFactorOperator
from honf_forward_core.interaction_response.organizer import InputOnlySupportScorer
from honf_forward_core.interaction_response.types import ResponseFactor, ValidityNeighborhood
from honf_forward_core.interaction_response.support_search import enumerate_unary_pair_candidates
from honf_inverse_core.contracts import NamedContext, PhysicalDesign


def _fixture() -> tuple[PhysicalDesign, NamedContext, tuple[str, ...], dict[str, object]]:
    design = PhysicalDesign(
        module_centers=np.asarray([[3.0, 1.0], [5.0, 2.0], [7.0, 3.0]], dtype=np.float32),
        module_present=np.ones(3),
        heat_powers=np.asarray([2.0, 3.0, 4.0]),
    )
    context = NamedContext(
        feature_names=CONTEXT_FEATURE_NAMES,
        vector=np.asarray([50.0, 1.0, 0.02, 0.10, 0.01, 1.0, 1.0, 0.45, 10.0, 4.0]),
        schema_name="thermalchannel_inverse_context",
    )
    grid = np.asarray([[0.0, 1.0], [0.0, 2.0], [10.0, 1.0], [10.0, 2.0]], dtype=np.float32)
    state: dict[str, object] = {
        "fluid_fields": np.zeros((4, 5), dtype=np.float32),
        "grid_xy": grid,
        "grid_valid_mask": np.ones(4, dtype=bool),
        "channel_order": ("u", "v", "p", "omega", "temperature"),
        "interface": np.zeros((3, 2), dtype=np.float32),
        "interface_angles": np.asarray([0.0, np.pi / 2, np.pi], dtype=np.float32),
        "interface_module_ids": ("a", "b", "c"),
        "solid_temperature": np.asarray([[10.0], [12.0], [14.0]], dtype=np.float32),
        "solid_local_xy": np.zeros((3, 2), dtype=np.float32),
        "solid_module_ids": ("a", "b", "c"),
    }
    return design, context, ("a", "b", "c"), state


def test_baseline_cache_uses_geometry_but_no_solved_values() -> None:
    design, context, ids, state = _fixture()
    feature_spec = ResponseFeatureSpec(heat_scale=4.0)
    cache, queries = build_response_model_inputs(design, context, ids, state, feature_spec, device="cpu")
    changed = dict(state)
    changed["fluid_fields"] = np.full((4, 5), 10000.0, dtype=np.float32)
    changed["interface"] = np.full((3, 2), 10000.0, dtype=np.float32)
    changed["solid_temperature"] = np.full((3, 1), 10000.0, dtype=np.float32)
    cache_changed, queries_changed = build_response_model_inputs(
        design, context, ids, changed, feature_spec, device="cpu"
    )
    assert torch.equal(cache.module_features, cache_changed.module_features)
    assert torch.equal(cache.baseline_design, cache_changed.baseline_design)
    assert torch.equal(cache.baseline_context, cache_changed.baseline_context)
    for role in queries:
        assert torch.equal(queries[role].features, queries_changed[role].features)


def test_masked_baseline_unknowns_are_not_treated_as_observed_zero() -> None:
    design, context, ids, state = _fixture()
    field = np.asarray(state["fluid_fields"]).copy()
    field[1, :] = np.nan
    state["fluid_fields"] = field
    state["grid_valid_mask"] = np.asarray([True, False, True, True])
    interface = np.asarray(state["interface"]).copy()
    interface[1, 0] = np.nan
    state["interface"] = interface
    state["interface_valid_mask"] = np.asarray([[True, True], [False, True], [True, True]])
    solid = np.asarray(state["solid_temperature"]).copy()
    solid[2, 0] = np.nan
    state["solid_temperature"] = solid
    state["solid_valid_mask"] = np.asarray([[True], [True], [False]])
    _, queries = build_response_model_inputs(
        design, context, ids, state, ResponseFeatureSpec(heat_scale=4.0), device="cpu"
    )
    assert not bool(queries["fluid_fields"].mask[0, 1])
    assert not bool(queries["interface"].mask[0, 1])
    assert not bool(queries["solid_temperature"].mask[0, 2])
    state["grid_valid_mask"] = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="Observed baseline fluid"):
        build_response_model_inputs(
            design, context, ids, state, ResponseFeatureSpec(heat_scale=4.0), device="cpu"
        )


def test_oracle_rejects_rebased_trial_and_ignores_other_donor_delta() -> None:
    design, context, ids, state = _fixture()
    device = torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cpu")
    model = AnchoredResponseFactorOperator(
        module_feature_dim=2, design_dim=3, context_dim=10, delta_dim=3, hidden_dim=24
    ).to(device)
    scorer = InputOnlySupportScorer(
        module_feature_dim=2, design_dim=3, context_dim=10, hidden_dim=24
    ).to(device)
    oracle = HonfThermalResponseOracle(
        operator=model,
        scorer=scorer,
        baseline_design=design,
        baseline_context=context,
        module_ids_by_slot=ids,
        baseline_output_state=state,
        feature_spec=ResponseFeatureSpec(heat_scale=4.0),
        support_threshold=-100.0,
        query_chunk_size=2,
    )
    prepared = oracle.prepare_baseline(design, context, ids)
    unary = next(factor for factor in prepared.factors if factor.donor_ids == ("a",))
    first = oracle.predict_factor_response(prepared, unary, {"a": np.asarray([0.10, 0.0])})
    second = oracle.predict_factor_response(
        prepared, unary, {"a": np.asarray([0.10, 0.0]), "b": np.asarray([100.0, 0.0])}
    )
    assert set(first) == {"fluid_fields", "interface", "solid_temperature"}
    for role in first:
        np.testing.assert_array_equal(first[role], second[role])
    forged = replace(unary, donor_ids=("b",))
    with pytest.raises(ValueError, match="exact factor object"):
        oracle.predict_factor_response(prepared, forged, {"b": np.asarray([0.10, 0.0])})
    moved = PhysicalDesign(
        module_centers=design.module_centers + np.asarray([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        module_present=design.module_present,
        heat_powers=design.heat_powers,
    )
    with pytest.raises(ValueError, match="rebuilt"):
        oracle.prepare_baseline(moved, context, ids)


def test_oracle_validity_is_unknown_without_atlas_and_preserves_measured_axis_radii() -> None:
    design, context, ids, state = _fixture()
    model = AnchoredResponseFactorOperator(
        module_feature_dim=2, design_dim=3, context_dim=10, delta_dim=3, hidden_dim=24
    )
    scorer = InputOnlySupportScorer(
        module_feature_dim=2, design_dim=3, context_dim=10, hidden_dim=24
    )
    feature_spec = ResponseFeatureSpec(heat_scale=4.0)
    oracle = HonfThermalResponseOracle(
        operator=model,
        scorer=scorer,
        baseline_design=design,
        baseline_context=context,
        module_ids_by_slot=ids,
        baseline_output_state=state,
        feature_spec=feature_spec,
        support_threshold=-100.0,
    )
    prepared = oracle.prepare_baseline(design, context, ids)
    unary = next(factor for factor in prepared.factors if factor.donor_ids == ("a",))
    assert unary.validity.max_abs_delta_by_module["a"] == (None, None, None)
    assert oracle.factor_validity_status(prepared, unary, {"a": np.asarray([0.1, 0.0])}) is None

    measured_atlas_factor = ResponseFactor(
        factor_id=unary.factor_id,
        donor_ids=unary.donor_ids,
        output_roles=unary.output_roles,
        validity=ValidityNeighborhood(
            max_abs_delta_by_module={"a": (0.02, None, None)},
            anchor_family_ids=("training-anchor-family",),
        ),
        evidence_sources=("reference_solver",),
        evidence_anchor_ids=("training-anchor-0001",),
    )
    measured_oracle = HonfThermalResponseOracle(
        operator=model,
        scorer=scorer,
        baseline_design=design,
        baseline_context=context,
        module_ids_by_slot=ids,
        baseline_output_state=state,
        feature_spec=feature_spec,
        support_threshold=-100.0,
        atlas_factors=(measured_atlas_factor,),
    )
    measured_prepared = measured_oracle.prepare_baseline(design, context, ids)
    measured_unary = next(factor for factor in measured_prepared.factors if factor.factor_id == unary.factor_id)
    assert measured_unary.validity.max_abs_delta_by_module["a"] == (0.02, None, None)
    assert measured_unary.evidence_sources == ("reference_solver",)
    assert measured_unary.evidence_anchor_ids == ("training-anchor-0001",)
    assert measured_oracle.factor_validity_status(
        measured_prepared, measured_unary, {"a": np.asarray([0.1, 0.0])}
    ) is None
    assert measured_oracle.factor_validity_status(
        measured_prepared, measured_unary, {"a": np.asarray([0.3, 0.0])}
    ) is False

    radius_scaled_oracle = HonfThermalResponseOracle(
        operator=model,
        scorer=scorer,
        baseline_design=design,
        baseline_context=context,
        module_ids_by_slot=ids,
        baseline_output_state=state,
        feature_spec=ResponseFeatureSpec(heat_scale=4.0, position_delta_scale=(0.45, 0.45)),
        support_threshold=-100.0,
        atlas_factors=(measured_atlas_factor,),
    )
    radius_scaled_prepared = radius_scaled_oracle.prepare_baseline(design, context, ids)
    radius_scaled_unary = next(
        factor for factor in radius_scaled_prepared.factors if factor.factor_id == unary.factor_id
    )
    assert radius_scaled_oracle.factor_validity_status(
        radius_scaled_prepared, radius_scaled_unary, {"a": np.asarray([0.1, 0.0])}
    ) is False


def test_all_unaries_plus_top1_pair_retains_unit_weights_and_freezes_pair_per_baseline() -> None:
    design, context, ids, state = _fixture()
    device = torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cpu")
    model = AnchoredResponseFactorOperator(
        module_feature_dim=2, design_dim=3, context_dim=10, delta_dim=3, hidden_dim=24
    ).to(device)
    scorer = InputOnlySupportScorer(
        module_feature_dim=2, design_dim=3, context_dim=10, hidden_dim=24
    ).to(device)
    oracle = HonfThermalResponseOracle(
        operator=model,
        scorer=scorer,
        baseline_design=design,
        baseline_context=context,
        module_ids_by_slot=ids,
        baseline_output_state=state,
        feature_spec=ResponseFeatureSpec(heat_scale=4.0),
        support_threshold=0.0,
        support_selection_mode="all_unaries_plus_top1_pair",
    )
    prepared = oracle.prepare_baseline(design, context, ids)
    all_factors = enumerate_unary_pair_candidates(ids, tuple(prepared.queries))
    all_scores = prepared.metadata["score_by_factor_id"]
    pair_ids = [factor.factor_id for factor in all_factors if len(factor.donor_ids) == 2]
    expected_top1 = min(pair_ids, key=lambda factor_id: (-all_scores[factor_id], factor_id))
    retained_ids = {factor.factor_id for factor in prepared.factors}
    unary_ids = {factor.factor_id for factor in all_factors if len(factor.donor_ids) == 1}
    assert retained_ids == unary_ids | {expected_top1}
    assert prepared.metadata["support_selection_mode"] == "all_unaries_plus_top1_pair"
    assert prepared.metadata["selected_top1_pair_factor_id"] == expected_top1
    assert prepared.metadata["actual_retained_factor_count"] == len(ids) + 1
    assert prepared.metadata["active_unary"] == len(ids)
    assert prepared.metadata["active_joint"] == 1
    weights = prepared.metadata["weight_by_factor_id"]
    assert all(weights[factor_id] == 1.0 for factor_id in retained_ids)
    assert all(weights[factor_id] == 0.0 for factor_id in set(all_scores) - retained_ids)


def test_unmeasured_pair_pressure_increment_is_zero_but_unary_pressure_is_retained(monkeypatch) -> None:
    design, context, ids, state = _fixture()
    device = torch.device("cuda:2")
    model = AnchoredResponseFactorOperator(
        module_feature_dim=2, design_dim=3, context_dim=10, delta_dim=3, hidden_dim=24
    ).to(device)
    scorer = InputOnlySupportScorer(
        module_feature_dim=2, design_dim=3, context_dim=10, hidden_dim=24
    ).to(device)
    oracle = HonfThermalResponseOracle(
        operator=model,
        scorer=scorer,
        baseline_design=design,
        baseline_context=context,
        module_ids_by_slot=ids,
        baseline_output_state=state,
        feature_spec=ResponseFeatureSpec(heat_scale=4.0),
        support_threshold=0.0,
        support_selection_mode="all_unaries_plus_top1_pair",
    )
    prepared = oracle.prepare_baseline(design, context, ids)
    pair = next(factor for factor in prepared.factors if len(factor.donor_ids) == 2)
    unary = next(factor for factor in prepared.factors if len(factor.donor_ids) == 1)

    def fixed_response(cache, factor, delta_by_module_id, queries, query_chunk_size=None):
        return {
            "fluid_fields": torch.full((1, 4, 5), 7.0, device=device),
            "interface": torch.full((1, 3, 2), 5.0, device=device),
            "solid_temperature": torch.full((1, 3, 1), 3.0, device=device),
        }

    monkeypatch.setattr(model, "predict_factor_response", fixed_response)
    deltas = {module_id: np.asarray([0.01, 0.0]) for module_id in pair.donor_ids}
    pair_response = oracle.predict_factor_response(prepared, pair, deltas)
    unary_response = oracle.predict_factor_response(
        prepared, unary, {unary.donor_ids[0]: np.asarray([0.01, 0.0])}
    )
    pressure_channel = ("u", "v", "p", "omega", "temperature").index("p")
    assert np.all(pair_response["fluid_fields"][:, pressure_channel] == 0.0)
    assert np.all(unary_response["fluid_fields"][:, pressure_channel] == 7.0)
    assert np.all(pair_response["interface"][:, 1] == 5.0)  # q_normal remains visible, not zero-filled.
    assert (
        prepared.metadata["pair_pressure_increment_policy"]
        == "zero_pair_fluid_pressure_increment_assumed_analytic_wake_additivity_null; "
        "pair_pressure_channel_was_unmeasured_and_is_not_learned"
    )
