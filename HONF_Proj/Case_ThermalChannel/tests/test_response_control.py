from __future__ import annotations

import json
import os
import pickle
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from channelthermal.interaction_evidence import (
    DesignState,
    EvidenceSource,
    EvidenceSplit,
    MeasuredQuantity,
    ModuleState,
    OperatingContext,
    PhysicalSolveOutput,
    ResponseStencil,
    RoleOutput,
    SolveRecord,
    SolveStatus,
    load_solve_record,
    persist_solve_record,
)
from channelthermal.response_control import (
    AbsolutePrediction,
    DesignInput,
    DifferentiableThermalOperator,
    MixedResponseSpec,
    RoleQuery,
    StagedTrainingConfig,
    ThermalLossScales,
    TrainingStage,
    calibrate_operator_weights,
    check_pressure_peak_ad_fd,
    evaluate_absolute_record,
    evaluate_stencil,
    load_staged_training_config,
    predict_stencil,
    pressure_drop_from_field,
    reduce_native_thermal_quantities,
    restore_checkpoint_payload,
    role_queries_from_record,
    run_paired_staged_fits,
    run_staged_fit,
)
from channelthermal.response_control import paired as paired_module
from channelthermal.response_control.historical import _sample_value_record
from channelthermal.response_control.losses import (
    FixedHeatNullControl,
    compute_stencil_loss_terms,
    fixed_heat_control_loss_terms,
    fixed_heat_material_peak_coverage,
)
from channelthermal.response_control.resume_provenance import (
    _validate_fit_arm_gate_accounting,
    file_sha256,
    training_config_mapping,
    validate_checkpoint_only_review_provenance,
    validate_paired_resume_provenance,
)
from channelthermal.response_control.runner import (
    _R0_RESPONSE_TERMS,
    _atomic_json,
    _audit_named_buffers,
    _combine_r0_response_gradient,
    _configure_native_expanded_response_interface_scope,
    _configure_native_nonlinear_interface_scope,
    _configure_native_output_head_scope,
    _dataclass_record,
    _historical_calibration_case_map,
    _json_default,
    _load_expanded_probe_evidence,
    _load_frozen_loss_scales,
    _load_frozen_response_weights,
    _load_r0_projection_evidence,
    _load_safe_response_checkpoint,
    _native_checkpoint_initialization,
    _prediction_difference_summary,
    _r0_gradient_loss_terms,
    _restore_review_arm_state,
    _sampling_summary_mapping,
    _select_geometry_x_probe,
    _validate_rehydrated_raw_baseline,
    _validate_review_gate,
    build_parser,
    derive_training_scales,
)
from channelthermal.response_control.sampling import (
    ReceiverSamplingConfig,
    SamplingSummary,
    sample_training_stencil,
)
from channelthermal.response_control.training import (
    TrainingStep,
    _family_visit_counts,
    historical_replay_coverage,
    historical_replay_sequence,
)
from torch import nn


def _record(
    label: str,
    *,
    position_x: float = 5.0,
    heating: float = 1.0,
    fluid_x: np.ndarray | None = None,
    split: EvidenceSplit = EvidenceSplit.TRAIN,
    case_dir: str | None = None,
) -> SolveRecord:
    design = DesignState(
        anchor_id="anchor-1",
        physical_family_id="family-1",
        split=split,
        modules=(ModuleState("m0", (position_x, 5.0), heating),),
    )
    x = np.asarray(
        [0.2, 0.9, 3.0, 9.0, 11.1, 11.8] if fluid_x is None else fluid_x,
        dtype=np.float64,
    )
    fluid_query = np.stack((x, np.full_like(x, 5.0)), axis=-1)
    signal = position_x * heating
    fluid_values = (10.0 - 2.0 * x / 12.0 + signal)[:, None]
    interface_query = np.asarray([[0.0, 1.0, 0.0], [np.pi, 1.0, 0.0]])
    interface_values = np.full((2, 2), signal, dtype=np.float64)
    solid_query = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    solid_values = (300.0 + signal + solid_query[:, :1])
    roles = {
        "fluid_fields": RoleOutput(
            role="fluid_fields",
            query_features=fluid_query,
            values=fluid_values,
            channel_names=("p",),
            channel_units=("Pa",),
            valid_mask=np.ones_like(fluid_values, dtype=bool),
            quadrature_weights=np.ones(len(x)),
            query_ids=tuple(f"grid:{idx}" for idx in range(len(x))),
            coordinate_kind="eulerian",
        ),
        "interface": RoleOutput(
            role="interface",
            query_features=interface_query,
            values=interface_values,
            channel_names=("normal_flux", "temperature"),
            channel_units=("W/m2", "K"),
            valid_mask=np.ones_like(interface_values, dtype=bool),
            quadrature_weights=np.ones(2),
            query_ids=("m0:port:0", "m0:port:1"),
            receiver_module_ids=("m0", "m0"),
            coordinate_kind="interface_material_angle",
        ),
        "solid_temperature": RoleOutput(
            role="solid_temperature",
            query_features=solid_query,
            values=solid_values,
            channel_names=("temperature",),
            channel_units=("K",),
            valid_mask=np.ones_like(solid_values, dtype=bool),
            quadrature_weights=np.ones(2),
            query_ids=("m0:solid:0", "m0:solid:1"),
            receiver_module_ids=("m0", "m0"),
            coordinate_kind="solid_material_normalized_xy",
        ),
    }
    output = PhysicalSolveOutput(
        roles=roles,
        quantities={"pressure_drop": MeasuredQuantity(2.0, "Pa")},
        active_module_ids=("m0",),
        module_peak_temperature={"m0": float(301.0 + signal)},
        units_metadata={"temperature": "K"},
        case_dir=case_dir,
    )
    return SolveRecord(
        record_id=label,
        design=design,
        context=OperatingContext(
            {"domain_length_x": 12.0, "module_radius": 0.2, "re": 50.0}
        ),
        source=EvidenceSource.REFERENCE_SOLVER,
        status=SolveStatus.CONVERGED,
        elapsed_seconds=0.1,
        provenance={
            "solver_invoked": True,
            "raw_solver_completed": True,
            "physical_wall_time_available": True,
            "raw_case_dir": case_dir,
        },
        output=output,
    )


def _stencil(split: EvidenceSplit = EvidenceSplit.TRAIN) -> ResponseStencil:
    base = _record("base", split=split)
    variants = {
        "heat": _record("heat", heating=1.2, split=split),
        "shift": _record("shift", position_x=5.5, split=split),
        "both": _record("both", position_x=5.5, heating=1.2, split=split),
    }
    return ResponseStencil(base, variants)


def _fixed_heat_record(label: str, heating: tuple[float, float]) -> SolveRecord:
    module_ids = ("m0", "m1")
    design = DesignState(
        anchor_id="fixed-heat-anchor",
        physical_family_id="fixed-heat-family",
        split=EvidenceSplit.TRAIN,
        modules=(
            ModuleState("m0", (4.0, 5.0), heating[0]),
            ModuleState("m1", (8.0, 5.0), heating[1]),
        ),
    )
    x = np.asarray([0.2, 0.8, 3.0, 6.0, 9.0, 11.2, 11.8], dtype=np.float64)
    fluid_values = np.stack(
        (
            np.zeros_like(x),
            np.zeros_like(x),
            20.0 - 2.0 * x,
            np.zeros_like(x),
            np.full_like(x, 300.0 + 0.5 * sum(heating)),
        ),
        axis=1,
    )
    fluid_role = RoleOutput(
        role="fluid_fields",
        query_features=np.stack((x, np.full_like(x, 5.0)), axis=1),
        values=fluid_values,
        channel_names=("u", "v", "p", "omega", "temperature"),
        channel_units=("m/s", "m/s", "Pa", "1/s", "dataset temperature units"),
        valid_mask=np.ones_like(fluid_values, dtype=bool),
        quadrature_weights=np.ones_like(x),
        query_ids=tuple(f"fluid:{index}" for index in range(len(x))),
        coordinate_kind="eulerian",
    )
    interface_features = np.asarray(
        [[0.0, 1.0, 0.0], [np.pi, -1.0, 0.0], [0.0, 1.0, 0.0], [np.pi, -1.0, 0.0]],
        dtype=np.float64,
    )
    interface_slots = np.asarray([0, 0, 1, 1], dtype=np.int64)
    interface_factor = 1.0 + interface_features[:, 0] / np.pi
    interface_heat = np.asarray(heating, dtype=np.float64)[interface_slots]
    interface_values = np.stack(
        (300.0 + interface_heat * interface_factor, 2.0 * interface_heat * interface_factor),
        axis=1,
    )
    interface_role = RoleOutput(
        role="interface",
        query_features=interface_features,
        values=interface_values,
        channel_names=("T_surface", "q_normal"),
        channel_units=("dataset temperature units", "dataset heat-flux units"),
        valid_mask=np.ones_like(interface_values, dtype=bool),
        quadrature_weights=np.ones(4),
        query_ids=("m0:port:0", "m0:port:1", "m1:port:0", "m1:port:1"),
        receiver_module_ids=("m0", "m0", "m1", "m1"),
        coordinate_kind="interface_material_angle",
    )
    local_xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [9.0, 0.0], [0.5, 0.0]], dtype=np.float64)
    solid_features = np.tile(local_xy, (2, 1))
    solid_receiver_ids = tuple(module_id for module_id in module_ids for _ in range(4))
    solid_values = np.concatenate(
        [
            (300.0 + value * local_xy[:, :1])
            for value in heating
        ],
        axis=0,
    )
    solid_values[[2, 6], 0] = 900.0
    solid_valid = np.tile(np.asarray([[True], [True], [False], [True]]), (2, 1))
    solid_role = RoleOutput(
        role="solid_temperature",
        query_features=solid_features,
        values=solid_values,
        channel_names=("temperature",),
        channel_units=("dataset temperature units",),
        valid_mask=solid_valid,
        # Each module's valid material peak receiver has zero integral weight.
        quadrature_weights=np.tile(np.asarray([1.0, 0.0, 1.0, 1.0]), 2),
        query_ids=tuple(f"{module_id}:solid:{index}" for module_id in module_ids for index in range(4)),
        receiver_module_ids=solid_receiver_ids,
        coordinate_kind="solid_material_normalized_xy",
    )
    solid_values_by_id = {
        module_id: float(np.max(solid_role.values[
            (np.asarray(solid_receiver_ids) == module_id) & solid_role.valid_mask[:, 0], 0
        ]))
        for module_id in module_ids
    }
    output = PhysicalSolveOutput(
        roles={
            "fluid_fields": fluid_role,
            "interface": interface_role,
            "solid_temperature": solid_role,
        },
        quantities={"pressure_drop": MeasuredQuantity(5.0, "Pa")},
        active_module_ids=module_ids,
        module_peak_temperature=solid_values_by_id,
        units_metadata={"temperature": "dataset temperature units; no SI metadata identified"},
    )
    return SolveRecord(
        record_id=label,
        design=design,
        context=OperatingContext(
            {"domain_length_x": 12.0, "module_radius": 0.2, "re": 50.0}
        ),
        source=EvidenceSource.REFERENCE_SOLVER,
        status=SolveStatus.CONVERGED,
        elapsed_seconds=0.1,
        provenance={"solver_invoked": True, "raw_solver_completed": True},
        output=output,
    )


class _FixedHeatOperator(nn.Module):
    def __init__(self, *, flow_leak: float, thermal_gain: float) -> None:
        super().__init__()
        self.flow_leak = nn.Parameter(torch.tensor(flow_leak, dtype=torch.float32))
        self.thermal_gain = nn.Parameter(torch.tensor(thermal_gain, dtype=torch.float32))

    def forward(self, design, context, role_queries):
        allocation = design.module_heating[0] - design.module_heating[1]
        result = {}
        for name, query in role_queries.items():
            if name == "fluid_fields":
                x = query.query_features[:, 0]
                y = query.query_features[:, 1]
                zero = torch.zeros_like(x)
                fluid_temperature = 300.0 + 0.5 * design.module_heating.sum() + zero
                result[name] = torch.stack(
                    (
                        self.flow_leak * allocation + zero,
                        self.flow_leak * allocation * y,
                        20.0 - 2.0 * x + self.flow_leak * allocation * x,
                        zero,
                        fluid_temperature,
                    ),
                    dim=1,
                )
            elif name == "interface":
                slots = torch.as_tensor(query.receiver_slots, dtype=torch.long)
                heater = design.module_heating.index_select(0, slots)
                factor = 1.0 + query.query_features[:, 0] / torch.pi
                result[name] = torch.stack(
                    (
                        300.0 + self.thermal_gain * heater * factor,
                        2.0 * self.thermal_gain * heater * factor,
                    ),
                    dim=1,
                )
            else:
                slots = torch.as_tensor(query.receiver_slots, dtype=torch.long)
                heater = design.module_heating.index_select(0, slots)
                local_x = query.query_features[:, 0]
                result[name] = (300.0 + self.thermal_gain * heater * local_x)[:, None]
        return AbsolutePrediction(role_values=result)


class _AbsoluteField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.5))
        self.received_contexts: list[dict[str, Any]] = []

    def forward(self, design, context, role_queries):
        self.received_contexts.append(dict(context))
        signal = design.module_positions[0, 0] * design.module_heating[0]
        values = {}
        for name, query in role_queries.items():
            features = query.query_features
            if name == "fluid_fields":
                predicted = 10.0 - 2.0 * features[:, :1] / 12.0 + self.gain * signal
            elif name == "solid_temperature":
                predicted = 300.0 + self.gain * signal + features[:, :1]
            else:
                predicted = self.gain * signal + features[:, :1] * 0.0
            values[name] = predicted.expand(features.shape[0], len(query.channel_names))
        return AbsolutePrediction(role_values=values)


def test_fixed_heat_control_preserves_null_flow_and_full_valid_material_peaks() -> None:
    baseline = _fixed_heat_record("fixed-base", (2.0, 1.0))
    trial = _fixed_heat_record("fixed-trial", (1.0, 2.0))
    control = FixedHeatNullControl("fixed-test", baseline, trial).sampled_for_panel(baseline)
    coverage = fixed_heat_material_peak_coverage(trial)
    assert set(coverage) == {"m0", "m1"}
    assert all(row["receiver_count"] == 4 for row in coverage.values())
    assert all(row["valid_receiver_count"] == 3 for row in coverage.values())
    assert all(row["exact_target_match"] is True for row in coverage.values())
    assert coverage["m0"]["peak_receiver_ids"] == ("m0:solid:1",)
    assert coverage["m1"]["peak_receiver_ids"] == ("m1:solid:1",)
    assert control.sampled_role_indices["solid_temperature"] == tuple(range(8))

    role_scales = {
        "fluid_fields": (1.0, 1.0, 1.0, 1.0, 1.0),
        "interface": (1.0, 1.0),
        "solid_temperature": (1.0,),
    }
    scales = ThermalLossScales(
        value=role_scales,
        finite=role_scales,
        mixed=role_scales,
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=100.0,
        pressure_boundary=1.0,
        solid_temperature=1.0,
        smooth_peak_beta=2.0,
        near_limit_band=1.0,
    )

    suppressed_thermal = _FixedHeatOperator(flow_leak=0.0, thermal_gain=0.0)
    suppressed_terms, suppressed_diagnostics = fixed_heat_control_loss_terms(
        suppressed_thermal, control, scales=scales
    )
    assert float(suppressed_terms["fixed_heat_null"].detach()) == pytest.approx(0.0, abs=1.0e-10)
    assert float(suppressed_terms["fixed_heat_thermal"].detach()) > 0.0
    assert float(suppressed_diagnostics["fixed_heat_thermal/peak_per_id"]) > 0.0
    suppressed_terms["fixed_heat_thermal"].backward()
    assert suppressed_thermal.thermal_gain.grad is not None
    assert float(suppressed_thermal.thermal_gain.grad.abs()) > 0.0

    matching = _FixedHeatOperator(flow_leak=0.0, thermal_gain=1.0)
    matching_terms, matching_diagnostics = fixed_heat_control_loss_terms(
        matching, control, scales=scales
    )
    assert float(matching_terms["fixed_heat_null"].detach()) == pytest.approx(0.0, abs=1.0e-10)
    assert float(matching_terms["fixed_heat_thermal"].detach()) == pytest.approx(0.0, abs=1.0e-6)
    assert float(matching_diagnostics["fixed_heat_thermal/peak_per_id"]) == pytest.approx(0.0, abs=1.0e-6)

    spurious_flow = _FixedHeatOperator(flow_leak=0.1, thermal_gain=1.0)
    flow_terms, _ = fixed_heat_control_loss_terms(spurious_flow, control, scales=scales)
    assert float(flow_terms["fixed_heat_null"].detach()) > 0.0
    flow_terms["fixed_heat_null"].backward()
    assert spurious_flow.flow_leak.grad is not None
    assert float(spurious_flow.flow_leak.grad.abs()) > 0.0

    changed_pressure_output = replace(
        trial.output,
        quantities={"pressure_drop": MeasuredQuantity(5.000001, "Pa")},
    )
    changed_pressure = replace(trial, output=changed_pressure_output)
    with pytest.raises(ValueError, match="exact resolved pressure-drop null target"):
        FixedHeatNullControl("bad-pressure", baseline, changed_pressure)


def test_fixed_heat_review_rehydrates_raw_baseline_and_keeps_strict_physical_quadrature() -> None:
    atlas_baseline = _fixed_heat_record("atlas-baseline", (2.0, 1.0))
    raw_control = _fixed_heat_record("raw-control", (1.0, 2.0))
    atlas_roles = dict(atlas_baseline.output.roles)  # type: ignore[union-attr]
    raw_baseline_roles: dict[str, RoleOutput] = {}
    raw_control_roles = dict(raw_control.output.roles)  # type: ignore[union-attr]
    physical_weights = {
        "fluid_fields": 0.0087890625,
        "interface": 0.04417864669,
        "solid_temperature": 0.00020408163,
    }
    for role_name, atlas_role in atlas_roles.items():
        raw_features = np.array(atlas_role.query_features, copy=True)
        if role_name == "solid_temperature":
            raw_features[0, 0] += 2.8383e-8
        weights = np.full(atlas_role.values.shape[0], physical_weights[role_name])
        raw_baseline_roles[role_name] = replace(
            atlas_role,
            query_features=raw_features,
            quadrature_weights=weights,
        )
        raw_control_roles[role_name] = replace(
            raw_control_roles[role_name],
            query_features=raw_features,
            quadrature_weights=weights,
        )
    raw_baseline = replace(
        atlas_baseline,
        output=replace(atlas_baseline.output, roles=raw_baseline_roles),  # type: ignore[arg-type,union-attr]
    )
    raw_control = replace(
        raw_control,
        output=replace(raw_control.output, roles=raw_control_roles),  # type: ignore[arg-type,union-attr]
    )

    binding = _validate_rehydrated_raw_baseline(atlas_baseline, raw_baseline)
    assert binding["status"] == "passed"
    assert binding["role_checks"]["solid_temperature"]["coordinates_max_abs_difference"] == pytest.approx(
        2.8383e-8, abs=1e-14
    )
    atlas_stencil = ResponseStencil(atlas_baseline, {"heat": raw_control})
    with pytest.raises(ValueError, match="quadrature weights differ"):
        atlas_stencil.finite_change("heat", "fluid_fields")

    raw_stencil = ResponseStencil(raw_baseline, {"heat": raw_control})
    for role_name in raw_baseline.output.roles:  # type: ignore[union-attr]
        assert raw_stencil.finite_change("heat", role_name).valid_mask.any()

    invalid_roles = dict(raw_baseline.output.roles)  # type: ignore[union-attr]
    invalid_features = np.array(invalid_roles["fluid_fields"].query_features, copy=True)
    invalid_features[0, 0] += 2.0e-7
    invalid_roles["fluid_fields"] = replace(
        invalid_roles["fluid_fields"],
        query_features=invalid_features,
    )
    unbound_baseline = replace(
        raw_baseline,
        output=replace(raw_baseline.output, roles=invalid_roles),  # type: ignore[arg-type,union-attr]
    )
    with pytest.raises(ValueError, match="coordinates exceed the declared atlas-rounding tolerance"):
        _validate_rehydrated_raw_baseline(atlas_baseline, unbound_baseline)


def test_fixed_heat_control_cursor_round_robins_with_resume_sampler_tail() -> None:
    panel = tuple(
        SimpleNamespace(physical_family_id=f"family-{index}")
        for index in range(8)
    )
    # At update 10, one complete epoch and the last two indices of the next
    # shuffled epoch have been consumed; the remaining tail identifies the
    # exact families whose control cursor advanced.
    visits = _family_visit_counts(panel, completed_updates=10, remaining_order=(0, 1, 2, 3, 4, 5))
    assert visits["family-0"] == 1
    assert visits["family-6"] == 2
    assert visits["family-7"] == 2
    next_four_controls = [visits["family-6"] + step for step in range(4)]
    assert [index % 4 for index in next_four_controls] == [2, 3, 0, 1]


def test_callback_is_target_free_and_response_algebra_uses_absolute_states() -> None:
    stencil = _stencil()
    model = _AbsoluteField()
    predicted = predict_stencil(model, stencil)
    assert len(model.received_contexts) == 4
    assert all(set(context) == {"domain_length_x", "module_radius", "re"} for context in model.received_contexts)
    finite = predicted.finite("heat", "interface")
    torch.testing.assert_close(finite, torch.ones_like(finite) * (5.0 * 0.2 * model.gain))
    spec = MixedResponseSpec("interface", "both", "heat", "shift")
    mixed = predicted.mixed(spec)
    torch.testing.assert_close(mixed, torch.ones_like(mixed) * (0.5 * 0.2 * model.gain))


def test_native_quantity_reduction_preserves_pressure_and_material_peak_semantics() -> None:
    x = torch.arange(0.0, 12.0001, 0.25)
    fluid_xy = torch.stack((x, torch.full_like(x, 3.0)), dim=1)
    fluid_query = RoleQuery(
        role="fluid_fields",
        query_features=fluid_xy,
        channel_names=("u", "v", "p", "omega", "temperature"),
        channel_units=("m/s", "m/s", "Pa", "1/s", "K"),
        receiver_slots=None,
        coordinate_kind="eulerian",
    )
    solid_query = RoleQuery(
        role="solid_temperature",
        query_features=torch.tensor(
            [[-0.5, 0.0], [0.0, 0.0], [-0.5, 0.0], [0.0, 0.0]], dtype=torch.float32
        ),
        channel_names=("temperature",),
        channel_units=("K",),
        receiver_slots=(0, 0, 1, 1),
        coordinate_kind="solid_material_normalized_xy",
    )
    design = DesignInput(
        module_positions=torch.tensor([[6.0, 3.0], [9.5, 3.0]], dtype=torch.float32),
        module_heating=torch.tensor([1.0, 2.0], dtype=torch.float32),
        module_present=torch.tensor([True, True]),
    )
    pressure_slope = torch.tensor(2.0, requires_grad=True)
    pressure = pressure_slope * x
    fluid_values = torch.stack(
        (torch.zeros_like(x), torch.zeros_like(x), pressure, torch.zeros_like(x), torch.zeros_like(x)),
        dim=1,
    )
    solid_values = torch.tensor([[301.0], [999.0], [305.0], [304.0]], requires_grad=True)
    prediction = AbsolutePrediction(
        role_values={"fluid_fields": fluid_values, "solid_temperature": solid_values}
    )

    quantities = reduce_native_thermal_quantities(
        prediction,
        design,
        {"fluid_fields": fluid_query, "solid_temperature": solid_query},
        {"domain_length_x": 12.0, "module_radius": 0.45},
        module_ids=("physical-A", "physical-B"),
        solid_valid_mask=torch.tensor([[True], [False], [True], [True]]),
    )

    # The pressure scalar uses the existing geometric fluid exclusion and
    # maintained 8% inlet/outlet bands; its gradient remains connected.
    expected = pressure_drop_from_field(
        fluid_values,
        fluid_query,
        design,
        {"domain_length_x": 12.0, "module_radius": 0.45},
    )
    torch.testing.assert_close(quantities.pressure_drop, expected)
    assert quantities.pressure_drop_units == "Pa"
    assert quantities.temperature_units == "K"
    assert set(quantities.module_peak_temperature) == {"physical-A", "physical-B"}
    torch.testing.assert_close(quantities.module_peak_temperature["physical-A"], torch.tensor(301.0))
    torch.testing.assert_close(quantities.module_peak_temperature["physical-B"], torch.tensor(305.0))

    (quantities.pressure_drop + sum(quantities.module_peak_temperature.values())).backward()
    assert pressure_slope.grad is not None and pressure_slope.grad < 0.0
    torch.testing.assert_close(
        solid_values.grad,
        torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    )


def test_historical_absolute_replay_is_target_free_and_uses_reference_peak_mask() -> None:
    original = _record("historical")
    solid = original.output.roles["solid_temperature"]
    masked_solid = replace(
        solid,
        values=np.asarray([[305.0], [999.0]], dtype=np.float64),
        valid_mask=np.asarray([[True], [False]]),
    )
    roles = dict(original.output.roles)
    roles["solid_temperature"] = masked_solid
    output = replace(
        original.output,
        roles=roles,
        module_peak_temperature={"m0": 305.0},
    )
    record = replace(original, output=output)

    queries = role_queries_from_record(record)
    assert set(queries) == set(record.output.roles)
    assert not hasattr(queries["solid_temperature"], "valid_mask")
    assert not hasattr(queries["fluid_fields"], "values")
    np.testing.assert_array_equal(
        queries["solid_temperature"].query_features.cpu().numpy(),
        masked_solid.query_features,
    )

    metrics = evaluate_absolute_record(_AbsoluteField(), record, pressure_limit=2.1)
    peak = metrics["solid_peaks"][0]
    assert peak["reference_peak"] == pytest.approx(305.0)
    # The invalid 999 K receiver does not enter either reference or predicted peaks.
    assert peak["predicted_peak"] == pytest.approx(302.5)
    assert metrics["solid_peak_summary"][0]["reference_value"] == pytest.approx(305.0)
    assert len(metrics["pressure"]) == 1
    assert metrics["pressure"][0]["units"] == "Pa"


def test_native_adapter_maps_material_rows_around_inactive_padded_slots() -> None:
    active_slots = (True, False, True, False)
    interface = RoleQuery(
        role="interface",
        query_features=torch.tensor([[0.0, 1.0, 0.0], [torch.pi, -1.0, 0.0]]),
        channel_names=("temperature", "normal_flux"),
        channel_units=("K", "W/m2"),
        receiver_slots=(0, 2),
        coordinate_kind="interface_material_angle",
    )
    solid = RoleQuery(
        role="solid_temperature",
        query_features=torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]),
        channel_names=("temperature",),
        channel_units=("K",),
        receiver_slots=(0, 0, 2, 2),
        coordinate_kind="solid_material_normalized_xy",
    )
    interface_rows = DifferentiableThermalOperator._rows_by_slot(interface, 4, active_slots)
    assert [row.tolist() for row in interface_rows] == [[0], [], [1], []]

    adapter = object.__new__(DifferentiableThermalOperator)
    adapter.max_modules = 4
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter.normalize_inputs = False
    condition, teacher_ports, rows = adapter._interface_inputs(interface, 4, active_slots)
    assert condition.shape == (1, 4, 1, 8)
    assert torch.count_nonzero(condition[:, 1]) == 0
    assert torch.count_nonzero(condition[:, 3]) == 0
    assert teacher_ports.shape == (1, 4, 1, 5)
    assert [row.tolist() for row in rows] == [[0], [], [1], []]

    local_query, solid_rows = adapter._solid_inputs(solid, 4, active_slots)
    assert local_query.shape == (1, 2, 2)
    assert [row.tolist() for row in solid_rows] == [[0, 1], [], [2, 3], []]

    misplaced = replace(
        interface,
        query_features=torch.cat((interface.query_features, interface.query_features[:1]), dim=0),
        receiver_slots=(0, 1, 2),
    )
    with pytest.raises(ValueError, match="Inactive module slots"):
        DifferentiableThermalOperator._rows_by_slot(misplaced, 4, active_slots)


def test_native_output_scope_selects_only_existing_final_layers() -> None:
    class _Head(nn.Module):
        def __init__(self, widths: tuple[int, ...]) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for input_width, output_width in pairwise(widths):
                layers.extend((nn.Linear(input_width, output_width), nn.Tanh()))
            self.net = nn.Sequential(*layers[:-1])

    class _ScopeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Module()
            self.core.common = nn.Module()
            self.core.common.field_head = _Head((3, 5, 2))
            self.local_coupling = nn.Module()
            self.local_coupling.has_local_surrogate = True
            self.local_coupling.port_head = _Head((4, 6, 2))
            self.local_coupling.port_refinement_head = _Head((5, 7, 2))
            self.backbone = nn.Linear(2, 2)

    model = _ScopeModel()
    scope = _configure_native_output_head_scope(model)  # type: ignore[arg-type]
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert scope["name"] == "native_output_heads"
    assert trainable == {
        "core.common.field_head.net.2.weight",
        "core.common.field_head.net.2.bias",
        "local_coupling.port_head.net.2.weight",
        "local_coupling.port_head.net.2.bias",
        "local_coupling.port_refinement_head.net.2.weight",
        "local_coupling.port_refinement_head.net.2.bias",
    }
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_native_nonlinear_scope_trains_all_existing_head_layers_in_eval_mode() -> None:
    class _ScopeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Module()
            self.core.common = nn.Module()
            self.core.common.field_head = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Dropout(0.25), nn.Linear(5, 2))
            self.local_coupling = nn.Module()
            self.local_coupling.has_local_surrogate = True
            self.local_coupling.port_head = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Dropout(0.25), nn.Linear(6, 2))
            self.local_coupling.port_refinement_head = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Dropout(0.25), nn.Linear(7, 2))
            self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(0.5))

    model = _ScopeModel()
    scope = _configure_native_nonlinear_interface_scope(model)  # type: ignore[arg-type]
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {
        name for name, _parameter in model.named_parameters()
        if name.startswith((
            "core.common.field_head.",
            "local_coupling.port_head.",
            "local_coupling.port_refinement_head.",
        ))
    }
    assert scope["name"] == "native_nonlinear_interface"
    assert trainable == expected
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(not module.training for module in model.modules())
    assert scope["dropout_modules_disabled_by_eval"]
    assert scope["frozen_buffer_names"] == ["backbone.1.running_mean", "backbone.1.running_var", "backbone.1.num_batches_tracked"]


def test_native_expanded_response_scope_selects_exact_heads_and_backend_blocks() -> None:
    class _ScopeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Module()
            self.core.common = nn.Module()
            self.core.common.field_head = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
            self.core.common.context_builder = nn.Linear(2, 2)
            self.core.backend = nn.Module()
            for name in (
                "mm_message", "me_message", "em_message", "module_update", "env_update",
                "query_module_message", "query_module_output", "env_query",
                "env_attention", "env_geometry_bias",
            ):
                setattr(self.core.backend, name, nn.Sequential(nn.Linear(4, 4), nn.Tanh()))
            self.core.backend.unselected_policy = nn.Linear(2, 2)
            self.local_coupling = nn.Module()
            self.local_coupling.has_local_surrogate = True
            self.local_coupling.port_head = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
            self.local_coupling.port_refinement_head = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
            self.local_coupling.local_latent_fusion = nn.Linear(2, 2)
            self.local_coupling.local_response_summary_proj = nn.Linear(2, 2)
            self.local_coupling.flux_correction_head = nn.Linear(2, 2)
            self.local_coupling.local_surrogate = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
            self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))

    model = _ScopeModel()
    model.train()
    scope = _configure_native_expanded_response_interface_scope(model)  # type: ignore[arg-type]
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    selected_modules = {
        "core.common.field_head",
        "local_coupling.port_head",
        "local_coupling.port_refinement_head",
        *(f"core.backend.{name}" for name in (
            "mm_message", "me_message", "em_message", "module_update", "env_update",
            "query_module_message", "query_module_output", "env_query", "env_attention",
            "env_geometry_bias",
        )),
    }
    expected = {
        name for name, _parameter in model.named_parameters()
        if any(name.startswith(f"{module_name}.") for module_name in selected_modules)
    }
    assert scope["name"] == "native_expanded_response_interface"
    assert trainable == expected
    assert set(scope["trainable_module_names"]) == selected_modules
    assert set(scope["trainable_parameter_count_by_block"]) == {
        "field_head", "port_head", "port_refinement_head",
        "native_module_environment_updates", "native_fine_receiver_reads",
    }
    assert all(not parameter.requires_grad for parameter in model.core.common.context_builder.parameters())
    assert all(not parameter.requires_grad for parameter in model.core.backend.unselected_policy.parameters())
    assert all(not parameter.requires_grad for parameter in model.local_coupling.local_surrogate.parameters())
    assert all(not module.training for module in model.modules())
    assert scope["dropout_modules_disabled_by_eval"]
    assert scope["frozen_buffer_names"] == [
        "backbone.1.running_mean", "backbone.1.running_var", "backbone.1.num_batches_tracked"
    ]


def test_native_checkpoint_initialization_keeps_incumbent_as_a_separate_model() -> None:
    class _NativeSource(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.config = SimpleNamespace(
                core_honf=SimpleNamespace(forward_architecture="dense_pairwise_field")
            )

    source = _NativeSource()
    target, provenance = _native_checkpoint_initialization(source)  # type: ignore[arg-type]

    assert target is not source
    assert provenance["prediction_identity_claim"] is True
    assert torch.equal(target.weight, source.weight)
    assert target.weight.data_ptr() != source.weight.data_ptr()
    with torch.no_grad():
        target.weight.add_(1.0)
    assert torch.equal(source.weight, torch.tensor([1.0, 2.0]))


def test_r0_projection_gate_uses_calibrated_combined_gradient_not_any_term() -> None:
    value = np.asarray([1.0, 0.0])
    combined, gate = _combine_r0_response_gradient(
        value,
        {
            "finite": np.asarray([-1.0, 0.0]),
            "finite_peak": np.asarray([1.0, 0.0]),
        },
        {"finite": 1.0, "finite_peak": 1.0},
    )
    assert np.array_equal(combined, np.zeros(2))
    assert gate["dot_product"] == 0.0
    assert gate["adverse"] is False

    _combined, adverse_gate = _combine_r0_response_gradient(
        value,
        {
            "finite": np.asarray([-1.0, 0.0]),
            "finite_peak": np.asarray([1.0, 0.0]),
        },
        {"finite": 1.0, "finite_peak": 0.25},
    )
    assert adverse_gate["dot_product"] == -0.75
    assert adverse_gate["adverse"] is True


def test_r0_gradient_objectives_use_all_calibrated_response_term_keys() -> None:
    calibrated_weights = {term: 1.0 for term in _R0_RESPONSE_TERMS}
    stencil_losses = {
        "value": torch.tensor(2.0, requires_grad=True),
        "finite": torch.tensor(0.5, requires_grad=True),
        "finite_peak": torch.tensor(0.4, requires_grad=True),
        "pressure_value": torch.tensor(0.3, requires_grad=True),
        "pressure_response": torch.tensor(0.2, requires_grad=True),
    }
    objectives = _r0_gradient_loss_terms(
        stencil_losses,
        torch.tensor(1.0, requires_grad=True),
        {"value": 1.0, **calibrated_weights},
    )
    assert set(_R0_RESPONSE_TERMS).issubset(objectives)

    value_gradient = np.asarray([1.0, 0.0])
    response_gradients = {
        "finite": np.asarray([-1.0, 0.0]),
        "finite_peak": np.asarray([0.4, 0.0]),
        "pressure_value": np.asarray([0.3, 0.0]),
        "pressure_response": np.asarray([0.5, 0.0]),
    }
    combined, gate = _combine_r0_response_gradient(
        value_gradient,
        response_gradients,
        calibrated_weights,
    )
    assert np.array_equal(combined, np.asarray([0.2, 0.0]))
    assert gate["adverse"] is False

    adverse_weights = {**calibrated_weights, "finite": 2.0}
    _combined, adverse_gate = _combine_r0_response_gradient(
        value_gradient,
        response_gradients,
        adverse_weights,
    )
    assert adverse_gate["adverse"] is True


def test_training_step_result_serialization_preserves_mappingproxy_fields() -> None:
    step = TrainingStep(
        completed_update=1,
        attempted_optimizer_step=1,
        training_stencil_index=0,
        stage="response",
        active_terms=("finite",),
        total_loss=0.5,
        term_losses=MappingProxyType({"finite": 0.5}),
        active_term_weights=MappingProxyType({"finite": 1.0}),
        response_gradient_dot_before=MappingProxyType({"field_head": 0.0}),
    )
    result = _dataclass_record(step)
    encoded = json.dumps(result, default=_json_default)
    assert json.loads(encoded)["term_losses"] == {"finite": 0.5}
    assert json.loads(encoded)["active_term_weights"] == {"finite": 1.0}


def test_historical_replay_coverage_reports_partial_and_complete_first_passes() -> None:
    case_order = tuple(f"case-{index:03d}" for index in range(600))
    probe_ids = historical_replay_sequence(case_order, start_update=0, update_count=80)
    probe_coverage = historical_replay_coverage(
        case_order, initial_update=0, segment_case_ids=probe_ids
    )
    assert probe_coverage["unique_train_cases_visited"] == 80
    assert probe_coverage["first_pass_train_cases_visited"] == 80
    assert probe_coverage["first_pass_complete"] is False
    continuation_ids = historical_replay_sequence(case_order, start_update=200, update_count=400)
    formal_coverage = historical_replay_coverage(
        case_order, initial_update=200, segment_case_ids=continuation_ids
    )
    assert formal_coverage["updates_through_gate"] == 600
    assert formal_coverage["unique_train_cases_visited"] == 600
    assert formal_coverage["first_pass_train_cases_visited"] == 600
    assert formal_coverage["first_pass_complete"] is True
    assert formal_coverage["first_pass_case_ids"] == list(case_order)
    with pytest.raises(ValueError, match="does not match the fixed train-case sequence"):
        historical_replay_coverage(
            case_order, initial_update=200, segment_case_ids=tuple(reversed(continuation_ids))
        )


def test_staged_config_exposes_bounded_review_gates_and_runs_actual_updates() -> None:
    configured = load_staged_training_config()
    assert configured.max_optimizer_updates == 2000
    assert configured.max_epochs == 500
    assert configured.review_updates == (100, 300, 1000, 2000)
    assert tuple(stage.stop_update for stage in configured.stages) == (20, 100, 300, 2000)
    assert configured.active_terms(0) == ("value",)
    assert configured.active_terms(19) == ("value",)
    assert configured.active_terms(20) == ("value", "finite")
    assert configured.active_terms(99) == ("value", "finite")
    assert configured.active_terms(100) == (
        "value", "finite", "decision", "constraint"
    )
    assert configured.active_terms(300) == (
        "value", "finite", "decision", "constraint"
    )
    assert configured.active_terms(2000) == (
        "value", "finite", "decision", "constraint"
    )
    assert configured.active_terms(2001) == (
        "value", "finite", "decision", "constraint"
    )
    value_config = load_staged_training_config(arm="B_value")
    assert all(value_config.active_terms(update) == ("value",) for update in (0, 19, 20, 99, 100, 300, 1999))
    nonlinear_recipe = Path(__file__).resolve().parents[1] / "configs" / "response_control_native_nonlinear_interface.json"
    response_config = load_staged_training_config(str(nonlinear_recipe), arm="R_response")
    value_recipe_config = load_staged_training_config(str(nonlinear_recipe), arm="R_value")
    assert response_config.review_updates == (200, 500, 1000)
    assert response_config.active_terms(9) == ("value",)
    assert set(response_config.active_terms(49)) == {
        "value", "finite", "finite_peak", "pressure_value", "pressure_response",
        "fixed_heat_null", "fixed_heat_thermal",
    }
    assert all(response_config.term_multiplier(term, 49) == 1.0 for term in response_config.required_response_terms)
    assert response_config.project_response_gradient_blockwise is True
    assert all(value_recipe_config.active_terms(update) == ("value",) for update in (0, 9, 10, 49, 999))
    assert _validate_review_gate(100, configured, 3) == 1500
    with pytest.raises(ValueError, match="review gate 2000 is unreachable"):
        _validate_review_gate(2000, configured, 3)

    expanded_recipe = Path(__file__).resolve().parents[1] / "configs" / "response_control_native_expanded_interface.json"
    expanded = load_staged_training_config(str(expanded_recipe), arm="R_response")
    expanded_value = load_staged_training_config(str(expanded_recipe), arm="R_value")
    assert expanded.max_optimizer_updates == 600
    assert expanded.max_epochs == 75
    assert expanded.total_optimizer_update_ceiling == 600
    assert expanded.review_updates == (200, 600)
    assert expanded.project_response_gradient_blockwise is False
    assert expanded.active_terms(9) == ("value",)
    assert set(expanded.active_terms(49)) == {
        "value", "finite", "finite_peak", "pressure_value", "pressure_response",
        "fixed_heat_null", "fixed_heat_thermal",
    }
    assert set(expanded.required_response_terms) == {
        "finite", "finite_peak", "pressure_value", "pressure_response",
    }
    assert set(expanded.required_control_terms) == {"fixed_heat_null", "fixed_heat_thermal"}
    assert all(expanded_value.active_terms(update) == ("value",) for update in (0, 9, 10, 49, 599))
    with expanded_recipe.open("r", encoding="utf-8") as stream:
        expanded_payload = json.load(stream)
    assert expanded_payload["probe_optimizer_updates"] == 80
    assert expanded_payload["shared_remedy_optimizer_call_ceiling"] == 600
    assert expanded_payload["historical_train_case_count"] == 600
    assert expanded_payload["expected_source_checkpoint_sha256"] == (
        "71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066"
    )

    config = StagedTrainingConfig(
        arm="B_value",
        max_optimizer_updates=2,
        max_epochs=2,
        total_optimizer_update_ceiling=2,
        checkpoint_every_updates=1,
        review_updates=(1, 2),
        stages=(TrainingStage("value", 0, 2, ("value",)),),
    )
    model = _AbsoluteField()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    checkpoint_labels: list[str] = []
    result = run_staged_fit(
        model,
        model,
        optimizer,
        [_stencil()],
        scales=scales,
        config=config,
        on_checkpoint=lambda _payload, label: checkpoint_labels.append(label),
    )
    assert result.actual_optimizer_updates == 1
    assert result.attempted_optimizer_steps == 1
    assert result.completed_epochs == 1
    assert [step.completed_update for step in result.history] == [1]
    assert checkpoint_labels == ["zero_update_preflight", "training_checkpoint"]
    assert result.stopped_at_review is True

    continued_model = _AbsoluteField()
    continued_optimizer = torch.optim.SGD(continued_model.parameters(), lr=1e-4)
    continued = run_staged_fit(
        continued_model,
        continued_model,
        continued_optimizer,
        [_stencil()],
        scales=scales,
        config=config,
        stop_at_update=2,
        on_review=lambda _step: "continue",
    )
    assert continued.actual_optimizer_updates == 2
    assert [step.completed_update for step in continued.history] == [1, 2]
    assert continued.stopped_at_review is True


def test_r_arm_writes_first_finite_update_checkpoint_without_changing_b_schedule() -> None:
    config = StagedTrainingConfig(
        arm="R_value",
        max_optimizer_updates=2,
        max_epochs=2,
        total_optimizer_update_ceiling=2,
        checkpoint_every_updates=2,
        review_updates=(2,),
        stages=(TrainingStage("value", 0, 2, ("value",)),),
    )
    model = _AbsoluteField()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    labels: list[str] = []
    result = run_staged_fit(
        model,
        model,
        optimizer,
        [_stencil()],
        scales=scales,
        config=config,
        on_checkpoint=lambda _payload, label: labels.append(label),
    )
    assert result.actual_optimizer_updates == 2
    assert labels == [
        "zero_update_preflight",
        "training_checkpoint",  # R_value u1 checkpoint even though interval is 2.
        "training_checkpoint",  # configured u2 review checkpoint.
    ]


def test_lazy_nonpersistent_buffer_audit_uses_registered_names_not_state_dict_keys() -> None:
    model = nn.Module()
    core = nn.Module()
    core.position_fourier = nn.Module()
    core.position_fourier.register_buffer("frequencies", torch.tensor([1.0]), persistent=False)
    core.receiver_fourier = nn.Module()
    model.core = core
    snapshot = {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
    }
    core.receiver_fourier.register_buffer("frequencies", torch.tensor([2.0]), persistent=False)

    audit = _audit_named_buffers(model, snapshot)

    assert audit["passed"] is True
    assert audit["missing_names"] == []
    assert audit["changed_names"] == []
    assert audit["added_names"] == ["core.receiver_fourier.frequencies"]
    assert audit["added_nonpersistent_names"] == ["core.receiver_fourier.frequencies"]
    assert audit["nonpersistent_names"] == [
        "core.position_fourier.frequencies",
        "core.receiver_fourier.frequencies",
    ]


def test_r0_projection_gate_accepts_only_the_verified_lazy_buffer_recovery(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source checkpoint")
    gradient_path = tmp_path / "m10_gradient.json"
    gradient_path.write_text(json.dumps({
        "family_id": "stored_family:0350",
        "combined_response_gradient_vs_combined_value_gradient": {
            "field_head": {
                "adverse": True,
                "dot_product": -0.1,
                "combined_value_gradient_norm": 0.2,
                "cosine": -0.25,
            },
            "port_head": {
                "adverse": False,
                "dot_product": 0.1,
                "combined_value_gradient_norm": 0.2,
                "cosine": 0.25,
            },
            "port_refinement_head": {
                "adverse": False,
                "dot_product": 0.0,
                "combined_value_gradient_norm": 0.0,
                "cosine": None,
            },
        },
        "individual_response_term_diagnostics": {"finite": {"field_head": -0.1}},
        "calibrated_weights": {"finite": 1.0, "finite_peak": 1.0, "pressure_value": 1.0, "pressure_response": 1.0},
    }), encoding="utf-8")
    report_path = tmp_path / "r0_manifest.json"
    report_path.write_text(json.dumps({
        "mode": "r0_missing_native_nonlinear_interface_scope",
        "status": "failed",
        "error_type": "KeyError",
        "error": "'core.position_fourier.frequencies'",
        "checkpoint_sha256": file_sha256(checkpoint),
        "optimizer_updates_completed": 80,
        "optimizer_updates_attempted": 80,
        "prior_optimizer_attempts_preserved": 100,
        "total_r0_optimizer_attempts_including_prior": 180,
        "reference_solver_calls": 0,
        "m10_projection_justified": True,
        "m10_gradient_diagnostic": str(gradient_path),
    }), encoding="utf-8")

    evidence = _load_r0_projection_evidence(report_path, checkpoint)

    assert evidence["projection_enabled_for_fit"] is True
    assert evidence["evidence_status"].startswith("recovered_train_only_fit")
    assert evidence["combined_gradient_gate_by_block"]["field_head"]["adverse"] is True
    assert evidence["individual_response_term_diagnostics"] == {"finite": {"field_head": -0.1}}
    invalid = json.loads(report_path.read_text(encoding="utf-8"))
    invalid["error"] = "different failure"
    report_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="neither a passed direct diagnostic nor the verified"):
        _load_r0_projection_evidence(report_path, checkpoint)


def test_response_resume_checkpoint_uses_restricted_weights_only_loading(tmp_path: Path) -> None:
    safe_path = tmp_path / "safe_response.pt"
    numpy_state = np.random.get_state()
    torch.save(
        {"model": {"weight": torch.tensor([1.0])}, "numpy_rng_state": numpy_state},
        safe_path,
    )
    loaded = _load_safe_response_checkpoint(safe_path)
    assert torch.equal(loaded["model"]["weight"], torch.tensor([1.0]))
    assert isinstance(loaded["numpy_rng_state"], tuple)

    marker = tmp_path / "arbitrary_pickle_executed"

    class DangerousPickle:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return os.system, (f"touch {marker}",)

    malicious_path = tmp_path / "untrusted_response.pt"
    torch.save({"model": {}, "untrusted": DangerousPickle()}, malicious_path)
    with pytest.raises(pickle.UnpicklingError):
        _load_safe_response_checkpoint(malicious_path)
    assert not marker.exists()


def test_paired_calibration_restores_through_late_persistent_buffer_materialization() -> None:
    class LazyPersistentField(_AbsoluteField):
        def forward(self, design, context, role_queries):
            if "calibration_marker" not in dict(self.named_buffers()):
                self.register_buffer("calibration_marker", torch.tensor([7.0]))
            return super().forward(design, context, role_queries)

    config = StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=1,
        checkpoint_every_updates=1,
        review_updates=(1,),
        stages=(TrainingStage("value", 0, 1, ("value",)),),
    )
    model = LazyPersistentField()
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )

    def optimizer_factory(current: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(current.parameters(), lr=1.0e-4)

    paired = run_paired_staged_fits(
        model,
        lambda current: current,
        optimizer_factory,
        (_stencil(),),
        scales=scales,
        mixed_specs=(),
        config=config,
    )

    assert paired.arms["R_value"].actual_optimizer_updates == 1
    assert paired.arms["R_response"].actual_optimizer_updates == 1
    assert paired.buffer_integrity["R_value"]["passed"] is True
    assert paired.buffer_integrity["R_value"]["calibration_added_names"] == [
        "calibration_marker"
    ]


def test_matched_review_rows_load_distinct_value_and_response_arm_states() -> None:
    model = _AbsoluteField()
    checkpoint_payloads = {
        "R_value": {"model": {"gain": torch.tensor(0.1)}},
        "R_response": {"model": {"gain": torch.tensor(1.1)}},
    }
    stencil = _stencil(EvidenceSplit.FINAL_REVIEW)
    review_errors: dict[str, float] = {}
    for arm in ("R_value", "R_response"):
        _restore_review_arm_state(model, arm, checkpoint_payloads)
        with torch.no_grad():
            rows = evaluate_stencil(model, stencil, pressure_limit=3.0)
        fluid_metric = next(
            row for row in rows["absolute_roles"]
            if row["role"] == "fluid_fields"
        )
        review_errors[arm] = float(fluid_metric["weighted_mae"])

    assert not np.isclose(review_errors["R_value"], review_errors["R_response"])


def test_r_arms_are_first_class_and_finite_peak_pressure_terms_are_continuous() -> None:
    stencil = _stencil()
    model = _AbsoluteField()
    scales = ThermalLossScales(
        value={name: 1.0 for name in ("fluid_fields", "interface", "solid_temperature")},
        finite={name: 1.0 for name in ("fluid_fields", "interface", "solid_temperature")},
        mixed={name: 1.0 for name in ("fluid_fields", "interface", "solid_temperature")},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
        pressure_limit_by_family={"family-1": 3.0},
    )
    predictions = predict_stencil(model, stencil)
    losses = compute_stencil_loss_terms(
        predictions,
        stencil,
        scales=scales,
        enabled_terms=("finite_peak", "pressure_value", "pressure_response"),
        include_feasibility_bce=False,
    )
    assert set(losses.terms) == {"finite_peak", "pressure_value", "pressure_response"}
    assert "finite_peak/heat/m0" in losses.diagnostics
    assert all(bool(torch.isfinite(term)) for term in losses.terms.values())
    assert losses.terms["finite_peak"].requires_grad

    config = StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=1,
        checkpoint_every_updates=1,
        review_updates=(1,),
        required_response_terms=("finite",),
        stages=(TrainingStage("response", 0, 1, ("value", "finite")),),
    )
    paired = run_paired_staged_fits(
        _AbsoluteField(),
        lambda current: current,
        lambda current: torch.optim.SGD(current.parameters(), lr=1.0e-4),
        [stencil],
        scales=scales,
        mixed_specs=(),
        config=config,
        stop_at_update=1,
    )
    assert set(paired.arms) == {"R_value", "R_response"}
    assert paired.arms["R_value"].actual_optimizer_updates == 1
    assert paired.arms["R_response"].actual_optimizer_updates == 1
    assert paired.arms["R_value"].history[0].active_terms == ("value",)
    assert paired.arms["R_response"].history[0].active_terms == ("value", "finite")
    assert paired.gradient_calibration["scope"] == "equal_family_train_calibration"
    calibration = paired.gradient_calibration["diagnostics"]
    assert calibration["family_gradient_rms"]["family-1"]["finite"] > 0.0
    assert calibration["family_gradient_to_value_ratio"]["family-1"]["finite"] > 0.0


def test_whole_callback_ad_fd_keeps_pressure_and_peak_derivatives_separate() -> None:
    stencil = _stencil()
    model = _AbsoluteField()
    heating = check_pressure_peak_ad_fd(
        model,
        stencil,
        variable="heating",
        step=1.0e-2,
    )
    geometry = check_pressure_peak_ad_fd(
        model,
        stencil,
        variable="position_x",
        step=1.0e-3,
    )
    assert heating["passed"] is True
    assert geometry["passed"] is True
    assert heating["pressure_drop"]["autodiff"] == pytest.approx(0.0, abs=1.0e-7)
    assert heating["solid_sampled_peak"]["autodiff"] > 0.0
    assert geometry["solid_sampled_peak"]["autodiff"] > 0.0


def test_geometry_probe_uses_typed_module_position_xy_and_safe_clearance() -> None:
    stencil = _stencil()
    slot, step = _select_geometry_x_probe(stencil)
    module = stencil.baseline.design.modules[slot]
    radius = float(stencil.baseline.context.values["module_radius"])
    domain_x = float(stencil.baseline.context.values["domain_length_x"])
    clearance = min(module.position_xy[0] - radius, domain_x - radius - module.position_xy[0])
    assert slot == 0
    assert 0.0 < step <= min(1.0e-3, clearance * 0.25)


def test_permutation_diagnostic_reports_interface_channels_and_physical_units() -> None:
    query = RoleQuery(
        role="interface",
        query_features=torch.zeros((2, 3)),
        channel_names=("surface_temperature", "normal_flux"),
        channel_units=("K", "W/m2"),
        receiver_slots=(0, 0),
        coordinate_kind="interface_material_angle",
    )
    left = AbsolutePrediction(
        {"interface": torch.tensor([[17.3, 26.3268], [0.0, 0.0]])}
    )
    right = AbsolutePrediction(
        {"interface": torch.tensor([[17.3000057, 26.32686], [0.0, 6.0e-5]])}
    )
    summary = _prediction_difference_summary(left, right, {"interface": query})
    channels = summary["role_channels"]["interface"]
    assert [(row["channel"], row["unit"]) for row in channels] == [
        ("surface_temperature", "K"),
        ("normal_flux", "W/m2"),
    ]
    assert channels[0]["max_absolute"] == pytest.approx(5.7e-6, abs=3.0e-8)
    assert channels[1]["max_absolute"] == pytest.approx(6.1e-5, abs=5.0e-8)
    assert channels[1]["allclose_at_3e-6_atol_3e-5_rtol"] is False
    assert channels[1]["permutation_allowed_max_absolute"] == pytest.approx(
        3.0e-6 + 3.0e-6 * 26.32686
    )
    assert summary["permutation_gate_passed"] is True

    outside = AbsolutePrediction(
        {"interface": torch.tensor([[17.3000057, 26.32686], [0.0, 1.0e-4]])}
    )
    rejected = _prediction_difference_summary(left, outside, {"interface": query})
    assert rejected["permutation_gate_passed"] is False


def test_immutable_sampling_metadata_serializes_without_asdict() -> None:
    summary = SamplingSummary(
        physical_family_id="family-1",
        split="train",
        original_counts={"fluid_fields": 10},
        sampled_counts={"fluid_fields": 5},
        protected_counts={"pressure_bands": 2},
        solid_peak_query_coverage={"baseline:m0": True},
        inverse_probability_weighting=True,
    )
    payload = _sampling_summary_mapping(summary)
    restored = json.loads(json.dumps(payload))
    assert restored["sampled_counts"] == {"fluid_fields": 5}
    assert restored["solid_peak_query_coverage"] == {"baseline:m0": True}
    generic = json.loads(
        json.dumps({"coverage": summary.solid_peak_query_coverage}, default=_json_default)
    )
    assert generic == {"coverage": {"baseline:m0": True}}


def test_sampler_protects_pressure_and_all_state_near_interface_union() -> None:
    fluid_x = np.linspace(0.0, 12.0, 241)
    stencil = ResponseStencil(
        _record("base", fluid_x=fluid_x),
        {
            "heat": _record("heat", heating=1.2, fluid_x=fluid_x),
            "shift": _record("shift", position_x=5.5, fluid_x=fluid_x),
            "both": _record("both", position_x=5.5, heating=1.2, fluid_x=fluid_x),
        },
    )
    sampled = sample_training_stencil(
        stencil,
        config=ReceiverSamplingConfig(
            max_fluid_queries=50,
            solid_queries_per_module=2,
            hot_solid_points_per_module=1,
            random_seed=12,
        ),
    )
    baseline = stencil.baseline.output.roles["fluid_fields"]
    sampled_role = sampled.stencil.baseline.output.roles["fluid_fields"]
    sampled_ids = set(sampled_role.query_ids)
    pressure_ids = {
        baseline.query_ids[index]
        for index, x_value in enumerate(baseline.query_features[:, 0])
        if x_value <= 0.08 * 12.0 or x_value >= 0.92 * 12.0
    }
    near_ids: set[str] = set()
    for record in stencil.records:
        role = record.output.roles["fluid_fields"]
        centers = np.asarray([module.position_xy for module in record.design.modules])
        surface_distance = np.linalg.norm(
            role.query_features[:, None, :2] - centers[None, :, :], axis=-1
        ).min(axis=1) - 0.2
        near = (surface_distance >= 0.0) & (surface_distance <= 0.25)
        near_ids.update(role.query_ids[index] for index in np.flatnonzero(near))

    assert pressure_ids <= sampled_ids
    assert near_ids <= sampled_ids
    assert len(near_ids) > 0
    assert sampled.summary.protected_counts["fluid_pressure_bands"] == len(pressure_ids)
    assert sampled.summary.protected_counts["fluid_near_interface_union"] == len(near_ids)
    assert sampled.summary.protected_counts["fluid_protected_union"] == len(pressure_ids | near_ids)
    assert sampled.summary.sampled_counts["fluid_fields"] == len(pressure_ids | near_ids)


def test_frozen_train_calibration_loads_exact_scales_and_response_weights(tmp_path: Path) -> None:
    stencil = _stencil()
    scales_path = tmp_path / "scales.json"
    scales_payload = {
        "frozen_scales": {
            "value": {"fluid_fields": [1.0], "interface": [2.0], "solid_temperature": [3.0]},
            "finite": {"fluid_fields": [0.1], "interface": [0.2], "solid_temperature": [0.3]},
            "mixed": {"fluid_fields": [0.1], "interface": [0.2], "solid_temperature": [0.3]},
            "pressure_value": 4.0,
            "pressure_response": 0.4,
            "pressure_boundary": 0.2,
            "solid_temperature": 3.0,
            "smooth_peak_beta": 1.0,
            "near_limit_band": 0.2,
            "pressure_limit_by_family": {"family-1": 2.1},
        }
    }
    scales_path.write_text(json.dumps(scales_payload), encoding="utf-8")
    scales, scales_source = _load_frozen_loss_scales(
        scales_path, (stencil,), smooth_peak_beta=1.0
    )
    assert scales.value["fluid_fields"] == (1.0,)
    assert scales.finite["interface"] == (0.2,)
    assert scales.pressure_limit == pytest.approx(2.1)
    assert scales.pressure_limit_by_family == {"family-1": 2.1}
    assert len(scales_source["sha256"]) == 64

    weights_path = tmp_path / "weights.json"
    weights = {"value": 1.0, "finite": 0.07, "decision": 0.4, "constraint": 0.003}
    weights_path.write_text(json.dumps({"calibrated_response_weights": weights}), encoding="utf-8")
    loaded_weights, weights_source = _load_frozen_response_weights(weights_path)
    assert loaded_weights == weights
    assert len(weights_source["sha256"]) == 64

    scales_payload["frozen_scales"]["pressure_limit_by_family"] = {"other-family": 2.1}
    scales_path.write_text(json.dumps(scales_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match the training panel"):
        _load_frozen_loss_scales(scales_path, (stencil,), smooth_peak_beta=1.0)
    scales_payload["frozen_scales"]["pressure_limit_by_family"] = {"family-1": 2.2}
    scales_path.write_text(json.dumps(scales_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match 1.05x its train baseline"):
        _load_frozen_loss_scales(scales_path, (stencil,), smooth_peak_beta=1.0)


def _resume_provenance_fixture(tmp_path: Path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    train_atlas = tmp_path / "train_case.npz"
    train_atlas.write_bytes(b"train atlas")
    train_metadata = train_atlas.with_suffix(".json")
    train_metadata.write_text("{}", encoding="utf-8")
    dev_atlas = tmp_path / "dev_case.npz"
    dev_atlas.write_bytes(b"development atlas")
    scales_path = tmp_path / "scales.json"
    scale_snapshot = {"value": {"fluid_fields": [1.0]}, "pressure_limit": 2.0}
    scales_path.write_text(json.dumps({"frozen_scales": scale_snapshot}), encoding="utf-8")

    config = replace(load_staged_training_config(), max_wall_seconds=900.0)
    weights = {"value": 1.0, "finite": 0.07, "decision": 0.4, "constraint": 0.003}
    sampling = [{"physical_family_id": "train-family", "sampled_counts": {"fluid_fields": 3}}]
    refit_config = {
        "forward_architecture": "three_term_full_access_honf",
        "source_prediction_mode": "local_surrogate",
        "prediction_identity_claim": False,
    }
    checkpoint_paths = {}
    checkpoint_entries = {}
    payloads = {}
    rng = {
        "python_rng_state": (1, (2, 3), None),
        "numpy_rng_state": ("MT19937", np.asarray([4, 5], dtype=np.uint32), 0, 0, 0.0),
        "torch_rng_state": torch.tensor([6, 7], dtype=torch.uint8),
        "cuda_rng_state_by_model_device": None,
        "sampler_rng_state": (3, (8, 9), None),
        "sampler_remaining_order": [0, 0, 0],
    }
    for arm in ("B_value", "B_response"):
        checkpoint = tmp_path / f"{arm}_u100.pt"
        checkpoint.write_bytes(arm.encode())
        checkpoint_paths[arm] = checkpoint
        checkpoint_entries[arm] = {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)}
        payloads[arm] = {
            "arm": arm,
            "actual_optimizer_updates": 100,
            "attempted_optimizer_steps": 100,
            "model": {"weight": torch.tensor([1.0])},
            "optimizer": {"state": {}, "param_groups": [{}]},
            "training_config": training_config_mapping(config, arm=arm),
            **rng,
            "python_rng_state": (1, (2, 3), None) if arm == "B_value" else (1, (2, 4), None),
            "torch_rng_state": torch.tensor([6, 7], dtype=torch.uint8)
            if arm == "B_value"
            else torch.tensor([6, 8], dtype=torch.uint8),
            "calibrated_loss_weights": weights if arm == "B_response" else {},
        }

    fit_path = tmp_path / "response_control_manifest.json"
    fit = {
        "status": "passed",
        "mode": "paired_staged_fit",
        "checkpoint": str(source.resolve()),
        "checkpoint_sha256": file_sha256(source),
        "train_atlas_paths": [str(train_atlas.resolve())],
        "development_paths": [str(dev_atlas.resolve())],
        "train_family_ids": ["train-family"],
        "train_sampling": sampling,
        "refit_config": refit_config,
        "loss_scales_source": {"path": str(scales_path.resolve()), "sha256": file_sha256(scales_path)},
        "loss_scales": scale_snapshot,
        "review_cap": 100,
        "review_decisions": {"100": "stop"},
        "arms": {
            arm: {
                "initial_update": 0,
                "actual_optimizer_updates": 100,
                "attempted_optimizer_steps": 100,
                "total_attempted_optimizer_steps": 100,
                "final_update": 100,
            }
            for arm in ("B_value", "B_response")
        },
        "checkpoint_paths": {
            arm: [str(checkpoint_paths[arm].resolve())]
            for arm in ("B_value", "B_response")
        },
        "calibrated_response_weights": weights,
    }
    fit_path.write_text(json.dumps(fit), encoding="utf-8")

    replay_path = tmp_path / "train_u100_replay_manifest.json"
    replay = {
        "status": "passed",
        "mode": "read_only_full_grid_train_replay",
        "optimizer_updates": 0,
        "reference_solves": 0,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "refit_config": refit_config,
        "train_stencils": [
            {
                "path": str(train_atlas.resolve()),
                "sha256": file_sha256(train_atlas),
                "json_sha256": file_sha256(train_metadata),
            }
        ],
        "arm_checkpoints": checkpoint_entries,
    }
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    kwargs = {
        "fit_manifest_path": fit_path,
        "replay_manifest_path": replay_path,
        "source_checkpoint_path": source,
        "train_atlas_paths": [train_atlas],
        "development_atlas_paths": [dev_atlas],
        "train_family_ids": ["train-family"],
        "train_sampling": sampling,
        "frozen_scales_path": scales_path,
        "loss_scales": scale_snapshot,
        "refit_config": refit_config,
        "resume_checkpoint_paths": checkpoint_paths,
        "resume_payloads": payloads,
        "training_config": config,
    }
    return kwargs, fit_path, replay_path, checkpoint_paths


def test_segmented_resume_gate_accounting_preserves_updates_and_attempts() -> None:
    # This no-optimizer fixture mirrors the staged native schedule. The
    # second manifest reports 300 local updates while its checkpoint is at
    # cumulative u500; that manifest must remain eligible for a u600 resume.
    staged_arms = (
        {"initial_update": 0, "actual_optimizer_updates": 200, "attempted_optimizer_steps": 200, "total_attempted_optimizer_steps": 200, "final_update": 200},
        {"initial_update": 200, "actual_optimizer_updates": 300, "attempted_optimizer_steps": 300, "total_attempted_optimizer_steps": 500, "final_update": 500},
        {"initial_update": 500, "actual_optimizer_updates": 100, "attempted_optimizer_steps": 100, "total_attempted_optimizer_steps": 600, "final_update": 600},
    )
    for gate, fit_arm in zip((200, 500, 600), staged_arms, strict=True):
        _validate_fit_arm_gate_accounting(fit_arm, arm="R_response", required_update=gate)

    with pytest.raises(ValueError, match="fit manifest does not end at u500"):
        _validate_fit_arm_gate_accounting(
            {
                "initial_update": 200,
                "actual_optimizer_updates": 299,
                "attempted_optimizer_steps": 300,
                "final_update": 500,
            },
            arm="R_response",
            required_update=500,
        )

    with pytest.raises(ValueError, match="attempt accounting differs from its completed updates"):
        _validate_fit_arm_gate_accounting(
            {
                "initial_update": 200,
                "actual_optimizer_updates": 300,
                "attempted_optimizer_steps": 301,
                "total_attempted_optimizer_steps": 500,
                "final_update": 500,
            },
            arm="R_response",
            required_update=500,
        )


def test_resumed_u500_manifest_is_eligible_for_u600_gate_resume(tmp_path: Path) -> None:
    kwargs, fit_path, _replay_path, _checkpoints = _resume_provenance_fixture(tmp_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit["review_cap"] = 500
    fit["review_decisions"] = {"500": "stop"}
    for arm in ("B_value", "B_response"):
        fit["arms"][arm].update({
            "initial_update": 200,
            "actual_optimizer_updates": 300,
            "attempted_optimizer_steps": 300,
            "total_attempted_optimizer_steps": 500,
            "final_update": 500,
        })
        kwargs["resume_payloads"][arm]["actual_optimizer_updates"] = 500
        kwargs["resume_payloads"][arm]["attempted_optimizer_steps"] = 500
    fit_path.write_text(json.dumps(fit), encoding="utf-8")

    provenance = validate_paired_resume_provenance(**kwargs, required_update=500)
    assert provenance["resume_gate_update"] == 500

    fit["arms"]["B_response"]["attempted_optimizer_steps"] = 299
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    with pytest.raises(ValueError, match="attempt accounting differs from its completed updates"):
        validate_paired_resume_provenance(**kwargs, required_update=500)


def test_scale_only_resume_provenance_binds_pair_scales_and_rng(tmp_path: Path) -> None:
    kwargs, _fit_path, _replay_path, _checkpoints = _resume_provenance_fixture(tmp_path)
    provenance = validate_paired_resume_provenance(**kwargs)
    assert provenance["resume_gate_update"] == 100
    assert provenance["read_only_replay_manifest"]["optimizer_updates"] == 0
    assert provenance["frozen_loss_scales"]["sha256"] == file_sha256(kwargs["frozen_scales_path"])
    assert provenance["calibrated_response_weights"]["finite"] == 0.07
    assert (
        "training schedule, per-arm RNG state, and shared sampler continuity"
        in provenance["validated_contracts"]
    )


def test_scale_only_resume_rejects_recipe_source_or_checkpoint_mismatch(tmp_path: Path) -> None:
    kwargs, _fit_path, _replay_path, checkpoints = _resume_provenance_fixture(tmp_path)
    wrong_source = tmp_path / "wrong_source.pt"
    wrong_source.write_bytes(b"different source")
    with pytest.raises(ValueError, match="source checkpoint path differs"):
        validate_paired_resume_provenance(**{**kwargs, "source_checkpoint_path": wrong_source})

    wrong_sampling = [{"physical_family_id": "train-family", "sampled_counts": {"fluid_fields": 4}}]
    with pytest.raises(ValueError, match="sampled training panel changed"):
        validate_paired_resume_provenance(**{**kwargs, "train_sampling": wrong_sampling})

    changed_config = dict(kwargs["resume_payloads"]["B_response"]["training_config"])
    changed_config["random_seed"] += 1
    changed_payload = dict(kwargs["resume_payloads"]["B_response"])
    changed_payload["training_config"] = changed_config
    with pytest.raises(ValueError, match="B_response checkpoint training schedule changed"):
        validate_paired_resume_provenance(
            **{
                **kwargs,
                "resume_payloads": {
                    **kwargs["resume_payloads"],
                    "B_response": changed_payload,
                },
            }
        )

    original = checkpoints["B_response"].read_bytes()
    checkpoints["B_response"].write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="B_response checkpoint SHA-256 differs"):
        validate_paired_resume_provenance(**kwargs)


def test_scale_only_resume_rejects_changed_frozen_scale_source(tmp_path: Path) -> None:
    kwargs, _fit_path, _replay_path, _checkpoints = _resume_provenance_fixture(tmp_path)
    scales_path = Path(kwargs["frozen_scales_path"])
    scales_path.write_text('{"frozen_scales": {"value": {"fluid_fields": [9.0]}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="frozen scale source SHA-256 differs"):
        validate_paired_resume_provenance(**kwargs)


def test_native_resume_rejects_changed_output_scope_and_shape(tmp_path: Path) -> None:
    kwargs, fit_path, _replay_path, _checkpoints = _resume_provenance_fixture(tmp_path)
    historical_dataset = tmp_path / "packed_train.h5"
    historical_dataset.write_bytes(b"train cases")
    historical_order = ["train0001", "train0002"]
    native_scope = {"mode": "final_output_heads", "names": ["field_head.weight"]}
    inventory = {
        "trainable_parameter_names": ["field_head.weight"],
        "trainable_parameter_shapes": {"field_head.weight": [2, 3]},
    }
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit["initialization_mode"] = "native_checkpoint"
    fit["native_trainable_scope"] = native_scope
    fit["parameter_inventory"] = inventory
    fit["historical_value_replay"] = {
        "dataset": str(historical_dataset.resolve()),
        "dataset_size_bytes": historical_dataset.stat().st_size,
        "dataset_mtime_ns": historical_dataset.stat().st_mtime_ns,
        "train_case_order": historical_order,
    }
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    payloads = {
        arm: {
            **payload,
            "historical_case_order": historical_order,
            "historical_next_index": 100 % len(historical_order),
        }
        for arm, payload in kwargs["resume_payloads"].items()
    }
    native_kwargs = {
        **kwargs,
        "native_trainable_scope": native_scope,
        "native_parameter_inventory": inventory,
        "historical_case_order": historical_order,
        "historical_dataset_path": historical_dataset,
        "resume_payloads": payloads,
    }
    assert validate_paired_resume_provenance(**native_kwargs)["resume_gate_update"] == 100
    with pytest.raises(ValueError, match="native trainable scope changed"):
        validate_paired_resume_provenance(
            **{**native_kwargs, "native_trainable_scope": {**native_scope, "names": ["other.weight"]}}
        )
    with pytest.raises(ValueError, match="native trainable_parameter_shapes changed"):
        validate_paired_resume_provenance(
            **{
                **native_kwargs,
                "native_parameter_inventory": {
                    **inventory,
                    "trainable_parameter_shapes": {"field_head.weight": [3, 3]},
                },
            }
        )


def test_gradient_calibration_includes_historical_value_objective() -> None:
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    model = _AbsoluteField()
    stencil = _stencil()
    stencil_only = calibrate_operator_weights(
        model, (stencil,), scales=scales, parameters=model.parameters()
    )
    historical_record = _record("historical-train", heating=2.0)
    source = SimpleNamespace(
        case_ids=(historical_record.record_id,),
        load=lambda case_id: historical_record,
    )
    diagnostics: dict[str, Any] = {}
    combined = calibrate_operator_weights(
        model, (stencil,), scales=scales, historical_value_source=source,
        parameters=model.parameters(),
        enabled_terms=("value", "finite", "finite_peak", "pressure_value", "pressure_response"),
        include_feasibility_bce=False,
        diagnostic_sink=diagnostics,
    )
    assert combined["value"] == pytest.approx(1.0)
    assert combined["finite"] != pytest.approx(stencil_only["finite"])
    assert diagnostics["family_ids"] == ["family-1"]
    assert diagnostics["family_gradient_rms"]["family-1"]["finite"] > 0.0
    assert diagnostics["family_gradient_to_value_ratio"]["family-1"]["finite"] > 0.0
    assert set(diagnostics["enabled_terms"]) == {
        "value", "finite", "finite_peak", "pressure_value", "pressure_response"
    }


def test_r1_historical_scale_examples_are_assigned_per_sorted_family_slot() -> None:
    first = _record("hist-2", case_dir="hist-2")
    second = replace(
        _record("hist-1", case_dir="hist-1"),
        design=replace(_record("hist-1", case_dir="hist-1").design, physical_family_id="family-2"),
    )
    source = SimpleNamespace(
        case_ids=("hist-1", "hist-2"),
        load=lambda case_id: {"hist-1": second, "hist-2": first}[case_id],
    )
    assert _historical_calibration_case_map(("family-2", "family-1"), source) == {
        "family-1": "hist-1",
        "family-2": "hist-2",
    }
    stencil = _stencil()
    without_historical = derive_training_scales((stencil,))
    with_historical = derive_training_scales((stencil,), historical_value_source=source)
    assert with_historical.value["fluid_fields"] != without_historical.value["fluid_fields"]


def test_paired_fit_uses_frozen_response_weights_without_recalibration(monkeypatch) -> None:
    config = StagedTrainingConfig(
        arm="B_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=1,
        checkpoint_every_updates=1,
        review_updates=(1,),
        stages=(TrainingStage("value", 0, 1, ("value",)),),
    )
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    frozen_weights = {"value": 1.0, "finite": 0.07, "decision": 0.4, "constraint": 0.003}

    def reject_recalibration(*args, **kwargs):
        pytest.fail("Frozen recipe calibration must bypass gradient recalibration.")

    monkeypatch.setattr(paired_module, "calibrate_operator_weights", reject_recalibration)
    model = _AbsoluteField()
    result = run_paired_staged_fits(
        model,
        lambda current: current,
        lambda current: torch.optim.SGD(current.parameters(), lr=1.0e-4),
        (_stencil(),),
        scales=scales,
        mixed_specs=(),
        config=config,
        stop_at_update=1,
        fixed_response_weights=frozen_weights,
    )
    assert dict(result.calibrated_response_weights) == frozen_weights
    assert result.arms["B_value"].actual_optimizer_updates == 1
    assert result.arms["B_response"].actual_optimizer_updates == 1


def test_family_pressure_limits_remain_anchored_to_each_original_start() -> None:
    first = _stencil()
    second_family = "family-2"

    def update_record(record: SolveRecord, pressure: float) -> SolveRecord:
        assert record.output is not None
        design = replace(record.design, physical_family_id=second_family)
        quantities = dict(record.output.quantities)
        quantities["pressure_drop"] = MeasuredQuantity(pressure, "Pa")
        return replace(record, design=design, output=replace(record.output, quantities=quantities))

    second = ResponseStencil(
        update_record(first.baseline, 4.0),
        {label: update_record(record, 4.1) for label, record in first.variants.items()},
    )
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    ).with_train_reference_limits((first, second))
    assert scales.pressure_limit_by_family == {"family-1": 2.1, "family-2": 4.2}


def test_paired_review_gate_stops_by_default_and_requires_recorded_continuation() -> None:
    config = StagedTrainingConfig(
        arm="B_response",
        max_optimizer_updates=2,
        max_epochs=2,
        total_optimizer_update_ceiling=2,
        checkpoint_every_updates=1,
        review_updates=(1, 2),
        stages=(TrainingStage("value", 0, 2, ("value",)),),
    )
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )

    def run(continuations: tuple[int, ...]):
        model = _AbsoluteField()
        return run_paired_staged_fits(
            model,
            lambda current: current,
            lambda current: torch.optim.SGD(current.parameters(), lr=1.0e-4),
            (_stencil(),),
            scales=scales,
            mixed_specs=(),
            config=config,
            stop_at_update=2 if continuations else 1,
            review_continuations=continuations,
        )

    stopped = run(())
    assert stopped.arms["B_value"].final_update == 1
    assert stopped.arms["B_response"].final_update == 1
    assert stopped.review_decisions == {1: "stop"}
    assert tuple(step.training_stencil_index for step in stopped.arms["B_value"].history) == tuple(
        step.training_stencil_index for step in stopped.arms["B_response"].history
    )

    continued = run((1,))
    assert continued.arms["B_value"].final_update == 2
    assert continued.arms["B_response"].final_update == 2
    assert continued.review_decisions == {1: "continue", 2: "stop"}


def test_paired_zero_update_checkpoint_roundtrip_serializes_both_arms(tmp_path: Path) -> None:
    config = StagedTrainingConfig(
        arm="B_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=1,
        checkpoint_every_updates=1,
        max_wall_seconds=1.0e-12,
        review_updates=(1,),
        stages=(TrainingStage("value", 0, 1, ("value",)),),
    )
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    model = _AbsoluteField()
    optimizers: dict[str, torch.optim.Optimizer] = {}
    saved_labels: dict[str, list[str]] = {"B_value": [], "B_response": []}

    def make_optimizer(current: torch.nn.Module) -> torch.optim.Optimizer:
        arm = "B_value" if "B_value" not in optimizers else "B_response"
        optimizer = torch.optim.SGD(current.parameters(), lr=1.0e-4)
        optimizers[arm] = optimizer
        return optimizer

    def roundtrip(arm: str, payload: dict[str, Any], label: str) -> None:
        saved_labels[arm].append(label)
        path = tmp_path / f"{arm}.pt"
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        arm_config = StagedTrainingConfig.from_mapping(loaded["training_config"])
        restored_update, attempted, _sampler, remaining, _weights = restore_checkpoint_payload(
            model, optimizers[arm], loaded, config=arm_config
        )
        assert restored_update == 0
        assert attempted == 0
        assert remaining == []
        assert loaded["sampler_rng_state"] is not None
        legacy_payload = dict(loaded)
        legacy_config = dict(loaded["training_config"])
        for key in (
            "deterministic_eval_mode",
            "project_response_gradient_blockwise",
            "response_ramp_start_update",
            "response_ramp_end_update",
            "response_ramp_terms",
            "required_response_terms",
            "include_feasibility_bce",
        ):
            legacy_config.pop(key)
        legacy_payload["training_config"] = legacy_config
        restore_checkpoint_payload(
            model, optimizers[arm], legacy_payload, config=arm_config
        )

    paired = run_paired_staged_fits(
        model,
        lambda current: current,
        make_optimizer,
        (_stencil(),),
        scales=scales,
        mixed_specs=(),
        config=config,
        stop_at_update=1,
        on_checkpoint=roundtrip,
    )
    assert paired.arms["B_value"].actual_optimizer_updates == 0
    assert paired.arms["B_response"].actual_optimizer_updates == 0
    assert saved_labels == {
        "B_value": ["zero_update_preflight"],
        "B_response": ["zero_update_preflight"],
    }


def test_held_stencil_evaluator_reports_role_pressure_and_peak_evidence() -> None:
    stencil = _stencil(EvidenceSplit.FINAL_REVIEW)
    rows = evaluate_stencil(
        _AbsoluteField(),
        stencil,
        pressure_limit=1.9,
        mixed_specs=(MixedResponseSpec("interface", "both", "heat", "shift"),),
    )
    assert rows["absolute_roles"]
    assert rows["finite_roles"]
    assert rows["mixed_roles"]
    assert len(rows["solid_peak_summary"]) == 2 * len(stencil.records)
    assert all(row["units"] == "K" for row in rows["solid_peak_summary"])
    assert all(row["units"] == "K" for row in rows["solid_peak_response_summary"])
    baseline_pressure = next(
        row for row in rows["pressure"] if row["kind"] == "absolute" and row["label"] == "baseline"
    )
    assert baseline_pressure["reference_feasible"] is False
    assert baseline_pressure["predicted_feasible"] is True
    assert baseline_pressure["false_feasible"] is True
    finite_pressure = next(row for row in rows["pressure"] if row["kind"] == "finite")
    assert finite_pressure["reference_feasible"] is None
    assert finite_pressure["predicted_feasible"] is None
    assert finite_pressure["false_feasible"] is None


def test_held_stencil_evaluator_reports_full_grid_near_interface_rows() -> None:
    x = np.asarray(
        [0.2, 0.9, 5.0, 5.21, 5.25, 5.44, 4.75, 4.74, 11.1, 11.8],
        dtype=np.float64,
    )
    stencil = ResponseStencil(
        _record("base", fluid_x=x),
        {
            "heat": _record("heat", heating=1.2, fluid_x=x),
            "shift": _record("shift", position_x=5.5, fluid_x=x),
            "both": _record("both", position_x=5.5, heating=1.2, fluid_x=x),
        },
    )
    rows = evaluate_stencil(
        _AbsoluteField(), stencil, pressure_limit=1.9, smooth_peak_beta=1.0
    )
    baseline = next(
        row for row in rows["near_interface_fluid"] if row["label"] == "baseline"
    )
    role = stencil.baseline.output.roles["fluid_fields"]  # type: ignore[union-attr]
    expected_near = np.asarray(
        [False, False, False, True, True, True, True, True, False, False]
    )
    assert baseline["observed_count"] == int(expected_near.sum())
    assert baseline["reference_rms"] == pytest.approx(
        np.sqrt(np.mean(np.square(role.values[expected_near, 0])))
    )


def test_all_feasible_stencil_skips_feasibility_bce_but_keeps_pressure_losses() -> None:
    stencil = _stencil()
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
        pressure_limit_by_family={"family-1": 3.0},
    )
    predictions = predict_stencil(_AbsoluteField(), stencil)
    terms = compute_stencil_loss_terms(
        predictions, stencil, scales=scales, enabled_terms=("constraint",)
    )
    assert "constraint" in terms.terms
    assert terms.diagnostics["constraint/feasibility_bce_skipped_single_class"].item() == 1.0
    assert "constraint/pressure_value_mean" in terms.diagnostics
    assert "constraint/pressure_response_mean" in terms.diagnostics
    assert not any(key.endswith("/feasibility_bce") for key in terms.diagnostics)


def test_serialization_failure_keeps_successful_solve_and_does_not_reissue(tmp_path: Path) -> None:
    raw_case = tmp_path / "raw-solver-case"
    raw_case.mkdir()
    (raw_case / "solver.log").write_text("successful raw solve\n", encoding="utf-8")
    calls: list[str] = []

    def successful_solve() -> SolveRecord:
        calls.append("solver")
        return _record("successful", case_dir=str(raw_case))

    record = successful_solve()

    def fail_serialization(directory: Path, _record: SolveRecord) -> None:
        (directory / "partial.bin").write_bytes(b"partial")
        raise OSError("forced serialization failure")

    failed = persist_solve_record(record, tmp_path / "failed-record", writer=fail_serialization)
    assert failed.serialization_status == "serialization_failed"
    assert failed.solve_status is SolveStatus.CONVERGED
    assert failed.solver_invoked is True
    assert failed.raw_solver_completed is True
    assert failed.physical_wall_time_available is True
    assert failed.physical_attempt_count == 1
    assert failed.raw_case_dir == str(raw_case)
    assert (raw_case / "solver.log").is_file()
    assert not failed.destination.exists()
    assert calls == ["solver"]


def test_solve_record_storage_round_trips_raw_path_and_arrays(tmp_path: Path) -> None:
    raw_case = tmp_path / "raw-solver-case"
    raw_case.mkdir()
    (raw_case / "raw.dat").write_text("kept", encoding="utf-8")
    record = _record("roundtrip", case_dir=str(raw_case))
    result = persist_solve_record(record, tmp_path / "stored-record")
    assert result.serialization_status == "stored"
    restored = load_solve_record(result.destination)
    assert restored.record_id == record.record_id
    assert restored.output is not None
    assert restored.output.case_dir == str(raw_case)
    np.testing.assert_array_equal(
        restored.output.roles["fluid_fields"].values,
        record.output.roles["fluid_fields"].values,
    )
    assert (raw_case / "raw.dat").is_file()


def test_paired_historical_value_replay_matches_arms_and_exact_resume() -> None:
    class TwoCaseSource:
        case_ids = ("historical-A", "historical-B")

        def load(self, case_id: str) -> SolveRecord:
            return {
                "historical-A": _record("historical-A", heating=0.8),
                "historical-B": _record("historical-B", heating=1.3),
            }[case_id]

    source = TwoCaseSource()
    config = StagedTrainingConfig(
        arm="B_response",
        max_optimizer_updates=2,
        max_epochs=2,
        total_optimizer_update_ceiling=2,
        checkpoint_every_updates=1,
        review_updates=(1, 2),
        stages=(TrainingStage("value", 0, 2, ("value",)),),
    )
    scales = ThermalLossScales(
        value={name: 1.0 for name in ("fluid_fields", "interface", "solid_temperature")},
        finite={name: 1.0 for name in ("fluid_fields", "interface", "solid_temperature")},
        mixed={name: 1.0 for name in ("fluid_fields", "interface", "solid_temperature")},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    snapshots: dict[tuple[str, int], dict[str, Any]] = {}

    def checkpoint(arm: str, payload: dict[str, Any], _label: str) -> None:
        snapshots[(arm, payload["actual_optimizer_updates"])] = payload

    def fit(model: _AbsoluteField, *, replay_source: Any = source, **kwargs: Any):
        return run_paired_staged_fits(
            model,
            lambda current: current,
            lambda current: torch.optim.SGD(current.parameters(), lr=1e-4),
            [_stencil()],
            historical_value_source=replay_source,
            scales=scales,
            mixed_specs=(),
            config=config,
            stop_at_update=2,
            review_continuations=(1,),
            on_checkpoint=checkpoint,
            **kwargs,
        )

    result = fit(_AbsoluteField(), fixed_response_weights={
        "value": 1.0, "finite": 1.0, "decision": 1.0, "constraint": 1.0,
    })
    for arm in ("B_value", "B_response"):
        steps = result.arms[arm].history
        assert [step.historical_case_id for step in steps] == list(source.case_ids)
        assert all(np.isfinite(step.term_losses["historical_value"]) for step in steps)
        assert snapshots[(arm, 1)]["historical_case_order"] == list(source.case_ids)
        assert snapshots[(arm, 1)]["historical_next_index"] == 1
    assert result.arms["B_value"].history[0].term_losses["historical_value"] == result.arms["B_response"].history[0].term_losses["historical_value"]

    first_gate = {arm: snapshots[(arm, 1)] for arm in ("B_value", "B_response")}
    original_final = {
        arm: {name: tensor.clone() for name, tensor in snapshots[(arm, 2)]["model"].items()}
        for arm in first_gate
    }
    resumed = fit(_AbsoluteField(), resume_payloads=first_gate)
    assert all(result.arms[arm].history[1].historical_case_id == resumed.arms[arm].history[0].historical_case_id for arm in first_gate)
    for arm in first_gate:
        for name, tensor in original_final[arm].items():
            torch.testing.assert_close(tensor, snapshots[(arm, 2)]["model"][name], rtol=0, atol=0)

    class ReorderedSource(TwoCaseSource):
        case_ids = ("historical-B", "historical-A")

    with pytest.raises(ValueError, match="historical train cohort/order differs"):
        fit(_AbsoluteField(), replay_source=ReorderedSource(), resume_payloads=first_gate)


def test_historical_sampler_preserves_pressure_ports_and_each_exact_peak() -> None:
    original = _record("historical-coverage")
    sampled = _sample_value_record(
        original,
        ReceiverSamplingConfig(max_fluid_queries=2, solid_queries_per_module=1, hot_solid_points_per_module=1),
    )
    assert sampled.output is not None and original.output is not None
    original_fluid = original.output.roles["fluid_fields"]
    fluid = sampled.output.roles["fluid_fields"]
    required = {
        original_fluid.query_ids[index]
        for index, x in enumerate(original_fluid.query_features[:, 0])
        if x <= 0.08 * 12.0 or x >= 0.92 * 12.0
    }
    assert required.issubset(fluid.query_ids)
    assert fluid.values.shape[0] >= len(required)
    assert sampled.output.roles["interface"].query_ids == original.output.roles["interface"].query_ids
    solid = sampled.output.roles["solid_temperature"]
    assert "m0:solid:1" in solid.query_ids


def test_checkpoint_only_review_provenance_requires_exact_u200_pair_and_zero_updates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Run1804_e4738.pt"
    source.write_bytes(b"trusted source checkpoint fixture")
    recipe = Path(__file__).resolve().parents[1] / "configs" / "response_control_native_nonlinear_interface.json"
    r0 = tmp_path / "r0.json"
    r0.write_text('{"status":"passed"}', encoding="utf-8")
    scales_path = tmp_path / "native_train_loss_scales.json"
    frozen_scales = {
        "pressure_limit_by_family": {f"family-{index}": 1.05 for index in range(8)},
        "smooth_peak_beta": 1.0,
    }
    scales_path.write_text(json.dumps({"frozen_scales": frozen_scales}), encoding="utf-8")
    train_paths = [tmp_path / f"train-{index}.npz" for index in range(8)]
    dev_paths = [tmp_path / f"dev-{index}.npz" for index in range(4)]
    for path in (*train_paths, *dev_paths):
        path.write_bytes(path.name.encode())
        path.with_suffix(".json").write_text("{}", encoding="utf-8")
    failed_manifest = tmp_path / "response_control_manifest.json"
    failed_manifest.write_text(json.dumps({
        "status": "failed",
        "mode": "paired",
        "initialization_mode": "native_checkpoint",
        "error_type": "ValueError",
        "error": "Role 'fluid_fields' quadrature weights differ across stencil members.",
        "checkpoint": str(source.resolve()),
        "recipe_config": str(recipe.resolve()),
        "r0_diagnostic_json": str(r0.resolve()),
        "train_stencils": [str(path.resolve()) for path in train_paths],
        "development_stencils": [str(path.resolve()) for path in dev_paths],
        "max_wall_seconds": 1800.0,
    }), encoding="utf-8")
    config = replace(
        load_staged_training_config(str(recipe), arm="R_response"),
        max_wall_seconds=1800.0,
    )
    scope = {
        "name": "native_nonlinear_interface",
        "trainable_parameter_names": ["head.weight"],
    }
    scale_source = {"path": str(scales_path.resolve()), "sha256": file_sha256(scales_path)}
    calibration = {
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "training_recipe": str(recipe.resolve()),
        "training_recipe_sha256": file_sha256(recipe),
        "loss_scales_source": scale_source,
        "loss_scales": frozen_scales,
        "active_scope": scope,
        "frozen_buffer_checkpoint_audit": {"passed": True},
    }
    arm_paths = {
        arm: tmp_path / f"response_control_{arm}_training_checkpoint_u00200.pt"
        for arm in ("R_value", "R_response")
    }
    arm_payloads: dict[str, dict[str, Any]] = {}
    for arm, path in arm_paths.items():
        path.write_bytes(f"trusted {arm} u200 fixture".encode())
        arm_payloads[arm] = {
            "arm": arm,
            "actual_optimizer_updates": 200,
            "attempted_optimizer_steps": 200,
            "training_config": training_config_mapping(config, arm=arm),
            "model": {"head.weight": torch.tensor([1.0])},
            "response_control_calibration_provenance": calibration,
            "calibrated_loss_weights": {"value": 1.0, "finite": 0.2},
            "sampler_rng_state": (1, (2, 3), None),
            "sampler_remaining_order": [1, 0],
            "historical_case_order": ["case-1", "case-2"],
            "historical_next_index": 2,
        }

    kwargs = {
        "failed_manifest_path": failed_manifest,
        "source_checkpoint_path": source,
        "recipe_path": recipe,
        "r0_diagnostic_path": r0,
        "frozen_scales_path": scales_path,
        "train_atlas_paths": train_paths,
        "development_atlas_paths": dev_paths,
        "arm_checkpoint_paths": arm_paths,
        "arm_payloads": arm_payloads,
        "training_config": config,
    }
    evidence = validate_checkpoint_only_review_provenance(**kwargs)
    assert evidence["mode"] == "checkpoint_only_review"
    assert evidence["optimizer_calls"] == 0
    assert evidence["optimizer_instances_created"] == 0
    assert evidence["r0_content_hash_attested_by_source_run"] is False
    assert set(evidence["checkpoints"]) == {"R_value", "R_response"}

    changed_payloads = {arm: dict(payload) for arm, payload in arm_payloads.items()}
    changed_payloads["R_response"]["attempted_optimizer_steps"] = 199
    with pytest.raises(ValueError, match="exactly 200 attempted and completed"):
        validate_checkpoint_only_review_provenance(**{**kwargs, "arm_payloads": changed_payloads})


def test_checkpoint_review_cli_mode_is_explicitly_u200_only() -> None:
    parsed = build_parser().parse_args([
        "--mode", "checkpoint_review",
        "--checkpoint", "Run1804_e4738.pt",
        "--train-stencil", "train.npz",
        "--output-dir", "diagnostics/generated/review",
    ])
    assert parsed.mode == "checkpoint_review"
    assert parsed.review_update == 200


def test_atomic_review_json_recursively_serializes_config_stages_and_enums(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "response_control_native_nonlinear_interface.json"
    config = load_staged_training_config(str(config_path), arm="R_response")
    path = tmp_path / "review_progress.json"
    _atomic_json(path, {
        "training_config": config,
        "split": EvidenceSplit.TRAIN,
        "stages": (config.stages[0], config.stages[1]),
    })
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["training_config"]["arm"] == "R_response"
    assert written["training_config"]["stages"][0]["name"] == config.stages[0].name
    assert written["stages"][1]["active_terms"] == list(config.stages[1].active_terms)
    assert written["split"] == EvidenceSplit.TRAIN.value


def test_expanded_review500_recipe_changes_only_review_schedule() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    probe_path = config_dir / "response_control_native_expanded_interface.json"
    paired_path = config_dir / "response_control_native_expanded_interface_review500.json"
    probe_recipe = json.loads(probe_path.read_text(encoding="utf-8"))
    paired_recipe = json.loads(paired_path.read_text(encoding="utf-8"))
    probe_schedule = probe_recipe.pop("review_updates")
    paired_schedule = paired_recipe.pop("review_updates")
    assert probe_schedule == [200, 600]
    assert paired_schedule == [200, 500, 600]
    assert probe_recipe == paired_recipe
    config = load_staged_training_config(str(paired_path), arm="R_response")
    assert config.review_updates == (200, 500, 600)
    assert config.max_optimizer_updates == 600
    assert config.total_optimizer_update_ceiling == 600


def test_expanded_probe_evidence_attests_review_schedule_only_recipe_delta(tmp_path: Path) -> None:
    probe_recipe_path = tmp_path / "response_control_native_expanded_interface.json"
    paired_recipe_path = tmp_path / "response_control_native_expanded_interface_review500.json"
    base_recipe = {
        "name": "native_expanded_response_interface",
        "architecture": "Run1804_dense_pairwise_field_native_checkpoint",
        "review_updates": [200, 600],
        "learning_rate": 1.0e-5,
    }
    paired_recipe = {**base_recipe, "review_updates": [200, 500, 600]}
    probe_recipe_path.write_text(json.dumps(base_recipe), encoding="utf-8")
    paired_recipe_path.write_text(json.dumps(paired_recipe), encoding="utf-8")
    checkpoint_path = tmp_path / "Run1804_e4738.pt"
    checkpoint_path.write_bytes(b"checkpoint identity fixture")
    train_paths = []
    train_hashes = []
    family_ids = [f"train-family-{index}" for index in range(8)]
    for index in range(8):
        atlas_path = tmp_path / f"train_{index:04d}_responses.npz"
        metadata_path = atlas_path.with_suffix(".json")
        atlas_path.write_bytes(f"atlas-{index}".encode())
        metadata_path.write_text(f'{{"family_id": "{family_ids[index]}"}}', encoding="utf-8")
        train_paths.append(atlas_path)
        train_hashes.append({
            "sha256": file_sha256(atlas_path),
            "metadata_sha256": file_sha256(metadata_path),
        })
    probe_report = {
        "status": "passed",
        "mode": "expanded_response_fit_capability_probe",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "recipe": str(probe_recipe_path.resolve()),
        "recipe_sha256": file_sha256(probe_recipe_path),
        "native_trainable_scope": {"name": "native_expanded_response_interface"},
        "optimizer_updates_completed": 80,
        "optimizer_updates_attempted": 80,
        "optimizer_calls_charged_to_shared_remedy_ledger": 80,
        "reference_solver_calls": 0,
        "development_stencil_count": 0,
        "capability_gate": {"supports_matched_trial": True},
        "train_atlas_hashes": train_hashes,
        "train_family_ids": family_ids,
    }
    evidence_path = tmp_path / "expanded_probe.json"
    evidence_path.write_text(json.dumps(probe_report), encoding="utf-8")

    evidence = _load_expanded_probe_evidence(
        evidence_path,
        checkpoint_path=checkpoint_path,
        recipe_path=paired_recipe_path,
        train_atlas_paths=train_paths,
        expected_family_ids=family_ids,
    )
    assert evidence["probe_recipe_path"] == str(probe_recipe_path.resolve())
    assert evidence["probe_recipe_sha256"] == file_sha256(probe_recipe_path)
    assert evidence["paired_recipe_path"] == str(paired_recipe_path.resolve())
    assert evidence["paired_recipe_sha256"] == file_sha256(paired_recipe_path)
    assert evidence["recipe_schedule_difference"] == {
        "field": "review_updates",
        "probe": [200, 600],
        "paired": [200, 500, 600],
    }

    probe_report["optimizer_calls_charged_to_shared_remedy_ledger"] = 79
    evidence_path.write_text(json.dumps(probe_report), encoding="utf-8")
    with pytest.raises(ValueError, match="shared-remedy ledger charge must equal its 80 completed attempts"):
        _load_expanded_probe_evidence(
            evidence_path,
            checkpoint_path=checkpoint_path,
            recipe_path=paired_recipe_path,
            train_atlas_paths=train_paths,
            expected_family_ids=family_ids,
        )

    probe_report["optimizer_calls_charged_to_shared_remedy_ledger"] = 80
    evidence_path.write_text(json.dumps(probe_report), encoding="utf-8")
    paired_recipe["learning_rate"] = 5.0e-5
    paired_recipe_path.write_text(json.dumps(paired_recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="may differ from the probe-pinned recipe only in review_updates"):
        _load_expanded_probe_evidence(
            evidence_path,
            checkpoint_path=checkpoint_path,
            recipe_path=paired_recipe_path,
            train_atlas_paths=train_paths,
            expected_family_ids=family_ids,
        )
