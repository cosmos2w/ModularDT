from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
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
from channelthermal.response_control.losses import compute_stencil_loss_terms
from channelthermal.response_control.resume_provenance import (
    file_sha256,
    training_config_mapping,
    validate_paired_resume_provenance,
)
from channelthermal.response_control.runner import (
    _configure_native_output_head_scope,
    _json_default,
    _load_frozen_loss_scales,
    _load_frozen_response_weights,
    _prediction_difference_summary,
    _sampling_summary_mapping,
    _select_geometry_x_probe,
    _validate_review_gate,
)
from channelthermal.response_control.sampling import (
    ReceiverSamplingConfig,
    SamplingSummary,
    sample_training_stencil,
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
    assert _validate_review_gate(100, configured, 3) == 1500
    with pytest.raises(ValueError, match="review gate 2000 is unreachable"):
        _validate_review_gate(2000, configured, 3)

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
    combined = calibrate_operator_weights(
        model, (stencil,), scales=scales, historical_value_source=source,
        parameters=model.parameters(),
    )
    assert combined["value"] == pytest.approx(1.0)
    assert combined["finite"] != pytest.approx(stencil_only["finite"])


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
