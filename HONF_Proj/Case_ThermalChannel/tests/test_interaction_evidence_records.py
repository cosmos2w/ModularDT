from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
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
    attach_role_noise_floors,
    estimate_tightening_response_floor,
    pressure_drop_8pct,
    validate_family_splits,
)
from channelthermal.interaction_evidence.reference_adapter import (
    design_state_to_physical_design,
    load_stored_reference_case,
    operating_context_from_config,
    read_embedded_case_config,
)


def _role(
    role: str,
    values: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    noise_floor: np.ndarray | None = None,
) -> RoleOutput:
    values = np.asarray(values, dtype=np.float64)
    n, channels = values.shape
    if role == "fluid_fields":
        query = np.stack([np.arange(n), np.zeros(n)], axis=-1)
        names = tuple(f"c{i}" for i in range(channels))
        units = tuple("dataset units" for _ in names)
        query_ids = tuple(f"grid:{i}" for i in range(n))
        receiver = None
        kind = "eulerian"
    elif role == "interface":
        query = np.stack([np.linspace(0.0, 2.0 * np.pi, n, endpoint=False), np.ones(n), np.zeros(n)], axis=-1)
        names = tuple(f"i{i}" for i in range(channels))
        units = tuple("dataset units" for _ in names)
        query_ids = tuple(f"m0:port:{i}" for i in range(n))
        receiver = tuple("m0" for _ in range(n))
        kind = "interface_material_angle"
    else:
        query = np.stack([np.linspace(-1.0, 1.0, n), np.zeros(n)], axis=-1)
        names = tuple(f"s{i}" for i in range(channels))
        units = tuple("dataset units" for _ in names)
        query_ids = tuple(f"m0:solid:{i}" for i in range(n))
        receiver = tuple("m0" for _ in range(n))
        kind = "solid_material_normalized_xy"
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    return RoleOutput(
        role=role,
        query_features=query,
        values=values,
        channel_names=names,
        channel_units=units,
        valid_mask=valid,
        quadrature_weights=np.ones(n),
        query_ids=query_ids,
        receiver_module_ids=receiver,
        coordinate_kind=kind,
        noise_floor=noise_floor,
    )


def _record(
    label: str,
    *,
    split: EvidenceSplit = EvidenceSplit.TRAIN,
    source: EvidenceSource = EvidenceSource.REFERENCE_SOLVER,
    fluid: np.ndarray | None = None,
    fluid_valid: np.ndarray | None = None,
    fluid_floor: np.ndarray | None = None,
) -> SolveRecord:
    if fluid is None:
        fluid = np.zeros((3, 1), dtype=np.float64)
    design = DesignState(
        anchor_id="anchor",
        physical_family_id="family",
        split=split,
        modules=(ModuleState("m0", (3.0, 3.0), 1.0),),
    )
    roles = {
        "fluid_fields": _role("fluid_fields", fluid, valid=fluid_valid, noise_floor=fluid_floor),
        "interface": _role("interface", np.zeros((2, 2))),
        "solid_temperature": _role("solid_temperature", np.zeros((2, 1))),
    }
    output = PhysicalSolveOutput(
        roles=roles,
        quantities={"pressure_drop": MeasuredQuantity(1.0, "dataset pressure units")},
        active_module_ids=("m0",),
        module_peak_temperature={"m0": 0.0},
        units_metadata={"temperature": "dataset units"},
    )
    return SolveRecord(
        record_id=label,
        design=design,
        context=OperatingContext({"re": 50.0}),
        source=source,
        status=SolveStatus.CONVERGED,
        elapsed_seconds=0.1,
        provenance={"source": source.value},
        output=output,
    )


def _linear_stencil(*, floors: bool = False) -> ResponseStencil:
    h1, h2 = 0.2, 0.4
    baseline_floor = np.full((3, 1), 100.0) if floors else None
    corner_floor = np.full((3, 1), 2.0) if floors else None
    baseline = _record("base", fluid=np.zeros((3, 1)), fluid_floor=baseline_floor)
    values = {
        "i_plus": np.full((3, 1), 1.0),
        "i_minus": np.full((3, 1), -1.0),
        "j_plus": np.full((3, 1), 2.0),
        "j_minus": np.full((3, 1), -2.0),
        "pp": np.full((3, 1), 3.0 * h1 * h2),
        "pm": np.full((3, 1), -3.0 * h1 * h2),
        "mp": np.full((3, 1), -3.0 * h1 * h2),
        "mm": np.full((3, 1), 3.0 * h1 * h2),
    }
    variants = {name: _record(name, fluid=value, fluid_floor=corner_floor) for name, value in values.items()}
    return ResponseStencil(baseline, variants)


def test_centered_mixed_difference_uses_joint_corners_and_corner_only_noise_floor() -> None:
    stencil = _linear_stencil(floors=True)
    result = stencil.centered_mixed_difference(
        role="fluid_fields",
        joint_pp="pp",
        joint_pm="pm",
        joint_mp="mp",
        joint_mm="mm",
        first_step=0.2,
        second_step=0.4,
    )
    np.testing.assert_allclose(result.delta, 3.0)
    # Four corner floors are 2 each: sqrt(4*2^2)/(4*.2*.4)=12.5.
    # The deliberately large baseline floor must not enter a centered derivative.
    np.testing.assert_allclose(result.noise_floor, 12.5)
    assert not np.any(result.resolved_sign_mask)


def test_common_support_is_stencil_wide_and_unresolved_values_are_not_zero_filled() -> None:
    baseline = _record("base", fluid=np.zeros((3, 1)))
    valid = np.ones((3, 1), dtype=bool)
    valid[2, 0] = False
    variants = {
        "i": _record("i", fluid=np.ones((3, 1)), fluid_valid=valid),
        "j": _record("j", fluid=np.full((3, 1), 2.0)),
        "ij": _record("ij", fluid=np.full((3, 1), 4.0)),
    }
    stencil = ResponseStencil(baseline, variants)
    block = stencil.anchored_interaction(role="fluid_fields", joint_variant="ij", first_variant="i", second_variant="j")
    assert block.observed_mask[:, 0].tolist() == [True, True, False]
    assert np.all(np.isnan(block.delta[~block.observed_mask]))
    assert block.resolved_sign_mask is None


def test_noise_floor_attachment_keeps_observation_and_sign_resolution_separate() -> None:
    base = _record("base", fluid=np.zeros((3, 1)))
    trial = _record("trial", fluid=np.asarray([[1.0], [0.1], [0.0]]))
    base = attach_role_noise_floors(base, {"fluid_fields": np.full((3, 1), 0.2)})
    trial = attach_role_noise_floors(trial, {"fluid_fields": np.full((3, 1), 0.2)})
    block = ResponseStencil(base, {"trial": trial}).finite_change("trial", "fluid_fields")
    assert block.observed_mask[:, 0].all()
    assert block.resolved_sign_mask[:, 0].tolist() == [True, False, False]
    assert block.unresolved_sign_mask[:, 0].tolist() == [False, True, True]


def test_tightening_floor_is_pointwise_and_requires_matching_queries() -> None:
    nominal = ResponseStencil(
        _record("base"),
        {"trial": _record("trial", fluid=np.ones((3, 1)))},
    ).finite_change("trial", "fluid_fields")
    tightened = ResponseStencil(
        _record("base_tight"),
        {"trial": _record("trial_tight", fluid=np.asarray([[1.5], [0.8], [0.0]]))},
    ).finite_change("trial", "fluid_fields")
    estimated = estimate_tightening_response_floor(nominal, tightened)
    np.testing.assert_allclose(estimated.noise_floor[:, 0], [0.5, 0.2, 1.0])
    assert estimated.resolved_sign_mask[:, 0].tolist() == [True, True, False]
    assert estimated.unresolved_sign_mask[:, 0].tolist() == [False, False, True]
    misaligned = replace(tightened, query_features=tightened.query_features + 1.0)
    with pytest.raises(ValueError, match="fixed, aligned queries"):
        estimate_tightening_response_floor(nominal, misaligned)


def test_stencil_rejects_mixed_sources_and_family_split_leakage() -> None:
    baseline = _record("base")
    with pytest.raises(ValueError, match="mix evidence sources"):
        ResponseStencil(baseline, {"teacher": _record("teacher", source=EvidenceSource.SURROGATE_TEACHER)})
    dev = _record("dev", split=EvidenceSplit.DEVELOPMENT)
    with pytest.raises(ValueError, match="cannot cross evidence splits"):
        ResponseStencil(baseline, {"dev": dev})
    with pytest.raises(ValueError, match="crosses splits"):
        validate_family_splits((baseline, dev))


def test_material_receiver_ids_must_stay_fixed_across_stencil() -> None:
    baseline = _record("base")
    trial = _record("trial")
    roles = dict(trial.output.roles)
    roles["interface"] = replace(roles["interface"], receiver_module_ids=("other", "other"))
    output = replace(trial.output, roles=roles)
    trial = replace(trial, output=output)
    with pytest.raises(ValueError, match="material receiver identities changed"):
        ResponseStencil(baseline, {"trial": trial}).common_mask("interface")


def test_pressure_drop_uses_fixed_fluid_8_percent_sections() -> None:
    x = np.asarray([[0.5, 3.0, 9.0, 11.5]])
    p = np.asarray([[4.0, 3.0, 2.0, 1.0]])
    fluid = np.ones_like(p, dtype=bool)
    value, counts = pressure_drop_8pct(p, x, fluid, domain_length_x=12.0)
    assert value == 3.0
    assert counts == {"inlet_count": 1, "outlet_count": 1}


def test_stored_case_adapter_keeps_source_and_outputs_typed() -> None:
    hdf5_path = "/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5"
    import os

    if not os.path.isfile(hdf5_path):
        pytest.skip("The external ThermalChannel HDF5 dataset is not mounted.")
    config, _ = read_embedded_case_config(hdf5_path, "0001")
    context = operating_context_from_config(config)
    assert context.values["re"] == 50.0
    assert context.values["nu"] == pytest.approx(0.018)
    record = load_stored_reference_case(
        hdf5_path,
        "0001",
        physical_family_id="duplicate_family:0001+0273",
        split_override=EvidenceSplit.TRAIN,
    )
    assert record.source is EvidenceSource.STORED_REFERENCE
    assert record.design.active_module_ids == tuple(f"0001:module:{i}" for i in range(3))
    output = record.output
    assert output is not None
    mapped = output.as_mapping()
    assert mapped["fluid_fields"].shape == (8192, 5)
    assert mapped["grid_xy"].shape == (8192, 2)
    assert mapped["interface"].shape == (192, 2)
    assert mapped["solid_temperature"].shape == (9288, 1)
    assert mapped["interface_module_ids"][:64] == ("0001:module:0",) * 64
    assert mapped["active_module_ids"] == record.design.active_module_ids
    assert "dataset pressure units" in output.quantities["pressure_drop"].units


def test_inverse_design_conversion_roundtrips_inputs_through_float32() -> None:
    design = DesignState(
        anchor_id="float32-anchor",
        physical_family_id="float32-family",
        split=EvidenceSplit.TRAIN,
        modules=(ModuleState("m0", (1.234567891, 2.345678912), 1.456789123),),
    )
    physical = design_state_to_physical_design(design)
    np.testing.assert_array_equal(
        np.asarray(physical.module_centers, dtype=np.float32),
        np.asarray([[1.234567891, 2.345678912]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(physical.heat_powers, dtype=np.float32),
        np.asarray([1.456789123], dtype=np.float32),
    )
