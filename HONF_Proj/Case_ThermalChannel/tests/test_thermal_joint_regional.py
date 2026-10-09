"""Focused CPU contracts for the fresh joint Thermal regional adapter/provider."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.joint_regional import (
    JOINT_THERMAL_CHANNEL_ORDER,
    JOINT_THERMAL_MODES,
    JointThermalRegionalAdapter,
    load_joint_thermal_checkpoint,
)
from channelthermal.training.joint_task import JointThermalTask


def _stats():
    return {
        "field_mean_by_channel": np.asarray([1.0, -2.0, 0.5, 3.0, 10.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "interface_targets_mean": np.asarray([10.0, 0.0], dtype=np.float32),
        "interface_targets_std": np.asarray([4.0, 5.0], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.0], dtype=np.float32),
    }


def _structure(module_count: int):
    centers = []
    for index in range(module_count):
        centers.append([1.0 + 1.75 * (index % 6), 1.0 + 3.7 * (index // 6)])
    return {
        "module_centers": torch.tensor([centers], dtype=torch.float32),
        "module_present": torch.ones(1, module_count, dtype=torch.float32),
        "material_params": torch.tensor([[0.02, 0.01, 0.02, 1.0, 2.0, 0.22]], dtype=torch.float32),
        "re": torch.tensor([[50.0]], dtype=torch.float32),
        "u_in": torch.ones(1, 1, dtype=torch.float32),
        "domain_length_x": torch.tensor([[12.0]], dtype=torch.float32),
        "domain_length_y": torch.tensor([[6.0]], dtype=torch.float32),
        "module_source_ids": tuple(f"physical-{index}" for index in range(module_count)),
    }


def _adapter(mode="J-H"):
    torch.manual_seed(91)
    return JointThermalRegionalAdapter(
        mode=mode,
        normalization_stats=_stats(),
        hidden=12,
        message=12,
        regional_anchors=6,
        depth=2,
        receiver_tile=3,
        nx=8,
        ny=4,
        environment_nx=4,
        environment_ny=2,
        seed=91,
    )


def _prepare(model, module_count):
    structure = _structure(module_count)
    xy = torch.tensor([[[0.75, 0.75], [3.75, 2.25], [8.25, 3.75]]], dtype=torch.float32)
    local = torch.tensor([[[[0.0, 0.0], [0.35, 0.0]], [[0.0, 0.0], [0.35, 0.0]]]], dtype=torch.float32)
    if module_count != 2:
        local = torch.zeros(1, module_count, 2, 2, dtype=torch.float32)
    prepared = model.prepare_native(structure, xy, local_query_points=local, ntheta=4, chunk_size=3)
    return structure, xy, local, prepared


class _ForbiddenTarget:
    def __getattribute__(self, name):
        raise AssertionError(f"forbidden target/input access: {name}")


@pytest.mark.parametrize("mode", JOINT_THERMAL_MODES)
def test_joint_native_heat_null_affinity_and_target_poisoning(mode):
    model = _adapter(mode)
    structure, xy, local, _ = _prepare(model, 2)
    structure.update(
        heat_powers=_ForbiddenTarget(),
        steady_field=_ForbiddenTarget(),
        predicted_ports=_ForbiddenTarget(),
        observed_temperature=_ForbiddenTarget(),
    )
    prepared = model.prepare_native(structure, xy, local_query_points=local, ntheta=4, chunk_size=3)
    assert prepared.context.dependency.output_law == "affine"
    assert prepared.context.dependency.output_roles == ("temperature",)
    assert prepared.context.model_output_laws == {"flow": "nonlinear", "temperature_response": "affine"}

    heat = torch.tensor([[0.7, 1.1]], dtype=torch.float64)
    delta = torch.tensor([[0.125, -0.25]], dtype=torch.float64)
    baseline = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)
    changed = model.apply_native(prepared, heat + delta, accumulation_dtype=torch.float64)
    increment = model.apply_native(
        prepared,
        delta,
        increment=True,
        accumulation_dtype=torch.float64,
    )
    assert torch.equal(baseline["pred_field"][..., :4], changed["pred_field"][..., :4])
    assert torch.equal(increment["pred_field"][..., :4], torch.zeros_like(increment["pred_field"][..., :4]))
    torch.testing.assert_close(
        changed["fluid_temperature"] - baseline["fluid_temperature"],
        increment["fluid_temperature"],
        atol=1e-13,
        rtol=1e-8,
    )
    torch.testing.assert_close(
        changed["pred_interface"] - baseline["pred_interface"],
        increment["pred_interface"],
        atol=1e-13,
        rtol=1e-8,
    )
    assert increment["fluid_temperature"].dtype == torch.float64
    assert "module_material_peak" not in increment
    assert "pred_port_condition" not in increment


@pytest.mark.parametrize("module_count", [1, 12])
def test_low_and_high_source_count_both_heads_reach_shared_hypergraph_parameters(module_count):
    model = _adapter("J-H")
    _, _, _, prepared = _prepare(model, module_count)
    heat = torch.linspace(0.2, 1.0, module_count)[None]
    output = model.apply_native(prepared, heat)
    response = model.apply_native(prepared, torch.full_like(heat, 0.17), increment=True)
    flow_loss = output["pred_field"][..., :4].square().mean()
    thermal_loss = (
        response["fluid_temperature"].square().mean()
        + response["pred_interface"].square().mean()
        + response["pred_internal_temperature"].square().mean()
    )
    parameters = dict(model.named_parameters())
    shared_names = [
        "core.source_encoder.0.weight",
        "core.source_read.0.weight",
        "core.edge_updates.0.0.weight",
        "core.receiver_query.0.weight",
    ]
    # With one physical source, a normalized source-incidence distribution has
    # no source competition and its membership-score derivative is exactly 0.
    # The high-M panel tests that learned incidence is live in both heads.
    if module_count == 12:
        shared_names.append("core.source_membership_score.0.weight")
    for loss in (flow_loss, thermal_loss):
        gradients = torch.autograd.grad(loss, tuple(parameters[name] for name in shared_names), retain_graph=True)
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


def test_joint_checkpoint_roundtrip_is_self_contained_and_stale_context_is_rejected():
    model = _adapter("J-geometry")
    _, _, _, prepared = _prepare(model, 3)
    before = model.apply_native(prepared, torch.tensor([[0.5, 0.8, 1.1]]))["pred_field"].detach()
    payload = model.checkpoint_payload(provider_identity={"scope": "focused_cpu_test"})
    assert payload["channel_order"] == list(JOINT_THERMAL_CHANNEL_ORDER)
    assert all(name.startswith("core.") for name in payload["model_state_dict"])
    assert not any(name.startswith(("parent.", "flow.", "thermal.")) for name in payload["model_state_dict"])
    restored = load_joint_thermal_checkpoint(payload)
    _, _, _, restored_prepared = _prepare(restored, 3)
    after = restored.apply_native(restored_prepared, torch.tensor([[0.5, 0.8, 1.1]]))["pred_field"].detach()
    torch.testing.assert_close(after, before, atol=0, rtol=0)

    with torch.no_grad():
        next(model.parameters()).add_(1.0e-3)
    with pytest.raises(ValueError, match="weights changed"):
        model.apply_native(prepared, torch.ones(1, 3))


def test_native_record_increment_returns_flow_null_and_precise_temperature_roles():
    model = _adapter("J-H")
    module = SimpleNamespace(
        module_id="module-0",
        position_xy=(3.0, 2.0),
        active=True,
        heating=0.75,
    )
    roles = {
        "fluid_fields": SimpleNamespace(
            query_features=np.asarray([[0.75, 0.75], [3.75, 2.25], [8.25, 3.75]], dtype=np.float32)
        ),
        "interface": SimpleNamespace(
            receiver_module_ids=np.asarray(["module-0"] * 4),
            query_features=np.stack((np.arange(4) * (2 * np.pi / 4), np.zeros(4)), axis=-1).astype(np.float32),
        ),
        "solid_temperature": SimpleNamespace(
            receiver_module_ids=np.asarray(["module-0"] * 2),
            query_features=np.asarray([[0.0, 0.0], [0.35, 0.0]], dtype=np.float32),
        ),
    }
    record = SimpleNamespace(
        design=SimpleNamespace(modules=[module]),
        context=SimpleNamespace(values={
            "nu": 0.02,
            "solid_alpha": 0.01,
            "fluid_alpha": 0.02,
            "solid_k": 1.0,
            "fluid_k": 2.0,
            "module_radius": 0.22,
            "re": 50.0,
            "u_in": 1.0,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
        }),
        output=SimpleNamespace(roles=roles),
    )
    prepared = model.prepare_record(record)
    base = model.apply_record(prepared, accumulation_dtype=torch.float64)
    delta = torch.tensor([0.125], dtype=torch.float64)
    plus = model.apply_record(
        prepared,
        torch.tensor([0.875], dtype=torch.float64),
        accumulation_dtype=torch.float64,
    )
    increment = model.apply_record_increment(prepared, delta, accumulation_dtype=torch.float64)
    assert torch.equal(increment["fluid_fields"][:, :4], torch.zeros_like(increment["fluid_fields"][:, :4]))
    torch.testing.assert_close(
        plus["fluid_fields"] - base["fluid_fields"], increment["fluid_fields"], atol=1e-13, rtol=1e-8
    )
    torch.testing.assert_close(plus["interface"] - base["interface"], increment["interface"], atol=1e-13, rtol=1e-8)
    torch.testing.assert_close(
        plus["solid_temperature"] - base["solid_temperature"],
        increment["solid_temperature"],
        atol=1e-13,
        rtol=1e-8,
    )


def test_joint_provider_has_one_all_parameter_group_and_balanced_native_loss_families():
    model = _adapter("J-H")
    provider = object.__new__(JointThermalTask)
    provider.model = model
    provider.joint_mode = "J-H"
    provider.total_epochs = 2500
    provider.flow_role_weights = {role: 0.5 / 4 for role in ("u", "v", "p", "omega")}
    provider.thermal_role_weights = {role: 0.5 / 3 for role in ("fluid", "surface", "material")}
    provider.q_proxy_weight = 0.05
    provider.response_weight = 0.1
    provider.operator_weight = 0.2
    provider.use_operator = True
    provider._joint_loss_metadata = provider._build_loss_metadata()

    first = provider.optimizer_groups(model, "J-H", "joint")
    second = provider.optimizer_groups(model, "J-H", "warmup")
    assert first == second
    assert len(first) == 1
    assert set(first[0].parameter_names) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert all(name.startswith("core.") for name in first[0].parameter_names)
    assert first[0].schedule.total_epochs == 2500
    assert first[0].schedule.peak_lr == pytest.approx(3.0e-4)
    assert first[0].schedule.final_lr == pytest.approx(3.0e-6)

    values = {
        **{f"flow/{role}": torch.tensor([1.0, 2.0]) for role in ("u", "v", "p", "omega")},
        **{f"temperature/{role}": torch.tensor([3.0, 4.0]) for role in ("fluid", "surface", "material")},
        "q_proxy": torch.tensor([5.0, 6.0]),
    }
    values["flow_group"] = torch.stack([values[f"flow/{role}"] for role in ("u", "v", "p", "omega")]).mean(0)
    values["thermal_group"] = torch.stack(
        [values[f"temperature/{role}"] for role in ("fluid", "surface", "material")]
    ).mean(0)
    provider._native_role_losses = lambda *args: values
    provider._operator_loss = lambda *args: (
        torch.tensor([7.0, 8.0]),
        {"operator_rows": 256, "unique_stencil_receiver_rows": 64},
    )
    predictions = SimpleNamespace(
        model=model,
        execution_mode="J-H",
        native_main={},
        native_prepared=None,
        work={},
    )
    targets = SimpleNamespace(case_ids=("a", "b"), case_indices=(0, 1), response=None)
    terms = provider.loss_terms(predictions, targets, "joint", {})
    assert sum(terms[f"flow/{role}"].weight for role in ("u", "v", "p", "omega")) == pytest.approx(0.5)
    assert sum(terms[f"temperature/{role}"].weight for role in ("fluid", "surface", "material")) == pytest.approx(0.5)
    assert terms["q_proxy"].weight == pytest.approx(0.05)
    assert terms["operator_residual"].weight == pytest.approx(0.2)
    metadata = provider.loss_metadata()
    assert metadata["q_proxy"]["weight"] == "0.05 outside the equal flow/thermal group balance"
    assert metadata["flow/u"]["panel"] == "native_prediction"
    assert metadata["operator_residual"]["panel"] == "physics"


def test_native_metrics_report_near_far_query_counts_and_sampled_module_peaks():
    model = _adapter("J-H")
    _, _, _, prepared = _prepare(model, 3)
    output = model.apply_native(prepared, torch.tensor([[0.7, 0.8, 0.9]]))
    provider = object.__new__(JointThermalTask)
    provider.stats = _stats()
    provider.formal_full = False
    provider._validation_ids = ("synthetic-train-preflight",)
    predictions = SimpleNamespace(native_prepared=prepared, native_main=output)
    targets = SimpleNamespace(
        case_ids=("synthetic-train-preflight",),
        field_targets=torch.zeros_like(output["pred_field"]),
        point_weights=torch.ones(output["pred_field"].shape[:2]),
        interface_target=torch.zeros_like(output["pred_interface"]),
        material_targets=torch.zeros_like(output["pred_internal_temperature"][..., 0]),
    )
    record = provider.validation_metrics(predictions, targets, {})
    row = record["case_rows"][0]
    assert row["near_fluid_temperature_query_count"] > 0
    assert row["far_fluid_temperature_query_count"] > 0
    assert row["near_fluid_temperature_query_count"] + row["far_fluid_temperature_query_count"] == 3
    assert len(row["material_peak_abs_error_by_source"]) == 3
    assert np.isfinite(row["sampled_material_peak_rmse"])
    reduced = provider.reduce_native_metrics([record])
    assert reduced["near_fluid_temperature_query_count"] == row["near_fluid_temperature_query_count"]
    assert reduced["far_fluid_temperature_query_count"] == row["far_fluid_temperature_query_count"]
    assert reduced["sampled_material_peak_rmse_mean"] == pytest.approx(row["sampled_material_peak_rmse"])
