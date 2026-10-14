"""CPU contracts for the native Thermal content-hypergraph adapter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.content_hypergraph import ThermalContentHypergraphAdapter
from channelthermal.flow_curl import physical_uvp_native_curl
from channelthermal.training.content_hypergraph_task import (
    ThermalContentHypergraphTask,
    _endpoint_material_maximum_rows,
)


def _stats() -> dict[str, np.ndarray]:
    return {
        "field_mean_by_channel": np.asarray([1.0, -2.0, 0.5, 3.0, 10.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "interface_targets_mean": np.asarray([10.0, 0.0], dtype=np.float32),
        "interface_targets_std": np.asarray([4.0, 5.0], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.0], dtype=np.float32),
    }


def _model() -> ThermalContentHypergraphAdapter:
    torch.manual_seed(11)
    return ThermalContentHypergraphAdapter(
        _stats(),
        nx=8,
        ny=6,
        core_config={
            "architecture": "A-H6",
            "context_dim": 32,
            "num_heads": 4,
            "context_depth": 1,
            "num_hyperedges": 6,
            "thermal_response_rank": 4,
            "local_hidden_dim": 16,
            "init_seed": 7,
        },
    )


def _structure(*, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    centers = torch.tensor([[[2.0, 2.0], [7.5, 4.0]]])
    if requires_grad:
        centers.requires_grad_()
    return {
        "module_centers": centers,
        "module_present": torch.ones(1, 2),
        "module_source_ids": torch.tensor([[31, 47]], dtype=torch.long),
        "material_params": torch.tensor([[0.02, 0.01, 0.02, 1.0, 2.0, 0.22]]),
        "re": torch.tensor([[50.0]]),
        "u_in": torch.ones(1, 1),
        "domain_length_x": torch.tensor([[12.0]]),
        "domain_length_y": torch.tensor([[6.0]]),
    }


def _receivers() -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        (torch.arange(6, dtype=torch.float32) + 0.5),
        (torch.arange(8, dtype=torch.float32) + 0.5) * 1.5,
        indexing="ij",
    )
    fluid_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), -1)[None]
    local = torch.tensor(
        [[[[0.0, 0.0], [0.35, 0.0]], [[0.0, 0.0], [0.35, 0.0]]]],
    )
    return fluid_xy, local


def _prepared(model: ThermalContentHypergraphAdapter, *, requires_grad: bool = False):
    fluid, local = _receivers()
    return model.prepare_native(
        _structure(requires_grad=requires_grad), fluid, local_query_points=local,
        ntheta=16, scene_key=("synthetic",)
    )


def test_native_union_keeps_logical_roles_and_physical_curl_order():
    model = _model().eval()
    prepared = _prepared(model)
    assert prepared.role_shapes == {
        "fluid": (48,), "surface": (2, 16), "outside": (2, 16), "material": (2, 2),
    }
    assert prepared.scene_batch.environment_features.shape == (1, 192, 8)
    assert prepared.scene_batch.module_ids.tolist() == [[31, 47]]
    assert prepared.query_batch.local_weights.ndim == 1
    assert prepared.query_batch.local_weights.numel() == 2

    output = model.apply_native(prepared, torch.tensor([[0.4, 0.8]]))
    raw = model.core.decode(
        prepared.organization, prepared.query_batch, controls=torch.tensor([[0.4, 0.8]])
    )
    expected_flow = physical_uvp_native_curl(
        raw.nonlinear_field,
        prepared.curl_stencil,
        model.field_mean,
        model.field_std,
    )
    torch.testing.assert_close(output["pred_field"][..., :4], expected_flow, rtol=0, atol=0)
    assert output["pred_interface"].shape == (1, 2, 16, 2)
    assert output["pred_internal_temperature"].shape == (1, 2, 2, 1)


def test_surface_receiver_count_is_explicit_for_monitor_and_native_endpoints():
    model = _model().eval()
    fluid, local = _receivers()
    monitoring = model.prepare_native(_structure(), fluid, local_query_points=local, ntheta=16)
    endpoint = model.prepare_native(_structure(), fluid, local_query_points=local)
    assert monitoring.ntheta == 16
    assert monitoring.role_shapes["surface"] == (2, 16)
    assert endpoint.ntheta == 64
    assert endpoint.role_shapes["surface"] == (2, 64)
    assert endpoint.role_shapes["outside"] == (2, 64)


def test_native_endpoint_batch_rebuilds_full_grid_instead_of_reusing_sampled_queries():
    task = object.__new__(ThermalContentHypergraphTask)
    task.device = torch.device("cpu")
    task.adapter = SimpleNamespace(nx=128, ny=64)
    x_grid = (np.arange(128, dtype=np.float32) + 0.5) * (12.0 / 128.0)
    y_grid = (np.arange(64, dtype=np.float32) + 0.5) * (6.0 / 64.0)
    case = {
        "case_id": "synthetic-native",
        "x_grid": x_grid,
        "y_grid": y_grid,
        "steady_field": np.zeros((64, 128, 5), dtype=np.float32),
        "module_mask": np.zeros((64, 128), dtype=np.uint8),
        "structure": {
            "re": np.asarray([50.0], dtype=np.float32),
            "u_in": np.asarray([1.0], dtype=np.float32),
            "module_centers": np.asarray([[6.0, 3.0]], dtype=np.float32),
            "heat_powers": np.asarray([0.0], dtype=np.float32),
            "module_present": np.asarray([1.0], dtype=np.float32),
            "material_params": np.asarray([0.02, 0.01, 0.02, 1.0, 2.0, 0.2], dtype=np.float32),
            "domain_length_x": np.asarray([12.0], dtype=np.float32),
            "domain_length_y": np.asarray([6.0], dtype=np.float32),
        },
        # Native endpoint evaluation does not use this sampled training set.
        "query_xy": np.zeros((17, 2), dtype=np.float32),
        "interface_target": np.zeros((1, 64, 2), dtype=np.float32),
        "interface_condition_valid_mask": np.ones((1, 64), dtype=np.float32),
        "module_internal_query_points": np.zeros((2, 2), dtype=np.float32),
        "module_internal_temperature_points": np.zeros((1, 2), dtype=np.float32),
    }
    batch = task._native_case_batch([case], (0,), split="train", ntheta=64)
    assert batch.receivers.ntheta == 64
    assert batch.receivers.fluid_xy.shape == (1, 8192, 2)
    assert batch.targets.field_targets.shape == (1, 8192, 5)
    assert batch.targets.interface_target.shape == (1, 1, 64, 2)


def test_heat_is_separate_zero_intercept_and_factorized_native_jvp_vjp():
    model = _model().eval()
    prepared = _prepared(model)
    heat = torch.tensor([[0.4, 0.8]])
    zero = model.apply_native(prepared, torch.zeros_like(heat))
    heated = model.apply_native(prepared, heat)
    assert torch.count_nonzero(zero["pred_field"][..., 4]) == 0
    assert torch.equal(heated["pred_field"][..., :4], zero["pred_field"][..., :4])

    increment = model.apply_increment(prepared, heat, accumulation_dtype=torch.float64)
    jvp = model.heat_jvp(prepared, heat, accumulation_dtype=torch.float64)
    torch.testing.assert_close(increment["fluid_temperature"], jvp["fluid_temperature"], rtol=0, atol=0)
    torch.testing.assert_close(
        increment["fluid_temperature"],
        heated["fluid_temperature"].to(torch.float64),
        rtol=2e-6,
        atol=2e-6,
    )
    cotangent = torch.linspace(-1.0, 1.0, 48).reshape(1, 48)
    vjp = model.heat_vjp(prepared, cotangent, accumulation_dtype=torch.float64)
    lhs = (jvp["fluid_temperature"][..., 0] * cotangent).sum()
    rhs = (heat * vjp).sum()
    torch.testing.assert_close(lhs, rhs, rtol=2e-8, atol=2e-8)


def test_prepared_native_state_rejects_transform_and_metadata_mutations():
    model = _model().eval()
    prepared = _prepared(model)
    with torch.no_grad():
        model.field_std[0].add_(0.1)
    with pytest.raises(ValueError, match="physical transforms"):
        model.apply_native(prepared, torch.ones(1, 2))

    model = _model().eval()
    prepared = _prepared(model)
    model.normalization_stats["field_mean_by_channel"][0] += 0.25
    with pytest.raises(ValueError, match="normalization metadata"):
        model.apply_native(prepared, torch.ones(1, 2))


def test_prepared_native_rejects_moved_geometry_but_rebuilds_ids_and_masks():
    model = _model().eval()
    structure = _structure()
    fluid, local = _receivers()
    prepared = model.prepare_native(structure, fluid, local_query_points=local, ntheta=16)
    moved = copy.deepcopy(structure)
    moved["module_centers"][0, 0] = torch.tensor([2.25, 1.5])
    with pytest.raises(ValueError, match="geometry or prescribed context moved"):
        model.validate_prepared(prepared, structure=moved)

    rebuilt = model.prepare_native(moved, fluid, local_query_points=local, ntheta=16)
    assert rebuilt.scene_batch.module_ids.tolist() == [[31, 47]]
    assert not torch.equal(prepared.native_solid_mask, rebuilt.native_solid_mask)


def test_checkpoint_adapter_payload_reconstructs_without_dataset_access():
    model = _model().eval()
    prepared = _prepared(model)
    heat = torch.tensor([[0.4, 0.8]])
    expected = model.apply_native(prepared, heat)
    restored = ThermalContentHypergraphAdapter.from_state_payload(
        model.state_payload(), state_dict=model.state_dict(), device="cpu"
    ).eval()
    fluid, local = _receivers()
    actual = restored.apply_native(
        restored.prepare_native(
            _structure(), fluid, local_query_points=local, ntheta=16, scene_key=("synthetic",)
        ), heat
    )
    torch.testing.assert_close(actual["pred_field"], expected["pred_field"], rtol=0, atol=0)
    torch.testing.assert_close(actual["pred_interface"], expected["pred_interface"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["pred_internal_temperature"], expected["pred_internal_temperature"], rtol=0, atol=0
    )


def test_adapter_canonicalizes_hdf5_train_fitter_interface_stat_names():
    stats = _stats()
    stats["interface_target_mean"] = stats.pop("interface_targets_mean")
    stats["interface_target_std"] = stats.pop("interface_targets_std")
    model = ThermalContentHypergraphAdapter(
        stats,
        nx=8,
        ny=6,
        core_config=_model().core.get_config(),
    )
    assert model.normalization_stats["interface_targets_mean"] == [10.0, 0.0]
    assert model.normalization_stats["interface_targets_std"] == [4.0, 5.0]


def test_complete_geometry_vjp_matches_two_centered_differences():
    model = _model().double().eval()
    fluid_xy, local = _receivers()
    fluid_xy, local = fluid_xy.double(), local.double()
    heat = torch.tensor([[0.4, 0.8]], dtype=torch.float64)
    base = _structure()
    base = {name: value.double() for name, value in base.items()}
    base["module_centers"].requires_grad_()

    prepared = model.prepare_native(base, fluid_xy, local_query_points=local, ntheta=16)
    assert not prepared.fluid_xy.requires_grad
    assert bool(prepared.stencils["fluid"].valid.all())
    assert bool(prepared.stencils["material"].valid.all())
    assert bool(prepared.stencils["surface"].valid.all())
    assert bool(prepared.stencils["outside"].valid.all())
    assert prepared.stencils["material"].weights.requires_grad
    assert prepared.query_batch.local_weights.requires_grad
    output = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)
    objective = (
        output["pred_field"][..., 4].square().mean()
        + 0.03 * output["pred_field"][..., 3].sin().mean()
        + 0.1 * output["pred_interface"][..., 0].square().mean()
        + 0.001 * output["pred_interface"][..., 1].square().mean()
        + 0.05 * output["pred_internal_temperature"].square().mean()
    )
    gradient = torch.autograd.grad(objective, base["module_centers"])[0][0, 0, 0].item()

    def value_at(offset: float) -> float:
        structure = {name: value.detach().clone() for name, value in base.items()}
        structure["module_centers"][0, 0, 0] += offset
        native = model.prepare_native(structure, fluid_xy, local_query_points=local, ntheta=16)
        pred = model.apply_native(native, heat, accumulation_dtype=torch.float64)
        return float((
            pred["pred_field"][..., 4].square().mean()
            + 0.03 * pred["pred_field"][..., 3].sin().mean()
            + 0.1 * pred["pred_interface"][..., 0].square().mean()
            + 0.001 * pred["pred_interface"][..., 1].square().mean()
            + 0.05 * pred["pred_internal_temperature"].square().mean()
        ).item())

    step = 2.0e-5
    central_one = (value_at(step) - value_at(-step)) / (2 * step)
    central_two = (value_at(2 * step) - value_at(-2 * step)) / (4 * step)
    assert abs(gradient) > 1.0e-8
    assert gradient == pytest.approx(central_one, rel=1.0e-4, abs=1.0e-5)
    assert gradient == pytest.approx(central_two, rel=1.0e-4, abs=1.0e-5)


@pytest.mark.parametrize(
    ("protocol", "total_epochs", "hold_through_epoch"),
    (("development", 2500, 1000), ("full_train1000", 5000, 2000)),
)
def test_protocol_specific_absolute_lr_hold(protocol, total_epochs, hold_through_epoch):
    task = object.__new__(ThermalContentHypergraphTask)
    task.protocol = protocol
    task.total_epochs = total_epochs
    assert task._schedule_payload()["hold_through_epoch"] == hold_through_epoch


def test_endpoint_material_maximum_metrics_and_module_count_summaries():
    task = object.__new__(ThermalContentHypergraphTask)
    task.validation_scope = "fixed test scope"
    task.validation_case_ids = ("case-m3", "case-m5")
    prediction = np.asarray([[1.0, 4.0], [2.0, 3.0]])
    target = np.asarray([[2.0, 4.0], [1.0, 5.0]])
    valid = np.ones_like(target, dtype=bool)
    module_rows = _endpoint_material_maximum_rows(prediction, target, valid)
    assert [row["error"] for row in module_rows] == [0.0, -2.0]

    def row(case_id, module_count, errors):
        return {
            "case_id": case_id,
            "module_count": module_count,
            "field_score": 1.0,
            "flow_group_standardized_mse": 1.0,
            "thermal_group_standardized_mse": 1.0,
            "flow_rmse": {name: 1.0 for name in ("u", "v", "p", "omega")},
            "roles": {
                name: {"rmse": 1.0} for name in (
                    "temperature_fluid", "temperature_surface", "temperature_material"
                )
            } | {
                "temperature_material_endpoint_maxima": {
                    "per_module": [
                        {"module_index": index, "predicted_max": error, "target_max": 0.0, "error": error}
                        for index, error in enumerate(errors)
                    ]
                }
            },
            "q_proxy_rmse": 1.0,
        }

    reduced = task.reduce_native_metrics(
        [{"case_rows": [row("case-m3", 3, [1.0, -2.0, 3.0]), row("case-m5", 5, [4.0, -1.0])]}]
    )
    endpoint = reduced["temperature_material_endpoint_maxima"]
    assert endpoint["module_count"] == 5
    assert endpoint["absolute_error_mean"] == pytest.approx(2.2)
    assert endpoint["absolute_error_p90"] == pytest.approx(3.6)
    assert endpoint["absolute_error_p95"] == pytest.approx(3.8)
    assert endpoint["absolute_error_worst"] == pytest.approx(4.0)
    assert endpoint["by_module_count"]["M3"]["absolute_error_mean"] == pytest.approx(2.0)
    assert endpoint["by_module_count"]["M5"]["absolute_error_worst"] == pytest.approx(4.0)
    assert endpoint["by_module_count"]["M7"]["case_count"] == 0
    assert endpoint["by_module_count"]["M10"]["absolute_error_mean"] is None
