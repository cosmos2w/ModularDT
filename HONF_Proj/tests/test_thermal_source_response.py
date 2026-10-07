"""Native physical extraction and actual learned-response dependency checks."""

import copy
import hashlib
import sys

import numpy as np
import pytest
import torch
from channelthermal.dependency_flow import CASE_CAPABILITY, DEPENDENCY_ID, ThermalFlowReader
from channelthermal.source_response import (
    FIELD_ORDER,
    SOURCE_RESPONSE_CAPABILITY,
    SOURCE_RESPONSE_ID,
    SourceResponseThermalModel,
    ThermalSourceResponse,
    load_source_response_model,
)
from channelthermal.source_response_residual import (
    DiscreteThermalBalance,
    module_grid_ids,
    sampled_kernel_residual,
)
from channelthermal.training.checkpoints import atomic_save_checkpoint_payload


def structure():
    return {
        "module_centers": torch.tensor([[[3.1, 2.1], [7.2, 3.2]]]),
        "module_present": torch.ones(1, 2),
        "material_params": torch.tensor([[0.018, 0.01, 0.02, 1.0, 2.0, 0.45]]),
        "re": torch.tensor([[50.0]]),
        "u_in": torch.ones(1, 1),
        "domain_length_x": torch.tensor([[12.0]]),
        "domain_length_y": torch.tensor([[6.0]]),
    }


def thermal(mode="group"):
    torch.manual_seed(3)
    return ThermalSourceResponse(
        {"hidden": 8, "message": 8, "mode": mode}, nx=16, ny=8, environment_nx=3, environment_ny=2
    )


def stats():
    return {
        "field_mean_by_channel": np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32),
        "field_std_by_channel": np.array([2.0, 3.0, 4.0, 5.0, 6.0], np.float32),
    }


def test_frozen_forcing_scale_requires_matching_core_and_adapter_units():
    with pytest.raises(ValueError, match="frozen forcing scales differ"):
        ThermalSourceResponse({"forcing_scale": 2.0})
    matched = ThermalSourceResponse({"hidden": 8, "message": 8, "forcing_scale": 2.0}, forcing_scale=2.0)
    assert matched.core.config["forcing_scale"] == matched.adapter_config()["forcing_scale"] == 2.0


def test_native_cell_center_interpolation_sign_units_and_material_maximum():
    model = thermal()
    s = structure()
    xy = torch.tensor([[[0.375, 0.375], [11.625, 5.625], [3.0, 2.0]]])
    p = model.prepare_native(s, xy, local_query_points=torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]]), ntheta=4)
    # A declared synthetic linear shared grid tests native interpolation, not
    # accuracy or reference generation. It is evaluated at predicted-grid slots.
    coords = torch.stack((p.grid_indices % 16 + 0.5, p.grid_indices // 16 + 0.5), -1) * 0.75
    grid = 2 * coords[..., :1] + 3 * coords[..., 1:2]
    interp = model._interpolate(grid, p.stencils["fluid"])
    torch.testing.assert_close(interp, 2 * xy[..., :1] + 3 * xy[..., 1:2])
    surface = model._interpolate(grid, p.stencils["surface"]).reshape(1, 2, 4, 1)
    outside = model._interpolate(grid, p.stencils["outside"]).reshape(1, 2, 4, 1)
    q = -p.interface_conductivity[:, None, :, None] / p.delta[:, None, :, None] * (outside - surface)
    normal = 2 * p.theta.cos() + 3 * p.theta.sin()
    torch.testing.assert_close(q[0, 0, :, 0], -p.interface_conductivity[0, 0] * normal, atol=2.0e-5, rtol=2.0e-5)
    output = model.apply_native(p, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(output["module_material_peak"], output["pred_internal_temperature"][..., 0].amax(-1))
    assert "pred_port_condition_raw" not in output
    assert output["initial_port_status"].startswith("not_applicable")
    assert "simulate_channelthermal" not in sys.modules


@pytest.mark.parametrize("mode", ["direct", "group"])
def test_actual_affine_response_kernel_ad_and_heat_input_poisoning(mode):
    model = thermal(mode)
    s = structure()
    xy = torch.tensor([[[1.0, 1.0], [4.0, 2.0], [9.0, 4.0]]])

    class Poison:
        def __getattribute__(self, name):
            raise AssertionError("Forbidden current heat/target read")

    s.update(heat_powers=Poison(), steady_field=Poison(), predicted_ports=Poison())
    p = model.prepare_native(s, xy, local_query_points=torch.zeros(1, 2, 2), ntheta=4)
    heat = torch.tensor([[0.5, 1.0]], requires_grad=True)
    d = torch.tensor([[0.25, -0.25]])
    left = model.apply_native(p, heat)
    right = model.apply_native(p, heat + d)
    increment = model.apply_native(p, d, increment=True)
    for key in ("fluid_temperature", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(right[key] - left[key], increment[key], atol=2.0e-6, rtol=2.0e-5)
    zero = model.apply_native(p, torch.zeros_like(heat))
    assert torch.equal(zero["fluid_temperature"], torch.zeros_like(zero["fluid_temperature"]))
    kernels = model.export_native_kernels(p)
    grad = torch.autograd.grad(left["fluid_temperature"].sum(), heat)[0]
    torch.testing.assert_close(grad, kernels["fluid"][..., 0].sum(1), atol=2.0e-6, rtol=2.0e-5)
    # Structural affinity is not a physical-response accuracy assertion.
    twice = model.apply_native(p, 2 * heat)["pred_interface"]
    torch.testing.assert_close(twice, 2 * left["pred_interface"], atol=2.0e-6, rtol=2.0e-5)


def test_composed_actual_flow_null_and_live_geometry_query_gradients():
    t = thermal()
    f = ThermalFlowReader("D-sep", {"hidden": 8, "message": 8})
    model = SourceResponseThermalModel(t, f, stats())
    s = structure()
    s["module_centers"] = s["module_centers"].requires_grad_()
    s["u_in"] = s["u_in"].requires_grad_()
    heat = torch.tensor([[0.8, 1.2]], requires_grad=True)
    xy = torch.tensor([[[1.13, 1.1], [4.1, 2.2], [9.2, 4.1]]], requires_grad=True)
    p = model.prepare_native(s, xy, local_query_points=torch.zeros(1, 2, 2), ntheta=4)
    a = model.apply_native(p, heat)
    b = model.apply_native(p, heat + torch.tensor([[0.1, 0.3]]))
    assert torch.equal(a["pred_field"][..., :4], b["pred_field"][..., :4])
    grad = torch.autograd.grad(
        a["pred_field"][..., :4].square().sum(),
        (heat, s["module_centers"], s["u_in"], xy),
        allow_unused=True,
        retain_graph=True,
    )
    assert grad[0] is None or torch.equal(grad[0], torch.zeros_like(heat))
    # A concatenated thermal channel can make autograd return explicit zeros
    # for a flow-only slice. The actual flow call has no heating dependency.
    standalone = torch.autograd.grad(f(s, xy).sum(), heat, allow_unused=True, retain_graph=True)[0]
    assert standalone is None
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad[1:])
    tempgrad = torch.autograd.grad(a["pred_field"][..., 4].sum(), (heat, s["module_centers"], xy))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in tempgrad)


def test_prepared_stale_geometry_ids_and_receiver_catalog_rejected():
    t = thermal()
    s = structure()
    s["module_source_ids"] = ("a", "b")
    xy = torch.tensor([[[1.0, 1.0], [4.0, 2.0]]])
    p = t.prepare_native(s, xy, ntheta=4)
    s["module_source_ids"] = ("b", "a")
    with pytest.raises(ValueError, match="Geometry/context changed"):
        t.apply_native(p, torch.ones(1, 2))
    model = SourceResponseThermalModel(t, ThermalFlowReader("D-sep", {"hidden": 8, "message": 8}), stats())
    p = model.prepare_native(structure(), xy, ntheta=4)
    xy.add_(0.1)
    with pytest.raises(ValueError, match="Receiver catalogue changed"):
        model.apply_native(p, torch.ones(1, 2))


@pytest.mark.parametrize('mode', ['direct', 'group'])
def test_precise_native_small_increments_and_physical_kernel_vjp(mode):
    model = thermal(mode)
    s = structure()
    p = model.prepare_native(s, torch.tensor([[[1.1, 1.3], [4.1, 2.2]]]),
                             local_query_points=torch.zeros(1, 2, 2), ntheta=4)
    heat = torch.tensor([[0.8000000001, 1.2000000003]], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([[1e-8, -1e-8]], dtype=torch.float64)
    kwargs = {'accumulation_dtype': torch.float64}
    baseline = model.apply_native(p, heat, **kwargs)
    changed = model.apply_native(p, heat + delta, **kwargs)
    increment = model.apply_native(p, delta, increment=True, **kwargs)
    kernels = model.export_native_kernels(p, **kwargs)
    for key in ('fluid_temperature', 'pred_interface', 'pred_internal_temperature'):
        assert increment[key].dtype == torch.float64
        torch.testing.assert_close(changed[key] - baseline[key], increment[key], atol=1e-14, rtol=1e-6)
    for key, kernel in [('fluid_temperature', kernels['fluid']),
                        ('pred_interface', torch.cat((kernels['surface'], kernels['q_normal']), -1))]:
        actual = torch.autograd.grad(baseline[key].sum(), heat, retain_graph=True)[0]
        expected = kernel.sum(dim=tuple(i for i in range(1, kernel.ndim) if i != kernel.ndim - 2))
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    assert 'module_material_peak' not in increment
    assert 'pred_port_condition' not in increment
    s['module_centers'].add_(0.1)
    with pytest.raises(ValueError, match='Geometry/context changed'):
        model.apply_native(p, delta, increment=True, **kwargs)


def test_discrete_balance_zero_constant_adjoint_and_sampled_all_column_parity():
    s = structure()
    nx, ny = 16, 8
    lengths = torch.tensor([[12.0, 6.0]])
    yy, xx = torch.meshgrid((torch.arange(ny) + 0.5) * 0.75, (torch.arange(nx) + 0.5) * 0.75, indexing="ij")
    ids = module_grid_ids(xx, yy, s["module_centers"], s["module_present"], s["material_params"][:, 5])
    balance = DiscreteThermalBalance.from_training_fields(
        torch.ones(1, ny, nx), torch.full((1, ny, nx), -0.1), ids, s["material_params"], lengths, s["module_present"]
    )
    zero = torch.zeros(1, ny, nx)
    assert torch.equal(balance.residual(zero, torch.zeros(1, 2)), zero)
    constant = balance.apply(torch.ones_like(zero))
    torch.testing.assert_close(constant[:, 1:-1, 1:-1], torch.zeros_like(constant[:, 1:-1, 1:-1]), atol=2.0e-7, rtol=0)
    double = DiscreteThermalBalance(balance.coefficients.double(), balance.source_slots, balance.present.double())
    x = torch.randn(1, ny, nx, dtype=torch.float64)
    y = torch.randn_like(x)
    torch.testing.assert_close((double.apply(x) * y).sum(), (x * double.adjoint(y)).sum(), atol=1.0e-10, rtol=1.0e-10)
    model = thermal("direct")
    context = model.prepare_context(s)
    rows = torch.tensor([[0, 17, 35, 60, 127]])
    sampled, receipt = sampled_kernel_residual(model.core, context, balance, rows, lengths)
    xy = torch.stack((xx, yy), -1).reshape(1, -1, 2)
    kernel = model.core.read_kernel(context, xy)[..., 0].reshape(1, ny, nx, 2)
    full = balance.kernel_residual(kernel).reshape(1, -1, 2)
    torch.testing.assert_close(sampled, full[:, rows[0]], atol=2.0e-7, rtol=2.0e-5)
    assert receipt["operator_rows"] == 5 and receipt["neural_stencil_receiver_rows"] <= 25


def test_new_checkpoint_loader_roundtrip_no_thermal_parent_and_old_loader_rejection(tmp_path):
    t = thermal()
    flow = ThermalFlowReader("D-sep", {"hidden": 8, "message": 8})
    manifest = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
    dataset = {
        "development_manifest": "manifest",
        "development_manifest_sha256": manifest,
        "development_subset": {
            "manifest_sha256": manifest,
            "partitions": {
                "train": {"case_ids": [str(i) for i in range(150)]},
                "test": {"case_ids": [str(i) for i in range(150, 172)]},
            },
        },
    }
    identity = {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
    }
    f = dict(
        identity,
        dependency_identity=DEPENDENCY_ID,
        case_capability=CASE_CAPABILITY,
        dependency_policy="D-sep",
        flow_reader_config=flow.reader.config,
        flow_state_dict=flow.state_dict(),
        global_normalization_stats=stats(),
        train_config={"dataset": dataset},
    )
    fp = tmp_path / "flow.pt"
    atomic_save_checkpoint_payload(fp, f)
    child = dict(
        identity,
        source_response_identity=SOURCE_RESPONSE_ID,
        case_capability=SOURCE_RESPONSE_CAPABILITY,
        channel_order=list(FIELD_ORDER),
        source_response_config={"core": t.core_config, "adapter": t.adapter_config()},
        thermal_state_dict=t.state_dict(),
        flow_checkpoint=str(fp),
        flow_checkpoint_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),
        global_normalization_stats=stats(),
        train_config={"dataset": dataset},
    )
    cp = tmp_path / "response.pt"
    atomic_save_checkpoint_payload(cp, child)
    restored, _ = load_source_response_model(cp)
    assert all(torch.equal(value, restored.thermal.state_dict()[key]) for key, value in t.state_dict().items())
    assert not hasattr(restored, "local_coupling")
    from channelthermal.evaluation.loading import load_model

    with pytest.raises(ValueError, match="source_response"):
        load_model(cp, torch.device("cpu"))
    bad = copy.deepcopy(child)
    bad["global_normalization_stats"]["field_std_by_channel"][0] += 1
    with pytest.raises(ValueError, match="normalization"):
        load_source_response_model(cp, checkpoint=bad)
    bad = copy.deepcopy(child)
    bad["global_normalization_stats"].pop("field_std_by_channel")
    with pytest.raises(ValueError, match="normalization key sets"):
        load_source_response_model(cp, checkpoint=bad)
    bad = copy.deepcopy(child)
    bad["train_config"]["dataset"].pop("development_manifest_sha256")
    with pytest.raises(ValueError, match="literal fixed25_v1"):
        load_source_response_model(cp, checkpoint=bad)
    sealed = copy.deepcopy(child)
    sealed["fit_identity"] = {
        "mode": "group",
        "recipe": {
            "core_configs": {"group": copy.deepcopy(t.core_config)},
            "adapter_config": t.adapter_config(),
        },
    }
    load_source_response_model(cp, checkpoint=sealed)
    for drift in ("core_scale", "adapter_scale", "mode"):
        bad = copy.deepcopy(sealed)
        if drift == "mode":
            bad["fit_identity"]["mode"] = "direct"
        else:
            key = "core" if drift == "core_scale" else "adapter"
            bad["source_response_config"][key]["forcing_scale"] = 2.0
        with pytest.raises(ValueError, match="sealed fit identity recipe"):
            load_source_response_model(cp, checkpoint=bad)
