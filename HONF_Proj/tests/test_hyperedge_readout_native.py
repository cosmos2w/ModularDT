"""Native Thermal transforms and source-affine laws with the Round6 context."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from channelthermal.content_hypergraph import ThermalInheritedAttentionAdapterV1

from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1
from honf_forward_core.inherited_attention.readout_model import HyperedgeReadoutCoreV1


def fixture(reduction):
    config = {"module_input_dim": 8, "environment_input_dim": 8, "coordinate_dim": 2,
              "operating_input_dim": 14, "output_dim": 3, "thermal_response": True,
              "moment_variant": "H", "context_depth": 1, "collect_exact_work_counts": False}
    parent = MomentHyperedgeCoreV1(**config)
    child = HyperedgeReadoutCoreV1(**config, receiver_reduction=reduction,
                                 receiver_access="local_cover", geometry_mode="scene_frame_normalized")
    child.load_state_dict({**child.state_dict(), **parent.state_dict()}, strict=True)
    stats = {
        "field_mean_by_channel": np.asarray([1., -2., .5, 3., 10.], dtype=np.float32),
        "field_std_by_channel": np.asarray([2., 3., 4., 5., 6.], dtype=np.float32),
        "interface_targets_mean": np.asarray([10., 0.], dtype=np.float32),
        "interface_targets_std": np.asarray([4., 5.], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.], dtype=np.float32),
    }
    adapters = [ThermalInheritedAttentionAdapterV1(stats, core=core, nx=8, ny=6)
                for core in (parent, child)]
    structure = {
        "module_centers": torch.tensor([[[2., 2.], [7.5, 4.]]]),
        "module_present": torch.ones(1, 2), "module_source_ids": torch.tensor([[31, 47]]),
        "material_params": torch.tensor([[.02, .01, .02, 1., 2., .22]]),
        "re": torch.tensor([[50.]]), "u_in": torch.ones(1, 1),
        "domain_length_x": torch.tensor([[12.]]), "domain_length_y": torch.tensor([[6.]]),
    }
    yy, xx = torch.meshgrid(torch.arange(6) + .5, (torch.arange(8) + .5) * 1.5, indexing="ij")
    fluid = torch.stack((xx.flatten(), yy.flatten()), dim=-1)[None]
    local = torch.tensor([[[[0., 0.], [.35, 0.]], [[0., 0.], [.35, 0.]]]])
    return adapters, structure, fluid, local


@pytest.mark.parametrize("reduction", ["pool_first", "edge_first"])
def test_native_zero_output_parent_recovery_and_live_input_derivative(reduction):
    adapters, structure, fluid, local = fixture(reduction)
    heat = torch.tensor([[1., -2.]])
    records = []
    for adapter in adapters:
        # Freezing weights must retain the requested input-coordinate graph.
        adapter.requires_grad_(False)
        points = fluid.clone().requires_grad_()
        prepared = adapter.prepare_native(structure, points, local_query_points=local, ntheta=16)
        output = adapter.apply_native(prepared, heat)
        objective = output["fluid_temperature"].square().sum()
        gradient, = torch.autograd.grad(objective, points, create_graph=True)
        curvature, = torch.autograd.grad(gradient.square().sum(), points)
        records.append((output, gradient, curvature))
    for key in ("pred_field", "fluid_temperature", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(records[0][0][key], records[1][0][key], rtol=0., atol=0.)
    torch.testing.assert_close(records[0][1], records[1][1], rtol=0., atol=0.)
    torch.testing.assert_close(records[0][2], records[1][2], rtol=0., atol=0.)


@pytest.mark.parametrize("reduction", ["pool_first", "edge_first"])
def test_trained_nonzero_context_keeps_native_heat_null_affine_and_transpose(reduction):
    adapters, structure, fluid, local = fixture(reduction)
    adapter = adapters[1].eval()
    with torch.no_grad():
        output_weight = adapter.core.receiver_readout.output_projection.weight
        output_weight.copy_(.01 * torch.sin(torch.arange(output_weight.numel()).reshape_as(output_weight)))
        prepared = adapter.prepare_native(structure, fluid, local_query_points=local, ntheta=16)
        heat = torch.tensor([[1., -.5]], dtype=torch.float64)
        delta = torch.tensor([[-.2, .3]], dtype=torch.float64)
        zero = adapter.apply_native(prepared, torch.zeros_like(heat), accumulation_dtype=torch.float64)
        first = adapter.apply_native(prepared, heat, accumulation_dtype=torch.float64)
        second = adapter.apply_native(prepared, heat + delta, accumulation_dtype=torch.float64)
        increment = adapter.apply_increment(prepared, delta, accumulation_dtype=torch.float64)
        for key in ("fluid_temperature", "pred_interface", "pred_internal_temperature"):
            assert torch.count_nonzero(zero[key]) == 0
            torch.testing.assert_close(second[key] - first[key], increment[key], rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(first["pred_field"][..., :4], second["pred_field"][..., :4], rtol=0., atol=0.)
        query = prepared.query_batch
        affine = adapter.core.prepare_affine_query(prepared.organization, query)
        kernels = adapter.core.export_source_kernels(prepared.organization, query, accumulation_dtype=torch.float64)
        cotangent = torch.sin(torch.arange(query.num_queries, dtype=torch.float64))[None]
        transpose = adapter.core.heat_vjp_prepared_affine(affine, cotangent, accumulation_dtype=torch.float64)
        torch.testing.assert_close(transpose, torch.einsum("bqm,bq->bm", kernels, cotangent), rtol=2e-11, atol=2e-11)
