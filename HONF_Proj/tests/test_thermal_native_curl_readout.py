"""Generator parity and nonlinear gradient contracts for the opt-in readout."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from channelthermal.flow_curl import (
    NATIVE_CURL_READOUT_LAW,
    build_native_curl_read_stencil,
    generator_discrete_vorticity,
    physical_uvp_native_curl,
)
from channelthermal.joint_regional import (
    InteractionPreservingThermalAdapter,
    load_interaction_preserving_thermal_checkpoint,
)
from test_interaction_preserving_runtime import (
    _joint_recipe_module,
    _thermal_engine_payload,
    _thermal_inputs,
    _thermal_stats,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_curl_matches_generator_at_boundaries_solids_and_ragged_unions(dtype):
    nx, ny = 8, 6
    y, x = np.meshgrid((np.arange(ny) + 0.5) / 2, (np.arange(nx) + 0.5) / 2,
                       indexing="ij")
    uvp = np.stack((np.sin(x) + 0.2 * y, np.cos(y) - 0.3 * x, x * y), axis=-1)
    solid = np.zeros((2, ny, nx), dtype=bool)
    solid[0, 2:4, 3:5] = True
    solid[1, 0, 0] = True
    selected = torch.tensor([[0, 47, 10, 10, 18, 25, 40], [47, 0, 7, 7, 3, 20, 32]])
    points = torch.tensor(np.stack((x.ravel(), y.ravel()), axis=-1), dtype=dtype)
    stencil = build_native_curl_read_stencil(
        points[selected], torch.tensor([[4., 3.], [4., 3.]], dtype=dtype),
        torch.tensor(solid), nx=nx, ny=ny,
    )
    mean = torch.tensor([0.17, -0.08, 0.2, 0.31], dtype=dtype)
    std = torch.tensor([1.4, 0.73, 1., 2.3], dtype=dtype)
    union_uvp = torch.tensor(uvp.reshape(-1, 3), dtype=dtype)[stencil.node_indices]
    normalized = ((union_uvp - mean[:3]) / std[:3]).requires_grad_()
    physical = physical_uvp_native_curl(normalized, stencil, mean, std)
    references = []
    for row in range(2):
        field = uvp.copy()
        field[solid[row], :2] = 0
        omega = generator_discrete_vorticity(field[..., 0], field[..., 1], solid[row], 0.5, 0.5)
        references.append(np.concatenate((field, omega[..., None]), axis=-1).reshape(-1, 4)[selected[row]])
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(physical, torch.tensor(np.stack(references), dtype=dtype),
                               rtol=tolerance, atol=tolerance)
    assert torch.equal(stencil.query_xy[:, 0], points[selected][:, 0])
    assert stencil.unique_counts[0] != stencil.unique_counts[1]
    assert torch.equal(physical[1, 1, [0, 1, 3]], torch.zeros(3, dtype=dtype))
    # The physical solid curl is zero even with a nonzero normalization mean.
    assert (physical[1, 1, 3] - mean[3]) / std[3] != 0
    physical[..., 3].square().sum().backward()
    assert torch.isfinite(normalized.grad).all()
    assert normalized.grad[..., :2].abs().sum(dim=(0, 1)).min() > 0
    assert normalized.grad[..., 2].count_nonzero() == 0


@pytest.mark.parametrize("invalid", ["off_grid", "half", "mask_float", "mask_shape"])
def test_native_curl_rejects_ambiguous_coordinates_and_mask_metadata(invalid):
    points = torch.tensor([[[0.25, 0.25]]])
    lengths = torch.tensor([[4., 3.]])
    mask = torch.zeros(1, 6, 8, dtype=torch.bool)
    if invalid == "off_grid":
        points[..., 0] += 0.01
    elif invalid == "half":
        points = points.half()
    elif invalid == "mask_float":
        mask = mask.float()
    else:
        mask = mask[:, :5]
    with pytest.raises((ValueError, TypeError)):
        build_native_curl_read_stencil(points, lengths, mask, nx=8, ny=6)


def _model(mode):
    return InteractionPreservingThermalAdapter(
        mode=mode, normalization_stats=_thermal_stats(), flow_readout_law=NATIVE_CURL_READOUT_LAW,
        regional_anchors=0 if mode == "P" else 16,
        locality_prior_strength=None if mode == "P" else 1.,
    )


def _native_inputs():
    structure, _, local = _thermal_inputs()
    receivers = torch.tensor([[[0.046875, 0.046875], [3.796875, 2.296875], [8.296875, 3.796875]]])
    mask = torch.zeros(1, 64, 128, dtype=torch.bool)
    return structure, receivers, local, mask


@pytest.mark.parametrize("mode", ["P", "P-G", "P-H"])
def test_native_adapter_zero_collective_roundtrip_and_curl_gradient(mode):
    model = _model(mode).eval()
    direct = _model("P").eval()
    structure, receivers, local, mask = _native_inputs()
    kwargs = {"local_query_points": local, "ntheta": 4, "native_solid_mask": mask}
    heat = torch.tensor([[0.5, 0.9]], dtype=torch.float64)
    prepared = model.prepare_native(structure, receivers, **kwargs)
    output = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)
    reference = direct.apply_native(direct.prepare_native(structure, receivers, **kwargs), heat,
                                    accumulation_dtype=torch.float64)
    for key in ("pred_field", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(output[key], reference[key], rtol=0, atol=0)
    assert model.core.field_head[-1].weight.shape[0] == 3
    assert model.model_config()["field_outputs"] == 3
    changed_heat = model.apply_native(prepared, 2 * heat, accumulation_dtype=torch.float64)
    assert torch.equal(output["pred_field"][..., :4], changed_heat["pred_field"][..., :4])
    assert not torch.equal(output["pred_field"][..., 4], changed_heat["pred_field"][..., 4])
    gradients = torch.autograd.grad(output["pred_field"][..., 3].square().mean(),
                                    tuple(model.parameters()), retain_graph=True, allow_unused=True)
    named = {name: gradient for (name, _), gradient in zip(model.named_parameters(), gradients)}
    projection = named["core.field_head.2.weight"]
    assert torch.isfinite(projection).all()
    assert projection[:2].abs().sum(dim=1).min() > 0
    assert projection[2].count_nonzero() == 0
    for prefix in ("core.source_read.", "core.module_messages.0.", "core.environment_messages.0.",
                   "core.source_updates.0.", "core.environment_updates.0."):
        active = [g for name, g in named.items() if name.startswith(prefix) and g is not None]
        assert active and all(torch.isfinite(g).all() for g in active)
        assert sum(float(g.abs().sum()) for g in active) > 0
    envelope = _thermal_engine_payload(model)
    envelope["arm"] = mode
    identity = envelope["experiment_identity"]
    identity["recipe"].update(
        mode=mode, regional_anchors=0 if mode == "P" else 16,
        flow_readout_law=NATIVE_CURL_READOUT_LAW,
        model_contract=_joint_recipe_module()._recovery_model_contract("thermal", mode, NATIVE_CURL_READOUT_LAW),
    )
    identity["provider_identity"]["model_family"] = model.family_id
    for payload in (model.checkpoint_payload(), envelope):
        restored = load_interaction_preserving_thermal_checkpoint(payload).eval()
        restored_output = restored.apply_native(restored.prepare_native(structure, receivers, **kwargs),
                                                heat, accumulation_dtype=torch.float64)
        torch.testing.assert_close(restored_output["pred_field"], output["pred_field"], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["missing_mask", "changed_mask", "changed_stencil", "changed_law"])
def test_native_adapter_rejects_missing_stale_and_mislabeled_state(mutation):
    model = _model("P")
    structure, receivers, local, mask = _native_inputs()
    if mutation == "missing_mask":
        with pytest.raises(ValueError, match="geometry-only"):
            model.prepare_native(structure, receivers, local_query_points=local, ntheta=4)
        return
    if mutation == "changed_law":
        payload = copy.deepcopy(model.checkpoint_payload())
        payload["adapter_config"]["native_curl_contract"]["omega_projection_rows"] = 1
        with pytest.raises(ValueError, match="contract"):
            load_interaction_preserving_thermal_checkpoint(payload)
        return
    prepared = model.prepare_native(structure, receivers, local_query_points=local, ntheta=4,
                                    native_solid_mask=mask)
    if mutation == "changed_mask":
        mask[0, 0, 0] = True
    else:
        prepared.native_curl_stencil.query_xy[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="native-curl"):
        model.apply_native(prepared, torch.tensor([[0.5, 0.9]]))
