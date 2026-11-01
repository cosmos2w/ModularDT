"""Finite Thermal local-response feedback through original module records.

The angular channels are surface temperature (benchmark-native units), followed by signed native
q-proxy, at theta_j=2*pi*j/64 counter-clockwise from +x. The physical q
convention and finite outside offset are those of the native dataset.
"""
from dataclasses import replace

import torch

from .native_observations import interpolate_native
from .training.local_expert_task import prescribed_h


def response_scene(scene, response, mean, scale):
    """Append normalized 64-angle response and presence flag to original M."""
    present = scene.module_mask.bool()
    if scene.module_features.shape[-1] != 9:
        raise ValueError("Expected the nine prescribed Thermal module features.")
    if response is None:
        block = scene.module_features.new_zeros(*present.shape, 129)
    else:
        if response.shape != (*present.shape, 64, 2):
            raise ValueError("Thermal response must contain 64 aligned T/q pairs per module.")
        bad = present[..., None, None] & ~torch.isfinite(response)
        if bad.any():
            case, module, angle, channel = torch.nonzero(bad)[0].tolist()
            ids = getattr(scene, "scene_key", None)
            case_id = ids[case] if isinstance(ids, (tuple, list)) else case
            module_ids = getattr(scene, "module_ids", None)
            module_id = int(module_ids[case, module]) if module_ids is not None else module
            raise ValueError(f"Nonfinite interface response for case {case_id}, module {module_id}, angle {angle}, channel {channel}.")
        mean = torch.as_tensor(mean, device=response.device, dtype=response.dtype)
        scale = torch.as_tensor(scale, device=response.device, dtype=response.dtype)
        if mean.shape != (2,) or scale.shape != (2,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Response normalization requires finite positive TRAIN scales.")
        normalized = (response - mean) / scale
        normalized = torch.where(present[..., None, None], normalized, torch.zeros_like(normalized))
        block = torch.cat((normalized[..., 0], normalized[..., 1], present[..., None].to(response)), -1)
    return replace(scene, module_features=torch.cat((scene.module_features, block), -1))


def transplant_widened_model(model, source_state):
    """Strictly transplant the saved 9-feature core into a 138-feature core."""
    target = model.state_dict()
    if set(target) != set(source_state):
        raise ValueError("Transfer source and widened model have different tensor names.")
    changed = [name for name in target if target[name].shape != source_state[name].shape]
    if len(changed) != 1 or not changed[0].endswith("module_encoder.0.weight"):
        raise ValueError(f"Unexpected transfer shape differences: {changed}.")
    name = changed[0]
    old, new = source_state[name], target[name].clone()
    if old.ndim != 2 or old.shape[1] != 25 or new.shape != (old.shape[0], 154):
        raise ValueError("Transfer requires actual 9+16 and 138+16 encoder inputs.")
    new.zero_()
    new[:, :9] = old[:, :9]
    new[:, 138:] = old[:, 9:]
    transferred = dict(source_state)
    transferred[name] = new
    model.load_state_dict(transferred, strict=True)
    return name


def prepare_response(adapter, initial, response, mean, scale):
    scene = replace(initial.scene_batch, module_features=initial.scene_batch.module_features[..., :9])
    scene = response_scene(scene, response, mean, scale)
    organization = adapter.core.prepare(scene)
    return replace(initial, scene_batch=scene, organization=organization)


def outside_geometry(adapter, prepared):
    """Construct the shared, parameter-independent native port geometry once."""
    return adapter._query_union(prepared.scene_batch, prepared.fluid_xy[:, :1], None, 64)


def outside64(adapter, prepared, mode, geometry=None):
    query, union, stencils, *rest = outside_geometry(adapter, prepared) if geometry is None else geometry
    output = adapter.core.decode(prepared.organization, query, intervention={"read_mode": mode})
    temp = adapter._physical_temperature(output.joint_fields[..., 3])
    outside = interpolate_native(temp[..., None], stencils["outside"]).reshape(prepared.source_present.shape[0], *rest[-1]["outside"])
    valid = stencils["outside"].valid.reshape_as(outside)
    bad = prepared.source_present[..., None] & (~valid | ~torch.isfinite(outside))
    if bad.any():
        case, module, angle = torch.nonzero(bad)[0].tolist()
        case_ids = prepared.scene_key
        case_id = case_ids[case] if isinstance(case_ids, (tuple, list)) else case
        module_id = int(prepared.scene_batch.module_ids[case, module])
        raise ValueError(f"Incomplete/nonfinite outside64 for case {case_id}, module {module_id}, angle {angle}.")
    return outside, {**output.work_counts, "native_union_rows": int(union.unique_counts.sum())}


def local_response(adapter, local, prepared, heat, local_points, mode, geometry=None):
    outside, counts = outside64(adapter, prepared, mode, geometry)
    theta = torch.arange(64, device=outside.device, dtype=outside.dtype) * (2 * torch.pi / 64)
    angles = torch.stack((theta, theta.cos(), theta.sin()), -1).view(1, 1, 64, 3).expand(*outside.shape, 3)
    h = prescribed_h(prepared.structure, adapter.nx, adapter.ny)[:, None, :].expand_as(outside)
    ports = torch.cat((angles, outside[..., None], h[..., None]), -1)
    ports = torch.where(prepared.source_present[..., None, None], ports, torch.zeros_like(ports))
    result = local(prepared.structure, heat, ports, local_points)
    bad = prepared.source_present[..., None, None] & ~torch.isfinite(result["interface_pred"])
    if bad.any():
        case, module, angle, channel = torch.nonzero(bad)[0].tolist()
        ids = prepared.scene_key
        case_id = ids[case] if isinstance(ids, (tuple, list)) else case
        module_id = int(prepared.scene_batch.module_ids[case, module])
        raise FloatingPointError(f"Nonfinite local response for case {case_id}, module {module_id}, angle {angle}, channel {channel}.")
    return result, counts
