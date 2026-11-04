"""Native Thermal port extraction using the existing field decode."""
from dataclasses import replace
import torch
from .native_observations import NativeStencil, interpolate_native, _column


def prescribed_h(structure, nx, ny):
    centers = structure["module_centers"]
    lengths = torch.cat((_column(structure, "domain_length_x", centers, 12.0),
                         _column(structure, "domain_length_y", centers, 6.0)), -1)
    material = structure["material_params"]
    radius = material[:, 5:6]
    delta = torch.minimum((lengths / lengths.new_tensor([nx, ny])).amin(-1, keepdim=True), .15 * radius)
    solid, fluid = material[:, 3:4], material[:, 4:5]
    if bool(((solid <= 0) | (fluid <= 0) | (delta <= 0)).any()):
        raise ValueError("Prescribed conductivity and outside offset must be positive.")
    return (2 * solid * fluid / (solid + fluid)) / delta


def native_angle_view(prepared, ntheta):
    """Keep the 64-port union while exposing the requested native angle roles."""
    if ntheta == 64:
        return prepared
    if ntheta != 16 or prepared.ntheta != 64:
        raise ValueError("Expected the native 16- or 64-angle receiver catalogue.")
    batch, modules = prepared.source_present.shape
    stencils = dict(prepared.stencils)
    shapes = dict(prepared.role_shapes)
    for name in ("surface", "outside"):
        stencil = stencils[name]
        def select(value):
            return value.reshape(batch, modules, 64, *value.shape[2:])[:, :, ::4].reshape(batch, modules * 16, *value.shape[2:])
        stencils[name] = NativeStencil(select(stencil.indices), select(stencil.weights), select(stencil.valid))
        shapes[name] = (modules, 16)
    return replace(prepared, stencils=stencils, role_shapes=shapes, ntheta=16)


def outside_from_native_output(adapter, prepared64, output):
    """Extract physical 64-angle ports from an already executed native decode."""
    temperature = adapter._physical_temperature(output.joint_fields[..., 3])
    stencil = prepared64.stencils["outside"]
    outside = interpolate_native(temperature[..., None], stencil).reshape(
        prepared64.source_present.shape[0], *prepared64.role_shapes["outside"])
    valid = stencil.valid.reshape_as(outside)
    bad = prepared64.source_present[..., None] & (~valid | ~torch.isfinite(outside))
    if bad.any():
        case, module, angle = torch.nonzero(bad)[0].tolist()
        ids = prepared64.scene_key
        case_id = ids[case] if isinstance(ids, (tuple, list)) else case
        module_id = int(prepared64.scene_batch.module_ids[case, module])
        raise ValueError(f"Incomplete/nonfinite outside64 for case {case_id}, module {module_id}, angle {angle}.")
    return outside


def local_response_from_outside(adapter, local, prepared, heat, local_points, outside):
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
    return result
