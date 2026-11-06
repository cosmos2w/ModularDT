"""Target-free native Thermal inputs and physical design scoring.

The design is prescribed input metadata, not a physical reference record.
Each call builds a fresh scene and its full Eulerian and attached receivers.
The native model remains responsible for preparation and field decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .data.datasets import _local_disk_query_points, prescribed_raster_geometry
from .native_functionals import native_design_functionals


@dataclass(frozen=True)
class NativeThermalDesign:
    """One complete prescribed scene; original member and receiver IDs persist."""

    case_id: str
    member_ids: tuple[str, ...]
    structure: Mapping[str, torch.Tensor]
    heat: torch.Tensor
    fluid_xy: torch.Tensor
    local_query_points: torch.Tensor
    module_id_grid: np.ndarray
    fluid_xy_source: np.ndarray
    surface_xy_source: np.ndarray
    outside_xy_source: np.ndarray
    material_xy_source: np.ndarray
    fluid_ids: tuple[str, ...]
    surface_ids: tuple[str, ...]
    material_ids: tuple[str, ...]
    fluid_valid: np.ndarray
    surface_valid: np.ndarray
    outside_valid: np.ndarray
    material_valid: np.ndarray
    domain_length_x: float
    nx: int
    ny: int


def build_native_design(config: Mapping[str, Any], case_id: str, *,
                        member_ids: Sequence[str] | None = None,
                        device: torch.device | str = "cpu") -> NativeThermalDesign:
    """Build from physical configuration alone, without opening field values.

    Raster ownership follows the source generator's FP64 grid, inclusive disk
    boundary and last-member-wins ordering. Neural coordinates are cast only
    afterwards. Attached receiver coordinates follow each current member.
    """
    domain, layout = config["domain"], config["layout"]
    flow, thermal, local = config["flow"], config["thermal"], config["local_module"]
    nx, ny = int(domain["nx"]), int(domain["ny"])
    lx, ly, radius = (float(domain[k]) for k in ("lx", "ly", "module_radius"))
    centers = np.asarray(layout["centers"], dtype=np.float64)
    heat = np.asarray(layout["heat_powers"], dtype=np.float64)
    if (nx < 2 or ny < 2 or min(lx, ly, radius) <= 0 or not np.isfinite([lx, ly, radius]).all()
            or centers.ndim != 2 or centers.shape[1:] != (2,) or not len(centers)
            or heat.shape != (len(centers),) or not np.isfinite(centers).all() or not np.isfinite(heat).all()
            or int(layout["num_modules"]) != len(centers)):
        raise ValueError("Invalid prescribed Thermal grid, original-member geometry, or controls.")
    case_id = str(case_id)
    if not case_id:
        raise ValueError("A case ID is required to preserve attached receiver identities.")
    ids = tuple(member_ids) if member_ids is not None else tuple(
        f"{case_id}:module:{i}" for i in range(len(centers))
    )
    if len(ids) != len(centers) or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Original member IDs must be unique, nonempty strings in slot order.")
    ntheta, local_size = int(local["n_interface_points"]), int(local["local_grid_size"])
    if ntheta != 64 or local_size < 2:
        raise ValueError("Native Thermal scoring requires 64 surface angles and a nonempty local disk grid.")

    # Match build_uniform_grid/module_id_map, including FP64 operation order.
    x = np.linspace(0.0, lx, nx, endpoint=False) + 0.5 * lx / nx
    y = np.linspace(0.0, ly, ny, endpoint=False) + 0.5 * ly / ny
    xx, yy = np.meshgrid(x, y)
    owner = np.full((ny, nx), -1, dtype=np.int32)
    for slot, (cx, cy) in enumerate(centers):
        owner[np.hypot(xx - cx, yy - cy) <= radius] = slot
    fluid_xy = np.stack((xx.ravel(), yy.ravel()), axis=-1)

    axis = np.linspace(-1.0, 1.0, local_size, dtype=np.float64)
    local_x, local_y = np.meshgrid(axis, axis)
    disk = local_x * local_x + local_y * local_y <= 1.0
    local_source = np.stack((local_x[disk], local_y[disk]), axis=-1)
    # Stored raw interface normals have FP32 precision before the reference
    # adapter reconstructs attached world coordinates.
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float64)
    normals = np.stack((np.cos(theta), np.sin(theta)), axis=-1).astype(np.float32).astype(np.float64)
    delta = min(lx / nx, ly / ny, 0.15 * radius)
    surface = centers[:, None, :] + radius * normals[None]
    outside = centers[:, None, :] + (radius + delta) * normals[None]
    material = centers[:, None, :] + radius * local_source[None]

    def valid(xy: np.ndarray) -> np.ndarray:
        return ((xy[..., 0] >= 0.5 * lx / nx) & (xy[..., 0] <= lx - 0.5 * lx / nx)
                & (xy[..., 1] >= 0.5 * ly / ny) & (xy[..., 1] <= ly - 0.5 * ly / ny))

    nu = flow.get("nu")
    if nu is None:
        nu = float(flow.get("viscosity_scale", 1.0)) * float(flow["u_in"]) * 2.0 * radius / max(float(flow["re"]), 1e-12)
    material_params = [nu, *(thermal[k] for k in ("solid_alpha", "fluid_alpha", "solid_k", "fluid_k")), radius]
    if not np.isfinite(np.asarray(material_params, dtype=np.float64)).all() or min(map(float, material_params)) <= 0:
        raise ValueError("Native Thermal material coefficients must be positive and finite.")
    present = np.ones(len(centers), dtype=np.float32)
    centers64, raster_geometry = prescribed_raster_geometry(config, present)
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)[None]
    structure = {
        "module_centers": tensor(centers), "module_present": tensor(present),
        "material_params": tensor(material_params), "re": tensor([flow["re"]]),
        "u_in": tensor([flow["u_in"]]), "domain_length_x": tensor([lx]),
        "domain_length_y": tensor([ly]), "t_in": tensor([thermal["t_in"]]),
        "t_wall": tensor([thermal["t_wall"]]),
        "raster_module_centers": torch.as_tensor(centers64, device=device)[None],
        "raster_geometry": torch.as_tensor(raster_geometry, device=device)[None],
    }
    return NativeThermalDesign(
        case_id=case_id, member_ids=ids, structure=structure, heat=tensor(heat),
        fluid_xy=tensor(fluid_xy), local_query_points=tensor(_local_disk_query_points(disk)),
        module_id_grid=owner, fluid_xy_source=fluid_xy, surface_xy_source=surface,
        outside_xy_source=outside, material_xy_source=material,
        fluid_ids=tuple(f"grid:{j}:{i}" for j in range(ny) for i in range(nx)),
        surface_ids=tuple(f"{member}:port:{i}" for member in ids for i in range(ntheta)),
        material_ids=tuple(f"{member}:solid:{i}" for member in ids for i in range(len(local_source))),
        fluid_valid=(owner < 0).reshape(-1), surface_valid=valid(surface),
        outside_valid=valid(outside), material_valid=valid(material),
        domain_length_x=lx, nx=nx, ny=ny,
    )


def score_native_design(model: Any, design: NativeThermalDesign, *, tau: float,
                        mode: str = "restricted", intervention: Any = None) -> dict[str, Any]:
    """Freshly prepare/decode one design and return physical fields, J/P and work.

    Tensors and their gradients are retained. The caller owns evaluation mode,
    synchronization, timing, and any separate reference/physical-acceptance join.
    """
    if (model.nx, model.ny) != (design.nx, design.ny):
        raise ValueError("The model grid and prescribed native design grid differ.")
    if mode not in ("restricted", "full"):
        raise ValueError("Native member read mode must be restricted or full.")
    if model.training:
        raise ValueError("Native design scoring requires the model in evaluation mode.")
    prepared = model.prepare_native(
        design.structure, design.fluid_xy, physical_heat=design.heat,
        local_query_points=design.local_query_points, ntheta=64, scene_key=(design.case_id,),
    )
    options = {} if intervention is None else {"intervention": intervention}
    with model.core.read_execution(mode):
        native, counts, _ = model.apply_native_with_work_counts(prepared, design.heat, **options)
    field, interface = native["pred_field"], native["pred_interface"]
    material = native["pred_internal_temperature"][..., 0]
    present = prepared.source_present[..., None].bool()
    geometry = lambda mask: torch.as_tensor(mask, dtype=torch.bool, device=field.device)[None]
    fvalid = geometry(design.fluid_valid) & prepared.stencils["fluid"].valid.reshape(field.shape[:-1])
    svalid = geometry(design.surface_valid) & present & prepared.stencils["surface"].valid.reshape(interface.shape[:-1])
    qvalid = svalid & geometry(design.outside_valid) & prepared.stencils["outside"].valid.reshape(interface.shape[:-1])
    mvalid = geometry(design.material_valid) & present & prepared.stencils["material"].valid.reshape(material.shape)
    fields = {name: {"prediction": field[..., index], "mask": fvalid}
              for index, name in enumerate(("flow/u", "flow/v", "flow/p", "flow/omega", "temperature/fluid"))}
    fields.update({"temperature/surface": {"prediction": interface[..., 0], "mask": svalid},
                   "q_proxy": {"prediction": interface[..., 1], "mask": qvalid},
                   "temperature/material": {"prediction": material, "mask": mvalid}})
    j, p = native_design_functionals(material, mvalid, field[..., 2], fvalid,
                                     design.fluid_xy[..., 0], design.domain_length_x, tau)
    work = {"native_fluid_queries": int(design.fluid_xy.shape[1]),
            "native_union_rows": int(prepared.union.unique_counts.sum().item()),
            "native_local_pairs": int(prepared.query_batch.local_weights.numel()),
            f"member_routing_read_mode_{mode}": 1}
    work.update({f"core_prepare_primary_{key}": int(value) for key, value in prepared.organization.work_counts.items()})
    work.update({f"core_decode_primary_{key}": int(value) for key, value in counts.items()})
    return {"fields": fields, "functionals": {"J": j, "P": p}, "work": work,
            "native": native, "prepared": prepared}


__all__ = ["NativeThermalDesign", "build_native_design", "score_native_design"]
