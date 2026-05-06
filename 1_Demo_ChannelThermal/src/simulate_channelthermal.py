"""Run a 2-D nonperiodic channel thermal modular-design case.

This first Demo 1 simulator is intentionally lightweight and reproducible. It
does not attempt high-fidelity CFD. Instead, it builds a steady laminar channel
velocity/pressure approximation with no-slip circular solids, then advances a
single shared temperature grid over fluid and solid cells:

* fluid: advection + diffusion
* solid modules: diffusion + internal heat generation
* top/bottom walls: isothermal
* inlet: prescribed temperature
* outlet: zero-gradient temperature

The shared grid approximates conjugate interface coupling by diffusion across
the module mask boundary. The saved data contract is designed for future
hypergraph-organized neural-field training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from channelthermal_common import (
    SimulationConfig,
    backup_config_file,
    bilinear_sample,
    build_uniform_grid,
    compute_vorticity,
    config_from_dict,
    dataclass_to_dict,
    default_config_dir,
    default_data_dir,
    derive_runtime,
    kinematic_viscosity,
    local_disk_grid,
    make_case_dir,
    materialize_layout,
    module_id_map,
    read_json,
    resolve_config_path,
    resolve_data_path,
    string_array,
    write_json,
)


INTERFACE_FEATURE_NAMES = (
    "theta",
    "normal_x",
    "normal_y",
    "T_surface",
    "T_outside",
    "q_normal",
    "u_normal",
    "u_tangent",
    "h_proxy",
)


def progress_enabled() -> bool:
    """Use tqdm bars only in interactive terminals."""
    return sys.stdout.isatty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a nonperiodic channel thermal case.")
    parser.add_argument(
        "--config-json",
        type=str,
        default="config_channelthermal.json",
        help=f"JSON config file. Relative paths are loaded from {default_config_dir()}.",
    )
    parser.add_argument("--case-id", type=str, default=None, help="Override case identifier.")
    parser.add_argument("--num-modules", type=int, default=None, help="Override module count and resample layout.")
    parser.add_argument("--re", type=float, default=None, help="Override Reynolds number.")
    parser.add_argument("--seed", type=int, default=None, help="Override layout seed and resample layout/powers.")
    parser.add_argument("--device", choices=["cpu", "gpu"], default=None, help="Record CPU/GPU selection in config.")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU index metadata when --device gpu.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=f"Override output root. Relative paths are placed under {default_data_dir()}.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional tag in the case directory name.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SimulationConfig:
    """Load template config, apply CLI overrides, and materialize the layout."""
    loaded_config_path = resolve_config_path(args.config_json)
    raw = read_json(loaded_config_path)
    cfg = config_from_dict(raw)
    cfg.save.root_dir = str(resolve_data_path(cfg.save.root_dir))

    if args.case_id is not None:
        cfg.save.case_id = args.case_id
    if args.num_modules is not None:
        cfg.layout.num_modules = args.num_modules
        cfg.layout.centers = None
        cfg.layout.heat_powers = None
    if args.re is not None:
        cfg.flow.re = args.re
    if args.seed is not None:
        cfg.layout.seed = args.seed
        cfg.layout.centers = None
        cfg.layout.heat_powers = None
    if args.device is not None:
        cfg.execution.device = args.device
    if args.gpu_id is not None:
        cfg.execution.gpu_id = args.gpu_id
    if args.root_dir is not None:
        cfg.save.root_dir = str(resolve_data_path(args.root_dir))
    if args.tag is not None:
        cfg.save.tag = args.tag

    cfg = materialize_layout(cfg.finalize())
    backup_path = backup_config_file(loaded_config_path, cfg.save.case_id, name="Configs_channelthermal")
    tqdm.write(f"Loaded config: {loaded_config_path}")
    tqdm.write(f"Backed up config to: {backup_path}")
    tqdm.write(
        "Prepared channel case: "
        f"case_id={cfg.save.case_id}, modules={cfg.layout.num_modules}, "
        f"Re={cfg.flow.re}, device={cfg.execution.device}, seed={cfg.layout.seed}"
    )
    return cfg


def configure_runtime_device(cfg: SimulationConfig) -> None:
    """Record device selection without importing GPU frameworks."""
    if cfg.execution.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tqdm.write("Simulation kernel: NumPy CPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.execution.gpu_id)
        tqdm.write(
            "Simulation kernel: NumPy CPU "
            f"(GPU {cfg.execution.gpu_id} requested for metadata/future solvers)"
        )


def build_channel_flow(cfg: SimulationConfig, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic, nonperiodic channel flow approximation.

    The field uses a laminar parabolic inlet profile, obstacle wake deficits,
    and local pressure perturbations. Module cells are set to no-slip.
    """
    xx, yy = build_uniform_grid(cfg)
    lx = float(cfg.domain.lx)
    ly = float(cfg.domain.ly)
    radius = float(cfg.domain.module_radius)
    u_in = float(cfg.flow.u_in)
    nu = kinematic_viscosity(cfg)

    eta = np.clip(yy / ly, 0.0, 1.0)
    wall_profile = 6.0 * eta * (1.0 - eta)
    wall_taper = np.sin(np.pi * eta) ** 0.75
    u = u_in * wall_profile
    v = np.zeros_like(u)

    pressure_drop = 12.0 * nu * u_in * lx / max(ly * ly, 1e-12)
    p = pressure_drop * (1.0 - xx / lx)

    for cx, cy in cfg.layout.centers or []:
        dx = xx - float(cx)
        dy = yy - float(cy)
        dist = np.hypot(dx, dy)
        near = np.exp(-np.maximum(dist - radius, 0.0) ** 2 / max((0.75 * radius) ** 2, 1e-12))
        wake = (dx > 0.0) * np.exp(-dx / max(4.5 * radius, 1e-12)) * np.exp(-(dy / max(1.35 * radius, 1e-12)) ** 2)
        upstream = np.exp(-((dx + 0.75 * radius) / max(1.1 * radius, 1e-12)) ** 2 - (dy / max(1.4 * radius, 1e-12)) ** 2)
        u *= np.clip(1.0 - 0.28 * near - 0.70 * wake - 0.16 * upstream, 0.02, 1.20)

        circulation = np.exp(-((dist - 1.2 * radius) / max(0.9 * radius, 1e-12)) ** 2)
        wake_skew = np.exp(-np.maximum(dx, 0.0) / max(5.0 * radius, 1e-12))
        v += 0.12 * u_in * np.tanh(dy / max(0.35 * radius, 1e-12)) * circulation * wake_skew

        p += 0.55 * u_in * u_in * upstream
        p -= 0.28 * u_in * u_in * wake

    u *= wall_taper
    v *= wall_taper * (1.0 - 0.15 * xx / lx)
    u[ids >= 0] = 0.0
    v[ids >= 0] = 0.0
    p -= float(np.mean(p[:, -1]))
    omega = compute_vorticity(u, v, cfg)
    omega[ids >= 0] = 0.0
    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32), omega.astype(np.float32)


def enforce_temperature_boundaries(temperature: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Apply inlet, outlet, and isothermal wall thermal boundary conditions."""
    temperature[:, 0] = float(cfg.thermal.t_in)
    if temperature.shape[1] > 1:
        temperature[:, -1] = temperature[:, -2]
    temperature[0, :] = float(cfg.thermal.t_wall)
    temperature[-1, :] = float(cfg.thermal.t_wall)
    return temperature


def diffusion_rhs(temperature: np.ndarray, alpha: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Compute ``div(alpha grad T)`` using centered finite-volume fluxes."""
    dx = float(cfg.domain.lx) / int(cfg.domain.nx)
    dy = float(cfg.domain.ly) / int(cfg.domain.ny)
    rhs = np.zeros_like(temperature, dtype=np.float64)

    alpha_x = 0.5 * (alpha[:, 1:] + alpha[:, :-1])
    flux_x = alpha_x * (temperature[:, 1:] - temperature[:, :-1]) / (dx * dx)
    rhs[:, :-1] += flux_x
    rhs[:, 1:] -= flux_x

    alpha_y = 0.5 * (alpha[1:, :] + alpha[:-1, :])
    flux_y = alpha_y * (temperature[1:, :] - temperature[:-1, :]) / (dy * dy)
    rhs[:-1, :] += flux_y
    rhs[1:, :] -= flux_y
    return rhs


def advection_rhs(temperature: np.ndarray, u: np.ndarray, v: np.ndarray, fluid_mask: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Compute first-order upwind advection on fluid cells only."""
    dx = float(cfg.domain.lx) / int(cfg.domain.nx)
    dy = float(cfg.domain.ly) / int(cfg.domain.ny)

    left = np.empty_like(temperature)
    right = np.empty_like(temperature)
    down = np.empty_like(temperature)
    up = np.empty_like(temperature)
    left[:, 0] = temperature[:, 0]
    left[:, 1:] = temperature[:, :-1]
    right[:, -1] = temperature[:, -1]
    right[:, :-1] = temperature[:, 1:]
    down[0, :] = temperature[0, :]
    down[1:, :] = temperature[:-1, :]
    up[-1, :] = temperature[-1, :]
    up[:-1, :] = temperature[1:, :]

    dtdx = np.where(u >= 0.0, (temperature - left) / dx, (right - temperature) / dx)
    dtdy = np.where(v >= 0.0, (temperature - down) / dy, (up - temperature) / dy)
    rhs = -(u * dtdx + v * dtdy)
    rhs[~fluid_mask] = 0.0
    return rhs


def stable_substeps(cfg: SimulationConfig, u: np.ndarray, v: np.ndarray, alpha: np.ndarray) -> int:
    """Choose explicit thermal substeps for diffusion and advection stability."""
    dx = float(cfg.domain.lx) / int(cfg.domain.nx)
    dy = float(cfg.domain.ly) / int(cfg.domain.ny)
    max_u = max(float(np.max(np.abs(u))), 1e-12)
    max_v = max(float(np.max(np.abs(v))), 1e-12)
    max_alpha = max(float(np.max(alpha)), 1e-12)
    adv_dt = 0.45 * min(dx / max_u, dy / max_v)
    diff_dt = 0.20 * min(dx * dx, dy * dy) / max_alpha
    stable_dt = max(min(adv_dt, diff_dt), 1e-12)
    return max(1, int(math.ceil(float(cfg.flow.dt) / stable_dt)))


def heat_source_field(cfg: SimulationConfig, ids: np.ndarray) -> np.ndarray:
    """Return a solid-cell internal heat generation field."""
    source = np.zeros_like(ids, dtype=np.float64)
    powers = cfg.layout.heat_powers or []
    for idx, power in enumerate(powers):
        source[ids == idx] = float(power)
    return source


def step_temperature(
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    heat_source: np.ndarray,
    fluid_mask: np.ndarray,
    cfg: SimulationConfig,
    *,
    heat_active: bool,
    n_substeps: int,
) -> np.ndarray:
    """Advance the shared temperature grid for one configured time step."""
    dt_sub = float(cfg.flow.dt) / float(n_substeps)
    source = heat_source if heat_active and cfg.thermal.enabled else 0.0
    for _ in range(n_substeps):
        rhs = diffusion_rhs(temperature, alpha, cfg)
        rhs += advection_rhs(temperature, u, v, fluid_mask, cfg)
        temperature = temperature + dt_sub * (rhs + source)
        temperature = enforce_temperature_boundaries(temperature, cfg)
    return temperature


def extract_internal_temperatures(temperature: np.ndarray, cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Sample each module on a fixed normalized local disk grid."""
    local_size = int(cfg.local_module.local_grid_size)
    xi, eta, local_mask = local_disk_grid(local_size)
    radius = float(cfg.domain.module_radius)
    modules: List[np.ndarray] = []
    for cx, cy in cfg.layout.centers or []:
        sample_x = float(cx) + radius * xi
        sample_y = float(cy) + radius * eta
        sampled = bilinear_sample(temperature, sample_x, sample_y, cfg, fill_value=0.0)
        sampled = np.where(local_mask, sampled, 0.0)
        modules.append(sampled.astype(np.float32))
    if modules:
        return np.stack(modules, axis=0), local_mask.astype(np.uint8)
    return np.zeros((0, local_size, local_size), dtype=np.float32), local_mask.astype(np.uint8)


def extract_interface_response(
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Sample temperature, flux, and velocity response around every module."""
    n_theta = int(cfg.local_module.n_interface_points)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float64)
    normals_x = np.cos(theta)
    normals_y = np.sin(theta)
    tangents_x = -normals_y
    tangents_y = normals_x
    radius = float(cfg.domain.module_radius)
    delta = min(float(cfg.domain.lx) / int(cfg.domain.nx), float(cfg.domain.ly) / int(cfg.domain.ny), 0.15 * radius)
    k_interface = 2.0 * float(cfg.thermal.solid_k) * float(cfg.thermal.fluid_k) / max(
        float(cfg.thermal.solid_k) + float(cfg.thermal.fluid_k),
        1e-12,
    )

    response = np.zeros((int(cfg.layout.num_modules), n_theta, len(INTERFACE_FEATURE_NAMES)), dtype=np.float32)
    for module_idx, (cx, cy) in enumerate(cfg.layout.centers or []):
        cx = float(cx)
        cy = float(cy)
        xb = cx + radius * normals_x
        yb = cy + radius * normals_y
        xo = cx + (radius + delta) * normals_x
        yo = cy + (radius + delta) * normals_y
        t_surface = bilinear_sample(temperature, xb, yb, cfg, fill_value=np.nan)
        t_outside = bilinear_sample(temperature, xo, yo, cfg, fill_value=np.nan)
        u_out = bilinear_sample(u, xo, yo, cfg, fill_value=0.0)
        v_out = bilinear_sample(v, xo, yo, cfg, fill_value=0.0)
        q_normal = -k_interface * (t_outside - t_surface) / max(delta, 1e-12)
        u_normal = u_out * normals_x + v_out * normals_y
        u_tangent = u_out * tangents_x + v_out * tangents_y
        h_proxy = np.abs(q_normal) / (np.abs(t_surface - t_outside) + 1e-6)
        response[module_idx, :, 0] = theta
        response[module_idx, :, 1] = normals_x
        response[module_idx, :, 2] = normals_y
        response[module_idx, :, 3] = np.nan_to_num(t_surface)
        response[module_idx, :, 4] = np.nan_to_num(t_outside)
        response[module_idx, :, 5] = np.nan_to_num(q_normal)
        response[module_idx, :, 6] = np.nan_to_num(u_normal)
        response[module_idx, :, 7] = np.nan_to_num(u_tangent)
        response[module_idx, :, 8] = np.nan_to_num(h_proxy)
    return response


def save_frame(
    frame_path: Path,
    *,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    omega: np.ndarray,
    temperature: np.ndarray,
    ids: np.ndarray,
    cfg: SimulationConfig,
) -> Dict[str, np.ndarray]:
    """Write one compressed frame and return sampled module/interface arrays."""
    internal_temperature, internal_mask = extract_internal_temperatures(temperature, cfg)
    interface_response = extract_interface_response(temperature, u, v, cfg)
    module_mask = (ids >= 0).astype(np.uint8)
    payload = {
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "p": p.astype(np.float32),
        "omega": omega.astype(np.float32),
        "temperature": temperature.astype(np.float32),
        "module_mask": module_mask,
        "module_id": ids.astype(np.int32),
        "module_internal_temperature": internal_temperature.astype(np.float32),
        "module_internal_mask": internal_mask.astype(np.uint8),
        "interface_response": interface_response.astype(np.float32),
        "interface_feature_names": string_array(INTERFACE_FEATURE_NAMES),
    }
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(frame_path, **payload)
    return payload


def run_case(cfg: SimulationConfig) -> Path:
    """Execute one global channel thermal case and save raw frame data."""
    configure_runtime_device(cfg)
    case_dir = make_case_dir(cfg.save)
    scene_dir = case_dir / "scene"
    tqdm.write(f"Created case directory: {case_dir}")

    runtime = derive_runtime(cfg)
    ids = module_id_map(cfg)
    solid_mask = ids >= 0
    fluid_mask = ~solid_mask
    u, v, p, omega = build_channel_flow(cfg, ids)
    alpha = np.where(solid_mask, float(cfg.thermal.solid_alpha), float(cfg.thermal.fluid_alpha)).astype(np.float64)
    heat_source = heat_source_field(cfg, ids)
    n_substeps = stable_substeps(cfg, u, v, alpha)
    temperature = np.full((int(cfg.domain.ny), int(cfg.domain.nx)), float(cfg.thermal.t_in), dtype=np.float64)
    temperature = enforce_temperature_boundaries(temperature, cfg)

    config_payload = dataclass_to_dict(cfg)
    config_payload["runtime"].update({"thermal_substeps": n_substeps})
    config_payload["physical_assumptions"] = [
        "NumPy nonperiodic channel-flow approximation; not a high-fidelity CFD solver.",
        "Velocity and pressure are deterministic steady fields with no-slip module masks and wake deficits.",
        "One shared temperature grid is used across fluid and solid cells.",
        "Fluid cells advect and diffuse; solid cells diffuse and receive internal heat generation.",
        "Interface coupling is approximated by diffusion across neighboring grid cells.",
    ]
    config_payload["interface_feature_names"] = list(INTERFACE_FEATURE_NAMES)
    write_json(case_dir / "case_config.json", config_payload)

    xx, yy = build_uniform_grid(cfg)
    np.savez_compressed(case_dir / "grid.npz", x_grid=xx.astype(np.float32), y_grid=yy.astype(np.float32))

    rows: List[Dict[str, object]] = []
    previous_saved_temperature: np.ndarray | None = None
    saved_frame = 0
    tqdm.write(
        "Runtime summary: "
        f"steps={runtime['num_steps']}, dt={cfg.flow.dt}, save_stride={cfg.flow.save_stride}, "
        f"thermal_substeps={n_substeps}"
    )
    with tqdm(
        total=int(runtime["num_steps"]),
        desc="Channel thermal steps",
        unit="step",
        dynamic_ncols=True,
        disable=not progress_enabled(),
    ) as step_bar:
        for step in range(int(runtime["num_steps"])):
            step_number = step + 1
            physical_time = step_number * float(cfg.flow.dt)
            heat_active = bool(cfg.thermal.enabled and physical_time >= float(cfg.thermal.heat_start_time))
            temperature = step_temperature(
                temperature,
                u.astype(np.float64),
                v.astype(np.float64),
                alpha,
                heat_source,
                fluid_mask,
                cfg,
                heat_active=heat_active,
                n_substeps=n_substeps,
            )

            should_save = (step_number % int(cfg.flow.save_stride) == 0) or step_number == int(runtime["num_steps"])
            if should_save:
                frame_path = scene_dir / f"frame_{saved_frame:06d}.npz"
                save_frame(
                    frame_path,
                    u=u,
                    v=v,
                    p=p,
                    omega=omega,
                    temperature=temperature,
                    ids=ids,
                    cfg=cfg,
                )
                if previous_saved_temperature is None:
                    delta_inf = 0.0
                else:
                    delta_inf = float(np.max(np.abs(temperature - previous_saved_temperature)))
                previous_saved_temperature = temperature.copy()
                rows.append(
                    {
                        "saved_frame": saved_frame,
                        "file": frame_path.name,
                        "step": step_number,
                        "time": f"{physical_time:.8f}",
                        "heat_active": int(heat_active),
                        "warmup_complete": int(physical_time >= float(cfg.flow.warmup_time)),
                        "thermal_substeps": n_substeps,
                        "max_temperature": f"{float(np.max(temperature)):.8f}",
                        "mean_temperature": f"{float(np.mean(temperature)):.8f}",
                        "delta_inf": f"{delta_inf:.8e}",
                    }
                )
                saved_frame += 1
                tqdm.write(
                    "Saved frame: "
                    f"frame={saved_frame - 1}, step={step_number}, time={physical_time:.4f}, "
                    f"heat_active={int(heat_active)}"
                )
            if progress_enabled():
                step_bar.set_postfix(saved=saved_frame, t=f"{physical_time:.2f}")
            step_bar.update(1)

    with (case_dir / "frame_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tqdm.write(f"Simulation complete. Saved {saved_frame} frames to: {case_dir}")
    return case_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    run_case(cfg)


if __name__ == "__main__":
    main()
