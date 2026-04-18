"""Run a 2-D multi-cylinder PhiFlow benchmark with periodic boundaries.

Features
--------
* Uses `phi.torch.flow` by default so PhiFlow operates on the PyTorch backend.
* Supports inert cylinders (velocity / pressure / vorticity) and active cylinders
  with one-way thermal coupling (temperature advected by the flow field).
* Stores each saved frame in PhiFlow scene-compatible `.npz` files plus a JSON
  config and a CSV frame index for post-processing.

Important note
--------------
The active mode currently uses a Gaussian shell source around each cylinder,
which is intentionally lightweight. It is a good starting point for the modular
DT workflow and can later be upgraded to solid conduction or conjugate heat
transfer without changing the dataset structure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

from tqdm.auto import tqdm

from multicyl_common import (
    SimulationConfig,
    backup_config_file,
    config_from_dict,
    dataclass_to_dict,
    default_config_dir,
    default_data_dir,
    derive_runtime,
    kinematic_viscosity,
    thermal_diffusivity,
    make_case_dir,
    materialize_layout,
    resolve_config_path,
    resolve_data_path,
    write_json,
)


STEP_REPORT_INTERVAL = 10


def progress_enabled() -> bool:
    return sys.stdout.isatty()


def should_report_step(step_number: int, total_steps: int, interval: int = STEP_REPORT_INTERVAL) -> bool:
    return step_number == 1 or step_number == total_steps or (interval > 0 and step_number % interval == 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a periodic multi-cylinder wake case with PhiFlow.")
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help=f"Optional JSON config file. Relative paths are loaded from {default_config_dir()}.",
    )
    parser.add_argument("--case-id", type=str, default=None, help="Override case identifier.")
    parser.add_argument("--mode", type=str, choices=["inert", "active"], default=None, help="Simulation mode.")
    parser.add_argument("--num-cylinders", type=int, default=None, help="Override cylinder count.")
    parser.add_argument("--re", type=float, default=None, help="Override Reynolds number.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for layout / powers.")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default=None, help="Run on CPU or GPU.")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU index to use when --device gpu.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=f"Override output root directory. Relative paths are placed under {default_data_dir()}.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional text tag in the case directory name.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SimulationConfig:
    config = SimulationConfig().finalize()
    loaded_config_path: Path | None = None
    if args.config_json is not None:
        loaded_config_path = resolve_config_path(args.config_json)
        with loaded_config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        config = config_from_dict(raw)
        tqdm.write(f"Loaded config file: {loaded_config_path}")
    else:
        tqdm.write("No config file provided; using built-in defaults plus CLI overrides.")

    config.save.root_dir = str(resolve_data_path(config.save.root_dir))

    if args.case_id is not None:
        config.save.case_id = args.case_id
    if args.mode is not None:
        config.mode = args.mode
    if args.num_cylinders is not None:
        config.layout.num_cylinders = args.num_cylinders
        config.layout.centers = None
        config.layout.heat_powers = None
    if args.re is not None:
        config.flow.re = args.re
    if args.seed is not None:
        config.layout.seed = args.seed
        config.layout.centers = None
        config.layout.heat_powers = None
    if args.device is not None:
        config.execution.device = args.device
    if args.gpu_id is not None:
        config.execution.gpu_id = args.gpu_id
    if args.root_dir is not None:
        config.save.root_dir = str(resolve_data_path(args.root_dir))
    if args.tag is not None:
        config.save.tag = args.tag
    config = materialize_layout(config.finalize())

    if loaded_config_path is not None:
        backup_path = backup_config_file(loaded_config_path, case_id=config.save.case_id)
        tqdm.write(f"Backed up config to: {backup_path}")

    tqdm.write(
        "Prepared configuration: "
        f"case_id={config.save.case_id}, mode={config.mode}, "
        f"device={config.execution.device}, gpu_id={config.execution.gpu_id}, "
        f"cylinders={config.layout.num_cylinders}, Re={config.flow.re}"
    )

    return config


def configure_runtime_device(cfg: SimulationConfig) -> None:
    """Select CPU or a specific GPU before importing PhiFlow/Torch."""
    if cfg.execution.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tqdm.write("Simulation device: CPU")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.execution.gpu_id)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run with GPU selection.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU was requested (gpu_id={cfg.execution.gpu_id}) but CUDA is not available to PyTorch."
        )

    torch.cuda.set_device(0)
    tqdm.write(f"Simulation device: GPU {cfg.execution.gpu_id}")


# PhiFlow is imported lazily so the script stays importable on systems where the
# package is not installed yet.

def import_phiflow() -> Dict[str, Any]:
    from phi.torch.flow import (  # type: ignore
        Box,
        CenteredGrid,
        Obstacle,
        Scene,
        Solve,
        Sphere,
        StaggeredGrid,
        advect,
        diffuse,
        extrapolation,
        field,
        fluid,
        math,
    )

    return {
        "Box": Box,
        "CenteredGrid": CenteredGrid,
        "Obstacle": Obstacle,
        "Scene": Scene,
        "Solve": Solve,
        "Sphere": Sphere,
        "StaggeredGrid": StaggeredGrid,
        "advect": advect,
        "diffuse": diffuse,
        "extrapolation": extrapolation,
        "field": field,
        "fluid": fluid,
        "math": math,
    }


def build_case_objects(cfg: SimulationConfig, api: Dict[str, Any]) -> Dict[str, Any]:
    Box = api["Box"]
    Sphere = api["Sphere"]
    Obstacle = api["Obstacle"]

    bounds = Box(x=cfg.domain.lx, y=cfg.domain.ly)
    obstacles = [Obstacle(Sphere(x=float(cx), y=float(cy), radius=cfg.domain.cylinder_radius)) for cx, cy in cfg.layout.centers or []]
    return {"bounds": bounds, "obstacles": tuple(obstacles)}


def make_scalar_grid(value: float, cfg: SimulationConfig, api: Dict[str, Any], *, periodic: bool = True):
    CenteredGrid = api["CenteredGrid"]
    extrapolation = api["extrapolation"]
    return CenteredGrid(
        value,
        extrapolation.PERIODIC if periodic else extrapolation.ZERO,
        x=cfg.domain.nx,
        y=cfg.domain.ny,
        bounds=api["Box"](x=cfg.domain.lx, y=cfg.domain.ly),
    )


def make_constant_velocity_field(cfg: SimulationConfig, api: Dict[str, Any], *, x_value: float | None = None, y_value: float = 0.0):
    """Create a uniform staggered vector field.

    Used both for the initial velocity and for simple body-force templates.
    """
    StaggeredGrid = api["StaggeredGrid"]
    extrapolation = api["extrapolation"]
    return StaggeredGrid(
        (cfg.flow.u_bulk if x_value is None else x_value, y_value),
        extrapolation.PERIODIC,
        x=cfg.domain.nx,
        y=cfg.domain.ny,
        bounds=api["Box"](x=cfg.domain.lx, y=cfg.domain.ly),
    )


def center_velocity_field(velocity, cfg: SimulationConfig, api: Dict[str, Any]):
    """Resample staggered velocity to cell centers for saving and visualization."""
    CenteredGrid = api["CenteredGrid"]
    extrapolation = api["extrapolation"]
    template = CenteredGrid(
        (0.0, 0.0),
        extrapolation.PERIODIC,
        x=cfg.domain.nx,
        y=cfg.domain.ny,
        bounds=api["Box"](x=cfg.domain.lx, y=cfg.domain.ly),
    )
    return velocity @ template


def build_cylinder_mask_field(cfg: SimulationConfig, api: Dict[str, Any]):
    """Create a centered binary cylinder mask field for later post-processing."""
    CenteredGrid = api["CenteredGrid"]
    math_mod = api["math"]
    extrapolation = api["extrapolation"]

    lx, ly = cfg.domain.lx, cfg.domain.ly
    radius = cfg.domain.cylinder_radius
    centers = cfg.layout.centers or []

    def mask_fn(x):
        x_comp = x.vector["x"]
        y_comp = x.vector["y"]
        mask = 0.0
        for cx, cy in centers:
            dx = ((x_comp - cx + 0.5 * lx) % lx) - 0.5 * lx
            dy = ((y_comp - cy + 0.5 * ly) % ly) - 0.5 * ly
            dist = math_mod.sqrt(dx * dx + dy * dy)
            mask = math_mod.maximum(mask, math_mod.where(dist <= radius, 1.0, 0.0))
        return mask

    return CenteredGrid(mask_fn, extrapolation.PERIODIC, x=cfg.domain.nx, y=cfg.domain.ny, bounds=api["Box"](x=lx, y=ly))


def build_heat_source_field(cfg: SimulationConfig, api: Dict[str, Any]):
    """Create a smooth shell source around each cylinder.

    The shell formulation is intentionally simple and robust for a first demo.
    It approximates heat released near the cylinder-fluid interface without
    solving a separate solid domain.
    """
    CenteredGrid = api["CenteredGrid"]
    math_mod = api["math"]
    extrapolation = api["extrapolation"]

    if not cfg.thermal.enabled:
        return None

    lx, ly = cfg.domain.lx, cfg.domain.ly
    radius = cfg.domain.cylinder_radius
    sigma = cfg.thermal.source_sigma
    centers = cfg.layout.centers or []
    powers = cfg.layout.heat_powers or [0.0 for _ in centers]

    def source_fn(x):
        x_comp = x.vector["x"]
        y_comp = x.vector["y"]
        total = 0.0
        for (cx, cy), qdot in zip(centers, powers):
            dx = ((x_comp - cx + 0.5 * lx) % lx) - 0.5 * lx
            dy = ((y_comp - cy + 0.5 * ly) % ly) - 0.5 * ly
            dist = math_mod.sqrt(dx * dx + dy * dy)
            ring = math_mod.exp(-0.5 * ((dist - radius) / sigma) ** 2)
            total += float(qdot) * ring
        return total

    return CenteredGrid(
        source_fn,
        extrapolation.PERIODIC,
        x=cfg.domain.nx,
        y=cfg.domain.ny,
        bounds=api["Box"](x=lx, y=ly),
    )


def step_simulation(velocity, pressure, temperature, forcing_field, heat_source, obstacles, cfg: SimulationConfig, api: Dict[str, Any]):
    """Advance one time step.

    This function is isolated so later experiments can swap in alternative
    advection schemes, forcing laws, or multiphysics couplings without changing
    the storage and CLI logic.
    """
    advect = api["advect"]
    diffuse = api["diffuse"]
    fluid = api["fluid"]
    Solve = api["Solve"]

    dt = cfg.flow.dt
    nu = kinematic_viscosity(cfg)

    velocity = advect.mac_cormack(velocity, velocity, dt=dt)
    velocity = diffuse.explicit(velocity, nu, dt=dt, substeps=cfg.flow.diffusion_substeps)
    velocity = velocity + forcing_field * (dt / max(cfg.flow.forcing_relaxation, 1e-8))
    velocity = fluid.apply_boundary_conditions(velocity, obstacles)
    velocity, pressure = fluid.make_incompressible(
        velocity,
        obstacles,
        Solve(x0=pressure, rank_deficiency=cfg.flow.pressure_rank_deficiency),
    )

    if cfg.thermal.enabled and temperature is not None and heat_source is not None:
        alpha = thermal_diffusivity(cfg)
        temperature = advect.mac_cormack(temperature, velocity, dt=dt)
        temperature = diffuse.explicit(temperature, alpha, dt=dt, substeps=cfg.thermal.diffusion_substeps)
        temperature = temperature + heat_source * dt

    return velocity, pressure, temperature


def save_frame(scene, frame: int, sim_step: int, physical_time: float, velocity, pressure, temperature, cylinder_mask, cfg: SimulationConfig, api: Dict[str, Any]) -> None:
    """Save one frame to PhiFlow scene files."""
    field = api["field"]

    velocity_centered = center_velocity_field(velocity, cfg, api)
    payload = {"Velocity": velocity_centered}
    if cfg.save.save_pressure:
        payload["Pressure"] = pressure
    if cfg.save.save_vorticity:
        payload["Vorticity"] = field.curl(velocity)
    if cfg.thermal.enabled and cfg.save.save_temperature and temperature is not None:
        payload["Temperature"] = temperature
    if cfg.save.save_cylinder_mask and frame == 0 and cylinder_mask is not None:
        payload["CylinderMask"] = cylinder_mask
    scene.write(payload, frame=frame)


def run_case(cfg: SimulationConfig) -> Path:
    tqdm.write("Starting simulation run...")
    configure_runtime_device(cfg)
    api = import_phiflow()
    case_dir = make_case_dir(cfg.save)
    scene_dir = case_dir / "scene"
    scene_dir.mkdir(exist_ok=True)
    tqdm.write(f"Created case directory: {case_dir}")

    runtime = derive_runtime(cfg)
    write_json(case_dir / "case_config.json", dataclass_to_dict(cfg))
    tqdm.write(
        "Runtime summary: "
        f"num_steps={runtime['num_steps']}, warmup_time={runtime['warmup_time']:.3f}, "
        f"save_stride={runtime['save_stride']}, expected_saved_frames={runtime['expected_saved_frames']}"
    )

    scene = api["Scene"].at(str(scene_dir))
    objects = build_case_objects(cfg, api)
    forcing_acceleration = (cfg.flow.u_bulk ** 2) / max(cfg.domain.lx, 1e-8)
    forcing_field = make_constant_velocity_field(cfg, api, x_value=forcing_acceleration, y_value=0.0)
    velocity = make_constant_velocity_field(cfg, api)
    pressure = make_scalar_grid(0.0, cfg, api)
    temperature = make_scalar_grid(cfg.thermal.ambient_temperature, cfg, api) if cfg.thermal.enabled else None
    heat_source = build_heat_source_field(cfg, api)
    cylinder_mask = build_cylinder_mask_field(cfg, api) if cfg.save.save_cylinder_mask else None
    expected_saved_frames = max(1, int(runtime["expected_saved_frames"]))

    with (case_dir / "frame_index.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["saved_frame", "step", "time", "is_warmup_complete"])
        writer.writeheader()

        saved_frame = 0
        with tqdm(
            total=runtime["num_steps"],
            desc="Simulation steps",
            unit="step",
            dynamic_ncols=True,
            disable=not progress_enabled(),
        ) as step_bar:
            with tqdm(
                total=expected_saved_frames,
                desc="Saved frames",
                unit="frame",
                dynamic_ncols=True,
                disable=not progress_enabled(),
            ) as save_bar:
                for step in range(runtime["num_steps"]):
                    step_number = step + 1
                    physical_time = step_number * cfg.flow.dt
                    velocity, pressure, temperature = step_simulation(
                        velocity,
                        pressure,
                        temperature,
                        forcing_field,
                        heat_source,
                        objects["obstacles"],
                        cfg,
                        api,
                    )

                    warmup_complete = physical_time >= runtime["warmup_time"]
                    should_save = warmup_complete and (step_number % runtime["save_stride"] == 0)
                    if should_save:
                        save_frame(
                            scene,
                            frame=saved_frame,
                            sim_step=step_number,
                            physical_time=physical_time,
                            velocity=velocity,
                            pressure=pressure,
                            temperature=temperature,
                            cylinder_mask=cylinder_mask,
                            cfg=cfg,
                            api=api,
                        )
                        writer.writerow(
                            {
                                "saved_frame": saved_frame,
                                "step": step_number,
                                "time": f"{physical_time:.8f}",
                                "is_warmup_complete": int(warmup_complete),
                            }
                        )
                        saved_frame += 1
                        save_bar.update(1)
                        tqdm.write(
                            "Saved frame report: "
                            f"frame={saved_frame - 1}, step={step_number}, time={physical_time:.4f}, "
                            f"scene_dir={scene_dir}"
                        )

                    if should_report_step(step_number, runtime["num_steps"]):
                        tqdm.write(
                            "Step report: "
                            f"step={step_number}/{runtime['num_steps']}, time={physical_time:.4f}, "
                            f"saved_frames={saved_frame}"
                        )

                    if progress_enabled():
                        step_bar.set_postfix(saved=saved_frame, time=f"{physical_time:.2f}")
                    step_bar.update(1)

                if saved_frame > expected_saved_frames:
                    save_bar.total = saved_frame
                save_bar.n = saved_frame
                save_bar.refresh()

    return case_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    case_dir = run_case(cfg)
    tqdm.write(f"Simulation complete. Saved case to: {case_dir}")


if __name__ == "__main__":
    main()
