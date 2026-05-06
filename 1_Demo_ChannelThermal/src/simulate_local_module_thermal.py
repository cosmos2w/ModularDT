"""Generate one raw local thermal-module surrogate case.

Scope
-----
This script handles the **local module** part of Demo 1. It reads
``Configs/config_local_module.json`` by default, samples known-before-solve
boundary conditions, solves a steady conduction problem inside one circular
solid module, and writes a raw local case under
``Data_Saved/LocalModule_Raw/case_*``.

Outputs
-------
Each case contains ``case_config.json``, ``frame_index.csv``, and
``local_solution.npz`` plus a scene-compatible ``scene/frame_000000.npz``. The
raw arrays are intentionally split into leakage-free inputs and targets:

* ``port_tokens`` contains only interface input/condition features.
* ``interface_targets`` contains solved ``T_surface`` and ``q_normal``.
* ``q_internal`` is saved once as a module-level scalar.

Training role
-------------
These raw cases are packed by ``preprocess_local_module_dataset.py`` for the
future Stage-A local module surrogate. Stage-A should learn internal
temperature and interface response from module parameters plus port inputs.

The local problem solves steady conduction inside a unit disk with uniform
internal heat generation and a Robin boundary condition:

    -k_s dT/dn = h(theta) * (T_surface - T_env(theta))

Boundary functions are sampled from low-frequency Fourier modes. The solver is
a robust finite-difference SOR iteration on a square grid with a circular mask.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm.auto import tqdm

from channelthermal_common import (
    SimulationConfig,
    backup_config_file,
    config_from_dict,
    dataclass_to_dict,
    default_config_dir,
    default_data_dir,
    local_bilinear_sample,
    local_disk_grid,
    make_case_dir,
    read_json,
    resolve_config_path,
    resolve_data_path,
    string_array,
    write_json,
)


CONDITION_COEFFICIENT_NAMES = ("mode", "T_cos", "T_sin", "h_cos", "h_sin")
PORT_INPUT_FEATURE_NAMES = ("theta", "cos_theta", "sin_theta", "T_env", "h")
INTERFACE_TARGET_NAMES = ("T_surface", "q_normal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate one local module thermal conduction case.")
    parser.add_argument(
        "--config-json",
        type=str,
        default="config_local_module.json",
        help=f"JSON config file. Relative paths are loaded from {default_config_dir()}.",
    )
    parser.add_argument("--case-id", type=str, default=None, help="Override local case identifier.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=f"Override output root. Relative paths are placed under {default_data_dir()}.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional tag in the case directory name.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SimulationConfig:
    """Load JSON config and apply command-line overrides.

    The local simulator does not need a global channel layout, but it reuses the
    shared dataclass schema for path handling, material parameters, and seeds.
    The template config is backed up for the same traceability convention used
    by the global simulator.
    """
    config_path = resolve_config_path(args.config_json)
    cfg = config_from_dict(read_json(config_path))
    cfg.save.root_dir = str(resolve_data_path(cfg.save.root_dir))
    if args.case_id is not None:
        cfg.save.case_id = args.case_id
    if args.seed is not None:
        cfg.layout.seed = int(args.seed)
    if args.root_dir is not None:
        cfg.save.root_dir = str(resolve_data_path(args.root_dir))
    if args.tag is not None:
        cfg.save.tag = args.tag
    cfg = cfg.finalize()
    backup_path = backup_config_file(config_path, cfg.save.case_id, name="Configs_local_module")
    tqdm.write(f"Loaded local config: {config_path}")
    tqdm.write(f"Backed up config to: {backup_path}")
    return cfg


def sample_boundary_conditions(cfg: SimulationConfig) -> Dict[str, np.ndarray | float]:
    """Sample q, T_env(theta), h(theta), and compact Fourier coefficients.

    This block generates **known-before-solve** inputs only. The Fourier
    coefficients are saved for analysis/visualization; the actual model-facing
    interface inputs are the evaluated ``T_env`` and ``h`` arrays in
    ``port_tokens``.
    """
    local = cfg.local_module
    rng = np.random.default_rng(int(cfg.layout.seed))
    n_theta = int(local.n_interface_points)
    n_modes = int(local.n_boundary_modes)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float64)
    q_internal = float(rng.uniform(float(local.q_min), float(local.q_max)))
    t_base = float(rng.uniform(float(local.t_env_min), float(local.t_env_max)))
    h_base = float(rng.uniform(float(local.h_min), float(local.h_max)))

    coeffs = np.zeros((n_modes, len(CONDITION_COEFFICIENT_NAMES)), dtype=np.float64)
    t_env = np.full_like(theta, t_base)
    h_factor = np.ones_like(theta)
    t_span = float(local.t_env_max) - float(local.t_env_min)
    for mode_idx in range(n_modes):
        mode = mode_idx + 1
        t_scale = 0.20 * t_span / max(mode, 1)
        h_scale = 0.25 / max(mode, 1)
        t_cos = float(rng.normal(0.0, t_scale))
        t_sin = float(rng.normal(0.0, t_scale))
        h_cos = float(rng.normal(0.0, h_scale))
        h_sin = float(rng.normal(0.0, h_scale))
        t_env += t_cos * np.cos(mode * theta) + t_sin * np.sin(mode * theta)
        h_factor += h_cos * np.cos(mode * theta) + h_sin * np.sin(mode * theta)
        coeffs[mode_idx] = [mode, t_cos, t_sin, h_cos, h_sin]

    t_env = np.clip(t_env, float(local.t_env_min), float(local.t_env_max))
    h = np.clip(h_base * h_factor, float(local.h_min), float(local.h_max))
    return {
        "theta": theta,
        "T_env": t_env,
        "h": h,
        "q_internal": q_internal,
        "condition_coefficients": coeffs,
        "T_base": t_base,
        "h_base": h_base,
    }


def interpolate_periodic(theta_grid: np.ndarray, values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Periodic linear interpolation over theta in [0, 2*pi)."""
    n = len(theta_grid)
    scaled = (np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi)) * n
    i0 = np.floor(scaled).astype(np.int64) % n
    i1 = (i0 + 1) % n
    w = scaled - np.floor(scaled)
    return (1.0 - w) * values[i0] + w * values[i1]


def solve_disk_conduction(
    cfg: SimulationConfig,
    theta: np.ndarray,
    t_env: np.ndarray,
    h_theta: np.ndarray,
    q_internal: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Solve the masked disk conduction problem by SOR iteration.

    The square grid keeps the implementation cheap. Cells outside the disk are
    handled through a Robin ghost-value approximation using local ``h(theta)``
    and ``T_env(theta)``. The loop stops when the max point update falls below
    ``local_module.solver_tolerance`` or the iteration budget is exhausted.
    """
    size = int(cfg.local_module.local_grid_size)
    xi, eta, mask = local_disk_grid(size)
    dx = 2.0 / max(size - 1, 1)
    ks = float(cfg.thermal.solid_k)
    relaxation = float(cfg.local_module.relaxation)
    max_iterations = int(cfg.local_module.solver_iterations)
    tolerance = float(cfg.local_module.solver_tolerance)

    mean_env = float(np.mean(t_env))
    mean_h = max(float(np.mean(h_theta)), 1e-8)
    temperature = np.zeros((size, size), dtype=np.float64)
    temperature[mask] = mean_env + q_internal / max(2.0 * mean_h, 1e-8)
    inside_indices = np.argwhere(mask)
    theta_cell = np.arctan2(eta, xi)
    h_cell = interpolate_periodic(theta, h_theta, theta_cell)
    env_cell = interpolate_periodic(theta, t_env, theta_cell)

    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        residual = 0.0
        for j, i in inside_indices:
            old = temperature[j, i]
            neighbor_sum = 0.0
            for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nj = int(j + dj)
                ni = int(i + di)
                if 0 <= nj < size and 0 <= ni < size and mask[nj, ni]:
                    neighbor_sum += temperature[nj, ni]
                else:
                    h_value = float(h_cell[j, i])
                    env_value = float(env_cell[j, i])
                    boundary_value = ((ks / dx) * old + h_value * env_value) / max((ks / dx) + h_value, 1e-12)
                    neighbor_sum += boundary_value
            target = 0.25 * (neighbor_sum + q_internal * dx * dx / max(ks, 1e-12))
            new_value = (1.0 - relaxation) * old + relaxation * target
            temperature[j, i] = new_value
            residual = max(residual, abs(new_value - old))
        if residual < tolerance:
            break

    temperature[~mask] = 0.0
    return temperature, xi, eta, iteration, float(residual)


def extract_boundary_targets(
    temperature: np.ndarray,
    theta: np.ndarray,
    t_env: np.ndarray,
    h_theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build separated interface inputs and targets around the disk boundary."""
    sample_radius = 0.995
    xi = sample_radius * np.cos(theta)
    eta = sample_radius * np.sin(theta)
    t_surface = local_bilinear_sample(temperature, xi, eta, fill_value=0.0)
    q_normal = h_theta * (t_surface - t_env)

    # Leakage guard: port_tokens are model inputs, so they contain only
    # condition features that are available before solving the temperature.
    port_tokens = np.stack([theta, np.cos(theta), np.sin(theta), t_env, h_theta], axis=-1)
    interface_targets = np.stack([t_surface, q_normal], axis=-1)
    return t_surface.astype(np.float32), q_normal.astype(np.float32), port_tokens.astype(np.float32), interface_targets.astype(np.float32)


def run_case(cfg: SimulationConfig) -> Path:
    """Run one local case and save raw arrays for preprocessing."""
    # 1. Sample boundary-condition functions and a scalar heat generation rate.
    conditions = sample_boundary_conditions(cfg)
    theta = conditions["theta"].astype(np.float64)
    t_env = conditions["T_env"].astype(np.float64)
    h_theta = conditions["h"].astype(np.float64)
    q_internal = float(conditions["q_internal"])
    # 2. Solve the steady disk conduction problem.
    temperature, xi, eta, iterations, residual = solve_disk_conduction(cfg, theta, t_env, h_theta, q_internal)
    solver_limit = int(cfg.local_module.solver_iterations)
    solver_tolerance = float(cfg.local_module.solver_tolerance)
    converged = bool(residual < solver_tolerance)
    tqdm.write(
        "Local module solver report: "
        f"converged={int(converged)}, "
        f"solver_iterations_used={iterations}/{solver_limit}, "
        f"residual={residual:.3e}, tolerance={solver_tolerance:.3e}"
    )
    disk_mask = (xi * xi + eta * eta <= 1.0).astype(np.uint8)

    # 3. Sample boundary inputs and solved targets on the same theta ports.
    t_surface, q_normal, port_tokens, interface_targets = extract_boundary_targets(temperature, theta, t_env, h_theta)

    # 4. Save one raw frame. These names are consumed directly by the local
    # preprocessor; keep additions backward-compatible.
    case_dir = make_case_dir(cfg.save)
    scene_dir = case_dir / "scene"
    payload = {
        "local_x": xi.astype(np.float32),
        "local_y": eta.astype(np.float32),
        "temperature": temperature.astype(np.float32),
        "disk_mask": disk_mask,
        "theta": theta.astype(np.float32),
        "T_surface": t_surface,
        "q_normal": q_normal,
        "T_env": t_env.astype(np.float32),
        "h": h_theta.astype(np.float32),
        "q_internal": np.asarray([q_internal], dtype=np.float32),
        "condition_coefficients": conditions["condition_coefficients"].astype(np.float32),
        "condition_coefficient_names": string_array(CONDITION_COEFFICIENT_NAMES),
        "port_tokens": port_tokens,
        "port_input_feature_names": string_array(PORT_INPUT_FEATURE_NAMES),
        "port_feature_names": string_array(PORT_INPUT_FEATURE_NAMES),
        "interface_targets": interface_targets,
        "interface_target_names": string_array(INTERFACE_TARGET_NAMES),
    }
    np.savez_compressed(scene_dir / "frame_000000.npz", **payload)
    np.savez_compressed(case_dir / "local_solution.npz", **payload)

    config_payload = dataclass_to_dict(cfg)
    config_payload["local_solution"] = {
        "q_internal": q_internal,
        "T_base": float(conditions["T_base"]),
        "h_base": float(conditions["h_base"]),
        "iterations": int(iterations),
        "solver_iterations_used": int(iterations),
        "solver_iterations_limit": solver_limit,
        "solver_tolerance": solver_tolerance,
        "converged": converged,
        "residual": residual,
        "physics": "steady disk conduction with Robin interface boundary functions",
    }
    write_json(case_dir / "case_config.json", config_payload)

    # 5. Write a one-row frame index for consistency with global cases.
    with (case_dir / "frame_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "saved_frame",
                "file",
                "iterations",
                "solver_iterations_used",
                "solver_iterations_limit",
                "solver_tolerance",
                "converged",
                "residual",
                "q_internal",
                "mean_temperature",
                "max_temperature",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "saved_frame": 0,
                "file": "frame_000000.npz",
                "iterations": iterations,
                "solver_iterations_used": iterations,
                "solver_iterations_limit": solver_limit,
                "solver_tolerance": f"{solver_tolerance:.8e}",
                "converged": int(converged),
                "residual": f"{residual:.8e}",
                "q_internal": f"{q_internal:.8f}",
                "mean_temperature": f"{float(np.mean(temperature[disk_mask.astype(bool)])):.8f}",
                "max_temperature": f"{float(np.max(temperature)):.8f}",
            }
        )

    tqdm.write(
        "Local module case complete: "
        f"case_dir={case_dir}, converged={int(converged)}, "
        f"solver_iterations_used={iterations}/{solver_limit}, "
        f"residual={residual:.3e}, q={q_internal:.3f}"
    )
    return case_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    run_case(cfg)


if __name__ == "__main__":
    main()
