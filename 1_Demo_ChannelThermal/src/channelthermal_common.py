"""Shared utilities for the channel thermal modular-design demo.

Scope
-----
This module is imported by every Demo 1 data-generation script. It defines the
shared dataclass config schema, default paths, layout sampling, case-directory
creation, grid helpers, masks, interpolation, and small physics estimates.

Inputs and outputs
------------------
The helpers read JSON dictionaries from ``Configs/*.json`` and produce resolved
``SimulationConfig`` objects. They also create the timestamped raw case layout
used by both global channel cases and local module cases.

Training role
-------------
Keeping these helpers centralized prevents Stage-A local surrogate data and
Stage-B global channel data from drifting into incompatible naming, geometry, or
path conventions.

The demo intentionally keeps configuration, layout sampling, path handling, and
small numerical helpers in this module so the raw simulators, preprocessors,
visualizers, and future training scripts use one consistent data contract.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
DEMO_DIR = SRC_DIR.parent
DEFAULT_DATA_DIR = DEMO_DIR / "Data_Saved"
DEFAULT_CONFIG_DIR = DEMO_DIR / "Configs"
DEFAULT_CONFIG_BK_DIR = DEFAULT_CONFIG_DIR / "Config_bk"


# ----------------------------- Configuration blocks -----------------------------


@dataclass
class DomainConfig:
    """Nonperiodic 2-D channel geometry and grid settings."""

    nx: int = 256
    ny: int = 96
    lx: float = 12.0
    ly: float = 4.0
    module_radius: float = 0.45
    min_gap: float = 0.20
    inlet_margin: float = 1.0
    outlet_margin: float = 1.0
    wall_margin: float = 0.55


@dataclass
class FlowConfig:
    """Lightweight channel-flow settings."""

    re: float = 50.0
    u_in: float = 1.0
    dt: float = 0.01
    warmup_time: float = 5.0
    solve_time: float = 20.0
    save_stride: int = 20
    viscosity_scale: float = 1.0
    nu: Optional[float] = None
    pressure_rank_deficiency: int = 0
    flow_model: str = "analytic_wake"
    apply_projection: bool = False
    projection_iterations: int = 200
    projection_relaxation: float = 1.5


@dataclass
class ThermalConfig:
    """Conjugate-style scalar temperature settings.

    The global simulator uses one grid temperature over fluid and solids. Fluid
    cells advect and diffuse; solid cells only diffuse and receive source terms.
    The shared grid gives a cheap first-order interface coupling suitable for
    dataset prototyping.
    """

    enabled: bool = True
    t_in: float = 0.0
    t_wall: float = 0.0
    solid_alpha: float = 0.01
    fluid_alpha: float = 0.02
    solid_k: float = 1.0
    fluid_k: float = 1.0
    heat_power_min: float = 0.5
    heat_power_max: float = 2.0
    heat_start_time: float = 5.0
    stop_on_convergence: bool = True
    min_solve_time: float = 2.0
    max_solve_time: Optional[float] = None
    max_steps: Optional[int] = None
    convergence_window: int = 20
    convergence_tol: float = 1e-4
    convergence_rel_tol: float = 1e-5
    require_heat_active_for_convergence: bool = True


@dataclass
class LayoutConfig:
    """Module count, random seed, and optional explicit materialized layout."""

    num_modules: int = 4
    seed: int = 1
    layout_mode: str = "mixed"
    tandem_fraction: float = 0.25
    staggered_fraction: float = 0.25
    centers: Optional[List[List[float]]] = None
    heat_powers: Optional[List[float]] = None


@dataclass
class SaveConfig:
    """Raw and processed output settings."""

    root_dir: str = str(DEFAULT_DATA_DIR)
    case_id: str = "0001"
    tag: str = "channelthermal"
    save_pressure: bool = True
    save_vorticity: bool = True
    save_temperature: bool = True
    save_internal_temperature: bool = True
    save_interface_response: bool = True
    final_window_frames: int = 20


@dataclass
class LocalModuleConfig:
    """Cheap local conduction-surrogate data settings."""

    local_grid_size: int = 64
    n_interface_points: int = 64
    q_min: float = 0.5
    q_max: float = 2.0
    h_min: float = 0.2
    h_max: float = 3.0
    t_env_min: float = -0.5
    t_env_max: float = 0.5
    n_boundary_modes: int = 6
    boundary_modes_min: int = 2
    boundary_modes_max: Optional[int] = None
    h_log_perturb_scale: float = 0.10
    t_env_perturb_scale: float = 0.20
    solver_type: str = "polar_fd"
    polar_radial_points: int = 64
    polar_theta_points: Optional[int] = None
    polar_regularize_center: bool = True
    interface_sample_radius: float = 0.95
    solver_iterations: int = 4000
    solver_tolerance: float = 1e-6
    relaxation: float = 1.4


@dataclass
class ExecutionConfig:
    """Runtime device selection.

    The current numerical kernels are NumPy-based. The device fields are kept in
    the config so batch launchers and future GPU solvers can share one schema.
    """

    device: str = "cpu"
    gpu_id: int = 0


@dataclass
class SimulationConfig:
    """Top-level config used by both global and local Demo 1 scripts."""

    domain: DomainConfig = field(default_factory=DomainConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    local_module: LocalModuleConfig = field(default_factory=LocalModuleConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def finalize(self) -> "SimulationConfig":
        """Validate user settings and synchronize derived layout counts."""
        if self.domain.nx < 8 or self.domain.ny < 8:
            raise ValueError("domain.nx and domain.ny must both be at least 8.")
        if self.domain.lx <= 0.0 or self.domain.ly <= 0.0:
            raise ValueError("domain.lx and domain.ly must be positive.")
        if self.domain.module_radius <= 0.0:
            raise ValueError("domain.module_radius must be positive.")
        if self.layout.centers is not None:
            self.layout.num_modules = len(self.layout.centers)
        if self.layout.num_modules < 0:
            raise ValueError("layout.num_modules must be non-negative.")
        self.layout.layout_mode = str(self.layout.layout_mode).lower().strip()
        if self.layout.layout_mode not in {"random", "tandem", "staggered", "mixed"}:
            raise ValueError("layout.layout_mode must be one of 'random', 'tandem', 'staggered', or 'mixed'.")
        if self.layout.tandem_fraction < 0.0 or self.layout.staggered_fraction < 0.0:
            raise ValueError("layout tandem/staggered fractions must be non-negative.")
        if self.layout.tandem_fraction + self.layout.staggered_fraction > 1.0:
            raise ValueError("layout.tandem_fraction + layout.staggered_fraction must be <= 1.")
        if self.layout.heat_powers is not None and len(self.layout.heat_powers) != int(self.layout.num_modules):
            raise ValueError("layout.heat_powers length must match layout.num_modules.")
        if self.thermal.heat_power_min > self.thermal.heat_power_max:
            raise ValueError("thermal.heat_power_min must be <= thermal.heat_power_max.")
        if self.flow.dt <= 0.0:
            raise ValueError("flow.dt must be positive.")
        if self.flow.save_stride <= 0:
            raise ValueError("flow.save_stride must be positive.")
        if self.flow.flow_model not in {"analytic_wake", "projected_analytic_wake"}:
            raise ValueError("flow.flow_model must be 'analytic_wake' or 'projected_analytic_wake'.")
        if self.flow.projection_iterations < 0:
            raise ValueError("flow.projection_iterations must be non-negative.")
        if self.flow.projection_relaxation <= 0.0:
            raise ValueError("flow.projection_relaxation must be positive.")
        if self.thermal.min_solve_time < 0.0:
            raise ValueError("thermal.min_solve_time must be non-negative.")
        if self.thermal.max_solve_time is not None and self.thermal.max_solve_time <= 0.0:
            raise ValueError("thermal.max_solve_time must be positive when set.")
        if self.thermal.max_steps is not None and self.thermal.max_steps <= 0:
            raise ValueError("thermal.max_steps must be positive when set.")
        if self.thermal.convergence_window <= 0:
            raise ValueError("thermal.convergence_window must be positive.")
        if self.thermal.convergence_tol < 0.0:
            raise ValueError("thermal.convergence_tol must be non-negative.")
        if self.thermal.convergence_rel_tol < 0.0:
            raise ValueError("thermal.convergence_rel_tol must be non-negative.")
        if self.save.final_window_frames <= 0:
            raise ValueError("save.final_window_frames must be positive.")
        if self.local_module.local_grid_size < 8:
            raise ValueError("local_module.local_grid_size must be at least 8.")
        if self.local_module.n_interface_points < 8:
            raise ValueError("local_module.n_interface_points must be at least 8.")
        if self.local_module.n_boundary_modes < 1:
            raise ValueError("local_module.n_boundary_modes must be positive.")
        if self.local_module.boundary_modes_min < 0:
            raise ValueError("local_module.boundary_modes_min must be non-negative.")
        max_modes = (
            int(self.local_module.n_boundary_modes)
            if self.local_module.boundary_modes_max is None
            else int(self.local_module.boundary_modes_max)
        )
        if max_modes < self.local_module.boundary_modes_min:
            raise ValueError("local_module.boundary_modes_max must be >= boundary_modes_min.")
        if max_modes > self.local_module.n_boundary_modes:
            raise ValueError("local_module.boundary_modes_max must be <= n_boundary_modes.")
        if self.local_module.solver_type not in {"polar_fd", "cartesian_mask"}:
            raise ValueError("local_module.solver_type must be 'polar_fd' or 'cartesian_mask'.")
        if self.local_module.polar_radial_points < 8:
            raise ValueError("local_module.polar_radial_points must be at least 8.")
        if self.local_module.polar_theta_points is not None and self.local_module.polar_theta_points < 8:
            raise ValueError("local_module.polar_theta_points must be at least 8 when set.")
        if not (0.0 < self.local_module.interface_sample_radius <= 1.0):
            raise ValueError("local_module.interface_sample_radius must be in (0, 1].")
        if self.execution.device.lower().strip() not in {"cpu", "gpu"}:
            raise ValueError("execution.device must be 'cpu' or 'gpu'.")
        self.execution.device = self.execution.device.lower().strip()
        if self.execution.gpu_id < 0:
            raise ValueError("execution.gpu_id must be non-negative.")
        return self


# ----------------------------- Serialization helpers ----------------------------


T = TypeVar("T")


def _filtered_dataclass(cls: Type[T], payload: Dict[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})  # type: ignore[arg-type]


def config_from_dict(raw: Dict[str, Any]) -> SimulationConfig:
    """Rebuild nested dataclasses from a JSON dictionary.

    Unknown metadata keys such as ``runtime`` or ``generated_at`` are ignored so
    this function can load both template configs and resolved ``case_config``
    files.
    """

    cfg = SimulationConfig(
        domain=_filtered_dataclass(DomainConfig, raw.get("domain", {})),
        flow=_filtered_dataclass(FlowConfig, raw.get("flow", {})),
        thermal=_filtered_dataclass(ThermalConfig, raw.get("thermal", {})),
        layout=_filtered_dataclass(LayoutConfig, raw.get("layout", {})),
        save=_filtered_dataclass(SaveConfig, raw.get("save", {})),
        local_module=_filtered_dataclass(LocalModuleConfig, raw.get("local_module", {})),
        execution=_filtered_dataclass(ExecutionConfig, raw.get("execution", {})),
    )
    return cfg.finalize()


def dataclass_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Convert a config to a JSON-safe dict with runtime metadata."""
    if not is_dataclass(config):
        raise TypeError("dataclass_to_dict expects a SimulationConfig dataclass.")
    payload = asdict(config)
    payload["runtime"] = derive_runtime(config)
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def default_data_dir() -> Path:
    """Return this demo's canonical data directory."""
    return DEFAULT_DATA_DIR


def default_config_dir() -> Path:
    """Return this demo's canonical config directory."""
    return DEFAULT_CONFIG_DIR


def default_config_backup_dir(name: str | None = None) -> Path:
    """Return the config backup directory, optionally with a named subfolder."""
    if name is None:
        return DEFAULT_CONFIG_BK_DIR
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return DEFAULT_CONFIG_BK_DIR / safe_name


def resolve_data_path(path_like: str | Path | None) -> Path:
    """Resolve a user-facing data path.

    Relative case names are interpreted under ``1_Demo_ChannelThermal/Data_Saved``.
    Relative paths that already begin with ``Data_Saved`` are interpreted from
    the demo root.
    """
    if path_like is None:
        return DEFAULT_DATA_DIR.resolve()

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()

    normalized_parts = [part for part in path.parts if part not in {"", "."}]
    if not normalized_parts:
        return DEFAULT_DATA_DIR.resolve()
    if normalized_parts[0] == DEFAULT_DATA_DIR.name:
        return (DEMO_DIR / Path(*normalized_parts)).resolve()
    return (DEFAULT_DATA_DIR / Path(*normalized_parts)).resolve()


def resolve_config_path(path_like: str | Path) -> Path:
    """Resolve a config path, defaulting relative names to ``Configs/``."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = DEFAULT_CONFIG_DIR / path
    return path.resolve()


def backup_config_file(config_path: Path, case_id: str, stamp: str | None = None, name: str | None = None) -> Path:
    """Copy a template config into ``Configs/Config_bk`` for traceability."""
    resolved_config_path = config_path.expanduser().resolve()
    backup_dir = default_config_backup_dir(name)
    resolved_backup_dir = backup_dir.resolve()
    if resolved_backup_dir == resolved_config_path.parent or resolved_backup_dir in resolved_config_path.parents:
        return resolved_config_path

    timestamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_case_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in case_id)
    backup_name = f"{resolved_config_path.stem}_case_{safe_case_id}_{timestamp}{resolved_config_path.suffix}"
    backup_path = backup_dir / backup_name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_config_path, backup_path)
    return backup_path


# ------------------------------- Case management --------------------------------


def make_case_dir(save_cfg: SaveConfig) -> Path:
    """Create a timestamped raw case directory with ``scene`` and ``plots``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_case_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in save_cfg.case_id)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in save_cfg.tag)
    case_name = f"case_{safe_case_id}_{stamp}_{safe_tag}"
    case_dir = Path(save_cfg.root_dir).expanduser().resolve() / case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / "scene").mkdir(exist_ok=True)
    (case_dir / "plots").mkdir(exist_ok=True)
    return case_dir


def find_case_dirs(input_root: Path) -> List[Tuple[str, Path]]:
    """Return raw case directories, preserving optional train/test split labels."""
    root = input_root.expanduser().resolve()
    records: List[Tuple[str, Path]] = []
    if not root.exists():
        return records

    has_split_dirs = (root / "train").exists() or (root / "test").exists()
    if has_split_dirs:
        for split_name in ("train", "test"):
            split_root = root / split_name
            if not split_root.exists():
                continue
            for path in sorted(split_root.iterdir()):
                if path.is_dir() and (path / "case_config.json").exists():
                    records.append((split_name, path))
        return records

    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "case_config.json").exists():
            records.append(("unsplit", path))
    return records


# ------------------------------- Physics estimates ------------------------------


def module_diameter(config: SimulationConfig) -> float:
    """Return the module diameter in physical units."""
    return 2.0 * config.domain.module_radius


def kinematic_viscosity(config: SimulationConfig) -> float:
    """Compute kinematic viscosity from ``U D / Re`` unless ``flow.nu`` is set."""
    if config.flow.nu is not None:
        return float(config.flow.nu)
    re = max(float(config.flow.re), 1e-12)
    return float(config.flow.viscosity_scale) * float(config.flow.u_in) * module_diameter(config) / re


def derive_runtime(config: SimulationConfig) -> Dict[str, Any]:
    """Return simple time-step and save-cadence metadata."""
    warmup_time = max(0.0, float(config.flow.warmup_time))
    solve_time = max(0.0, float(config.flow.solve_time))
    max_solve_time = float(config.thermal.max_solve_time) if config.thermal.max_solve_time is not None else solve_time
    total_time = warmup_time + solve_time
    num_steps = max(1, int(math.ceil(total_time / max(float(config.flow.dt), 1e-12))))
    expected_saved_frames = max(1, int(math.ceil(num_steps / max(1, int(config.flow.save_stride)))))
    return {
        "warmup_time": warmup_time,
        "solve_time": solve_time,
        "max_solve_time": max_solve_time,
        "total_time": total_time,
        "num_steps": num_steps,
        "save_stride": int(config.flow.save_stride),
        "expected_saved_frames": expected_saved_frames,
        "nu": kinematic_viscosity(config),
        "stop_on_convergence": bool(config.thermal.stop_on_convergence),
        "convergence_window": int(config.thermal.convergence_window),
        "convergence_tol": float(config.thermal.convergence_tol),
        "convergence_rel_tol": float(config.thermal.convergence_rel_tol),
    }


# ----------------------------- Layout / design helpers ---------------------------


def _layout_bounds(config: SimulationConfig) -> Tuple[float, float, float, float, float]:
    r = float(config.domain.module_radius)
    xmin = float(config.domain.inlet_margin) + r
    xmax = float(config.domain.lx) - float(config.domain.outlet_margin) - r
    ymin = float(config.domain.wall_margin) + r
    ymax = float(config.domain.ly) - float(config.domain.wall_margin) - r
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Domain is too small for the requested module radius and margins.")
    min_center_dist = 2.0 * r + float(config.domain.min_gap)
    return xmin, xmax, ymin, ymax, min_center_dist


def _valid_nonoverlap(centers: Sequence[Sequence[float]], min_center_dist: float) -> bool:
    for idx, (cx, cy) in enumerate(centers):
        for ox, oy in centers[:idx]:
            if float(np.hypot(float(cx) - float(ox), float(cy) - float(oy))) < min_center_dist:
                return False
    return True


def _sample_random_module_centers(config: SimulationConfig, rng: np.random.Generator) -> List[List[float]]:
    """Sample non-overlapping circular modules in a nonperiodic channel."""
    xmin, xmax, ymin, ymax, min_center_dist = _layout_bounds(config)

    centers: List[List[float]] = []
    attempts = 0
    max_attempts = 20_000
    while len(centers) < int(config.layout.num_modules):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Failed to sample a valid non-overlapping module layout.")
        candidate = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)
        if all(float(np.hypot(candidate[0] - cx, candidate[1] - cy)) >= min_center_dist for cx, cy in centers):
            centers.append(candidate.tolist())
    return centers


def _sample_ordered_module_centers(config: SimulationConfig, rng: np.random.Generator, *, staggered: bool) -> List[List[float]]:
    num_modules = int(config.layout.num_modules)
    if num_modules <= 0:
        return []
    xmin, xmax, ymin, ymax, min_center_dist = _layout_bounds(config)
    y_span = ymax - ymin
    x_span = xmax - xmin
    base_y = float(rng.uniform(ymin + 0.2 * y_span, ymax - 0.2 * y_span)) if y_span > 0.0 else ymin
    max_jitter_x = 0.08 * x_span / max(num_modules - 1, 1)
    max_jitter_y = min(0.35 * min_center_dist, 0.18 * y_span) if y_span > 0.0 else 0.0
    stagger_offset = min(0.55 * min_center_dist, 0.35 * y_span) if y_span > 0.0 else 0.0
    for _attempt in range(2000):
        xs = np.linspace(xmin, xmax, num_modules, dtype=float)
        if num_modules > 2 and max_jitter_x > 0.0:
            xs[1:-1] += rng.uniform(-max_jitter_x, max_jitter_x, size=num_modules - 2)
        ys = np.full(num_modules, base_y, dtype=float)
        if staggered:
            signs = np.where(np.arange(num_modules) % 2 == 0, -1.0, 1.0)
            ys += signs * stagger_offset
            ys += rng.uniform(-0.15 * max(stagger_offset, max_jitter_y), 0.15 * max(stagger_offset, max_jitter_y), size=num_modules)
        elif max_jitter_y > 0.0:
            ys += rng.uniform(-max_jitter_y, max_jitter_y, size=num_modules)
        ys = np.clip(ys, ymin, ymax)
        centers = [[float(x), float(y)] for x, y in zip(xs, ys)]
        if _valid_nonoverlap(centers, min_center_dist):
            return centers
        base_y = float(rng.uniform(ymin, ymax))
        max_jitter_y = min(0.75 * min_center_dist, 0.35 * y_span) if y_span > 0.0 else 0.0
        stagger_offset = min(0.85 * min_center_dist, 0.45 * y_span) if y_span > 0.0 else 0.0
    return _sample_random_module_centers(config, rng)


def _resolve_layout_mode(config: SimulationConfig, rng: np.random.Generator) -> str:
    mode = str(config.layout.layout_mode).lower().strip()
    if mode != "mixed":
        return mode
    tandem = float(config.layout.tandem_fraction)
    staggered = float(config.layout.staggered_fraction)
    choice = float(rng.uniform())
    if choice < tandem:
        return "tandem"
    if choice < tandem + staggered:
        return "staggered"
    return "random"


def sample_module_centers(config: SimulationConfig) -> List[List[float]]:
    """Sample non-overlapping modules with random, tandem, staggered, or mixed layout modes."""
    rng = np.random.default_rng(int(config.layout.seed))
    mode = _resolve_layout_mode(config, rng)
    if mode == "tandem":
        return _sample_ordered_module_centers(config, rng, staggered=False)
    if mode == "staggered":
        return _sample_ordered_module_centers(config, rng, staggered=True)
    return _sample_random_module_centers(config, rng)


def sample_heat_powers(config: SimulationConfig) -> List[float]:
    """Sample positive per-module internal heat generation rates."""
    rng = np.random.default_rng(int(config.layout.seed) + 1009)
    powers = rng.uniform(
        low=float(config.thermal.heat_power_min),
        high=float(config.thermal.heat_power_max),
        size=int(config.layout.num_modules),
    )
    return [float(value) for value in powers]


def materialize_layout(config: SimulationConfig) -> SimulationConfig:
    """Fill in module centers and heat powers when the JSON leaves them null."""
    # Layout materialization is intentionally deterministic from layout.seed.
    # Batch launchers therefore need only record the seed to reproduce a case.
    if config.layout.centers is None:
        config.layout.centers = sample_module_centers(config)
    else:
        config.layout.centers = [[float(cx), float(cy)] for cx, cy in config.layout.centers]
        config.layout.num_modules = len(config.layout.centers)

    if config.layout.heat_powers is None:
        config.layout.heat_powers = sample_heat_powers(config)
    else:
        config.layout.heat_powers = [float(value) for value in config.layout.heat_powers]

    return config.finalize()


# ----------------------------- Grid / field helpers -----------------------------


def build_uniform_grid(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return cell-center ``x`` and ``y`` grids with shape ``[ny, nx]``."""
    x = np.linspace(0.0, float(config.domain.lx), int(config.domain.nx), endpoint=False)
    y = np.linspace(0.0, float(config.domain.ly), int(config.domain.ny), endpoint=False)
    x = x + 0.5 * float(config.domain.lx) / int(config.domain.nx)
    y = y + 0.5 * float(config.domain.ly) / int(config.domain.ny)
    return np.meshgrid(x, y)


def module_id_map(config: SimulationConfig) -> np.ndarray:
    """Return ``-1`` in fluid cells and module index in solid cells."""
    xx, yy = build_uniform_grid(config)
    ids = np.full(xx.shape, -1, dtype=np.int32)
    radius = float(config.domain.module_radius)
    for idx, (cx, cy) in enumerate(config.layout.centers or []):
        dist = np.hypot(xx - float(cx), yy - float(cy))
        ids[dist <= radius] = idx
    return ids


def module_mask(config: SimulationConfig) -> np.ndarray:
    """Return a boolean mask for all solid module cells."""
    return module_id_map(config) >= 0


def local_disk_grid(local_grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local normalized ``xi, eta`` grids and a disk mask."""
    axis = np.linspace(-1.0, 1.0, int(local_grid_size), dtype=np.float64)
    xi, eta = np.meshgrid(axis, axis)
    mask = xi * xi + eta * eta <= 1.0
    return xi, eta, mask


def bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray, config: SimulationConfig, fill_value: float = np.nan) -> np.ndarray:
    """Sample a ``[ny, nx]`` cell-centered field at physical coordinates."""
    arr = np.asarray(field, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    dx = float(config.domain.lx) / int(config.domain.nx)
    dy = float(config.domain.ly) / int(config.domain.ny)
    gx = x_arr / dx - 0.5
    gy = y_arr / dy - 0.5
    outside = (gx < 0.0) | (gy < 0.0) | (gx > arr.shape[1] - 1.0) | (gy > arr.shape[0] - 1.0)
    gx = np.clip(gx, 0.0, arr.shape[1] - 1.0)
    gy = np.clip(gy, 0.0, arr.shape[0] - 1.0)
    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gy).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, arr.shape[1] - 1)
    j1 = np.clip(j0 + 1, 0, arr.shape[0] - 1)
    wx = gx - i0
    wy = gy - j0
    sampled = (
        (1.0 - wx) * (1.0 - wy) * arr[j0, i0]
        + wx * (1.0 - wy) * arr[j0, i1]
        + (1.0 - wx) * wy * arr[j1, i0]
        + wx * wy * arr[j1, i1]
    )
    if np.isscalar(sampled):
        return np.array(fill_value if bool(outside) else sampled)
    sampled = np.asarray(sampled)
    sampled[outside] = fill_value
    return sampled


def local_bilinear_sample(field: np.ndarray, xi: np.ndarray, eta: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """Sample a local ``[-1, 1]^2`` grid at normalized coordinates."""
    arr = np.asarray(field, dtype=np.float64)
    size_y, size_x = arr.shape
    gx = (np.asarray(xi, dtype=np.float64) + 1.0) * 0.5 * (size_x - 1)
    gy = (np.asarray(eta, dtype=np.float64) + 1.0) * 0.5 * (size_y - 1)
    outside = (gx < 0.0) | (gy < 0.0) | (gx > size_x - 1.0) | (gy > size_y - 1.0)
    gx = np.clip(gx, 0.0, size_x - 1.0)
    gy = np.clip(gy, 0.0, size_y - 1.0)
    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gy).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, size_x - 1)
    j1 = np.clip(j0 + 1, 0, size_y - 1)
    wx = gx - i0
    wy = gy - j0
    sampled = (
        (1.0 - wx) * (1.0 - wy) * arr[j0, i0]
        + wx * (1.0 - wy) * arr[j0, i1]
        + (1.0 - wx) * wy * arr[j1, i0]
        + wx * wy * arr[j1, i1]
    )
    sampled = np.asarray(sampled)
    sampled[outside] = fill_value
    return sampled


def compute_vorticity(u: np.ndarray, v: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Compute ``omega = dv/dx - du/dy`` on the cell-centered grid."""
    dx = float(config.domain.lx) / int(config.domain.nx)
    dy = float(config.domain.ly) / int(config.domain.ny)
    dvdx = np.gradient(v, dx, axis=1, edge_order=1)
    dudy = np.gradient(u, dy, axis=0, edge_order=1)
    return dvdx - dudy


def write_frame_index(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write a CSV frame index from dictionaries."""
    import csv

    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def string_array(values: Sequence[str]) -> np.ndarray:
    """Return a compact UTF-8 string array suitable for HDF5 or NPZ metadata."""
    return np.asarray(list(values), dtype="S")
