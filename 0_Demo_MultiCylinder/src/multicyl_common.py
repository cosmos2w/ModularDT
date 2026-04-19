"""Shared utilities for the multi-cylinder PhiFlow demo benchmark.

This module keeps all non-PhiFlow logic in one place so later extensions
(model training, web demos, alternative simulators) can reuse the same
configuration and dataset conventions.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
DEMO_DIR = SRC_DIR.parent
DEFAULT_DATA_DIR = DEMO_DIR / "Data_Saved"
DEFAULT_CONFIG_DIR = DEMO_DIR / "Configs"
DEFAULT_CONFIG_BK_DIR = DEFAULT_CONFIG_DIR / "Config_bk"
DEFAULT_DOMAIN_SHAPE_DIR = DEMO_DIR / "Domain_shape"


# ----------------------------- Configuration blocks -----------------------------


@dataclass
class DomainConfig:
    """Geometric and discretization settings for the periodic 2-D domain."""

    nx: int = 256
    ny: int = 128
    lx: float = 24.0
    ly: float = 12.0
    cylinder_radius: float = 0.5
    min_gap: float = 0.35
    edge_margin: float = 2.0


@dataclass
class FlowConfig:
    """Fluid settings for the wake benchmark."""

    re: float = 100.0
    u_bulk: float = 1.0
    dt: float = 0.02
    forcing_relaxation: float = 12.0
    warmup_cycles: float = 1.0
    save_cycles: float = 2.5
    frames_per_cycle: int = 18
    pressure_rank_deficiency: int = 0
    diffusion_substeps: int = 1


@dataclass
class ThermalConfig:
    """One-way thermal settings.

    The current demo uses a Gaussian ring source around each cylinder instead of
    solving a full solid conduction problem. This keeps the workflow lightweight
    while still yielding meaningful module-level thermal metrics.
    """

    enabled: bool = False
    pr: float = 0.71
    ambient_temperature: float = 0.0
    source_sigma: float = 0.12
    power_min: float = 0.5
    power_max: float = 1.5
    diffusion_substeps: int = 1


@dataclass
class LayoutConfig:
    """Layout definition for variable-cardinality cylinder configurations."""

    num_cylinders: int = 4
    seed: int = 7
    centers: Optional[List[List[float]]] = None
    heat_powers: Optional[List[float]] = None


@dataclass
class SaveConfig:
    """Output settings."""

    root_dir: str = str(DEFAULT_DATA_DIR)
    case_id: str = "0001"
    tag: str = "multicyl"
    save_pressure: bool = True
    save_temperature: bool = True
    save_vorticity: bool = True
    save_cylinder_mask: bool = True


@dataclass
class ExecutionConfig:
    """Runtime device selection for the simulation backend."""

    device: str = "cpu"  # cpu | gpu
    gpu_id: int = 0


@dataclass
class SimulationConfig:
    """Top-level configuration for one simulation case."""

    mode: str = "inert"  # inert | active
    domain: DomainConfig = field(default_factory=DomainConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def finalize(self) -> "SimulationConfig":
        """Synchronize mode-dependent options and derived defaults."""
        self.mode = self.mode.lower().strip()
        self.thermal.enabled = self.mode == "active"
        self.execution.device = self.execution.device.lower().strip()
        if self.execution.device not in {"cpu", "gpu"}:
            raise ValueError("execution.device must be 'cpu' or 'gpu'.")
        if self.execution.gpu_id < 0:
            raise ValueError("execution.gpu_id must be non-negative.")
        if self.domain.edge_margin < 2.0 * self.domain.cylinder_radius:
            self.domain.edge_margin = 2.0 * self.domain.cylinder_radius
        if self.layout.centers is not None:
            self.layout.num_cylinders = len(self.layout.centers)
        return self


# ----------------------------- Serialization helpers ----------------------------


def config_from_dict(raw: Dict[str, Any]) -> SimulationConfig:
    """Rebuild nested dataclasses from a plain JSON dictionary."""
    cfg = SimulationConfig(
        mode=raw.get("mode", "inert"),
        domain=DomainConfig(**raw.get("domain", {})),
        flow=FlowConfig(**raw.get("flow", {})),
        thermal=ThermalConfig(**raw.get("thermal", {})),
        layout=LayoutConfig(**raw.get("layout", {})),
        save=SaveConfig(**raw.get("save", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
    )
    return cfg.finalize()


def dataclass_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Convert config to a JSON-safe dict with derived runtime metadata."""
    payload = asdict(config)
    runtime = derive_runtime(config)
    payload["runtime"] = runtime
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON using UTF-8 and stable formatting for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def default_data_dir() -> Path:
    """Return the canonical demo data directory under 0_Demo_MultiCylinder."""
    return DEFAULT_DATA_DIR


def default_config_dir() -> Path:
    """Return the canonical config directory under 0_Demo_MultiCylinder."""
    return DEFAULT_CONFIG_DIR


def default_config_backup_dir() -> Path:
    """Return the config backup directory under Configs/."""
    return DEFAULT_CONFIG_BK_DIR


def default_domain_shape_dir() -> Path:
    """Return the domain-shape output directory under 0_Demo_MultiCylinder."""
    return DEFAULT_DOMAIN_SHAPE_DIR


def resolve_data_path(path_like: str | Path | None) -> Path:
    """Resolve a user-facing data path.

    Relative case names are interpreted under the demo's Data_Saved directory.
    Relative paths that already start with `Data_Saved` are interpreted from the
    demo root so they do not become `Data_Saved/Data_Saved/...`.
    """
    if path_like is None:
        return DEFAULT_DATA_DIR

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()

    normalized_parts = [part for part in path.parts if part not in {"", "."}]
    if not normalized_parts:
        return DEFAULT_DATA_DIR.resolve()

    if normalized_parts[0] == DEFAULT_DATA_DIR.name:
        path = DEMO_DIR / Path(*normalized_parts)
    else:
        path = DEFAULT_DATA_DIR / Path(*normalized_parts)

    return path.resolve()


def resolve_config_path(path_like: str | Path) -> Path:
    """Resolve a config path, defaulting relative names to the Configs directory."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = DEFAULT_CONFIG_DIR / path
    return path.resolve()


def backup_config_file(config_path: Path, case_id: str, stamp: str | None = None) -> Path:
    """Copy a config file into Configs/Config_bk with case id and timestamp in the name."""
    timestamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_case_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in case_id)
    backup_name = f"{config_path.stem}_case_{safe_case_id}_{timestamp}{config_path.suffix}"
    backup_path = DEFAULT_CONFIG_BK_DIR / backup_name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, backup_path)
    return backup_path


# ------------------------------- Case management --------------------------------


def make_case_dir(save_cfg: SaveConfig) -> Path:
    """Create a timestamped case directory.

    Example:
        runs/case_0001_20260417_132015_multicyl
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_name = f"case_{save_cfg.case_id}_{stamp}_{save_cfg.tag}"
    case_dir = Path(save_cfg.root_dir).expanduser().resolve() / case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / "scene").mkdir(exist_ok=True)
    (case_dir / "plots").mkdir(exist_ok=True)
    return case_dir


# ------------------------------- Physics estimates ------------------------------


def cylinder_diameter(config: SimulationConfig) -> float:
    """Return the cylinder diameter in physical units."""
    return 2.0 * config.domain.cylinder_radius


def kinematic_viscosity(config: SimulationConfig) -> float:
    """Compute kinematic viscosity from U, D, Re."""
    return config.flow.u_bulk * cylinder_diameter(config) / config.flow.re


def thermal_diffusivity(config: SimulationConfig) -> float:
    """Compute thermal diffusivity alpha = nu / Pr for the active case."""
    return kinematic_viscosity(config) / config.thermal.pr


def estimate_strouhal(re: float) -> float:
    """Approximate 2-D cylinder shedding Strouhal number for moderate Re.

    This empirical estimate is only used to pick a sensible runtime and save
    interval. It does not affect the solver.
    """
    if re <= 47.0:
        return 0.0
    if re < 180.0:
        return max(0.05, 0.212 * (1.0 - 21.2 / re))
    return 0.20


def derive_runtime(config: SimulationConfig) -> Dict[str, Any]:
    """Estimate warm-up time, save time, and frame cadence.

    The intention is to store roughly 2-3 shedding cycles with enough temporal
    resolution for visualization while keeping storage bounded.
    """
    st = estimate_strouhal(config.flow.re)
    diameter = cylinder_diameter(config)
    if st > 0:
        shedding_period = diameter / (st * config.flow.u_bulk)
    else:
        shedding_period = diameter / max(config.flow.u_bulk, 1e-8)
    warmup_time = config.flow.warmup_cycles * shedding_period
    save_time = config.flow.save_cycles * shedding_period
    total_time = warmup_time + save_time
    num_steps = int(math.ceil(total_time / config.flow.dt))
    save_stride = max(1, int(round(shedding_period / (config.flow.frames_per_cycle * config.flow.dt))))
    expected_saved_frames = max(1, int(math.ceil(save_time / (save_stride * config.flow.dt))))
    return {
        "strouhal_estimate": st,
        "shedding_period_estimate": shedding_period,
        "warmup_time": warmup_time,
        "save_time": save_time,
        "total_time": total_time,
        "num_steps": num_steps,
        "save_stride": save_stride,
        "expected_saved_frames": expected_saved_frames,
    }


# ----------------------------- Layout / design helpers ---------------------------


def _pairwise_periodic_distance(a: np.ndarray, b: np.ndarray, lx: float, ly: float) -> float:
    """Minimum-image distance in a periodic rectangle."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, lx - dx)
    dy = min(dy, ly - dy)
    return float(np.hypot(dx, dy))


def sample_cylinder_centers(config: SimulationConfig) -> List[List[float]]:
    """Sample a non-overlapping cylinder layout.

    The sampler uses periodic distance checks so future datasets can place
    modules near periodic boundaries without special-casing overlap detection.
    """
    rng = np.random.default_rng(config.layout.seed)
    r = config.domain.cylinder_radius
    margin = config.domain.edge_margin
    min_center_dist = 2.0 * r + config.domain.min_gap
    xmin, xmax = margin, config.domain.lx - margin
    ymin, ymax = margin, config.domain.ly - margin
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Domain is too small for the requested cylinder radius and edge margin.")

    centers: List[List[float]] = []
    attempts = 0
    max_attempts = 10_000
    while len(centers) < config.layout.num_cylinders:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Failed to sample a valid cylinder layout. Reduce cylinder count, radius, or min_gap."
            )
        candidate = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)
        if all(
            _pairwise_periodic_distance(candidate, np.array(existing), config.domain.lx, config.domain.ly)
            >= min_center_dist
            for existing in centers
        ):
            centers.append(candidate.tolist())
    return centers


def sample_heat_powers(config: SimulationConfig) -> List[float]:
    """Sample per-cylinder heat-source strengths for the active mode."""
    rng = np.random.default_rng(config.layout.seed + 101)
    powers = rng.uniform(
        low=config.thermal.power_min,
        high=config.thermal.power_max,
        size=config.layout.num_cylinders,
    )
    return [float(v) for v in powers]


def materialize_layout(config: SimulationConfig) -> SimulationConfig:
    """Fill in layout coordinates and active powers if not already specified."""
    if config.layout.centers is None:
        config.layout.centers = sample_cylinder_centers(config)
    if config.thermal.enabled and config.layout.heat_powers is None:
        config.layout.heat_powers = sample_heat_powers(config)
    if not config.thermal.enabled:
        config.layout.heat_powers = [0.0 for _ in range(config.layout.num_cylinders)]
    return config


# ----------------------------- Visualization utilities --------------------------


def build_uniform_grid(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-center coordinates for post-processing and plotting."""
    x = np.linspace(0.0, config.domain.lx, config.domain.nx, endpoint=False) + 0.5 * config.domain.lx / config.domain.nx
    y = np.linspace(0.0, config.domain.ly, config.domain.ny, endpoint=False) + 0.5 * config.domain.ly / config.domain.ny
    return np.meshgrid(x, y)


def periodic_offsets(dx: np.ndarray, length: float) -> np.ndarray:
    """Apply minimum-image convention to coordinate differences."""
    return (dx + 0.5 * length) % length - 0.5 * length


def cylinder_mask(config: SimulationConfig, shell_width: float = 0.0) -> np.ndarray:
    """Build a binary mask for cylinders or thin shells on the post-processing grid."""
    xx, yy = build_uniform_grid(config)
    rr = np.zeros_like(xx, dtype=bool)
    radius_inner = config.domain.cylinder_radius
    radius_outer = radius_inner + shell_width
    for cx, cy in config.layout.centers or []:
        dx = periodic_offsets(xx - cx, config.domain.lx)
        dy = periodic_offsets(yy - cy, config.domain.ly)
        dist = np.sqrt(dx * dx + dy * dy)
        if shell_width <= 0:
            rr |= dist <= radius_inner
        else:
            rr |= (dist >= radius_inner) & (dist <= radius_outer)
    return rr
