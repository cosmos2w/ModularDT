"""Stable Thermal reference configuration contract and source defaults.

These dataclasses and their finalization mirror the dataset-generating solver's
configuration schema. They do not import or execute the legacy solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / '1_Demo_ChannelThermal' / 'Data_Saved'


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
