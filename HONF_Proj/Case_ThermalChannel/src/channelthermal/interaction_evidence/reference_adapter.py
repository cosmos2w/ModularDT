"""Adapters for the local analytic-wake solver and stored baseline targets.

The optional physical path delegates to the repository's local, ignored
``1_Demo_ChannelThermal`` source tree. That workflow is a low-fidelity analytic
flow approximation coupled to a shared-grid thermal update; it is not CFD.
The adapter has no teacher fallback and never modifies archived config files.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .quantities import CHANNEL_ORDER, module_peak_temperatures, pressure_drop_8pct
from .types import (
    DesignState,
    EvidenceSource,
    EvidenceSplit,
    MeasuredQuantity,
    ModuleState,
    OperatingContext,
    PhysicalSolveOutput,
    RoleOutput,
    SolveRecord,
    SolveStatus,
)

SOLVER_SCOPE = (
    "low-fidelity analytic_wake channel flow plus numerically advanced shared-grid "
    "thermal model; not CFD or a Navier-Stokes solve"
)
CONTEXT_FEATURES = (
    "re", "u_in", "nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k",
    "module_radius", "domain_length_x", "domain_length_y",
)
CONTEXT_SCHEMA_NAME = "thermalchannel_inverse_context"
CONTEXT_SCHEMA_VERSION = 1


class ReferenceCapabilityUnavailable(RuntimeError):
    """Raised when the explicitly configured local physical workflow is absent."""


def operating_context_from_config(config: Mapping[str, Any]) -> OperatingContext:
    """Extract the ten maintained NamedContext-v1 fields from a solver config."""

    flow = config["flow"]
    thermal = config["thermal"]
    domain = config["domain"]
    configured_nu = flow.get("nu")
    if configured_nu is None:
        reynolds = max(float(flow["re"]), 1.0e-12)
        configured_nu = (
            float(flow.get("viscosity_scale", 1.0))
            * float(flow["u_in"])
            * (2.0 * float(domain["module_radius"]))
            / reynolds
        )
    values = {
        "re": float(flow["re"]),
        "u_in": float(flow["u_in"]),
        "nu": float(configured_nu),
        "solid_alpha": float(thermal["solid_alpha"]),
        "fluid_alpha": float(thermal["fluid_alpha"]),
        "solid_k": float(thermal["solid_k"]),
        "fluid_k": float(thermal["fluid_k"]),
        "module_radius": float(domain["module_radius"]),
        "domain_length_x": float(domain["lx"]),
        "domain_length_y": float(domain["ly"]),
    }
    return OperatingContext(values)


def operating_context_to_named(context: OperatingContext) -> Any:
    """Convert to the existing inverse ``NamedContext`` without losing names."""

    from honf_inverse_core.contracts import NamedContext

    missing = [name for name in CONTEXT_FEATURES if name not in context.values]
    if missing:
        raise ValueError(f"OperatingContext is missing NamedContext-v1 fields: {missing}.")
    vector = np.asarray([float(context.values[name]) for name in CONTEXT_FEATURES], dtype=np.float32)
    return NamedContext(CONTEXT_FEATURES, vector, CONTEXT_SCHEMA_NAME, CONTEXT_SCHEMA_VERSION)


def design_state_from_physical_design(
    design: Any,
    *,
    anchor_id: str,
    physical_family_id: str,
    split: EvidenceSplit | str,
    module_ids: Sequence[str] | None = None,
) -> DesignState:
    """Preserve padded inverse slot IDs when converting an existing design."""

    centers = np.asarray(design.module_centers, dtype=np.float64)
    present = np.asarray(design.module_present, dtype=bool).reshape(-1)
    heat = np.asarray(design.heat_powers, dtype=np.float64).reshape(-1)
    if centers.shape != (present.size, 2) or heat.shape != present.shape:
        raise ValueError("PhysicalDesign center/presence/heating shapes do not align.")
    if module_ids is None:
        module_ids = tuple(f"{anchor_id}:module:{index}" for index in range(present.size))
    if len(module_ids) != present.size:
        raise ValueError("module_ids must include one explicit ID for each design slot.")
    modules = tuple(
        ModuleState(str(module_ids[index]), (float(centers[index, 0]), float(centers[index, 1])), float(heat[index]), bool(present[index]))
        for index in range(present.size)
    )
    return DesignState(anchor_id, physical_family_id, split, modules)


def design_state_to_physical_design(design: DesignState, *, module_family_id: str = "thermal_disk") -> Any:
    """Convert explicit labeled module slots to the maintained inverse contract."""

    from honf_inverse_core.contracts import PhysicalDesign

    centers = np.asarray([module.position_xy for module in design.modules], dtype=np.float32)
    present = np.asarray([module.active for module in design.modules], dtype=np.float32)
    heat = np.asarray([module.heating for module in design.modules], dtype=np.float32)
    return PhysicalDesign(centers, present, heat, module_family_id=module_family_id)


def read_embedded_case_config(hdf5_path: str | Path, case_id: str) -> tuple[dict[str, Any], str]:
    """Read the exact embedded replay config and stored split for one HDF5 case."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-specific dependency.
        raise RuntimeError("h5py is required to read the stored ThermalChannel case config.") from exc
    path = Path(hdf5_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ThermalChannel dataset not found: {path}")
    with h5py.File(path, "r") as handle:
        key = str(case_id)
        if key not in handle["cases"]:
            raise KeyError(f"Case {key!r} is absent from {path}.")
        group = handle["cases"][key]
        raw = group["case_config_json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        config = json.loads(str(raw))
        split = group.attrs.get("split", "development")
        if isinstance(split, bytes):
            split = split.decode("utf-8")
    return config, str(split)


def load_stored_reference_case(
    hdf5_path: str | Path,
    case_id: str,
    *,
    physical_family_id: str | None = None,
    split_override: EvidenceSplit | str | None = None,
) -> SolveRecord:
    """Load one existing HDF5 solution as stored_reference, never as a solve."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-specific dependency.
        raise RuntimeError("h5py is required to read stored ThermalChannel evidence.") from exc
    path = Path(hdf5_path).expanduser().resolve()
    key = str(case_id)
    if not path.is_file():
        raise FileNotFoundError(f"ThermalChannel dataset not found: {path}")
    with h5py.File(path, "r") as handle:
        group = handle["cases"][key]
        config_raw = group["case_config_json"][()]
        if isinstance(config_raw, bytes):
            config_raw = config_raw.decode("utf-8")
        config = json.loads(str(config_raw))
        stored_split = group.attrs.get("split", "development")
        if isinstance(stored_split, bytes):
            stored_split = stored_split.decode("utf-8")
        stored_label = str(stored_split).lower()
        mapped_split = {
            "train": EvidenceSplit.TRAIN,
            "calibration": EvidenceSplit.CALIBRATION,
            "validation": EvidenceSplit.CALIBRATION,
            "dev": EvidenceSplit.DEVELOPMENT,
            "development": EvidenceSplit.DEVELOPMENT,
            "test": EvidenceSplit.DEVELOPMENT,
        }.get(stored_label, EvidenceSplit.DEVELOPMENT)
        split = EvidenceSplit(split_override) if split_override is not None else mapped_split
        centers = group["module_centers"][...].astype(np.float64)
        heat = group["heat_powers"][...].astype(np.float64)
        active = group["module_present"][...].astype(bool)
        field = group["steady_field"][...].astype(np.float32)
        grid_x = group["x_grid"][...].astype(np.float32)
        grid_y = group["y_grid"][...].astype(np.float32)
        module_mask = group["module_mask"][...].astype(bool)
        interface = group["interface_target"][...].astype(np.float32)
        port_mask = group["interface_condition_valid_mask"][...].astype(bool)
        internal = group["module_internal_temperature"][...].astype(np.float32)
        internal_grid_mask = group["module_internal_mask"][...].astype(bool)
        group_attrs = {str(k): _json_scalar(v) for k, v in group.attrs.items()}
        material = {str(k): float(v) for k, v in group["material_parameters"].attrs.items()}
    family_id = physical_family_id or f"stored_case:{key}"
    modules = tuple(
        ModuleState(f"{key}:module:{slot}", tuple(map(float, centers[slot])), float(heat[slot]), bool(active[slot]))
        for slot in range(centers.shape[0])
    )
    design = DesignState(key, family_id, split, modules)
    context = operating_context_from_config(config)
    active_slots = np.flatnonzero(active)
    active_ids = tuple(modules[index].module_id for index in active_slots)
    ny, nx = grid_x.shape
    grid_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    grid_valid = (~module_mask).reshape(-1)
    field_role = RoleOutput(
        role="fluid_fields",
        query_features=grid_xy,
        values=field.reshape(-1, field.shape[-1]),
        channel_names=CHANNEL_ORDER,
        channel_units=("dataset velocity units", "dataset velocity units", "dataset pressure units", "dataset vorticity units", "dataset temperature units"),
        valid_mask=np.broadcast_to(grid_valid[:, None], (grid_xy.shape[0], field.shape[-1])),
        quadrature_weights=np.full(grid_xy.shape[0], (float(config["domain"]["lx"]) / nx) * (float(config["domain"]["ly"]) / ny)),
        query_ids=tuple(f"grid:{j}:{i}" for j in range(ny) for i in range(nx)),
        coordinate_kind="eulerian",
    )
    port_count = interface.shape[1]
    theta = np.linspace(0.0, 2.0 * np.pi, port_count, endpoint=False, dtype=np.float64)
    port_features = np.stack([theta, np.cos(theta), np.sin(theta)], axis=-1)
    active_interface = interface[active_slots].reshape(-1, 2)
    active_port_mask = port_mask[active_slots].reshape(-1, 1)
    active_port_mask = np.broadcast_to(active_port_mask, active_interface.shape)
    port_ids = tuple(modules[index].module_id for index in active_slots for _ in range(port_count))
    radius = float(config["domain"]["module_radius"])
    interface_role = RoleOutput(
        role="interface",
        query_features=np.tile(port_features, (len(active_slots), 1)),
        values=active_interface,
        channel_names=("T_surface", "q_normal"),
        channel_units=("dataset temperature units", "dataset heat-flux proxy units"),
        valid_mask=active_port_mask & np.isfinite(active_interface),
        quadrature_weights=np.full(len(active_slots) * port_count, radius * 2.0 * np.pi / port_count),
        query_ids=tuple(f"{module_id}:port:{port}" for module_id in active_ids for port in range(port_count)),
        receiver_module_ids=port_ids,
        coordinate_kind="interface_material_angle",
    )
    local_size = internal_grid_mask.shape[0]
    axis = np.linspace(-1.0, 1.0, local_size, dtype=np.float64)
    local_x, local_y = np.meshgrid(axis, axis)
    local_xy = np.stack([local_x[internal_grid_mask], local_y[internal_grid_mask]], axis=-1)
    internal_samples = internal[active_slots][:, internal_grid_mask].reshape(-1, 1)
    solid_ids = tuple(module_id for module_id in active_ids for _ in range(local_xy.shape[0]))
    local_area = (radius**2) * (4.0 / max(local_size - 1, 1) ** 2)
    solid_role = RoleOutput(
        role="solid_temperature",
        query_features=np.tile(local_xy, (len(active_slots), 1)),
        values=internal_samples,
        channel_names=("temperature",),
        channel_units=("dataset temperature units",),
        valid_mask=np.isfinite(internal_samples),
        quadrature_weights=np.full(internal_samples.shape[0], local_area),
        query_ids=tuple(f"{module_id}:solid:{sample}" for module_id in active_ids for sample in range(local_xy.shape[0])),
        receiver_module_ids=solid_ids,
        coordinate_kind="solid_material_normalized_xy",
    )
    peaks = module_peak_temperatures(solid_role.values, solid_ids, solid_role.valid_mask)
    fluid_2d = ~module_mask
    drop, counts = pressure_drop_8pct(field[..., 2], grid_x, fluid_2d, float(config["domain"]["lx"]))
    quantities = {
        "pressure_drop": MeasuredQuantity(
            drop,
            "dataset pressure units",
            metadata={"definition": "mean pressure on fluid x<=0.08Lx minus fluid x>=0.92Lx", **counts},
        ),
        "internal_temperature_max": MeasuredQuantity(max(peaks.values()), "dataset temperature units"),
    }
    output = PhysicalSolveOutput(
        roles={"fluid_fields": field_role, "interface": interface_role, "solid_temperature": solid_role},
        quantities=quantities,
        active_module_ids=active_ids,
        module_peak_temperature=peaks,
        units_metadata={
            "position": "dataset length units",
            "pressure": "dataset pressure units; no SI metadata identified",
            "temperature": "dataset temperature units; no SI metadata identified",
            "interface_flux": "dataset heat-flux proxy units; no SI metadata identified",
        },
        case_dir=str(group_attrs.get("source_case_dir", "")) or None,
    )
    converged = bool(group_attrs.get("converged", False))
    status = SolveStatus.CONVERGED if converged else SolveStatus.UNCONVERGED
    provenance = {
        "source": EvidenceSource.STORED_REFERENCE.value,
        "dataset_path": str(path),
        "case_id": key,
        "stored_split": str(stored_split),
        "split_assignment": split.value,
        "source_case_dir": str(group_attrs.get("source_case_dir", "")),
        "target_mode": str(group_attrs.get("target_mode", "unknown")),
        "group_attributes": group_attrs,
        "material_parameters": material,
        "embedded_config_path": f"{path}::/cases/{key}/case_config_json",
        "fidelity_scope": "stored dataset target only; not a new solve at a perturbed design",
    }
    return SolveRecord(
        record_id=f"stored:{key}",
        design=design,
        context=context,
        source=EvidenceSource.STORED_REFERENCE,
        status=status,
        elapsed_seconds=0.0,
        provenance=provenance,
        output=output,
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class AnalyticWakeReferenceAdapter:
    """Run the explicit local NumPy reference workflow without archive mutation."""

    def __init__(
        self,
        *,
        demo_root: str | Path,
        output_root: str | Path,
        case_template: Mapping[str, Any],
    ) -> None:
        self.demo_root = Path(demo_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.case_template = copy.deepcopy(dict(case_template))
        source = self.demo_root / "src" / "simulate_channelthermal.py"
        common = self.demo_root / "src" / "_helpers_forward" / "channelthermal_common.py"
        if not source.is_file() or not common.is_file():
            raise ReferenceCapabilityUnavailable(
                "The local analytic-wake reference source is unavailable. Expected explicit local dependency at "
                f"{source}; no surrogate fallback is provided."
            )
        self.solver_path = source
        self.common_path = common
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._solver = None

    @classmethod
    def from_hdf5_case(
        cls,
        *,
        demo_root: str | Path,
        output_root: str | Path,
        hdf5_path: str | Path,
        case_id: str,
    ) -> tuple[AnalyticWakeReferenceAdapter, OperatingContext]:
        config, _ = read_embedded_case_config(hdf5_path, case_id)
        adapter = cls(demo_root=demo_root, output_root=output_root, case_template=config)
        return adapter, operating_context_from_config(config)

    def solve(
        self,
        design: DesignState,
        context: OperatingContext,
        *,
        record_id: str,
        baseline_design: DesignState | None = None,
        physical_step_scales: Mapping[str, float] | None = None,
    ) -> SolveRecord:
        """Execute one fixed-context design and preserve the raw solver output."""

        step_scales = {str(name): float(value) for name, value in (physical_step_scales or {}).items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in step_scales.values()):
            raise ValueError("Physical step scales must be positive and finite before a solve starts.")
        module = self._load_solver()
        raw_config = copy.deepcopy(self.case_template)
        _apply_context(raw_config, context.values)
        active = design.active_modules
        raw_config.setdefault("layout", {})["num_modules"] = len(active)
        raw_config["layout"]["centers"] = [list(module.position_xy) for module in active]
        raw_config["layout"]["heat_powers"] = [float(module.heating) for module in active]
        raw_config.setdefault("save", {})["root_dir"] = str(self.output_root)
        raw_config["save"]["case_id"] = _safe_name(record_id)
        raw_config["save"]["tag"] = "reference_solver"
        raw_config.setdefault("execution", {})["device"] = "cpu"
        raw_config["execution"]["gpu_id"] = 2  # metadata only; this solver executes NumPy on CPU.
        start = time.perf_counter()
        case_dir: Path | None = None
        try:
            cfg = module.config_from_dict(raw_config)
            cfg = module.materialize_layout(cfg.finalize())
            previous_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                # The legacy entry point changes this variable even though its kernels are CPU NumPy.
                # Restore it immediately after the call so a caller's GPU-2 process state is retained.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    case_dir = Path(module.run_case(cfg))
            finally:
                if previous_visibility is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_visibility
            elapsed = time.perf_counter() - start
            output, converged, runtime = _read_solver_output(
                case_dir, design, cfg, grid_builder=module.build_uniform_grid
            )
            status = SolveStatus.CONVERGED if converged else SolveStatus.UNCONVERGED
            provenance = self._provenance(design, context, raw_config, case_dir, runtime)
            deltas = _design_delta(baseline_design, design) if baseline_design is not None else {}
            return SolveRecord(
                record_id=record_id,
                design=design,
                context=context,
                source=EvidenceSource.REFERENCE_SOLVER,
                status=status,
                elapsed_seconds=elapsed,
                provenance=provenance,
                output=output,
                perturbation_by_module=deltas,
                physical_step_scales=step_scales,
            )
        except ReferenceCapabilityUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve unexpected solver/runtime failures as typed records.
            elapsed = time.perf_counter() - start
            provenance = self._provenance(design, context, raw_config, case_dir, {})
            provenance["failure_type"] = type(exc).__name__
            provenance["failure_reason"] = str(exc)
            return SolveRecord(
                record_id=record_id,
                design=design,
                context=context,
                source=EvidenceSource.REFERENCE_SOLVER,
                status=SolveStatus.FAILED,
                elapsed_seconds=elapsed,
                provenance=provenance,
                output=None,
                perturbation_by_module=_design_delta(baseline_design, design) if baseline_design is not None else {},
                physical_step_scales=step_scales,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

    def load_record(
        self,
        case_dir: str | Path,
        design: DesignState,
        context: OperatingContext,
        *,
        record_id: str | None = None,
        elapsed_seconds: float = 0.0,
        baseline_design: DesignState | None = None,
        physical_step_scales: Mapping[str, float] | None = None,
    ) -> SolveRecord:
        """Rebuild a typed reference record from an existing raw solver case."""

        return load_solve_record(
            case_dir,
            design,
            context,
            solver_module=self._load_solver(),
            solver_path=self.solver_path,
            common_path=self.common_path,
            record_id=record_id,
            elapsed_seconds=elapsed_seconds,
            baseline_design=baseline_design,
            physical_step_scales=physical_step_scales,
        )

    def _load_solver(self) -> Any:
        if self._solver is not None:
            return self._solver
        if not self.solver_path.is_file():
            raise ReferenceCapabilityUnavailable(
                f"Local analytic-wake solver was removed or is unavailable: {self.solver_path}"
            )
        source_root = self.demo_root / "src"
        helpers_root = source_root / "_helpers_forward"
        for path in (str(source_root), str(helpers_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        module_name = "channelthermal_local_reference_solver"
        existing = sys.modules.get(module_name)
        if existing is not None:
            self._solver = existing
            return existing
        spec = importlib.util.spec_from_file_location(module_name, self.solver_path)
        if spec is None or spec.loader is None:
            raise ReferenceCapabilityUnavailable(f"Cannot load local solver source: {self.solver_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        self._solver = module
        return module

    def _provenance(
        self,
        design: DesignState,
        context: OperatingContext,
        raw_config: Mapping[str, Any],
        case_dir: Path | None,
        runtime: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "source": EvidenceSource.REFERENCE_SOLVER.value,
            "solver_scope": SOLVER_SCOPE,
            "solver_source_path": str(self.solver_path),
            "common_source_path": str(self.common_path),
            "actual_device": "NumPy CPU",
            "gpu_used": False,
            "case_dir": str(case_dir) if case_dir else None,
            "physical_family_id": design.physical_family_id,
            "anchor_id": design.anchor_id,
            "split": EvidenceSplit(design.split).value,
            "context": {name: float(context.values[name]) for name in CONTEXT_FEATURES if name in context.values},
            "runtime": dict(runtime),
            "solver_controls": {
                section: dict(raw_config.get(section, {}))
                for section in ("domain", "flow", "thermal", "local_module", "execution")
            },
            "solver_source_version": "local ignored source tree; no tracked revision identifier available",
            "units_note": "Dataset units are used; no SI conversion metadata was found.",
            "physical_assumptions": (
                "analytic wake channel flow; deterministic steady velocity/pressure; optional projection is not Navier-Stokes; "
                "one shared thermal grid with fluid advection/diffusion, solid diffusion/heating, and approximate interface coupling"
            ),
        }


def _apply_context(config: dict[str, Any], values: Mapping[str, Any]) -> None:
    """Map NamedContext-v1 fields to the exact legacy solver config keys."""

    sections = {"domain", "flow", "thermal", "local_module", "execution", "save"}
    for section in sections:
        incoming = values.get(section)
        if isinstance(incoming, Mapping):
            config.setdefault(section, {}).update(copy.deepcopy(dict(incoming)))
    mapping = {
        "re": ("flow", "re"), "u_in": ("flow", "u_in"), "nu": ("flow", "nu"),
        "solid_alpha": ("thermal", "solid_alpha"), "fluid_alpha": ("thermal", "fluid_alpha"),
        "solid_k": ("thermal", "solid_k"), "fluid_k": ("thermal", "fluid_k"),
        "module_radius": ("domain", "module_radius"),
        "domain_length_x": ("domain", "lx"), "domain_length_y": ("domain", "ly"),
    }
    for input_name, (section, output_name) in mapping.items():
        if input_name in values:
            config.setdefault(section, {})[output_name] = float(values[input_name])


def _read_solver_output(
    case_dir: Path,
    design: DesignState,
    cfg: Any,
    *,
    grid_builder: Any,
) -> tuple[PhysicalSolveOutput, bool, dict[str, Any]]:
    with (case_dir / "frame_index.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Reference solver saved no frames: {case_dir}")
    last = rows[-1]
    with np.load(case_dir / "scene" / last["file"], allow_pickle=False) as payload:
        frame = {key: payload[key].copy() for key in payload.files}
    config_payload = json.loads((case_dir / "case_config.json").read_text(encoding="utf-8"))
    runtime = dict(config_payload.get("runtime", {}))
    runtime["last_saved_step"] = int(last.get("step", -1))
    runtime["last_saved_time"] = float(last.get("time", "nan"))
    nx, ny = int(cfg.domain.nx), int(cfg.domain.ny)
    lx, ly = float(cfg.domain.lx), float(cfg.domain.ly)
    x_grid, y_grid = grid_builder(cfg)
    grid_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1)
    fluid_2d = frame["module_mask"] == 0
    fluid_valid = fluid_2d.reshape(-1)
    fields = np.stack([frame[name] for name in CHANNEL_ORDER], axis=-1).astype(np.float32)
    field_values = fields.reshape(-1, len(CHANNEL_ORDER))
    cell_area = (lx / nx) * (ly / ny)
    grid_ids = tuple(f"grid:{j}:{i}" for j in range(ny) for i in range(nx))
    field_role = RoleOutput(
        role="fluid_fields",
        query_features=grid_xy,
        values=field_values,
        channel_names=CHANNEL_ORDER,
        channel_units=("dataset velocity units", "dataset velocity units", "dataset pressure units", "dataset vorticity units", "dataset temperature units"),
        valid_mask=np.broadcast_to(fluid_valid[:, None], field_values.shape),
        quadrature_weights=np.full(grid_xy.shape[0], cell_area),
        query_ids=grid_ids,
        coordinate_kind="eulerian",
    )
    active_modules = design.active_modules
    active_ids = tuple(module.module_id for module in active_modules)
    radius = float(cfg.domain.module_radius)
    theta = np.asarray(frame["interface_response"][0, :, 0], dtype=np.float64)
    normals_x = np.asarray(frame["interface_response"][0, :, 1], dtype=np.float64)
    normals_y = np.asarray(frame["interface_response"][0, :, 2], dtype=np.float64)
    port_count = int(theta.size)
    interface_response = frame["interface_response"][:len(active_modules)]
    interface_values = interface_response[..., [3, 5]].reshape(-1, 2).astype(np.float32)
    interface_features = np.tile(np.stack([theta, normals_x, normals_y], axis=-1), (len(active_modules), 1))
    dx, dy = lx / nx, ly / ny
    delta = min(dx, dy, 0.15 * radius)
    centers = np.asarray([module.position_xy for module in active_modules], dtype=np.float64)
    surface_x = centers[:, None, 0] + radius * normals_x[None, :]
    surface_y = centers[:, None, 1] + radius * normals_y[None, :]
    outside_x = centers[:, None, 0] + (radius + delta) * normals_x[None, :]
    outside_y = centers[:, None, 1] + (radius + delta) * normals_y[None, :]
    sample_min_x, sample_max_x = 0.5 * dx, lx - 0.5 * dx
    sample_min_y, sample_max_y = 0.5 * dy, ly - 0.5 * dy
    valid_surface = (
        (surface_x >= sample_min_x) & (surface_x <= sample_max_x)
        & (surface_y >= sample_min_y) & (surface_y <= sample_max_y)
    )
    valid_outside = (
        (outside_x >= sample_min_x) & (outside_x <= sample_max_x)
        & (outside_y >= sample_min_y) & (outside_y <= sample_max_y)
    )
    interface_mask = np.stack([valid_surface, valid_surface & valid_outside], axis=-1).reshape(-1, 2)
    interface_mask &= np.isfinite(interface_values)
    interface_receiver_ids = tuple(module_id for module_id in active_ids for _ in range(port_count))
    interface_role = RoleOutput(
        role="interface",
        query_features=interface_features,
        values=interface_values,
        channel_names=("T_surface", "q_normal"),
        channel_units=("dataset temperature units", "dataset heat-flux proxy units"),
        valid_mask=interface_mask,
        quadrature_weights=np.full(len(active_modules) * port_count, radius * 2.0 * np.pi / port_count),
        query_ids=tuple(f"{module_id}:port:{port}" for module_id in active_ids for port in range(port_count)),
        receiver_module_ids=interface_receiver_ids,
        coordinate_kind="interface_material_angle",
    )
    internal_grid = frame["module_internal_temperature"][:len(active_modules)]
    local_mask = frame["module_internal_mask"].astype(bool)
    local_size_y, local_size_x = local_mask.shape
    local_axis_x = np.linspace(-1.0, 1.0, local_size_x, dtype=np.float64)
    local_axis_y = np.linspace(-1.0, 1.0, local_size_y, dtype=np.float64)
    local_x, local_y = np.meshgrid(local_axis_x, local_axis_y)
    local_xy_one = np.stack([local_x[local_mask], local_y[local_mask]], axis=-1)
    local_xy = np.tile(local_xy_one, (len(active_modules), 1))
    solid_values = internal_grid[:, local_mask].reshape(-1, 1).astype(np.float32)
    solid_mask_by_module = []
    for module in active_modules:
        world_x = module.position_xy[0] + radius * local_xy_one[:, 0]
        world_y = module.position_xy[1] + radius * local_xy_one[:, 1]
        valid = (
            (world_x >= sample_min_x) & (world_x <= sample_max_x)
            & (world_y >= sample_min_y) & (world_y <= sample_max_y)
        )
        solid_mask_by_module.append(valid)
    solid_valid = np.concatenate(solid_mask_by_module)[:, None] & np.isfinite(solid_values)
    solid_ids = tuple(module_id for module_id in active_ids for _ in range(local_xy_one.shape[0]))
    local_area = radius**2 * (4.0 / max(local_size_x - 1, 1) ** 2)
    solid_role = RoleOutput(
        role="solid_temperature",
        query_features=local_xy,
        values=solid_values,
        channel_names=("temperature",),
        channel_units=("dataset temperature units",),
        valid_mask=solid_valid,
        quadrature_weights=np.full(solid_values.shape[0], local_area),
        query_ids=tuple(f"{module_id}:solid:{sample}" for module_id in active_ids for sample in range(local_xy_one.shape[0])),
        receiver_module_ids=solid_ids,
        coordinate_kind="solid_material_normalized_xy",
    )
    peaks = module_peak_temperatures(solid_values, solid_ids, solid_valid)
    drop, counts = pressure_drop_8pct(fields[..., 2], x_grid, fluid_2d, lx)
    quantities = {
        "pressure_drop": MeasuredQuantity(
            drop,
            "dataset pressure units",
            metadata={"definition": "mean pressure on fluid x<=0.08Lx minus fluid x>=0.92Lx", **counts},
        ),
        "internal_temperature_max": MeasuredQuantity(max(peaks.values()), "dataset temperature units"),
    }
    output = PhysicalSolveOutput(
        roles={"fluid_fields": field_role, "interface": interface_role, "solid_temperature": solid_role},
        quantities=quantities,
        active_module_ids=active_ids,
        module_peak_temperature=peaks,
        units_metadata={
            "position": "dataset length units",
            "pressure": "dataset pressure units; no SI metadata identified",
            "temperature": "dataset temperature units; no SI metadata identified",
            "interface_flux": "dataset heat-flux proxy units; no SI metadata identified",
        },
        case_dir=str(case_dir),
    )
    return output, bool(runtime.get("converged", False)), runtime


def load_solve_record(
    case_dir: str | Path,
    design: DesignState,
    context: OperatingContext,
    *,
    solver_module: Any,
    solver_path: str | Path,
    common_path: str | Path,
    record_id: str | None = None,
    elapsed_seconds: float = 0.0,
    baseline_design: DesignState | None = None,
    physical_step_scales: Mapping[str, float] | None = None,
) -> SolveRecord:
    """Reload one completed raw analytic-wake case without running it again."""

    step_scales = {str(name): float(value) for name, value in (physical_step_scales or {}).items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in step_scales.values()):
        raise ValueError("Physical step scales must be positive and finite.")
    path = Path(case_dir).expanduser().resolve()
    config_path = path / "case_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Raw reference case config not found: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = solver_module.config_from_dict(raw_config)
    cfg = solver_module.materialize_layout(cfg.finalize())
    output, converged, runtime = _read_solver_output(
        path, design, cfg, grid_builder=solver_module.build_uniform_grid
    )
    provenance = {
        "source": EvidenceSource.REFERENCE_SOLVER.value,
        "solver_scope": SOLVER_SCOPE,
        "solver_source_path": str(Path(solver_path).resolve()),
        "common_source_path": str(Path(common_path).resolve()),
        "actual_device": "NumPy CPU",
        "gpu_used": False,
        "case_dir": str(path),
        "physical_family_id": design.physical_family_id,
        "anchor_id": design.anchor_id,
        "split": EvidenceSplit(design.split).value,
        "context": {name: float(context.values[name]) for name in CONTEXT_FEATURES if name in context.values},
        "runtime": runtime,
        "solver_controls": {
            section: dict(raw_config.get(section, {}))
            for section in ("domain", "flow", "thermal", "local_module", "execution")
        },
        "solver_source_version": "local ignored source tree; no tracked revision identifier available",
        "units_note": "Dataset units are used; no SI conversion metadata was found.",
    }
    return SolveRecord(
        record_id=record_id or path.name,
        design=design,
        context=context,
        source=EvidenceSource.REFERENCE_SOLVER,
        status=SolveStatus.CONVERGED if converged else SolveStatus.UNCONVERGED,
        elapsed_seconds=elapsed_seconds,
        provenance=provenance,
        output=output,
        perturbation_by_module=_design_delta(baseline_design, design) if baseline_design is not None else {},
        physical_step_scales=step_scales,
    )


def _design_delta(baseline: DesignState | None, trial: DesignState) -> dict[str, tuple[float, ...]]:
    if baseline is None:
        return {}
    if baseline.active_module_ids != trial.active_module_ids:
        raise ValueError("Baseline and trial module IDs must match for an anchored response.")
    before = {module.module_id: module for module in baseline.active_modules}
    changes = {}
    for module in trial.active_modules:
        old = before[module.module_id]
        changes[module.module_id] = (
            float(module.position_xy[0] - old.position_xy[0]),
            float(module.position_xy[1] - old.position_xy[1]),
            float(module.heating - old.heating),
        )
    return changes


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    if not safe:
        raise ValueError("record_id must contain at least one filesystem-safe character.")
    return safe[:96]


__all__ = [
    "CONTEXT_FEATURES",
    "SOLVER_SCOPE",
    "AnalyticWakeReferenceAdapter",
    "ReferenceCapabilityUnavailable",
    "design_state_from_physical_design",
    "design_state_to_physical_design",
    "load_solve_record",
    "load_stored_reference_case",
    "operating_context_from_config",
    "operating_context_to_named",
    "read_embedded_case_config",
]
