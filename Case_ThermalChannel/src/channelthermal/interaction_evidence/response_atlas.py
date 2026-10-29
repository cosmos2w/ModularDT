"""Load the compact physical-response-atlas NPZ format as typed stencils.

The atlas is an existing, ignored local artifact. This adapter never invokes
the reference solver; it reconstructs immutable :class:`ResponseStencil`
objects from the saved role arrays and their adjacent JSON metadata.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .response_dataset import ResponseStencil
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


_ROLE_PREFIX = {
    "fluid_fields": "fluid_fields",
    "interface": "interface",
    "solid_temperature": "solid_temperature",
}


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _context(metadata: dict[str, Any]) -> OperatingContext:
    config = metadata.get("case_config", {})
    domain = config.get("domain", {})
    flow = config.get("flow", {})
    thermal = config.get("thermal", {})
    runtime = metadata.get("runtime", {})
    reynolds = float(flow["re"])
    inflow_speed = float(flow["u_in"])
    radius = float(domain["module_radius"])
    if not all(math.isfinite(value) for value in (reynolds, inflow_speed, radius)):
        raise ValueError("Thermal atlas Re, inlet speed, and module radius must be finite.")
    if reynolds <= 0.0 or inflow_speed <= 0.0 or radius <= 0.0:
        raise ValueError("Thermal atlas Re, inlet speed, and module radius must be positive.")
    configured_nu = flow.get("nu")
    runtime_nu = runtime.get("nu")
    if configured_nu is None:
        calculated_nu = (
            float(flow.get("viscosity_scale", 1.0))
            * inflow_speed
            * (2.0 * radius)
            / reynolds
        )
        if runtime_nu is not None:
            resolved_nu = float(runtime_nu)
            if not math.isclose(
                calculated_nu, resolved_nu, rel_tol=1.0e-6, abs_tol=1.0e-12
            ):
                raise ValueError(
                    f"Derived nu={calculated_nu} disagrees with recorded runtime nu={runtime_nu}."
                )
        else:
            resolved_nu = calculated_nu
    else:
        resolved_nu = float(configured_nu)
        if runtime_nu is not None and not math.isclose(
            resolved_nu, float(runtime_nu), rel_tol=1.0e-6, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Configured nu={resolved_nu} disagrees with recorded runtime nu={runtime_nu}."
            )
    values = {
        "re": reynolds,
        "u_in": inflow_speed,
        "nu": resolved_nu,
        "solid_alpha": float(thermal["solid_alpha"]),
        "fluid_alpha": float(thermal["fluid_alpha"]),
        "solid_k": float(thermal["solid_k"]),
        "fluid_k": float(thermal["fluid_k"]),
        "module_radius": radius,
        "domain_length_x": float(domain["lx"]),
        "domain_length_y": float(domain["ly"]),
    }
    invalid = [name for name, value in values.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"Thermal atlas operating context has non-finite fields: {invalid}.")
    if resolved_nu <= 0.0:
        raise ValueError("Thermal atlas viscosity must be positive.")
    return OperatingContext(values)


def _design(
    label: str,
    centers: np.ndarray,
    heating: np.ndarray,
    module_ids: tuple[str, ...],
    anchor_id: str,
    family_id: str,
    split: EvidenceSplit,
) -> DesignState:
    if centers.shape != (len(module_ids), 2) or heating.shape != (len(module_ids),):
        raise ValueError(f"Atlas design {label!r} does not align with active module IDs.")
    modules = tuple(
        ModuleState(
            module_id=module_id,
            position_xy=(float(centers[index, 0]), float(centers[index, 1])),
            heating=float(heating[index]),
            active=True,
        )
        for index, module_id in enumerate(module_ids)
    )
    return DesignState(anchor_id, family_id, split, modules)


def _role_output(
    arrays: Any,
    role: str,
    row: int,
    module_ids: tuple[str, ...],
) -> RoleOutput:
    prefix = _ROLE_PREFIX[role]
    values = np.asarray(arrays[f"{prefix}_values"][row])
    query_features = np.asarray(arrays[f"{prefix}_query_features"])
    channel_names = tuple(str(value) for value in arrays[f"{prefix}_channel_names"])
    channel_units = tuple(str(value) for value in arrays[f"{prefix}_channel_units"])
    query_ids = tuple(str(value) for value in arrays[f"{prefix}_query_ids"])
    receiver_array = np.asarray(arrays[f"{prefix}_receiver_module_ids"])
    receiver_ids = tuple(str(value) for value in receiver_array) if receiver_array.size else None
    coordinate_kind = {
        "fluid_fields": "eulerian",
        "interface": "interface_material_angle",
        "solid_temperature": "solid_material_normalized_xy",
    }[role]
    if receiver_ids is not None and any(value not in module_ids for value in receiver_ids):
        raise ValueError(f"Atlas role {role!r} refers to an unknown active module.")
    return RoleOutput(
        role=role,
        query_features=query_features,
        values=values,
        channel_names=channel_names,
        channel_units=channel_units,
        valid_mask=np.asarray(arrays[f"{prefix}_valid_mask"][row], dtype=bool),
        # Atlas samples are uniform on each Eulerian, interface, and local
        # material grid. No quadrature weights are stored in this format.
        quadrature_weights=np.ones(query_features.shape[0], dtype=np.float64),
        query_ids=query_ids,
        receiver_module_ids=receiver_ids,
        coordinate_kind=coordinate_kind,
    )


def load_response_atlas_stencil(npz_path: str | Path) -> tuple[ResponseStencil, dict[str, Any]]:
    """Load one family NPZ plus its same-stem JSON companion.

    The returned metadata includes the complete case configuration for
    provenance. Only the compact numeric operating context is placed on the
    typed records and exposed to a target-free operator callback.
    """

    source_path = Path(npz_path)
    metadata_path = source_path.with_suffix(".json")
    if not source_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Atlas NPZ/JSON pair is incomplete: {source_path}.")
    metadata = _read_json(metadata_path)
    with np.load(source_path, allow_pickle=False) as arrays:
        labels = tuple(str(value) for value in arrays["all_labels"])
        if labels.count("baseline") != 1:
            raise ValueError(f"Atlas family {source_path} must contain exactly one baseline.")
        baseline_index = labels.index("baseline")
        module_ids = tuple(str(value) for value in arrays["active_module_ids"])
        family_id = str(metadata.get("family_id", ""))
        anchor_id = str(metadata.get("anchor_id", ""))
        family_metadata = json.loads(_scalar_text(arrays["family_metadata_json"]))
        context = _context(family_metadata)
        source_values = np.asarray(arrays["source"])
        split_values = np.asarray(arrays["split"])
        centers_values = np.asarray(arrays["module_centers_xy"])
        heating_values = np.asarray(arrays["heating"])
        elapsed_values = np.asarray(arrays["elapsed_seconds"])
        elapsed_measured = np.asarray(arrays["elapsed_seconds_measured"], dtype=bool)
        pressures = np.asarray(arrays["pressure_drop"], dtype=np.float64)
        peaks = np.asarray(arrays["module_peak_temperature"], dtype=np.float64)

        def record_at(index: int, label: str) -> SolveRecord:
            split = EvidenceSplit(str(split_values[index]))
            source = EvidenceSource(str(source_values[index]))
            design = _design(
                label,
                centers_values[index],
                heating_values[index],
                module_ids,
                anchor_id,
                family_id,
                split,
            )
            if peaks[index].shape != (len(module_ids),):
                raise ValueError(f"Atlas module peaks are not aligned for {label!r}.")
            roles = {
                role: _role_output(arrays, role, index, module_ids)
                for role in _ROLE_PREFIX
            }
            pressure_units = roles["fluid_fields"].channel_units[
                roles["fluid_fields"].channel_names.index("p")
            ]
            record_metadata = metadata.get("records", {}).get(label, {})
            output = PhysicalSolveOutput(
                roles=roles,
                quantities={
                    "pressure_drop": MeasuredQuantity(
                        value=float(pressures[index]), units=pressure_units
                    )
                },
                active_module_ids=module_ids,
                module_peak_temperature={
                    module_id: float(peaks[index, module_index])
                    for module_index, module_id in enumerate(module_ids)
                },
                units_metadata={
                    f"{role}.{channel}": unit
                    for role, output_role in roles.items()
                    for channel, unit in zip(output_role.channel_names, output_role.channel_units)
                },
                case_dir=record_metadata.get("case_dir"),
            )
            deltas = {
                module_id: (
                    float(centers_values[index, module_index, 0] - centers_values[baseline_index, module_index, 0]),
                    float(centers_values[index, module_index, 1] - centers_values[baseline_index, module_index, 1]),
                    float(heating_values[index, module_index] - heating_values[baseline_index, module_index]),
                )
                for module_index, module_id in enumerate(module_ids)
            }
            scales = {
                str(name): float(value)
                for name, value in record_metadata.get("physical_step_scales", {}).items()
                if float(value) > 0.0
            }
            return SolveRecord(
                record_id=f"{family_id}:{label}",
                design=design,
                context=context,
                source=source,
                status=SolveStatus.CONVERGED,
                elapsed_seconds=max(0.0, float(elapsed_values[index])),
                provenance={
                    "source": source.value,
                    "split": split.value,
                    "physical_family_id": family_id,
                    "anchor_id": anchor_id,
                    "atlas_npz": str(source_path),
                    "atlas_json": str(metadata_path),
                    "elapsed_seconds_measured": bool(elapsed_measured[index]),
                    "physical_assumptions": family_metadata.get("physical_assumptions", []),
                },
                output=output,
                perturbation_by_module=deltas,
                physical_step_scales=scales,
            )

        baseline = record_at(baseline_index, "baseline")
        variants = {
            label: record_at(index, label)
            for index, label in enumerate(labels)
            if label != "baseline"
        }
    return ResponseStencil(baseline=baseline, variants=variants), metadata


__all__ = ["load_response_atlas_stencil"]
