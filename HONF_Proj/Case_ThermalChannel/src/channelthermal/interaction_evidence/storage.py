"""Typed physical-record persistence with separate serialization status.

A successful physical solve remains successful if artifact serialization fails.
The returned result retains the original in-memory record and raw solver path so
callers can recover metadata or retry persistence without running the solver.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

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

SerializationStatus = Literal["stored", "serialization_failed"]


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return value


def _role_array_key(index: int, name: str, field: str) -> str:
    return f"role_{index}_{name}_{field}"


def _write_record_directory(directory: Path, record: SolveRecord) -> None:
    if record.output is None:
        arrays: dict[str, np.ndarray] = {}
        output_payload = None
    else:
        arrays = {}
        role_payload = []
        for index, (name, role) in enumerate(record.output.roles.items()):
            entries = {
                "query_features": role.query_features,
                "values": role.values,
                "valid_mask": role.valid_mask,
                "quadrature_weights": role.quadrature_weights,
            }
            if role.noise_floor is not None:
                entries["noise_floor"] = role.noise_floor
            role_payload.append({
                "role": name,
                "channel_names": list(role.channel_names),
                "channel_units": list(role.channel_units),
                "query_ids": list(role.query_ids),
                "receiver_module_ids": None if role.receiver_module_ids is None else list(role.receiver_module_ids),
                "coordinate_kind": role.coordinate_kind,
                "arrays": {field: _role_array_key(index, name, field) for field in entries},
            })
            arrays.update({
                _role_array_key(index, name, field): np.asarray(value)
                for field, value in entries.items()
            })
        output_payload = {
            "roles": role_payload,
            "quantities": {
                name: {
                    "value": quantity.value,
                    "units": quantity.units,
                    "resolved": quantity.resolved,
                    "metadata": _json_value(quantity.metadata),
                }
                for name, quantity in record.output.quantities.items()
            },
            "active_module_ids": list(record.output.active_module_ids),
            "module_peak_temperature": dict(record.output.module_peak_temperature),
            "units_metadata": _json_value(record.output.units_metadata),
            "case_dir": record.output.case_dir,
        }
    if arrays:
        np.savez_compressed(directory / "physical_output.npz", **arrays)
    metadata = {
        "schema_version": 1,
        "record_id": record.record_id,
        "source": record.source.value,
        "status": record.status.value,
        "elapsed_seconds": record.elapsed_seconds,
        "design": {
            "anchor_id": record.design.anchor_id,
            "physical_family_id": record.design.physical_family_id,
            "split": record.design.split.value,
            "modules": [
                {
                    "module_id": module.module_id,
                    "position_xy": list(module.position_xy),
                    "heating": module.heating,
                    "active": module.active,
                }
                for module in record.design.modules
            ],
        },
        "context": _json_value(record.context.values),
        "provenance": _json_value(record.provenance),
        "perturbation_by_module": _json_value(record.perturbation_by_module),
        "physical_step_scales": _json_value(record.physical_step_scales),
        "failure_reason": record.failure_reason,
        "output": output_payload,
    }
    with (directory / "record.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class RecordPersistenceResult:
    """Serialization outcome while preserving physical-attempt provenance."""

    serialization_status: SerializationStatus
    record: SolveRecord
    destination: Path
    error: str | None
    solver_invoked: bool | None
    raw_solver_completed: bool | None
    physical_wall_time_available: bool | None
    raw_case_dir: str | None
    physical_attempt_count: int | None

    @property
    def solve_status(self) -> SolveStatus:
        return self.record.status


def persist_solve_record(
    record: SolveRecord,
    destination: str | Path,
    *,
    writer: Callable[[Path, SolveRecord], None] = _write_record_directory,
) -> RecordPersistenceResult:
    """Atomically persist one record and return serializer errors as a distinct state.

    The function never calls a reference adapter or solver. On serialization
    failure the original record, its processed arrays, and the raw solver
    ``case_dir`` remain attached to the result for recovery without a new call.
    """

    final = Path(destination).expanduser().resolve()
    provenance = record.provenance
    raw_case_dir = (
        record.output.case_dir if record.output is not None and record.output.case_dir
        else provenance.get("raw_case_dir") or provenance.get("case_dir")
    )
    solver_invoked = provenance.get("solver_invoked")
    raw_completed = provenance.get("raw_solver_completed")
    wall_available = provenance.get("physical_wall_time_available")
    attempt_count = 1 if solver_invoked is True else 0 if solver_invoked is False else None
    status: SerializationStatus
    error: str | None = None
    temporary: Path | None = None
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(f"Persistence destination already exists: {final}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent))
        writer(temporary, record)
        os.replace(temporary, final)
        temporary = None
        status = "stored"
    except Exception as exc:  # noqa: BLE001 - serialization failure must preserve solve success
        status = "serialization_failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return RecordPersistenceResult(
        serialization_status=status,
        record=record,
        destination=final,
        error=error,
        solver_invoked=solver_invoked if isinstance(solver_invoked, bool) else None,
        raw_solver_completed=raw_completed if isinstance(raw_completed, bool) else None,
        physical_wall_time_available=wall_available if isinstance(wall_available, bool) else None,
        raw_case_dir=None if raw_case_dir is None else str(raw_case_dir),
        physical_attempt_count=attempt_count,
    )


def load_solve_record(directory: str | Path) -> SolveRecord:
    """Read one persisted typed record without invoking any physical solver."""

    root = Path(directory).expanduser().resolve()
    metadata = json.loads((root / "record.json").read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported persisted solve-record schema version.")
    design_data = metadata["design"]
    design = DesignState(
        anchor_id=design_data["anchor_id"],
        physical_family_id=design_data["physical_family_id"],
        split=EvidenceSplit(design_data["split"]),
        modules=tuple(ModuleState(
            module_id=item["module_id"],
            position_xy=tuple(item["position_xy"]),
            heating=float(item["heating"]),
            active=bool(item["active"]),
        ) for item in design_data["modules"]),
    )
    output_data = metadata["output"]
    output = None
    if output_data is not None:
        with np.load(root / "physical_output.npz", allow_pickle=False) as archive:
            roles = {}
            for role_data in output_data["roles"]:
                arrays = {
                    name: np.array(archive[key], copy=True)
                    for name, key in role_data["arrays"].items()
                }
                roles[role_data["role"]] = RoleOutput(
                    role=role_data["role"],
                    query_features=arrays["query_features"],
                    values=arrays["values"],
                    channel_names=tuple(role_data["channel_names"]),
                    channel_units=tuple(role_data["channel_units"]),
                    valid_mask=arrays["valid_mask"],
                    quadrature_weights=arrays["quadrature_weights"],
                    query_ids=tuple(role_data["query_ids"]),
                    receiver_module_ids=(
                        None if role_data["receiver_module_ids"] is None
                        else tuple(role_data["receiver_module_ids"])
                    ),
                    coordinate_kind=role_data["coordinate_kind"],
                    noise_floor=arrays.get("noise_floor"),
                )
        output = PhysicalSolveOutput(
            roles=roles,
            quantities={
                name: MeasuredQuantity(
                    value=float(item["value"]),
                    units=item["units"],
                    resolved=bool(item["resolved"]),
                    metadata=item.get("metadata", {}),
                )
                for name, item in output_data["quantities"].items()
            },
            active_module_ids=tuple(output_data["active_module_ids"]),
            module_peak_temperature=output_data["module_peak_temperature"],
            units_metadata=output_data["units_metadata"],
            case_dir=output_data.get("case_dir"),
        )
    return SolveRecord(
        record_id=metadata["record_id"],
        design=design,
        context=OperatingContext(metadata["context"]),
        source=EvidenceSource(metadata["source"]),
        status=SolveStatus(metadata["status"]),
        elapsed_seconds=float(metadata["elapsed_seconds"]),
        provenance=metadata["provenance"],
        output=output,
        perturbation_by_module=metadata.get("perturbation_by_module", {}),
        physical_step_scales=metadata.get("physical_step_scales", {}),
        failure_reason=metadata.get("failure_reason"),
    )


__all__ = [
    "RecordPersistenceResult",
    "SerializationStatus",
    "load_solve_record",
    "persist_solve_record",
]
