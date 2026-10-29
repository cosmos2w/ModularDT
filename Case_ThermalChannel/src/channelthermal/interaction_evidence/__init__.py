"""Typed physical, teacher, and synthetic interaction-response evidence."""

from .quantities import field_response_norms, module_peak_temperatures, pressure_drop_8pct, smooth_peak_temperature
from .storage import RecordPersistenceResult, load_solve_record, persist_solve_record
from .response_dataset import (
    ResponseBlock,
    ResponseExample,
    ResponseStencil,
    attach_role_noise_floors,
    estimate_tightening_response_floor,
    validate_family_splits,
)
from .types import (
    DesignState,
    EvidenceSource,
    EvidenceSplit,
    InteractionEvidenceRecord,
    MeasuredQuantity,
    ModuleState,
    OperatingContext,
    PhysicalSolveOutput,
    RoleOutput,
    SolveRecord,
    SolveStatus,
)
from .response_atlas import load_response_atlas_stencil

__all__ = [
    "DesignState",
    "EvidenceSource",
    "EvidenceSplit",
    "InteractionEvidenceRecord",
    "MeasuredQuantity",
    "ModuleState",
    "OperatingContext",
    "PhysicalSolveOutput",
    "ResponseBlock",
    "ResponseExample",
    "ResponseStencil",
    "RecordPersistenceResult",
    "RoleOutput",
    "SolveRecord",
    "SolveStatus",
    "attach_role_noise_floors",
    "estimate_tightening_response_floor",
    "field_response_norms",
    "load_solve_record",
    "load_response_atlas_stencil",
    "module_peak_temperatures",
    "persist_solve_record",
    "pressure_drop_8pct",
    "smooth_peak_temperature",
    "validate_family_splits",
]
