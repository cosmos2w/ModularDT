"""Typed physical, teacher, and synthetic interaction-response evidence."""

from .quantities import field_response_norms, module_peak_temperatures, pressure_drop_8pct, smooth_peak_temperature
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
    "RoleOutput",
    "SolveRecord",
    "SolveStatus",
    "attach_role_noise_floors",
    "estimate_tightening_response_floor",
    "field_response_norms",
    "module_peak_temperatures",
    "pressure_drop_8pct",
    "smooth_peak_temperature",
    "validate_family_splits",
]
