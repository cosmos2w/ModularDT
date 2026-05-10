from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ThermalModule(BaseModel):
    x: float
    y: float
    heat_power: float = Field(default=1.0, ge=0.0)


class DesignRequest(BaseModel):
    model_id: str
    reference_split: str = "test"
    reference_case_index: int = Field(default=0, ge=0)
    reference_case_id: Optional[str] = None
    re: Optional[float] = None
    u_in: Optional[float] = None
    modules: List[ThermalModule]
    field: str = "temperature"
    display_scale: int = Field(default=3, ge=1, le=8)
    display_smoothing: bool = True
    render_interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bicubic"
    return_kpis: bool = True
    return_organization: bool = True


class ValidationResult(BaseModel):
    valid: bool
    warnings: List[str]
    max_num_modules: int
    domain_length_x: float
    domain_length_y: float
    module_radius: float
    min_center_distance: float
    heat_power_min: float
    heat_power_max: float
    total_heat_power: float


class InferenceResponse(BaseModel):
    job_id: str
    status: str
    result_url: str


class ModelAvailability(BaseModel):
    id: str
    label: str
    enabled: bool
    available: bool
    run_dir: str
    checkpoint_path: str
    config_path: str
    checkpoint_exists: bool
    config_exists: bool
    missing_files: List[str] = Field(default_factory=list)
    reason_unavailable: Optional[str] = None
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KpiTargetSpec(BaseModel):
    enabled: bool = True
    name: str
    mode: Literal["exact", "range", "max", "min"] = "max"
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    weight: float = Field(default=1.0, ge=0.0)


class InverseConstraintSpec(BaseModel):
    num_modules_min: int = Field(default=3, ge=1)
    num_modules_max: int = Field(default=8, ge=1)
    min_center_distance: float = Field(default=1.1, ge=0.0)
    wall_clearance: float = Field(default=0.08, ge=0.0)
    inlet_clearance: float = Field(default=0.30, ge=0.0)
    outlet_clearance: float = Field(default=0.30, ge=0.0)
    heat_power_total: Optional[float] = Field(default=None, ge=0.0)


class InverseSamplingSpec(BaseModel):
    n_samples: int = Field(default=32, ge=1)
    n_steps: int = Field(default=4, ge=1)
    seed: int = 123
    count_mode: Literal["uniform", "sample", "argmax"] = "uniform"


class InverseRunRequest(BaseModel):
    inverse_model_id: str
    forward_model_id: str
    target_name: Optional[str] = None
    kpis: List[KpiTargetSpec]
    constraints: InverseConstraintSpec
    preferences: Dict[str, Any] = Field(default_factory=dict)
    sampling: InverseSamplingSpec = Field(default_factory=InverseSamplingSpec)
    reference_split: str = "test"
    reference_case_index: int = Field(default=0, ge=0)


class InverseRunResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    result_url: str
