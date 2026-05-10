from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

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
    design_source: Literal["reference_case", "diy", "candidate"] = "diy"
    re: Optional[float] = None
    u_in: Optional[float] = None
    modules: List[ThermalModule]
    field: str = "temperature"
    display_scale: int = Field(default=3, ge=1, le=8)
    display_smoothing: bool = True
    render_interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bicubic"
    return_kpis: bool = True
    return_organization: bool = True
    return_ground_truth: bool = False
    return_error: bool = False


class ForwardSimulationRequest(BaseModel):
    design: DesignRequest
    prediction_job_id: Optional[str] = None
    max_runtime_seconds: int = Field(default=900, ge=10, le=7200)


class ForwardSimulationResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    result_url: str


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


class BoxSpec(BaseModel):
    x: Tuple[float, float]
    y: Tuple[float, float]


class HeatLoadSpec(BaseModel):
    mode: Literal[
        "per_module",
        "per_module_range",
        "uniform",
        "uniform_range",
        "total_only",
        "from_reference",
        "none",
    ] = "from_reference"
    values: Optional[List[float]] = None
    ranges: Optional[List[Tuple[float, float]]] = None
    value: Optional[float] = None
    range: Optional[Tuple[float, float]] = None
    total: Optional[float] = None
    sort_mode: Literal["heat_desc_then_xy", "slot_order", "anonymous"] = "heat_desc_then_xy"


class StructureConstraintSpec(BaseModel):
    enabled: bool = False
    strength: float = Field(default=0.0, ge=0.0)
    x_span: Optional[Tuple[float, float]] = None
    y_span: Optional[Tuple[float, float]] = None
    min_x_coverage: Optional[float] = None
    min_y_coverage: Optional[float] = None
    min_mean_pair_distance: Optional[float] = None
    centroid: Optional[Tuple[float, float]] = None
    centroid_tolerance: Optional[Tuple[float, float]] = None
    avoid_vertical_stack: bool = False
    keepout_boxes: List[BoxSpec] = Field(default_factory=list)
    protected_boxes: List[BoxSpec] = Field(default_factory=list)
    preferred_boxes: List[BoxSpec] = Field(default_factory=list)
    sketch_maps: Optional[Dict[str, List[List[float]]]] = None


class ThermalLimitSpec(BaseModel):
    solid_temperature_max: Optional[float] = None
    module_temperature_spread_max: Optional[float] = None
    pressure_drop_max: Optional[float] = None
    wall_hot_delta_T: Optional[float] = None
    outlet_hot_delta_T: Optional[float] = None


class ObjectiveWeightsSpec(BaseModel):
    safety: float = Field(default=1.0, ge=0.0)
    uniformity: float = Field(default=0.8, ge=0.0)
    pressure: float = Field(default=0.4, ge=0.0)
    outlet_mixing: float = Field(default=0.5, ge=0.0)
    wall_protection: float = Field(default=0.4, ge=0.0)
    plume_avoidance: float = Field(default=0.6, ge=0.0)
    coverage: float = Field(default=0.3, ge=0.0)


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
    target_mode: Literal["design_intent", "legacy_kpi"] = "legacy_kpi"
    kpis: List[KpiTargetSpec] = Field(default_factory=list)
    constraints: InverseConstraintSpec = Field(default_factory=InverseConstraintSpec)
    heat_loads: HeatLoadSpec = Field(default_factory=HeatLoadSpec)
    structure_constraints: StructureConstraintSpec = Field(default_factory=StructureConstraintSpec)
    thermal_limits: ThermalLimitSpec = Field(default_factory=ThermalLimitSpec)
    objective_weights: ObjectiveWeightsSpec = Field(default_factory=ObjectiveWeightsSpec)
    field_preferences: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    sampling: InverseSamplingSpec = Field(default_factory=InverseSamplingSpec)
    guidance_scale: Optional[float] = None
    diversity_rerank_weight: Optional[float] = None
    candidate_pool_multiplier: Optional[float] = None
    reference_split: str = "test"
    reference_case_index: int = Field(default=0, ge=0)


class InverseRunResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    result_url: str
