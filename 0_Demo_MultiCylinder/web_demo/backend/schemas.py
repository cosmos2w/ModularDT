from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Cylinder(BaseModel):
    x: float
    y: float


class GenerativeOptions(BaseModel):
    num_samples: int = Field(default=4, ge=1)
    n_steps: int = Field(default=16, ge=1)
    seed: Optional[int] = None
    noise_mode: str = "harmonic"


class DesignRequest(BaseModel):
    model_id: str
    mode: Literal["deterministic", "generative"]
    re: float
    cylinders: List[Cylinder]
    phase_bins: int = Field(default=36, ge=1, le=512)
    resolution_nx: int = Field(default=192, ge=8, le=2048)
    resolution_ny: int = Field(default=96, ge=8, le=2048)
    field: str = "omega"
    display_smoothing: bool = True
    display_scale: int = Field(default=3, ge=1, le=8)
    render_interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bicubic"
    return_hypergraph: bool = True
    return_kpis: bool = True
    generative: GenerativeOptions = Field(default_factory=GenerativeOptions)


class ValidationResult(BaseModel):
    valid: bool
    warnings: List[str]
    max_num_cylinders: int
    domain_length_x: float
    domain_length_y: float
    requested_phase_bins: int = 36
    effective_phase_bins: int = 36
    max_phase_bins: int = 36
    phase_bin_policy: Literal["cap", "reject"] = "cap"


class InferenceResponse(BaseModel):
    job_id: str
    status: str
    result_url: str


class ModelAvailability(BaseModel):
    id: str
    label: str
    mode: str
    enabled: bool
    available: bool
    preload: bool = False
    stage: Optional[int] = None
    run_dir: str
    checkpoint_path: str
    config_path: str
    checkpoint_exists: bool = False
    config_exists: bool = False
    missing_files: List[str] = Field(default_factory=list)
    reason_unavailable: Optional[str] = None
    note: Optional[str] = None
    status: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KpiTargetSpec(BaseModel):
    enabled: bool = True
    name: str
    mode: Literal["exact", "range", "max", "min", "minimize", "maximize"] = "exact"
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    weight: float = Field(default=1.0, ge=0.0)


class InverseConstraintSpec(BaseModel):
    re: float
    num_cylinders_min: int = Field(default=1, ge=1)
    num_cylinders_max: int = Field(default=8, ge=1)
    min_center_distance: float = Field(default=1.1, ge=0.0)
    min_x_span: Optional[float] = Field(default=None, ge=0.0)
    min_y_span: Optional[float] = Field(default=None, ge=0.0)


class InverseSamplingSpec(BaseModel):
    n_samples: int = Field(default=64, ge=1)
    verify_top_k: int = Field(default=16, ge=0)
    save_verified_top_k: int = Field(default=4, ge=0)
    n_steps: int = Field(default=32, ge=1)
    seed: Optional[int] = None


class InverseVerificationSpec(BaseModel):
    forward_verifier_model_id: str
    forward_backend: Literal["deterministic", "generative"] = "deterministic"
    phase_bins: int = Field(default=12, ge=1, le=512)
    nx: int = Field(default=96, ge=8, le=2048)
    ny: int = Field(default=48, ge=8, le=2048)
    generative_num_samples: int = Field(default=4, ge=1)
    generative_n_steps: int = Field(default=16, ge=1)
    generative_ode_solver: Literal["euler", "heun"] = "heun"
    uncertainty_penalty_weight: float = Field(default=0.05, ge=0.0)


class InverseRunRequest(BaseModel):
    inverse_model_id: str
    target_name: Optional[str] = None
    kpis: List[KpiTargetSpec]
    constraints: InverseConstraintSpec
    sampling: InverseSamplingSpec = Field(default_factory=InverseSamplingSpec)
    verification: InverseVerificationSpec
    simulation_enabled: bool = False


class InverseRunResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    result_url: str


class InverseCandidate(BaseModel):
    rank: Optional[int] = None
    sample_index: int
    score: Optional[float] = None
    centers: List[List[float]]
    count: int
    validity: Dict[str, Any] = Field(default_factory=dict)
    kpis: Dict[str, Any] = Field(default_factory=dict)
    kpi_comparison: Dict[str, Any] = Field(default_factory=dict)
    frame_urls: Optional[Dict[str, List[str]]] = None
    hypergraph: Optional[Dict[str, Any]] = None
    quick_validation_status: str = "not_started"
    simulation_validation_status: str = "not_started"


class CandidateQuickValidationRequest(BaseModel):
    verification: Optional[InverseVerificationSpec] = None


class CandidateSimulationValidationRequest(BaseModel):
    simulation_mode: Literal["inert", "active"] = "inert"
    simulation_device: Optional[Literal["cpu", "gpu"]] = None
    simulation_gpu_id: Optional[int] = None
    simulation_preprocess_device: Optional[str] = None
    simulation_nx: Optional[int] = Field(default=None, ge=8)
    simulation_ny: Optional[int] = Field(default=None, ge=8)
    simulation_phase_bins: Optional[int] = Field(default=None, ge=1)
    simulation_warmup_cycles: Optional[float] = Field(default=None, ge=0.0)
    simulation_save_cycles: Optional[float] = Field(default=None, ge=0.0)
    simulation_frames_per_cycle: Optional[int] = Field(default=None, ge=1)
    simulation_dt: Optional[float] = Field(default=None, gt=0.0)
