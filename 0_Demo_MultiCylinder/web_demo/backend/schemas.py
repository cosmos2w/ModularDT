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
