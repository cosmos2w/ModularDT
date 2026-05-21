"""Shared dataclasses for the unified forward-model sandbox.

The types in this module are intentionally small and dependency-light. They
define the ablation surface used by the sandbox without importing either demo
folder or copying any dataset artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover - only used for type flexibility.
    torch = None  # type: ignore


DECODER_MODES = {
    "hyper_only",
    "hyper_plus_global",
    "hyper_plus_direct_residual",
    "hyper_plus_near_module",
    "current_like",
}


@dataclass
class UnifiedForwardConfig:
    """Configuration for the minimal unified hypergraph neural field."""

    field_dim: int = 5
    max_num_modules: int = 12
    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45

    num_env_tokens_x: int = 16
    num_env_tokens_y: int = 6
    num_hyperedges: int = 4
    hidden_dim: int = 128
    dropout: float = 0.05
    use_layer_norm: bool = True

    geometry_mode: str = "nonperiodic"
    query_time_mode: str = "none"

    decoder_mode: str = "hyper_only"
    use_hyper_geometry_bias: bool = True
    hyper_geometry_bias_scale: float = 1.0
    direct_residual_gate_init: float = 0.0
    use_A_me_auxiliary: bool = True
    use_direct_module_env_decoder: bool = False
    use_near_module_context: bool = False
    use_global_context: bool = True

    output_mean_residual_split: bool = False
    use_dynamic_tokens: bool = False
    use_local_surrogate_patch: bool = False

    def __post_init__(self) -> None:
        if self.geometry_mode not in {"nonperiodic", "periodic"}:
            raise ValueError("geometry_mode must be 'nonperiodic' or 'periodic'.")
        if self.query_time_mode not in {"none", "phase", "physical_time"}:
            raise ValueError("query_time_mode must be 'none', 'phase', or 'physical_time'.")
        if self.decoder_mode not in DECODER_MODES:
            allowed = ", ".join(sorted(DECODER_MODES))
            raise ValueError(f"decoder_mode must be one of: {allowed}")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UnifiedForwardConfig":
        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)


@dataclass
class CaseConfig:
    """Dataset and case-selection settings for a single forward-model case."""

    case_name: str = "synthetic"
    dataset_path: Optional[str] = None
    batch_size: int = 1
    points_per_case: int = 256
    allow_synthetic_fallback: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CaseConfig":
        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)


@dataclass
class AblationConfig:
    """One row of the forward-model ablation ladder."""

    name: str = "hyper_only"
    decoder_mode: str = "hyper_only"
    use_A_me_auxiliary: bool = True
    use_direct_module_env_decoder: bool = False
    use_near_module_context: bool = False
    use_global_context: bool = True
    num_hyperedges: int = 4
    notes: str = ""

    def __post_init__(self) -> None:
        if self.decoder_mode not in DECODER_MODES:
            allowed = ", ".join(sorted(DECODER_MODES))
            raise ValueError(f"decoder_mode must be one of: {allowed}")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AblationConfig":
        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)


@dataclass
class BatchData:
    """Canonical one-batch data container consumed by the unified model."""

    module_centers: Any
    module_present: Any
    module_features: Any
    global_context: Any
    query_xy: Any
    query_time: Optional[Any]
    target_field: Optional[Any]
    case_name: str
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BatchData":
        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)

    def to(self, device: Any) -> "BatchData":
        """Move tensor fields to a device and return a new BatchData object."""
        if torch is None:
            return self
        payload: Dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = value.to(device) if torch.is_tensor(value) else value
        return BatchData(**payload)


def _dataclass_from_dict(cls: Any, payload: Dict[str, Any]) -> Any:
    names = {item.name for item in fields(cls)}
    filtered = {key: value for key, value in dict(payload).items() if key in names}
    return cls(**filtered)


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    def convert(obj: Any) -> Any:
        if torch is not None and torch.is_tensor(obj):
            return {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }
        if is_dataclass(obj):
            return {key: convert(val) for key, val in asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(key): convert(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(val) for val in obj]
        return obj

    return convert(value)
