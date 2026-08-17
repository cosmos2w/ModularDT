"""CORE HONF dataclasses and configuration.

Inputs are generic module centers/features, global context, optional query
time, query coordinates, and optional generic environment coordinates/features.
Outputs are configuration and batch containers consumed by the reusable HONF
core. This module is reusable across domains and contains no ChannelThermal
wall, inlet, outlet, or material assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
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
    "hyper_plus_global_near",
    "hyper_plus_global_direct",
    "hyper_plus_near_direct",
    "no_hyper_global_near",
    "no_hyper_current_like_direct",
    "current_like",
    "enhanced_honf_pairwise",
    "enhanced_honf_pairwise_only",
}

DECODER_COMPONENTS = {
    "hyper_only": {"hyper"},
    "hyper_plus_global": {"hyper", "global"},
    "hyper_plus_direct_residual": {"hyper", "direct"},
    "hyper_plus_near_module": {"hyper", "near"},
    "hyper_plus_global_near": {"hyper", "global", "near"},
    "hyper_plus_global_direct": {"hyper", "global", "direct"},
    "hyper_plus_near_direct": {"hyper", "near", "direct"},
    "no_hyper_global_near": {"global", "near"},
    "no_hyper_current_like_direct": {"global", "near", "direct"},
    "current_like": {"global", "near", "direct"},
    "enhanced_honf_pairwise": {"hyper", "pairwise", "global", "near"},
    "enhanced_honf_pairwise_only": {"hyper", "pairwise", "global", "near"},
}

# Accepted only when loading historical configs/checkpoints and never
# serialized by the cleaned configuration. Decoder mode supersedes the old
# component flags; runtime batch tensors supersede the old module-count cap.
LEGACY_IGNORED_CORE_KEYS = {
    "max_num_modules",
    "use_hyper_context",
    "use_hypergraph_gated_pairwise_kernel",
    "use_direct_module_env_decoder",
    "use_near_module_context",
    "use_global_context",
    "use_dynamic_tokens",
    "use_local_surrogate_patch",
}


@dataclass
class UnifiedForwardConfig:
    """Configuration for the minimal unified hypergraph neural field."""

    field_dim: int = 5
    domain_length_x: float = 12.0
    domain_length_y: float = 4.0
    module_radius: float = 0.45
    coordinate_scale: Optional[list[float]] = None
    periodic_axes: Optional[list[int]] = None
    local_context_scale: Optional[float] = None

    num_env_tokens_x: int = 16
    num_env_tokens_y: int = 6
    num_hyperedges: int = 4
    hidden_dim: int = 128
    dropout: float = 0.05
    use_layer_norm: bool = True

    geometry_mode: str = "nonperiodic"
    query_time_mode: str = "none"

    decoder_mode: str = "hyper_only"
    use_hyper_value_context: bool = True
    query_fourier_frequencies: int = 4
    boundary_feature_mode: str = "rectangular"
    position_fourier_frequencies: int = 2
    use_position_fourier_for_modules: bool = True
    use_position_fourier_for_env: bool = True
    use_hyper_mechanism_encoder: bool = True
    mechanism_include_geometry: bool = True
    mechanism_include_masses: bool = True
    mechanism_hidden_dim: Optional[int] = None
    hyper_module_assignment_mode: str = "learned"
    hyper_query_attention_mode: str = "learned"
    hyper_attention_topk: int = 0
    hyper_attention_temperature: float = 1.0
    sparse_hyper_attention_detach_mask: bool = True
    pairwise_kernel_hidden_dim: Optional[int] = None
    pairwise_kernel_num_layers: int = 3
    pairwise_kernel_gate_init: float = 0.10
    pairwise_kernel_use_fourier: bool = True
    pairwise_kernel_fourier_frequencies: int = 2
    pairwise_kernel_include_module_token: bool = True
    pairwise_kernel_include_module_features: bool = True
    pairwise_kernel_normalize_by_edge_mass: bool = True
    use_hyper_geometry_bias: bool = True
    hyper_geometry_bias_scale: float = 1.0
    direct_residual_gate_init: float = 0.0
    use_A_me_auxiliary: bool = True
    output_mean_residual_split: bool = False

    def __post_init__(self) -> None:
        """Validate mode names and numerical routing constraints."""

        if self.geometry_mode not in {"nonperiodic", "periodic"}:
            raise ValueError("geometry_mode must be 'nonperiodic' or 'periodic'.")
        if self.query_time_mode not in {"none", "phase", "physical_time"}:
            raise ValueError("query_time_mode must be 'none', 'phase', or 'physical_time'.")
        if self.boundary_feature_mode not in {"none", "rectangular", "channel"}:
            raise ValueError("boundary_feature_mode must be 'none' or 'rectangular' ('channel' is a legacy alias).")
        if self.local_context_scale is not None and float(self.local_context_scale) <= 0.0:
            raise ValueError("local_context_scale must be positive when provided.")
        if self.coordinate_scale is not None:
            if len(self.coordinate_scale) != 2 or any(float(value) <= 0.0 for value in self.coordinate_scale):
                raise ValueError("coordinate_scale must contain two positive values for the current 2-D core.")
        if self.periodic_axes is not None:
            axes = [int(value) for value in self.periodic_axes]
            if len(set(axes)) != len(axes) or any(value not in {0, 1} for value in axes):
                raise ValueError("periodic_axes must contain unique axis indices from {0,1}.")
        if self.hyper_module_assignment_mode not in {"learned", "uniform"}:
            raise ValueError("hyper_module_assignment_mode must be 'learned' or 'uniform'.")
        if self.hyper_query_attention_mode not in {"learned", "uniform"}:
            raise ValueError("hyper_query_attention_mode must be 'learned' or 'uniform'.")
        if int(self.hyper_attention_topk) < 0:
            raise ValueError("hyper_attention_topk must be >= 0.")
        if float(self.hyper_attention_temperature) <= 0:
            raise ValueError("hyper_attention_temperature must be > 0.")
        if self.decoder_mode not in DECODER_MODES:
            allowed = ", ".join(sorted(DECODER_MODES))
            raise ValueError(f"decoder_mode must be one of: {allowed}")
        if self.decoder_mode == "enhanced_honf_pairwise_only":
            self.use_hyper_value_context = False

    def spatial_scale(self) -> tuple[float, float]:
        """Return neutral 2-D coordinate scales with legacy length fallback."""

        if self.coordinate_scale is not None:
            return float(self.coordinate_scale[0]), float(self.coordinate_scale[1])
        return float(self.domain_length_x), float(self.domain_length_y)

    def periodic_dimensions(self) -> tuple[int, ...]:
        """Return periodic axes, interpreting legacy periodic mode as all axes."""

        if self.periodic_axes is not None:
            return tuple(sorted(int(value) for value in self.periodic_axes))
        return (0, 1) if self.geometry_mode == "periodic" else ()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UnifiedForwardConfig":
        """Construct a strict core configuration from a mapping."""

        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this core configuration to plain Python values."""

        return _to_plain_dict(self)

    def decoder_uses(self, component: str) -> bool:
        """Report whether ``decoder_mode`` enables a named context component."""

        return str(component) in DECODER_COMPONENTS[self.decoder_mode]


@dataclass
class BatchData:
    """Canonical one-batch data container consumed by the HONF model."""

    module_centers: Any
    module_present: Any
    module_features: Any
    global_context: Any
    query_xy: Any
    query_time: Optional[Any]
    target_field: Optional[Any]
    case_name: str
    metadata: Dict[str, Any]
    env_coords: Optional[Any] = None
    env_features: Optional[Any] = None
    query_features: Optional[Any] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BatchData":
        """Validate and construct the canonical batch container."""

        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        """Describe batch values as serializable metadata."""

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
    """Build a strict config dataclass while allowing underscore note fields."""

    names = {item.name for item in fields(cls)}
    unknown = sorted(
        key
        for key in payload
        if key not in names and key not in LEGACY_IGNORED_CORE_KEYS and not str(key).startswith("_")
    )
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {unknown}")
    filtered = {key: value for key, value in dict(payload).items() if key in names}
    return cls(**filtered)


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    """Recursively convert dataclasses and tensors to JSON-safe descriptions."""

    def convert(obj: Any) -> Any:
        """Convert one nested value to its plain representation."""

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
