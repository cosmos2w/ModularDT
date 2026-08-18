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


_FORWARD_MODE_DEFAULTS: Dict[str, Any] = {
    "organizer_mode": "fixed_projection",
    "mechanism_state_mode": "residual_concat",
    "field_assembly_mode": "context_fusion",
    "module_assignment_normalizer": "softmax",
    "environment_assignment_normalizer": "softmax",
    "query_assignment_normalizer": "softmax",
    "routing_execution": "dense",
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
    organizer_mode: str = "fixed_projection"
    edge_capacity: int = 0
    initial_active_edges: int = 6
    minimum_active_edges: int = 1
    slot_refinement_steps: int = 2
    slot_code_mode: str = "sinusoidal"

    edge_selection_mode: str = "all"
    selection_warmup_epochs: int = 200
    selection_start_epoch: int = -1
    selection_transition_epochs: int = 0
    selection_warmup_mode: str = "legacy"
    selection_minimum_module_mass_fraction: float = 0.01
    selection_minimum_environment_mass_fraction: float = 0.01
    selection_coverage_rate: float = 0.95
    selection_token_threshold: float = 0.50
    selection_maximum_redundancy: float = 0.85
    candidate_module_mass_fraction_floor: float = 0.01
    candidate_environment_mass_fraction_floor: float = 0.01

    module_assignment_normalizer: str = "softmax"
    environment_assignment_normalizer: str = "softmax"
    query_assignment_normalizer: str = "softmax"
    module_sparsity_start_epoch: int = -1
    module_sparsity_transition_epochs: int = 0
    environment_sparsity_start_epoch: int = -1
    environment_sparsity_transition_epochs: int = 0
    query_sparsity_start_epoch: int = -1
    query_sparsity_transition_epochs: int = 0
    entmax_alpha: float = 1.5

    environment_locality_mode: str = "none"
    environment_locality_strength: float = 1.0
    query_locality_mode: str = "inherit_environment"
    locality_radius_cap: float = 3.0
    minimum_region_scale: float = 0.05

    mechanism_state_mode: str = "residual_concat"
    mechanism_latent_residual_scale: float = 0.35

    field_assembly_mode: str = "context_fusion"
    additive_edge_gate_init: float = 0.10
    additive_output_init_std: float = 1.0e-3
    routing_execution: str = "dense"
    gathered_execution_start_epoch: int = -1
    query_edge_limit: int = 0
    query_module_limit: int = 0
    query_edge_retained_mass_floor: float = 0.0
    module_incidence_retained_mass_floor: float = 0.0

    topology_signature_enabled: bool = False
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

        if self.organizer_mode not in {"fixed_projection", "exchangeable_slots"}:
            raise ValueError("organizer_mode must be 'fixed_projection' or 'exchangeable_slots'.")
        if self.organizer_mode == "fixed_projection" and int(self.num_hyperedges) <= 0:
            raise ValueError("fixed_projection organizer_mode requires num_hyperedges > 0.")
        if self.organizer_mode == "exchangeable_slots":
            if int(self.edge_capacity) <= 0:
                raise ValueError("exchangeable_slots organizer_mode requires edge_capacity > 0.")
            if int(self.initial_active_edges) <= 0 or int(self.initial_active_edges) > int(self.edge_capacity):
                raise ValueError("initial_active_edges must be in [1, edge_capacity].")
            if int(self.minimum_active_edges) <= 0 or int(self.minimum_active_edges) > int(self.initial_active_edges):
                raise ValueError("minimum_active_edges must be in [1, initial_active_edges].")
        if int(self.edge_capacity) < 0:
            raise ValueError("edge_capacity must be >= 0.")
        if int(self.slot_refinement_steps) <= 0:
            raise ValueError("slot_refinement_steps must be positive.")
        if self.slot_code_mode not in {"sinusoidal", "low_discrepancy"}:
            raise ValueError("slot_code_mode must be 'sinusoidal' or 'low_discrepancy'.")
        if self.edge_selection_mode not in {"all", "quality_coverage"}:
            raise ValueError("edge_selection_mode must be 'all' or 'quality_coverage'.")
        if int(self.selection_warmup_epochs) < 0:
            raise ValueError("selection_warmup_epochs must be >= 0.")
        if int(self.selection_start_epoch) < -1:
            raise ValueError("selection_start_epoch must be >= -1.")
        if int(self.selection_transition_epochs) < 0:
            raise ValueError("selection_transition_epochs must be nonnegative.")
        if self.selection_warmup_mode not in {"legacy", "all_viable"}:
            raise ValueError("selection_warmup_mode must be 'legacy' or 'all_viable'.")
        if not 0.0 < float(self.selection_minimum_module_mass_fraction) <= 1.0:
            raise ValueError("selection_minimum_module_mass_fraction must be in (0, 1].")
        if not 0.0 < float(self.selection_minimum_environment_mass_fraction) <= 1.0:
            raise ValueError("selection_minimum_environment_mass_fraction must be in (0, 1].")
        if not 0.0 < float(self.selection_coverage_rate) <= 1.0:
            raise ValueError("selection_coverage_rate must be in (0, 1].")
        if not 0.0 < float(self.selection_token_threshold) <= 1.0:
            raise ValueError("selection_token_threshold must be in (0, 1].")
        if not 0.0 <= float(self.selection_maximum_redundancy) <= 1.0:
            raise ValueError("selection_maximum_redundancy must be in [0, 1].")
        if not 0.0 < float(self.candidate_module_mass_fraction_floor) <= 1.0:
            raise ValueError("candidate_module_mass_fraction_floor must be in (0, 1].")
        if not 0.0 < float(self.candidate_environment_mass_fraction_floor) <= 1.0:
            raise ValueError("candidate_environment_mass_fraction_floor must be in (0, 1].")
        normalizers = {
            self.module_assignment_normalizer,
            self.environment_assignment_normalizer,
            self.query_assignment_normalizer,
        }
        if not normalizers <= {"softmax", "entmax15", "scheduled"}:
            raise ValueError("assignment normalizers must be 'softmax', 'entmax15', or 'scheduled'.")
        sparsity_schedules = (
            ("module", self.module_assignment_normalizer, self.module_sparsity_start_epoch, self.module_sparsity_transition_epochs),
            ("environment", self.environment_assignment_normalizer, self.environment_sparsity_start_epoch, self.environment_sparsity_transition_epochs),
            ("query", self.query_assignment_normalizer, self.query_sparsity_start_epoch, self.query_sparsity_transition_epochs),
        )
        for name, normalizer, start, transition in sparsity_schedules:
            if int(start) < -1 or int(transition) < 0:
                raise ValueError(f"{name} sparsity schedule values must be >= -1/0 respectively.")
            if normalizer == "scheduled" and (int(start) < 0 or int(transition) <= 0):
                raise ValueError(f"scheduled {name} normalization requires a nonnegative start and positive transition.")
        if not 1.0 < float(self.entmax_alpha) <= 2.0:
            raise ValueError("entmax_alpha must be in (1, 2].")
        if normalizers & {"entmax15", "scheduled"} and abs(float(self.entmax_alpha) - 1.5) > 1.0e-8:
            raise ValueError("entmax15 and scheduled assignment modes require entmax_alpha=1.5.")
        locality_modes = {"none", "compact_kernel", "bounded_gaussian", "gaussian_bounded"}
        if self.environment_locality_mode not in locality_modes:
            raise ValueError(
                "environment_locality_mode must be 'none', 'compact_kernel', "
                "'bounded_gaussian', or 'gaussian_bounded'."
            )
        if self.query_locality_mode not in locality_modes | {"inherit_environment"}:
            raise ValueError("query_locality_mode must be a locality mode or 'inherit_environment'.")
        if float(self.environment_locality_strength) < 0.0:
            raise ValueError("environment_locality_strength must be >= 0.")
        if float(self.locality_radius_cap) <= 0.0:
            raise ValueError("locality_radius_cap must be positive.")
        if float(self.minimum_region_scale) <= 0.0:
            raise ValueError("minimum_region_scale must be positive.")
        if self.mechanism_state_mode not in {"residual_concat", "descriptor_first"}:
            raise ValueError("mechanism_state_mode must be 'residual_concat' or 'descriptor_first'.")
        if not 0.0 <= float(self.mechanism_latent_residual_scale) <= 1.0:
            raise ValueError("mechanism_latent_residual_scale must be in [0, 1].")
        if self.field_assembly_mode not in {"context_fusion", "edge_additive"}:
            raise ValueError("field_assembly_mode must be 'context_fusion' or 'edge_additive'.")
        if not 0.0 < float(self.additive_edge_gate_init) < 1.0:
            raise ValueError("additive_edge_gate_init must be in (0, 1).")
        if float(self.additive_output_init_std) <= 0.0:
            raise ValueError("additive_output_init_std must be positive.")
        if self.field_assembly_mode == "edge_additive":
            components = DECODER_COMPONENTS.get(self.decoder_mode, set())
            if not {"hyper", "pairwise"} <= components:
                raise ValueError("edge_additive field_assembly_mode requires a hyper-plus-pairwise decoder_mode.")
            if self.output_mean_residual_split:
                raise ValueError("output_mean_residual_split is available only with context_fusion field assembly.")
        if self.routing_execution not in {"dense", "gathered", "scheduled"}:
            raise ValueError("routing_execution must be 'dense', 'gathered', or 'scheduled'.")
        if int(self.gathered_execution_start_epoch) < -1:
            raise ValueError("gathered_execution_start_epoch must be >= -1.")
        if self.routing_execution == "scheduled" and int(self.gathered_execution_start_epoch) < 0:
            raise ValueError("scheduled routing execution requires gathered_execution_start_epoch >= 0.")
        if int(self.query_edge_limit) < 0 or int(self.query_module_limit) < 0:
            raise ValueError("gathered routing limits must be nonnegative.")
        if not 0.0 <= float(self.query_edge_retained_mass_floor) <= 1.0:
            raise ValueError("query_edge_retained_mass_floor must be in [0, 1].")
        if not 0.0 <= float(self.module_incidence_retained_mass_floor) <= 1.0:
            raise ValueError("module_incidence_retained_mass_floor must be in [0, 1].")
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

        resolved = dict(payload)
        for key, value in _FORWARD_MODE_DEFAULTS.items():
            resolved.setdefault(key, value)
        return _dataclass_from_dict(cls, resolved)

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
