"""CORE HONF dataclasses and configuration.

Inputs are generic module centers/features, global context, optional query
time, query coordinates, and optional generic environment coordinates/features.
Outputs are configuration and batch containers consumed by the reusable HONF
core. This module is reusable across domains and contains no ChannelThermal
wall, inlet, outlet, or material assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import math
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
    "additive_background_mode": "dense_query_attention",
    "module_assignment_normalizer": "softmax",
    "environment_assignment_normalizer": "softmax",
    "query_assignment_normalizer": "softmax",
    "routing_execution": "dense",
    "pairwise_aggregation_mode": "edge_explicit",
    "pairwise_kernel_mode": "legacy_mlp",
    "query_module_retained_mass_floor": 1.0,
}


FORWARD_ARCHITECTURES = {
    "legacy_honf",
    "dense_pairwise_field",
    "geometry_latent_field",
    "sparse_interface_honf",
    "regional_response_honf",
    "hierarchical_regional_honf",
}

LEGACY_ARCHITECTURE_KEYS = {
    "num_hyperedges", "organizer_mode", "edge_capacity", "initial_active_edges",
    "minimum_active_edges", "slot_refinement_steps", "slot_code_mode",
    "residual_stop_fraction", "residual_soft_stop_temperature",
    "residual_factor_refinement_steps", "residual_coupling_fourier_frequencies",
    "residual_interaction_dim", "residual_mechanism_cap_multiplier",
    "case_edge_selection_mode", "case_edge_probe_source", "case_edge_probe_limit",
    "case_edge_probe_relative_rms_tolerance", "case_edge_probe_channel_tolerance",
    "case_edge_probe_search", "edge_selection_mode", "selection_warmup_epochs",
    "selection_start_epoch", "selection_transition_epochs", "selection_warmup_mode",
    "selection_minimum_module_mass_fraction", "selection_minimum_environment_mass_fraction",
    "selection_coverage_rate", "selection_token_threshold", "selection_maximum_redundancy",
    "candidate_module_mass_fraction_floor", "candidate_environment_mass_fraction_floor",
    "module_assignment_normalizer", "environment_assignment_normalizer",
    "query_assignment_normalizer", "module_sparsity_start_epoch",
    "module_sparsity_transition_epochs", "environment_sparsity_start_epoch",
    "environment_sparsity_transition_epochs", "query_sparsity_start_epoch",
    "query_sparsity_transition_epochs", "entmax_alpha", "environment_locality_mode",
    "environment_locality_strength", "query_locality_mode", "query_locality_strength",
    "locality_radius_cap", "minimum_region_scale", "mechanism_state_mode",
    "mechanism_latent_residual_scale", "field_assembly_mode", "additive_background_mode",
    "additive_edge_gate_init", "additive_output_init_std", "routing_execution",
    "gathered_execution_start_epoch", "query_edge_limit", "query_module_limit",
    "query_edge_retained_mass_floor", "module_incidence_retained_mass_floor",
    "pairwise_aggregation_mode", "pairwise_kernel_mode", "query_module_retained_mass_floor",
    "topology_signature_enabled", "decoder_mode", "use_hyper_value_context",
    "use_hyper_mechanism_encoder", "mechanism_include_geometry", "mechanism_include_masses",
    "mechanism_hidden_dim", "hyper_module_assignment_mode", "hyper_query_attention_mode",
    "hyper_attention_topk", "hyper_attention_temperature", "sparse_hyper_attention_detach_mask",
    "pairwise_kernel_hidden_dim", "pairwise_kernel_num_layers", "pairwise_kernel_gate_init",
    "pairwise_kernel_use_fourier", "pairwise_kernel_fourier_frequencies",
    "pairwise_kernel_include_module_token", "pairwise_kernel_include_module_features",
    "pairwise_kernel_normalize_by_edge_mass", "use_hyper_geometry_bias",
    "hyper_geometry_bias_scale", "direct_residual_gate_init", "use_A_me_auxiliary",
    "output_mean_residual_split",
}


@dataclass
class InterfaceFieldConfig:
    """Matched-family settings shared by non-legacy interface fields."""

    message_hidden_dim: int = 128
    attention_heads: int = 4
    coarse_latent_count: int = 8
    coarse_blocks: int = 1
    main_latent_count: int = 16
    main_latent_blocks: int = 2
    local_radius_factor: float = 2.5
    support_spacing_factor: Optional[float] = None
    # Historical sparse readers include an explicit null entry.  The
    # geometry-envelope mode is opt-in for the reader-recovery experiment.
    group_read_mode: str = "null_softmax"
    relative_fourier_frequencies: int = 4
    receiver_chunk_size: int = 128
    activation_checkpointing: bool = False
    # Physical extents of one response region along the adapter-provided
    # environment grid axes.  The adapter owns membership; the backend only
    # consumes the resulting IDs, masses, and centroids.
    response_region_block_shape: list[int] = field(default_factory=lambda: [2, 2])
    response_tree_opening_interval: list[float] = field(default_factory=lambda: [1.0, 2.0])
    coarse_module_source: str = "module_states"

    def __post_init__(self) -> None:
        if int(self.message_hidden_dim) <= 0:
            raise ValueError("interface_model.message_hidden_dim must be positive.")
        if int(self.attention_heads) <= 0:
            raise ValueError("interface_model.attention_heads must be positive.")
        if int(self.coarse_latent_count) <= 0 or int(self.coarse_blocks) <= 0:
            raise ValueError("interface_model coarse latent count/blocks must be positive.")
        if int(self.main_latent_count) <= 0 or int(self.main_latent_blocks) <= 0:
            raise ValueError("interface_model main latent count/blocks must be positive.")
        if float(self.local_radius_factor) <= 0.0:
            raise ValueError("interface_model.local_radius_factor must be positive.")
        if self.support_spacing_factor is not None and float(self.support_spacing_factor) <= 0.0:
            raise ValueError("interface_model.support_spacing_factor must be positive when provided.")
        if self.group_read_mode not in {"null_softmax", "geometry_envelope_attention"}:
            raise ValueError(
                "interface_model.group_read_mode must be 'null_softmax' or "
                "'geometry_envelope_attention'."
            )
        if int(self.relative_fourier_frequencies) < 0:
            raise ValueError("interface_model.relative_fourier_frequencies must be nonnegative.")
        if int(self.receiver_chunk_size) <= 0:
            raise ValueError("interface_model.receiver_chunk_size must be positive.")
        if (
            not isinstance(self.response_region_block_shape, (list, tuple))
            or len(self.response_region_block_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0
                   for value in self.response_region_block_shape)
        ):
            raise ValueError(
                "interface_model.response_region_block_shape must contain two positive integers."
            )
        self.response_region_block_shape = [int(value) for value in self.response_region_block_shape]
        interval = self.response_tree_opening_interval
        if (
            not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in interval)
            or not 0.0 < float(interval[0]) < float(interval[1])
        ):
            raise ValueError("interface_model.response_tree_opening_interval must contain increasing positive radii.")
        self.response_tree_opening_interval = [float(value) for value in interval]
        if self.coarse_module_source not in {"module_states", "group_states"}:
            raise ValueError("interface_model.coarse_module_source must be 'module_states' or 'group_states'.")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "InterfaceFieldConfig":
        return _dataclass_from_dict(cls, dict(payload or {}))


@dataclass
class UnifiedForwardConfig:
    """Configuration for the minimal unified hypergraph neural field."""

    field_dim: int = 5
    forward_architecture: str = "legacy_honf"
    interface_model: Optional[InterfaceFieldConfig] = None
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
    residual_stop_fraction: float = 0.02
    residual_soft_stop_temperature: float = 0.05
    residual_factor_refinement_steps: int = 2
    residual_coupling_fourier_frequencies: int = 2
    residual_interaction_dim: int = 32
    residual_mechanism_cap_multiplier: float = 1.5

    # Evaluation-only case-specific query routing.  This is deliberately
    # separate from ``edge_selection_mode``: the organizer always constructs
    # and trains the complete fixed bank, while probe fidelity may retain a
    # smaller routing support after case preparation.
    case_edge_selection_mode: str = "none"
    case_edge_probe_source: str = "environment_plus_module_local"
    case_edge_probe_limit: int = 256
    case_edge_probe_relative_rms_tolerance: float = 0.005
    case_edge_probe_channel_tolerance: float = 0.01
    case_edge_probe_search: str = "exhaustive_small_bank"

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
    query_locality_strength: Optional[float] = None
    locality_radius_cap: float = 3.0
    minimum_region_scale: float = 0.05

    mechanism_state_mode: str = "residual_concat"
    mechanism_latent_residual_scale: float = 0.35

    field_assembly_mode: str = "context_fusion"
    additive_background_mode: str = "dense_query_attention"
    additive_edge_gate_init: float = 0.10
    additive_output_init_std: float = 1.0e-3
    routing_execution: str = "dense"
    gathered_execution_start_epoch: int = -1
    query_edge_limit: int = 0
    query_module_limit: int = 0
    query_edge_retained_mass_floor: float = 0.0
    module_incidence_retained_mass_floor: float = 0.0
    pairwise_aggregation_mode: str = "edge_explicit"
    pairwise_kernel_mode: str = "legacy_mlp"
    query_module_retained_mass_floor: float = 1.0

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
    # Historical profiles omit this field and therefore continue to use the
    # original two-dimensional parameter/feature path.  A three-dimensional
    # path is opt-in and is intentionally limited to the supported WindFarm
    # fixed organizer and dense interface backend.  It is appended to retain
    # the positional order of every pre-existing configuration field.
    spatial_dim: int = 2

    def __post_init__(self) -> None:
        """Validate mode names and numerical routing constraints."""

        if int(self.spatial_dim) not in {2, 3}:
            raise ValueError("spatial_dim must be 2 or 3.")
        self.spatial_dim = int(self.spatial_dim)
        if isinstance(self.interface_model, dict):
            self.interface_model = InterfaceFieldConfig.from_dict(self.interface_model)
        if self.forward_architecture not in FORWARD_ARCHITECTURES:
            allowed = ", ".join(sorted(FORWARD_ARCHITECTURES))
            raise ValueError(f"forward_architecture must be one of: {allowed}")
        if self.forward_architecture == "legacy_honf":
            if self.interface_model is not None:
                raise ValueError("legacy_honf does not accept an interface_model block.")
        elif self.interface_model is None:
            raise ValueError(f"{self.forward_architecture} requires an interface_model block.")
        elif self.forward_architecture == "sparse_interface_honf":
            if self.interface_model.support_spacing_factor is None:
                raise ValueError(
                    "sparse_interface_honf requires interface_model.support_spacing_factor."
                )
        elif self.interface_model.support_spacing_factor is not None:
            raise ValueError(
                "interface_model.support_spacing_factor is only valid for sparse_interface_honf."
            )
        if self.interface_model is not None and int(self.hidden_dim) % int(self.interface_model.attention_heads) != 0:
            raise ValueError("hidden_dim must be divisible by interface_model.attention_heads.")
        if (self.interface_model is not None
                and self.interface_model.coarse_module_source == "group_states"
                and self.forward_architecture != "sparse_interface_honf"):
            raise ValueError("coarse_module_source='group_states' requires sparse_interface_honf group preparation.")

        if self.organizer_mode not in {
            "fixed_projection",
            "exchangeable_slots",
            "case_adaptive_residual",
            "case_adaptive_tensor_residual",
        }:
            raise ValueError(
                "organizer_mode must be 'fixed_projection', 'exchangeable_slots', "
                "'case_adaptive_residual', or 'case_adaptive_tensor_residual'."
            )
        if self.organizer_mode == "fixed_projection" and int(self.num_hyperedges) <= 0:
            raise ValueError("fixed_projection organizer_mode requires num_hyperedges > 0.")
        if self.organizer_mode == "exchangeable_slots":
            if int(self.edge_capacity) <= 0:
                raise ValueError("exchangeable_slots organizer_mode requires edge_capacity > 0.")
            if int(self.initial_active_edges) <= 0 or int(self.initial_active_edges) > int(self.edge_capacity):
                raise ValueError("initial_active_edges must be in [1, edge_capacity].")
            if int(self.minimum_active_edges) <= 0 or int(self.minimum_active_edges) > int(self.initial_active_edges):
                raise ValueError("minimum_active_edges must be in [1, initial_active_edges].")
        if self.organizer_mode == "case_adaptive_residual":
            if int(self.num_hyperedges) != 0:
                raise ValueError("case_adaptive_residual organizer_mode requires num_hyperedges == 0.")
            if int(self.edge_capacity) != 0:
                raise ValueError("case_adaptive_residual organizer_mode requires edge_capacity == 0.")
            if int(self.minimum_active_edges) <= 0:
                raise ValueError("case_adaptive_residual requires minimum_active_edges >= 1.")
            if not 0.0 < float(self.residual_stop_fraction) < 1.0:
                raise ValueError("residual_stop_fraction must be in (0, 1).")
            if float(self.residual_soft_stop_temperature) <= 0.0:
                raise ValueError("residual_soft_stop_temperature must be positive.")
            if int(self.residual_factor_refinement_steps) <= 0:
                raise ValueError("residual_factor_refinement_steps must be positive.")
            if int(self.residual_coupling_fourier_frequencies) < 0:
                raise ValueError("residual_coupling_fourier_frequencies must be nonnegative.")
            if self.field_assembly_mode != "context_fusion":
                raise ValueError("case_adaptive_residual requires context_fusion field assembly.")
            if self.pairwise_aggregation_mode != "fused_query_module":
                raise ValueError(
                    "case_adaptive_residual requires fused_query_module pairwise aggregation."
                )
        if self.organizer_mode == "case_adaptive_tensor_residual":
            if int(self.num_hyperedges) != 0:
                raise ValueError(
                    "case_adaptive_tensor_residual organizer_mode requires num_hyperedges == 0."
                )
            if int(self.edge_capacity) != 0:
                raise ValueError(
                    "case_adaptive_tensor_residual organizer_mode requires edge_capacity == 0."
                )
            if int(self.minimum_active_edges) <= 0:
                raise ValueError("case_adaptive_tensor_residual requires minimum_active_edges >= 1.")
            if int(self.residual_interaction_dim) <= 0:
                raise ValueError("residual_interaction_dim must be positive.")
            if float(self.residual_mechanism_cap_multiplier) < 1.0:
                raise ValueError("residual_mechanism_cap_multiplier must be >= 1.")
            if not 0.0 < float(self.residual_stop_fraction) < 1.0:
                raise ValueError("residual_stop_fraction must be in (0, 1).")
            if float(self.residual_soft_stop_temperature) <= 0.0:
                raise ValueError("residual_soft_stop_temperature must be positive.")
            if int(self.residual_factor_refinement_steps) <= 0:
                raise ValueError("residual_factor_refinement_steps must be positive.")
            if int(self.residual_coupling_fourier_frequencies) < 0:
                raise ValueError("residual_coupling_fourier_frequencies must be nonnegative.")
            if self.field_assembly_mode != "context_fusion":
                raise ValueError(
                    "case_adaptive_tensor_residual requires context_fusion field assembly."
                )
            if self.pairwise_aggregation_mode != "fused_query_module":
                raise ValueError(
                    "case_adaptive_tensor_residual requires fused_query_module pairwise aggregation."
                )
        if self.case_edge_selection_mode not in {"none", "probe_fidelity"}:
            raise ValueError(
                "case_edge_selection_mode must be 'none' or 'probe_fidelity'."
            )
        if self.case_edge_probe_source != "environment_plus_module_local":
            raise ValueError(
                "case_edge_probe_source must be 'environment_plus_module_local'."
            )
        if int(self.case_edge_probe_limit) <= 0:
            raise ValueError("case_edge_probe_limit must be positive.")
        if float(self.case_edge_probe_relative_rms_tolerance) < 0.0:
            raise ValueError("case_edge_probe_relative_rms_tolerance must be nonnegative.")
        if float(self.case_edge_probe_channel_tolerance) < 0.0:
            raise ValueError("case_edge_probe_channel_tolerance must be nonnegative.")
        if self.case_edge_probe_search != "exhaustive_small_bank":
            raise ValueError(
                "case_edge_probe_search must be 'exhaustive_small_bank'."
            )
        if (
            self.case_edge_selection_mode == "probe_fidelity"
            and self.organizer_mode != "fixed_projection"
        ):
            raise ValueError(
                "probe_fidelity case-edge selection requires fixed_projection organization."
            )
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
        if self.query_locality_strength is not None and float(self.query_locality_strength) < 0.0:
            raise ValueError("query_locality_strength must be >= 0 when provided.")
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
        if self.additive_background_mode not in {
            "dense_query_attention",
            "global_pooled_attention",
        }:
            raise ValueError(
                "additive_background_mode must be 'dense_query_attention' "
                "or 'global_pooled_attention'."
            )
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
        if self.pairwise_aggregation_mode not in {"edge_explicit", "fused_query_module"}:
            raise ValueError(
                "pairwise_aggregation_mode must be 'edge_explicit' or "
                "'fused_query_module'."
            )
        if self.pairwise_kernel_mode not in {"legacy_mlp", "factorized_gated"}:
            raise ValueError(
                "pairwise_kernel_mode must be 'legacy_mlp' or 'factorized_gated'."
            )
        if not 0.0 <= float(self.query_module_retained_mass_floor) <= 1.0:
            raise ValueError("query_module_retained_mass_floor must be in [0, 1].")
        if (
            self.pairwise_aggregation_mode == "fused_query_module"
            and self.field_assembly_mode != "context_fusion"
        ):
            raise ValueError(
                "fused_query_module pairwise aggregation requires context_fusion "
                "field assembly."
            )
        if (
            self.pairwise_kernel_mode == "factorized_gated"
            and self.pairwise_aggregation_mode != "fused_query_module"
        ):
            raise ValueError(
                "factorized_gated pairwise kernels require fused_query_module "
                "pairwise aggregation."
            )
        if self.geometry_mode not in {"nonperiodic", "periodic"}:
            raise ValueError("geometry_mode must be 'nonperiodic' or 'periodic'.")
        if self.query_time_mode not in {"none", "phase", "physical_time"}:
            raise ValueError("query_time_mode must be 'none', 'phase', or 'physical_time'.")
        if self.boundary_feature_mode not in {"none", "rectangular", "channel"}:
            raise ValueError("boundary_feature_mode must be 'none' or 'rectangular' ('channel' is a legacy alias).")
        if self.local_context_scale is not None and float(self.local_context_scale) <= 0.0:
            raise ValueError("local_context_scale must be positive when provided.")
        if self.coordinate_scale is not None:
            if len(self.coordinate_scale) != self.spatial_dim or any(
                float(value) <= 0.0 for value in self.coordinate_scale
            ):
                raise ValueError(
                    f"coordinate_scale must contain {self.spatial_dim} positive values."
                )
        if self.spatial_dim == 3:
            if self.coordinate_scale is None:
                raise ValueError("spatial_dim=3 requires an explicit three-value coordinate_scale.")
            if self.forward_architecture not in {"legacy_honf", "dense_pairwise_field"}:
                raise ValueError(
                    "spatial_dim=3 is supported only by legacy_honf or dense_pairwise_field."
                )
            if self.forward_architecture == "legacy_honf" and self.organizer_mode != "fixed_projection":
                raise ValueError("spatial_dim=3 legacy_honf requires fixed_projection organization.")
            if self.geometry_mode != "nonperiodic":
                raise ValueError("spatial_dim=3 currently supports only nonperiodic geometry.")
            if self.periodic_axes not in (None, []):
                raise ValueError("spatial_dim=3 currently does not support periodic_axes.")
            if self.boundary_feature_mode != "none":
                raise ValueError("spatial_dim=3 requires boundary_feature_mode='none'.")
            if self.forward_architecture == "legacy_honf":
                if self.field_assembly_mode != "context_fusion":
                    raise ValueError("spatial_dim=3 legacy_honf requires context_fusion assembly.")
                if self.pairwise_kernel_mode != "legacy_mlp":
                    raise ValueError("spatial_dim=3 legacy_honf requires the legacy_mlp pair kernel.")
                if self.pairwise_aggregation_mode != "fused_query_module":
                    raise ValueError("spatial_dim=3 legacy_honf requires fused_query_module aggregation.")
                if self.routing_execution != "dense" or self.edge_selection_mode != "all":
                    raise ValueError("spatial_dim=3 legacy_honf requires dense full-support routing.")
                if self.query_edge_limit != 0 or self.query_module_limit != 0:
                    raise ValueError("spatial_dim=3 legacy_honf does not support query routing truncation.")
                if self.hyper_attention_topk != 0:
                    raise ValueError("spatial_dim=3 legacy_honf does not support hyperedge top-k routing.")
        if self.periodic_axes is not None:
            axes = [int(value) for value in self.periodic_axes]
            allowed_axes = set(range(self.spatial_dim))
            if len(set(axes)) != len(axes) or any(value not in allowed_axes for value in axes):
                raise ValueError(
                    f"periodic_axes must contain unique axis indices from {sorted(allowed_axes)}."
                )
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

    def spatial_scale(self) -> tuple[float, ...]:
        """Return coordinate scales with the historical 2-D fallback."""

        if self.coordinate_scale is not None:
            return tuple(float(value) for value in self.coordinate_scale)
        # The 3-D validator requires an explicit scale.  Keep a direct error
        # here as a guard for callers that inspect a partially-built object.
        if self.spatial_dim == 3:
            raise ValueError("spatial_dim=3 requires an explicit coordinate_scale.")
        return float(self.domain_length_x), float(self.domain_length_y)

    def periodic_dimensions(self) -> tuple[int, ...]:
        """Return periodic axes, interpreting legacy periodic mode as all axes."""

        if self.periodic_axes is not None:
            return tuple(sorted(int(value) for value in self.periodic_axes))
        return tuple(range(self.spatial_dim)) if self.geometry_mode == "periodic" else ()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UnifiedForwardConfig":
        """Construct a strict core configuration from a mapping."""

        resolved = dict(payload)
        for key, value in _FORWARD_MODE_DEFAULTS.items():
            resolved.setdefault(key, value)
        if isinstance(resolved.get("interface_model"), dict):
            resolved["interface_model"] = InterfaceFieldConfig.from_dict(resolved["interface_model"])
        return _dataclass_from_dict(cls, resolved)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this core configuration to plain Python values."""

        payload = _to_plain_dict(self)
        # Keep historical resolved configurations byte-for-byte compatible
        # when the new dimension selector is at its omitted/default value.
        # New 3-D checkpoints retain the explicit selector.
        if self.spatial_dim == 2:
            payload.pop("spatial_dim", None)
        # Historical resolved configs/checkpoints predate the architecture
        # selector. Keep their serialized shape unchanged and do not inject an
        # empty new-family block while loading or resaving them.
        if self.forward_architecture == "legacy_honf":
            payload.pop("forward_architecture", None)
            payload.pop("interface_model", None)
        else:
            for key in LEGACY_ARCHITECTURE_KEYS:
                payload.pop(key, None)
            interface_payload = payload.get("interface_model")
            if isinstance(interface_payload, dict):
                if self.forward_architecture != "sparse_interface_honf":
                    interface_payload.pop("support_spacing_factor", None)
                    interface_payload.pop("group_read_mode", None)
                else:
                    # Main latent settings belong only to the latent-attention
                    # baseline and are not sparse-HONF capacity parameters.
                    interface_payload.pop("main_latent_count", None)
                    interface_payload.pop("main_latent_blocks", None)
                if self.forward_architecture not in {"regional_response_honf", "hierarchical_regional_honf"}:
                    interface_payload.pop("response_region_block_shape", None)
                if self.forward_architecture != "hierarchical_regional_honf":
                    interface_payload.pop("response_tree_opening_interval", None)
                if self.interface_model.coarse_module_source == "module_states":
                    interface_payload.pop("coarse_module_source", None)
        return payload

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
    # Optional adapter-owned physical region IDs aligned with ``env_coords``.
    # They are consumed only by regional-response families; established
    # families ignore the field and retain the fine environment route.  Keep
    # this after the historical fields so positional BatchData construction
    # retains its prior ordering.
    env_region_ids: Optional[Any] = None
    # Optional typed hierarchy supplied by the case adapter, aligned to the
    # original fine environment. Historical batches leave it unset.
    env_hierarchy: Optional[Any] = None
    # Optional adapter-owned environmental quadrature mass.  Appended after
    # all historical fields so positional BatchData construction is stable.
    env_weights: Optional[Any] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BatchData":
        """Validate and construct the canonical batch container."""

        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        """Describe batch values as serializable metadata."""

        payload = _to_plain_dict(self)
        # Keep the historical absent-measure representation unchanged.  A
        # supplied tensor is still described explicitly for adapter diagnostics.
        if self.env_weights is None:
            payload.pop("env_weights", None)
        return payload

    def to(self, device: Any) -> "BatchData":
        """Move tensor fields to a device and return a new BatchData object."""
        if torch is None:
            return self
        payload: Dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = (
                value.to(device) if torch.is_tensor(value) or (item.name == "env_hierarchy" and value is not None)
                else value
            )
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
