"""CORE HONF dataclasses and configuration.

Inputs are generic module centers/features, global context, optional query
time, query coordinates, and optional generic environment coordinates/features.
Outputs are configuration and batch containers consumed by the reusable HONF
core. This module is reusable across domains and contains no ChannelThermal
wall, inlet, outlet, or material assumptions.
"""

from __future__ import annotations

import math
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
    "enhanced_honf_pairwise_no_global",
    "enhanced_honf_pairwise_no_global_near",
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
    "enhanced_honf_pairwise_no_global": {"hyper", "pairwise", "near"},
    "enhanced_honf_pairwise_no_global_near": {"hyper", "pairwise"},
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
    "pairwise_module_token_source": "base",
    "query_module_retained_mass_floor": 1.0,
}


FORWARD_ARCHITECTURES = {
    "legacy_honf",
    "dense_pairwise_field",
    "geometry_latent_field",
    "sparse_interface_honf",
    "regional_response_honf",
    "hierarchical_regional_honf",
    "routed_pairwise_honf",
    "fixed_group_pairwise_honf",
    "group_control_pairwise_honf",
    "phase_shared_group_control_honf",
    "hypergraph_quadrature_honf",
    "budgeted_group_control_honf",
    "occupancy_adaptive_group_control_honf",
    "mass_competitive_group_control_honf",
    "sparse_incidence_group_control_honf",
    "coalesced_sparse_incidence_honf",
    "converged_identity_preserving_coalescence_honf",
    "continuous_functional_coalescence_honf",
    "adaptive_hyperedge_opening_honf",
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
    "pairwise_aggregation_mode", "pairwise_kernel_mode", "pairwise_module_token_source",
    "query_module_retained_mass_floor",
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


ROUTING_TYPED_TEMPERATURE_NAMES = (
    "log_temperature_source_module",
    "log_temperature_source_environment",
    "log_temperature_query_module",
    "log_temperature_query_environment",
)


@dataclass
class RoutingSparsificationConfig:
    """Opt-in science settings for the routed sparse-execution trial.

    The historical routed profiles leave this block absent when serialized.
    When enabled, the only trainable additions are the four typed log
    temperatures named in :data:`ROUTING_TYPED_TEMPERATURE_NAMES`; the only
    auxiliary objective is the induced fine-pair cost term.
    """

    enabled: bool = False
    learn_typed_temperatures: bool = False
    relative_density_epsilon: float = 0.05
    cost_weight: float = 0.0
    module_pair_cost: float = 1.0
    environment_pair_cost: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("routing.sparsification.enabled must be boolean.")
        if not isinstance(self.learn_typed_temperatures, bool):
            raise TypeError("routing.sparsification.learn_typed_temperatures must be boolean.")
        epsilon = float(self.relative_density_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("routing.sparsification.relative_density_epsilon must be finite and positive.")
        cost_weight = float(self.cost_weight)
        if not math.isfinite(cost_weight) or cost_weight < 0.0:
            raise ValueError("routing.sparsification.cost_weight must be finite and nonnegative.")
        for name in ("module_pair_cost", "environment_pair_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"routing.sparsification.{name} must be finite and positive.")
        if not self.enabled and (self.learn_typed_temperatures or cost_weight != 0.0):
            raise ValueError(
                "routing.sparsification.enabled must be true when typed temperatures or "
                "the induced pair-cost objective is configured."
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> RoutingSparsificationConfig:
        return _dataclass_from_dict(cls, dict(payload or {}))


@dataclass
class RoutingIndexConfig:
    """Scalar routing settings; fine response capacity remains Dense's."""

    strategy: str = "module_hubs"
    descriptor_dim: int = 32
    router_hidden_dim: int = 64
    source_normalizer: str = "sparsemax"
    query_normalizer: str = "source_measure_sparsemax"
    temperature: float = 1.0
    content_scale: float = 2.0
    geometry_scale: float = 0.25
    propensity_scale: float = 0.25
    resistance_mode: str = "adapter"
    execution: str = "gathered"
    fine_pair_chunk_size: int = 16384
    # Run 2100 is intentionally a fixed three-step experiment.  These fields
    # are omitted from historical module-hub profiles and materialize only
    # when the new strategy is selected.
    mean_shift_steps: int = 3
    mean_shift_feature_bandwidth: float = 1.0
    # Appended after all historical fields so positional construction and
    # serialized historical routing blocks remain compatible.
    qe_backend: str = "torch"
    sparsification: RoutingSparsificationConfig = field(default_factory=RoutingSparsificationConfig)

    def __post_init__(self) -> None:
        if isinstance(self.sparsification, dict):
            self.sparsification = RoutingSparsificationConfig.from_dict(self.sparsification)
        for name in (
            "descriptor_dim",
            "router_hidden_dim",
            "fine_pair_chunk_size",
            "mean_shift_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"routing.{name} must be a positive integer.")
        for name in ("temperature", "content_scale", "geometry_scale", "propensity_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or (name == "temperature" and value == 0):
                raise ValueError(f"routing.{name} must be finite and nonnegative (temperature positive).")
        if not math.isfinite(float(self.mean_shift_feature_bandwidth)) or self.mean_shift_feature_bandwidth <= 0:
            raise ValueError("routing.mean_shift_feature_bandwidth must be finite and positive.")
        required = {"strategy": "module_hubs", "source_normalizer": "sparsemax",
                    "query_normalizer": "source_measure_sparsemax", "resistance_mode": "adapter"}
        if self.strategy not in {"module_hubs", "mean_shift"}:
            raise ValueError("routing.strategy currently supports 'module_hubs' or 'mean_shift'.")
        if self.execution not in {"gathered", "optimized_exact", "compiled_exact"}:
            raise ValueError(
                "routing.execution currently supports 'gathered', 'optimized_exact', or 'compiled_exact'."
            )
        if self.qe_backend not in {"torch", "triton"}:
            raise ValueError("routing.qe_backend must be 'torch' or 'triton'.")
        for name, expected in required.items():
            if name == "strategy":
                continue
            if getattr(self, name) != expected:
                raise ValueError(f"routing.{name} currently supports only {expected!r}.")
        if self.strategy == "mean_shift" and self.mean_shift_steps != 3:
            raise ValueError("routing.mean_shift_steps is fixed at exactly 3 for mean_shift.")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RoutingIndexConfig":
        return _dataclass_from_dict(cls, dict(payload))


@dataclass
class CaseGroupBudgetConfig:
    """Opt-in hard-concrete availability settings for Run 1409."""

    enabled: bool = False
    gate_hidden_dim: int = 32
    hard_concrete_temperature: float = 2.0 / 3.0
    stretch_lower: float = -0.1
    stretch_upper: float = 1.1
    initial_optional_open_probability: float = 0.95
    always_available_group: int = 0
    normalization: str = "gate_reference_overlap"
    execution_mode: str = "full_width"
    # New rescue semantics are opt-in.  Old model_config payloads omit these
    # fields and therefore reconstruct the historical privileged-group path.
    rescue_mode: bool = False
    schedule: str = "static"
    routing_initial_scale: float = 0.1
    routing_full_epoch: int = 25
    compression_start_epoch: int = 25
    hardening_epoch: int = 150

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("interface_model.case_group_budget.enabled must be boolean.")
        if not isinstance(self.rescue_mode, bool):
            raise TypeError("interface_model.case_group_budget.rescue_mode must be boolean.")
        if isinstance(self.gate_hidden_dim, bool) or int(self.gate_hidden_dim) <= 0:
            raise ValueError("interface_model.case_group_budget.gate_hidden_dim must be positive.")
        for name in ("hard_concrete_temperature", "stretch_lower", "stretch_upper"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"interface_model.case_group_budget.{name} must be finite.")
        if float(self.hard_concrete_temperature) <= 0.0:
            raise ValueError("interface_model.case_group_budget.hard_concrete_temperature must be positive.")
        if not float(self.stretch_lower) < 0.0 < float(self.stretch_upper):
            raise ValueError("interface_model.case_group_budget stretch must straddle zero.")
        probability = float(self.initial_optional_open_probability)
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError(
                "interface_model.case_group_budget.initial_optional_open_probability must be in (0,1)."
            )
        if (
            isinstance(self.always_available_group, bool)
            or int(self.always_available_group) < 0
            or int(self.always_available_group) >= 12
        ):
            raise ValueError(
                "interface_model.case_group_budget.always_available_group must be an index in [0, 12)."
            )
        if not self.rescue_mode and int(self.always_available_group) != 0:
            raise ValueError("the historical budget path requires always_available_group=0.")
        if self.schedule not in {"static", "dense_to_sparse_v2"}:
            raise ValueError(
                "interface_model.case_group_budget.schedule must be 'static' or 'dense_to_sparse_v2'."
            )
        if self.schedule == "dense_to_sparse_v2" and not self.rescue_mode:
            raise ValueError(
                "dense_to_sparse_v2 requires interface_model.case_group_budget.rescue_mode=true."
            )
        routing_initial_scale = float(self.routing_initial_scale)
        if not math.isfinite(routing_initial_scale) or not 0.0 < routing_initial_scale <= 1.0:
            raise ValueError(
                "interface_model.case_group_budget.routing_initial_scale must be finite and in (0,1]."
            )
        for name in ("routing_full_epoch", "compression_start_epoch", "hardening_epoch"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(
                    f"interface_model.case_group_budget.{name} must be a positive integer."
                )
        if int(self.compression_start_epoch) > int(self.hardening_epoch):
            raise ValueError(
                "interface_model.case_group_budget.compression_start_epoch must not exceed hardening_epoch."
            )
        if self.normalization != "gate_reference_overlap":
            raise ValueError(
                "interface_model.case_group_budget.normalization must be 'gate_reference_overlap'."
            )
        if self.execution_mode not in {"full_width", "compact"}:
            raise ValueError(
                "interface_model.case_group_budget.execution_mode must be 'full_width' or 'compact'."
            )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "CaseGroupBudgetConfig":
        return _dataclass_from_dict(cls, dict(payload or {}))


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
    routing: Optional[RoutingIndexConfig] = None
    # Run 1405's fixed-group reader is deliberately appended so historical
    # positional construction remains stable.  These values are validated as
    # compact contracts for the opt-in fixed/group-control readers below.
    group_count: int = 6
    source_normalizer: str = "entmax15"
    query_normalizer: str = "entmax15"
    module_temperature: float = 1.0
    environment_temperature: float = 1.0
    query_temperature: float = 1.0
    group_code_dim: int = 32
    # Run 1406's low-dimensional control width.  Appended so historical
    # positional InterfaceFieldConfig construction remains unchanged.
    group_control_dim: int = 16
    # Run 1408's fixed environmental quadrature budget. Appended so
    # historical positional InterfaceFieldConfig construction remains
    # unchanged and omitted from historical serialized profiles.
    samples_per_group: int = 4
    # Run 1409's case-level availability plan.  Appended to preserve every
    # historical positional constructor and omitted from old architectures.
    case_group_budget: Optional[CaseGroupBudgetConfig] = None
    # Run 1502's sole candidate hook.  The sparse-incidence backend applies
    # this only to the final environmental refinement; all proposal/module
    # assignments and query routing keep their historical normalizers.
    environment_refinement_normalizer: str = "entmax15"
    # Run 1503 reversible fusion settings are serialized only by the
    # coalescence architectures that consume them.
    fusion_max_iterations: int = 64
    fusion_eta_final: float = 0.5
    fusion_ramp_epochs: int = 150
    fusion_eps_abs: float = 1.0e-9
    fusion_eps_rel: float = 1.0e-8
    # Run 1503-v4 uses one train-input-only, fixed binary tree. These fields
    # are serialized only for that opt-in architecture.
    functional_tree_subsets: list[list[int]] = field(default_factory=list)
    read_input_close_rms: float = 0.02
    read_input_keep_rms: float = 0.06

    def __post_init__(self) -> None:
        if isinstance(self.routing, dict):
            self.routing = RoutingIndexConfig.from_dict(self.routing)
        if isinstance(self.case_group_budget, dict):
            self.case_group_budget = CaseGroupBudgetConfig.from_dict(self.case_group_budget)
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
        if isinstance(self.group_count, bool) or not isinstance(self.group_count, int) or int(self.group_count) <= 0:
            raise ValueError("interface_model.group_count must be a positive integer.")
        if self.source_normalizer not in {"softmax", "entmax15"}:
            raise ValueError("interface_model.source_normalizer must be 'softmax' or 'entmax15'.")
        if self.query_normalizer not in {"softmax", "entmax15", "sparsemax"}:
            raise ValueError(
                "interface_model.query_normalizer must be 'softmax', 'entmax15', or 'sparsemax'."
            )
        for name in ("module_temperature", "environment_temperature", "query_temperature"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"interface_model.{name} must be finite and positive.")
        if (
            isinstance(self.group_code_dim, bool)
            or not isinstance(self.group_code_dim, int)
            or int(self.group_code_dim) <= 0
        ):
            raise ValueError("interface_model.group_code_dim must be a positive integer.")
        if (
            isinstance(self.group_control_dim, bool)
            or not isinstance(self.group_control_dim, int)
            or int(self.group_control_dim) <= 0
        ):
            raise ValueError("interface_model.group_control_dim must be a positive integer.")
        if (
            isinstance(self.samples_per_group, bool)
            or not isinstance(self.samples_per_group, int)
            or int(self.samples_per_group) <= 0
        ):
            raise ValueError("interface_model.samples_per_group must be a positive integer.")
        if self.environment_refinement_normalizer not in {"entmax15", "sparsemax"}:
            raise ValueError(
                "interface_model.environment_refinement_normalizer must be "
                "'entmax15' or 'sparsemax'."
            )
        if (
            isinstance(self.fusion_max_iterations, bool)
            or not isinstance(self.fusion_max_iterations, int)
            or self.fusion_max_iterations <= 0
        ):
            raise ValueError("interface_model.fusion_max_iterations must be a positive integer.")
        if not math.isfinite(float(self.fusion_eta_final)) or float(self.fusion_eta_final) <= 0.0:
            raise ValueError("interface_model.fusion_eta_final must be finite and positive.")
        if (
            isinstance(self.fusion_ramp_epochs, bool)
            or not isinstance(self.fusion_ramp_epochs, int)
            or self.fusion_ramp_epochs <= 0
        ):
            raise ValueError("interface_model.fusion_ramp_epochs must be a positive integer.")
        for name in ("fusion_eps_abs", "fusion_eps_rel"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"interface_model.{name} must be finite and nonnegative."
                )
        for name in ("read_input_close_rms", "read_input_keep_rms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"interface_model.{name} must be finite and positive.")
        if float(self.read_input_close_rms) >= float(self.read_input_keep_rms):
            raise ValueError("read_input_close_rms must be below read_input_keep_rms.")

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
    # Appended after all historical fields so positional construction remains
    # compatible. The default preserves the accepted Run-1401 pair arithmetic.
    pairwise_module_token_source: str = "base"

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
        elif self.forward_architecture in {
            "group_control_pairwise_honf",
            "phase_shared_group_control_honf",
            "hypergraph_quadrature_honf",
            "budgeted_group_control_honf",
            "occupancy_adaptive_group_control_honf",
            "mass_competitive_group_control_honf",
            "sparse_incidence_group_control_honf",
            "coalesced_sparse_incidence_honf",
            "converged_identity_preserving_coalescence_honf",
            "continuous_functional_coalescence_honf",
            "adaptive_hyperedge_opening_honf",
        }:
            controlled = self.interface_model
            if controlled.support_spacing_factor is not None:
                raise ValueError(
                    "interface_model.support_spacing_factor is only valid for sparse_interface_honf."
                )
            if controlled.source_normalizer != "entmax15":
                raise ValueError(
                        f"{self.forward_architecture} requires interface_model.source_normalizer='entmax15'."
                )
            expected_query_normalizer = (
                "sparsemax"
                if self.forward_architecture in {
                    "sparse_incidence_group_control_honf",
                    "coalesced_sparse_incidence_honf",
                    "converged_identity_preserving_coalescence_honf",
                    "continuous_functional_coalescence_honf",
                    "adaptive_hyperedge_opening_honf",
                }
                else "entmax15"
            )
            if controlled.query_normalizer != expected_query_normalizer:
                raise ValueError(
                    f"{self.forward_architecture} requires interface_model.query_normalizer="
                    f"'{expected_query_normalizer}'."
                )
            for name in ("module_temperature", "environment_temperature", "query_temperature"):
                if float(getattr(controlled, name)) != 1.0:
                    raise ValueError(
                        f"{self.forward_architecture} requires interface_model.{name}=1.0."
                    )
            if self.forward_architecture in {
                "phase_shared_group_control_honf",
                "hypergraph_quadrature_honf",
            }:
                if int(controlled.group_count) != 6:
                    raise ValueError(
                        f"{self.forward_architecture} requires interface_model.group_count exactly 6."
                    )
                if int(controlled.group_control_dim) != 16:
                    raise ValueError(
                        f"{self.forward_architecture} requires interface_model.group_control_dim exactly 16."
                    )
            if self.forward_architecture == "budgeted_group_control_honf":
                budget = controlled.case_group_budget
                if budget is None or not budget.enabled:
                    raise ValueError(
                        "budgeted_group_control_honf requires an enabled interface_model.case_group_budget block."
                    )
                if int(controlled.group_count) != 12:
                    raise ValueError(
                        "budgeted_group_control_honf requires interface_model.group_count exactly 12."
                    )
                if int(controlled.group_control_dim) != 16:
                    raise ValueError(
                        "budgeted_group_control_honf requires interface_model.group_control_dim exactly 16."
                    )
            if self.forward_architecture in {
                "occupancy_adaptive_group_control_honf",
                "mass_competitive_group_control_honf",
                "sparse_incidence_group_control_honf",
                "coalesced_sparse_incidence_honf",
                "converged_identity_preserving_coalescence_honf",
                "continuous_functional_coalescence_honf",
                "adaptive_hyperedge_opening_honf",
            }:
                if controlled.case_group_budget is not None:
                    raise ValueError(
                        f"{self.forward_architecture} does not accept case_group_budget."
                    )
                if int(controlled.group_count) != 12:
                    raise ValueError(
                        f"{self.forward_architecture} requires interface_model.group_count exactly 12."
                    )
                if int(controlled.group_control_dim) != 16:
                    raise ValueError(
                        f"{self.forward_architecture} requires interface_model.group_control_dim exactly 16."
                    )
            if self.forward_architecture in {
                "adaptive_hyperedge_opening_honf",
                "coalesced_sparse_incidence_honf",
                "converged_identity_preserving_coalescence_honf",
                "continuous_functional_coalescence_honf",
            }:
                if controlled.environment_refinement_normalizer != "sparsemax":
                    raise ValueError(
                        f"{self.forward_architecture} requires "
                        "interface_model.environment_refinement_normalizer='sparsemax'."
                    )
            if self.forward_architecture == "coalesced_sparse_incidence_honf":
                expected_fusion_settings = {
                    "fusion_max_iterations": 64,
                    "fusion_eta_final": 0.5,
                    "fusion_ramp_epochs": 150,
                }
                actual_fusion_settings = {
                    name: getattr(controlled, name)
                    for name in expected_fusion_settings
                }
                if actual_fusion_settings != expected_fusion_settings:
                    raise ValueError(
                        "coalesced_sparse_incidence_honf requires the prescribed "
                        "fusion settings: max_iterations=64, eta_final=0.5, "
                        "ramp_epochs=150."
                    )
            elif self.forward_architecture == "converged_identity_preserving_coalescence_honf":
                expected_fusion_settings = {
                    "fusion_max_iterations": 512,
                    "fusion_eta_final": 0.5,
                    "fusion_ramp_epochs": 150,
                    "fusion_eps_abs": 1.0e-9,
                    "fusion_eps_rel": 1.0e-8,
                }
                actual_fusion_settings = {
                    name: getattr(controlled, name)
                    for name in expected_fusion_settings
                }
                if actual_fusion_settings != expected_fusion_settings:
                    raise ValueError(
                        "converged_identity_preserving_coalescence_honf requires "
                        "fusion settings: max_iterations=512, eps_abs=1e-9, "
                        "eps_rel=1e-8, eta_final=0.5, ramp_epochs=150."
                    )
            elif self.forward_architecture == "continuous_functional_coalescence_honf":
                if len(controlled.functional_tree_subsets) != 11:
                    raise ValueError("continuous functional coalescence requires eleven tree subsets.")
                nodes: list[frozenset[int]] = []
                for subset in controlled.functional_tree_subsets:
                    if not isinstance(subset, (list, tuple)) or len(subset) < 2:
                        raise ValueError("each functional tree subset must have at least two leaves.")
                    if any(
                        isinstance(leaf, bool)
                        or not isinstance(leaf, int)
                        or leaf not in range(12)
                        for leaf in subset
                    ):
                        raise ValueError("functional tree leaves must be integer IDs in [0, 11].")
                    node = frozenset(subset)
                    if len(node) != len(subset) or node in nodes:
                        raise ValueError("functional tree subsets must have unique leaves and nodes.")
                    nodes.append(node)
                if nodes[-1] != frozenset(range(12)):
                    raise ValueError("the final functional tree node must contain all twelve leaves.")
                leaves = [frozenset((index,)) for index in range(12)]
                for index, node in enumerate(nodes):
                    for previous in nodes[:index]:
                        if previous & node and not previous < node:
                            raise ValueError("functional tree nodes must be disjoint or bottom-up nested.")
                    candidates = [child for child in [*leaves, *nodes[:index]] if child < node]
                    children = [
                        child for child in candidates
                        if not any(child < ancestor < node for ancestor in nodes[:index])
                    ]
                    if len(children) != 2 or children[0] | children[1] != node:
                        raise ValueError("each functional tree node must have exactly two children.")
                if (
                    float(controlled.read_input_close_rms),
                    float(controlled.read_input_keep_rms),
                ) != (0.02, 0.06):
                    raise ValueError("this candidate requires fixed read-input RMS scales 0.02 and 0.06.")
            elif (
                self.forward_architecture not in {
                    "sparse_incidence_group_control_honf",
                    "coalesced_sparse_incidence_honf",
                    "converged_identity_preserving_coalescence_honf",
                    "continuous_functional_coalescence_honf",
                }
                and controlled.environment_refinement_normalizer != "entmax15"
            ):
                raise ValueError(
                    f"{self.forward_architecture} requires "
                    "interface_model.environment_refinement_normalizer='entmax15'."
                )
            if self.forward_architecture == "hypergraph_quadrature_honf" and int(controlled.samples_per_group) != 4:
                raise ValueError(
                    "hypergraph_quadrature_honf requires interface_model.samples_per_group exactly 4."
                )
        elif self.forward_architecture == "fixed_group_pairwise_honf":
            fixed = self.interface_model
            if fixed.support_spacing_factor is not None:
                raise ValueError(
                    "interface_model.support_spacing_factor is only valid for sparse_interface_honf."
                )
            if int(fixed.group_count) != 6:
                raise ValueError("fixed_group_pairwise_honf requires interface_model.group_count exactly 6.")
            if fixed.source_normalizer != "entmax15":
                raise ValueError("fixed_group_pairwise_honf requires interface_model.source_normalizer='entmax15'.")
            if fixed.query_normalizer != "entmax15":
                raise ValueError("fixed_group_pairwise_honf requires interface_model.query_normalizer='entmax15'.")
            for name in ("module_temperature", "environment_temperature", "query_temperature"):
                if float(getattr(fixed, name)) != 1.0:
                    raise ValueError(
                        f"fixed_group_pairwise_honf requires interface_model.{name}=1.0."
                    )
            if int(fixed.group_code_dim) != 32:
                raise ValueError("fixed_group_pairwise_honf requires interface_model.group_code_dim exactly 32.")
        elif self.interface_model.support_spacing_factor is not None:
            raise ValueError(
                "interface_model.support_spacing_factor is only valid for sparse_interface_honf."
            )
        if (
            self.interface_model is not None
            and self.forward_architecture not in {
                "sparse_incidence_group_control_honf",
                "coalesced_sparse_incidence_honf",
                "converged_identity_preserving_coalescence_honf",
                "continuous_functional_coalescence_honf",
                "adaptive_hyperedge_opening_honf",
            }
            and self.interface_model.environment_refinement_normalizer != "entmax15"
        ):
            raise ValueError(
                f"{self.forward_architecture} requires "
                "interface_model.environment_refinement_normalizer='entmax15'."
            )
        if self.forward_architecture == "routed_pairwise_honf":
            if self.interface_model.routing is None:
                raise ValueError("routed_pairwise_honf requires interface_model.routing.")
        elif self.interface_model is not None and self.interface_model.routing is not None:
            raise ValueError("interface_model.routing is only valid for routed_pairwise_honf.")
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
        if self.pairwise_module_token_source not in {
            "base",
            "organizer_contextualized",
        }:
            raise ValueError(
                "pairwise_module_token_source must be 'base' or "
                "'organizer_contextualized'."
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
            if self.forward_architecture not in {
                "legacy_honf",
                "dense_pairwise_field",
                "routed_pairwise_honf",
                "fixed_group_pairwise_honf",
            }:
                raise ValueError(
                    "spatial_dim=3 requires legacy_honf, dense_pairwise_field, routed_pairwise_honf, "
                    "or fixed_group_pairwise_honf."
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
        if self.pairwise_module_token_source == "base":
            payload.pop("pairwise_module_token_source", None)
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
                if self.forward_architecture not in {
                    "sparse_incidence_group_control_honf",
                    "coalesced_sparse_incidence_honf",
                    "converged_identity_preserving_coalescence_honf",
                    "continuous_functional_coalescence_honf",
                    "adaptive_hyperedge_opening_honf",
                }:
                    # The Run-1502 hook is not part of any historical
                    # architecture's serialized contract.
                    interface_payload.pop("environment_refinement_normalizer", None)
                elif self.interface_model.environment_refinement_normalizer == "entmax15":
                    # Preserve the historical Run-1501/checkpoint shape when
                    # the new hook is at its default value.
                    interface_payload.pop("environment_refinement_normalizer", None)
                if self.forward_architecture not in {
                    "coalesced_sparse_incidence_honf",
                    "converged_identity_preserving_coalescence_honf",
                }:
                    interface_payload.pop("fusion_max_iterations", None)
                    interface_payload.pop("fusion_eta_final", None)
                    interface_payload.pop("fusion_ramp_epochs", None)
                if (
                    self.forward_architecture
                    != "converged_identity_preserving_coalescence_honf"
                ):
                    interface_payload.pop("fusion_eps_abs", None)
                    interface_payload.pop("fusion_eps_rel", None)
                if self.forward_architecture != "continuous_functional_coalescence_honf":
                    interface_payload.pop("functional_tree_subsets", None)
                    interface_payload.pop("read_input_close_rms", None)
                    interface_payload.pop("read_input_keep_rms", None)
                budget_payload = interface_payload.get("case_group_budget")
                if (
                    self.forward_architecture == "budgeted_group_control_honf"
                    and isinstance(budget_payload, dict)
                    and not bool(budget_payload.get("rescue_mode", False))
                ):
                    # Appended rescue fields must not be injected into old
                    # v1 model_config payloads when they are resaved.
                    for key in (
                        "rescue_mode",
                        "schedule",
                        "routing_initial_scale",
                        "routing_full_epoch",
                        "compression_start_epoch",
                        "hardening_epoch",
                    ):
                        budget_payload.pop(key, None)
                if self.interface_model.routing is None:
                    interface_payload.pop("routing", None)
                elif self.interface_model.routing.strategy != "mean_shift":
                    # Keep the historical module-hub serialization shape
                    # stable; these fields are meaningful only for Run 2100.
                    routing_payload = interface_payload.get("routing")
                    if isinstance(routing_payload, dict):
                        routing_payload.pop("mean_shift_steps", None)
                        routing_payload.pop("mean_shift_feature_bandwidth", None)
                routing_payload = interface_payload.get("routing")
                if isinstance(routing_payload, dict):
                    # A disabled, zero-weight sparsification block is an
                    # additive historical default.  Omit it, and the default
                    # Torch QE selector alongside it, so old checkpoints and
                    # profiles retain their established routing shape.  Keep
                    # an explicit QE selector on the enabled science profile
                    # even when it requests the default Torch implementation;
                    # the requested backend is part of that experiment's
                    # reproducible configuration.
                    sparse_payload = routing_payload.get("sparsification")
                    default_sparse = isinstance(sparse_payload, dict) and sparse_payload == {
                        "enabled": False,
                        "learn_typed_temperatures": False,
                        "relative_density_epsilon": 0.05,
                        "cost_weight": 0.0,
                        "module_pair_cost": 1.0,
                        "environment_pair_cost": 1.0,
                    }
                    if default_sparse:
                        routing_payload.pop("sparsification", None)
                        if routing_payload.get("qe_backend") == "torch":
                            routing_payload.pop("qe_backend", None)
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
                if self.forward_architecture not in {
                    "fixed_group_pairwise_honf",
                    "group_control_pairwise_honf",
                    "phase_shared_group_control_honf",
                    "hypergraph_quadrature_honf",
                    "budgeted_group_control_honf",
                    "occupancy_adaptive_group_control_honf",
                    "mass_competitive_group_control_honf",
                    "sparse_incidence_group_control_honf",
                    "coalesced_sparse_incidence_honf",
                    "converged_identity_preserving_coalescence_honf",
                    "continuous_functional_coalescence_honf",
                    "adaptive_hyperedge_opening_honf",
                }:
                    for key in (
                        "group_count",
                        "source_normalizer",
                        "query_normalizer",
                        "module_temperature",
                        "environment_temperature",
                        "query_temperature",
                        "group_code_dim",
                        "group_control_dim",
                        "samples_per_group",
                        "case_group_budget",
                    ):
                        interface_payload.pop(key, None)
                elif self.forward_architecture == "fixed_group_pairwise_honf":
                    # The fixed-group reader has no common coarse bank or
                    # local-neighbour branch.  Keep the serialized profile
                    # compact while defaults remain available to old callers.
                    for key in (
                        "coarse_latent_count",
                        "coarse_blocks",
                        "main_latent_count",
                        "main_latent_blocks",
                        "local_radius_factor",
                        "group_read_mode",
                        "response_region_block_shape",
                        "response_tree_opening_interval",
                        "coarse_module_source",
                        "group_control_dim",
                        "samples_per_group",
                    ):
                        interface_payload.pop(key, None)
                else:
                    # Run 1406 uses one low-dimensional control/code width;
                    # do not serialize the fixed-group reader's redundant
                    # H-wide code selector under the new architecture.  The
                    # three-term reader also has no coarse/local bank.
                    for key in (
                        "coarse_latent_count",
                        "coarse_blocks",
                        "main_latent_count",
                        "main_latent_blocks",
                        "local_radius_factor",
                        "group_read_mode",
                        "response_region_block_shape",
                        "response_tree_opening_interval",
                        "coarse_module_source",
                    ):
                        interface_payload.pop(key, None)
                    interface_payload.pop("group_code_dim", None)
                    if self.forward_architecture != "hypergraph_quadrature_honf":
                        interface_payload.pop("samples_per_group", None)
                    if self.forward_architecture != "budgeted_group_control_honf":
                        interface_payload.pop("case_group_budget", None)
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
    # Runtime case geometry; never stored as an executable checkpoint callback.
    routing_geometry: Optional[Any] = None
    # Optional adapter-owned regular-grid metadata for the sampled
    # environmental reader. Appended to preserve positional compatibility.
    sampler_layout: Optional[Any] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BatchData":
        """Validate and construct the canonical batch container."""

        return _dataclass_from_dict(cls, payload)

    def to_dict(self) -> Dict[str, Any]:
        """Describe batch values as serializable metadata."""

        payload = _to_plain_dict(self)
        # Keep the historical absent-measure representation unchanged.  A
        # supplied tensor is still described explicitly for adapter diagnostics.
        payload.pop("routing_geometry", None)
        if self.sampler_layout is None:
            payload.pop("sampler_layout", None)
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
