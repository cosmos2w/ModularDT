"""Response-aware fitting utilities for the ThermalChannel absolute operator."""

from .algebra import MixedResponseSpec, StencilPredictions, predict_stencil
from .contracts import AbsoluteOperator, AbsolutePrediction, DesignInput, RoleQuery, role_queries_from_record
from .derivative_check import check_pressure_peak_ad_fd
from .evaluation import (
    ChannelMetric,
    PressureMetric,
    SolidPeakMetric,
    SolidPeakSummaryMetric,
    evaluate_absolute_record,
    evaluate_stencil,
)
from .losses import (
    StencilLossTerms,
    ThermalLossScales,
    compute_stencil_loss_terms,
    weighted_masked_mse,
)
from .native import DifferentiableThermalOperator
from .paired import PairedFitResult, run_paired_staged_fits, write_paired_training_curves
from .sampling import (
    ReceiverSamplingConfig,
    SampledResponseStencil,
    SamplingSummary,
    sample_training_panel,
    sample_training_stencil,
)
from .thermal import (
    NativeThermalQuantities,
    module_peak_temperatures_from_role,
    pressure_drop_from_field,
    pressure_section_masks,
    reduce_native_thermal_quantities,
    smooth_module_peak,
    target_module_peak_slots,
)
from .training import (
    StagedFitResult,
    StagedTrainingConfig,
    TrainingCheckpointError,
    TrainingStage,
    calibrate_gradient_weights,
    calibrate_operator_weights,
    checkpoint_payload,
    load_staged_training_config,
    restore_checkpoint_payload,
    run_staged_fit,
)

__all__ = [
    "AbsoluteOperator",
    "AbsolutePrediction",
    "ChannelMetric",
    "DesignInput",
    "DifferentiableThermalOperator",
    "MixedResponseSpec",
    "NativeThermalQuantities",
    "PairedFitResult",
    "PressureMetric",
    "ReceiverSamplingConfig",
    "RoleQuery",
    "SampledResponseStencil",
    "SamplingSummary",
    "SolidPeakMetric",
    "SolidPeakSummaryMetric",
    "StagedFitResult",
    "StagedTrainingConfig",
    "StencilLossTerms",
    "StencilPredictions",
    "ThermalLossScales",
    "TrainingCheckpointError",
    "TrainingStage",
    "calibrate_gradient_weights",
    "calibrate_operator_weights",
    "check_pressure_peak_ad_fd",
    "checkpoint_payload",
    "compute_stencil_loss_terms",
    "evaluate_absolute_record",
    "evaluate_stencil",
    "load_staged_training_config",
    "module_peak_temperatures_from_role",
    "predict_stencil",
    "pressure_drop_from_field",
    "pressure_section_masks",
    "reduce_native_thermal_quantities",
    "restore_checkpoint_payload",
    "role_queries_from_record",
    "run_paired_staged_fits",
    "run_staged_fit",
    "sample_training_panel",
    "sample_training_stencil",
    "smooth_module_peak",
    "target_module_peak_slots",
    "weighted_masked_mse",
    "write_paired_training_curves",
]
