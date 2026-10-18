"""Strict case-local pilot configuration."""
from __future__ import annotations
from pathlib import Path
import yaml

REQUIRED = {
    "rng": {"root_seed", "key_hash", "pilot_namespace_disjoint_from_formal"},
    "physics": {"domain", "sheet_model", "lateral_electrical_bc", "lateral_thermal_bc", "contact_heat_fraction_to_sheet", "temperature_dependent_coefficients", "units"},
    "environment": {"tile_shape", "reconstruction", "family_weights", "electrical_contrast_bound", "thermal_contrast_bound", "cooling_level", "cooling_contrast_bound", "latent_correlation_modes"},
    "contacts": {"counts", "radius", "total_conductance", "footprint", "quadrature_order_per_axis", "edge_extra_clearance", "interpad_extra_clearance", "core_voltage_range"},
    "grid": {"first_representative", "refinement_levels", "optional_reference"},
    "solver": {"method", "preconditioner", "precision_policy", "fast_true_rtol", "fast_absolute_residual_tolerance", "reference_true_rtol", "reference_absolute_residual_tolerance", "max_iterations", "max_refinement_corrections", "stopping_check_interval", "near_zero_power_absolute_tolerance", "nonnegative_absolute_tolerance"},
    "pilot": {"representative_count", "counterfactual_quartets", "throughput_states", "batch_sizes", "timing_repeats", "max_distinct_states", "max_coupled_evaluations", "max_linear_system_attempts"},
    "resources": {"physical_gpu", "required_gpu_uuid", "min_free_gpu_gib", "min_free_gpu_fraction", "initial_process_allocation_cap_gib", "elapsed_budget_hours", "gpu_associated_budget_hours", "retained_budget_gib", "temporary_budget_gib", "minimum_free_disk_gib", "data2_base"},
}

def load_config(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    expected = set(REQUIRED) | {"schema", "formal_generation_authorized", "ml_training_authorized"}
    if set(data) != expected:
        raise ValueError(f"configuration top-level mismatch: {set(data) ^ expected}")
    for section, keys in REQUIRED.items():
        if set(data[section]) != keys:
            raise ValueError(f"configuration {section} mismatch: {set(data[section]) ^ keys}")
    if data["schema"] != "hetmf_pilot_config_v1" or data["formal_generation_authorized"] or data["ml_training_authorized"]:
        raise ValueError("invalid scope gates")
    if data["resources"]["physical_gpu"] != 2 or data["resources"]["required_gpu_uuid"] != "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39":
        raise ValueError("GPU2 identity mismatch")
    if data["pilot"]["max_distinct_states"] < 292:
        raise ValueError("pilot state ceiling below declared scope")
    if min(data["solver"]["fast_true_rtol"], data["solver"]["reference_true_rtol"]) <= 0:
        raise ValueError("invalid numerical tolerance")
    return data
