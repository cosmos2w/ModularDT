from __future__ import annotations

import copy
import json

import pytest

from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin, require_model_family


def test_split_config_composes_deterministically() -> None:
    first = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    second = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    assert first.config_hash == second.config_hash
    assert first.effective["model"]["core_honf"]["decoder_mode"] == "enhanced_honf_pairwise"
    assert "max_num_modules" not in first.effective["model"]["core_honf"]
    assert first.effective["case"]["selection"]["dataset_id"] == "thermal_channel_global_v1"


def test_case_plugin_loads_without_core_case_branch() -> None:
    bundle = load_config_bundle("project://src/config_core/forward/hyper_plus_global_near.json")
    plugin = load_case_plugin(bundle.case["plugin"])
    assert plugin.case_id == "ThermalChannel"
    assert plugin.version


def test_unknown_nested_core_setting_fails_before_dispatch(tmp_path) -> None:
    source = load_config_bundle("project://src/config_core/forward/hyper_plus_global_near.json")
    payload = dict(source.core)
    payload["case"] = dict(payload["case"])
    payload["case"]["config"] = str(source.case_source)
    payload["training"] = dict(payload["training"])
    payload["training"]["learnng_rate"] = payload["training"].pop("learning_rate")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="learnng_rate"):
        load_config_bundle(path)


def test_model_family_registry_rejects_reserved_and_unknown_families() -> None:
    assert require_model_family("honf_forward", "forward").available
    with pytest.raises(NotImplementedError):
        require_model_family("honf_inverse", "inverse")
    with pytest.raises(ValueError, match="Unknown model_family"):
        require_model_family("imaginary_baseline", "forward")


def test_strict_experiment_overlay_changes_only_declared_keys(tmp_path) -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/enhanced_honf_pairwise.json",
        experiment_overlay="project://src/config_core/forward/experiments/uniform_h_assignment.json",
    )
    core = bundle.effective["model"]["core_honf"]
    assert core["decoder_mode"] == "enhanced_honf_pairwise_only"
    assert core["hyper_module_assignment_mode"] == "uniform"
    assert bundle.experiment_source is not None
    assert bundle.core_source_payload["model"]["core_honf"]["decoder_mode"] == "enhanced_honf_pairwise"

    invalid = tmp_path / "invalid_overlay.json"
    invalid.write_text(
        '{"schema_version":1,"core":{"model":{"core_honf":{"invented":true}}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot introduce"):
        load_config_bundle(
            "project://src/config_core/forward/enhanced_honf_pairwise.json",
            experiment_overlay=invalid,
        )


def test_global_only_overlay_disables_local_dependency() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/enhanced_honf_pairwise.json",
        experiment_overlay="project://src/config_core/forward/experiments/global_only.json",
    )
    assert bundle.case["model"]["local_coupling"]["use_local_surrogate"] is False
    assert bundle.case["model"]["channelthermal"]["internal_prediction_mode"] == "global_head"


def test_stage1_fixed_additive_overlay_changes_only_the_scientific_bridge() -> None:
    base = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    bundle = load_config_bundle(
        "project://src/config_core/forward/enhanced_honf_pairwise.json",
        experiment_overlay="project://src/config_core/forward/experiments/stage1_fixed_additive_soft.json",
    )
    core = bundle.effective["model"]["core_honf"]
    expected = {
        "organizer_mode": "fixed_projection",
        "num_hyperedges": 6,
        "mechanism_state_mode": "descriptor_first",
        "field_assembly_mode": "edge_additive",
        "module_assignment_normalizer": "softmax",
        "environment_assignment_normalizer": "softmax",
        "query_assignment_normalizer": "softmax",
        "environment_locality_mode": "none",
        "routing_execution": "dense",
        "query_edge_limit": 0,
        "query_module_limit": 0,
        "edge_selection_mode": "all",
    }
    assert {key: core[key] for key in expected} == expected
    assert bundle.effective["training"]["learning_rate"] == 1.0e-4
    assert bundle.case == base.case
    assert bundle.effective["loss"] == base.effective["loss"]
    assert bundle.effective["dataset"] == base.effective["dataset"]


def test_stage2_exchangeable_soft_overlay_changes_only_organizer_parameterization() -> None:
    base = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    bundle = load_config_bundle(
        "project://src/config_core/forward/enhanced_honf_pairwise.json",
        experiment_overlay="project://src/config_core/forward/experiments/stage2_exchangeable_soft.json",
    )
    core = bundle.effective["model"]["core_honf"]
    expected = {
        "organizer_mode": "exchangeable_slots",
        "edge_capacity": 6,
        "initial_active_edges": 6,
        "minimum_active_edges": 1,
        "slot_refinement_steps": 2,
        "edge_selection_mode": "all",
        "module_assignment_normalizer": "softmax",
        "environment_assignment_normalizer": "softmax",
        "query_assignment_normalizer": "softmax",
        "environment_locality_mode": "none",
        "mechanism_state_mode": "descriptor_first",
        "field_assembly_mode": "edge_additive",
        "routing_execution": "dense",
        "query_edge_limit": 0,
        "query_module_limit": 0,
    }
    assert {key: core[key] for key in expected} == expected
    assert bundle.effective["training"]["learning_rate"] == 1.0e-4
    assert bundle.case == base.case
    assert bundle.effective["loss"] == base.effective["loss"]
    assert bundle.effective["dataset"] == base.effective["dataset"]


def test_stage3_scheduled_profile_has_staggered_transitions_and_delayed_gathering() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/adaptive_sparse_additive.json",
        experiment_overlay="project://src/config_core/forward/experiments/stage3_scheduled_adaptive_sparse.json",
    )
    core = bundle.effective["model"]["core_honf"]

    assert core["edge_capacity"] == 8
    assert core["minimum_active_edges"] == 2
    assert (core["selection_start_epoch"], core["selection_transition_epochs"]) == (150, 250)
    assert core["selection_warmup_mode"] == "all_viable"
    assert core["selection_coverage_rate"] == 0.99
    assert (core["query_sparsity_start_epoch"], core["query_sparsity_transition_epochs"]) == (250, 250)
    assert (core["module_sparsity_start_epoch"], core["module_sparsity_transition_epochs"]) == (350, 300)
    assert (core["environment_sparsity_start_epoch"], core["environment_sparsity_transition_epochs"]) == (350, 300)
    assert core["environment_locality_mode"] == "gaussian_bounded"
    assert core["environment_locality_strength"] == 0.25
    assert core["query_locality_mode"] == "none"
    assert core["routing_execution"] == "scheduled"
    assert core["gathered_execution_start_epoch"] == 650
    assert core["query_edge_retained_mass_floor"] == 0.98
    assert core["module_incidence_retained_mass_floor"] == 0.95
    assert bundle.effective["training"]["learning_rate"] == 1.0e-4


def test_adaptive_sparse_additive_base_is_the_dense_formal_stage3_profile() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/adaptive_sparse_additive.json"
    )
    core = bundle.effective["model"]["core_honf"]

    assert (core["edge_capacity"], core["initial_active_edges"], core["minimum_active_edges"]) == (8, 8, 2)
    assert core["additive_background_mode"] == "dense_query_attention"
    assert bundle.effective["training"]["learning_rate"] == 1.0e-4
    assert bundle.effective["training"]["organizer_learning_rate"] is None
    assert (core["selection_start_epoch"], core["selection_transition_epochs"]) == (150, 250)
    assert core["selection_warmup_mode"] == "all_viable"
    assert core["selection_coverage_rate"] == 0.99
    assert core["selection_minimum_module_mass_fraction"] == 0.01
    assert core["selection_minimum_environment_mass_fraction"] == 0.01
    assert core["candidate_module_mass_fraction_floor"] == 0.01
    assert core["candidate_environment_mass_fraction_floor"] == 0.01
    assert core["module_assignment_normalizer"] == "scheduled"
    assert core["environment_assignment_normalizer"] == "scheduled"
    assert core["query_assignment_normalizer"] == "scheduled"
    assert (core["module_sparsity_start_epoch"], core["module_sparsity_transition_epochs"]) == (350, 300)
    assert (core["environment_sparsity_start_epoch"], core["environment_sparsity_transition_epochs"]) == (350, 300)
    assert (core["query_sparsity_start_epoch"], core["query_sparsity_transition_epochs"]) == (250, 250)
    assert core["environment_locality_mode"] == "gaussian_bounded"
    assert core["environment_locality_strength"] == 0.25
    assert core["minimum_region_scale"] == 0.10
    assert core["query_locality_mode"] == "none"
    assert core["routing_execution"] == "dense"
    assert core["query_edge_retained_mass_floor"] == 0.98
    assert core["module_incidence_retained_mass_floor"] == 0.95


@pytest.mark.parametrize(
    ("overlay", "background_mode", "learning_rate", "organizer_learning_rate"),
    [
        (
            "stage4_uniform_lr2e4_dense_background.json",
            "dense_query_attention",
            2.0e-4,
            None,
        ),
        (
            "stage4_split_lr_dense_background.json",
            "dense_query_attention",
            3.0e-4,
            1.0e-4,
        ),
        (
            "stage4_split_lr_pooled_background.json",
            "global_pooled_attention",
            3.0e-4,
            1.0e-4,
        ),
    ],
)
def test_stage4_overlays_resolve_strictly_with_expected_provenance(
    overlay: str,
    background_mode: str,
    learning_rate: float,
    organizer_learning_rate: float | None,
) -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/adaptive_sparse_additive.json",
        experiment_overlay=f"project://src/config_core/forward/experiments/{overlay}",
    )
    assert bundle.effective["model"]["core_honf"]["additive_background_mode"] == background_mode
    assert bundle.effective["training"]["learning_rate"] == learning_rate
    assert bundle.effective["training"]["organizer_learning_rate"] == organizer_learning_rate
    assert bundle.experiment_source is not None
    assert bundle.experiment_source.name == overlay
    assert bundle.experiment["core"]["training"]["learning_rate"] == learning_rate


def test_organizer_learning_rate_must_be_null_or_positive(tmp_path) -> None:
    source = load_config_bundle("project://src/config_core/forward/adaptive_sparse_additive.json")
    payload = copy.deepcopy(source.core)
    payload["case"]["config"] = str(source.case_source)
    payload["training"]["organizer_learning_rate"] = 0.0
    path = tmp_path / "invalid_optimizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="organizer_learning_rate"):
        load_config_bundle(path)
