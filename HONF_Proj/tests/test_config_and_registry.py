from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin, require_model_family


def test_forward_profile_registry_is_complete_and_keeps_metadata_out_of_profiles() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry_path = project_root / "src/config_core/forward/profile_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    profiles = registry["profiles"]
    names = [profile["name"] for profile in profiles]
    assert len(names) == len(set(names)) == 25
    assert registry["recommended_forward_profile"] == "stage7_structured_context"
    assert {profile["status"] for profile in profiles} <= {
        "current",
        "compatibility",
        "frozen_experiment",
        "evaluation_only",
        "candidate",
        "promoted_execution",
        "rejected",
        "completed_research",
    }
    by_name = {profile["name"]: profile for profile in profiles}
    assert by_name["stage7_structured_context"]["status"] == "current"
    assert by_name["stage7_fused_query_module"]["status"] == "promoted_execution"
    assert by_name["stage7_factorized_gated_r96"]["status"] == "rejected"
    assert by_name["stage7_k4_fused_audit"]["status"] == "completed_research"
    assert by_name["stage7_k8_fused_audit"]["status"] == "rejected"
    assert by_name["enhanced_honf_pairwise"]["status"] == "compatibility"
    assert by_name["stage5_fixed_residual_concat_uniform_lr3e4"]["status"] == "frozen_experiment"
    assert by_name["case_adaptive_residual_context"]["status"] == "candidate"
    assert by_name["case_adaptive_tensor_residual_context"]["status"] == "candidate"
    assert registry["recommended_forward_profile"] != "case_adaptive_residual_context"
    assert registry["recommended_forward_profile"] != "case_adaptive_tensor_residual_context"

    for profile in profiles:
        path = project_root / profile["path"].removeprefix("project://")
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "status" not in payload
        assert "checkpoint_compatibility" not in payload
        if profile["base"] is None:
            load_config_bundle(profile["path"])
        else:
            load_config_bundle(
                by_name[profile["base"]]["path"],
                experiment_overlay=profile["path"],
            )


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
    assert bundle.effective["training"]["plot_every_epochs"] == 50
    assert bundle.effective["checkpointing"]["save_latest_every_epochs"] == 10


def test_stage7_structured_context_profile_is_explicit_run1000_style() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/stage7_structured_context.json"
    )
    core = bundle.effective["model"]["core_honf"]

    assert core["organizer_mode"] == "fixed_projection"
    assert (core["num_hyperedges"], core["edge_capacity"]) == (6, 0)
    assert core["edge_selection_mode"] == "all"
    assert core["module_assignment_normalizer"] == "softmax"
    assert core["environment_assignment_normalizer"] == "softmax"
    assert core["query_assignment_normalizer"] == "softmax"
    assert core["environment_locality_mode"] == "none"
    assert core["query_locality_mode"] == "none"
    assert core["mechanism_state_mode"] == "residual_concat"
    assert core["use_hyper_mechanism_encoder"] is False
    assert core["field_assembly_mode"] == "context_fusion"
    assert core["routing_execution"] == "dense"
    assert (core["query_edge_limit"], core["query_module_limit"]) == (0, 0)
    assert core["pairwise_aggregation_mode"] == "edge_explicit"
    assert core["query_module_retained_mass_floor"] == 1.0
    assert core["pairwise_kernel_mode"] == "legacy_mlp"
    assert core["use_hyper_value_context"] is True
    assert core["hyper_query_attention_mode"] == "learned"
    assert core["hyper_attention_topk"] == 0
    assert core["hyper_attention_temperature"] == 1.0
    assert core["use_hyper_geometry_bias"] is True
    assert core["hyper_geometry_bias_scale"] == 1.0
    assert core["use_A_me_auxiliary"] is True
    assert core["direct_residual_gate_init"] == 0.0
    assert core["output_mean_residual_split"] is False

    training = bundle.effective["training"]
    assert training["learning_rate"] == 3.0e-4
    assert training["organizer_learning_rate"] is None
    assert training["weight_decay"] == 1.0e-5
    assert training["plot_every_epochs"] == 50
    assert bundle.effective["loss"]["organizer_regularization"]["enabled"] is False
    assert bundle.effective["checkpointing"]["save_latest_every_epochs"] == 10
    assert bundle.effective["checkpointing"]["save_epoch_milestones"] == [
        500, 1000, 2500, 5000, 7500, 10000
    ]


def test_stage7_enhancement_overlays_are_minimal_and_strict() -> None:
    fused = load_config_bundle(
        "project://src/config_core/forward/stage7_structured_context.json",
        experiment_overlay=(
            "project://src/config_core/forward/experiments/"
            "stage7_fused_query_module.json"
        ),
    )
    fused_core = fused.effective["model"]["core_honf"]
    assert fused_core["num_hyperedges"] == 6
    assert fused_core["pairwise_aggregation_mode"] == "fused_query_module"
    assert fused_core["pairwise_kernel_mode"] == "legacy_mlp"
    assert fused_core["routing_execution"] == "dense"
    assert fused_core["query_module_retained_mass_floor"] == 1.0
    assert fused_core["query_module_limit"] == 0
    assert fused.experiment["core"] == {
        "model": {
            "core_honf": {
                "num_hyperedges": 6,
                "pairwise_aggregation_mode": "fused_query_module",
                "query_module_retained_mass_floor": 1.0,
                "pairwise_kernel_mode": "legacy_mlp",
                "routing_execution": "dense",
                "query_module_limit": 0,
            }
        }
    }
    assert fused.effective["training"]["epochs"] == 5000

    factorized = load_config_bundle(
        "project://src/config_core/forward/stage7_structured_context.json",
        experiment_overlay=(
            "project://src/config_core/forward/experiments/"
            "stage7_factorized_gated_r96.json"
        ),
    )
    factorized_core = factorized.effective["model"]["core_honf"]
    assert factorized_core["pairwise_aggregation_mode"] == "fused_query_module"
    assert factorized_core["pairwise_kernel_mode"] == "factorized_gated"
    assert factorized_core["pairwise_kernel_hidden_dim"] == 96
    assert factorized_core["pairwise_kernel_num_layers"] == 2
    assert factorized.effective["training"]["epochs"] == 500
    assert factorized.effective["checkpointing"]["save_epoch_milestones"] == [500]


@pytest.mark.parametrize(
    ("overlay", "num_hyperedges"),
    [
        ("stage7_k4_fused_audit.json", 4),
        ("stage7_k8_fused_audit.json", 8),
    ],
)
def test_stage7_k_audit_overlays_change_only_k_and_requested_milestones(
    overlay: str,
    num_hyperedges: int,
) -> None:
    base_path = "project://src/config_core/forward/stage7_structured_context.json"
    fused = load_config_bundle(
        base_path,
        experiment_overlay=(
            "project://src/config_core/forward/experiments/"
            "stage7_fused_query_module.json"
        ),
    )
    audit = load_config_bundle(
        base_path,
        experiment_overlay=f"project://src/config_core/forward/experiments/{overlay}",
    )

    expected_overlay_core = {
        "num_hyperedges": num_hyperedges,
        "pairwise_aggregation_mode": "fused_query_module",
        "query_module_retained_mass_floor": 1.0,
        "pairwise_kernel_mode": "legacy_mlp",
        "routing_execution": "dense",
        "query_module_limit": 0,
    }
    assert audit.experiment["core"] == {
        "model": {"core_honf": expected_overlay_core},
        "checkpointing": {"save_epoch_milestones": [500, 2500, 5000]},
    }

    fused_effective = copy.deepcopy(fused.effective)
    fused_effective["model"]["core_honf"]["num_hyperedges"] = num_hyperedges
    fused_effective["checkpointing"]["save_epoch_milestones"] = [500, 2500, 5000]
    assert audit.effective == fused_effective
    assert audit.effective["training"]["epochs"] == 5000
    assert audit.effective["training"]["seed"] == 0


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


@pytest.mark.parametrize(
    ("overlay", "organizer_mode", "edge_capacity"),
    [
        ("stage5_exchangeable_soft_organized.json", "exchangeable_slots", 6),
        ("stage5_fixed_softmax_modern.json", "fixed_projection", 0),
    ],
)
def test_stage5_profiles_are_matched_all_soft_dense_models(
    overlay: str,
    organizer_mode: str,
    edge_capacity: int,
) -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/adaptive_sparse_additive.json",
        experiment_overlay=f"project://src/config_core/forward/experiments/{overlay}",
    )
    core = bundle.effective["model"]["core_honf"]

    assert core["organizer_mode"] == organizer_mode
    assert core["num_hyperedges"] == 6
    assert core["edge_capacity"] == edge_capacity
    assert core["edge_selection_mode"] == "all"
    assert (core["selection_start_epoch"], core["selection_transition_epochs"]) == (-1, 0)
    for prefix in ("module", "environment", "query"):
        assert core[f"{prefix}_assignment_normalizer"] == "softmax"
        assert core[f"{prefix}_sparsity_start_epoch"] == -1
        assert core[f"{prefix}_sparsity_transition_epochs"] == 0
    assert core["environment_locality_mode"] == "none"
    assert core["query_locality_mode"] == "none"
    assert core["mechanism_state_mode"] == "descriptor_first"
    assert core["field_assembly_mode"] == "edge_additive"
    assert core["additive_background_mode"] == "dense_query_attention"
    assert core["routing_execution"] == "dense"
    assert core["query_edge_limit"] == 0
    assert core["query_module_limit"] == 0
    assert bundle.effective["training"]["learning_rate"] == 3.0e-4
    assert bundle.effective["training"]["organizer_learning_rate"] == 1.0e-4
    assert bundle.effective["checkpointing"]["save_epoch_milestones"] == [
        250, 500, 1000, 1500, 2500
    ]


def test_stage6_frozen_screen_profile_resolves_to_supported_s0_control() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/adaptive_sparse_additive.json",
        experiment_overlay=(
            "project://src/config_core/forward/experiments/"
            "stage6_fixed_role_consistent_additive.json"
        ),
    )
    core = bundle.effective["model"]["core_honf"]

    assert core["organizer_mode"] == "fixed_projection"
    assert core["num_hyperedges"] == 6
    assert core["edge_capacity"] == 0
    assert core["edge_selection_mode"] == "all"
    assert core["module_assignment_normalizer"] == "softmax"
    assert core["environment_assignment_normalizer"] == "softmax"
    assert core["query_assignment_normalizer"] == "softmax"
    assert core["query_locality_mode"] == "none"
    assert core["query_locality_strength"] is None
    assert core["mechanism_state_mode"] == "descriptor_first"
    assert core["mechanism_latent_residual_scale"] == 0.35
    assert core["field_assembly_mode"] == "edge_additive"
    assert core["additive_background_mode"] == "dense_query_attention"
    assert core["routing_execution"] == "dense"
    assert bundle.effective["training"]["learning_rate"] == 3.0e-4
    assert bundle.effective["training"]["organizer_learning_rate"] == 1.0e-4


@pytest.mark.parametrize("milestones", [[250, 250], [0], [True], "250,500"])
def test_checkpoint_milestones_reject_non_unique_positive_integer_lists(
    tmp_path,
    milestones,
) -> None:
    source = load_config_bundle("project://src/config_core/forward/adaptive_sparse_additive.json")
    payload = copy.deepcopy(source.core)
    payload["case"]["config"] = str(source.case_source)
    payload["checkpointing"]["save_epoch_milestones"] = milestones
    path = tmp_path / "invalid_milestones.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="save_epoch_milestones"):
        load_config_bundle(path)


def test_organizer_learning_rate_must_be_null_or_positive(tmp_path) -> None:
    source = load_config_bundle("project://src/config_core/forward/adaptive_sparse_additive.json")
    payload = copy.deepcopy(source.core)
    payload["case"]["config"] = str(source.case_source)
    payload["training"]["organizer_learning_rate"] = 0.0
    path = tmp_path / "invalid_optimizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="organizer_learning_rate"):
        load_config_bundle(path)
