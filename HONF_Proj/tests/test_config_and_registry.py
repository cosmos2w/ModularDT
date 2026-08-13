from __future__ import annotations

import json

import pytest

from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin, require_model_family


def test_split_config_composes_deterministically() -> None:
    first = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    second = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    assert first.config_hash == second.config_hash
    assert first.effective["model"]["core_honf"]["decoder_mode"] == "enhanced_honf_pairwise"
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
