"""Development preparation contracts; no native models, checkpoints or launches."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("thermal_development", TOOLS / "thermal_development.py")
development = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(development)


def manifest(train_ids=("0001", "0304", "0318", "0320", "0333")):
    # Profile composition accepts a manifest already validated by the canonical loader.
    return {"manifest_sha256": "a" * 64,
            "partitions": {"train": {"case_ids": list(train_ids)}, "test": {"case_ids": ["0601", "0602"]}}}


def test_five_bound_profiles_share_full_selected_budget_horizon_and_response_pool():
    profiles = development.development_profiles(manifest=manifest(), stage=500, manifest_path="/data/shared.json")
    common = []
    for config in profiles.values():
        assert config["case"]["dataset"] == {
            "development_manifest": "/data/shared.json", "development_manifest_sha256": "a" * 64}
        training = config["training"]
        assert training["epochs"] == 500 and training["seed"] == 0 and training["amp"] is False
        assert training["max_train_batches_per_epoch"] is None and training["max_val_batches"] is None
        assert training["campaign"]["schedule_total_epochs"] == 1000
        assert training["campaign"]["microbatch_size"] == 8 and training["campaign"]["require_full_epoch"]
        assert len(training["campaign"]["response_stencils"]) == 4
        assert all(Path(p).name.removeprefix("train_").removesuffix("_responses.npz")
                   in manifest()["partitions"]["train"]["case_ids"] for p in training["campaign"]["response_stencils"])
        assert config["checkpointing"] == {
            "save_best": False, "save_best_field_mse": True, "save_best_temperature_mse": False,
            "save_best_predicted": False, "save_latest": True, "save_latest_every_epochs": 100,
            "save_epoch_milestones": list(range(100, 1001, 100)), "save_best_every_epochs": 100}
        assert training["plot_every_epochs"] == 100
        comparable = copy.deepcopy(config)
        comparable.pop("profile_name")
        comparable["model"]["core_honf"].pop("forward_architecture")
        comparable["training"]["campaign"].pop("arm")
        comparable.pop("run")
        common.append(comparable)
    assert all(config == common[0] for config in common)


def test_zero_response_anchor_allows_initial100_but_refuses_continuation():
    selected = manifest(("0002", "0003"))
    first = development.development_profiles(manifest=selected)
    assert all(not config["training"]["campaign"]["response_stencils"] for config in first.values())
    for stage in (500, 1000):
        with pytest.raises(ValueError, match="selected-train response"):
            development.development_profiles(manifest=selected, stage=stage)
    with pytest.raises(ValueError, match="100, 500 or 1000"):
        development.development_profiles(stage=5000)


def test_faithfulness_named_profiles_have_bounded_candidates_and_separate_dense_budget():
    initial = development.faithfulness_profiles(manifest=manifest(("0348",)))
    assert set(initial) == {"Tree-L", "Tree-F", "Pair-F", "Dense-new"}
    assert initial["Tree-F"]["run"]["id"] == "3201"
    assert initial["Pair-F"]["run"]["id"] == "3202"
    for arm in ("Tree-F", "Pair-F"):
        assert initial[arm]["case"]["config"].endswith("case_source_local.json")
        assert initial[arm]["training"]["epochs"] == 100
    assert "channelthermal" not in initial["Tree-L"]["model"]
    for age in (200, 300, 400, 500):
        extended = development.faithfulness_profiles(stage=age, manifest=manifest(("0348",)))
        assert "Tree-L" not in extended
        for arm in ("Tree-F", "Pair-F"):
            campaign = extended[arm]["training"]["campaign"]
            assert campaign["schedule_total_epochs"] == 1000
            assert campaign["physical_loss_policy_version"] == 2
            assert campaign["native_loss_denominators_start_epoch"] == 101
            assert len(campaign["response_stencils"]) == 1
    dense = development.faithfulness_profiles(stage=1000, manifest=manifest(("0348",)))
    assert set(dense) == {"Dense-new"}
    assert dense["Dense-new"]["training"]["epochs"] == 1000


def test_fine_maturation_control_has_source_local_inputs_and_common_auxiliary_schedule():
    from channelthermal.training.campaign import validate_campaign


    config = development.maturation_profiles(manifest=manifest(("0348",)), stage=100)["Fine-F"]
    assert config["model"]["core_honf"]["forward_architecture"] == "three_term_full_access_honf"
    assert config["case"]["config"].endswith("case_source_local.json")
    assert config["run"]["id"] == "3301"
    campaign = config["training"]["campaign"]
    assert campaign["structural_weight"] == 0.0
    assert campaign["schedule_total_epochs"] == 1000
    assert campaign["native_loss_denominators_start_epoch"] == 101
    assert campaign["heat_null_response"] == {"benchmark_verified": True, "cases_per_epoch": 2,
                                              "fluid_queries": 256, "fraction": .1}
    assert len(campaign["response_stencils"]) == 1
    # Native schema restrictions remain separate from profile preparation.
    effective = copy.deepcopy(config)
    effective["model"]["channelthermal"] = {"global_feature_schema": "source_local_v3"}
    validate_campaign(effective)
    effective["model"]["channelthermal"]["global_feature_schema"] = "legacy"
    with pytest.raises(ValueError, match="source_local_v3"):
        validate_campaign(effective)


def test_limited_available_response_diversity_is_not_expanded_by_excluded_cases():
    profiles = development.development_profiles(manifest=manifest(("0304",)), stage=1000)
    assert all(len(config["training"]["campaign"]["response_stencils"]) == 1 for config in profiles.values())
    assert all(config["training"]["campaign"]["response_stencils"][0].endswith("train_0304_responses.npz")
               for config in profiles.values())


def test_maintained_unbound_templates_regenerate_and_do_not_rewrite_formal_profiles():
    root = TOOLS.parent / "src/config_core/forward"
    formal = {p: p.read_bytes() for p in (root / "thermal_campaign").glob("*.json")}
    profiles = development.development_profiles()
    for arm, config in profiles.items():
        assert json.loads((root / "thermal_development" / f"{arm.lower()}_e100.json").read_text()) == config
        assert config["case"]["dataset"]["development_manifest_sha256"] is None
    assert formal and all(p.read_bytes() == before for p, before in formal.items())


def test_prepare_only_explicit_arm_and_preserve_existing_preparation(monkeypatch, tmp_path, capsys):
    from channelthermal.data import development_split

    monkeypatch.setattr(development_split, "load_development_manifest", lambda *args, **kwargs: manifest())
    monkeypatch.setattr(development, "registry_dataset_path", lambda profile: tmp_path / "data.h5")
    args = ["--arm", "H-local", "--manifest", str(tmp_path / "manifest.json"),
            "--dataset", str(tmp_path / "data.h5"), "--first-run-id", "3101", "--output-dir", str(tmp_path / "profiles")]
    assert development.main(args) == 0
    output = tmp_path / "profiles"
    assert sorted(p.name for p in output.iterdir()) == ["h-local_e100.json", "preparation_index.json"]
    saved = {p: p.read_bytes() for p in output.iterdir()}
    prepared = json.loads((output / "h-local_e100.json").read_text())
    assert "packed_h5_path" not in prepared["case"]["dataset"]
    index = json.loads((output / "preparation_index.json").read_text())
    assert index["profiles"]["H-local"]["response_case_count"] == 4
    assert index["profiles"]["H-local"]["response_case_ids"] == ["0001", "0304", "0318", "0320"]
    command = capsys.readouterr().out
    assert "--dry-run" in command and "--yes" not in command
    assert "H-local" in command and "B-native" not in command
    with pytest.raises(FileExistsError, match="preserve"):
        development.main(args)
    assert all(p.read_bytes() == before for p, before in saved.items())
    with pytest.raises(SystemExit):
        development.parse_args([*args, "--arm", "H-local", "H-local"])


def test_readonly_scope_defaults_bound_manifest_and_full_optout_keeps_no_filter(monkeypatch):
    from channelthermal.data import development_split

    seen = []
    selected = manifest()
    monkeypatch.setattr(development_split, "resolve_development_manifest",
                        lambda settings, path, **kwargs: seen.append(kwargs) or selected)
    assert development.resolve_evaluation_manifest({}, "/data/native.h5") is selected
    assert seen == [{"allow_embedded": True}]
    assert development.evaluation_dataset_kwargs(selected, "val") == {"case_ids": ("0601", "0602")}
    assert development.resolve_evaluation_manifest({}, "/data/native.h5", scope="formal-full") is None
    assert len(seen) == 1 and development.evaluation_dataset_kwargs(None, "test") == {}
    with pytest.raises(ValueError, match="cannot also"):
        development.resolve_evaluation_manifest({}, "/data/native.h5", scope="formal-full", manifest_path=Path("other.json"))
    monkeypatch.setattr(development_split, "resolve_development_manifest", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="requires"):
        development.resolve_evaluation_manifest({}, "/data/native.h5", scope="development")


def test_explicit_reader_manifest_cannot_replace_checkpoint_membership(monkeypatch):
    from channelthermal.data import development_split

    saved, other = manifest(), manifest()
    other["manifest_sha256"] = "b" * 64
    monkeypatch.setattr(development_split, "resolve_development_manifest", lambda *args, **kwargs: saved)
    monkeypatch.setattr(development_split, "load_development_manifest", lambda *args, **kwargs: other)
    with pytest.raises(ValueError, match="differs"):
        development.resolve_evaluation_manifest({}, "/data/native.h5", manifest_path=Path("other.json"))


def test_preparation_rejects_unignored_repository_sibling_output(monkeypatch, tmp_path):
    from channelthermal.data import development_split

    monkeypatch.setattr(development_split, "load_development_manifest", lambda *args, **kwargs: manifest())
    monkeypatch.setattr(development, "registry_dataset_path", lambda profile: tmp_path / "data.h5")
    forbidden = development.REPOSITORY_ROOT / "development_profiles"
    with pytest.raises(ValueError, match="outside tracked"):
        development.main(["--arm", "H-local", "--manifest", str(tmp_path / "manifest.json"),
            "--dataset", str(tmp_path / "data.h5"), "--first-run-id", "3101", "--output-dir", str(forbidden)])
    assert not forbidden.exists()


def test_shared_generated_output_guard_allows_external_and_only_established_ignored_roots(tmp_path):
    assert development.validate_generated_output(tmp_path / "outside") == tmp_path / "outside"
    for name in ("diagnostics", "Trained_Results"):
        path = development.PROJECT_ROOT / name / "future_evidence"
        assert development.validate_generated_output(path) == path.resolve()
    for path in (development.REPOSITORY_ROOT / "sibling.json", development.PROJECT_ROOT / "src/generated.json"):
        with pytest.raises(ValueError, match="outside tracked"):
            development.validate_generated_output(path)


def test_native_registry_placement_rejects_mismatch_before_write_and_real_plugin_accepts_profile(monkeypatch, tmp_path):
    from channelthermal.data import development_split
    from channelthermal.plugin import ThermalChannelPlugin

    from honf_runtime.config_loader import load_config_bundle

    selected = manifest()
    monkeypatch.setattr(development_split, "load_development_manifest", lambda *args, **kwargs: selected)
    profile = development.development_profiles(manifest=selected)["H-local"]
    native_path = development.registry_dataset_path(profile)
    output = tmp_path / "profiles"
    args = ["--arm", "H-local", "--manifest", str(tmp_path / "manifest.json"),
        "--dataset", str(tmp_path / "relocated.h5"), "--first-run-id", "3101", "--output-dir", str(output)]
    with pytest.raises(ValueError, match="native registry path"):
        development.main(args)
    assert not output.exists()
    args[args.index("--dataset") + 1] = str(native_path)
    assert development.main(args) == 0
    bundle = load_config_bundle(output / "h-local_e100.json")
    ThermalChannelPlugin().validate_config(bundle)
    resolved = ThermalChannelPlugin()._registry(bundle).resolve(profile["case"]["dataset_id"])
    assert resolved.path.resolve() == native_path
    assert "packed_h5_path" not in bundle.core["case"]["dataset"]
