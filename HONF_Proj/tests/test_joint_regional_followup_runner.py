from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
for source in (ROOT / "src", ROOT / "Case_ThermalChannel/src", ROOT / "Case_WindFarm/src"):
    sys.path.insert(0, str(source))
import joint_regional_train as runner


def thermal_recipe() -> dict:
    objective = copy.deepcopy(runner._recovery_objective_contract("thermal"))
    objective["primary_cohort"] = "all original600 TRAIN cases; canonical89 from original90 exposed validation"
    objective["calibration_scope"] = "fresh full600 TRAIN P initialization; no DEV target reads/materialization; zero optimizer steps"
    objective["response_addendum"]["primary_cohort_overlap"] = ["0001", "0318", "0333", "0348"]
    return {
        "schema_version": 1,
        "task": "thermal",
        "mode": "P",
        "run_id": "T4201",
        "seed": 0,
        "hidden": 128,
        "message": 128,
        "regional_anchors": 0,
        "depth": 2,
        "receiver_tile": 512,
        "primary_queries": 1024,
        "microbatch_cases": 8,
        "effective_cases": 48,
        "formal_full": False,
        "total_epochs": 5000,
        "initialization": "fresh_all_trainable",
        "launch_policy": "root_protocol_go",
        "dataset_protocol": "thermal_original600_train_canonical89_validation_followup_v1",
        "locality_prior_strength": 0.0,
        "collective_width": 64,
        "max_sources": 12,
        "environment_token_shape": [24, 8],
        "model_contract": runner._recovery_model_contract("thermal", "P", runner.NATIVE_CURL_THERMAL_LAW),
        "objective_contract": objective,
        "engine_schedule": runner.RECOVERY_ENGINE_SCHEDULE,
        "optimizer_name": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1.0e-8,
        "gradient_clip_norm": 1.0,
        "optimizer_schedule": {
            "peak_lr": 3.0e-4,
            "warmup_start_lr": 3.0e-5,
            "warmup_epochs": 20,
            "hold_through_epoch": 1000,
            "final_lr": 3.0e-6,
        },
        "weight_decay": 1.0e-4,
        "checkpoint_epochs": list(range(100, 1001, 100)),
        "flow_readout_law": runner.NATIVE_CURL_THERMAL_LAW,
        "validation_scope": "canonical89",
        "native_sampling_protocol": "baseline_formal_v1",
        "execution_protocol": runner.FULL_TRAIN_FOLLOWUP_PROTOCOL,
        "full_train_followup1000": True,
        "approved_stop_after": 1000,
        "root_protocol_go": True,
        "manual_full_followup": True,
        "parent_checkpoint": None,
        "auxiliary_calibration_fallback": False,
        "population_binding": {
            "training_count": 600,
            "training_membership_sha256": "c991acbcdced62e887385bc2763d4da7f4686256f72e44f797b69185522f3e3e",
            "validation_count": 89,
            "validation_membership_sha256": "51f0bea2278c26b24a994a29a52ad6b8af1e3e85941dc75d9b20bbebce5d8728",
        },
        "mechanism_reference": {"recipe_file_sha256": "a" * 64},
        "calibration_receipt": {
            "receipt_path": "calibration.json",
            "required_payload_sha256": "sealed_after_fresh_P_native_curl_calibration",
            "dataset_protocol": "thermal_original600_train_canonical89_validation_followup_v1",
        },
    }


def test_full_read_recipe_binds_calibration_dataset_protocol_and_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "honf"
    run_root.mkdir()
    monkeypatch.setattr(runner, "ROOT", run_root)
    source_identity = {"source_git_commit": "clean-source-fixture", "source_set_sha256": "b" * 64}
    calls: list[dict] = []

    def clean_source_identity(**kwargs):
        calls.append(kwargs)
        return source_identity

    monkeypatch.setattr(runner, "training_source_identity", clean_source_identity)
    receipt = {
        "response_coefficient": 1.0,
        "operator_coefficient": 1.0,
        "flow_readout_law": runner.NATIVE_CURL_THERMAL_LAW,
        "training_source_identity": source_identity,
    }
    receipt["payload_sha256"] = runner._canonical_payload_sha256(receipt)
    (run_root / "calibration.json").write_text(json.dumps(receipt))
    candidate = thermal_recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(candidate))
    original_reader = runner.read_recipe
    reader_paths: list[str] = []

    def observe_reader(path):
        reader_paths.append(str(path))
        return original_reader(path)

    monkeypatch.setattr(runner, "read_recipe", observe_reader)
    resolved = runner.resolved_recipe(argparse.Namespace(recipe_json=str(recipe_path), command="start"))
    assert reader_paths == [str(recipe_path)]
    assert calls == [{
        "require_clean": True,
        "flow_readout_law": runner.NATIVE_CURL_THERMAL_LAW,
        "full_train_followup": True,
    }]
    assert resolved["calibration_receipt"] == {
        "receipt_path": "calibration.json",
        "required_payload_sha256": receipt["payload_sha256"],
        "file_sha256": runner.hashlib.sha256((run_root / "calibration.json").read_bytes()).hexdigest(),
        "dataset_protocol": candidate["dataset_protocol"],
    }
    assert resolved["auxiliary_calibration"] == receipt


def test_legacy_receipt_binding_keeps_its_original_descriptor_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "legacy-honf"
    run_root.mkdir()
    monkeypatch.setattr(runner, "ROOT", run_root)
    source_identity = {"source_git_commit": "legacy-source-fixture"}
    monkeypatch.setattr(runner, "training_source_identity", lambda **_kwargs: source_identity)
    monkeypatch.setattr(runner, "validate_recipe", lambda recipe: recipe)
    receipt = {"response_coefficient": 0.2, "operator_coefficient": 0.3,
               "training_source_identity": source_identity}
    receipt["payload_sha256"] = runner._canonical_payload_sha256(receipt)
    (run_root / "legacy.json").write_text(json.dumps(receipt))
    bound = runner._bind_calibration_receipt({
        "calibration_receipt": {
            "receipt_path": "legacy.json",
            "required_payload_sha256": "sealed_after_fresh_P_calibration",
        },
    })
    assert set(bound["calibration_receipt"]) == {
        "receipt_path", "required_payload_sha256", "file_sha256",
    }


def test_full_followup_build_marks_source_identity_as_full_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from channelthermal.training import joint_task as thermal_provider

    model = torch.nn.Linear(2, 2)
    model.seed = 0
    monkeypatch.setattr(thermal_provider, "build_thermal_joint_task",
                        lambda **_kwargs: (model, SimpleNamespace()))
    source_calls: list[dict] = []

    def fake_source_identity(**kwargs):
        source_calls.append(kwargs)
        return {"source_git_commit": "clean-full-source-fixture"}

    monkeypatch.setattr(runner, "training_source_identity", fake_source_identity)
    recipe = thermal_recipe()
    recipe["response_coefficient"] = 1.0
    recipe["operator_coefficient"] = 1.0
    recipe["auxiliary_calibration"] = {"response_coefficient": 1.0, "operator_coefficient": 1.0}
    _model, _provider, _engine, identity = runner.build(
        recipe, device="cpu", calibration_only=True, require_clean_source=True,
    )
    assert source_calls == [{
        "require_clean": True,
        "flow_readout_law": runner.NATIVE_CURL_THERMAL_LAW,
        "full_train_followup": True,
    }]
    assert identity["training_source_identity"]["source_git_commit"] == "clean-full-source-fixture"


def test_calibration_main_records_full_followup_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    run_root = tmp_path / "honf"
    output_dir = tmp_path / "run-output"
    run_root.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(runner, "ROOT", run_root)
    candidate = thermal_recipe()
    source_identity = {"source_git_commit": "clean-full-calibration-fixture", "source_set_sha256": "c" * 64}
    build_calls: list[dict] = []

    class Provider:
        def calibrate_auxiliary_coefficients(self):
            return {
                "response_coefficient": 1.0,
                "operator_coefficient": 1.0,
                "primary_partition": "original600 TRAIN only",
                "response_family_partition": "original TRAIN response addendum; all four fixed families",
                "calibration_case_ids": ["0006", "0002", "0348", "0231"],
                "validation_values_read": False,
                "validation_cases_materialized": 0,
                "development_response_families_loaded": False,
                "optimizer_steps": 0,
            }

    def fake_build(recipe, *, device, calibration_only=False, require_clean_source=False):
        build_calls.append({
            "device": device,
            "calibration_only": calibration_only,
            "require_clean_source": require_clean_source,
            "full_train_followup1000": recipe.get("full_train_followup1000"),
        })
        return None, Provider(), None, {"training_source_identity": source_identity}

    monkeypatch.setattr(runner, "resolved_recipe", lambda _args: candidate)
    monkeypatch.setattr(runner, "build", fake_build)
    monkeypatch.setattr(runner, "_check_fresh_calibration_destinations",
                        lambda _recipe, _output: (run_root / "calibration.json", output_dir / "calibration_invocation.json"))
    monkeypatch.setattr(runner, "repository_head_observed", lambda: "observed-head")
    assert runner.main([
        "calibrate-thermal", "--recipe-json", "unused.json", "--output-dir", str(output_dir),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    receipt = json.loads((run_root / "calibration.json").read_text())
    assert build_calls == [{
        "device": "cpu",
        "calibration_only": True,
        "require_clean_source": True,
        "full_train_followup1000": True,
    }]
    assert payload["optimizer_steps"] == 0
    assert receipt["training_source_identity"] == source_identity
    assert receipt["optimizer_steps"] == 0
    assert json.loads((output_dir / "calibration_invocation.json").read_text())["training_source_identity"] == source_identity


def test_full_followup_dry_run_is_inert_and_never_calls_preflight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    class Engine:
        def preflight_one_update(self, *_args, **_kwargs):
            raise AssertionError("full follow-up dry-run must not perform an optimizer update")

    monkeypatch.setattr(runner, "resolved_recipe", lambda _args: thermal_recipe())
    monkeypatch.setattr(runner, "build", lambda *_args, **_kwargs: (None, None, Engine(), {}))
    assert runner.main(["dry-run", "--recipe-json", "unused.json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_train_followup_inert_validation"
    assert payload["optimizer_updates"] == 0
    assert payload["preflight_update_executed"] is False
    assert payload["checkpoint_written"] is False
