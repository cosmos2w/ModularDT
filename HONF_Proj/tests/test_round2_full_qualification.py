"""End-to-end fixtures for the measured Round2 full-TRAIN gate.

All checkpoint, evidence, source-copy, and recipe artifacts live under pytest's
temporary directory. The fixture builds a real but small v2 Thermal model and
checkpoint; it never opens dataset, validation, or solver data.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (
    PROJECT / "src",
    PROJECT / "Case_ThermalChannel/src",
    PROJECT / "Case_WindFarm/src",
    PROJECT / "tools",
):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from channelthermal.content_hypergraph import (  # noqa: E402
    THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2,
    ThermalContentHypergraphAdapterV2,
)
from honf_runtime.content_hypergraph_qualification import (  # noqa: E402
    FIRST_DIRECT,
    SCHEMA,
    canonical_sha256,
    recipe_architecture,
    validate_round2_qualification,
)


_LAUNCHER_PATH = PROJECT / "tools/content_hypergraph_train.py"
_LAUNCHER_SPEC = importlib.util.spec_from_file_location("round2_qualification_launcher", _LAUNCHER_PATH)
assert _LAUNCHER_SPEC is not None and _LAUNCHER_SPEC.loader is not None
launcher = importlib.util.module_from_spec(_LAUNCHER_SPEC)
_LAUNCHER_SPEC.loader.exec_module(launcher)

_RECIPE_DIR = PROJECT / "src/config_core/forward/content_hypergraph"
_CORRECTNESS_CHECKS = {
    "attention_reference",
    "masked_empty_rows",
    "staged_exchange",
    "measure_refinement",
    "source_coefficients_W_coupling",
    "affine_and_zero_offset",
    "native_observations",
    "heat_adjoint",
    "fresh_head_gradients",
    "resume_identity",
}
_THERMAL_ROLES = (
    "u",
    "v",
    "p",
    "omega",
    "temperature_fluid",
    "temperature_surface",
    "temperature_material",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=allow_nan) + "\n")


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _digest(path)}


def _artifact(path: Path, measurements: dict[str, Any], *, config_sha: str,
              checkpoint_sha: str, law_sha: str) -> dict[str, Any]:
    _write_json(path, {
        "status": "completed",
        "architecture_config_sha256": config_sha,
        "development_checkpoint_sha256": checkpoint_sha,
        "physical_output_law_sha256": law_sha,
        "qualified": True,
        "measurements": measurements,
    })
    return _descriptor(path)


def _native_record(path: Path, *, architecture: str, checkpoint_sha: str,
                   epoch: int, field_score: float, temperature_fluid: float) -> dict[str, str]:
    rows = []
    for index in range(22):
        roles = {name: {"rmse": 1.0} for name in _THERMAL_ROLES}
        roles["temperature_fluid"] = {"rmse": temperature_fluid}
        rows.append({
            "case_id": f"dev-{index:02d}",
            "module_count": 1 + (index % 12),
            "roles": roles,
            "q_proxy_rmse": 1.0,
        })
    _write_json(path, {
        "status": "completed",
        "panel": "native",
        "split": "validation",
        "checkpoint_sha256": checkpoint_sha,
        "completed_epoch": epoch,
        "architecture": architecture,
        "endpoint_panel_sha256": hashlib.sha256(b"fixed25-dev22-native-panel").hexdigest(),
        "metrics": {"field_score": field_score, "per_case": rows},
    })
    return _descriptor(path)


def _inspection_case(case_key: str, active_modules: int, *, edge_delta: float,
                     local_delta: float) -> dict[str, Any]:
    # These dimensions follow the real Thermal inspector's B,M,E,D and physical
    # field/query axes. Values are deliberately tiny summaries, not data arrays.
    return {
        "case_key": case_key,
        "scene_shape": {
            "batch_size": 1,
            "module_slots": 12,
            "active_module_count": active_modules,
            "environment_cells": 24 * 8,
            "coordinate_dim": 2,
            "native_grid_shape_yx": [64, 128],
        },
        "query_shape": {"batch_size": 1, "native_union_rows": 4096 + 16 * active_modules, "coordinate_dim": 2},
        "physical_output_shape": {"batch_size": 1, "native_union_rows": 4096 + 16 * active_modules, "channels": 5},
        "edge_selection": {"rule": "input-declared fixed edge0", "edge": 0},
        "conditions": {
            "input_declared_edge_removed": {
                "edge_access_frozen_without_renormalization": True,
                "output_delta_l2": edge_delta,
            },
            "local_branch_removed": {
                "edge_access_preserved": True,
                "output_delta_l2": local_delta,
            },
        },
    }


def _fixture(root: Path) -> dict[str, Any]:
    """Write a fully hash-bound positive qualification fixture under root."""
    root.mkdir(parents=True, exist_ok=True)
    development = json.loads((_RECIPE_DIR / "thermal_a2-h6_development.json").read_text())
    full = json.loads((_RECIPE_DIR / "thermal_a2-h6_full1000.json").read_text())
    assert recipe_architecture(development) == recipe_architecture(full)

    model = ThermalContentHypergraphAdapterV2(
        {
            "field_mean_by_channel": [0.0] * 5,
            "field_std_by_channel": [1.0] * 5,
            "interface_targets_std": [1.0, 1.0],
            "internal_temperature_std": [1.0],
        },
        core_config={
            "schema": "content_hypergraph_core_v2",
            "architecture": "A2-H6",
            **development["model"],
        },
    )
    model_config = model.get_config()
    config_sha = canonical_sha256(model_config)
    law_sha = canonical_sha256(THERMAL_CONTENT_NATIVE_LAW_CONTRACT_V2)

    # Hash source copies so the mutation test can invalidate evidence without
    # editing any repository source file.
    source_copies = []
    loaded_sources = {}
    source_names = (
        "Case_ThermalChannel/src/channelthermal/content_hypergraph.py",
        "Case_ThermalChannel/src/channelthermal/training/content_hypergraph_task_v2.py",
        "src/honf_runtime/content_hypergraph_qualification.py",
    )
    for index, name in enumerate(source_names):
        actual = PROJECT / name
        loaded_sources[name] = _digest(actual)
        copied = root / "source_evidence" / f"source_{index}.py"
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(actual.read_bytes())
        source_copies.append(_descriptor(copied))

    identity = {
        "workflow": "content_hypergraph_joint_v2",
        "recipe": development,
        "provider_identity": {"model_config": model_config},
        "loaded_source": {"files": loaded_sources},
    }
    identity_path = root / "development_identity.json"
    _write_json(identity_path, identity)

    checkpoint_path = root / "development_e2500_A2-H6.pt"
    torch.save({
        "epoch": 2500,
        "arm": "A2-H6",
        "experiment_identity": identity,
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)
    checkpoint = _descriptor(checkpoint_path)
    checkpoint.update({"epoch": 2500, "arm": "A2-H6"})

    organized_checkpoint_sha = checkpoint["sha256"]
    direct_checkpoint_sha = hashlib.sha256(b"matched A2-D checkpoint fixture").hexdigest()
    native = {
        "organized": _native_record(
            root / "native_organized.json", architecture="A2-H6",
            checkpoint_sha=organized_checkpoint_sha, epoch=2500,
            field_score=0.08, temperature_fluid=0.80,
        ),
        "matched_direct": _native_record(
            root / "native_matched_direct.json", architecture="A2-D",
            checkpoint_sha=direct_checkpoint_sha, epoch=2500,
            field_score=0.09, temperature_fluid=0.80,
        ),
        "first_direct": _native_record(
            root / "native_first_direct.json", architecture="A-D",
            checkpoint_sha=FIRST_DIRECT["thermal"], epoch=2500,
            field_score=0.10, temperature_fluid=1.0,
        ),
    }

    inspection_path = root / "content_inspection.json"
    _write_json(inspection_path, {
        "status": "completed",
        "architecture": "A2-H6",
        "completed_epoch": 2500,
        "checkpoint_sha256": organized_checkpoint_sha,
        "cases": [
            _inspection_case("low-M-train-scene", 1, edge_delta=0.125, local_delta=0.075),
            _inspection_case("high-M-train-scene", 12, edge_delta=0.250, local_delta=0.110),
        ],
    })

    measurements = {
        "correctness": {
            "checks": {name: True for name in sorted(_CORRECTNESS_CHECKS)},
            "failed_checks": 0,
            "cpu_test_pass_count": 1,
            "source_evidence": source_copies,
        },
        "content": {
            "inspection": _descriptor(inspection_path),
            "source_evidence": source_copies,
        },
        "cost": {
            "current_same_device_same_output_high_m_ratio": 0.5,
            "baseline_complete_high_m_median_seconds": 2.0,
            "candidate_complete_high_m_median_seconds": 1.0,
            "baseline_checkpoint_sha256": "0d4eb5eeb120c8be630dc439e7fc39e2c625e8fad3a28b53a644836ae032e117",
            "same_device_uuid": "GPU-fixture-uuid",
            "same_output_request_sha256": hashlib.sha256(b"same complete native output request").hexdigest(),
            "fixed_batch_backward": [
                {"architecture": architecture, "batch_size": 8, "requested_q_per_scene": q,
                 "status": "completed"}
                for architecture in ("A2-D", "A2-H6")
                for q in (1024, 4096, 8192)
            ],
            "source_evidence": source_copies,
        },
        "budget": {
            "forecast_gpu_associated_seconds": 100.0,
            "remaining_gpu_associated_seconds": 1000.0,
            "forecast_elapsed_seconds": 1000.0,
            "closeout_reserve_seconds": 7200.0,
            "remaining_elapsed_seconds": 10000.0,
            "source_evidence": source_copies,
        },
    }
    evidence = {
        **native,
        "correctness": _artifact(root / "correctness_evidence.json", measurements["correctness"],
                                  config_sha=config_sha, checkpoint_sha=organized_checkpoint_sha,
                                  law_sha=law_sha),
        "content": _artifact(root / "content_evidence.json", measurements["content"],
                              config_sha=config_sha, checkpoint_sha=organized_checkpoint_sha,
                              law_sha=law_sha),
        "cost": _artifact(root / "cost_evidence.json", measurements["cost"],
                           config_sha=config_sha, checkpoint_sha=organized_checkpoint_sha,
                           law_sha=law_sha),
        "budget": _artifact(root / "budget_evidence.json", measurements["budget"],
                             config_sha=config_sha, checkpoint_sha=organized_checkpoint_sha,
                             law_sha=law_sha),
    }

    floors = {name: 1.0e-8 for name in (
        "field_score", "u", "v", "p", "omega", "temperature_fluid",
        "temperature_surface", "temperature_material", "q_proxy",
    )}
    value = {
        "schema": SCHEMA,
        "dataset": "thermal",
        "full_recipe_sha256": canonical_sha256(full),
        "development_recipe": development,
        "approved_stop_after": 1000,
        "prospective_schedule_epochs": 5000,
        "architecture_config_sha256": config_sha,
        "physical_output_law_sha256": law_sha,
        "evidence": evidence,
        "inherited_train_floors": floors,
        "partition_binding": {
            "dataset": "thermal",
            "protocol": "full_train1000",
            "training_membership_sha256": hashlib.sha256(b"full thermal train membership fixture").hexdigest(),
            "validation_membership_sha256": hashlib.sha256(b"canonical thermal validation membership fixture").hexdigest(),
            "base_manifest_sha256": hashlib.sha256(b"thermal base manifest fixture").hexdigest(),
            "excluded_validation_id": "0273",
        },
        "development_identity_evidence": _descriptor(identity_path),
        "development_checkpoint_evidence": checkpoint,
    }
    qualification_path = root / "qualification.json"
    _write_json(qualification_path, value)
    return {
        "root": root,
        "full_recipe": full,
        "development_recipe": development,
        "qualification_path": qualification_path,
        "qualification": value,
        "inspection_path": inspection_path,
        "native_paths": {
            "organized": root / "native_organized.json",
            "matched_direct": root / "native_matched_direct.json",
            "first_direct": root / "native_first_direct.json",
        },
        "source_path": Path(source_copies[0]["path"]),
        "source_descriptors": source_copies,
        "full_recipe_path": root / "full_recipe.json",
    }


def _save_qualification(fixture: dict[str, Any]) -> None:
    _write_json(fixture["qualification_path"], fixture["qualification"])


def _refresh_evidence_descriptor(fixture: dict[str, Any], name: str) -> None:
    fixture["qualification"]["evidence"][name] = _descriptor(
        Path(fixture["qualification"]["evidence"][name]["path"])
    )
    _save_qualification(fixture)


def _run_launcher_before_build(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                               *, stop_after: int = 1000) -> None:
    recipe_path = fixture["root"] / "full_recipe.json"
    _write_json(recipe_path, fixture["full_recipe"])
    monkeypatch.setattr(launcher, "build", lambda *a, **k: pytest.fail("qualification failure reached model/data build"))
    launcher.main([
        "--recipe", str(recipe_path),
        "--qualification", str(fixture["qualification_path"]),
        "--stop-after", str(stop_after),
        "--device", "cpu",
        "--yes",
        "--output", str(fixture["root"] / "must-not-exist-run"),
    ])


def test_positive_full1000_qualification_is_hash_bound_and_complete(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "positive")
    result = validate_round2_qualification(fixture["qualification_path"], fixture["full_recipe"])

    assert result["schema"] == SCHEMA
    assert result["verified_field_decision"]["qualified"] is True
    assert set(result["evidence"]) >= {"organized", "matched_direct", "first_direct", "correctness", "content", "cost", "budget"}
    assert result["partition_binding"]["training_membership_sha256"]
    assert result["partition_binding"]["validation_membership_sha256"]
    assert result["full_recipe_sha256"] == canonical_sha256(fixture["full_recipe"])
    assert recipe_architecture(result["development_recipe"]) == recipe_architecture(fixture["full_recipe"])

    correctness_path = Path(result["evidence"]["correctness"]["path"])
    correctness = json.loads(correctness_path.read_text())["measurements"]["checks"]
    assert set(correctness) == _CORRECTNESS_CHECKS
    assert all(correctness.values())

    inspection = json.loads(fixture["inspection_path"].read_text())
    assert len(inspection["cases"]) == 2
    assert {case["scene_shape"]["active_module_count"] for case in inspection["cases"]} == {1, 12}
    for case in inspection["cases"]:
        assert case["scene_shape"]["module_slots"] == 12
        assert case["scene_shape"]["environment_cells"] == 192
        assert case["query_shape"]["coordinate_dim"] == 2
        assert case["physical_output_shape"]["channels"] == 5

    cost_path = Path(result["evidence"]["cost"]["path"])
    cost = json.loads(cost_path.read_text())["measurements"]
    assert {(item["architecture"], item["requested_q_per_scene"]) for item in cost["fixed_batch_backward"]} == {
        (architecture, q) for architecture in ("A2-D", "A2-H6") for q in (1024, 4096, 8192)
    }


def test_qualification_rejects_changed_hashed_source_before_build(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path / "changed-source")
    with fixture["source_path"].open("ab") as stream:
        stream.write(b"\n# changed after measurement\n")
    with pytest.raises(ValueError, match="source file changed"):
        _run_launcher_before_build(fixture, monkeypatch)
    assert not (fixture["root"] / "must-not-exist-run").exists()


def test_qualification_rejects_missing_content_use_before_build(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path / "missing-content")
    inspection = json.loads(fixture["inspection_path"].read_text())
    inspection["cases"][0]["conditions"]["input_declared_edge_removed"]["output_delta_l2"] = 0.0
    _write_json(fixture["inspection_path"], inspection)
    content_path = Path(fixture["qualification"]["evidence"]["content"]["path"])
    content = json.loads(content_path.read_text())
    content["measurements"]["inspection"] = _descriptor(fixture["inspection_path"])
    _write_json(content_path, content)
    _refresh_evidence_descriptor(fixture, "content")

    with pytest.raises(ValueError, match="edge-content/local output use"):
        _run_launcher_before_build(fixture, monkeypatch)
    assert not (fixture["root"] / "must-not-exist-run").exists()


def test_qualification_rejects_legacy_four_boolean_claims_before_build(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path / "legacy-booleans")
    fixture["qualification"] = {
        "schema": "content_hypergraph_full1000_qualification_v1",
        "dataset": "thermal",
        "correctness_pass": True,
        "edge_content_used": True,
        "nontrivial_train_dev_fields": True,
        "budget_forecast_fits": True,
    }
    _save_qualification(fixture)
    with pytest.raises(ValueError, match="Missing measured Round2 full-TRAIN qualification"):
        _run_launcher_before_build(fixture, monkeypatch)
    assert not (fixture["root"] / "must-not-exist-run").exists()


def test_qualification_rejects_nonfinite_budget_before_build(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path / "nonfinite-budget")
    budget_path = Path(fixture["qualification"]["evidence"]["budget"]["path"])
    budget = json.loads(budget_path.read_text())
    budget["measurements"]["forecast_gpu_associated_seconds"] = float("nan")
    _write_json(budget_path, budget, allow_nan=True)
    _refresh_evidence_descriptor(fixture, "budget")

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _run_launcher_before_build(fixture, monkeypatch)
    assert not (fixture["root"] / "must-not-exist-run").exists()


def test_qualification_rejects_wrong_native_epoch_and_wrong_stop_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_epoch = _fixture(tmp_path / "wrong-epoch")
    organized_path = wrong_epoch["native_paths"]["organized"]
    organized = json.loads(organized_path.read_text())
    organized["completed_epoch"] = 2499
    _write_json(organized_path, organized)
    wrong_epoch["qualification"]["evidence"]["organized"] = _descriptor(organized_path)
    _save_qualification(wrong_epoch)
    with pytest.raises(ValueError, match="literal2500"):
        _run_launcher_before_build(wrong_epoch, monkeypatch)
    assert not (wrong_epoch["root"] / "must-not-exist-run").exists()

    wrong_stop = _fixture(tmp_path / "wrong-stop")
    with pytest.raises(ValueError, match="Execution stop must lie in 1..1000"):
        _run_launcher_before_build(wrong_stop, monkeypatch, stop_after=1001)
    assert not (wrong_stop["root"] / "must-not-exist-run").exists()


@pytest.mark.parametrize("kind", ["full_recipe_hash", "unapproved_development_change"])
def test_qualification_binds_exact_recipe_and_only_permitted_full_differences(
    tmp_path: Path, kind: str,
) -> None:
    fixture = _fixture(tmp_path / kind)
    if kind == "full_recipe_hash":
        changed_full = copy.deepcopy(fixture["full_recipe"])
        changed_full["microbatch_size"] = 12
        with pytest.raises(ValueError, match="Full recipe differs"):
            validate_round2_qualification(fixture["qualification_path"], changed_full)
    else:
        fixture["qualification"]["development_recipe"]["model"]["context_dim"] = 64
        _save_qualification(fixture)
        with pytest.raises(ValueError, match="cannot silently change"):
            validate_round2_qualification(fixture["qualification_path"], fixture["full_recipe"])
