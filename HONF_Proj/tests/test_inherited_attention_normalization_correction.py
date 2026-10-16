from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tools"))

import inherited_attention_train

from honf_runtime import content_hypergraph as checkpoint_loader
from honf_runtime import inherited_attention_protocol
from honf_runtime.inherited_attention_child import (
    IAG_NORMALIZATION_CORRECTION_ALLOWED_PATHS,
    IAG_NORMALIZATION_CORRECTION_BINDING_SCHEMA,
    IAG_NORMALIZATION_CORRECTION_PATH,
    IAG_NORMALIZATION_CORRECTION_Q_PATHS,
    IAG_NORMALIZATION_CORRECTION_SCHEMA,
    IAG_NORMALIZATION_CORRECTION_SCOPE,
    IAG_NORMALIZATION_CORRECTION_TARGET_SHA256,
    IAG_NORMALIZATION_CORRECTION_THERMAL_PATHS,
    IAG_NORMALIZATION_CORRECTION_THERMAL_SCOPE,
    _json_sha256,
    _require_same_soft_identity,
    bind_inherited_attention_normalization_correction,
    validate_inherited_attention_normalization_correction_resume,
)
from honf_runtime.inherited_attention_protocol import IAG_PROTOCOL


def _files_for(architecture: str, *, external_delta: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    if architecture == "IAG-C":
        source = {
            "src/honf_forward_core/inherited_attention/__init__.py": "1" * 64,
            "src/honf_forward_core/inherited_attention/model.py": "2" * 64,
            IAG_NORMALIZATION_CORRECTION_PATH: "3" * 64,
            "src/honf_runtime/inherited_attention_child.py": "4" * 64,
            "src/honf_runtime/inherited_attention_protocol.py": "5" * 64,
            "tools/inherited_attention_train.py": "6" * 64,
        }
        target = {
            "src/honf_forward_core/inherited_attention/__init__.py": "7" * 64,
            "src/honf_forward_core/inherited_attention/model.py": "8" * 64,
            IAG_NORMALIZATION_CORRECTION_PATH: IAG_NORMALIZATION_CORRECTION_TARGET_SHA256,
            "src/honf_forward_core/inherited_attention/query_reader.py": "9" * 64,
            "src/honf_runtime/inherited_attention_child.py": "a" * 64,
            "src/honf_runtime/inherited_attention_protocol.py": "b" * 64,
            "tools/inherited_attention_train.py": "c" * 64,
        }
    else:
        source = {
            IAG_NORMALIZATION_CORRECTION_PATH: "1" * 64,
            "src/honf_runtime/inherited_attention_child.py": "2" * 64,
            "tools/inherited_attention_train.py": "3" * 64,
        }
        target = {
            IAG_NORMALIZATION_CORRECTION_PATH: IAG_NORMALIZATION_CORRECTION_TARGET_SHA256,
            "src/honf_runtime/inherited_attention_child.py": "4" * 64,
            "tools/inherited_attention_train.py": "5" * 64,
        }
    if external_delta:
        source["src/honf_runtime/unified_training.py"] = "7" * 64
        target["src/honf_runtime/unified_training.py"] = "8" * 64
    return source, target


def _loaded_source(files: dict[str, str]) -> dict[str, object]:
    normalized = dict(sorted(files.items()))
    return {"files": normalized, "sha256": _json_sha256(normalized)}


def _identity_pair(tmp_path: Path, architecture: str = "IAG-C", *, external_delta: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    representation = {"IAG-C": "iag_c", "IAG-Q": "iag_q"}[architecture]
    dataset = "thermal" if architecture == "IAG-C" else "wind"
    source_files, target_files = _files_for(architecture, external_delta=external_delta)
    source = {
        "workflow": "inherited_attention_joint_v1",
        "model_family": "inherited_attention_v1",
        "recipe": {
            "dataset": dataset,
            "architecture": architecture,
            "representation": representation,
            "protocol": IAG_PROTOCOL,
            "normalization": "soft",
            "trajectory": "fresh_soft",
        },
        "recipe_sha256": "a" * 64,
        "absolute_epoch_transition": None,
        "provider_identity": {
            "model_config": {
                "architecture": architecture,
                "representation": representation,
                "normalization": "soft",
                "scene_executor": "dense" if architecture == "IAG-C" else "cached_tokens",
                "query_executor": "indexed" if architecture == "IAG-C" else "cell_selected",
            },
            "training": {"normalization": "soft", "absolute_epoch_transition": None},
        },
        "optimizer_schedule_contract": [{"name": "all", "parameter_names": ["weight"]}],
        "loaded_source": _loaded_source(source_files),
    }
    target = copy.deepcopy(source)
    target["recipe"] = {
        **source["recipe"],
        "normalization": "measure_entmax15",
        "trajectory": "e500_sparse_child",
        "sparse_transition": {"start_epoch": 501, "end_epoch": 800},
    }
    target["recipe_sha256"] = "b" * 64
    target["absolute_epoch_transition"] = {"start_epoch": 501, "end_epoch": 800}
    target["provider_identity"]["model_config"]["normalization"] = "measure_entmax15"
    target["provider_identity"]["training"] = {
        "normalization": "measure_entmax15",
        "absolute_epoch_transition": {"start_epoch": 501, "end_epoch": 800},
    }
    target["loaded_source"] = _loaded_source(target_files)
    source_path = tmp_path / "parent_e500.pt"
    source_path.write_bytes(b"frozen soft e500 checkpoint")
    changes = {
        name: {"source_sha256": source_files.get(name), "target_sha256": target_files.get(name)}
        for name in source_files.keys() | target_files.keys()
        if source_files.get(name) != target_files.get(name)
    }
    receipt = {
        "schema": IAG_NORMALIZATION_CORRECTION_SCHEMA,
        "scope": IAG_NORMALIZATION_CORRECTION_SCOPE if architecture == "IAG-Q" else IAG_NORMALIZATION_CORRECTION_THERMAL_SCOPE,
        "dataset": dataset,
        "architecture": architecture,
        "representation": representation,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_epoch": 500,
        "source_recipe_sha256": source["recipe_sha256"],
        "target_recipe_sha256": target["recipe_sha256"],
        "source_loaded_source_sha256": source["loaded_source"]["sha256"],
        "target_loaded_source_sha256": target["loaded_source"]["sha256"],
        "source_files": source_files,
        "target_files": target_files,
        "file_changes": changes,
        "reason": "FP64 prefix-root evaluation repairs finite FP32 boundary misses.",
    }
    if architecture == "IAG-C":
        evidence_rows = []
        for case_key, module_count in zip(("0006", "0097", "0342", "0631"), (1, 4, 7, 12)):
            input_hash, output_hash, gradient_hash = "d" * 64, "e" * 64, "f" * 64
            evidence_rows.append({
                "case_key": case_key,
                "module_count": module_count,
                "source_input_sha256": input_hash,
                "target_input_sha256": input_hash,
                "source_output_sha256": output_hash,
                "target_output_sha256": output_hash,
                "source_named_gradient_sha256": gradient_hash,
                "target_named_gradient_sha256": gradient_hash,
            })
        evidence = {
            "schema": "round4_thermal_soft_runtime_equivalence_v1",
            "checkpoint_sha256": receipt["source_checkpoint_sha256"],
            "source_loaded_source_sha256": receipt["source_loaded_source_sha256"],
            "target_loaded_source_sha256": receipt["target_loaded_source_sha256"],
            "case_keys": ["0006", "0097", "0342", "0631"],
            "rows": evidence_rows,
            "passed": True,
        }
        evidence_path = tmp_path / "thermal_runtime_equivalence.json"
        evidence_bytes = json.dumps(evidence, sort_keys=True, indent=2).encode()
        evidence_path.write_bytes(evidence_bytes)
        receipt["thermal_runtime_equivalence"] = {
            "path": str(evidence_path.resolve()),
            "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        }
    receipt_path = tmp_path / "normalization_correction.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
    return source, target, source_path, receipt_path, receipt


@pytest.mark.parametrize("architecture", ["IAG-C", "IAG-Q"])
def test_correction_receipt_binds_exact_soft_parent_and_sparse_child_identity(tmp_path: Path, architecture: str):
    source, target, source_path, receipt_path, _ = _identity_pair(tmp_path, architecture)
    binding = bind_inherited_attention_normalization_correction(
        receipt_path, source, target, source_path=source_path,
    )
    assert binding["schema"] == IAG_NORMALIZATION_CORRECTION_BINDING_SCHEMA
    assert binding["receipt_sha256"] == _json_sha256(json.loads(binding["receipt_json"]))
    target["normalization_correction"] = binding
    transition = _require_same_soft_identity(source, target, source_path=source_path)
    assert transition["same_probability_function"] is True
    validate_inherited_attention_normalization_correction_resume(binding, target)
    resume_identity = copy.deepcopy(target)
    resume_identity.pop("normalization_correction")
    restored = inherited_attention_train._restore_normalization_correction_identity(
        {"normalization_correction": binding}, resume_identity,
    )
    assert restored == binding
    assert resume_identity["normalization_correction"] == binding


def test_no_receipt_keeps_loaded_source_identity_strict(tmp_path: Path):
    source, target, source_path, receipt_path, _ = _identity_pair(tmp_path)
    target["loaded_source"]["files"]["src/honf_runtime/unified_training.py"] = "9" * 64
    target["loaded_source"]["sha256"] = _json_sha256(target["loaded_source"]["files"])
    with pytest.raises(ValueError, match="changes a field beyond normalization"):
        _require_same_soft_identity(source, target)
    with pytest.raises(ValueError, match="file maps do not match"):
        bind_inherited_attention_normalization_correction(
            receipt_path, source, target, source_path=source_path,
        )


def test_receipt_rejects_file_delta_outside_fixed_allowlist(tmp_path: Path):
    source, target, source_path, receipt_path, _ = _identity_pair(tmp_path, external_delta=True)
    with pytest.raises(ValueError, match="exact fixed scope allowlist|exact old/new loaded-source maps"):
        bind_inherited_attention_normalization_correction(
            receipt_path, source, target, source_path=source_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scope", "entmax15_other_repair_v1", "scope or source epoch"),
        ("source_epoch", 499, "scope or source epoch"),
        ("source_checkpoint_sha256", "0" * 64, "exact e500 parent checkpoint"),
        ("target_recipe_sha256", "0" * 64, "exact parent and child recipe hashes"),
        ("representation", "iag_c_wrong", "different dataset or IAG representation"),
    ],
)
def test_receipt_rejects_wrong_scope_or_parent_target_binding(
    tmp_path: Path, field: str, value: object, message: str,
):
    source, target, source_path, receipt_path, receipt = _identity_pair(tmp_path)
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        bind_inherited_attention_normalization_correction(
            receipt_path, source, target, source_path=source_path,
        )


def test_correction_receipt_rejects_any_other_identity_change_and_resume_runtime_drift(tmp_path: Path):
    source, target, source_path, receipt_path, _ = _identity_pair(tmp_path)
    binding = bind_inherited_attention_normalization_correction(
        receipt_path, source, target, source_path=source_path,
    )
    target["normalization_correction"] = binding
    target["optimizer_schedule_contract"][0]["parameter_names"] = ["other"]
    with pytest.raises(ValueError, match="changes a field beyond normalization"):
        _require_same_soft_identity(source, target, source_path=source_path)

    source, target, source_path, receipt_path, _ = _identity_pair(tmp_path / "resume")
    binding = bind_inherited_attention_normalization_correction(
        receipt_path, source, target, source_path=source_path,
    )
    validate_inherited_attention_normalization_correction_resume(binding, target)
    drifted = copy.deepcopy(target)
    drifted["loaded_source"]["files"][IAG_NORMALIZATION_CORRECTION_PATH] = "0" * 64
    drifted["loaded_source"]["sha256"] = _json_sha256(drifted["loaded_source"]["files"])
    with pytest.raises(ValueError, match="differs from its bound target runtime"):
        validate_inherited_attention_normalization_correction_resume(binding, drifted)
    altered_binding = dict(binding)
    altered_binding["receipt_json"] += " "
    with pytest.raises(ValueError, match="fail their sealed hash"):
        validate_inherited_attention_normalization_correction_resume(altered_binding, target)


def test_thermal_runtime_equivalence_evidence_is_bound_and_required(tmp_path: Path):
    source, target, source_path, receipt_path, receipt = _identity_pair(tmp_path)
    evidence_binding = receipt["thermal_runtime_equivalence"]
    evidence_path = Path(evidence_binding["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["passed"] = False
    evidence_bytes = json.dumps(evidence, sort_keys=True, indent=2).encode()
    evidence_path.write_bytes(evidence_bytes)
    receipt["thermal_runtime_equivalence"]["sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match this parent and runtime transition"):
        bind_inherited_attention_normalization_correction(
            receipt_path, source, target, source_path=source_path,
        )

    evidence["passed"] = True
    evidence["rows"][0]["target_output_sha256"] = "0" * 64
    evidence_bytes = json.dumps(evidence, sort_keys=True, indent=2).encode()
    evidence_path.write_bytes(evidence_bytes)
    receipt["thermal_runtime_equivalence"]["sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="did not match bitwise"):
        bind_inherited_attention_normalization_correction(
            receipt_path, source, target, source_path=source_path,
        )


def test_correction_contract_has_distinct_fixed_q_and_thermal_path_sets():
    assert IAG_NORMALIZATION_CORRECTION_Q_PATHS == frozenset({
        "src/honf_forward_core/inherited_attention/normalization.py",
        "src/honf_runtime/inherited_attention_child.py",
        "tools/inherited_attention_train.py",
    })
    assert IAG_NORMALIZATION_CORRECTION_ALLOWED_PATHS == IAG_NORMALIZATION_CORRECTION_Q_PATHS | IAG_NORMALIZATION_CORRECTION_THERMAL_PATHS
    assert "src/honf_forward_core/inherited_attention/query_reader.py" in IAG_NORMALIZATION_CORRECTION_THERMAL_PATHS


def test_cli_resume_contract_matches_the_checkpoint_bound_receipt(tmp_path: Path):
    source, target, source_path, receipt_path, _ = _identity_pair(tmp_path)
    binding = bind_inherited_attention_normalization_correction(
        receipt_path, source, target, source_path=source_path,
    )
    target["normalization_correction"] = binding
    receipt = json.loads(binding["receipt_json"])
    expected = {
        "scope": receipt["scope"],
        "receipt_sha256": binding["receipt_sha256"],
        "target_loaded_source_sha256": receipt["target_loaded_source_sha256"],
    }
    payload = {
        "branch_parent": {
            "path": str(source_path.resolve()),
            "sha256": receipt["source_checkpoint_sha256"],
            "identity": source,
            "inherited_attention_sparse_child": {"normalization_correction": expected},
        },
    }
    inherited_attention_train._validate_normalization_correction_branch_binding(target, payload)
    payload["branch_parent"]["inherited_attention_sparse_child"]["normalization_correction"]["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from its normalization-correction identity"):
        inherited_attention_train._validate_normalization_correction_branch_binding(target, payload)


def test_loader_rejects_corrected_checkpoint_with_drifted_branch_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    source, target, source_path, receipt_path, receipt = _identity_pair(tmp_path, "IAG-C")
    source_model = {
        "normalization": "soft", "scene_executor": "dense", "query_executor": "indexed",
    }
    target_model = {
        **source_model, "normalization": "measure_entmax15", "scene_executor": "selected",
    }
    source["recipe"]["model"] = source_model
    target["recipe"]["model"] = target_model
    source["recipe"]["numerical_policy"] = {}
    target["recipe"]["numerical_policy"] = {}
    source["recipe_sha256"] = inherited_attention_train.canonical_sha256(source["recipe"])
    target["recipe_sha256"] = inherited_attention_train.canonical_sha256(target["recipe"])
    receipt["source_recipe_sha256"] = source["recipe_sha256"]
    receipt["target_recipe_sha256"] = target["recipe_sha256"]
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")

    source_core = {"schema": "inherited_attention_core_v1", "architecture": "IAG-C", **source_model}
    target_core = {"schema": "inherited_attention_core_v1", "architecture": "IAG-C", **target_model}
    source["provider_identity"]["model_config"] = source_core
    source["provider_identity"]["adapter_state"] = {"core": source_core}
    target["provider_identity"]["model_config"] = target_core
    target["provider_identity"]["adapter_state"] = {"core": target_core}
    binding = bind_inherited_attention_normalization_correction(
        receipt_path, source, target, source_path=source_path,
    )
    target["normalization_correction"] = binding

    monkeypatch.setattr(inherited_attention_protocol, "validate_inherited_attention_recipe", lambda value: value)
    monkeypatch.setattr(inherited_attention_protocol, "configure_inherited_attention_numerics", lambda value: None)
    monkeypatch.setattr(checkpoint_loader, "require_model_family", lambda *args: None)
    expected_summary = {
        "scope": receipt["scope"],
        "receipt_sha256": binding["receipt_sha256"],
        "target_loaded_source_sha256": receipt["target_loaded_source_sha256"],
    }
    expected_summary["receipt_sha256"] = "0" * 64
    transition = {"source": "dense", "target": "selected", "same_probability_function": True}
    payload = {
        "workflow": "inherited_attention_joint_v1",
        "arm": "IAG-C",
        "epoch": 501,
        "experiment_identity": target,
        "branch_parent": {
            "path": str(source_path.resolve()),
            "sha256": receipt["source_checkpoint_sha256"],
            "epoch": 500,
            "identity": source,
            "inherited_attention_sparse_child": {
                "schema": "inherited_attention_sparse_child_v1",
                "trajectory": "soft_e1_500_then_absolute_sparse_transition_e501_800",
                "source_epoch": 500,
                "target_recipe_sha256": target["recipe_sha256"],
                "scene_executor_transition": transition,
                "normalization_correction": expected_summary,
            },
        },
        "model_state_dict": {},
    }
    with pytest.raises(ValueError, match="branch correction summary differs"):
        checkpoint_loader._load_inherited_attention_checkpoint(payload, device="cpu")

    payload["branch_parent"]["inherited_attention_sparse_child"]["normalization_correction"] = {
        "scope": receipt["scope"],
        "receipt_sha256": binding["receipt_sha256"],
        "target_loaded_source_sha256": receipt["target_loaded_source_sha256"],
    }
    payload["branch_parent"]["inherited_attention_sparse_child"]["scene_executor_transition"] = {
        "source": "selected", "target": "selected", "same_probability_function": True,
    }
    with pytest.raises(ValueError, match="branch transition differs"):
        checkpoint_loader._load_inherited_attention_checkpoint(payload, device="cpu")


def test_cli_rejects_correction_receipt_outside_fresh_sparse_attachment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    builds: list[bool] = []
    monkeypatch.setattr(inherited_attention_train, "build", lambda *args, **kwargs: builds.append(True))
    with pytest.raises(ValueError, match="reserved for a fresh sparse e500 child attachment"):
        inherited_attention_train.main([
            "--recipe", str(PROJECT / "src/config_core/forward/inherited_attention/thermal_iag_c_soft_development_v1.json"),
            "--dry-run",
            "--normalization-correction", str(tmp_path / "receipt.json"),
        ])
    assert builds == []
