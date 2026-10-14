"""The explicit pooling repair cannot change sealed scientific bindings."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from honf_runtime.unified_training import (
    _content_pooling_source_amendment,
    _resume_identity_amendment,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def repair(tmp_path):
    path = "src/honf_forward_core/content_hypergraph/organization.py"
    old_files = {path: "a" * 64, "src/unchanged.py": "b" * 64}
    new_files = {**old_files, path: "c" * 64}
    saved = {
        "workflow": "content_hypergraph_joint_v1", "model_family": "honf_content_hypergraph",
        "recipe": {"dataset": "wind", "architecture": "A-H6", "protocol": "full_train1000",
                   "approved_stop_after": 1000, "primary_queries": 4096},
        "provider_identity": {"membership_sha256": "locked", "normalizer": "TRAIN-only"},
        "optimizer_schedule_contract": ["unchanged"],
        "loaded_source": {"files": old_files, "sha256": digest(old_files)},
    }
    current = {**copy.deepcopy(saved), "loaded_source": {"files": new_files, "sha256": digest(new_files)}}
    evidence = {"schema": "content_hypergraph_stable_pooling_correctness_v1", "optimizer_updates": 0,
                **{key: True for key in ("fp64_reference_pass", "extreme_finite_backward_pass",
                    "checkpoint_forward_pass", "native_backward_probe_pass",
                    "old_finite_forward_nonfinite_backward_reproduced")}}
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    declaration = {
        "schema": "content_hypergraph_pooling_numerical_repair_v1", "source_epoch": 800,
        "approved_stop_after": 1000, "source_checkpoint_sha256": "d" * 64,
        "saved_loaded_source_sha256": digest(old_files), "current_loaded_source_sha256": digest(new_files),
        "changes": [{"path": path, "old_sha256": "a" * 64, "new_sha256": "c" * 64}],
        "evidence_path": str(evidence_path),
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }
    return saved, current, declaration


def apply(repair, **kwargs):
    saved, current, declaration = repair
    return _content_pooling_source_amendment(saved, current, declaration,
        source_sha256="d" * 64, source_epoch=800, stop_after=1000, **kwargs)


def test_explicit_repair_changes_only_loaded_source(repair):
    result = apply(repair)
    assert result["scientific_identity_unchanged"] is True
    assert result["kind"] == "explicit_content_pooling_numerical_repair"
    with pytest.raises(ValueError, match="Checkpoint identity"):
        _resume_identity_amendment(repair[0], repair[1])


@pytest.mark.parametrize("field", ["recipe", "provider_identity", "optimizer_schedule_contract"])
def test_repair_rejects_scientific_identity_changes(repair, field):
    repair[1][field] = {"changed": True}
    with pytest.raises(ValueError, match="numerical pooling continuation"):
        apply(repair)


@pytest.mark.parametrize("field,value", [
    ("source_epoch", 700), ("source_checkpoint_sha256", "e" * 64),
    ("approved_stop_after", 5000), ("saved_loaded_source_sha256", "f" * 64),
    ("current_loaded_source_sha256", "f" * 64), ("changes", []),
    ("evidence_path", "relative/evidence.json"), ("evidence_sha256", "f" * 64),
])
def test_repair_rejects_inconsistent_declaration(repair, field, value):
    repair[2][field] = value
    with pytest.raises(ValueError, match="numerical pooling continuation"):
        apply(repair)


def test_repair_rejects_unallowlisted_or_added_source(repair):
    files = repair[1]["loaded_source"]["files"]
    files["src/unchanged.py"] = "e" * 64
    repair[1]["loaded_source"]["sha256"] = digest(files)
    repair[2]["current_loaded_source_sha256"] = digest(files)
    repair[2]["changes"].append({"path": "src/unchanged.py", "old_sha256": "b" * 64, "new_sha256": "e" * 64})
    with pytest.raises(ValueError, match="numerical pooling continuation"):
        apply(repair)
    files["src/new.py"] = "f" * 64
    with pytest.raises(ValueError, match="numerical pooling continuation"):
        apply(repair)


@pytest.mark.parametrize("value", [False, 1])
def test_repair_requires_boolean_measurement_pass(repair, value):
    from pathlib import Path

    evidence_path = Path(repair[2]["evidence_path"])
    evidence = json.loads(evidence_path.read_text())
    evidence["native_backward_probe_pass"] = value
    evidence_path.write_text(json.dumps(evidence))
    repair[2]["evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="numerical pooling continuation"):
        apply(repair)


def test_repair_never_authorizes_5000(repair):
    with pytest.raises(ValueError, match="numerical pooling continuation"):
        _content_pooling_source_amendment(*repair, source_sha256="d" * 64,
                                         source_epoch=800, stop_after=5000)
