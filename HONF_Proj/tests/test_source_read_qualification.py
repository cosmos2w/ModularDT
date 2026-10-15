"""The conditional full-data permission requires completed, bound evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from honf_runtime.source_read_qualification import validate_source_read_qualification


def _write(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _evidence(tmp_path, variant="D-scale"):
    project = Path(__file__).resolve().parents[1]
    recipe = json.loads((project / "src/config_core/forward/content_hypergraph/thermal_a2-d_source_read_full1000_d_scale_v1.json").read_text())
    recipe["source_reader_variant"] = variant
    if variant != "D-scale":
        recipe["dense_host_variant"] = "D-scale"
        recipe["model"].update(source_read_policy="utility", source_read_executor="packed")
    primary = {"temperature_fluid": .94, "v": 1., "p": 1.}
    other = {name: 1. for name in ("u", "omega", "temperature_surface", "temperature_material", "q_proxy", "module_peak_endpoint")}
    tails = {name: 1. for name in (*primary, "temperature_material", "module_peak_endpoint")}
    metrics = {"other_primary_means": primary, "primary_means": primary, "other_means": other,
               "primary_p90": tails, "native_tail_review_consistent": True,
               "not_materially_worse_than_equal_work_geometry": True}
    cost = {"complete_high_m_time_ratio": .88, "training_epoch_ratio": 1.01,
            "actual_fine_retained_fraction": .65, "b8_q8192_memory_ratio": .85}
    artifacts = {name: _write(tmp_path / (name + ".pt"), {"test_checkpoint": name})
                 for name in ("candidate_checkpoint", "dense_host_checkpoint")}
    native = {"candidate_epoch": 2500, "dense_host_epoch": 2500, "qualification_metrics": metrics,
              **{name + "_sha256": item["sha256"] for name, item in artifacts.items()}}
    checks = {name: True for name in ("native_laws", "same_function_executor", "plan_ownership",
                                     "query_consistency", "gradients", "real_optimizer", "checkpoint_resume", "full_stop")}
    for name, value in {"native_accuracy": native, "cost": {"qualification_metrics": cost},
                        "correctness": {"checks": checks}, "train_calibration": {"TRAIN_only": True},
                        "budget_forecast": {"fits_remaining_phase_budget": True},
                        "response_guard": {"no_new_gross_masking_failure": True}}.items():
        artifacts[name] = _write(tmp_path / (name + ".json"), value)
    relative = "src/honf_runtime/source_read_protocol.py"
    files = {relative: hashlib.sha256((project / relative).read_bytes()).hexdigest()}
    gate = {"schema": "source_read_full1000_qualification_v1", "dataset": "thermal",
            "source_reader_variant": variant, "development_epoch": 2500,
            "full_recipe_sha256": _canonical(recipe), "authorized_run_directory": str(tmp_path / "run"),
            "full_loaded_source": {"files": files, "sha256": _canonical(files)}, "evidence_files": artifacts}
    path = tmp_path / "qualification.json"
    _write(path, gate)
    return recipe, path, gate


def test_full_permission_is_bound_to_one_run_completed_age_and_immutable_evidence(tmp_path):
    recipe, path, gate = _evidence(tmp_path)
    assert validate_source_read_qualification(path, recipe, output_dir=tmp_path / "run")["development_epoch"] == 2500
    with pytest.raises(ValueError, match="exactly one"):
        validate_source_read_qualification(path, recipe, output_dir=tmp_path / "second_run")
    gate["development_epoch"] = 1000
    _write(path, gate)
    with pytest.raises(ValueError, match="saved age"):
        validate_source_read_qualification(path, recipe, output_dir=tmp_path / "run")
    gate["development_epoch"] = 2500
    _write(path, gate)
    Path(gate["evidence_files"]["candidate_checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_source_read_qualification(path, recipe, output_dir=tmp_path / "run")


@pytest.mark.parametrize("variant,artifact,key,value,match", [
    ("D-scale", "native_accuracy", "native_tail_review_consistent", False, "tails"),
    ("D-scale", "native_accuracy", "primary_p90", {"v": 1.0}, "ratios"),
    ("S-Utility", "cost", "actual_fine_retained_fraction", .71, "fine rows"),
    ("S-Utility", "cost", "complete_high_m_time_ratio", 1.2, "operating point"),
])
def test_predeclared_native_and_actual_work_failures_reject(tmp_path, variant, artifact, key, value, match):
    recipe, path, gate = _evidence(tmp_path, variant)
    item = gate["evidence_files"][artifact]
    data = json.loads(Path(item["path"]).read_text())
    data["qualification_metrics"][key] = value
    gate["evidence_files"][artifact] = _write(Path(item["path"]), data)
    _write(path, gate)
    with pytest.raises(ValueError, match=match):
        validate_source_read_qualification(path, recipe, output_dir=tmp_path / "run")
