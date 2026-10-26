"""Release integrity, immutable inheritance, and CPU preparation boundaries."""

from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from hetmf import formal
from hetmf.interventions import physical_input_hash
from hetmf.io import sha256_file, sha256_json, write_case_atomic
from hetmf.physics import solve_cpu
from hetmf.study_design import build_formal_plan, hash_plan
from hetmf.validation import audit_persisted


@pytest.fixture(scope="module")
def plan():
    return build_formal_plan("compact")


def _approval(plan, root, **extra):
    return {"schema": "hetmf_formal_approval_v1", "plan_hash": plan["plan_hash"],
            "max_states": len(plan["states"]), "grid": 48,
            "resolution_groups": {key: 48 for key in ("default", "m10", "m16", "high_contrast", "composite")},
            "precision": "fp64", "backend": formal.CPU_BACKEND, "gpu_uuid": None,
            "coefficient_bounds": {key: {"min": 1e-6, "max": 1e6} for key in ("s", "kappa", "beta")},
            "disk_root": str(root), "max_elapsed_seconds": 120, "batch_size": 2,
            "retained_budget_bytes": 2**28, "temporary_budget_bytes": 2**26,
            "minimum_free_bytes": 2**28, **extra}


def _fake_storage(monkeypatch, base):
    monkeypatch.setattr(formal, "OWNED_BASE", base)
    monkeypatch.setattr(formal, "require_owned_root", lambda path: Path(path).resolve())
    monkeypatch.setattr("hetmf.io.require_owned_root", lambda path: Path(path).resolve())
    monkeypatch.setattr(formal, "require_data2", lambda **kwargs: {"free_bytes": 2**40})


def _tiny(plan, count=1):
    small = deepcopy(plan)
    small["states"] = small["states"][:count]
    small["counts"]["total_states"] = count
    small["split_counts"] = {"train": count}
    small["quartets"] = []
    small["plan_hash"] = hash_plan(small)
    return small


def test_formal_rejects_false_split_counts_and_root_leakage(plan):
    corrupt = deepcopy(plan)
    corrupt["split_counts"]["train"] += 1
    corrupt["split_counts"]["test"] -= 1
    corrupt["plan_hash"] = hash_plan(corrupt)
    with pytest.raises(ValueError, match="split count"):
        formal.validate_formal_plan(corrupt)
    corrupt = deepcopy(plan)
    dev = next(row for row in corrupt["states"] if row["split"] == "dev")
    dev["environment_root_id"] = corrupt["states"][0]["environment_root_id"]
    corrupt["plan_hash"] = hash_plan(corrupt)
    with pytest.raises(ValueError, match="roots overlap"):
        formal.validate_formal_plan(corrupt)


def test_formal_checks_override_bounds_and_reports_pairing_confound(plan, tmp_path, monkeypatch):
    _fake_storage(monkeypatch, tmp_path)
    corrupt = deepcopy(plan)
    row = next(row for row in corrupt["states"] if row["split"] == "audit")
    row["input_override"]["environment"]["s"][0][0] = 1e7
    scene = {"environment": row["input_override"]["environment"], "contacts": row["input_override"]["contacts"]}
    row["input_hash"] = physical_input_hash(scene)
    corrupt["plan_hash"] = hash_plan(corrupt)
    root = tmp_path / "releases" / "bounded"
    with pytest.raises(ValueError, match="intervention coefficients exceed"):
        formal.validate_approval(corrupt, _approval(corrupt, root), root=root, yes=True, device="cpu")
    audit = formal.audit_formal_design(plan)
    assert audit["train_layout_morphology_confounded"]
    assert audit["splits"]["train"]["morphologies_per_layout_counts"] == {1: 240}
    assert audit["splits"]["dev"]["morphologies_per_layout_counts"] == {4: 48}


@pytest.mark.parametrize("field,index", [
    ("terminal_current", 0), ("jx_face", (5, 5)), ("jy_face", (5, 5)),
    ("heat_flux_x_face", (5, 5)), ("heat_flux_y_face", (5, 5)),
])
def test_persisted_audit_rejects_corrupted_constitutive_fields(plan, tmp_path, monkeypatch, field, index):
    _fake_storage(monkeypatch, tmp_path)
    for name in ("raw", "temporary"):
        (tmp_path / name).mkdir()
    scene = formal._safe_scene(plan, plan["states"][0])
    write_case_atomic(tmp_path, scene, solve_cpu(scene, 48), grid=48, solver_hash="a" * 64)
    assert audit_persisted(tmp_path, scene, 48)["accepted"]
    path = tmp_path / "raw" / f"{scene['state_id']}_n48.h5"
    with h5py.File(path, "r+") as handle:
        handle[field][index] += 0.1
    assert not audit_persisted(tmp_path, scene, 48)["accepted"]


def test_shard_reuse_checks_real_fields_and_metadata(plan, tmp_path, monkeypatch):
    _fake_storage(monkeypatch, tmp_path)
    for name in ("raw", "temporary"):
        (tmp_path / name).mkdir()
    scene = formal._safe_scene(plan, plan["states"][0])
    receipt = write_case_atomic(tmp_path, scene, solve_cpu(scene, 48), grid=48, solver_hash="a" * 64)
    members = [(scene, Path(receipt["raw_path"]))]
    shard = formal.pack_verified_shard(tmp_path, members, grid=48, shard_key="tiny")
    with h5py.File(shard["processed_path"], "r+") as handle:
        handle["fields"][0, 0, 0, 0] += 0.01
    with pytest.raises(RuntimeError, match="processed payload differs"):
        formal.pack_verified_shard(tmp_path, members, grid=48, shard_key="tiny")


def test_cpu_formal_never_binds_gpu_and_resumes_with_same_recipe(plan, tmp_path, monkeypatch):
    _fake_storage(monkeypatch, tmp_path)
    small = _tiny(plan)
    # Real plan validation is exercised above; this fixture isolates one state
    # for an actual CPU solve, atomic write, physical audit, and resume.
    monkeypatch.setattr(formal, "validate_formal_plan", lambda p: {
        "plan_hash": p["plan_hash"], "profile": p["profile"], "states": len(p["states"]),
        "split_counts": p["split_counts"]})
    monkeypatch.setattr(formal, "bind_gpu2_visibility", lambda: pytest.fail("CPU generation touched GPU binding"))
    monkeypatch.setattr(formal, "gpu2_status", lambda: pytest.fail("CPU generation inspected GPU"))
    root = tmp_path / "releases" / "cpu"
    approval = _approval(small, root)
    result = formal.run_formal(small, approval, root=root, yes=True, device="cpu")
    assert result["complete"] and result["physical_solves"] == 1
    recipes = json.loads((root / "inputs" / "numerical_recipes.json").read_text())
    assert recipes["48"]["backend"] == formal.CPU_BACKEND
    resumed = formal.run_formal(small, approval, root=root, yes=True, device="cpu", resume=True)
    assert resumed["complete"] and resumed["physical_solves"] == 0 and resumed["reused"] == 1
    # Resource-policy changes do not change the physical numerical recipe.
    assert formal._solver_hash(approval, 48) == formal._solver_hash({**approval, "batch_size": 1}, 48)


def test_inheritance_retains_original_solver_approval_bytes_and_readonly_resolution(plan, tmp_path, monkeypatch):
    _fake_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(formal, "validate_formal_plan", lambda p: {
        "plan_hash": p["plan_hash"], "profile": p["profile"], "states": len(p["states"]),
        "split_counts": p["split_counts"]})
    source_plan = _tiny(plan)
    source = tmp_path / "releases" / "source"
    source_approval = _approval(source_plan, source, backend=formal.GPU_BACKEND, gpu_uuid=formal.GPU2_UUID)
    # Exercise the immutable legacy GPU metadata format using a tiny CPU
    # fixture, without importing torch or occupying a GPU during tests.
    for name in ("inputs", "raw", "temporary"):
        (source / name).mkdir(parents=True, exist_ok=True)
    (source / "inputs" / "formal_plan.json").write_text(json.dumps(source_plan))
    (source / "inputs" / "formal_approval.json").write_text(json.dumps(source_approval))
    source_scene = formal._safe_scene(source_plan, source_plan["states"][0])
    source_receipt = write_case_atomic(source, source_scene, solve_cpu(source_scene, 48), grid=48,
                                       solver_hash=formal._solver_hash(source_approval, 48))
    formal._append_journal(source / "solver_ledger.jsonl", {
        **source_receipt, "approval_hash": sha256_json(source_approval), "status": "accepted"})
    context = formal.load_release_context(source)
    original = formal.resolve_release_row(source, source_plan["states"][0], context=context)
    original_path = Path(original["raw_path"])
    original_sha = sha256_file(original_path)
    destination_plan = _tiny(plan, count=2)
    destination = tmp_path / "releases" / "expanded"
    approval = _approval(destination_plan, destination, inheritance={
        "root": str(source), "plan_hash": source_plan["plan_hash"], "approval_hash": sha256_json(source_approval)})
    result = formal.run_formal(destination_plan, approval, root=destination, yes=True, device="cpu")
    assert result["complete"] and result["physical_solves"] == 1 and result["inherited"] == 1
    expanded_context = formal.load_release_context(destination)
    inherited = formal.resolve_release_row(destination, source_plan["states"][0]["state_id"], context=expanded_context)
    local = Path(inherited["raw_path"])
    assert local.stat().st_ino == original_path.stat().st_ino
    assert sha256_file(local) == original_sha == sha256_file(original_path)
    assert inherited["approval_hash"] == sha256_json(source_approval)
    assert inherited["solver_sha256"] == original["solver_sha256"]
    assert inherited["solver_sha256"] != formal._solver_hash(approval, 48)
    assert inherited["inheritance"]["source_raw_path"] == str(original_path)
    resumed = formal.run_formal(destination_plan, approval, root=destination, yes=True, device="cpu", resume=True)
    assert resumed["reused"] == 2 and resumed["physical_solves"] == 0
    # An unchanged H5/ledger pair cannot bypass the source release binding.
    bad = deepcopy(expanded_context)
    key = (inherited["scene"]["state_id"], 48)
    bad["ledger"][key]["inheritance"]["approval_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="release binding mismatch"):
        formal.resolve_release_row(destination, source_plan["states"][0]["state_id"], context=bad)
