"""Input-plan contracts; these tests never dispatch a physical solver."""

from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from hetmf.interventions import make_counterfactuals, physical_input_hash
from hetmf.study_design import (build_formal_plan, build_pilot_plan, hash_plan,
                                replay_pilot_layout_diagnostics, resolve_state)
from hetmf.formal import (bind_gpu2_visibility, check_materialization_budget, dry_run_formal,
                          pack_verified_shard, require_remaining_budget, validate_accepted_source,
                          validate_approval)
from hetmf.io import sha256_file, sha256_json


def test_pilot_is_frozen_and_deduplicates_quartet_parents():
    first = build_pilot_plan(20261017)
    second = build_pilot_plan(20261017)
    assert first["plan_hash"] == second["plan_hash"] == hash_plan(first)
    assert first["counts"] == {"representatives": 12, "quartets": 6, "quartet_endpoints": 24,
                               "throughput": 256, "distinct_states": 286}
    assert len({s["state_id"] for s in first["scenes"]}) == 286
    assert len({s["input_hash"] for s in first["scenes"]}) == 286
    assert set(first["representative_ids"]) == {f"R{i:02d}" for i in range(12)}
    assert len(first["throughput_ids"]) == 256
    reps = {s["state_id"]: s for s in first["scenes"][:12]}
    for changed in ("R08", "R09"):
        assert reps[changed]["contacts"] == reps["R02"]["contacts"]
        assert reps[changed]["environment"]["s"] == reps["R02"]["environment"]["s"]
    assert reps["R08"]["environment"]["beta"] == reps["R02"]["environment"]["beta"]
    assert reps["R09"]["environment"]["kappa"] == reps["R02"]["environment"]["kappa"]
    assert all(s["split"] == "pilot" for s in first["scenes"])


def test_pilot_quartets_freeze_independent_edits():
    pilot = build_pilot_plan()
    by_id = {s["state_id"]: s for s in pilot["scenes"]}
    for q in pilot["quartets"]:
        f00, f10, f01, f11 = (by_id[state_id] for state_id in q["endpoint_state_ids"])
        for scene in (f10, f01, f11):
            assert scene["environment_root_id"] == f00["environment_root_id"]
            assert scene["layout_root_id"] == f00["layout_root_id"]
            assert scene["input_hash"] == physical_input_hash(scene)
        if q["kind"] == "mm_voltages":
            assert [c["u"] for c in f11["contacts"]][2:] == [c["u"] for c in f00["contacts"]][2:]
            assert f10["contacts"][1] == f00["contacts"][1]
            assert f01["contacts"][0] == f00["contacts"][0]
            assert f11["contacts"][0] == f10["contacts"][0]
            assert f11["contacts"][1] == f01["contacts"][1]
        if q["kind"] == "ee_cross":
            assert f01["environment"]["s"] == f00["environment"]["s"]
            assert f10["environment"]["beta"] == f00["environment"]["beta"]
            assert f11["environment"]["s"] == f10["environment"]["s"]
            assert f11["environment"]["beta"] == f01["environment"]["beta"]
    assert [q["kind"] for q in pilot["quartets"]] == [
        "me_patch_move", "me_rotate_gamma", "mm_voltages", "mm_moves", "ee_electrical", "ee_cross"]


def test_pilot_layout_replay_sidecar_preserves_frozen_plan():
    pilot = build_pilot_plan()
    original_hash = pilot["plan_hash"]
    sidecar = replay_pilot_layout_diagnostics(pilot)
    assert sidecar["pilot_plan_hash"] == original_hash == hash_plan(pilot)
    assert sidecar["layout_roots"] == 266  # 10 independent representative + 256 throughput layouts
    assert sidecar["geometry_proposals"] == sidecar["rejected_geometry_proposals"] + 266
    altered = dict(pilot)
    altered["scenes"] = list(pilot["scenes"])
    altered["scenes"][0] = dict(altered["scenes"][0])
    altered["scenes"][0]["contacts"] = [dict(c) for c in altered["scenes"][0]["contacts"]]
    altered["scenes"][0]["contacts"][0]["x"] += 0.001
    altered["plan_hash"] = hash_plan(altered)
    try:
        replay_pilot_layout_diagnostics(altered)
    except ValueError:
        pass
    else:
        raise AssertionError("layout replay accepted a changed frozen contact")


def test_formal_counts_roots_degrees_and_nested_ordinary_inputs():
    plans = {name: build_formal_plan(name) for name in ("compact", "standard", "expanded")}
    for name, plan in plans.items():
        scale = {"compact": 1, "standard": 2, "expanded": 4}[name]
        assert len(plan["states"]) == 2304 * scale
        assert plan["split_counts"] == {"train": 1440 * scale, "dev": 240 * scale,
                                        "test": 240 * scale, "panel_new_environment": 64 * scale,
                                        "panel_new_layout": 64 * scale, "panel_new_pair": 64 * scale,
                                        "panel_extrapolation": 64 * scale, "audit": 128 * scale}
        ordinary = [r for r in plan["states"] if r["split"] in ("train", "dev", "test")]
        for split in ("train", "dev", "test"):
            rows = [r for r in ordinary if r["split"] == split]
            assert len({(r["environment_root_id"], r["layout_root_id"]) for r in rows}) == len(rows)
            assert set(Counter(r["environment_root_id"] for r in rows).values()) == {12}
            assert set(Counter(r["layout_root_id"] for r in rows).values()) == ({6} if split == "train" else {5})
        roots = {split: ({r["environment_root_id"] for r in ordinary if r["split"] == split},
                         {r["layout_root_id"] for r in ordinary if r["split"] == split})
                 for split in ("train", "dev", "test")}
        assert all(not (roots[a][axis] & roots[b][axis]) for a, b in
                   (("train", "dev"), ("train", "test"), ("dev", "test")) for axis in (0, 1))
        train_env, train_layout = roots["train"]
        for row in plan["states"]:
            if "environment_seen_in_train" in row:
                assert row["environment_seen_in_train"] == (row["environment_root_id"] in train_env)
                assert row["layout_seen_in_train"] == (row["layout_root_id"] in train_layout)
        assert len(plan["quartets"]) == 32 * scale
        assert plan["plan_hash"] == hash_plan(plan)
        assert resolve_state(plan, plan["states"][-1])["input_hash"] == plan["states"][-1]["input_hash"]
    for smaller, larger in (("compact", "standard"), ("standard", "expanded")):
        earlier = {r["state_id"]: r["input_hash"] for r in plans[smaller]["states"]}
        later = {r["state_id"]: r["input_hash"] for r in plans[larger]["states"]}
        assert all(later[state_id] == digest for state_id, digest in earlier.items())


def test_unknown_profile_and_intervention_refused():
    try:
        build_formal_plan("unapproved")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown profile accepted")
    try:
        make_counterfactuals(build_pilot_plan()["scenes"][0], "invented", "Qx")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown intervention accepted")


def test_formal_dry_run_is_read_only_and_execution_needs_yes(tmp_path):
    plan = build_formal_plan("compact")
    summary = dry_run_formal(plan, grid=96)
    assert summary["states"] == 2304
    assert summary["physical_solves"] == 0 and summary["writes"] == 0
    assert summary["approval_required"] is True
    assert list(tmp_path.iterdir()) == []
    try:
        validate_approval(plan, {}, root=tmp_path, yes=False)
    except PermissionError:
        pass
    else:
        raise AssertionError("formal approval bypassed explicit yes")
    corrupted = dict(plan)
    corrupted["plan_hash"] = "0" * 64
    try:
        dry_run_formal(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("formal dry-run accepted corrupt plan hash")


def test_formal_approval_binds_qualified_precision_and_coefficients(tmp_path, monkeypatch):
    from hetmf.resources import GPU2_UUID
    plan = build_formal_plan("compact")
    release = tmp_path / "releases" / "demo"
    monkeypatch.setattr("hetmf.formal.OWNED_BASE", tmp_path)
    monkeypatch.setattr("hetmf.formal.require_owned_root", lambda path: Path(path).resolve())
    monkeypatch.setattr("hetmf.formal.require_remaining_budget", lambda root, approval: {})
    approval = {"schema": "hetmf_formal_approval_v1", "plan_hash": plan["plan_hash"],
                "max_states": len(plan["states"]), "grid": 96,
                "resolution_groups": {name: 96 for name in
                                      ("default", "m10", "m16", "high_contrast", "composite")},
                "precision": "fp64", "coefficient_bounds": {key: {"min": 1e-5, "max": 1e5}
                                                        for key in ("s", "kappa", "beta")},
                "gpu_uuid": GPU2_UUID, "disk_root": str(release), "max_elapsed_seconds": 3600,
                "retained_budget_bytes": 2**30, "temporary_budget_bytes": 2**28,
                "minimum_free_bytes": 2**30, "batch_size": 32}
    assert validate_approval(plan, approval, root=release, yes=True)["precision"] == "fp64"
    with pytest.raises(ValueError, match="only GPU2 FP64 qualified"):
        validate_approval(plan, {**approval, "precision": "mixed"}, root=release, yes=True)
    with pytest.raises(ValueError, match="exceed approved bounds"):
        validate_approval(plan, {**approval, "coefficient_bounds":
                                 {**approval["coefficient_bounds"], "s": {"min": 1, "max": 1.1}}},
                          root=release, yes=True)
    concurrent = {"mode": "concurrent", "minimum_free_mib": 16384,
                  "process_memory_fraction": 0.25}
    assert validate_approval(plan, {**approval, "gpu_sharing": concurrent},
                             root=release, yes=True)["precision"] == "fp64"
    with pytest.raises(ValueError, match="GPU sharing policy"):
        validate_approval(plan, {**approval, "gpu_sharing":
                                 {**concurrent, "minimum_free_mib": 8192}}, root=release, yes=True)
    with pytest.raises(ValueError, match="GPU sharing policy"):
        validate_approval(plan, {**approval, "gpu_sharing":
                                 {**concurrent, "process_memory_fraction": 0.5}}, root=release, yes=True)


def test_formal_gpu2_sharing_checks_each_batch(monkeypatch):
    from hetmf.formal import _check_gpu2_sharing
    monkeypatch.setattr("hetmf.formal.gpu2_status", lambda: {
        "free_mib": 20000, "compute_processes": ["999,GPU-other-job"]})
    sharing = {"gpu_sharing": {"mode": "concurrent", "minimum_free_mib": 16384,
                              "process_memory_fraction": 0.25}}
    assert _check_gpu2_sharing(sharing)["free_mib"] == 20000
    with pytest.raises(RuntimeError, match="foreign GPU2 job"):
        _check_gpu2_sharing({})
    monkeypatch.setattr("hetmf.formal.gpu2_status", lambda: {
        "free_mib": 16000, "compute_processes": ["999,GPU-other-job"]})
    with pytest.raises(RuntimeError, match="below approved concurrent floor"):
        _check_gpu2_sharing(sharing)


def test_formal_shard_packs_verified_sources_without_removing_them(tmp_path):
    scene = build_pilot_plan()["scenes"][0]
    grid = 4
    source = tmp_path / "accepted.h5"
    shapes = {"potential": (grid, grid), "temperature_rise": (grid, grid),
              "q_sheet": (grid, grid), "q_contact": (grid, grid),
              "jx_face": (grid, grid + 1), "jy_face": (grid + 1, grid),
              "heat_flux_x_face": (grid, grid + 1), "heat_flux_y_face": (grid + 1, grid),
              "terminal_current": (2,)}
    with h5py.File(source, "w") as handle:
        handle.attrs["accepted"] = True
        handle.attrs["grid_n"] = grid
        handle.attrs["input_sha256"] = sha256_json(scene)
        for field, shape in shapes.items():
            handle.create_dataset(field, data=np.ones(shape, dtype=np.float32))
    result = pack_verified_shard(tmp_path, [(scene, source)], grid=grid, shard_key="test")
    assert not result["reused"] and source.exists()
    with h5py.File(result["raw_path"]) as raw, h5py.File(result["processed_path"]) as processed:
        assert raw["contact_mask"][:].sum() == 2
        assert processed["fields"].shape == (1, grid, grid, 5)
        assert np.all(processed["fields"][0, :, :, -1] == 2)
    assert pack_verified_shard(tmp_path, [(scene, source)], grid=grid, shard_key="test")["reused"]
    from pathlib import Path
    Path(result["processed_path"]).unlink()
    repaired = pack_verified_shard(tmp_path, [(scene, source)], grid=grid, shard_key="test")
    assert not repaired["reused"] and Path(repaired["processed_path"]).exists()
    with h5py.File(source, "r+") as handle:
        handle.attrs["input_sha256"] = "corrupt"
    try:
        pack_verified_shard(tmp_path, [(scene, source)], grid=grid, shard_key="test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("packer accepted source whose input hash changed")


def test_formal_prewrite_budgets_and_cuda_binding_refuse_unsafe_start(tmp_path, monkeypatch):
    approval = {"retained_budget_bytes": 100, "temporary_budget_bytes": 30,
                "minimum_free_bytes": 20}
    check_materialization_budget(retained_used=40, next_retained_upper=20,
                                 next_temporary_upper=20, free_bytes=100, approval=approval)
    for values in ((90, 20, 20, 100), (40, 20, 31, 100), (40, 20, 20, 50)):
        try:
            check_materialization_budget(retained_used=values[0], next_retained_upper=values[1],
                                         next_temporary_upper=values[2], free_bytes=values[3], approval=approval)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unsafe materialization accepted: {values}")
    (tmp_path / "prior_raw").write_bytes(b"x" * 60)
    monkeypatch.setattr("hetmf.formal.require_data2", lambda **kwargs: {"free_bytes": 70})
    incremental = require_remaining_budget(tmp_path, {**approval, "temporary_budget_bytes": 10})
    assert incremental["retained_used_bytes"] == 60 and incremental["retained_remaining_bytes"] == 40
    monkeypatch.setattr("hetmf.formal.require_data2", lambda **kwargs: {"free_bytes": 50})
    try:
        require_remaining_budget(tmp_path, {**approval, "temporary_budget_bytes": 10})
    except RuntimeError:
        pass
    else:
        raise AssertionError("remaining budget ignored disk headroom")
    monkeypatch.setitem(sys.modules, "torch", object())
    try:
        bind_gpu2_visibility()
    except RuntimeError:
        pass
    else:
        raise AssertionError("formal solver accepted pre-imported torch")


def test_formal_resume_requires_ledger_checksum_and_terminal_vector(tmp_path):
    scene = build_pilot_plan()["scenes"][0]
    grid = 4
    solver_hash = "f" * 64
    raw = tmp_path / "raw"
    raw.mkdir()
    target = raw / f"{scene['state_id']}_n{grid}.h5"
    shapes = {"potential": (grid, grid), "temperature_rise": (grid, grid),
              "q_sheet": (grid, grid), "q_contact": (grid, grid),
              "jx_face": (grid, grid + 1), "jy_face": (grid + 1, grid),
              "heat_flux_x_face": (grid, grid + 1), "heat_flux_y_face": (grid + 1, grid)}
    with h5py.File(target, "w") as handle:
        handle.attrs["accepted"] = True
        handle.attrs["grid_n"] = grid
        handle.attrs["state_id"] = scene["state_id"]
        handle.attrs["input_sha256"] = sha256_json(scene)
        handle.attrs["solver_sha256"] = solver_hash
        for field, shape in shapes.items():
            handle.create_dataset(field, data=np.ones(shape, dtype=np.float32))
        handle.create_dataset("terminal_current", data=np.zeros(2, dtype=np.float64))
    with (tmp_path / "solver_ledger.jsonl").open("w") as handle:
        handle.write(json.dumps({"state_id": scene["state_id"], "grid": grid, "raw_path": str(target),
                                 "raw_sha256": sha256_file(target), "input_sha256": sha256_json(scene),
                                 "solver_sha256": solver_hash}) + "\n")
    with h5py.File(target, "r+") as handle:
        handle["potential"][0, 0] += 0.1  # finite but no longer the ledgered source
    try:
        validate_accepted_source(tmp_path, scene, grid, solver_hash)
    except RuntimeError as exc:
        assert "differs from solver ledger" in str(exc)
    else:
        raise AssertionError("resumption accepted a raw file with no matching checksum")
    with h5py.File(target, "r+") as handle:
        handle["terminal_current"][0] = np.nan
    try:
        validate_accepted_source(tmp_path, scene, grid, solver_hash)
    except RuntimeError as exc:
        assert "terminal current" in str(exc) or "terminal-current" in str(exc)
    else:
        raise AssertionError("resumption accepted nonfinite terminal current")
