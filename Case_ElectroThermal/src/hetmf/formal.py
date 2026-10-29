"""Approval-bound formal generation; plan inspection has no physical solves.

The pilot never calls :func:`run_formal`. An approval is an exact, external
record for one frozen manifest and one resource policy; `--yes` is an additional
required execution switch. This module does not mint approval records.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from .interventions import physical_input_hash
from .io import initialize_root, sha256_file, sha256_json, validate_raw_case, write_case_atomic, write_json_atomic
from .resources import GPU2_UUID, OWNED_BASE, gpu2_status, require_data2, require_exclusive_gpu2, require_owned_root
from .study_design import hash_plan, resolve_state


CPU_BACKEND = "cpu_sparse_direct_fp64"
GPU_BACKEND = "gpu_batched_pcg_fp64"


def audit_formal_design(plan: dict) -> dict:
    """Describe root reuse and morphology exposure without changing frozen rows."""
    result = {}
    for split in ("train", "dev", "test"):
        rows = [row for row in plan["states"] if row["split"] == split]
        exposure = defaultdict(set)
        for row in rows:
            exposure[row["layout_root_id"]].add(plan["environments"][row["environment_root_id"]]["family"])
        result[split] = {
            "states": len(rows), "environment_roots": len({r["environment_root_id"] for r in rows}),
            "layout_roots": len(exposure),
            "environment_reuse_counts": dict(Counter(Counter(r["environment_root_id"] for r in rows).values())),
            "layout_reuse_counts": dict(Counter(Counter(r["layout_root_id"] for r in rows).values())),
            "morphologies_per_layout_counts": dict(Counter(map(len, exposure.values()))),
        }
    return {"schema": "hetmf_formal_design_audit_v1", "plan_hash": plan["plan_hash"], "splits": result,
            "train_layout_morphology_confounded": any(
                count == 1 for count in result["train"]["morphologies_per_layout_counts"])}


def validate_formal_plan(plan: dict) -> dict:
    """Check immutable input-plan identity and arithmetic without solving."""
    if plan.get("schema") != "hetmf_input_plan_v1" or plan.get("kind") != "formal_proposal":
        raise ValueError("formal input plan schema/kind required")
    if plan.get("formal_generation_authorized") is not False:
        raise ValueError("plan must remain an unapproved proposal")
    if plan.get("plan_hash") != hash_plan(plan):
        raise ValueError("formal plan hash mismatch")
    rows = plan["states"]
    if len(rows) != plan["counts"]["total_states"] or len({r["state_id"] for r in rows}) != len(rows):
        raise ValueError("formal state count or IDs inconsistent")
    if dict(Counter(row["split"] for row in rows)) != plan["split_counts"]:
        raise ValueError("formal split count mismatch")
    allowed_splits = {"train", "dev", "test", "audit", "panel_new_environment", "panel_new_layout",
                      "panel_new_pair", "panel_extrapolation"}
    if set(plan["split_counts"]) != allowed_splits:
        raise ValueError("formal split names inconsistent")
    roots = {split: ({r["environment_root_id"] for r in rows if r["split"] == split},
                     {r["layout_root_id"] for r in rows if r["split"] == split})
             for split in ("train", "dev", "test", "audit")}
    for i, first in enumerate(roots):
        for second in list(roots)[i + 1:]:
            if any(roots[first][axis] & roots[second][axis] for axis in (0, 1)):
                raise ValueError("formal core/audit roots overlap across splits")
    # IDs alone cannot prove independent physical roots: reject an identical
    # environment or layout relabelled into another protected partition.
    physical_roots = {}
    for split, (environment_ids, layout_ids) in roots.items():
        environment_hashes = {physical_input_hash({"environment": plan["environments"][key], "contacts": []})
                              for key in environment_ids}
        layout_hashes = {sha256_json(sorted(
            ([contact[name] for name in ("x", "y", "r", "gamma", "u")]
             for contact in plan["layouts"][key]["contacts"]))) for key in layout_ids}
        physical_roots[split] = (environment_hashes, layout_hashes)
        if len(environment_hashes) != len(environment_ids) or len(layout_hashes) != len(layout_ids):
            raise ValueError("formal catalogue repeats physical roots under different IDs")
    for i, first in enumerate(physical_roots):
        for second in list(physical_roots)[i + 1:]:
            if any(physical_roots[first][axis] & physical_roots[second][axis] for axis in (0, 1)):
                raise ValueError("formal physical roots leak across protected splits")
    seen_hashes = set()
    core_pairs = {(r["environment_root_id"], r["layout_root_id"]) for r in rows if r["split"] == "train"}
    for row in rows:
        if row["environment_root_id"] not in plan["environments"] or row["layout_root_id"] not in plan["layouts"]:
            raise ValueError("formal row refers to missing catalogue root")
        if not isinstance(row["input_hash"], str) or len(row["input_hash"]) != 64:
            raise ValueError("formal row has missing physical input hash")
        scene = resolve_state(plan, row)
        if scene["input_hash"] in seen_hashes:
            raise ValueError("duplicate formal physical inputs")
        seen_hashes.add(scene["input_hash"])
        if "input_override" in row and row["split"] != "audit":
            raise ValueError("formal input overrides belong only to audit quartets")
        if row["split"].startswith("panel_"):
            exposure = (row["environment_root_id"] in roots["train"][0],
                        row["layout_root_id"] in roots["train"][1])
            expected = {"panel_new_environment": (False, True), "panel_new_layout": (True, False),
                        "panel_new_pair": (True, True), "panel_extrapolation": (False, False)}[row["split"]]
            if exposure != expected or exposure != (row.get("environment_seen_in_train"), row.get("layout_seen_in_train")):
                raise ValueError("formal panel component exposure mismatch")
            if row["split"] == "panel_new_pair" and (row["environment_root_id"], row["layout_root_id"]) in core_pairs:
                raise ValueError("new-pair panel reused a TRAIN pair")
            if any(row["environment_root_id"] in roots[split][0] or row["layout_root_id"] in roots[split][1]
                   for split in ("dev", "test", "audit")):
                raise ValueError("formal panel exposes protected core/audit roots")
    by_id = {row["state_id"]: row for row in rows}
    audit_ids = set()
    for quartet in plan["quartets"]:
        endpoints = quartet["endpoint_state_ids"]
        if len(endpoints) != 4 or len(set(endpoints)) != 4:
            raise ValueError("formal quartet must contain four unique endpoints")
        members = [by_id.get(state_id) for state_id in endpoints]
        if any(row is None or row["split"] != "audit" or row.get("quartet_id") != quartet["quartet_id"] for row in members):
            raise ValueError("formal quartet membership mismatch")
        if ({row.get("quartet_endpoint") for row in members} != {"00", "10", "01", "11"} or
                len({(row["environment_root_id"], row["layout_root_id"]) for row in members}) != 1 or audit_ids & set(endpoints)):
            raise ValueError("formal quartet lineage mismatch")
        audit_ids.update(endpoints)
    if audit_ids != {r["state_id"] for r in rows if r["split"] == "audit"}:
        raise ValueError("formal audit states absent from quartets")
    return {"plan_hash": plan["plan_hash"], "profile": plan["profile"],
            "states": len(rows), "split_counts": plan["split_counts"],
            "environment_roots": len(plan["environments"]), "layout_roots": len(plan["layouts"])}


def dry_run_formal(plan: dict, *, root: Path | None = None, grid: int = 96) -> dict:
    """Read-only formal preflight: no directory creation and no PDE calls."""
    summary = validate_formal_plan(plan)
    if grid not in (48, 96, 192, 384):
        raise ValueError("unsupported formal grid")
    summary["requested_default_grid"] = grid
    summary["physical_solves"] = 0
    summary["writes"] = 0
    summary["approval_required"] = True
    if root is not None:
        capacity = require_data2()
        owned = require_owned_root(root)
        summary["proposed_root"] = str(owned)
        summary["free_bytes"] = capacity["free_bytes"]
    return summary


def validate_approval(plan: dict, approval: dict, *, root: Path, yes: bool,
                      device: str | None = None) -> dict:
    """Refuse any ambiguous or underspecified formal authorization."""
    summary = validate_formal_plan(plan)
    if not yes:
        raise PermissionError("formal generation requires explicit --yes")
    required = {"schema", "plan_hash", "max_states", "grid", "resolution_groups", "precision",
                "coefficient_bounds",
                "gpu_uuid", "disk_root", "max_elapsed_seconds", "retained_budget_bytes",
                "temporary_budget_bytes", "minimum_free_bytes", "batch_size"}
    missing = required - approval.keys()
    if missing:
        raise ValueError(f"formal approval missing fields: {sorted(missing)}")
    if approval["schema"] != "hetmf_formal_approval_v1":
        raise ValueError("formal approval schema mismatch")
    if approval["plan_hash"] != summary["plan_hash"] or int(approval["max_states"]) != summary["states"]:
        raise ValueError("formal approval plan hash/state bound mismatch")
    backend = approval.get("backend", GPU_BACKEND)
    if backend not in (GPU_BACKEND, CPU_BACKEND):
        raise ValueError("unsupported formal backend")
    expected_device = "cpu" if backend == CPU_BACKEND else "cuda:0"
    if device is not None and device != expected_device:
        raise ValueError("formal approval backend/device mismatch")
    if backend == GPU_BACKEND and approval["gpu_uuid"] != GPU2_UUID:
        raise ValueError("formal approval must bind physical GPU2 and cuda:0")
    if backend == CPU_BACKEND and approval["gpu_uuid"] is not None:
        raise ValueError("CPU formal approval requires gpu_uuid=null")
    owned = require_owned_root(root)
    releases = OWNED_BASE / "releases"
    if Path(approval["disk_root"]).resolve(strict=False) != owned or owned.parent != releases:
        raise ValueError("formal approval root must be an exact owned releases/<id> path")
    groups = approval["resolution_groups"]
    if not isinstance(groups, dict) or groups.get("default") != approval["grid"]:
        raise ValueError("formal approval grid/group mismatch")
    if set(groups) != {"default", "m10", "m16", "high_contrast", "composite"}:
        raise ValueError("formal approval must bind every declared resolution group")
    if not all(name in {"default", "m10", "m16", "high_contrast", "composite"} and
               value in (48, 96, 192, 384) for name, value in groups.items()):
        raise ValueError("unsupported formal resolution group")
    if approval["precision"] != "fp64":
        if backend == CPU_BACKEND:
            raise ValueError("CPU sparse-direct formal backend requires FP64")
        raise ValueError("only GPU2 FP64 qualified in this pilot; formal mixed precision is unavailable")
    bounds = approval["coefficient_bounds"]
    if not isinstance(bounds, dict) or set(bounds) != {"s", "kappa", "beta"}:
        raise ValueError("formal approval must bind s/kappa/beta coefficient bounds")
    for key in ("s", "kappa", "beta"):
        entry = bounds[key]
        if not isinstance(entry, dict) or set(entry) != {"min", "max"}:
            raise ValueError(f"formal {key} bounds need min/max")
        low, high = float(entry["min"]), float(entry["max"])
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low <= high):
            raise ValueError(f"invalid formal {key} bounds")
        for environment in plan["environments"].values():
            values = np.asarray(environment[key], dtype=np.float64)
            if not np.isfinite(values).all() or values.min() < low or values.max() > high:
                raise ValueError(f"formal {key} control coefficients exceed approved bounds")
        for row in plan["states"]:
            if "input_override" in row:
                values = np.asarray(row["input_override"]["environment"][key], dtype=np.float64)
                if not np.isfinite(values).all() or values.min() < low or values.max() > high:
                    raise ValueError(f"formal {key} intervention coefficients exceed approved bounds")
    if int(approval["batch_size"]) < 1 or int(approval["batch_size"]) > 32:
        raise ValueError("formal batch size outside 1..32")
    sharing = approval.get("gpu_sharing", {"mode": "exclusive"})
    if sharing == {"mode": "exclusive"}:
        pass
    elif (isinstance(sharing, dict) and set(sharing) ==
          {"mode", "minimum_free_mib", "process_memory_fraction"} and
          sharing["mode"] == "concurrent" and
          isinstance(sharing["minimum_free_mib"], int) and
          sharing["minimum_free_mib"] >= 16 * 1024 and
          isinstance(sharing["process_memory_fraction"], (int, float)) and
          0 < sharing["process_memory_fraction"] <= 0.25):
        pass
    else:
        raise ValueError("formal GPU sharing policy is invalid or exceeds bounded concurrent limits")
    if backend == CPU_BACKEND and "gpu_sharing" in approval:
        raise ValueError("CPU formal approval cannot request GPU sharing")
    if any(int(approval[key]) <= 0 for key in
           ("max_elapsed_seconds", "retained_budget_bytes", "temporary_budget_bytes", "minimum_free_bytes")):
        raise ValueError("formal time/storage limits must be positive")
    require_remaining_budget(owned, approval)
    if "inheritance" in approval:
        _load_inheritance(plan, approval)
    return {**summary, "approval_hash": sha256_json(approval), "disk_root": str(owned),
            "resolution_groups": groups, "precision": approval["precision"], "backend": backend}


def _grid(row: dict, approval: dict) -> int:
    challenge = row.get("challenge", "default")
    groups = approval["resolution_groups"]
    if challenge != "default" and challenge not in groups:
        raise ValueError(f"unapproved resolution group: {challenge}")
    return int(groups[challenge])


def _safe_scene(plan: dict, row: dict) -> dict:
    scene = resolve_state(plan, row)
    if physical_input_hash(scene) != row["input_hash"]:
        raise ValueError("formal row physical hash mismatch")
    scene["formal_state_id"] = scene["state_id"]
    scene["state_id"] = "f_" + hashlib.sha256(scene["formal_state_id"].encode()).hexdigest()[:24]
    return scene


def _load_accepted_ledger(root: Path) -> dict[tuple[str, int], dict]:
    ledger = Path(root) / "solver_ledger.jsonl"
    if not ledger.is_file():
        return {}
    index = {}
    for line in ledger.read_text().splitlines():
        record = json.loads(line)
        index[(record.get("state_id"), record.get("grid"))] = record
    return index


def _solver_recipe(approval: dict, grid: int) -> dict:
    """Describe numerical identity separately from where a run may execute."""
    if approval.get("backend", GPU_BACKEND) == GPU_BACKEND:
        return {"approval_hash": sha256_json(approval), "grid": grid,
                "precision": approval["precision"], "solver": "batched_matrix_free_pcg_v1"}
    import scipy
    return {"backend": CPU_BACKEND, "grid": grid, "precision": "fp64", "quadrature_order": 4,
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "sources": {name: sha256_file(Path(__file__).with_name(name + ".py"))
                          for name in ("physics", "finite_volume", "contacts", "environment")}}


def _solver_hash(approval: dict, grid: int) -> str:
    """Keep legacy GPU identity; CPU recipes are independent of resource approval."""
    return sha256_json(_solver_recipe(approval, grid))


def _load_inheritance(plan: dict, approval: dict) -> dict | None:
    """Read and bind an immutable source release; never change its provenance."""
    spec = approval.get("inheritance")
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {"root", "plan_hash", "approval_hash"}:
        raise ValueError("inheritance requires exact root, plan_hash and approval_hash")
    source = require_owned_root(Path(spec["root"]))
    if source.parent != OWNED_BASE / "releases" or source == Path(approval["disk_root"]).resolve():
        raise ValueError("inheritance source must be a different exact owned release")
    source_plan = json.loads((source / "inputs" / "formal_plan.json").read_text())
    source_approval = json.loads((source / "inputs" / "formal_approval.json").read_text())
    if (source_plan.get("plan_hash") != spec["plan_hash"] or hash_plan(source_plan) != spec["plan_hash"] or
            sha256_json(source_approval) != spec["approval_hash"] or
            source_approval.get("plan_hash") != spec["plan_hash"] or
            Path(source_approval.get("disk_root", "")).resolve() != source):
        raise ValueError("inheritance source plan/approval identity mismatch")
    if "inheritance" in source_approval:
        raise ValueError("chained release inheritance is not supported")
    destination = {r["state_id"]: r for r in plan["states"]}
    for row in source_plan["states"]:
        current = destination.get(row["state_id"])
        if current is None or current["input_hash"] != row["input_hash"] or current["split"] != row["split"]:
            raise ValueError("inherited release is not an exact nested input/split subset")
        if _grid(row, source_approval) != _grid(current, approval):
            raise ValueError("inherited source grid differs from approved destination grid")
    return {"root": source, "plan": source_plan, "approval": source_approval,
            "rows": {r["state_id"]: r for r in source_plan["states"]},
            "ledger": _load_accepted_ledger(source)}


def validate_release_source(root: Path, scene: dict, grid: int, approval: dict, *,
                            ledger_index: dict | None = None,
                            inheritance_context: dict | None = None) -> dict:
    """Validate a local release row, including immutable inherited source lineage.

    Callers scanning a release should pass a frozen ledger and the result of
    ``_load_inheritance(plan, approval)`` once, avoiding repeated catalogue IO.
    The returned source path always resides in this release's ``raw`` folder.
    """
    index = _load_accepted_ledger(root) if ledger_index is None else ledger_index
    record = index.get((scene["state_id"], grid))
    if record is None:
        raise RuntimeError("release source absent from solver ledger")
    raw_path = Path(root) / "raw" / f"{scene['state_id']}_n{grid}.h5"
    with h5py.File(raw_path, "r") as handle:
        if (handle.attrs.get("state_id") != scene["state_id"] or
                handle.attrs.get("units") != "dimensionless" or handle.attrs.get("field_dtype") != "float64" or
                any(key not in handle or handle[key].dtype != np.dtype("float64")
                    for key in (*RAW_FIELDS, "terminal_current"))):
            raise RuntimeError("formal raw source FP64/state/units contract mismatch")
    inheritance = record.get("inheritance")
    expected_solver = _solver_hash(approval, grid)
    if inheritance is not None:
        if "inheritance" not in approval:
            raise RuntimeError("unapproved release source inheritance")
        context = inheritance_context
        if context is None:
            plan = json.loads((Path(root) / "inputs" / "formal_plan.json").read_text())
            context = _load_inheritance(plan, approval)
        spec = approval["inheritance"]
        if any(inheritance.get(key) != spec[key] for key in ("root", "plan_hash", "approval_hash")):
            raise RuntimeError("inherited source release binding mismatch")
        formal_id = scene.get("formal_state_id")
        if formal_id not in context["rows"]:
            raise RuntimeError("inherited state absent from approved source plan")
        source_scene = _safe_scene(context["plan"], context["rows"][formal_id])
        if sha256_json(source_scene) != sha256_json(scene):
            raise RuntimeError("inherited full input differs from approved source")
        expected_solver = _solver_hash(context["approval"], grid)
        source_audit = validate_accepted_source(context["root"], source_scene, grid, expected_solver,
                                               ledger_index=context["ledger"])
        source_path = context["root"] / "raw" / f"{source_scene['state_id']}_n{grid}.h5"
        if (inheritance.get("source_raw_path") != str(source_path) or
                inheritance.get("source_raw_sha256") != source_audit["raw_sha256"] or
                inheritance.get("source_solver_sha256") != expected_solver or
                inheritance.get("method") != "immutable_hardlink" or
                record.get("approval_hash") != spec["approval_hash"]):
            raise RuntimeError("inherited source checksum/provenance mismatch")
    elif record.get("approval_hash") != sha256_json(approval):
        raise RuntimeError("release source approval provenance mismatch")
    audit = validate_accepted_source(root, scene, grid, expected_solver, ledger_index=index)
    return {**audit, "raw_path": str(Path(root) / "raw" / f"{scene['state_id']}_n{grid}.h5"),
            "solver_sha256": expected_solver, "approval_hash": record.get("approval_hash"),
            "inheritance": inheritance}


def load_release_context(root: Path) -> dict:
    """Pin a release's on-disk plan, approval and ledgers for read-only consumers."""
    owned = require_owned_root(root)
    plan = json.loads((owned / "inputs" / "formal_plan.json").read_text())
    approval = json.loads((owned / "inputs" / "formal_approval.json").read_text())
    validate_formal_plan(plan)
    if approval.get("plan_hash") != plan["plan_hash"] or Path(approval.get("disk_root", "")).resolve() != owned:
        raise RuntimeError("release plan/approval/root binding mismatch")
    return {"root": owned, "plan": plan, "approval": approval, "approval_hash": sha256_json(approval),
            "rows": {row["state_id"]: row for row in plan["states"]},
            "ledger": _load_accepted_ledger(owned), "inheritance": _load_inheritance(plan, approval)}


def resolve_release_row(root: Path, row: dict | str, *, context: dict | None = None) -> dict:
    """Resolve and physically verify a row without trusting a caller-supplied hash."""
    context = load_release_context(root) if context is None else context
    if Path(root).resolve() != context["root"]:
        raise ValueError("release context belongs to another root")
    state_id = row if isinstance(row, str) else row["state_id"]
    frozen = context["rows"].get(state_id)
    if frozen is None or (not isinstance(row, str) and row != frozen):
        raise ValueError("release row differs from frozen input plan")
    scene = _safe_scene(context["plan"], frozen)
    grid = _grid(frozen, context["approval"])
    audit = validate_release_source(context["root"], scene, grid, context["approval"],
                                    ledger_index=context["ledger"], inheritance_context=context["inheritance"])
    return {**audit, "scene": scene, "grid": grid, "row": frozen}


def validate_accepted_source(root: Path, scene: dict, grid: int, solver_hash: str,
                             *, ledger_index: dict | None = None) -> dict:
    """Require an accepted ledger checksum and post-storage physical audit."""
    target = Path(root) / "raw" / f"{scene['state_id']}_n{grid}.h5"
    validation = validate_raw_case(target, scene, solver_hash)
    with h5py.File(target, "r") as handle:
        if ("terminal_current" not in handle or
                handle["terminal_current"].shape != (len(scene["contacts"]),) or
                not np.isfinite(handle["terminal_current"][:]).all()):
            raise RuntimeError(f"invalid terminal-current vector: {target}")
    index = _load_accepted_ledger(root) if ledger_index is None else ledger_index
    receipt = index.get((scene["state_id"], grid))
    if receipt is None or receipt.get("raw_path") != str(target):
        raise RuntimeError(f"accepted source absent from solver ledger: {target}")
    if (receipt.get("raw_sha256") != validation["raw_sha256"] or
            receipt.get("input_sha256") != sha256_json(scene) or
            receipt.get("solver_sha256") != solver_hash):
        raise RuntimeError(f"accepted source differs from solver ledger: {target}")
    from .validation import audit_persisted
    physical = audit_persisted(Path(root), scene, grid)
    if not physical["accepted"]:
        raise RuntimeError(f"accepted source fails post-storage physical audit: {target}")
    return {"raw_sha256": validation["raw_sha256"], "raw_bytes": validation["raw_bytes"],
            "physical_audit": physical}


def _append_journal(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _check_gpu2_sharing(approval: dict) -> dict:
    status = gpu2_status()
    foreign = [line for line in status["compute_processes"]
               if line.split(",", 1)[0].strip() != str(os.getpid())]
    sharing = approval.get("gpu_sharing", {"mode": "exclusive"})
    if sharing["mode"] == "exclusive" and foreign:
        raise RuntimeError(f"foreign GPU2 job appeared: {foreign}")
    if sharing["mode"] == "concurrent" and status["free_mib"] < sharing["minimum_free_mib"]:
        raise RuntimeError(f"GPU2 free memory below approved concurrent floor: {status}")
    return status


def bind_gpu2_visibility() -> None:
    """Pin CUDA before torch import so cuda:0 cannot refer to another GPU."""
    if "torch" in sys.modules:
        raise RuntimeError("formal GPU2 binding requires a fresh process before importing torch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, GPU2_UUID):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose physical GPU2 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU2_UUID


def check_materialization_budget(*, retained_used: int, next_retained_upper: int,
                                 next_temporary_upper: int, free_bytes: int,
                                 approval: dict) -> None:
    """Reject a batch or shard before any next payload is written."""
    if min(retained_used, next_retained_upper, next_temporary_upper, free_bytes) < 0:
        raise ValueError("negative materialization accounting")
    if next_temporary_upper > int(approval["temporary_budget_bytes"]):
        raise RuntimeError("formal temporary budget cannot hold next materialization")
    if retained_used + next_retained_upper > int(approval["retained_budget_bytes"]):
        raise RuntimeError("formal retained budget cannot hold next materialization")
    if free_bytes - next_retained_upper - next_temporary_upper < int(approval["minimum_free_bytes"]):
        raise RuntimeError("formal next materialization would breach filesystem headroom")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def require_remaining_budget(root: Path, approval: dict) -> dict:
    """Reserve only bytes not yet materialized in this exact release root."""
    minimum = int(approval["minimum_free_bytes"])
    capacity = require_data2(minimum_free_bytes=minimum)
    already = _tree_bytes(root)
    retained_limit = int(approval["retained_budget_bytes"])
    temporary_limit = int(approval["temporary_budget_bytes"])
    if already > retained_limit:
        raise RuntimeError("formal release already exceeds retained-byte approval")
    remaining = retained_limit - already
    if capacity["free_bytes"] - remaining - temporary_limit < minimum:
        raise RuntimeError("formal remaining budget would breach filesystem headroom")
    return {"free_bytes": capacity["free_bytes"], "retained_used_bytes": already,
            "retained_remaining_bytes": remaining, "temporary_budget_bytes": temporary_limit}


def _inherit_source_case(root: Path, scene: dict, grid: int, context: dict, spec: dict) -> dict:
    """Hardlink one audited immutable case, retaining its original H5 attributes."""
    source_row = context["rows"][scene["formal_state_id"]]
    source_scene = _safe_scene(context["plan"], source_row)
    if sha256_json(scene) != sha256_json(source_scene):
        raise RuntimeError("nested case differs from immutable inherited source")
    solver_hash = _solver_hash(context["approval"], grid)
    original = context["ledger"].get((scene["state_id"], grid), {})
    if original.get("approval_hash") != spec["approval_hash"]:
        raise RuntimeError("source ledger differs from approved inheritance provenance")
    audit = validate_release_source(context["root"], source_scene, grid, context["approval"],
                                    ledger_index=context["ledger"])
    source_path = context["root"] / "raw" / f"{scene['state_id']}_n{grid}.h5"
    target = Path(root) / "raw" / source_path.name
    if source_path.is_symlink():
        raise RuntimeError("inherited source case must be a regular immutable file")
    if target.exists():
        if sha256_file(target) != audit["raw_sha256"]:
            raise RuntimeError("existing inherited case differs from approved source")
    else:
        os.link(source_path, target)
    return {**original, "raw_path": str(target), "formal_state_id": scene["formal_state_id"],
            "status": "accepted_inherited", "inheritance": {
                **spec, "source_raw_path": str(source_path), "source_raw_sha256": audit["raw_sha256"],
                "source_solver_sha256": solver_hash, "method": "immutable_hardlink"}}


def run_formal(plan: dict, approval: dict, *, root: Path, yes: bool = False,
               device: str = "cuda:0", resume: bool = False) -> dict:
    """Generate approved formal rows, stopping safely at a batch boundary.

    Completed H5 cases are reused only after their frozen input/solver hashes
    and stored field shapes validate. One writer owns the root. A STOP file in
    `receipts/` requests a clean boundary; no unrelated process is signalled.
    """
    binding = validate_approval(plan, approval, root=root, yes=yes, device=device)
    backend = binding["backend"]
    inheritance_context = _load_inheritance(plan, approval)
    if backend == GPU_BACKEND:
        bind_gpu2_visibility()
    sharing = approval.get("gpu_sharing", {"mode": "exclusive"})
    if backend == CPU_BACKEND:
        if device != "cpu":
            raise ValueError("CPU formal backend requires device=cpu")
    elif sharing["mode"] == "exclusive":
        require_exclusive_gpu2()
    else:
        _check_gpu2_sharing(approval)
        import torch
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(sharing["process_memory_fraction"], 0)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("physical GPU2 must be the sole visible CUDA device")
    owned = Path(binding["disk_root"])
    require_remaining_budget(owned, approval)
    stable_budget = {"retained_budget_bytes": int(approval["retained_budget_bytes"]),
                     "temporary_budget_bytes": int(approval["temporary_budget_bytes"]),
                     "minimum_free_bytes": int(approval["minimum_free_bytes"]),
                     "peak_new_bytes": int(approval["retained_budget_bytes"]) + int(approval["temporary_budget_bytes"])}
    initialize_root(owned, stable_budget)
    lock_path = owned / "receipts" / "formal_writer.lock"
    started = time.monotonic()
    new, reused, inherited = 0, 0, 0
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another formal writer owns this root") from exc
        plan_path = owned / "inputs" / "formal_plan.json"
        approval_path = owned / "inputs" / "formal_approval.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("existing formal plan differs from approved input")
        if approval_path.exists() and json.loads(approval_path.read_text()) != approval:
            raise RuntimeError("existing formal approval differs from this invocation")
        if not plan_path.exists():
            write_json_atomic(plan_path, plan)
        if not approval_path.exists():
            write_json_atomic(approval_path, approval)
        if backend == CPU_BACKEND:
            recipes_path = owned / "inputs" / "numerical_recipes.json"
            recipes = {str(grid): _solver_recipe(approval, grid) for grid in set(approval["resolution_groups"].values())}
            if recipes_path.exists() and json.loads(recipes_path.read_text()) != recipes:
                raise RuntimeError("CPU solver recipe changed; this release cannot silently resume")
            if not recipes_path.exists():
                write_json_atomic(recipes_path, recipes)
        ledger_index = _load_accepted_ledger(owned)
        rows = plan["states"]
        index = 0
        while index < len(rows):
            if (owned / "receipts" / "STOP").exists():
                break
            if time.monotonic() - started >= int(approval["max_elapsed_seconds"]):
                break
            capacity = require_data2(minimum_free_bytes=int(approval["minimum_free_bytes"]))
            retained_bytes = _tree_bytes(owned)
            if retained_bytes > int(approval["retained_budget_bytes"]):
                raise RuntimeError("formal retained-byte budget exceeded")
            batch_grid = _grid(rows[index], approval)
            batch_rows = []
            while index < len(rows) and len(batch_rows) < int(approval["batch_size"]) and _grid(rows[index], approval) == batch_grid:
                batch_rows.append(rows[index])
                index += 1
            solver_hash = _solver_hash(approval, batch_grid)
            pending = []
            for row in batch_rows:
                scene = _safe_scene(plan, row)
                target = owned / "raw" / f"{scene['state_id']}_n{batch_grid}.h5"
                can_inherit = inheritance_context is not None and row["state_id"] in inheritance_context["rows"]
                if can_inherit and (scene["state_id"], batch_grid) not in ledger_index:
                    if target.exists() and not resume:
                        raise RuntimeError("existing inherited state requires --resume")
                    source_path = inheritance_context["root"] / "raw" / target.name
                    inheritance_upper = source_path.stat().st_size + 4096
                    check_materialization_budget(retained_used=retained_bytes,
                                                 next_retained_upper=inheritance_upper,
                                                 next_temporary_upper=0,
                                                 free_bytes=capacity["free_bytes"], approval=approval)
                    ledger = _inherit_source_case(owned, scene, batch_grid, inheritance_context, approval["inheritance"])
                    _append_journal(owned / "solver_ledger.jsonl", ledger)
                    ledger_index[(scene["state_id"], batch_grid)] = ledger
                    retained_bytes += inheritance_upper
                    inherited += 1
                    continue
                if target.exists():
                    if not resume:
                        raise RuntimeError("existing formal state requires --resume")
                    if ((scene["state_id"], batch_grid) not in ledger_index or
                            ledger_index[(scene["state_id"], batch_grid)].get("approval_hash") is None):
                        # A cleanly written and physically accepted raw file
                        # can precede its append-only ledger row if interrupted.
                        # Reconstruct only that missing row after independent
                        # hash, shape, and post-storage physical verification.
                        recovered = validate_raw_case(target, scene, solver_hash)
                        from .validation import audit_persisted
                        if not audit_persisted(owned, scene, batch_grid)["accepted"]:
                            raise RuntimeError(f"unledgered formal source fails physical audit: {target}")
                        ledger = {"state_id": scene["state_id"], "formal_state_id": row["state_id"],
                                  "grid": batch_grid, "raw_path": str(target),
                                  "raw_sha256": recovered["raw_sha256"],
                                  "input_sha256": sha256_json(scene), "solver_sha256": solver_hash,
                                  "approval_hash": binding["approval_hash"],
                                  "status": "accepted_recovered_after_interruption"}
                        _append_journal(owned / "solver_ledger.jsonl", ledger)
                        _append_journal(owned / "diagnostics" / "formal_recovery.jsonl", ledger)
                        ledger_index[(scene["state_id"], batch_grid)] = ledger
                    validate_release_source(owned, scene, batch_grid, approval, ledger_index=ledger_index,
                                            inheritance_context=inheritance_context)
                    reused += 1
                else:
                    pending.append((row, scene))
            if not pending:
                continue
            # Bounds eight FP64 2D raw fields, staggered faces, and HDF5
            # metadata even if the current writer qualifies FP32 storage.
            estimated_case_bytes = 80 * batch_grid * batch_grid + 256 * 2**10
            check_materialization_budget(retained_used=retained_bytes,
                                         next_retained_upper=len(pending) * estimated_case_bytes,
                                         next_temporary_upper=estimated_case_bytes,
                                         free_bytes=capacity["free_bytes"], approval=approval)
            from .validation import audit_persisted, validate_solution
            gpu_before_batch = None
            if backend == CPU_BACKEND:
                from .physics import solve_cpu
                solutions = [solve_cpu(scene, batch_grid) for _, scene in pending]
            else:
                gpu_before_batch = _check_gpu2_sharing(approval)
                from .gpu_physics import solve_gpu_batch
                solutions = solve_gpu_batch([scene for _, scene in pending], batch_grid,
                                            precision=approval["precision"], device=device)
            for (row, scene), solution in zip(pending, solutions):
                quality = validate_solution(scene, solution,
                                            residual_limit=2e-10 if approval["precision"] == "fp64" else 2e-6)
                record = {"formal_state_id": row["state_id"], "raw_state_key": scene["state_id"],
                          "physical_input_hash": row["input_hash"], "grid": batch_grid,
                          "plan_hash": plan["plan_hash"], "approval_hash": binding["approval_hash"],
                          "backend": backend,
                          "gpu2_free_mib_before_batch": None if gpu_before_batch is None else gpu_before_batch["free_mib"],
                          "gpu2_foreign_processes_before_batch": [] if gpu_before_batch is None else [line for line in gpu_before_batch["compute_processes"]
                                                                   if line.split(",", 1)[0].strip() != str(os.getpid())],
                          "quality": quality, "timestamp_unix": time.time()}
                if not quality["accepted"]:
                    record["status"] = "failed_qualification"
                    _append_journal(owned / "diagnostics" / "formal_attempts.jsonl", record)
                    raise RuntimeError(f"formal quality failure: {row['state_id']}")
                receipt = write_case_atomic(owned, scene, solution, grid=batch_grid, solver_hash=solver_hash)
                storage = audit_persisted(owned, scene, batch_grid)
                record.update({"raw_sha256": receipt["raw_sha256"], "raw_bytes": receipt["raw_bytes"],
                               "post_storage": storage, "status": "accepted" if storage["accepted"] else "failed_post_storage"})
                _append_journal(owned / "diagnostics" / "formal_attempts.jsonl", record)
                if not storage["accepted"]:
                    raise RuntimeError(f"formal storage audit failed: {row['state_id']}")
                ledger = {"state_id": scene["state_id"], "formal_state_id": row["state_id"],
                          "grid": batch_grid, "raw_path": receipt["raw_path"],
                          "raw_sha256": receipt["raw_sha256"],
                          "input_sha256": sha256_json(scene), "solver_sha256": solver_hash,
                          "approval_hash": binding["approval_hash"], "status": "accepted"}
                _append_journal(owned / "solver_ledger.jsonl", ledger)
                ledger_index[(scene["state_id"], batch_grid)] = ledger
                new += 1
            print(f"formal: {index}/{len(rows)} planned, new={new} reused={reused} inherited={inherited}", flush=True)
    result = {"plan_hash": plan["plan_hash"], "approval_hash": binding["approval_hash"],
              "total_states": len(plan["states"]), "new": new, "reused": reused, "inherited": inherited,
              "backend": backend, "physical_solves": new,
              "inherited_sources_total": sum("inheritance" in record for record in ledger_index.values()),
              "complete": new + reused + inherited == len(plan["states"]), "elapsed_seconds": time.monotonic() - started}
    write_json_atomic(owned / "diagnostics" / "formal_progress.json", result)
    return result


RAW_FIELDS = ("potential", "temperature_rise", "q_sheet", "q_contact", "jx_face", "jy_face",
              "heat_flux_x_face", "heat_flux_y_face")
PROCESSED_FIELD_ORDER = ("potential", "jx_cell", "jy_cell", "temperature_rise", "q_total")


def _audit_packed_shard(path: Path, kind: str, members: list[tuple[dict, Path]], *,
                        grid: int, source_hashes: list[str], expected: str) -> None:
    """Compare saved shard metadata and every numerical payload to its sources."""
    from .processing import derive_fields
    state_ids = [scene.get("formal_state_id", scene["state_id"]) for scene, _ in members]
    with h5py.File(path, "r") as handle:
        if (handle.attrs.get("shard_sha256") != expected or not handle.attrs.get("accepted", False) or
                handle.attrs.get("grid_n") != grid or handle.attrs.get("kind") != kind or
                handle.attrs.get("units") != "dimensionless"):
            raise RuntimeError(f"conflicting or interrupted formal shard: {path}")
        for key, expected_values in (("state_id", state_ids), ("source_sha256", source_hashes)):
            if key not in handle or handle[key].shape != (len(members),) or handle[key].asstr()[:].tolist() != expected_values:
                raise RuntimeError(f"formal shard {key} mismatch: {path}")
        if kind == "processed":
            if (handle.attrs.get("field_order") != ",".join(PROCESSED_FIELD_ORDER) or
                    "fields" not in handle or handle["fields"].shape != (len(members), grid, grid, 5) or
                    handle["fields"].dtype != np.dtype("float32")):
                raise RuntimeError(f"formal processed field contract mismatch: {path}")
        else:
            if ("terminal_current" not in handle or handle["terminal_current"].shape != (len(members), 16) or
                    "contact_mask" not in handle or handle["contact_mask"].shape != (len(members), 16) or
                    handle["terminal_current"].dtype != np.dtype("float64") or
                    handle["contact_mask"].dtype != np.dtype("bool")):
                raise RuntimeError(f"formal raw terminal contract mismatch: {path}")
        for index, (_, source_path) in enumerate(members):
            with h5py.File(source_path, "r") as source:
                if kind == "processed":
                    if not np.array_equal(handle["fields"][index], derive_fields(source)):
                        raise RuntimeError(f"formal processed payload differs from source: {path}")
                else:
                    for key in RAW_FIELDS:
                        if (key not in handle or handle[key].shape != (len(members), *source[key].shape) or
                                handle[key].dtype != source[key].dtype or
                                not np.array_equal(handle[key][index], source[key][:])):
                            raise RuntimeError(f"formal raw {key} payload differs from source: {path}")
                    current = source["terminal_current"][:]
                    padded = np.zeros(16, dtype=np.float64)
                    padded[:len(current)] = current
                    mask = np.arange(16) < len(current)
                    if (not np.array_equal(handle["terminal_current"][index], padded) or
                            not np.array_equal(handle["contact_mask"][index], mask)):
                        raise RuntimeError(f"formal raw terminal payload differs from source: {path}")


def pack_verified_shard(root: Path, members: list[tuple[dict, Path]], *, grid: int,
                        shard_key: str, compression: str = "lzf") -> dict:
    """Atomically pack at most 64 verified per-case H5 files into raw/processed shards.

    The source files remain in place. `members` contain full physical scenes
    with safe raw keys, alongside their accepted case paths. This narrow
    function is also usable for a small pilot-sourced mechanical pack test.
    """
    if not 1 <= len(members) <= 64 or grid < 2:
        raise ValueError("shard needs 1..64 cases and a valid native grid")
    if not shard_key.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe shard key")
    if compression not in (None, "lzf", "gzip"):
        raise ValueError("unsupported lossless compression")
    root = Path(root)
    (root / "raw_shards").mkdir(parents=True, exist_ok=True)
    (root / "processed_shards").mkdir(parents=True, exist_ok=True)
    (root / "temporary").mkdir(parents=True, exist_ok=True)
    source_hashes, source_paths = [], []
    shapes = {"potential": (grid, grid), "temperature_rise": (grid, grid),
              "q_sheet": (grid, grid), "q_contact": (grid, grid),
              "jx_face": (grid, grid + 1), "jy_face": (grid + 1, grid),
              "heat_flux_x_face": (grid, grid + 1), "heat_flux_y_face": (grid + 1, grid)}
    for scene, path in members:
        path = Path(path)
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("accepted", False) or int(handle.attrs.get("grid_n", -1)) != grid:
                raise RuntimeError(f"unaccepted source or grid mismatch: {path}")
            if handle.attrs.get("input_sha256") != sha256_json(scene):
                raise RuntimeError(f"source input hash mismatch: {path}")
            for key in RAW_FIELDS:
                if key not in handle or handle[key].shape != shapes[key] or not np.isfinite(handle[key][:]).all():
                    raise RuntimeError(f"missing or nonfinite raw field {key}: {path}")
            if ("terminal_current" not in handle or
                    handle["terminal_current"].shape != (len(scene["contacts"]),) or
                    not np.isfinite(handle["terminal_current"][:]).all()):
                raise RuntimeError(f"invalid terminal-current vector: {path}")
        source_hashes.append(sha256_file(path))
        source_paths.append(path)
    expected = sha256_json({"grid": grid, "state_ids": [s["state_id"] for s, _ in members],
                            "source_sha256": source_hashes, "field_order": PROCESSED_FIELD_ORDER})
    targets = {"raw": root / "raw_shards" / f"{shard_key}.h5",
               "processed": root / "processed_shards" / f"{shard_key}.h5"}
    for kind, target in targets.items():
        if target.exists():
            _audit_packed_shard(target, kind, members, grid=grid, source_hashes=source_hashes, expected=expected)
    if all(target.exists() for target in targets.values()):
        return {"shard_key": shard_key, "states": len(members), "grid": grid, "shard_sha256": expected,
                "source_sha256": source_hashes, "raw_path": str(targets["raw"]),
                "processed_path": str(targets["processed"]), "reused": True,
                "raw_file_sha256": sha256_file(targets["raw"]),
                "processed_file_sha256": sha256_file(targets["processed"])}
    # A crash may leave one accepted half of the pair. Its hash was checked
    # above; rebuild both from the immutable source and replace atomically.
    tmp_raw = root / "temporary" / f"{shard_key}.raw.{os.getpid()}.tmp"
    tmp_processed = root / "temporary" / f"{shard_key}.processed.{os.getpid()}.tmp"
    strings = h5py.string_dtype("utf-8")
    try:
        with h5py.File(tmp_raw, "w") as raw, h5py.File(tmp_processed, "w") as processed:
            for handle, kind in ((raw, "raw"), (processed, "processed")):
                handle.attrs["shard_sha256"] = expected
                handle.attrs["accepted"] = False
                handle.attrs["grid_n"] = grid
                handle.attrs["units"] = "dimensionless"
                handle.attrs["kind"] = kind
                handle.create_dataset("state_id", data=np.asarray([s["formal_state_id"] if "formal_state_id" in s else s["state_id"]
                                                                    for s, _ in members], dtype=object), dtype=strings)
                handle.create_dataset("source_sha256", data=np.asarray(source_hashes, dtype=object), dtype=strings)
            raw_arrays = {key: [] for key in RAW_FIELDS}
            processed_arrays = []
            currents = np.zeros((len(members), 16), dtype=np.float64)
            mask = np.zeros((len(members), 16), dtype=np.bool_)
            for i, source in enumerate(source_paths):
                with h5py.File(source, "r") as handle:
                    arrays = {key: handle[key][:] for key in RAW_FIELDS}
                    for key, value in arrays.items():
                        raw_arrays[key].append(value)
                    current = handle["terminal_current"][:]
                    if len(current) > 16:
                        raise RuntimeError("formal shard contact padding supports at most 16")
                    currents[i, :len(current)] = current
                    mask[i, :len(current)] = True
                    jx = 0.5 * (arrays["jx_face"][:, :-1] + arrays["jx_face"][:, 1:])
                    jy = 0.5 * (arrays["jy_face"][:-1, :] + arrays["jy_face"][1:, :])
                    processed_arrays.append(np.stack((arrays["potential"], jx, jy,
                                                      arrays["temperature_rise"],
                                                      arrays["q_sheet"] + arrays["q_contact"]), axis=-1))
            for key, values in raw_arrays.items():
                raw.create_dataset(key, data=np.stack(values), chunks=(1, *values[0].shape), compression=compression)
            raw.create_dataset("terminal_current", data=currents, chunks=(1, 16), compression=compression)
            raw.create_dataset("contact_mask", data=mask, chunks=(1, 16), compression=compression)
            processed.attrs["field_order"] = ",".join(PROCESSED_FIELD_ORDER)
            processed.create_dataset("fields", data=np.stack(processed_arrays).astype(np.float32),
                                     chunks=(1, grid, grid, 5), compression=compression)
            raw.attrs["accepted"] = True
            processed.attrs["accepted"] = True
            raw.flush(); processed.flush()
        for path, kind in ((tmp_raw, "raw"), (tmp_processed, "processed")):
            _audit_packed_shard(path, kind, members, grid=grid, source_hashes=source_hashes, expected=expected)
        os.replace(tmp_raw, targets["raw"])
        os.replace(tmp_processed, targets["processed"])
    finally:
        tmp_raw.unlink(missing_ok=True)
        tmp_processed.unlink(missing_ok=True)
    return {"shard_key": shard_key, "states": len(members), "grid": grid, "shard_sha256": expected,
            "source_sha256": source_hashes, "raw_path": str(targets["raw"]),
            "processed_path": str(targets["processed"]), "reused": False,
            "raw_file_sha256": sha256_file(targets["raw"]),
            "processed_file_sha256": sha256_file(targets["processed"]),
            "raw_bytes": targets["raw"].stat().st_size, "processed_bytes": targets["processed"].stat().st_size}


def pack_formal_release(plan: dict, approval: dict, *, root: Path) -> dict:
    """Pack only a complete approved release; preserve all source cases."""
    binding = validate_approval(plan, approval, root=root, yes=True)
    owned = Path(binding["disk_root"])
    groups = defaultdict(list)
    ledger_index = _load_accepted_ledger(owned)
    inheritance_context = _load_inheritance(plan, approval)
    for row in plan["states"]:
        grid = _grid(row, approval)
        scene = _safe_scene(plan, row)
        source = owned / "raw" / f"{scene['state_id']}_n{grid}.h5"
        if not source.exists():
            raise RuntimeError(f"formal release incomplete: {row['state_id']}")
        validate_release_source(owned, scene, grid, approval, ledger_index=ledger_index,
                                inheritance_context=inheritance_context)
        groups[(row["split"], grid)].append((scene, source))
    results = []
    lock_path = owned / "receipts" / "formal_pack.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another formal packer owns this root") from exc
        for (split, grid), members in groups.items():
            for start in range(0, len(members), 64):
                shard_key = f"{split}_n{grid}_s{start // 64:04d}"
                chunk = members[start:start + 64]
                # Conservative uncompressed envelope for eight raw fields,
                # five processed channels, and HDF5 metadata. Atomic staging
                # can temporarily require this much space once more.
                upper = len(chunk) * (112 * grid * grid + 32 * grid + 256) + 4 * 2**20
                existing_raw = owned / "raw_shards" / f"{shard_key}.h5"
                existing_processed = owned / "processed_shards" / f"{shard_key}.h5"
                if not (existing_raw.exists() and existing_processed.exists()):
                    capacity = require_data2(minimum_free_bytes=int(approval["minimum_free_bytes"]))
                    check_materialization_budget(retained_used=_tree_bytes(owned),
                                                 next_retained_upper=upper,
                                                 next_temporary_upper=upper,
                                                 free_bytes=capacity["free_bytes"], approval=approval)
                results.append(pack_verified_shard(owned, chunk, grid=grid, shard_key=shard_key))
                _append_journal(owned / "diagnostics" / "formal_packing.jsonl", results[-1])
    receipt = {"plan_hash": plan["plan_hash"], "approval_hash": binding["approval_hash"],
               "states": len(plan["states"]), "shards": len(results),
               "raw_shard_bytes": sum(Path(r["raw_path"]).stat().st_size for r in results),
               "processed_shard_bytes": sum(Path(r["processed_path"]).stat().st_size for r in results),
               "source_cases_retained": True}
    write_json_atomic(owned / "diagnostics" / "formal_pack_receipt.json", receipt)
    return receipt
