"""Hash-bound, atomic pilot case storage."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from .resources import require_owned_root


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_root(root: Path, budget: dict) -> Path:
    root = require_owned_root(root)
    if not budget or budget.get("peak_new_bytes", 0) <= 0:
        raise ValueError("explicit generated-byte budget required")
    for name in ("inputs", "raw", "processed", "diagnostics", "figures", "logs", "temporary", "receipts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    path = root / "receipts" / "storage_reservation.json"
    if path.exists() and json.loads(path.read_text()) != budget:
        raise RuntimeError("root already has a different storage reservation")
    path.write_text(json.dumps(budget, indent=2) + "\n")
    return root


def write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stored_arrays(solution: dict, field_dtype: str) -> dict[str, np.ndarray]:
    if field_dtype not in ("float32", "float64"):
        raise ValueError("unsupported raw field dtype")
    aliases = {
        "potential": ("potential", "v"),
        "temperature_rise": ("temperature_rise", "theta"),
        "q_sheet": ("q_sheet",), "q_contact": ("q_contact",),
        "jx_face": ("jx_face", "j_face_x"), "jy_face": ("jy_face", "j_face_y"),
        "heat_flux_x_face": ("heat_flux_x_face", "heat_x_face", "heat_flux_face_x"),
        "heat_flux_y_face": ("heat_flux_y_face", "heat_y_face", "heat_flux_face_y"),
        "terminal_current": ("terminal_current", "currents", "terminal_currents"),
    }
    out = {}
    for key, choices in aliases.items():
        found = next((solution[choice] for choice in choices if choice in solution), None)
        if found is None:
            raise KeyError(f"missing raw array {key}")
        out[key] = np.asarray(found, dtype=np.float64 if key == "terminal_current" else field_dtype)
    return out


def write_case_atomic(root: Path, scene: dict, solution: dict, *, grid: int,
                      solver_hash: str, compression: str = "lzf",
                      field_dtype: str = "float64") -> dict:
    root = require_owned_root(root)
    state_id = str(scene["state_id"])
    if not state_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe state ID")
    input_hash = sha256_json(scene)
    target = root / "raw" / f"{state_id}_n{grid}.h5"
    if target.exists():
        with h5py.File(target) as handle:
            if handle.attrs.get("input_sha256") != input_hash or handle.attrs.get("solver_sha256") != solver_hash:
                raise RuntimeError(f"conflicting existing case: {target}")
            if not handle.attrs.get("accepted", False):
                raise RuntimeError(f"incomplete existing case: {target}")
        return {"state_id": state_id, "grid": grid, "input_sha256": input_hash,
                "solver_sha256": solver_hash, "raw_path": str(target),
                "raw_sha256": sha256_file(target), "reused": True}
    arrays = _stored_arrays(solution, field_dtype)
    temporary = root / "temporary" / f"{state_id}_n{grid}.{os.getpid()}.h5.tmp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["state_id"] = state_id
        handle.attrs["input_sha256"] = input_hash
        handle.attrs["solver_sha256"] = solver_hash
        handle.attrs["grid_n"] = grid
        handle.attrs["accepted"] = True
        handle.attrs["units"] = "dimensionless"
        handle.attrs["field_dtype"] = field_dtype
        for key, value in arrays.items():
            handle.create_dataset(key, data=value, compression=compression,
                                  chunks=True if value.ndim >= 2 else None)
    with h5py.File(temporary) as handle:
        if any(key not in handle for key in arrays) or not handle.attrs["accepted"]:
            raise RuntimeError("temporary case validation failed")
    os.replace(temporary, target)
    receipt = {"state_id": state_id, "grid": grid, "input_sha256": input_hash,
               "solver_sha256": solver_hash, "raw_path": str(target),
               "raw_sha256": sha256_file(target), "raw_bytes": target.stat().st_size,
               "timestamp_unix": time.time(), "reused": False}
    with (root / "solver_ledger.jsonl").open("a") as handle:
        handle.write(canonical_json(receipt) + "\n")
    return receipt


def validate_raw_case(path: Path, scene: dict, solver_hash: str) -> dict:
    expected = sha256_json(scene)
    with h5py.File(path) as handle:
        if (handle.attrs.get("input_sha256") != expected or
                handle.attrs.get("solver_sha256") != solver_hash or
                not handle.attrs.get("accepted", False)):
            raise RuntimeError(f"raw case hash/accepted mismatch: {path}")
        n = int(handle.attrs["grid_n"])
        shapes = {"potential": (n, n), "temperature_rise": (n, n),
                  "q_sheet": (n, n), "q_contact": (n, n),
                  "jx_face": (n, n + 1), "jy_face": (n + 1, n),
                  "heat_flux_x_face": (n, n + 1), "heat_flux_y_face": (n + 1, n)}
        for key, shape in shapes.items():
            if key not in handle or handle[key].shape != shape or not np.isfinite(handle[key][:]).all():
                raise RuntimeError(f"invalid raw field {key}: {path}")
        if ("terminal_current" not in handle or
                handle["terminal_current"].shape != (len(scene["contacts"]),) or
                not np.isfinite(handle["terminal_current"][:]).all()):
            raise RuntimeError(f"invalid terminal current: {path}")
        return {"state_id": handle.attrs["state_id"], "grid": n,
                "raw_bytes": Path(path).stat().st_size, "raw_sha256": sha256_file(path)}


def validate_ledger_case(root: Path, path: Path) -> str:
    """Require the accepted case bytes to match their append-only source receipt."""
    ledger = Path(root) / "solver_ledger.jsonl"
    if not ledger.exists():
        raise RuntimeError("missing source case ledger")
    matches = [json.loads(line) for line in ledger.read_text().splitlines()
               if json.loads(line).get("raw_path") == str(path)]
    if not matches:
        raise RuntimeError(f"missing source checksum receipt: {path}")
    expected = matches[-1]["raw_sha256"]
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"source checksum mismatch: {path}")
    return observed
