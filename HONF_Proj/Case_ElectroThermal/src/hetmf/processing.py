"""Derived model targets; source raw fields retain conservation geometry."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from .io import sha256_file
from .resources import require_owned_root


FIELD_ORDER = ("potential", "jx_cell", "jy_cell", "temperature_rise", "q_total")


def derive_fields(raw: h5py.File | dict) -> np.ndarray:
    jx = (np.asarray(raw["jx_face"][:, :-1]) + np.asarray(raw["jx_face"][:, 1:])) * 0.5
    jy = (np.asarray(raw["jy_face"][:-1, :]) + np.asarray(raw["jy_face"][1:, :])) * 0.5
    q = np.asarray(raw["q_sheet"]) + np.asarray(raw["q_contact"])
    return np.stack((raw["potential"], jx, jy, raw["temperature_rise"], q), axis=-1).astype(np.float32)


def pack_case(root: Path, state_id: str, grid: int, *, compression: str = "lzf") -> dict:
    root = require_owned_root(root)
    raw_path = root / "raw" / f"{state_id}_n{grid}.h5"
    target = root / "processed" / f"{state_id}_n{grid}.h5"
    with h5py.File(raw_path) as raw:
        if not raw.attrs.get("accepted", False):
            raise RuntimeError("raw source is not accepted")
        fields = derive_fields(raw)
        input_hash = raw.attrs["input_sha256"]
        solver_hash = raw.attrs["solver_sha256"]
    if target.exists():
        with h5py.File(target) as handle:
            if (handle.attrs.get("input_sha256") != input_hash or
                    handle.attrs.get("solver_sha256") != solver_hash or
                    handle["fields"].shape != fields.shape):
                raise RuntimeError("processed shard conflicts with source")
        return {"path": str(target), "bytes": target.stat().st_size,
                "sha256": sha256_file(target), "reused": True}
    temporary = root / "temporary" / f"{state_id}_n{grid}.{os.getpid()}.processed.tmp"
    with h5py.File(temporary, "w") as handle:
        handle.attrs["input_sha256"] = input_hash
        handle.attrs["solver_sha256"] = solver_hash
        handle.attrs["field_order"] = ",".join(FIELD_ORDER)
        handle.attrs["units"] = "dimensionless"
        handle.create_dataset("fields", data=fields, chunks=True, compression=compression)
    os.replace(temporary, target)
    return {"path": str(target), "bytes": target.stat().st_size,
            "sha256": sha256_file(target), "reused": False}
