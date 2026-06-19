"""CHANNELTHERMAL-SPECIFIC canonical hypergraph plan export.

Inputs are the legacy-compatible `organizer_aux` dictionary returned by the
ChannelThermal HONF wrapper plus the module-present mask for a case. Outputs
are small NumPy arrays describing the static organization: module/environment
assignments, source/region coordinates, masses, strengths, active masks, and
schema metadata.

The plan intentionally excludes dense learned `hyper_state`, query-dependent
`alpha_qk`, and raw module tokens. `alpha_qk` is recomputed by the forward
decoder for each query grid, and raw module tokens are recomputed from any
generated physical design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


SCHEMA_VERSION = 2
EPS = 1.0e-5


def _to_numpy(value: Any, *, detach: bool) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach() if detach else value
        return tensor.cpu().numpy()
    return np.asarray(value)


def _squeeze_batch(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _edge_permutation(plan: Dict[str, np.ndarray]) -> np.ndarray:
    active = np.asarray(plan["active_hyperedge_mask"], dtype=np.float32).reshape(-1) > 0.5
    src = np.asarray(plan["hyper_source_coords"], dtype=np.float64)
    dst = np.asarray(plan["hyper_region_coords"], dtype=np.float64)
    strength = np.asarray(plan["hyper_strength"], dtype=np.float64).reshape(-1)
    order = sorted(
        range(strength.shape[0]),
        key=lambda idx: (
            0 if active[idx] else 1,
            float(src[idx, 0]),
            float(src[idx, 1]),
            float(dst[idx, 0]),
            float(dst[idx, 1]),
            -float(strength[idx]),
        ),
    )
    return np.asarray(order, dtype=np.int64)


def _apply_edge_permutation(plan: Dict[str, np.ndarray], perm: np.ndarray) -> Dict[str, np.ndarray]:
    out = dict(plan)
    for key in ("A_mh", "A_eh"):
        if key in out:
            out[key] = np.asarray(out[key])[:, perm].astype(np.float32, copy=False)
    for key in (
        "hyper_source_coords",
        "hyper_region_coords",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_strength",
        "active_hyperedge_mask",
    ):
        if key in out:
            out[key] = np.asarray(out[key])[perm].astype(np.float32, copy=False)
    out["edge_permutation"] = perm.astype(np.int64)
    return out


def extract_hypergraph_plan(
    organizer_aux: Dict[str, Any],
    module_present: Any,
    *,
    detach: bool = True,
    domain_length_x: float | None = None,
    domain_length_y: float | None = None,
) -> Dict[str, np.ndarray]:
    """Extract a canonical static organizer plan suitable for inverse design."""

    plan: Dict[str, np.ndarray] = {}
    for key in (
        "A_mh",
        "A_eh",
        "hyper_source_coords",
        "hyper_region_coords",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_strength",
        "active_hyperedge_mask",
        "env_coords",
    ):
        if key in organizer_aux:
            plan[key] = _squeeze_batch(_to_numpy(organizer_aux[key], detach=detach)).astype(np.float32, copy=False)

    if "hyper_region_coords" not in plan and "hyper_thermal_region_coords" in organizer_aux:
        plan["hyper_region_coords"] = _squeeze_batch(
            _to_numpy(organizer_aux["hyper_thermal_region_coords"], detach=detach)
        ).astype(np.float32, copy=False)

    present = _squeeze_batch(_to_numpy(module_present, detach=detach)).astype(np.float32, copy=False)
    plan["module_present"] = present
    if "A_mh" in plan and plan["A_mh"].ndim == 2:
        # Module slots are stable design slots. Inactive slots remain present in
        # A_mh with zero mass so inverse code can preserve slot indexing.
        plan["A_mh"] = plan["A_mh"] * present[:, None]

    num_h = int(plan.get("hyper_strength", np.zeros((0,), dtype=np.float32)).shape[0])
    if "active_hyperedge_mask" not in plan:
        plan["active_hyperedge_mask"] = (plan["hyper_strength"] > 0.05).astype(np.float32)
    else:
        plan["active_hyperedge_mask"] = (plan["active_hyperedge_mask"] > 0.5).astype(np.float32)
    plan["schema_version"] = np.asarray(SCHEMA_VERSION, dtype=np.int32)
    plan["domain_length_x"] = np.asarray(0.0 if domain_length_x is None else float(domain_length_x), dtype=np.float32)
    plan["domain_length_y"] = np.asarray(0.0 if domain_length_y is None else float(domain_length_y), dtype=np.float32)
    plan["num_hyperedges"] = np.asarray(num_h, dtype=np.int32)
    plan["num_env_tokens"] = np.asarray(int(plan.get("A_eh", np.zeros((0, 0))).shape[0]), dtype=np.int32)
    plan["module_count"] = np.asarray(int(present.shape[0]), dtype=np.int32)
    if "env_coords" not in plan:
        plan["env_coords"] = np.zeros((int(plan["num_env_tokens"]), 2), dtype=np.float32)
    perm = _edge_permutation(plan)
    return _apply_edge_permutation(plan, perm)


def save_hypergraph_plan(path: str | Path, plan: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **plan)


def load_hypergraph_plan(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def validate_hypergraph_plan(plan: Dict[str, np.ndarray]) -> None:
    required = {
        "schema_version",
        "A_mh",
        "A_eh",
        "hyper_source_coords",
        "hyper_region_coords",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_strength",
        "active_hyperedge_mask",
        "module_present",
        "env_coords",
        "edge_permutation",
    }
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError(f"Hypergraph plan missing keys: {missing}")
    for key, value in plan.items():
        arr = np.asarray(value)
        if arr.dtype.kind in {"f", "i", "u"} and not np.isfinite(arr).all():
            raise ValueError(f"Hypergraph plan key {key!r} contains non-finite values.")
    A_mh = np.asarray(plan["A_mh"], dtype=np.float64)
    A_eh = np.asarray(plan["A_eh"], dtype=np.float64)
    module_present = np.asarray(plan["module_present"], dtype=np.float64).reshape(-1)
    num_h = int(np.asarray(plan["num_hyperedges"]))
    if A_mh.shape != (module_present.shape[0], num_h):
        raise ValueError(f"A_mh shape {A_mh.shape} does not match module_count/num_hyperedges.")
    if A_eh.ndim != 2 or A_eh.shape[1] != num_h:
        raise ValueError(f"A_eh shape {A_eh.shape} does not match num_hyperedges={num_h}.")
    present_rows = module_present > 0.5
    if present_rows.any() and not np.allclose(A_mh[present_rows].sum(axis=1), 1.0, atol=2.0e-4):
        raise ValueError("A_mh present rows are not normalized.")
    if (~present_rows).any() and not np.allclose(A_mh[~present_rows].sum(axis=1), 0.0, atol=2.0e-4):
        raise ValueError("A_mh inactive module rows should have zero mass.")
    if A_eh.shape[0] and not np.allclose(A_eh.sum(axis=1), 1.0, atol=2.0e-4):
        raise ValueError("A_eh rows are not normalized.")
    for key in ("hyper_module_mass", "hyper_env_mass"):
        mass = np.asarray(plan[key], dtype=np.float64).reshape(-1)
        if mass.shape[0] != num_h:
            raise ValueError(f"{key} length does not match num_hyperedges.")
        if not np.isclose(mass.sum(), 1.0, atol=2.0e-4):
            raise ValueError(f"{key} does not sum to one.")
    active = np.asarray(plan["active_hyperedge_mask"]).reshape(-1)
    if active.shape[0] != num_h or not np.isin(active, [0, 1]).all():
        raise ValueError("active_hyperedge_mask must be binary and edge-indexed.")
    env_coords = np.asarray(plan["env_coords"])
    if env_coords.shape != (A_eh.shape[0], 2):
        raise ValueError("env_coords shape must align with A_eh rows.")
    perm = np.asarray(plan["edge_permutation"]).reshape(-1)
    if sorted(perm.tolist()) != list(range(num_h)):
        raise ValueError("edge_permutation must be a permutation of original edge indices.")
    identity_order = _edge_permutation({key: np.asarray(value) for key, value in plan.items()})
    if not np.array_equal(identity_order, np.arange(num_h)):
        raise ValueError("Hyperedges are not in canonical order.")


def summarize_hypergraph_plan(plan: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "schema_version": int(np.asarray(plan["schema_version"])),
        "keys": sorted(plan.keys()),
        "shapes": {key: list(np.asarray(value).shape) for key, value in plan.items()},
        "num_hyperedges": int(np.asarray(plan["num_hyperedges"])),
        "num_env_tokens": int(np.asarray(plan["num_env_tokens"])),
        "module_count": int(np.asarray(plan["module_count"])),
        "active_hyperedge_count": int(np.asarray(plan["active_hyperedge_mask"]).sum()),
        "edge_permutation": np.asarray(plan["edge_permutation"], dtype=np.int64).tolist(),
        "note": (
            "Static canonical organizer plan. Query-dependent alpha_qk is "
            "recomputed by the HONF decoder; module tokens are recomputed from "
            "generated physical designs. A_mh preserves module slot indexing."
        ),
    }

