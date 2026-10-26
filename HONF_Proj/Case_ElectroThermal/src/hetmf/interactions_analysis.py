"""Measured physical endpoint and interaction diagnostics."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .io import write_json_atomic


def _load(root: Path, state_id: str, grid: int) -> dict:
    with h5py.File(root / "raw" / f"{state_id}_n{grid}.h5") as handle:
        return {key: handle[key][:].astype(np.float64) for key in
                ("potential", "temperature_rise", "terminal_current", "q_sheet", "q_contact")}


def _restrict(field: np.ndarray, ratio: int) -> np.ndarray:
    if field.ndim == 1:
        return field
    n = field.shape[0] // ratio
    return field.reshape(n, ratio, n, ratio).mean(axis=(1, 3))


def _norm(field: np.ndarray) -> float:
    return float(np.linalg.norm(field))


def analyze_quartet(root: Path, quartet: dict, *, grid: int = 96) -> dict:
    ids = quartet["endpoint_state_ids"]
    endpoints = [_load(root, sid, grid) for sid in ids]
    result = {"quartet_id": quartet["quartet_id"], "kind": quartet["kind"],
              "grid": grid, "endpoint_ids": ids,
              "endpoint_input_hashes": quartet["endpoint_input_hashes"],
              "interventions": quartet["interventions"]}
    for key in ("potential", "temperature_rise", "terminal_current"):
        a, b, c, d = [endpoint[key] for endpoint in endpoints]
        interaction = d - b - c + a
        scale = max(_norm(a), _norm(b), _norm(c), _norm(d), 1e-12)
        result[key] = {
            "A_change_l2": _norm(b - a), "B_change_l2": _norm(c - a),
            "joint_change_l2": _norm(d - a),
            "interaction_l2": _norm(interaction),
            "interaction_fraction_of_endpoint_scale": _norm(interaction) / scale,
            "interaction_max_absolute": float(np.max(abs(interaction))),
        }
    powers = [float((e["q_sheet"] + e["q_contact"]).mean()) for e in endpoints]
    peaks = [float(e["temperature_rise"].max()) for e in endpoints]
    result["power_endpoints"] = powers
    result["power_interaction"] = powers[3] - powers[1] - powers[2] + powers[0]
    result["temperature_peak_endpoints"] = peaks
    result["temperature_peak_interaction"] = peaks[3] - peaks[1] - peaks[2] + peaks[0]
    if all((root / "raw" / f"{sid}_n192.h5").exists() for sid in ids):
        fine = [_load(root, sid, 192) for sid in ids]
        floor = {}
        for key in ("potential", "temperature_rise", "terminal_current"):
            coarse_i = endpoints[3][key] - endpoints[1][key] - endpoints[2][key] + endpoints[0][key]
            fine_i = fine[3][key] - fine[1][key] - fine[2][key] + fine[0][key]
            discrepancy = _norm(coarse_i - _restrict(fine_i, 2))
            floor[key] = {"grid_interaction_difference_l2": discrepancy,
                          "interaction_over_grid_difference": _norm(coarse_i) / discrepancy if discrepancy > 0 else None}
        result["resolution_floor_96_vs_192"] = floor
    return result


def analyze_all(root: Path, plan: dict) -> dict:
    root = Path(root)
    rows = [analyze_quartet(root, q) for q in plan["quartets"]]
    output = {"quartets": rows}
    write_json_atomic(root / "diagnostics" / "counterfactuals.json", output)
    return output
