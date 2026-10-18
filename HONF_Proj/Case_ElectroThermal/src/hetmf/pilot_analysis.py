"""Compact evidence tables from saved pilot inputs and accepted attempts."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import h5py
import numpy as np

from .interactions_analysis import analyze_all
from .io import write_json_atomic
from .physics import terminal_conductance_matrix
from .qualification import refinement_board


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)), "max": float(array.max()),
            "mean": float(array.mean())}


def analyze_pilot(root: Path, plan: dict) -> dict:
    root = Path(root)
    attempts = [json.loads(line) for line in (root / "diagnostics" / "attempts.jsonl").read_text().splitlines()]
    accepted = [row for row in attempts if row["status"] == "accepted"]
    panel = [row for row in accepted if row["stage"] == "throughput"]
    if len(panel) != 256:
        raise RuntimeError("complete 256-state panel required")
    scenes = {s["state_id"]: s for s in plan["scenes"]}
    by_family = defaultdict(list)
    by_m = defaultdict(list)
    contact_power, theta_spatial_cv, temp_peak, current_magnitude = [], [], [], []
    realized_s_contrast, realized_k_contrast, realized_beta_contrast = [], [], []
    for row in panel:
        scene = scenes[row["state_id"]]
        family, m = scene["environment"]["family"], len(scene["contacts"])
        by_family[family].append(row["complete_seconds_amortized"])
        by_m[str(m)].append(row["complete_seconds_amortized"])
        diag = row["quality"]["diagnostics"]
        contact_power.append(diag["contact_power_fraction"])
        env = scene["environment"]
        for key, dest in (("s", realized_s_contrast), ("kappa", realized_k_contrast),
                          ("beta", realized_beta_contrast)):
            vals = np.asarray(env[key])
            dest.append(float(vals.max() / vals.min()))
        with h5py.File(root / "raw" / f"{scene['state_id']}_n96.h5") as handle:
            theta = handle["temperature_rise"][:]
            currents = handle["terminal_current"][:]
            theta_spatial_cv.append(float(theta.std() / max(abs(theta.mean()), 1e-12)))
            temp_peak.append(float(theta.max()))
            current_magnitude.append(float(np.abs(currents).max()))
    basis = {}
    for sid in ("R00", "R02", "R05", "R10"):
        response = terminal_conductance_matrix(scenes[sid], 96)
        basis[sid] = {**response, "matrix": response["matrix"].tolist()}
    write_json_atomic(root / "diagnostics" / "terminal_response_basis.json", basis)
    interactions = analyze_all(root, plan)
    refinement = refinement_board(root, plan["representative_ids"])
    max_residuals = {key: max(row["post_storage"][key] for row in accepted)
                     for key in ("electrical_residual_relative", "thermal_residual_relative",
                                 "current_balance_relative", "electrical_power_relative",
                                 "thermal_power_relative")}
    data = {"physical_states": len(plan["scenes"]), "accepted_raw_files": len(accepted),
            "failed_attempts": [row for row in attempts if row["status"] != "accepted"],
            "attempt_status_counts": dict(Counter(row["status"] for row in attempts)),
            "throughput_panel": {"states": len(panel), "family_counts": dict(Counter(scenes[r["state_id"]]["environment"]["family"] for r in panel)),
                                 "contact_count_counts": dict(Counter(str(len(scenes[r["state_id"]]["contacts"])) for r in panel)),
                                 "complete_seconds": _summary([r["complete_seconds_amortized"] for r in panel]),
                                 "by_family_seconds": {k: _summary(v) for k, v in by_family.items()},
                                 "by_contact_count_seconds": {k: _summary(v) for k, v in by_m.items()},
                                 "contact_power_fraction": _summary(contact_power),
                                 "temperature_spatial_cv": _summary(theta_spatial_cv),
                                 "temperature_peak": _summary(temp_peak),
                                 "max_absolute_terminal_current": _summary(current_magnitude),
                                 "realized_s_contrast": _summary(realized_s_contrast),
                                 "realized_kappa_contrast": _summary(realized_k_contrast),
                                 "realized_beta_contrast": _summary(realized_beta_contrast)},
            "max_post_storage_residuals": max_residuals,
            "basis_matrix_summary": {sid: {k: v for k, v in result.items() if k != "matrix"}
                                     for sid, result in basis.items()},
            "interaction_summary": {row["quartet_id"]: {"kind": row["kind"],
                                    "potential_interaction_fraction": row["potential"]["interaction_fraction_of_endpoint_scale"],
                                    "temperature_interaction_fraction": row["temperature_rise"]["interaction_fraction_of_endpoint_scale"],
                                    "resolution_floor": row.get("resolution_floor_96_vs_192")}
                                    for row in interactions["quartets"]},
            "refinement_max_by_pair": refinement["max_by_pair"]}
    write_json_atomic(root / "diagnostics" / "pilot_summary.json", data)
    return data
