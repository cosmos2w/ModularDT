"""Numerical acceptance and post-storage conservation checks."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .contacts import rasterize_contacts
from .environment import render_environment
from .finite_volume import (apply_operator, face_energy_heat, prepare_faces,
                            staggered_face_density)


def validate_solution(scene: dict, solution: dict, *, residual_limit: float = 1e-10,
                      balance_limit: float = 1e-5) -> dict:
    d = solution["diagnostics"]
    failures = []
    for key in ("electrical_residual", "thermal_residual"):
        value = d[key]["relative_l2"]
        if not np.isfinite(value) or value > residual_limit:
            failures.append(f"{key}={value}")
    for key in ("terminal_current_balance", "electrical_energy_balance", "thermal_energy_balance"):
        value = d[key]["relative"]
        if value is not None and (not np.isfinite(value) or abs(value) > balance_limit):
            failures.append(f"{key}={value}")
    u = np.asarray([c["u"] for c in scene["contacts"]], dtype=float)
    v = np.asarray(solution["v"])
    theta = np.asarray(solution["theta"])
    if v.min() < u.min() - 1e-5 or v.max() > u.max() + 1e-5:
        failures.append("potential maximum principle")
    if min(theta.min(), np.min(solution["q_sheet"]), np.min(solution["q_contact"])) < -1e-10:
        failures.append("negative heat or temperature")
    mass_error = float(np.max(abs(np.asarray(solution["contact_pre_integrals"]) - 1)))
    if mass_error > 0.002:
        failures.append(f"contact quadrature mass error={mass_error}")
    return {"accepted": not failures, "failures": failures, "contact_mass_error_max": mass_error,
            "diagnostics": d}


def audit_persisted(root: Path, scene: dict, grid: int, *, rtol: float = 1e-5) -> dict:
    path = Path(root) / "raw" / f"{scene['state_id']}_n{grid}.h5"
    with h5py.File(path) as handle:
        v = handle["potential"][:].astype(np.float64)
        theta = handle["temperature_rise"][:].astype(np.float64)
        qs = handle["q_sheet"][:].astype(np.float64)
        qc = handle["q_contact"][:].astype(np.float64)
        ix = handle["terminal_current"][:].astype(np.float64)
        jx = handle["jx_face"][:].astype(np.float64)
        jy = handle["jy_face"][:].astype(np.float64)
        hx = handle["heat_flux_x_face"][:].astype(np.float64)
        hy = handle["heat_flux_y_face"][:].astype(np.float64)
    s, k, beta = render_environment(scene["environment"], grid, grid)
    chi, _ = rasterize_contacts(scene["contacts"], grid, grid, 4)
    gamma = np.asarray([c["gamma"] for c in scene["contacts"]], dtype=np.float64)
    u = np.asarray([c["u"] for c in scene["contacts"]], dtype=np.float64)
    area = 1 / (grid * grid)
    density = gamma[:, None, None] * chi
    a = density.sum(axis=0)
    b = np.einsum("m,mij->ij", u, density)
    sx, sy = prepare_faces(s)
    kx, ky = prepare_faces(k)
    rv = apply_operator(v, sx, sy, area * a) - area * b
    rt = apply_operator(theta, kx, ky, area * beta) - area * (qs + qc)
    currents = gamma * area * np.einsum("mij,mij->m", chi, u[:, None, None] - v[None])
    face_power = float(np.sum(sx * np.diff(v, axis=1)**2) + np.sum(sy * np.diff(v, axis=0)**2))
    contact_power = float(area * qc.sum())
    supplied = float(np.dot(u, currents))
    cooling = float(area * np.sum(beta * theta))
    q_power = float(area * (qs + qc).sum())
    expected_qs, _, _ = face_energy_heat(v, s, sx, sy)
    expected_qc = np.sum(density * (u[:, None, None] - v[None])**2, axis=0)
    expected_jx, expected_jy = staggered_face_density(v, sx, sy)
    expected_hx, expected_hy = staggered_face_density(theta, kx, ky)

    def relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
        difference = float(np.linalg.norm(observed - expected))
        scale = float(np.linalg.norm(expected))
        return difference / scale if scale else (0.0 if difference == 0 else float("inf"))

    closure = {
        "electrical_residual_relative": float(np.linalg.norm(rv) / np.linalg.norm(area*b)),
        "thermal_residual_relative": float(np.linalg.norm(rt) / np.linalg.norm(area*(qs+qc))),
        "current_balance_relative": float(abs(currents.sum()) / np.abs(currents).sum()),
        "electrical_power_relative": float(abs(supplied - face_power - contact_power) / max(abs(supplied), q_power)),
        "thermal_power_relative": float(abs(cooling - q_power) / max(abs(cooling), q_power)),
        "terminal_current_storage_max_absolute": float(np.max(abs(ix-currents))),
        "terminal_current_storage_relative": relative_error(ix, currents),
        "sheet_heat_storage_relative": relative_error(qs, expected_qs),
        "contact_heat_storage_relative": relative_error(qc, expected_qc),
        "jx_storage_relative": relative_error(jx, expected_jx),
        "jy_storage_relative": relative_error(jy, expected_jy),
        "heat_flux_x_storage_relative": relative_error(hx, expected_hx),
        "heat_flux_y_storage_relative": relative_error(hy, expected_hy),
        "exterior_face_flux_max_absolute": float(max(np.max(abs(jx[:, (0,-1)])), np.max(abs(jy[(0,-1), :])),
                                                       np.max(abs(hx[:, (0,-1)])), np.max(abs(hy[(0,-1), :])))),
    }
    return {"accepted": bool(all(np.isfinite(x) and x <= rtol for key, x in closure.items()
                            if key.endswith("relative")) and closure["exterior_face_flux_max_absolute"] == 0
            and all(np.isfinite(x).all() for x in (v, theta, qs, qc, ix, jx, jy, hx, hy))
            and min(theta.min(), qs.min(), qc.min()) >= -1e-10),
            **closure}
