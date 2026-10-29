"""Independent CPU FP64 sparse-direct electro-thermal finite-volume reference."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse.linalg import splu, spsolve

from .contacts import rasterize_contacts
from .environment import render_environment
from .finite_volume import (apply_operator, assemble_matrix, centered_reconstruction,
                            face_energy_heat, prepare_faces, staggered_face_density)


def _residual_stats(residual: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    return {"rms_integrated": float(np.sqrt(np.mean(residual**2))),
            "max_integrated": float(np.max(np.abs(residual))),
            "l2_integrated": float(np.linalg.norm(residual)),
            "rhs_l2_integrated": float(np.linalg.norm(rhs)),
            "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(rhs)) if np.linalg.norm(rhs) > 1e-14 else np.nan}


def _balance(absolute: float, scale: float, *, threshold: float = 1e-12) -> dict[str, float | None]:
    """Use a declared unresolved relative ratio near zero; no arbitrary tiny divisor."""
    return {"absolute": float(absolute), "denominator": float(scale),
            "relative": float(absolute / scale) if scale > threshold else None,
            "relative_unresolved_below": threshold}


def prepare_scene(scene: dict[str, Any], ny: int, nx: int | None = None,
                  quadrature_order: int = 4) -> dict[str, Any]:
    """Prepare canonical FP64 grid terms shared by CPU and GPU solvers.

    `reaction_v`, `rhs_v`, and all face arrays use integrated-cell units.
    `reaction_t` is independent of voltage/contact layout. GPU callers may
    batch these exact arrays and form `rhs_t=area*q` after electrical solving.
    """
    nx = ny if nx is None else nx
    if min(nx, ny) < 2:
        raise ValueError("grid dimensions must be at least two")
    environment, contacts = scene["environment"], scene["contacts"]
    if len(contacts) < 2:
        raise ValueError("at least two contacts required")
    s, kappa, beta = render_environment(environment, ny, nx)
    chi, contact_pre_integrals = rasterize_contacts(contacts, ny, nx, quadrature_order)
    gamma = np.asarray([float(c["gamma"]) for c in contacts], dtype=np.float64)
    u = np.asarray([float(c["u"]) for c in contacts], dtype=np.float64)
    if not np.isfinite(gamma).all() or (gamma <= 0).any() or not np.isfinite(u).all():
        raise ValueError("invalid finite contact conductances or voltages")
    area = 1 / (nx * ny)
    contact_density = gamma[:, None, None] * chi
    a = contact_density.sum(axis=0)
    b = np.einsum("m,mij->ij", u, contact_density)
    face_s_x, face_s_y = prepare_faces(s)
    face_k_x, face_k_y = prepare_faces(kappa)
    return {"ny": ny, "nx": nx, "quadrature_order": quadrature_order,
            "s": s, "kappa": kappa, "beta": beta, "chi": chi,
            "contact_pre_integrals": contact_pre_integrals,
            "gamma": gamma, "u": u, "area": area, "contact_density": contact_density,
            "face_s_x": face_s_x, "face_s_y": face_s_y,
            "face_k_x": face_k_x, "face_k_y": face_k_y,
            "reaction_v": area * a, "rhs_v": area * b, "reaction_t": area * beta}


def heat_from_potential(prepared: dict[str, Any], v: np.ndarray) -> dict[str, np.ndarray]:
    """Produce the exact discrete face-energy source before thermal solving."""
    ny, nx = prepared["ny"], prepared["nx"]
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (ny, nx) or not np.isfinite(v).all():
        raise ValueError("potential must be a finite native-order cell field")
    s, face_s_x, face_s_y = (prepared[k] for k in ("s", "face_s_x", "face_s_y"))
    q_sheet, power_face_x, power_face_y = face_energy_heat(v, s, face_s_x, face_s_y)
    u, contact_density = prepared["u"], prepared["contact_density"]
    q_contact_by_pad = contact_density * (v[None, :, :] - u[:, None, None]) ** 2
    q_contact = q_contact_by_pad.sum(axis=0)
    q = q_sheet + q_contact
    return {"q": q, "q_sheet": q_sheet, "q_contact": q_contact,
            "power_face_x": power_face_x, "power_face_y": power_face_y,
            "rhs_t": prepared["area"] * q}


def finalize_solution(scene: dict[str, Any], prepared: dict[str, Any], v: np.ndarray,
                      theta: np.ndarray, *, solver: str = "external",
                      elapsed_s: float | None = None) -> dict[str, Any]:
    """Audit any solver's fields against canonical FP64 operators and heat law."""
    ny, nx = prepared["ny"], prepared["nx"]
    v, theta = np.asarray(v, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    if v.shape != (ny, nx) or theta.shape != (ny, nx) or not np.isfinite(theta).all():
        raise ValueError("solution fields must be finite and match prepared grid")
    heat = heat_from_potential(prepared, v)
    s, kappa, beta, chi, gamma, u, area = (prepared[k] for k in
                                               ("s", "kappa", "beta", "chi", "gamma", "u", "area"))
    q, q_sheet, q_contact = (heat[k] for k in ("q", "q_sheet", "q_contact"))
    face_s_x, face_s_y, face_k_x, face_k_y = (prepared[k] for k in
                                                    ("face_s_x", "face_s_y", "face_k_x", "face_k_y"))
    rhs_v, rhs_t = prepared["rhs_v"], heat["rhs_t"]
    electrical_residual = apply_operator(v, face_s_x, face_s_y, prepared["reaction_v"]) - rhs_v
    thermal_residual = apply_operator(theta, face_k_x, face_k_y, prepared["reaction_t"]) - rhs_t
    i_terminal = gamma * area * np.einsum("mij,mij->m", chi, u[:, None, None] - v[None, :, :])
    power_supply = float(np.dot(u, i_terminal))
    power_sheet = float(heat["power_face_x"].sum() + heat["power_face_y"].sum())
    power_contact = float(area * q_contact.sum())
    power_total = power_sheet + power_contact
    cooling_total = float(area * np.sum(beta * theta))
    j_face_x, j_face_y = staggered_face_density(v, face_s_x, face_s_y)
    heat_flux_face_x, heat_flux_face_y = staggered_face_density(theta, face_k_x, face_k_y)
    j_x, j_y = centered_reconstruction(j_face_x, j_face_y)
    heat_flux_x, heat_flux_y = centered_reconstruction(heat_flux_face_x, heat_flux_face_y)
    return {
        "state_id": scene.get("state_id"), "solver": solver,
        "ny": ny, "nx": nx, "quadrature_order": prepared["quadrature_order"],
        "s": s, "kappa": kappa, "beta": beta, "chi": chi,
        "contact_pre_integrals": prepared["contact_pre_integrals"],
        "v": v, "theta": theta, "q": q, "q_sheet": q_sheet, "q_contact": q_contact,
        "terminal_currents": i_terminal,
        "j_face_x": j_face_x, "j_face_y": j_face_y, "j_x": j_x, "j_y": j_y,
        "heat_flux_face_x": heat_flux_face_x, "heat_flux_face_y": heat_flux_face_y,
        "heat_flux_x": heat_flux_x, "heat_flux_y": heat_flux_y,
        "diagnostics": {
            "electrical_residual": _residual_stats(electrical_residual, rhs_v),
            "thermal_residual": _residual_stats(thermal_residual, rhs_t),
            "terminal_current_balance": _balance(float(i_terminal.sum()), float(np.abs(i_terminal).sum())),
            "electrical_energy_balance": _balance(power_supply - power_total,
                                                   max(abs(power_supply), power_total)),
            "thermal_energy_balance": _balance(cooling_total - power_total,
                                                max(abs(cooling_total), power_total)),
            "power_supply": power_supply, "power_sheet": power_sheet,
            "power_contact": power_contact, "power_total": power_total,
            "cooling_total": cooling_total,
            "contact_power_fraction": power_contact / power_total if power_total > 1e-12 else None,
            "elapsed_s": elapsed_s,
        },
    }


def solve_cpu(scene: dict[str, Any], ny: int, nx: int | None = None,
              quadrature_order: int = 4) -> dict[str, Any]:
    """Independent FP64 sparse-direct solve on integrated cell operators."""
    started = perf_counter()
    prepared = prepare_scene(scene, ny, nx, quadrature_order)
    ny, nx = prepared["ny"], prepared["nx"]
    matrix_v = assemble_matrix(prepared["face_s_x"], prepared["face_s_y"], prepared["reaction_v"])
    v = spsolve(matrix_v, prepared["rhs_v"].ravel()).reshape(ny, nx)
    heat = heat_from_potential(prepared, v)
    matrix_t = assemble_matrix(prepared["face_k_x"], prepared["face_k_y"], prepared["reaction_t"])
    theta = spsolve(matrix_t, heat["rhs_t"].ravel()).reshape(ny, nx)
    return finalize_solution(scene, prepared, v, theta, solver="cpu_fp64_sparse_direct",
                             elapsed_s=perf_counter() - started)


def solve_scene(scene: dict[str, Any], n: int, quadrature: int = 4) -> dict[str, Any]:
    """Square-grid convenience alias for manifest/pilot callers."""
    return solve_cpu(scene, n, n, quadrature)


def terminal_conductance_matrix(scene: dict[str, Any], n: int,
                                quadrature_order: int = 4) -> dict[str, Any]:
    """Input-only electrical unit-voltage basis and terminal response matrix.

    The `m` electrical basis solves are counted explicitly. Factorization is
    reused, but no large per-pad fields are returned or persisted by default.
    """
    prepared = prepare_scene(scene, n, n, quadrature_order)
    matrix = assemble_matrix(prepared["face_s_x"], prepared["face_s_y"], prepared["reaction_v"])
    factor = splu(matrix.tocsc())
    chi, gamma, area = prepared["chi"], prepared["gamma"], prepared["area"]
    m = len(gamma)
    response = np.empty((m, m), dtype=np.float64)
    max_relative_residual = 0.0
    for j in range(m):
        rhs = area * prepared["contact_density"][j]
        v = factor.solve(rhs.ravel()).reshape(n, n)
        residual = apply_operator(v, prepared["face_s_x"], prepared["face_s_y"],
                                  prepared["reaction_v"]) - rhs
        max_relative_residual = max(max_relative_residual, float(np.linalg.norm(residual) / np.linalg.norm(rhs)))
        response[:, j] = -gamma * area * np.einsum("mij,ij->m", chi, v)
        response[j, j] += gamma[j]
    symmetric = 0.5 * (response + response.T)
    return {"matrix": response, "basis_solves": m, "max_relative_residual": max_relative_residual,
            "max_symmetry_error": float(np.max(np.abs(response - response.T))),
            "max_row_sum_error": float(np.max(np.abs(response @ np.ones(m)))),
            "min_symmetric_eigenvalue": float(np.linalg.eigvalsh(symmetric).min())}
