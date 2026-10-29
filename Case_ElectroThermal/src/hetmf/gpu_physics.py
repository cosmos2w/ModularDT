"""Batched torch electro-thermal solves with canonical FP64 physical audits.

The caller owns device selection, process checks, and its memory policy.
This module does not select another GPU or alter another process. CPU torch is
supported for deterministic small tests of the same batched path.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .physics import finalize_solution, prepare_scene
from .solvers import PCGResult, RefinedResult, audit_true_residual, solve_fp32_refined, solve_pcg


class BatchSolveError(RuntimeError):
    """A linear solve failed; its last iterate must not be saved as a label."""

    def __init__(self, stage: str, failed_ids: list[str], reasons: list[str],
                 details: list[dict[str, Any]] | None = None):
        self.stage = stage
        self.failed_ids = failed_ids
        self.reasons = reasons
        self.details = details or []
        super().__init__(f"{stage} solve failed for {list(zip(failed_ids, reasons))}")


def _stack(prepared: Sequence[dict[str, Any]], key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack([p[key] for p in prepared]),
                           dtype=torch.float64, device=device)


def _heat_rhs(v: torch.Tensor, s: torch.Tensor, face_x: torch.Tensor,
              face_y: torch.Tensor, contact_density: torch.Tensor,
              u: torch.Tensor, area: float) -> torch.Tensor:
    """Integrated Joule source from unique faces and finite contacts.

    Each face power is divided by the two cell areas according to the two
    half-cell electrical resistances, exactly as in the CPU reference.
    """
    px = face_x * (v[:, :, :-1] - v[:, :, 1:]).square()
    py = face_y * (v[:, :-1, :] - v[:, 1:, :]).square()
    wx = s[:, :, 1:] / (s[:, :, :-1] + s[:, :, 1:])
    wy = s[:, 1:, :] / (s[:, :-1, :] + s[:, 1:, :])
    cell_power = F.pad(wx * px, (0, 1)) + F.pad((1 - wx) * px, (1, 0))
    cell_power = cell_power + F.pad(wy * py, (0, 0, 0, 1))
    cell_power = cell_power + F.pad((1 - wy) * py, (0, 0, 1, 0))
    q_contact = (contact_density * (v[:, None] - u[:, :, None, None]).square()).sum(dim=1)
    return cell_power + area * q_contact


def _solve_stage(face_x: torch.Tensor, face_y: torch.Tensor,
                 reaction: torch.Tensor, rhs: torch.Tensor, *,
                 precision: str, rtol: float, atol: float,
                 max_iterations: int, check_interval: int) -> PCGResult | RefinedResult:
    if precision == "fp64":
        return solve_pcg(face_x, face_y, reaction, rhs, rtol=rtol, atol=atol,
                         max_iterations=max_iterations, check_interval=check_interval)
    return solve_fp32_refined(face_x, face_y, reaction, rhs, rtol=rtol, atol=atol,
                              max_iterations=max_iterations, check_interval=check_interval,
                              max_corrections=0 if precision == "fp32" else 3)


def _solver_diagnostic(result: PCGResult | RefinedResult, i: int,
                       precision: str) -> dict[str, Any]:
    out = {
        "precision": precision,
        "accepted": bool(result.converged[i]),
        "failure_reason": result.failure_reason[i],
        "true_residual_l2_integrated": float(result.true_residual_norm[i]),
        "rhs_l2_integrated": float(result.rhs_norm[i]),
        "true_relative_residual": float(result.relative_residual[i]),
    }
    if isinstance(result, RefinedResult):
        out.update(initial_iterations=int(result.initial_iterations[i]),
                   total_iterations=int(result.total_iterations[i]),
                   corrections=int(result.corrections[i]))
    else:
        out.update(initial_iterations=int(result.iterations[i]),
                   total_iterations=int(result.iterations[i]), corrections=0)
    return out


@torch.no_grad()
def solve_gpu_batch(scenes: Sequence[dict[str, Any]], n: int,
                    precision: str = "fp64", device: str = "cuda:0", *,
                    quadrature_order: int = 4, rtol: float | None = None,
                    atol: float | None = None, max_iterations: int = 4000,
                    check_interval: int = 10) -> list[dict[str, Any]]:
    """Solve square-grid physical scenes with one batched electrical/thermal pair.

    ``fp64`` uses torch FP64 PCG. ``fp32`` accepts only canonical-FP64-audited
    FP32 results, while ``mixed`` allows up to three FP32 correction solves.
    Failed states raise ``BatchSolveError`` with exact case IDs and reasons;
    caller policy may retry those exact inputs with FP64 or CPU sparse direct.
    The returned fields are FP64 NumPy arrays and use ``finalize_solution`` for
    currents, power, face fluxes, and local/global conservation diagnostics.
    """
    if not scenes:
        return []
    if precision not in {"fp64", "fp32", "mixed"}:
        raise ValueError("precision must be fp64, fp32, or mixed")
    if n < 2:
        raise ValueError("grid dimension must be at least two")
    rtol = (1e-10 if precision == "fp64" else 1e-6) if rtol is None else rtol
    atol = (1e-13 if precision == "fp64" else 1e-10) if atol is None else atol
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    start = perf_counter()
    prepared = [prepare_scene(scene, n, n, quadrature_order) for scene in scenes]
    face_s_x = _stack(prepared, "face_s_x", target_device)
    face_s_y = _stack(prepared, "face_s_y", target_device)
    reaction_v = _stack(prepared, "reaction_v", target_device)
    rhs_v = _stack(prepared, "rhs_v", target_device)
    face_k_x = _stack(prepared, "face_k_x", target_device)
    face_k_y = _stack(prepared, "face_k_y", target_device)
    reaction_t = _stack(prepared, "reaction_t", target_device)

    electrical = _solve_stage(face_s_x, face_s_y, reaction_v, rhs_v,
                              precision=precision, rtol=rtol, atol=atol,
                              max_iterations=max_iterations, check_interval=check_interval)
    if not bool(electrical.converged.all()):
        failed = [i for i, ok in enumerate(electrical.converged.tolist()) if not ok]
        raise BatchSolveError("electrical", [str(scenes[i].get("state_id", i)) for i in failed],
                              [electrical.failure_reason[i] for i in failed],
                              [_solver_diagnostic(electrical, i, precision) for i in failed])
    v = electrical.solution.double()
    s = _stack(prepared, "s", target_device)
    pad_count = max(len(p["u"]) for p in prepared)
    contact_density = np.zeros((len(prepared), pad_count, n, n), dtype=np.float64)
    voltages = np.zeros((len(prepared), pad_count), dtype=np.float64)
    for i, p in enumerate(prepared):
        count = len(p["u"])
        contact_density[i, :count] = p["contact_density"]
        voltages[i, :count] = p["u"]
    density_tensor = torch.as_tensor(contact_density, dtype=torch.float64, device=target_device)
    voltage_tensor = torch.as_tensor(voltages, dtype=torch.float64, device=target_device)
    rhs_t = _heat_rhs(v, s, face_s_x, face_s_y, density_tensor, voltage_tensor, 1 / n**2)
    thermal = _solve_stage(face_k_x, face_k_y, reaction_t, rhs_t,
                           precision=precision, rtol=rtol, atol=atol,
                           max_iterations=max_iterations, check_interval=check_interval)
    if not bool(thermal.converged.all()):
        failed = [i for i, ok in enumerate(thermal.converged.tolist()) if not ok]
        raise BatchSolveError("thermal", [str(scenes[i].get("state_id", i)) for i in failed],
                              [thermal.failure_reason[i] for i in failed],
                              [_solver_diagnostic(thermal, i, precision) for i in failed])
    theta = thermal.solution.double()
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    v_np = v.cpu().numpy()
    theta_np = theta.cpu().numpy()
    out = []
    for i, (scene, p) in enumerate(zip(scenes, prepared)):
        result = finalize_solution(scene, p, v_np[i], theta_np[i],
                                   solver=f"torch_{precision}_batched_pcg")
        result["diagnostics"]["electrical_solver"] = _solver_diagnostic(electrical, i, precision)
        result["diagnostics"]["thermal_solver"] = _solver_diagnostic(thermal, i, precision)
        result["diagnostics"]["batch_size"] = len(scenes)
        out.append(result)
    elapsed = perf_counter() - start
    for result in out:
        result["diagnostics"]["elapsed_s"] = elapsed / len(scenes)
        result["diagnostics"]["batch_elapsed_s"] = elapsed
    return out
