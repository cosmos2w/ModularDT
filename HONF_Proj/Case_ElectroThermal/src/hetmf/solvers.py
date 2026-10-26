"""Matrix-free batched PCG for the integrated HET-MF finite-volume operators.

All arrays have native ``[batch, y, x]`` order. ``face_x`` and ``face_y``
contain *internal* integrated transmissibilities, with shapes
``[batch, ny, nx-1]`` and ``[batch, ny-1, nx]``. ``reaction`` is the integrated
cell reaction (cell area times the pointwise coefficient), and ``rhs`` is the
integrated source. No exterior faces are represented, so their flux is zero.
The thermal solve uses the same operator after the electrical/Joule step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualAudit:
    norm: torch.Tensor
    relative: torch.Tensor
    rhs_norm: torch.Tensor
    rms: torch.Tensor
    maximum: torch.Tensor


@dataclass(frozen=True)
class PCGResult:
    solution: torch.Tensor
    converged: torch.Tensor
    failed: torch.Tensor
    iterations: torch.Tensor
    true_residual_norm: torch.Tensor
    relative_residual: torch.Tensor
    rhs_norm: torch.Tensor
    failure_reason: tuple[str, ...]


@dataclass(frozen=True)
class RefinedResult:
    """Fast solve audited against canonical FP64 coefficients."""

    solution: torch.Tensor
    converged: torch.Tensor
    failed: torch.Tensor
    initial_iterations: torch.Tensor
    total_iterations: torch.Tensor
    corrections: torch.Tensor
    true_residual_norm: torch.Tensor
    relative_residual: torch.Tensor
    rhs_norm: torch.Tensor
    failure_reason: tuple[str, ...]


def _check_shapes(face_x: torch.Tensor, face_y: torch.Tensor,
                  reaction: torch.Tensor, rhs: torch.Tensor) -> None:
    if rhs.ndim != 3 or reaction.shape != rhs.shape:
        raise ValueError("rhs and reaction must have matching [batch, ny, nx] shape")
    batch, ny, nx = rhs.shape
    if min(batch, ny, nx) < 1:
        raise ValueError("batch and grid dimensions must be positive")
    if face_x.shape != (batch, ny, nx - 1) or face_y.shape != (batch, ny - 1, nx):
        raise ValueError("internal face shapes do not match rhs")
    if any(t.device != rhs.device or t.dtype != rhs.dtype for t in (face_x, face_y, reaction)):
        raise ValueError("all operator arrays must use the rhs device and dtype")
    if not rhs.is_floating_point():
        raise TypeError("floating-point tensors are required")


def apply_operator(field: torch.Tensor, face_x: torch.Tensor,
                   face_y: torch.Tensor, reaction: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric integrated five-point operator, without wrapping."""
    if field.ndim != 3 or reaction.shape != field.shape:
        raise ValueError("field and reaction must have matching [batch, ny, nx] shape")
    batch, ny, nx = field.shape
    if face_x.shape != (batch, ny, nx - 1) or face_y.shape != (batch, ny - 1, nx):
        raise ValueError("internal face shapes do not match field")
    out = reaction * field
    if nx > 1:
        difference_x = face_x * (field[:, :, :-1] - field[:, :, 1:])
        out = out + torch.nn.functional.pad(difference_x, (0, 1))
        out = out - torch.nn.functional.pad(difference_x, (1, 0))
    if ny > 1:
        difference_y = face_y * (field[:, :-1, :] - field[:, 1:, :])
        out = out + torch.nn.functional.pad(difference_y, (0, 0, 0, 1))
        out = out - torch.nn.functional.pad(difference_y, (0, 0, 1, 0))
    return out


def jacobi_diagonal(face_x: torch.Tensor, face_y: torch.Tensor,
                    reaction: torch.Tensor) -> torch.Tensor:
    """Return the exact diagonal of the integrated finite-volume operator."""
    batch, ny, nx = reaction.shape
    if face_x.shape != (batch, ny, nx - 1) or face_y.shape != (batch, ny - 1, nx):
        raise ValueError("internal face shapes do not match reaction")
    diagonal = reaction
    if nx > 1:
        diagonal = diagonal + torch.nn.functional.pad(face_x, (0, 1))
        diagonal = diagonal + torch.nn.functional.pad(face_x, (1, 0))
    if ny > 1:
        diagonal = diagonal + torch.nn.functional.pad(face_y, (0, 0, 0, 1))
        diagonal = diagonal + torch.nn.functional.pad(face_y, (0, 0, 1, 0))
    return diagonal


def audit_true_residual(solution: torch.Tensor, face_x: torch.Tensor,
                        face_y: torch.Tensor, reaction: torch.Tensor,
                        rhs: torch.Tensor) -> ResidualAudit:
    """Recompute residuals from the supplied operator, including canonical FP64.

    For a fast FP32 solve, pass the canonical FP64 coefficients and the solution
    converted to FP64. A zero right-hand side has an undefined relative residual
    (NaN); its absolute norm, RMS, and maximum remain meaningful.
    """
    _check_shapes(face_x, face_y, reaction, rhs)
    if solution.shape != rhs.shape or solution.dtype != rhs.dtype or solution.device != rhs.device:
        raise ValueError("solution must match canonical rhs shape, dtype and device")
    residual = rhs - apply_operator(solution, face_x, face_y, reaction)
    norm = torch.linalg.vector_norm(residual.flatten(1), dim=1)
    rhs_norm = torch.linalg.vector_norm(rhs.flatten(1), dim=1)
    relative = torch.where(rhs_norm > 0, norm / torch.where(rhs_norm > 0, rhs_norm, 1),
                           torch.full_like(norm, float("nan")))
    return ResidualAudit(
        norm=norm,
        relative=relative,
        rhs_norm=rhs_norm,
        rms=torch.sqrt(torch.mean(residual.square(), dim=(1, 2))),
        maximum=torch.amax(torch.abs(residual), dim=(1, 2)),
    )


def solve_pcg(face_x: torch.Tensor, face_y: torch.Tensor,
              reaction: torch.Tensor, rhs: torch.Tensor, *,
              rtol: float = 1e-10, atol: float = 1e-13,
              max_iterations: int = 4000, check_interval: int = 10,
              x0: torch.Tensor | None = None) -> PCGResult:
    """Solve independent SPD cases with Jacobi PCG and per-case true-residual checks.

    Acceptance requires ``||rhs-Ax||_2 <= max(atol, rtol*||rhs||_2)``.
    Cases that fail curvature, finite-value, or iteration checks are marked as
    failures; their last iterate is never a successful physical label. Host
    synchronization for early stopping happens only at residual-check intervals.
    """
    _check_shapes(face_x, face_y, reaction, rhs)
    if not (0 <= rtol < float("inf") and 0 <= atol < float("inf")):
        raise ValueError("rtol and atol must be finite and nonnegative")
    if max_iterations < 0 or check_interval < 1:
        raise ValueError("max_iterations must be nonnegative and check_interval positive")
    if x0 is not None and (x0.shape != rhs.shape or x0.dtype != rhs.dtype or x0.device != rhs.device):
        raise ValueError("x0 must match rhs shape, dtype and device")

    batch = rhs.shape[0]
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    diagonal = jacobi_diagonal(face_x, face_y, reaction)
    valid = (torch.isfinite(diagonal) & (diagonal > 0)).flatten(1).all(dim=1)
    valid &= torch.isfinite(rhs).flatten(1).all(dim=1)
    valid &= torch.isfinite(reaction).flatten(1).all(dim=1)
    valid &= (reaction >= 0).flatten(1).all(dim=1)
    valid &= (reaction > 0).flatten(1).any(dim=1)
    valid &= torch.isfinite(face_x).flatten(1).all(dim=1)
    valid &= torch.isfinite(face_y).flatten(1).all(dim=1)
    valid &= (face_x > 0).flatten(1).all(dim=1)
    valid &= (face_y > 0).flatten(1).all(dim=1)
    valid &= torch.isfinite(x).flatten(1).all(dim=1)
    # Sanitize invalid cases before arithmetic so a bad member cannot poison
    # the batched recurrence. Its failure flag is retained below.
    safe_face_x = torch.where(valid[:, None, None], face_x, 0)
    safe_face_y = torch.where(valid[:, None, None], face_y, 0)
    safe_reaction = torch.where(valid[:, None, None], reaction, 1)
    safe_rhs = torch.where(valid[:, None, None], rhs, 0)
    x = torch.where(valid[:, None, None], x, 0)
    diagonal = torch.where(valid[:, None, None], diagonal, 1)
    residual = safe_rhs - apply_operator(x, safe_face_x, safe_face_y, safe_reaction)
    rhs_norm = torch.linalg.vector_norm(safe_rhs.flatten(1), dim=1)
    threshold = torch.maximum(torch.full_like(rhs_norm, atol), rtol * rhs_norm)
    initial_norm = torch.linalg.vector_norm(residual.flatten(1), dim=1)
    converged = valid & (initial_norm <= threshold)
    failed = ~valid
    active = valid & ~converged
    pending = torch.zeros_like(active)
    iterations = torch.zeros(batch, dtype=torch.int32, device=rhs.device)
    z = residual / diagonal
    p = torch.where(active[:, None, None], z, 0)
    rz = torch.sum(residual * z, dim=(1, 2))
    breakdown = torch.zeros_like(active)

    for iteration in range(1, max_iterations + 1):
        if iteration == 1 or (iteration - 1) % check_interval == 0:
            if not bool((active | pending).any()):
                break
        ap = apply_operator(p, safe_face_x, safe_face_y, safe_reaction)
        curvature = torch.sum(p * ap, dim=(1, 2))
        bad = active & ((curvature <= 0) | ~torch.isfinite(curvature)
                        | (rz <= 0) | ~torch.isfinite(rz))
        breakdown |= bad
        failed |= bad
        active &= ~bad
        safe_curvature = torch.where(active, curvature, 1)
        alpha = torch.where(active, rz / safe_curvature, 0)
        x = x + alpha[:, None, None] * p
        residual = residual - alpha[:, None, None] * ap
        iterations = torch.where(active, iteration, iterations)
        z_new = residual / diagonal
        rz_new = torch.sum(residual * z_new, dim=(1, 2))
        safe_rz = torch.where(active, rz, 1)
        beta = torch.where(active, rz_new / safe_rz, 0)
        p = torch.where(active[:, None, None], z_new + beta[:, None, None] * p, 0)
        rz = rz_new
        recurrence_norm = torch.linalg.vector_norm(residual.flatten(1), dim=1)
        newly_pending = active & (recurrence_norm <= threshold)
        pending |= newly_pending
        active &= ~newly_pending
        p = torch.where(active[:, None, None], p, 0)
        if iteration % check_interval == 0 or iteration == max_iterations:
            true_r = safe_rhs - apply_operator(x, safe_face_x, safe_face_y, safe_reaction)
            true_norm = torch.linalg.vector_norm(true_r.flatten(1), dim=1)
            candidates = active | pending
            nonfinite = candidates & ~torch.isfinite(true_norm)
            failed |= nonfinite
            active &= ~nonfinite
            pending &= ~nonfinite
            newly_converged = candidates & (true_norm <= threshold)
            converged |= newly_converged
            active &= ~newly_converged
            # A recurrence candidate that misses its true-residual target is
            # restarted from the recomputed residual without changing tolerance.
            restart = pending & ~newly_converged
            residual = torch.where(restart[:, None, None], true_r, residual)
            z_new = residual / diagonal
            rz = torch.where(restart, torch.sum(residual * z_new, dim=(1, 2)), rz)
            p = torch.where(restart[:, None, None], z_new, p)
            active |= restart
            pending = torch.zeros_like(pending)
            p = torch.where(active[:, None, None], p, 0)

    final = audit_true_residual(x, safe_face_x, safe_face_y, safe_reaction, safe_rhs)
    converged &= torch.isfinite(final.norm) & (final.norm <= threshold)
    failed |= ~converged
    fail_cpu = failed.tolist()
    breakdown_cpu = breakdown.tolist()
    valid_cpu = valid.tolist()
    reasons = tuple(
        "invalid_input" if not valid_cpu[i] else
        "nonpositive_or_nonfinite_curvature" if breakdown_cpu[i] else
        "max_iterations_or_true_residual" if fail_cpu[i] else "converged"
        for i in range(batch)
    )
    # Invalid cases have an undefined residual against their original operator.
    true_norm = torch.where(valid, final.norm, torch.full_like(final.norm, float("nan")))
    rel = torch.where(valid, final.relative, torch.full_like(final.relative, float("nan")))
    input_rhs_norm = torch.linalg.vector_norm(torch.nan_to_num(rhs).flatten(1), dim=1)
    return PCGResult(x, converged, failed, iterations, true_norm, rel,
                     input_rhs_norm, reasons)


def solve_fp32_refined(face_x: torch.Tensor, face_y: torch.Tensor,
                       reaction: torch.Tensor, rhs: torch.Tensor, *,
                       rtol: float = 1e-6, atol: float = 1e-10,
                       max_iterations: int = 4000, check_interval: int = 10,
                       max_corrections: int = 3,
                       correction_rtol: float = 1e-4) -> RefinedResult:
    """FP32 PCG followed by at most three FP64-residual correction rounds.

    Inputs must be canonical FP64 tensors. Corrections solve the same physical
    operator rounded to FP32, while residuals and acceptance always use the
    original FP64 operator. A failed correction is returned for explicit FP64
    retry by the caller; tolerances and material coefficients are never changed.
    """
    _check_shapes(face_x, face_y, reaction, rhs)
    if rhs.dtype != torch.float64:
        raise TypeError("canonical inputs must be FP64")
    if max_corrections not in range(4):
        raise ValueError("max_corrections must be between zero and three")
    if not (0 < correction_rtol < 1):
        raise ValueError("correction_rtol must be between zero and one")
    fast_face_x, fast_face_y = face_x.float(), face_y.float()
    fast_reaction, fast_rhs = reaction.float(), rhs.float()
    # A mixed solve uses deliberately approximate FP32 inner solves. The final
    # acceptance threshold below remains the caller's canonical FP64 target.
    # Requiring the inner solve to cross an FP32 true-residual floor prevents
    # iterative refinement from starting on otherwise valid high-contrast rows.
    inner_rtol = max(rtol, correction_rtol) if max_corrections else rtol
    initial = solve_pcg(fast_face_x, fast_face_y, fast_reaction, fast_rhs,
                        rtol=inner_rtol, atol=atol, max_iterations=max_iterations,
                        check_interval=check_interval)
    x = initial.solution.double()
    rhs_norm = torch.linalg.vector_norm(rhs.flatten(1), dim=1)
    threshold = torch.maximum(torch.full_like(rhs_norm, atol), rtol * rhs_norm)
    audit = audit_true_residual(x, face_x, face_y, reaction, rhs)
    eligible = initial.converged.clone()
    converged = eligible & torch.isfinite(audit.norm) & (audit.norm <= threshold)
    correction_failed = torch.zeros_like(converged)
    corrections = torch.zeros_like(initial.iterations)
    total_iterations = initial.iterations.clone()
    for _ in range(max_corrections):
        pending = eligible & ~converged & ~correction_failed
        if not bool(pending.any()):
            break
        canonical_residual = rhs - apply_operator(x, face_x, face_y, reaction)
        correction_rhs = torch.where(pending[:, None, None], canonical_residual, 0).float()
        correction = solve_pcg(fast_face_x, fast_face_y, fast_reaction,
                               correction_rhs, rtol=correction_rtol, atol=0.0,
                               max_iterations=max_iterations,
                               check_interval=check_interval)
        correction_failed |= pending & correction.failed
        accepted_correction = pending & correction.converged
        x = x + torch.where(accepted_correction[:, None, None],
                            correction.solution.double(), 0)
        corrections += pending.to(corrections.dtype)
        total_iterations += torch.where(pending, correction.iterations, 0)
        audit = audit_true_residual(x, face_x, face_y, reaction, rhs)
        converged = eligible & torch.isfinite(audit.norm) & (audit.norm <= threshold)
    failed = ~converged
    eligible_cpu = eligible.tolist()
    correction_failed_cpu = correction_failed.tolist()
    converged_cpu = converged.tolist()
    reasons = tuple(
        "initial_" + initial.failure_reason[i] if not eligible_cpu[i] else
        "correction_failed" if correction_failed_cpu[i] and not converged_cpu[i] else
        "canonical_residual" if not converged_cpu[i] else "converged"
        for i in range(rhs.shape[0])
    )
    return RefinedResult(x, converged, failed, initial.iterations,
                         total_iterations, corrections, audit.norm,
                         audit.relative, rhs_norm, reasons)
