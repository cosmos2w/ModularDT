"""Focused CPU FP64 checks for the batched matrix-free finite-volume solver."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetmf.solvers import (  # noqa: E402
    apply_operator,
    audit_true_residual,
    solve_fp32_refined,
    solve_pcg,
)


def _reference_matrix(tx, ty, reaction):
    ny, nx = reaction.shape
    matrix = lil_matrix((ny * nx, ny * nx), dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
            p = y * nx + x
            matrix[p, p] += reaction[y, x]
            if x + 1 < nx:
                q = p + 1
                t = tx[y, x]
                matrix[p, p] += t
                matrix[q, q] += t
                matrix[p, q] -= t
                matrix[q, p] -= t
            if y + 1 < ny:
                q = p + nx
                t = ty[y, x]
                matrix[p, p] += t
                matrix[q, q] += t
                matrix[p, q] -= t
                matrix[q, p] -= t
    return matrix.tocsr()


def _problem(batch=3, ny=4, nx=5):
    rng = np.random.default_rng(17)
    tx = torch.tensor(rng.uniform(0.2, 1.8, (batch, ny, nx - 1)), dtype=torch.float64)
    ty = torch.tensor(rng.uniform(0.2, 1.8, (batch, ny - 1, nx)), dtype=torch.float64)
    reaction = torch.tensor(rng.uniform(0.03, 0.3, (batch, ny, nx)), dtype=torch.float64)
    rhs = torch.tensor(rng.normal(size=(batch, ny, nx)), dtype=torch.float64)
    return tx, ty, reaction, rhs


def test_operator_is_symmetric_spd_and_has_insulating_edges():
    tx, ty, reaction, _ = _problem(batch=1, ny=3, nx=4)
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(size=(1, 3, 4)), dtype=torch.float64)
    y = torch.tensor(rng.normal(size=(1, 3, 4)), dtype=torch.float64)
    ax = apply_operator(x, tx, ty, reaction)
    ay = apply_operator(y, tx, ty, reaction)
    assert torch.sum(x * ay).item() == pytest.approx(torch.sum(y * ax).item(), abs=1e-13)
    assert torch.sum(x * ax).item() > 0
    matrix = _reference_matrix(tx[0].numpy(), ty[0].numpy(), reaction[0].numpy())
    np.testing.assert_allclose(ax[0].numpy().ravel(), matrix @ x[0].numpy().ravel(), atol=1e-14)
    # With zero reaction a constant has no leakage through exterior boundaries.
    flux = apply_operator(torch.ones_like(x), tx, ty, torch.zeros_like(reaction))
    assert torch.count_nonzero(flux) == 0


def test_heterogeneous_batch_matches_independently_assembled_sparse_direct():
    tx, ty, reaction, rhs = _problem()
    rhs[0] = 0  # zero-response member remains inactive
    rhs[1] *= 1e-12  # small physical scale uses the absolute tolerance
    result = solve_pcg(tx, ty, reaction, rhs, rtol=1e-11, atol=1e-15,
                       max_iterations=1000, check_interval=7)
    assert result.converged.tolist() == [True, True, True]
    assert result.failed.tolist() == [False, False, False]
    assert result.iterations[0].item() == 0
    assert result.iterations[1].item() > 0
    assert result.iterations[2].item() > 0
    assert result.failure_reason == ("converged",) * 3
    for i in range(3):
        matrix = _reference_matrix(tx[i].numpy(), ty[i].numpy(), reaction[i].numpy())
        expected = spsolve(matrix, rhs[i].numpy().ravel()).reshape(rhs.shape[1:])
        np.testing.assert_allclose(result.solution[i].numpy(), expected, rtol=1e-10, atol=1e-13)
    audit = audit_true_residual(result.solution, tx, ty, reaction, rhs)
    assert audit.norm[0].item() == 0
    assert torch.isnan(audit.relative[0])
    assert torch.all(audit.norm <= torch.maximum(torch.tensor(1e-15), 1e-11 * audit.rhs_norm))


def test_one_cell_exact_solve_does_not_break_down_before_true_check():
    tx = torch.empty((1, 1, 0), dtype=torch.float64)
    ty = torch.empty((1, 0, 1), dtype=torch.float64)
    reaction = torch.tensor([[[2.0]]], dtype=torch.float64)
    rhs = torch.tensor([[[6.0]]], dtype=torch.float64)
    result = solve_pcg(tx, ty, reaction, rhs, check_interval=20)
    assert result.converged.item()
    assert result.iterations.item() == 1
    assert result.solution.item() == pytest.approx(3.0)


def test_max_iteration_failure_and_bad_case_are_isolated():
    tx, ty, reaction, rhs = _problem(batch=2, ny=5, nx=5)
    result = solve_pcg(tx, ty, reaction, rhs, max_iterations=1, check_interval=10)
    assert result.failed.tolist() == [True, True]
    assert result.failure_reason == ("max_iterations_or_true_residual",) * 2
    reaction[0, 0, 0] = -1
    result = solve_pcg(tx, ty, reaction, rhs, max_iterations=1000, check_interval=10)
    assert result.failure_reason[0] == "invalid_input"
    assert result.failed[0].item()
    assert result.converged[1].item()


def test_canonical_fp64_audit_detects_low_precision_operator_mismatch():
    tx, ty, reaction, rhs = _problem(batch=1, ny=6, nx=6)
    fast = solve_pcg(tx.float(), ty.float(), reaction.float(), rhs.float(),
                     rtol=1e-6, atol=1e-10, max_iterations=1000)
    assert fast.converged.item()
    canonical = audit_true_residual(fast.solution.double(), tx, ty, reaction, rhs)
    assert canonical.norm.item() > 0
    assert canonical.norm.item() > fast.true_residual_norm.item() * 1e-3
    assert canonical.maximum.item() > 0


def test_fp32_refinement_uses_canonical_operator_and_counts_corrections():
    tx = torch.empty((2, 1, 0), dtype=torch.float64)
    ty = torch.empty((2, 0, 1), dtype=torch.float64)
    reaction = torch.tensor([[[1.234567890123]], [[2.0]]], dtype=torch.float64)
    rhs = torch.tensor([[[0.876543210987]], [[0.0]]], dtype=torch.float64)
    result = solve_fp32_refined(tx, ty, reaction, rhs, rtol=1e-10,
                                atol=1e-13, max_corrections=3)
    assert result.converged.tolist() == [True, True]
    assert result.corrections.tolist() == [1, 0]
    assert result.initial_iterations.tolist() == [1, 0]
    assert result.total_iterations.tolist() == [2, 0]
    assert result.true_residual_norm[0].item() < 1e-10 * abs(rhs[0, 0, 0].item())
    assert result.failure_reason == ("converged", "converged")
