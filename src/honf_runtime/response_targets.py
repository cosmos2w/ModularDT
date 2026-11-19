"""Input-only bounded approximations of finite responses in an RBF space."""
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize


class ResponseTargetAdapter(Protocol):
    """A case supplies complete descriptions, legal edits and aligned observations."""
    def complete_input(self, scene: Any) -> Any: ...
    def target(self, scene: Any) -> tuple[Any, Any]: ...
    def chart(self, anchor: Any, scene: Any) -> np.ndarray: ...
    def aligned_observations(self, scenes: Any) -> Any: ...


@dataclass(frozen=True)
class BoundedTargetResult:
    coefficients: np.ndarray
    rho: float
    target_norm_squared: float
    objective: float
    success: bool
    message: str
    resolution: float


def _kernel(x, y, bandwidth):
    distance = np.square(x[:, None, :] - y[None, :, :]).sum(axis=-1)
    return np.exp(-distance / (2 * bandwidth**2))


def solve_bounded_rbf_target(chart, before, after, *, bandwidth=0.4914509773167929,
                             pair_weights=None, representation_resolution=0.0):
    """Solve one or an average of finite edits using CPU FP64 and fixed SLSQP.

    The inputs before/after have shape (pairs, coordinates), or (coordinates,).
    A target indistinguishable at the stated input resolution returns an explicit
    unsupported result with zero coefficients and rho=1.
    """
    x = np.asarray(chart, dtype=np.float64)
    before = np.atleast_2d(np.asarray(before, dtype=np.float64))
    after = np.atleast_2d(np.asarray(after, dtype=np.float64))
    if x.ndim != 2 or len(x) < 2 or before.shape != after.shape or before.shape[1:] != x.shape[1:] or not all(np.isfinite(z).all() for z in (x, before, after)):
        raise ValueError('finite compatible scene and edit charts are required')
    h = float(bandwidth)
    resolution = float(representation_resolution)
    if not np.isfinite(h) or h <= 0 or not np.isfinite(resolution) or resolution < 0:
        raise ValueError('invalid bandwidth or representation resolution')
    weights = np.ones(len(before), dtype=np.float64) / len(before) if pair_weights is None else np.asarray(pair_weights, dtype=np.float64)
    if weights.shape != (len(before),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('pair weights must be nonnegative and sum to one')
    points = np.concatenate((after, before))
    signs = np.concatenate((weights, -weights))
    # expm1 avoids subtracting near-unit kernels in the small-functional norm.
    d2 = np.square(points[:, None, :] - points[None, :, :]).sum(-1)
    ktt = float(signs @ np.expm1(-d2 / (2*h*h)) @ signs)
    numerical_floor = max(64 * np.finfo(np.float64).eps**2, (2 * resolution / h)**2)
    n = len(x)
    if not np.isfinite(ktt) or ktt <= numerical_floor:
        return BoundedTargetResult(np.zeros(n), 1., ktt, 1., False, 'target unresolved at input representation resolution', resolution)
    K = _kernel(x, x, h)
    # Stable difference even when before and after are nearby.
    db = np.square(x[:, None, :] - before[None, :, :]).sum(-1) / (2*h*h)
    da = np.square(x[:, None, :] - after[None, :, :]).sum(-1) / (2*h*h)
    kt = (np.exp(-db) * np.expm1(db - da)) @ weights
    def objective(z):
        a = z[:n] - z[n:]
        return float((a @ K @ a - 2*a @ kt + ktt)/ktt + .05*(a @ a))
    def gradient(z):
        a = z[:n] - z[n:]
        g = 2*(K @ a - kt)/ktt + .1*a
        return np.concatenate((g, -g))
    equality = lambda z: np.array([z[:n].sum()-1, z[n:].sum()-1])
    jacobian = np.zeros((2, 2*n)); jacobian[0, :n] = 1; jacobian[1, n:] = 1
    result = minimize(objective, np.full(2*n, 1/n), jac=gradient, method='SLSQP', bounds=[(0., 1.)]*(2*n),
                      constraints={'type': 'eq', 'fun': equality, 'jac': lambda z: jacobian},
                      options={'ftol': 1e-10, 'maxiter': 500})
    a = result.x[:n] - result.x[n:]
    residual = float((a @ K @ a - 2*a @ kt + ktt)/ktt)
    feasible = abs(a.sum()) <= 1e-8 and np.abs(a).sum() <= 2 + 1e-8 and np.isfinite(a).all()
    success = bool(result.success and feasible and residual >= -1e-7 and np.isfinite(residual))
    return BoundedTargetResult(a if success else np.zeros(n), float(np.sqrt(max(0., residual))) if success else 1.,
                               ktt, float(result.fun), success, str(result.message), resolution)
