"""Canonical 12 by 12 HET-MF material controls and log-bilinear renderer.

All randomness is keyed locally. Neither contacts nor solved fields enter this module.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

RENDERER = "log_bilinear_centers_border_v1"
CONTROL_SIZE = 12


def _rng(seed: int, channel: str) -> np.random.Generator:
    digest = hashlib.sha256(f"hetmf_contact_sheet_v1:{seed}:{channel}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _centers() -> np.ndarray:
    t = (np.arange(CONTROL_SIZE, dtype=np.float64) + 0.5) / CONTROL_SIZE
    xx, yy = np.meshgrid(t, t)
    return np.stack((xx, yy), axis=-1)


def _grf_factor(lengths: tuple[float, float], angle: float) -> tuple[np.ndarray, float]:
    xy = _centers().reshape(-1, 2)
    delta = xy[:, None, :] - xy[None, :, :]
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    scaled = delta @ rot / np.asarray(lengths, dtype=np.float64)
    covariance = np.exp(-0.5 * np.sum(scaled**2, axis=-1))
    for jitter in (1e-10, 1e-8):
        try:
            factor = np.linalg.cholesky(covariance + jitter * np.eye(xy.shape[0]))
            return factor, jitter
        except np.linalg.LinAlgError:
            continue
    raise ArithmeticError("GRF covariance Cholesky failed at maximum declared jitter")


def _draw_grf(factor: np.ndarray, seed: int, channel: str) -> np.ndarray:
    return (factor @ _rng(seed, channel).standard_normal(factor.shape[0])).reshape(12, 12)


def _morphology(seed: int, family: str, xy: np.ndarray) -> np.ndarray:
    rng = _rng(seed, "morphology")
    x, y = xy[..., 0], xy[..., 1]
    if family == "uniform":
        return np.zeros((12, 12), dtype=np.float64)
    if family == "correlated_mosaic":
        return np.zeros((12, 12), dtype=np.float64)
    if family in {"layered_barrier", "barrier_gap"}:
        angle = rng.uniform(0.15, np.pi - 0.15)
        normal = np.cos(angle) * (x - 0.5) + np.sin(angle) * (y - 0.5)
        width = rng.uniform(2.0, 3.2) / 12
        if family == "barrier_gap":
            tangent = -np.sin(angle) * (x - 0.5) + np.cos(angle) * (y - 0.5)
            gap = np.abs(tangent - rng.uniform(-0.14, 0.14)) < rng.uniform(2.0, 2.8) / 24
            return np.where((np.abs(normal) < width / 2) & ~gap, -2.2, 0.5)
        offsets = rng.uniform(-0.22, 0.22, size=int(rng.integers(2, 5)))
        bands = np.zeros_like(x)
        for offset in offsets:
            bands -= 1.9 * np.exp(-0.5 * ((normal - offset) / (width / 2)) ** 4)
        return bands + 0.4
    if family == "inclusions":
        z = np.zeros_like(x)
        for _ in range(int(rng.integers(2, 6))):
            cx, cy = rng.uniform(0.18, 0.82, size=2)
            a, b = rng.uniform(0.10, 0.24, size=2)
            phi = rng.uniform(0, np.pi)
            dx, dy = x - cx, y - cy
            xr = np.cos(phi) * dx + np.sin(phi) * dy
            yr = -np.sin(phi) * dx + np.cos(phi) * dy
            d = (xr / a) ** 2 + (yr / b) ** 2
            z += rng.choice((-2.2, 2.2)) * np.clip(1.5 - d, 0, 1)
        return z
    if family in {"corridors", "composite"}:
        z = np.full_like(x, -1.2)
        n_nodes = int(rng.integers(3, 5))
        nodes = np.column_stack((np.linspace(0.04, 0.96, n_nodes), rng.uniform(0.20, 0.80, n_nodes)))
        dist = np.full_like(x, np.inf)
        for p, q in zip(nodes[:-1], nodes[1:]):
            vx, vy = q - p
            t = np.clip(((x - p[0]) * vx + (y - p[1]) * vy) / (vx * vx + vy * vy), 0, 1)
            dist = np.minimum(dist, np.hypot(x - p[0] - t * vx, y - p[1] - t * vy))
        width = rng.uniform(2.0, 3.0) / 12
        z += 3.3 * np.clip((width / 2 - dist) * 12, 0, 1)
        if family == "composite":
            z -= 1.6 * (np.abs(y - 0.49) < 0.09) * (np.abs(x - 0.52) > 0.12)
        return z
    raise ValueError(f"unknown environment family: {family}")


def generate_environment(
    seed: int,
    family: str,
    *,
    contrast_s: float | None = None,
    contrast_k: float | None = None,
    contrast_beta: float | None = None,
    beta0: float | None = None,
    rho: float | None = None,
    correlation_lengths: tuple[float, float] | None = None,
    orientation: float | None = None,
) -> dict[str, Any]:
    """Generate one input-only catalogue; returned controls are JSON-compatible."""
    rng = _rng(seed, "parameters")
    cs = float(contrast_s if contrast_s is not None else np.exp(rng.uniform(np.log(3), np.log(30))))
    ck = float(contrast_k if contrast_k is not None else np.exp(rng.uniform(np.log(1.5), np.log(6))))
    cb = float(contrast_beta if contrast_beta is not None else np.exp(rng.uniform(0, np.log(3))))
    b0 = float(beta0 if beta0 is not None else np.exp(rng.uniform(np.log(2), np.log(12))))
    correlation = float(rho if rho is not None else rng.choice((0.0, 0.7)))
    if min(cs, ck, cb, b0) <= 0 or cs < 1 or ck < 1 or cb < 1 or abs(correlation) > 1:
        raise ValueError("invalid positive coefficient bounds or latent correlation")
    if correlation_lengths is None:
        long = rng.uniform(0.12, 0.35)
        short = long / rng.uniform(1, 3)
        correlation_lengths = (float(long), float(short))
    if min(correlation_lengths) <= 0:
        raise ValueError("correlation lengths must be positive")
    angle = float(orientation if orientation is not None else rng.uniform(0, np.pi))
    xy = _centers()
    morph = _morphology(seed, family, xy)
    if family == "uniform":
        gs = np.zeros((12, 12))
        gk = np.zeros((12, 12))
        gb = np.zeros((12, 12))
        jitter = None
    else:
        factor, jitter = _grf_factor(correlation_lengths, angle)
        gs_random = _draw_grf(factor, seed, "electrical")
        gk_random = _draw_grf(factor, seed, "thermal")
        gb = _draw_grf(factor, seed, "cooling")
        texture = float(rng.uniform(0.10, 0.20))
        gs = gs_random if family == "correlated_mosaic" else morph + texture * gs_random
        independent_k = gk_random
        gk = correlation * gs + np.sqrt(1 - correlation**2) * independent_k
    s = np.exp(0.5 * np.log(cs) * np.tanh(gs))
    kappa = np.exp(0.5 * np.log(ck) * np.tanh(gk))
    beta = b0 * np.exp(0.5 * np.log(cb) * np.tanh(gb))
    return {
        "family": family, "root_id": f"env-{seed:010d}", "seed": int(seed), "renderer": RENDERER,
        "s": s.tolist(), "kappa": kappa.tolist(), "beta": beta.tolist(),
        "provenance": {"contrast_s_bound": cs, "contrast_k_bound": ck,
                       "contrast_beta_bound": cb, "beta0": b0, "rho": correlation,
                       "correlation_lengths": list(correlation_lengths), "orientation": angle,
                       "grf_jitter": jitter,
                       "realized_s_ratio": float(s.max() / s.min()),
                       "realized_kappa_ratio": float(kappa.max() / kappa.min()),
                       "realized_beta_ratio": float(beta.max() / beta.min())},
    }


def _interpolate_log(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if values.shape != (12, 12) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("controls must be finite positive 12 by 12 arrays")
    ux = np.clip(12 * np.asarray(x, dtype=np.float64) - 0.5, 0, 11)
    uy = np.clip(12 * np.asarray(y, dtype=np.float64) - 0.5, 0, 11)
    ix, iy = np.floor(ux).astype(int), np.floor(uy).astype(int)
    jx, jy = np.minimum(ix + 1, 11), np.minimum(iy + 1, 11)
    wx, wy = ux - ix, uy - iy
    logv = np.log(values)
    z = ((1 - wy) * ((1 - wx) * logv[iy, ix] + wx * logv[iy, jx])
         + wy * ((1 - wx) * logv[jy, ix] + wx * logv[jy, jx]))
    return np.exp(z)


def render_points(environment: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render canonical positive coefficients at arbitrary points on the sheet."""
    if environment.get("renderer", RENDERER) != RENDERER:
        raise ValueError("unsupported renderer for canonical reference")
    return tuple(_interpolate_log(np.asarray(environment[key]), x, y) for key in ("s", "kappa", "beta"))


def render_environment(environment: dict[str, Any], ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(nx, ny) < 2:
        raise ValueError("grid dimensions must be at least two")
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    return render_points(environment, xx, yy)


def control_records(environment: dict[str, Any]) -> list[dict[str, float | int]]:
    """Expose all 144 supplied records with their input quadrature convention."""
    s, kappa, beta = (np.asarray(environment[k]) for k in ("s", "kappa", "beta"))
    return [{"record_id": iy * 12 + ix, "center_x": (ix + 0.5) / 12,
             "center_y": (iy + 0.5) / 12, "spacing_x": 1 / 12, "spacing_y": 1 / 12,
             "sampling_area": 1 / 144, "s": float(s[iy, ix]),
             "kappa": float(kappa[iy, ix]), "beta": float(beta[iy, ix])}
            for iy in range(12) for ix in range(12)]
