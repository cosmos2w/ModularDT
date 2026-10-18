"""Integrated cell-centered orthogonal finite-volume operators and face energy."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def prepare_faces(coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic transmissibilities on +x and +y internal faces."""
    c = np.asarray(coeff, dtype=np.float64)
    if c.ndim != 2 or min(c.shape) < 2 or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("coefficient must be a finite positive 2D field")
    ny, nx = c.shape
    tx = (2 * nx / ny) / (1 / c[:, :-1] + 1 / c[:, 1:])
    ty = (2 * ny / nx) / (1 / c[:-1, :] + 1 / c[1:, :])
    return tx, ty


def apply_operator(field: np.ndarray, face_x: np.ndarray, face_y: np.ndarray, reaction: np.ndarray) -> np.ndarray:
    """Apply the integrated SPD operator without periodic wraparound."""
    v = np.asarray(field)
    tx, ty, a = np.asarray(face_x), np.asarray(face_y), np.asarray(reaction)
    if tx.shape != (v.shape[0], v.shape[1] - 1) or ty.shape != (v.shape[0] - 1, v.shape[1]) or a.shape != v.shape:
        raise ValueError("face/reaction shapes disagree with cell field")
    out = a * v
    fx = tx * (v[:, :-1] - v[:, 1:])
    fy = ty * (v[:-1, :] - v[1:, :])
    out = out.copy()
    out[:, :-1] += fx
    out[:, 1:] -= fx
    out[:-1, :] += fy
    out[1:, :] -= fy
    return out


def assemble_matrix(face_x: np.ndarray, face_y: np.ndarray, reaction: np.ndarray) -> sparse.csr_matrix:
    """Independently assemble the five-point integrated sparse matrix."""
    a = np.asarray(reaction, dtype=np.float64)
    ny, nx = a.shape
    if face_x.shape != (ny, nx - 1) or face_y.shape != (ny - 1, nx):
        raise ValueError("face shapes disagree with reaction")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("reaction must be finite and nonnegative")
    ids = np.arange(ny * nx).reshape(ny, nx)
    p_x, r_x = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    p_y, r_y = ids[:-1, :].ravel(), ids[1:, :].ravel()
    p = np.r_[p_x, p_y]
    r = np.r_[r_x, r_y]
    t = np.r_[np.asarray(face_x).ravel(), np.asarray(face_y).ravel()]
    diagonal = a.ravel().copy()
    np.add.at(diagonal, p, t)
    np.add.at(diagonal, r, t)
    rows = np.r_[np.arange(ny * nx), p, r]
    cols = np.r_[np.arange(ny * nx), r, p]
    data = np.r_[diagonal, -t, -t]
    return sparse.coo_matrix((data, (rows, cols)), shape=(ny * nx, ny * nx)).tocsr()


def face_energy_heat(field: np.ndarray, coeff: np.ndarray, face_x: np.ndarray | None = None,
                     face_y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition unique-face power in proportion to half-cell resistance."""
    v = np.asarray(field, dtype=np.float64)
    c = np.asarray(coeff, dtype=np.float64)
    if v.shape != c.shape:
        raise ValueError("field and coefficient shapes disagree")
    ny, nx = v.shape
    tx, ty = prepare_faces(c) if face_x is None or face_y is None else (face_x, face_y)
    power_x = tx * (v[:, :-1] - v[:, 1:]) ** 2
    power_y = ty * (v[:-1, :] - v[1:, :]) ** 2
    cell_power = np.zeros_like(v)
    rx_left, rx_right = 1 / c[:, :-1], 1 / c[:, 1:]
    wx = rx_left / (rx_left + rx_right)
    cell_power[:, :-1] += wx * power_x
    cell_power[:, 1:] += (1 - wx) * power_x
    ry_lower, ry_upper = 1 / c[:-1, :], 1 / c[1:, :]
    wy = ry_lower / (ry_lower + ry_upper)
    cell_power[:-1, :] += wy * power_y
    cell_power[1:, :] += (1 - wy) * power_y
    return cell_power * (nx * ny), power_x, power_y


def staggered_face_density(field: np.ndarray, face_x: np.ndarray, face_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oriented +x/+y flux densities; exterior components are exactly zero."""
    v = np.asarray(field)
    ny, nx = v.shape
    out_x = np.zeros((ny, nx + 1), dtype=np.float64)
    out_y = np.zeros((ny + 1, nx), dtype=np.float64)
    out_x[:, 1:-1] = face_x * (v[:, :-1] - v[:, 1:]) * ny
    out_y[1:-1, :] = face_y * (v[:-1, :] - v[1:, :]) * nx
    return out_x, out_y


def centered_reconstruction(face_density_x: np.ndarray, face_density_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (0.5 * (face_density_x[:, :-1] + face_density_x[:, 1:]),
            0.5 * (face_density_y[:-1, :] + face_density_y[1:, :]))
