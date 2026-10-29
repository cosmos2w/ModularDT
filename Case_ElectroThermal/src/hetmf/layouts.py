"""Input-only legal finite-contact layout and operating-voltage generator."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _rng(seed: int, channel: str) -> np.random.Generator:
    digest = hashlib.sha256(f"hetmf_contact_sheet_v1:{seed}:{channel}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def validate_contacts(contacts: list[dict], clearance: float = 0.015) -> float:
    if len(contacts) < 2:
        raise ValueError("at least two contacts required")
    nearest = np.inf
    for i, ci in enumerate(contacts):
        x, y, r, g, u = (float(ci[k]) for k in ("x", "y", "r", "gamma", "u"))
        if not all(np.isfinite(z) for z in (x, y, r, g, u)) or r <= 0 or g <= 0:
            raise ValueError("contacts need finite coordinates, voltage and positive radius/conductance")
        if min(x, y, 1 - x, 1 - y) < r + 0.025 - 1e-12:
            raise ValueError("contact violates edge clearance")
        for cj in contacts[:i]:
            gap = np.hypot(x - float(cj["x"]), y - float(cj["y"])) - r - float(cj["r"])
            nearest = min(nearest, float(gap))
            if gap < clearance - 1e-12:
                raise ValueError("contacts overlap or violate pairwise clearance")
    voltages = [float(c["u"]) for c in contacts]
    if max(voltages) - min(voltages) <= 0:
        raise ValueError("ordinary case must have a nonzero voltage range")
    return float(nearest)


def _proposal_centers(rng: np.random.Generator, m: int, family: str, attempt: int,
                      radii: np.ndarray) -> tuple[np.ndarray, int]:
    if family == "dispersed":
        centers = []
        rejects = 0
        for i in range(m):
            margin = float(radii[i] + 0.025)
            for _ in range(2000):
                candidate = rng.uniform(margin, 1 - margin, size=2)
                if all(np.linalg.norm(candidate - previous) >= radii[i] + radii[j] + 0.015
                       for j, previous in enumerate(centers)):
                    centers.append(candidate)
                    break
                rejects += 1
            else:
                return np.full((m, 2), np.nan), rejects
        return np.asarray(centers), rejects
    if family == "rows":
        columns = int(np.ceil(np.sqrt(m)))
        rows = int(np.ceil(m / columns))
        xx, yy = np.meshgrid(np.linspace(-0.33, 0.33, columns), np.linspace(-0.32, 0.32, rows))
        base = np.column_stack((xx.ravel(), yy.ravel()))[:m]
        angle = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        jitter = rng.normal(scale=0.012 if m >= 8 else 0.02, size=base.shape)
        return 0.5 + base @ rot.T + jitter, 0
    if family == "two_cluster":
        # Construct each cluster on a staggered lattice, with continuous jitter
        # and rotation. No pad radius or clearance is modified on rejection.
        left = (m + 1) // 2
        right = m - left
        out = []
        for count, cx in ((left, 0.27), (right, 0.73)):
            if count == 0:
                continue
            cols = 2
            rows = int(np.ceil(count / cols))
            spacing = 0.20
            grid = []
            for k in range(count):
                iy, ix = divmod(k, cols)
                grid.append(((ix - (cols - 1) / 2) * spacing,
                             (iy - (rows - 1) / 2) * spacing))
            center = np.array([0.24 if cx < 0.5 else 0.76, 0.5 + rng.uniform(-0.008, 0.008)])
            angle = rng.uniform(-0.08, 0.08)
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            out.extend(center + np.asarray(grid) @ rot.T + rng.normal(scale=0.002, size=(count, 2)))
        return np.asarray(out), 0
    raise ValueError(f"unknown layout family: {family}")


def generate_layout(seed: int, m: int, family: str, voltage_family: str = "balanced") -> dict[str, Any]:
    if m not in (2, 3, 4, 6, 8, 10, 12, 16):
        raise ValueError("unsupported contact count")
    rng = _rng(seed, "geometry")
    radii = rng.uniform(0.06, 0.09, size=m)
    gamma = np.exp(rng.uniform(np.log(1), np.log(10), size=m))
    voltage_rng = _rng(seed, "voltage")
    if voltage_family in ("balanced", "two_level") or m == 2:
        signs = voltage_rng.integers(0, 2, size=m)
        signs[0], signs[1] = 0, 1
        voltage_rng.shuffle(signs)
        voltage = signs.astype(float) - 0.5
    elif voltage_family == "multilevel":
        voltage = voltage_rng.uniform(-0.4, 0.4, size=m)
        anchors = voltage_rng.choice(m, size=2, replace=False)
        voltage[anchors] = (-0.5, 0.5)
    else:
        raise ValueError(f"unknown voltage family: {voltage_family}")
    voltage -= 0.5 * (voltage.min() + voltage.max())
    point_rejections = 0
    for attempt in range(1, 10001):
        centers, rejected = _proposal_centers(rng, m, family, attempt, radii)
        point_rejections += rejected
        contacts = [{"id": f"pad-{i:02d}", "x": float(centers[i, 0]), "y": float(centers[i, 1]),
                     "r": float(radii[i]), "gamma": float(gamma[i]), "u": float(voltage[i])}
                    for i in range(m)]
        try:
            gap = validate_contacts(contacts)
            return {"family": family, "root_id": f"layout-{seed:010d}", "seed": int(seed),
                    "voltage_family": voltage_family, "contacts": contacts,
                    "proposal_count": attempt, "rejected_proposals": attempt - 1,
                    "rejected_point_proposals": point_rejections,
                    "minimum_pairwise_gap": gap}
        except ValueError:
            continue
    raise RuntimeError(f"failed to generate legal {family} M{m} layout in 10000 proposals")
