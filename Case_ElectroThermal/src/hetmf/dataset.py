"""Read-only pilot loader with input/target separation."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .contacts import rasterize_contacts
from .environment import control_records
from .environment import render_environment
from .processing import FIELD_ORDER


class PilotDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.plan = json.loads((self.root / "inputs" / "pilot_plan.json").read_text())
        self.scenes = {scene["state_id"]: scene for scene in self.plan["scenes"]}

    def inputs(self, state_id: str) -> dict:
        scene = self.scenes[state_id]
        contacts = [{key: contact[key] for key in ("id", "x", "y", "r", "gamma", "u")}
                    for contact in scene["contacts"]]
        return {"environment_records": control_records(scene["environment"]),
                "contacts": contacts, "renderer": "log_bilinear_centers_border_v1",
                "units": "dimensionless"}

    def fields(self, state_id: str, grid: int) -> np.ndarray:
        with h5py.File(self.root / "processed" / f"{state_id}_n{grid}.h5") as handle:
            if handle.attrs["field_order"] != ",".join(FIELD_ORDER):
                raise RuntimeError("field order mismatch")
            return handle["fields"][:]

    def query(self, state_id: str, grid: int, cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fields = self.fields(state_id, grid)
        ids = np.asarray(cell_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= grid * grid):
            raise IndexError("native query ID outside grid")
        y, x = np.divmod(ids, grid)
        xy = np.stack(((x + 0.5) / grid, (y + 0.5) / grid), axis=-1)
        return xy, fields[y, x]


def grid_baseline_inputs(scene: dict, grid: int, quadrature_order: int = 4) -> np.ndarray:
    """Complete input-only raster channels [s,kappa,beta,a,b,d]."""
    s, kappa, beta = render_environment(scene["environment"], grid, grid)
    chi, _ = rasterize_contacts(scene["contacts"], grid, grid, quadrature_order)
    gamma = np.asarray([c["gamma"] for c in scene["contacts"]], dtype=np.float64)
    u = np.asarray([c["u"] for c in scene["contacts"]], dtype=np.float64)
    contact_density = gamma[:, None, None] * chi
    a = contact_density.sum(axis=0)
    b = np.einsum("m,mij->ij", u, contact_density)
    d = np.einsum("m,mij->ij", u**2, contact_density)
    return np.stack((s, kappa, beta, a, b, d), axis=-1)
