"""Scene-wide organization evidence and small, dataset-independent plot primitives.

This module reads prepared arrays only. It never loads a model, query weights,
training targets, or a checkpoint. All member columns are packed to active M + E.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np


@dataclass
class OrganizationScene:
    name: str
    source: Path
    metadata: dict
    m_ids: tuple[str, ...]
    e_ids: tuple[str, ...]
    module_xy: np.ndarray
    environment_xy: np.ndarray
    C: np.ndarray
    gates: dict[str, np.ndarray]
    provenance: dict[str, str]
    module_radius: np.ndarray | None
    polygons: np.ndarray | None
    bounds: tuple[float, float, float, float]
    directory_key: str

    @property
    def n_m(self) -> int:
        return len(self.m_ids)

    @property
    def n_e(self) -> int:
        return len(self.e_ids)

    @property
    def member_labels(self) -> tuple[str, ...]:
        return self.m_ids + self.e_ids

    @property
    def own_columns(self) -> np.ndarray:
        return self.n_m + np.arange(self.n_e)

    @property
    def shared(self) -> np.ndarray:
        result = self.C.copy()
        result[np.arange(self.n_e), self.own_columns] = 0
        return result

    def identity(self) -> dict:
        return {"label": self.name, "source_npz": str(self.source),
                "case": self.metadata.get("case"), "epoch": self.metadata.get("epoch"),
                "partition": self.metadata.get("partition"),
                "checkpoint": self.metadata.get("checkpoint"),
                "directory_key": self.directory_key, "stage_provenance": self.provenance,
                "directory_provenance": self.metadata.get("route_provenance", self.metadata.get("directory_provenance")),
                "coordinate_units": self.metadata.get("coordinate_units", "source geometry units"),
                "projection": self.metadata.get("projection")}


@dataclass
class FigureArtifact:
    stem: str
    title: str
    caption: str
    evidence: dict


def _edges(values: np.ndarray) -> np.ndarray:
    mid = (values[:-1] + values[1:]) / 2
    return np.r_[values[0] - (mid[0] - values[0]), mid,
                 values[-1] + (values[-1] - mid[-1])]


def load_scene(name: str, source: Path, *, directory_key: str = "C_restricted",
               cycle: int | None = None) -> OrganizationScene:
    source = Path(source).resolve()
    metadata = json.loads(source.with_suffix(".json").read_text())
    with np.load(source, allow_pickle=False) as arrays:
        raw_m = np.asarray(arrays["module_xy"], dtype=float)
        e = np.asarray(arrays["environment_xy"], dtype=float)
        mask = np.asarray(arrays["module_mask"], dtype=bool)
        active = np.flatnonzero(mask)
        n_slots, ne = len(raw_m), len(e)
        if raw_m.shape != (n_slots, 2) or e.shape != (ne, 2) or mask.shape != (n_slots,):
            raise ValueError("Provide adapter-owned two-dimensional physical coordinates and an M-slot mask.")
        raw_m_ids = arrays["module_ids"].tolist()
        raw_e_ids = arrays["environment_ids"].tolist()
        m_ids = tuple(f"M{raw_m_ids[i]}" for i in active)
        e_ids = tuple(f"E{i}" for i in raw_e_ids)
        if len(set(m_ids)) != len(m_ids) or len(set(e_ids)) != ne:
            raise ValueError("Original member IDs must be unique within each role.")
        raw_c = np.asarray(arrays[directory_key], dtype=float)
        if raw_c.shape != (ne, n_slots + ne):
            raise ValueError("Directory rows must follow environment_ids; columns must be M slots followed by E IDs.")
        C = raw_c[:, np.r_[active, n_slots + np.arange(ne)]]
        if not np.isfinite(C).all() or (C < 0).any():
            raise ValueError("Directory weights must be finite and nonnegative.")
        if not np.isfinite(raw_m).all() or not np.isfinite(e).all() or ne == 0:
            raise ValueError("Coordinates must be finite and E addresses nonempty.")
        cycles = sorted({int(match.group(1)) for key in arrays.files
                         if (match := re.fullmatch(r"context_cycle(\d+)_(?:mm|me|em|ee)", key))})
        selected_cycle = cycle if cycle is not None else (cycles[-1] if cycles else None)
        gates, provenance = {}, {}
        for route in ("mm", "em", "ee", "me"):
            key = f"context_cycle{selected_cycle}_{route}"
            if key not in arrays:
                continue
            g = np.asarray(arrays[key], dtype=float)
            receivers = active if route[0] == "m" else np.arange(ne)
            donors = active if route[1] == "m" else np.arange(ne)
            expected = (n_slots if route[0] == "m" else ne,
                        n_slots if route[1] == "m" else ne)
            if g.shape != expected or not np.isfinite(g).all() or (g < 0).any():
                raise ValueError(f"Invalid actual contextual gate {key}.")
            gates[route] = g[np.ix_(receivers, donors)]
            provenance[route] = f"cycle{selected_cycle}/{route}"
        radius = None
        if "radius" in arrays:
            r = np.asarray(arrays["radius"], dtype=float)
            radius = np.full(len(active), float(r)) if r.ndim == 0 else r[active]
        polygons = np.asarray(arrays["environment_polygons"], dtype=float) if "environment_polygons" in arrays else None
        if polygons is not None and (polygons.ndim != 3 or polygons.shape[0] != ne or polygons.shape[-1] != 2):
            raise ValueError("environment_polygons must be [E, polygon vertices, 2].")
        if "domain_bounds" in arrays:
            bounds = tuple(float(x) for x in arrays["domain_bounds"])
        else:
            xs, ys = np.unique(e[:, 0]), np.unique(e[:, 1])
            if len(xs) > 1 and len(ys) > 1 and len(xs) * len(ys) == ne and len(np.unique(e, axis=0)) == ne:
                xx, yy = _edges(xs), _edges(ys)
                bounds = (float(xx[0]), float(xx[-1]), float(yy[0]), float(yy[-1]))
            else:
                points = np.concatenate([e, raw_m[active]], axis=0)
                lo, hi = points.min(0), points.max(0)
                pad = np.maximum(hi - lo, 1) * .05
                bounds = (float(lo[0]-pad[0]), float(hi[0]+pad[0]), float(lo[1]-pad[1]), float(hi[1]+pad[1]))
    return OrganizationScene(name, source, metadata, m_ids, e_ids, raw_m[active], e,
                             C, gates, provenance, radius, polygons, bounds, directory_key)


def require_matched(scenes: list[OrganizationScene]) -> None:
    if not scenes:
        raise ValueError("Supply at least one scene.")
    reference = scenes[0]
    for s in scenes[1:]:
        if s.m_ids != reference.m_ids or s.e_ids != reference.e_ids:
            raise ValueError("Comparative plots require aligned original M/E IDs; align exports in the adapter.")
        if not np.array_equal(s.module_xy, reference.module_xy) or not np.array_equal(s.environment_xy, reference.environment_xy):
            raise ValueError("This comparison requires the same original layout; use separate reports for different layouts.")


def apply_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#23343F", "axes.labelcolor": "#23343F",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def save_figure(fig, output: Path, stem: str, title: str, caption: str,
                evidence: dict) -> FigureArtifact:
    import matplotlib.pyplot as plt
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf")
    fig.savefig(folder / f"{stem}.png", dpi=150)
    plt.close(fig)
    return FigureArtifact(stem, title, caption, evidence)


def plot_cells(ax, scene: OrganizationScene, values: np.ndarray, **kwargs):
    """Use supplied polygons, an inferred complete Cartesian grid, or original points."""
    from matplotlib.collections import PolyCollection
    values = np.ma.asarray(values)
    if scene.polygons is not None:
        vmin, vmax = kwargs.pop("vmin", None), kwargs.pop("vmax", None)
        collection = PolyCollection(scene.polygons, array=values, **kwargs)
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        return collection
    xy = scene.environment_xy
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if len(xs) > 1 and len(ys) > 1 and len(xs)*len(ys) == scene.n_e and len(np.unique(xy, axis=0)) == scene.n_e:
        order = np.lexsort((xy[:, 0], xy[:, 1]))
        return ax.pcolormesh(_edges(xs), _edges(ys), values[order].reshape(len(ys), len(xs)),
                             shading="flat", **kwargs)
    return ax.scatter(*xy.T, c=values, marker="o", s=100, **kwargs)


def decorate_domain(ax, scene: OrganizationScene, *, labels: bool = False) -> None:
    from matplotlib.patches import Circle
    for i, xy in enumerate(scene.module_xy):
        if scene.module_radius is not None:
            ax.add_patch(Circle(xy, scene.module_radius[i], facecolor=(1, 1, 1, .75),
                                edgecolor="#7B8790", linewidth=.8, zorder=3))
        else:
            ax.scatter(*xy, s=65, marker="s", facecolor="white", edgecolor="#7B8790", zorder=3)
        if labels:
            ax.text(*xy, scene.m_ids[i], ha="center", va="center", fontsize=8, zorder=5)
    x0, x1, y0, y1 = scene.bounds
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="x", ylabel="y")
    ax.set_aspect("equal")


def member_colors(labels: list[str] | tuple[str, ...]) -> dict[str, tuple]:
    from matplotlib.colors import to_rgb
    base = ["#2878B5", "#E87913", "#CB3B46", "#169E75", "#9A6EC0", "#B6A233",
            "#38A6A5", "#7851A9", "#8EBED9", "#D08BA1", "#8C564B", "#68747E",
            "#E9C638", "#7F9E48", "#C47D4C", "#576ABD", "#C4618D", "#639C8A",
            "#A692BC", "#A6A39A"]
    if len(labels) > len(base):
        import matplotlib.pyplot as plt
        return {label: plt.get_cmap("turbo")(i/max(1, len(labels)-1)) for i, label in enumerate(labels)}
    return {label: to_rgb(base[i]) for i, label in enumerate(labels)}
