"""Deterministic, memory-bounded visual showcases for full-volume runs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

import numpy as np

from .io import WindFarmDataset

FieldName = Literal["speed", "ux", "uy", "uz"]
FIELD_LABELS: dict[str, str] = {
    "speed": "Velocity magnitude [m/s]",
    "ux": "Ux [m/s]",
    "uy": "Uy [m/s]",
    "uz": "Uz [m/s]",
}


def select_representative_indices(dataset: WindFarmDataset, count: int = 3) -> list[int]:
    """Return deterministic representative row indices.

    The default set is the first, middle, and last source rows.  This is
    intentionally explicit and stable: the source ordering covers the first
    and last layout families and all three wind directions.  For a smaller
    requested count, evenly spaced source rows are selected.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if dataset.n_runs == 0:
        return []
    if count == 1:
        return [0]
    if count == 2:
        return [0, dataset.n_runs - 1]
    if count == 3:
        return [0, dataset.n_runs // 2, dataset.n_runs - 1]
    selected = np.linspace(0, dataset.n_runs - 1, count, dtype=np.int64)
    return sorted({int(index) for index in selected})


def _resolve_axis_target(values: np.ndarray, target: float) -> int:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0:
        raise ValueError("Cannot select a cut from an empty coordinate axis")
    return int(np.nanargmin(np.abs(finite - float(target))))


def _stride(length: int, max_points: int) -> int:
    return max(1, math.ceil(length / max(2, max_points)))


def _scalar(values: np.ndarray, field: FieldName) -> np.ndarray:
    if field == "speed":
        return np.sqrt(np.sum(np.asarray(values, dtype=np.float32) ** 2, axis=-1))
    return np.asarray(values)[..., {"ux": 0, "uy": 1, "uz": 2}[field]]


def _finite_range(values: list[np.ndarray], field: FieldName) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return -1.0, 1.0
    if field in ("uy", "uz"):
        limit = float(np.percentile(np.abs(flat), 99.0))
        limit = max(limit, 1e-8)
        return -limit, limit
    low, high = np.percentile(flat, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(flat)), float(np.max(flat))
    if high <= low:
        high = low + 1e-8
    return float(low), float(high)


def _safe_case_name(case: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case).strip("_") or "case"


def render_case_showcase(
    dataset: WindFarmDataset,
    index: int,
    output: str | Path,
    *,
    field: FieldName = "speed",
    x_cut_m: float = 0.0,
    y_cut_m: float = 0.0,
    z_cut_m: float = 70.0,
    max_points: int = 180,
    dpi: int = 180,
    turbine_xy_m: np.ndarray | None = None,
    turbine_hub_height_m: float = 70.0,
    color_limits: tuple[float, float] | None = None,
    camera: tuple[float, float] = (25.0, -55.0),
) -> Path:
    """Render hub, vertical, and oblique 3-D cut-plane views for one case.

    Only the selected run is addressed in the memory-mapped arrays.  The hub
    plane is the nearest available ``z`` to 70 m (or ``z_cut_m``), the two
    vertical cuts are nearest to ``x_cut_m`` and ``y_cut_m``, and the 3-D panel
    uses an oblique camera angle so all three cut planes remain visible.
    """

    if field not in FIELD_LABELS:
        raise ValueError(f"field must be one of {tuple(FIELD_LABELS)}, got {field!r}")
    run = dataset.run(int(index))
    u = run.U_structured
    ix = _resolve_axis_target(run.x_m, x_cut_m)
    iy = _resolve_axis_target(run.y_m, y_cut_m)
    iz = _resolve_axis_target(run.z_m, z_cut_m)
    sx = _stride(run.nx, max_points)
    sy = _stride(run.ny, max_points)
    sz = _stride(run.nz, max_points)

    xs, ys, zs = run.x_m[::sx], run.y_m[::sy], run.z_m[::sz]
    # These are the only scalar surfaces needed for all three panels.  No
    # complete scalar volume is materialized.
    hub = _scalar(u[iz, ::sy, ::sx, :], field)
    vertical_x = _scalar(u[::sz, iy, ::sx, :], field)
    vertical_y = _scalar(u[::sz, ::sy, ix, :], field)
    scalar_surfaces = [hub, vertical_x, vertical_y]
    vmin, vmax = _finite_range(scalar_surfaces, field) if color_limits is None else color_limits
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("color_limits must be finite and strictly increasing")

    turbines: np.ndarray | None = None
    if turbine_xy_m is not None:
        candidate = np.asarray(turbine_xy_m, dtype=np.float64)
        if candidate.ndim != 2 or candidate.shape[1] != 2:
            raise ValueError("turbine_xy_m must have shape (n_turbines, 2)")
        turbines = candidate[np.all(np.isfinite(candidate), axis=1)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable

    if field in ("uy", "uz"):
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")
    elif field == "ux":
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(16, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.25))
    ax_hub = fig.add_subplot(grid[0, 0])
    ax_vertical = fig.add_subplot(grid[0, 1])
    ax_3d = fig.add_subplot(grid[0, 2], projection="3d")

    ax_hub.pcolormesh(xs, ys, hub, shading="auto", cmap=cmap, norm=norm)
    ax_hub.axvline(run.x_m[ix], color="white", linestyle="--", linewidth=0.8)
    ax_hub.axhline(run.y_m[iy], color="white", linestyle="--", linewidth=0.8)
    if turbines is not None and len(turbines):
        ax_hub.scatter(
            turbines[:, 0],
            turbines[:, 1],
            s=20,
            marker="o",
            facecolor="#ff4d4d",
            edgecolor="white",
            linewidth=0.45,
            label=f"turbines (n={len(turbines)})",
            zorder=4,
        )
        ax_hub.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_hub.set_aspect("equal", adjustable="box")
    ax_hub.set(
        title=f"Hub plane: z={run.z_m[iz]:.1f} m",
        xlabel="x [m] (streamwise)",
        ylabel="y [m] (crosswind)",
    )

    ax_vertical.pcolormesh(xs, zs, vertical_x, shading="auto", cmap=cmap, norm=norm)
    ax_vertical.axhline(run.z_m[iz], color="white", linestyle="--", linewidth=0.8)
    ax_vertical.set(
        title=f"Vertical cut: y={run.y_m[iy]:.1f} m",
        xlabel="x [m] (streamwise)",
        ylabel="z [m]",
    )

    # Full-volume showcase: three mutually orthogonal cut planes in a single
    # oblique 3-D view.  The hub and vertical cuts intentionally share the
    # same normalisation as the 3-D surfaces.
    Xxy, Yxy = np.meshgrid(xs, ys, indexing="xy")
    Xxz, Zxz = np.meshgrid(xs, zs, indexing="xy")
    Yyz, Zyz = np.meshgrid(ys, zs, indexing="xy")
    ax_3d.plot_surface(
        Xxy,
        Yxy,
        np.full_like(Xxy, run.z_m[iz]),
        facecolors=cmap(norm(hub)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.90,
    )
    ax_3d.plot_surface(
        Xxz,
        np.full_like(Xxz, run.y_m[iy]),
        Zxz,
        facecolors=cmap(norm(vertical_x)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.82,
    )
    ax_3d.plot_surface(
        np.full_like(Yyz, run.x_m[ix]),
        Yyz,
        Yyz * 0.0 + Zyz,
        facecolors=cmap(norm(vertical_y)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.76,
    )
    ax_3d.set_box_aspect((max(np.ptp(xs), 1.0), max(np.ptp(ys), 1.0), max(np.ptp(zs), 1.0)))
    if turbines is not None and len(turbines):
        ax_3d.scatter(
            turbines[:, 0],
            turbines[:, 1],
            np.full(len(turbines), float(turbine_hub_height_m)),
            s=8,
            c="#ff4d4d",
            depthshade=False,
        )
    ax_3d.view_init(elev=float(camera[0]), azim=float(camera[1]))
    ax_3d.set(
        title=(
            "Oblique 3-D cuts\n"
            f"x={run.x_m[ix]:.1f}, y={run.y_m[iy]:.1f}, z={run.z_m[iz]:.1f} m"
        ),
    )
    ax_3d.tick_params(axis="z", labelsize=7, pad=0)

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.asarray(hub).reshape(-1))
    colorbar = fig.colorbar(mappable, ax=(ax_hub, ax_vertical, ax_3d), shrink=0.86, pad=0.065)
    colorbar.set_label(FIELD_LABELS[field])
    fig.suptitle(
        f"Wind-farm full-volume showcase — {run.case} | layout {run.layout_index} | "
        f"wind direction {run.wind_direction_deg:g}°",
        fontsize=14,
        fontweight="semibold",
    )

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    return destination


def default_figure_path(output_dir: str | Path, dataset: WindFarmDataset, index: int) -> Path:
    """Return the stable figure filename used by the inspection workspace."""

    run = dataset.run(index)
    return Path(output_dir).expanduser().resolve() / (f"windfarm_case_{int(index):04d}_{_safe_case_name(run.case)}.png")
