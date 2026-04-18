"""
Visualize and post-process a saved multi-cylinder case directory.

The script reads the scene-compatible `.npz` arrays produced by the simulation
script, generates selected frame plots, optionally writes a GIF, and extracts a
small set of physically meaningful quantities of interest (QoIs).

python src/visualize_multicylinder_case.py --case_dir 0001 --save-gif

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from multicyl_common import (
    SimulationConfig,
    build_uniform_grid,
    config_from_dict,
    default_data_dir,
    periodic_offsets,
    resolve_data_path,
)


def resolve_case_dir(case_arg: str) -> Path:
    """Resolve a case argument to an actual case directory.

    Accepted forms:
    - bare case id, e.g. `0001`
    - case directory name, e.g. `case_0001_20260417_142729_multicyl`
    - path relative to `Data_Saved/`
    - absolute path
    """
    direct_path = resolve_data_path(case_arg)
    if (direct_path / "case_config.json").exists():
        return direct_path

    case_id = case_arg.strip()
    if case_id.isdigit():
        data_root = default_data_dir().resolve()
        matches = sorted(data_root.glob(f"case_{case_id}_*"))
        matches = [path for path in matches if (path / "case_config.json").exists()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple case directories matched case id {case_id}: "
                + ", ".join(path.name for path in matches)
            )

    raise FileNotFoundError(
        f"Could not resolve case '{case_arg}'. "
        "Pass a case id like '0001', a full case directory name, or an absolute path."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved multi-cylinder case.")
    parser.add_argument(
        "--case_dir",
        type=str,
        required=True,
        help=f"Case directory or case name under {default_data_dir()}.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated saved-frame indices to plot. Default: first, 25%%, 50%%, last.",
    )
    parser.add_argument("--save-gif", action="store_true", help="Render a GIF over all saved frames.")
    parser.add_argument(
        "--gif-field",
        choices=["vorticity", "speed", "temperature", "pressure"],
        default=None,
        help="Field to animate. Default: vorticity for inert, temperature for active.",
    )
    parser.add_argument("--gif-dpi", type=int, default=90, help="DPI used when rendering GIF frames.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for GIF output.")
    return parser.parse_args()


def load_case_config(case_dir: Path) -> SimulationConfig:
    with (case_dir / "case_config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_cfg = {k: v for k, v in raw.items() if k in {"mode", "domain", "flow", "thermal", "layout", "save"}}
    return config_from_dict(raw_cfg)


def load_frame_index(case_dir: Path) -> pd.DataFrame:
    csv_path = case_dir / "frame_index.csv"
    return pd.read_csv(csv_path)


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        last_key = list(data.keys())[-1]
        return np.array(data[last_key])


def _candidate_field_names(name: str) -> List[str]:
    return [name, name.lower(), name.capitalize()]


def _find_existing_field_path(scene_dir: Path, name: str, frame: int) -> Optional[Path]:
    for candidate in _candidate_field_names(name):
        path = scene_dir / f"{candidate}_{frame:06d}.npz"
        if path.exists():
            return path
    return None


def list_saved_frames(scene_dir: Path, property_name: str = "velocity") -> List[int]:
    files: List[Path] = []
    for candidate in _candidate_field_names(property_name):
        files = sorted(scene_dir.glob(f"{candidate}_*.npz"))
        if files:
            break
    return [int(f.stem.split("_")[-1]) for f in files]


def _normalize_scalar_field(arr: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Convert saved scalar fields to plotting/QoI shape (ny, nx)."""
    nx, ny = cfg.domain.nx, cfg.domain.ny

    if arr.shape == (ny, nx):
        return arr
    if arr.shape == (nx, ny):
        return arr.T
    if arr.shape == (ny + 1, nx + 1):
        return arr[:-1, :-1]
    if arr.shape == (nx + 1, ny + 1):
        return arr[:-1, :-1].T

    raise ValueError(
        f"Unsupported scalar field shape {arr.shape}. "
        f"Expected one of {(ny, nx)}, {(nx, ny)}, {(ny + 1, nx + 1)}, {(nx + 1, ny + 1)}."
    )


def _normalize_vector_field(arr: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Convert saved vector fields to plotting/QoI shape (ny, nx, 2)."""
    nx, ny = cfg.domain.nx, cfg.domain.ny

    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Unsupported vector field shape {arr.shape}. Expected a trailing vector dimension of size 2.")
    if arr.shape[:2] == (ny, nx):
        return arr
    if arr.shape[:2] == (nx, ny):
        return np.transpose(arr, (1, 0, 2))

    raise ValueError(f"Unsupported vector field shape {arr.shape}. Expected {(ny, nx, 2)} or {(nx, ny, 2)}.")


def load_fields_for_frame(scene_dir: Path, frame: int, cfg: SimulationConfig) -> Dict[str, Optional[np.ndarray]]:
    def maybe(name: str) -> Optional[np.ndarray]:
        path = _find_existing_field_path(scene_dir, name, frame)
        return _load_npz(path) if path is not None else None

    velocity = maybe("velocity")
    pressure = maybe("pressure")
    vorticity = maybe("vorticity")
    temperature = maybe("temperature")

    return {
        "velocity": _normalize_vector_field(velocity, cfg) if velocity is not None else None,
        "pressure": _normalize_scalar_field(pressure, cfg) if pressure is not None else None,
        "vorticity": _normalize_scalar_field(vorticity, cfg) if vorticity is not None else None,
        "temperature": _normalize_scalar_field(temperature, cfg) if temperature is not None else None,
    }


def choose_default_frames(frame_ids: Sequence[int]) -> List[int]:
    if not frame_ids:
        raise ValueError("No saved frames found.")
    idxs = [0, int(round(0.25 * (len(frame_ids) - 1))), int(round(0.5 * (len(frame_ids) - 1))), len(frame_ids) - 1]
    return sorted({frame_ids[i] for i in idxs})


def parse_frame_request(frame_ids: Sequence[int], frame_arg: Optional[str]) -> List[int]:
    if frame_arg is None:
        return choose_default_frames(frame_ids)
    requested = sorted({int(v.strip()) for v in frame_arg.split(",") if v.strip()})
    missing = [i for i in requested if i not in frame_ids]
    if missing:
        raise ValueError(f"Requested saved frames do not exist: {missing}")
    return requested


def overlay_cylinders(ax, cfg: SimulationConfig) -> None:
    for cx, cy in cfg.layout.centers or []:
        ax.add_patch(Circle((cx, cy), cfg.domain.cylinder_radius, fill=False, color="k", lw=1.5))


def compute_qois(cfg: SimulationConfig, fields: Dict[str, Optional[np.ndarray]]) -> Dict[str, float]:
    velocity = fields["velocity"]
    if velocity is None:
        raise ValueError("Velocity field is required for QoI extraction.")

    ux = velocity[..., 0]
    uy = velocity[..., 1]
    speed = np.sqrt(ux * ux + uy * uy)
    qoi: Dict[str, float] = {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "reverse_flow_fraction": float(np.mean(ux < 0.0)),
    }

    vort = fields["vorticity"]
    if vort is not None:
        qoi["mean_abs_vorticity"] = float(np.mean(np.abs(vort)))
        qoi["enstrophy"] = float(np.mean(vort * vort))

    temp = fields["temperature"]
    if temp is not None:
        qoi["mean_temperature"] = float(np.mean(temp))
        qoi["max_temperature"] = float(np.max(temp))

    xx, yy = build_uniform_grid(cfg)
    probe_dx = 3.0 * cfg.domain.cylinder_radius
    probe_dy = 2.0 * cfg.domain.cylinder_radius
    shell_width = max(cfg.domain.lx / cfg.domain.nx, cfg.domain.ly / cfg.domain.ny)

    for idx, (cx, cy) in enumerate(cfg.layout.centers or []):
        dx = periodic_offsets(xx - (cx + probe_dx), cfg.domain.lx)
        dy = periodic_offsets(yy - cy, cfg.domain.ly)
        wake_mask = (np.abs(dx) <= probe_dx) & (np.abs(dy) <= probe_dy)
        qoi[f"wake_deficit_cyl_{idx:02d}"] = float(cfg.flow.u_bulk - np.mean(ux[wake_mask]))

        if temp is not None:
            dx0 = periodic_offsets(xx - cx, cfg.domain.lx)
            dy0 = periodic_offsets(yy - cy, cfg.domain.ly)
            rr = np.sqrt(dx0 * dx0 + dy0 * dy0)
            annulus = (rr >= cfg.domain.cylinder_radius) & (rr <= cfg.domain.cylinder_radius + shell_width)
            qoi[f"surface_temperature_proxy_cyl_{idx:02d}"] = float(np.mean(temp[annulus]))

    return qoi


def render_frame(case_dir: Path, cfg: SimulationConfig, frame: int, fields: Dict[str, Optional[np.ndarray]], *, dpi: int = 150) -> Path:
    scene_dir = case_dir / "scene"
    plots_dir = case_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    velocity = fields["velocity"]
    if velocity is None:
        raise ValueError("Velocity field is required for plotting.")
    ux = velocity[..., 0]
    uy = velocity[..., 1]
    speed = np.sqrt(ux * ux + uy * uy)

    arrays: List[np.ndarray] = [speed]
    titles: List[str] = ["Speed"]
    cmaps: List[str] = ["viridis"]

    if fields["vorticity"] is not None:
        arrays.append(fields["vorticity"])
        titles.append("Vorticity")
        cmaps.append("RdBu_r")
    if cfg.thermal.enabled and fields["temperature"] is not None:
        arrays.append(fields["temperature"])
        titles.append("Temperature")
        cmaps.append("inferno")

    fig, axes = plt.subplots(1, len(arrays), figsize=(5.0 * len(arrays), 4.0), constrained_layout=True, dpi=dpi)
    if len(arrays) == 1:
        axes = [axes]

    extent = (0.0, cfg.domain.lx, 0.0, cfg.domain.ly)
    for ax, arr, title, cmap in zip(axes, arrays, titles, cmaps):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        overlay_cylinders(ax, cfg)
        ax.set_title(f"{title} | frame {frame:04d}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = plots_dir / f"frame_{frame:04d}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_gif(case_dir: Path, cfg: SimulationConfig, frame_ids: Sequence[int], gif_field: str, fps: int, dpi: int) -> Path:
    plots_dir = case_dir / "plots"
    gif_path = plots_dir / f"animation_{gif_field}.gif"
    frames_rgba: List[np.ndarray] = []
    extent = (0.0, cfg.domain.lx, 0.0, cfg.domain.ly)

    with tqdm(total=len(frame_ids), desc=f"Rendering GIF ({gif_field})", unit="frame", dynamic_ncols=True) as gif_bar:
        for frame in frame_ids:
            fields = load_fields_for_frame(case_dir / "scene", frame, cfg)
            velocity = fields["velocity"]
            if velocity is None:
                raise ValueError("Velocity field is required for GIF rendering.")
            ux = velocity[..., 0]
            uy = velocity[..., 1]
            speed = np.sqrt(ux * ux + uy * uy)

            if gif_field == "speed":
                arr, cmap, title = speed, "viridis", "Speed"
            elif gif_field == "temperature":
                if fields["temperature"] is None:
                    raise ValueError("Temperature field not found; cannot render temperature GIF.")
                arr, cmap, title = fields["temperature"], "inferno", "Temperature"
            elif gif_field == "pressure":
                if fields["pressure"] is None:
                    raise ValueError("Pressure field not found; cannot render pressure GIF.")
                arr, cmap, title = fields["pressure"], "magma", "Pressure"
            else:
                if fields["vorticity"] is None:
                    raise ValueError("Vorticity field not found; cannot render vorticity GIF.")
                arr, cmap, title = fields["vorticity"], "RdBu_r", "Vorticity"

            fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=dpi, constrained_layout=True)
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, aspect="equal")
            overlay_cylinders(ax, cfg)
            ax.set_title(f"{title} | frame {frame:04d}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba())
            frames_rgba.append(frame_rgba)
            plt.close(fig)
            gif_bar.update(1)

    imageio.mimsave(gif_path, frames_rgba, duration=1.0 / max(fps, 1))
    tqdm.write(f"Saved GIF to: {gif_path}")
    return gif_path


def save_qoi_outputs(case_dir: Path, qoi_rows: List[Dict[str, float]], csv_name: str, plot_timeseries: bool) -> None:
    plots_dir = case_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(qoi_rows)
    df.to_csv(plots_dir / csv_name, index=False)

    if not plot_timeseries or len(df) <= 1:
        return

    numeric_cols = [c for c in df.columns if c not in {"saved_frame", "time"}]
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8.0, max(2.0, 2.2 * len(numeric_cols))), constrained_layout=True)
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        ax.plot(df["time"], df[col], lw=1.8)
        ax.set_xlabel("time")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "qoi_timeseries.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    case_dir = resolve_case_dir(args.case_dir)
    scene_dir = case_dir / "scene"
    cfg = load_case_config(case_dir)
    frame_index = load_frame_index(case_dir)
    frame_ids = list_saved_frames(scene_dir, property_name="velocity")
    selected_frames = parse_frame_request(frame_ids, args.frames)

    selected_rows: List[Dict[str, float]] = []
    for frame in selected_frames:
        fields = load_fields_for_frame(scene_dir, frame, cfg)
        render_frame(case_dir, cfg, frame, fields)
        qois = compute_qois(cfg, fields)
        row = {"saved_frame": frame}
        row["time"] = float(frame_index.loc[frame_index["saved_frame"] == frame, "time"].iloc[0])
        row.update(qois)
        selected_rows.append(row)

    save_qoi_outputs(case_dir, selected_rows, csv_name="qoi_selected_frames.csv", plot_timeseries=False)

    if args.save_gif:
        default_gif_field = "temperature" if cfg.thermal.enabled else "vorticity"
        gif_field = args.gif_field or default_gif_field
        render_gif(case_dir, cfg, frame_ids, gif_field=gif_field, fps=args.fps, dpi=args.gif_dpi)

        full_rows: List[Dict[str, float]] = []
        for frame in frame_ids:
            fields = load_fields_for_frame(scene_dir, frame, cfg)
            qois = compute_qois(cfg, fields)
            row = {"saved_frame": frame}
            row["time"] = float(frame_index.loc[frame_index["saved_frame"] == frame, "time"].iloc[0])
            row.update(qois)
            full_rows.append(row)
        save_qoi_outputs(case_dir, full_rows, csv_name="qoi_all_frames.csv", plot_timeseries=True)


if __name__ == "__main__":
    main()
