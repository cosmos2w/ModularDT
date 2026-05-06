"""Visualize raw and processed global channel thermal cases.

Scope
-----
This script handles the **global channel** visualization layer for Demo 1. It
can read either raw case folders produced by ``simulate_channelthermal.py`` or
the packed HDF5 written by ``preprocess_channelthermal_dataset.py``.

Outputs
-------
The script writes PNG plots under each case or processed dataset ``plots/``
folder by default:

* raw frame fields ``u, v, p, omega, temperature``
* processed ``steady_field`` and ``rms_field``
* module internal temperature grids
* interface target curves ``T_surface(theta)`` and ``q_normal(theta)``
* optional GIF for raw temperature transients

Training role
-------------
These plots are smoke checks for the Stage-B global channel dataset. They keep
using the legacy full ``interface_response`` for visualization, while future
training should prefer the clean ``interface_condition`` and ``interface_target``
datasets.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - only needed for processed views
    h5py = None

from channelthermal_common import config_from_dict, read_json, resolve_data_path


FIELD_NAMES = ("u", "v", "p", "omega", "temperature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw or processed channel thermal cases.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Raw case directory to visualize.")
    parser.add_argument("--processed-h5", type=Path, default=None, help="Processed packed_dataset.h5 to visualize.")
    parser.add_argument("--case-id", type=str, default=None, help="Processed case id/key. Defaults to first case.")
    parser.add_argument("--frame", type=int, default=-1, help="Raw saved frame index; negative selects from the end.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG/GIF outputs.")
    parser.add_argument("--save-gif", action="store_true", help="Save a temperature transient GIF for raw cases.")
    return parser.parse_args()


def latest_raw_case() -> Path:
    """Return the most recent raw global case under ``Data_Saved``."""
    root = resolve_data_path("./Data_Saved")
    candidates = sorted(
        path for path in root.glob("case_*") if path.is_dir() and (path / "scene").exists() and (path / "case_config.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No raw channel thermal cases found under {root}.")
    return candidates[-1]


def resolve_case_dir(path: Path | None) -> Path:
    if path is None:
        return latest_raw_case()
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    direct = (Path.cwd() / expanded).resolve()
    if direct.exists():
        return direct
    return resolve_data_path(expanded)


def read_frame_index(case_dir: Path) -> List[Dict[str, str]]:
    with (case_dir / "frame_index.csv").open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_raw_frame(case_dir: Path, frame_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Load one raw frame payload and its frame-index row."""
    rows = read_frame_index(case_dir)
    if frame_index < 0:
        row = rows[frame_index]
    else:
        row = rows[min(frame_index, len(rows) - 1)]
    file_name = row.get("file") or f"frame_{int(row['saved_frame']):06d}.npz"
    with np.load(case_dir / "scene" / file_name, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    return payload, row


def overlay_modules(ax: plt.Axes, centers: Sequence[Sequence[float]], radius: float) -> None:
    """Draw circular module outlines on global channel plots."""
    for idx, (cx, cy) in enumerate(centers):
        circle = plt.Circle((float(cx), float(cy)), radius, fill=False, color="white", linewidth=1.2)
        ax.add_patch(circle)
        ax.text(float(cx), float(cy), str(idx), color="white", ha="center", va="center", fontsize=8)


def plot_fields(
    fields: Dict[str, np.ndarray],
    cfg_payload: Dict[str, object],
    title: str,
    output_path: Path,
) -> None:
    """Plot a five-channel global field tensor as small multiples."""
    cfg = config_from_dict(cfg_payload)
    extent = [0.0, float(cfg.domain.lx), 0.0, float(cfg.domain.ly)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, name in zip(axes_flat, FIELD_NAMES):
        image = ax.imshow(fields[name], origin="lower", extent=extent, aspect="auto", cmap="viridis")
        overlay_modules(ax, cfg.layout.centers or [], float(cfg.domain.module_radius))
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes_flat[-1].axis("off")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_internal_temperatures(payload: Dict[str, np.ndarray], output_path: Path) -> None:
    """Plot module-local internal temperature targets, ignoring padded modules."""
    internal = payload.get("module_internal_temperature")
    if internal is None or internal.shape[0] == 0:
        return
    present = payload.get("module_present")
    if present is not None:
        count = min(int(np.count_nonzero(present)), int(internal.shape[0]), 8)
    else:
        count = min(int(internal.shape[0]), 8)
    if count == 0:
        return
    cols = min(count, 4)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False, constrained_layout=True)
    for idx in range(rows * cols):
        ax = axes.ravel()[idx]
        if idx >= count:
            ax.axis("off")
            continue
        image = ax.imshow(internal[idx], origin="lower", extent=[-1, 1, -1, 1], cmap="inferno")
        ax.set_title(f"module {idx}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Module Internal Temperature")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_interface_targets(payload: Dict[str, np.ndarray], output_path: Path) -> None:
    """Plot solved interface target curves from the full response array."""
    response = payload.get("interface_response")
    names = payload.get("interface_feature_names")
    if response is None or response.shape[0] == 0 or names is None:
        return
    decoded = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]
    theta_idx = decoded.index("theta")
    temp_idx = decoded.index("T_surface")
    flux_idx = decoded.index("q_normal")
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True, constrained_layout=True)
    for module_idx in range(response.shape[0]):
        axes[0].plot(response[module_idx, :, theta_idx], response[module_idx, :, temp_idx], label=f"module {module_idx}")
        axes[1].plot(response[module_idx, :, theta_idx], response[module_idx, :, flux_idx], label=f"module {module_idx}")
    axes[0].set_ylabel("target: T_surface")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("target: q_normal")
    fig.suptitle("Interface Targets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_temperature_gif(case_dir: Path, cfg_payload: Dict[str, object], output_path: Path) -> None:
    """Save an optional raw-frame temperature transient GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    rows = read_frame_index(case_dir)
    cfg = config_from_dict(cfg_payload)
    extent = [0.0, float(cfg.domain.lx), 0.0, float(cfg.domain.ly)]
    frames = []
    for row in rows:
        file_name = row.get("file") or f"frame_{int(row['saved_frame']):06d}.npz"
        with np.load(case_dir / "scene" / file_name, allow_pickle=False) as data:
            frames.append(data["temperature"].astype(np.float32))
    vmin = min(float(np.min(frame)) for frame in frames)
    vmax = max(float(np.max(frame)) for frame in frames)
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    image = ax.imshow(frames[0], origin="lower", extent=extent, aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax)
    overlay_modules(ax, cfg.layout.centers or [], float(cfg.domain.module_radius))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    def update(frame_idx: int):
        image.set_array(frames[frame_idx])
        ax.set_title(f"temperature frame {frame_idx}")
        return (image,)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=160, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=6))
    plt.close(fig)


def visualize_raw(case_dir_arg: Path | None, frame_index: int, output_dir: Path | None, save_gif: bool) -> None:
    """Visualize one raw global case frame."""
    case_dir = resolve_case_dir(case_dir_arg)
    cfg_payload = read_json(case_dir / "case_config.json")
    payload, row = load_raw_frame(case_dir, frame_index)
    out_dir = output_dir.expanduser().resolve() if output_dir is not None else case_dir / "plots"
    fields = {name: payload[name] for name in FIELD_NAMES}
    frame_id = int(row.get("saved_frame", 0))
    plot_fields(fields, cfg_payload, f"Raw case {case_dir.name}, frame {frame_id}", out_dir / f"raw_frame_{frame_id:06d}_fields.png")
    plot_internal_temperatures(payload, out_dir / f"raw_frame_{frame_id:06d}_internal_temperature.png")
    plot_interface_targets(payload, out_dir / f"raw_frame_{frame_id:06d}_interface_targets.png")
    if save_gif:
        save_temperature_gif(case_dir, cfg_payload, out_dir / "temperature_transient.gif")
    print(f"Saved raw visualizations to: {out_dir}")


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def visualize_processed(processed_h5_arg: Path, case_id: str | None, output_dir: Path | None) -> None:
    """Visualize one processed global HDF5 case."""
    if h5py is None:
        raise ImportError("h5py is required for processed HDF5 visualization.")
    h5_path = processed_h5_arg.expanduser()
    if not h5_path.is_absolute():
        h5_path = resolve_data_path(h5_path)
    h5_path = h5_path.resolve()
    with h5py.File(h5_path, "r") as h5:
        cases_group = h5["cases"]
        key = case_id or sorted(cases_group.keys())[0]
        group = cases_group[key]
        cfg_payload = json.loads(decode_h5_string(group["case_config_json"][()]))
        steady = group["steady_field"][()]
        rms = group["rms_field"][()]
        fields = {name: steady[..., idx] for idx, name in enumerate(FIELD_NAMES)}
        rms_fields = {name: rms[..., idx] for idx, name in enumerate(FIELD_NAMES)}
        payload = {
            "module_internal_temperature": group["module_internal_temperature"][()],
            "module_present": group["module_present"][()],
            "interface_response": group["interface_response"][()],
            "interface_feature_names": h5["interface_feature_names"][()],
        }
    out_dir = output_dir.expanduser().resolve() if output_dir is not None else h5_path.parent / "plots"
    plot_fields(fields, cfg_payload, f"Processed steady case {key}", out_dir / f"processed_{key}_steady_fields.png")
    plot_fields(rms_fields, cfg_payload, f"Processed RMS case {key}", out_dir / f"processed_{key}_rms_fields.png")
    plot_internal_temperatures(payload, out_dir / f"processed_{key}_internal_temperature.png")
    plot_interface_targets(payload, out_dir / f"processed_{key}_interface_targets.png")
    print(f"Saved processed visualizations to: {out_dir}")


def main() -> int:
    args = parse_args()
    if args.case_dir is None and args.processed_h5 is None:
        visualize_raw(None, args.frame, args.output_dir, args.save_gif)
        return 0
    if args.case_dir is not None:
        visualize_raw(args.case_dir, args.frame, args.output_dir, args.save_gif)
    if args.processed_h5 is not None:
        visualize_processed(args.processed_h5, args.case_id, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
