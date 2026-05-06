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

from channelthermal_common import config_from_dict, find_case_dirs, read_json, resolve_data_path


FIELD_NAMES = ("u", "v", "p", "omega", "temperature")


def domain_extent(cfg_payload: Dict[str, object]) -> Tuple[List[float], float, float]:
    """Return imshow extent plus physical domain dimensions."""
    cfg = config_from_dict(cfg_payload)
    lx = float(cfg.domain.lx)
    ly = float(cfg.domain.ly)
    return [0.0, lx, 0.0, ly], lx, ly


def set_domain_aspect(ax: plt.Axes, lx: float, ly: float) -> None:
    """Render physical domain coordinates without stretching."""
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)
    ax.set_aspect("equal", adjustable="box")


def single_domain_figsize(lx: float, ly: float, width: float = 8.5) -> Tuple[float, float]:
    """Choose a figure size that follows the channel's physical aspect ratio."""
    height = max(2.4, width * ly / max(lx, 1e-12) + 1.0)
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw or processed channel thermal cases.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Raw case directory to visualize.")
    parser.add_argument("--processed-h5", type=Path, default=None, help="Processed packed_dataset.h5 to visualize.")
    parser.add_argument("--case-id", type=str, default=None, help="Processed case id/key. Defaults to first case.")
    parser.add_argument("--frame", type=int, default=-1, help="Raw saved frame index; negative selects from the end.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG/GIF outputs.")
    parser.add_argument("--save-gif", action="store_true", help="Save a temperature transient GIF for raw cases.")
    return parser.parse_args()


def raw_case_candidates() -> List[Path]:
    """Return all raw global channel case directories in deterministic order."""
    root = resolve_data_path("./Data_Saved")
    return sorted(
        path
        for _split, path in find_case_dirs(root)
        if (path / "scene").exists() and (path / "frame_index.csv").exists()
    )


def latest_raw_case() -> Path:
    """Return the most recent raw global case under ``Data_Saved``."""
    root = resolve_data_path("./Data_Saved")
    candidates = raw_case_candidates()
    if not candidates:
        raise FileNotFoundError(f"No raw channel thermal cases found under {root}.")
    return candidates[-1]


def normalize_case_id(case_id: str) -> str:
    """Normalize short numeric IDs to the global batch case-id convention."""
    cleaned = str(case_id).strip()
    if cleaned.isdigit():
        return f"{int(cleaned):04d}"
    if cleaned.startswith("case_"):
        parts = cleaned.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"{int(parts[1]):04d}"
    return cleaned


def resolve_raw_case_id(case_id: str) -> Path:
    """Resolve a raw global case by config save.case_id or directory name.

    Examples that resolve to the same case:
        ``1``, ``0001``, ``case_0001_20260506_161815_channelthermal``.
    """
    candidates = raw_case_candidates()
    if not candidates:
        raise FileNotFoundError(f"No raw channel thermal cases found under {resolve_data_path('./Data_Saved')}.")

    normalized = normalize_case_id(case_id)
    matches: List[Path] = []
    for path in candidates:
        cfg_payload = read_json(path / "case_config.json")
        saved_case_id = str(cfg_payload.get("save", {}).get("case_id", ""))
        if saved_case_id == normalized or path.name == case_id or path.name.startswith(f"case_{normalized}_"):
            matches.append(path)

    if not matches:
        available = [str(read_json(path / "case_config.json").get("save", {}).get("case_id", path.name)) for path in candidates[-20:]]
        raise FileNotFoundError(
            f"No raw channel thermal case matched --case-id {case_id!r} (normalized to {normalized!r}). "
            f"Recent available IDs: {available}"
        )
    return matches[-1]


def resolve_case_dir(path: Path | None, case_id: str | None = None) -> Path:
    if path is None:
        if case_id is not None:
            return resolve_raw_case_id(case_id)
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


def row_float(row: Dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def plot_convergence_history(
    frame_rows: Sequence[Dict[str, str]],
    cfg_payload: Dict[str, object],
    output_path: Path,
    selected_times: np.ndarray | None = None,
) -> None:
    """Plot saved-frame temperature convergence diagnostics."""
    if not frame_rows:
        return
    times = np.asarray([row_float(row, "time") for row in frame_rows], dtype=np.float64)
    delta_inf = np.asarray([row_float(row, "delta_inf") for row in frame_rows], dtype=np.float64)
    delta_l2_rel = np.asarray([row_float(row, "delta_l2_rel") for row in frame_rows], dtype=np.float64)
    max_temperature = np.asarray([row_float(row, "max_temperature") for row in frame_rows], dtype=np.float64)
    mean_temperature = np.asarray([row_float(row, "mean_temperature") for row in frame_rows], dtype=np.float64)
    if not np.any(np.isfinite(times)):
        return
    runtime = cfg_payload.get("runtime", {}) if isinstance(cfg_payload, dict) else {}
    converged_time = runtime.get("converged_time") if isinstance(runtime, dict) else None
    try:
        converged_time = float(converged_time) if converged_time is not None else np.nan
    except (TypeError, ValueError):
        converged_time = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    series = [
        (delta_inf, "delta_inf", True),
        (delta_l2_rel, "delta_l2_rel", True),
        (max_temperature, "max temperature", False),
        (mean_temperature, "mean temperature", False),
    ]
    for ax, (values, label, use_log) in zip(axes.ravel(), series):
        finite = np.isfinite(times) & np.isfinite(values)
        if np.any(finite):
            ax.plot(times[finite], values[finite], marker="o", linewidth=1.2)
        if use_log and np.any(values[finite] > 0.0):
            ax.set_yscale("log")
        if np.isfinite(converged_time):
            ax.axvline(converged_time, color="tab:green", linestyle="--", linewidth=1.1, label="converged")
        if selected_times is not None and len(selected_times) > 0:
            lo = float(np.min(selected_times))
            hi = float(np.max(selected_times))
            ax.axvspan(lo, hi, color="tab:orange", alpha=0.18, label="selected")
        ax.set_xlabel("time")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        if np.isfinite(converged_time) or (selected_times is not None and len(selected_times) > 0):
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("Temperature Convergence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_processed_metadata(metadata: Dict[str, object], output_path: Path) -> None:
    """Save a compact text panel for processed convergence metadata."""
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    ax.axis("off")
    lines = [
        f"case_id: {metadata.get('case_id', '')}",
        f"target_mode: {metadata.get('target_mode', '')}",
        f"converged: {metadata.get('converged', '')}",
        f"selected_times: {metadata.get('selected_times', '')}",
        f"final_delta_inf: {metadata.get('final_delta_inf', '')}",
        f"final_delta_l2_rel: {metadata.get('final_delta_l2_rel', '')}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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
    extent, lx, ly = domain_extent(cfg_payload)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, name in zip(axes_flat, FIELD_NAMES):
        image = ax.imshow(fields[name], origin="lower", extent=extent, aspect="equal", cmap="viridis")
        overlay_modules(ax, cfg.layout.centers or [], float(cfg.domain.module_radius))
        set_domain_aspect(ax, lx, ly)
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
        image = ax.imshow(internal[idx], origin="lower", extent=[-1, 1, -1, 1], aspect="equal", cmap="inferno")
        ax.set_aspect("equal", adjustable="box")
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
    extent, lx, ly = domain_extent(cfg_payload)
    frames = []
    for row in rows:
        file_name = row.get("file") or f"frame_{int(row['saved_frame']):06d}.npz"
        with np.load(case_dir / "scene" / file_name, allow_pickle=False) as data:
            frames.append(data["temperature"].astype(np.float32))
    vmin = min(float(np.min(frame)) for frame in frames)
    vmax = max(float(np.max(frame)) for frame in frames)
    fig, ax = plt.subplots(figsize=single_domain_figsize(lx, ly), constrained_layout=True)
    image = ax.imshow(frames[0], origin="lower", extent=extent, aspect="equal", cmap="inferno", vmin=vmin, vmax=vmax)
    overlay_modules(ax, cfg.layout.centers or [], float(cfg.domain.module_radius))
    set_domain_aspect(ax, lx, ly)
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


def visualize_raw(
    case_dir_arg: Path | None,
    frame_index: int,
    output_dir: Path | None,
    save_gif: bool,
    case_id_arg: str | None = None,
) -> None:
    """Visualize one raw global case frame."""
    case_dir = resolve_case_dir(case_dir_arg, case_id_arg)
    cfg_payload = read_json(case_dir / "case_config.json")
    payload, row = load_raw_frame(case_dir, frame_index)
    case_id = str(cfg_payload.get("save", {}).get("case_id", case_dir.name))
    out_dir = output_dir.expanduser().resolve() if output_dir is not None else case_dir / "plots"
    fields = {name: payload[name] for name in FIELD_NAMES}
    frame_id = int(row.get("saved_frame", 0))
    plot_fields(fields, cfg_payload, f"Raw case {case_id}, frame {frame_id}", out_dir / f"raw_frame_{frame_id:06d}_fields.png")
    plot_internal_temperatures(payload, out_dir / f"raw_frame_{frame_id:06d}_internal_temperature.png")
    plot_interface_targets(payload, out_dir / f"raw_frame_{frame_id:06d}_interface_targets.png")
    plot_convergence_history(read_frame_index(case_dir), cfg_payload, out_dir / "convergence_temperature.png")
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
        key = normalize_case_id(case_id) if case_id is not None else sorted(cases_group.keys())[0]
        if key not in cases_group:
            raise KeyError(f"Case '{key}' not found in {h5_path}. Available: {sorted(cases_group.keys())}")
        group = cases_group[key]
        cfg_payload = json.loads(decode_h5_string(group["case_config_json"][()]))
        steady = group["steady_field"][()]
        rms = group["rms_field"][()]
        selected_times = group["selected_times"][()] if "selected_times" in group else np.asarray([], dtype=np.float32)
        target_mode = decode_h5_string(h5.attrs.get("target_mode", group.attrs.get("target_mode", "unknown")))
        converged = bool(group.attrs.get("converged", False))
        final_delta_inf = float(group.attrs.get("final_delta_inf", np.nan))
        final_delta_l2_rel = float(group.attrs.get("final_delta_l2_rel", np.nan))
        fields = {name: steady[..., idx] for idx, name in enumerate(FIELD_NAMES)}
        rms_fields = {name: rms[..., idx] for idx, name in enumerate(FIELD_NAMES)}
        payload = {
            "module_internal_temperature": group["module_internal_temperature"][()],
            "module_present": group["module_present"][()],
            "interface_response": group["interface_response"][()],
            "interface_feature_names": h5["interface_feature_names"][()],
        }
    out_dir = output_dir.expanduser().resolve() if output_dir is not None else h5_path.parent / "plots"
    status = "converged" if converged else "unconverged"
    detail = f"{status}, mode={target_mode}, selected_times={np.asarray(selected_times).tolist()}"
    plot_fields(fields, cfg_payload, f"Processed steady case {key} ({detail})", out_dir / f"processed_{key}_steady_fields.png")
    plot_fields(rms_fields, cfg_payload, f"Processed RMS case {key} ({detail})", out_dir / f"processed_{key}_rms_fields.png")
    plot_internal_temperatures(payload, out_dir / f"processed_{key}_internal_temperature.png")
    plot_interface_targets(payload, out_dir / f"processed_{key}_interface_targets.png")
    plot_processed_metadata(
        {
            "case_id": key,
            "target_mode": target_mode,
            "converged": converged,
            "selected_times": np.asarray(selected_times).tolist(),
            "final_delta_inf": final_delta_inf,
            "final_delta_l2_rel": final_delta_l2_rel,
        },
        out_dir / f"processed_{key}_metadata.png",
    )
    print(
        "Processed case diagnostics: "
        f"case_id={key}, converged={converged}, target_mode={target_mode}, "
        f"selected_times={np.asarray(selected_times).tolist()}, "
        f"final_delta_inf={final_delta_inf:.6e}, final_delta_l2_rel={final_delta_l2_rel:.6e}"
    )
    print(f"Saved processed visualizations to: {out_dir}")


def main() -> int:
    args = parse_args()
    if args.case_dir is None and args.processed_h5 is None:
        visualize_raw(None, args.frame, args.output_dir, args.save_gif, args.case_id)
        return 0
    if args.case_dir is not None:
        visualize_raw(args.case_dir, args.frame, args.output_dir, args.save_gif, args.case_id)
    if args.processed_h5 is not None:
        visualize_processed(args.processed_h5, args.case_id, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
