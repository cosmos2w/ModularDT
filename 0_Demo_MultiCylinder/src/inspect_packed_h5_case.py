"""
Inspect and visualize a packed HDF5 inert multi-cylinder dataset.

This helper opens a preprocessed ``packed_dataset.h5`` file in read-only mode,
prints a detailed summary for one case, and can optionally export an animated
GIF of the canonical periodic attractor using the vorticity field.

Example
-------
python src/inspect_packed_h5_case.py \
    --h5-path ./Data_Saved/Processed_Inert_Dataset/packed_dataset.h5 \
    --case-id 0061 \
    --gif-out ./Data_Saved/Processed_Inert_Dataset/case_0061_omega.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect one case inside a packed HDF5 inert multi-cylinder dataset."
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        required=True,
        help="Path to the packed_dataset.h5 file.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        required=True,
        help="Exact case id string under cases/<case_id>.",
    )
    parser.add_argument(
        "--gif-out",
        type=Path,
        default=None,
        help="Optional output path for an animated GIF of the canonical vorticity cycle.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the exported GIF.",
    )
    return parser.parse_args()


def decode_scalar(value: object) -> object:
    """Convert HDF5 scalars into friendly Python objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_string_array(values: np.ndarray) -> List[str]:
    """Decode HDF5 string datasets/arrays into a plain Python string list."""
    decoded: List[str] = []
    for item in np.asarray(values).reshape(-1):
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        elif isinstance(item, np.generic):
            decoded.append(str(item.item()))
        else:
            decoded.append(str(item))
    return decoded


def safe_percentile_limits(field: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    """Compute stable symmetric color limits for a diverging colormap."""
    finite_values = field[np.isfinite(field)]
    if finite_values.size == 0:
        return -1.0, 1.0

    lo = float(np.percentile(finite_values, low))
    hi = float(np.percentile(finite_values, high))
    vmax = max(abs(lo), abs(hi), 1e-6)
    return -vmax, vmax


def format_heat_power(value: float) -> str:
    return f"{value:+.2f}" if abs(value) < 100.0 else f"{value:+.2e}"


def overlay_cylinders_with_heat(
    ax,
    centers: Optional[np.ndarray],
    heat_powers: Optional[np.ndarray],
    cylinder_radius: float,
) -> None:
    if centers is None:
        return
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    powers = (
        np.asarray(heat_powers, dtype=np.float32).reshape(-1)
        if heat_powers is not None
        else np.zeros((centers.shape[0],), dtype=np.float32)
    )
    max_abs_power = float(np.max(np.abs(powers))) if powers.size else 0.0
    show_heat = powers.size >= centers.shape[0] and max_abs_power > 0.0

    for idx, (cx, cy) in enumerate(centers):
        qdot = float(powers[idx]) if idx < powers.size else 0.0
        color = "#b2182b" if show_heat and qdot > 0.0 else "#2166ac" if show_heat and qdot < 0.0 else "k"
        ax.add_patch(plt.Circle((cx, cy), cylinder_radius, fill=False, color=color, lw=1.7 if show_heat else 1.0))
        if show_heat:
            ax.text(
                cx,
                cy + 1.35 * cylinder_radius,
                f"C{idx} q={format_heat_power(qdot)}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=color,
                weight="bold",
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.82, "boxstyle": "round,pad=0.18", "linewidth": 0.7},
                zorder=20,
            )


def print_root_summary(h5_file: h5py.File) -> None:
    """Print global attributes stored at the HDF5 root."""
    print("=== Packed Dataset Summary ===")
    print(f"HDF5 path: {Path(h5_file.filename).resolve()}")

    root_attrs = dict(h5_file.attrs.items())
    for key in ("dataset_type", "phase_bins", "save_cycles", "sampling_mode", "channel_order", "input_root", "output_root"):
        if key in root_attrs:
            print(f"{key}: {decode_scalar(root_attrs[key])}")

    if "cases" in h5_file:
        num_cases = len(h5_file["cases"].keys())
        print(f"num_cases: {num_cases}")
    print()


def print_case_summary(case_id: str, case_group: h5py.Group) -> None:
    """Print case-level metadata and dataset shapes."""
    print(f"=== Case Summary: {case_id} ===")

    for key, value in sorted(case_group.attrs.items()):
        print(f"{key}: {decode_scalar(value)}")
    print()

    print("Datasets:")
    for key in sorted(case_group.keys()):
        obj = case_group[key]
        if isinstance(obj, h5py.Dataset):
            print(f"  {key}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  {key}/")
            for child_key in sorted(obj.keys()):
                child = obj[child_key]
                if isinstance(child, h5py.Dataset):
                    print(f"    {child_key}: shape={child.shape}, dtype={child.dtype}")
    print()


def get_channel_order(case_group: h5py.Group, h5_file: h5py.File) -> List[str]:
    """Read channel order, preferring the case-local dataset when present."""
    if "channel_order" in case_group:
        return decode_string_array(case_group["channel_order"][...])

    root_channel_order = h5_file.attrs.get("channel_order")
    if root_channel_order is not None:
        if isinstance(root_channel_order, str):
            cleaned = root_channel_order.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    import json

                    return [str(item) for item in json.loads(cleaned)]
                except Exception:
                    pass
            return [part.strip() for part in cleaned.split(",") if part.strip()]
        return decode_string_array(np.asarray(root_channel_order))

    raise KeyError("Could not determine channel order from case dataset or root attributes.")


def plot_vorticity_gif(
    case_id: str,
    case_group: h5py.Group,
    channel_order: List[str],
    gif_out: Path,
    fps: int,
) -> None:
    """Create and save an animated GIF of the canonical vorticity cycle."""
    required = ("canonical_cycle", "phase_bin_centers", "x_grid", "y_grid")
    missing = [name for name in required if name not in case_group]
    if missing:
        raise KeyError(f"Missing required datasets for GIF export: {missing}")

    if "omega" not in channel_order:
        raise ValueError(f"Channel order does not include 'omega': {channel_order}")

    canonical_cycle = case_group["canonical_cycle"][...]
    phase_bin_centers = case_group["phase_bin_centers"][...]
    x_grid = case_group["x_grid"][...]
    y_grid = case_group["y_grid"][...]
    centers = case_group["cylinder_centers"][...] if "cylinder_centers" in case_group else None
    heat_powers = case_group["heat_powers"][...] if "heat_powers" in case_group else None
    cylinder_radius = float(case_group.attrs.get("cylinder_radius", 0.5))

    omega_idx = channel_order.index("omega")
    omega_cycle = canonical_cycle[..., omega_idx]
    vmin, vmax = safe_percentile_limits(omega_cycle, low=1.0, high=99.0)

    gif_out = gif_out.expanduser().resolve()
    gif_out.parent.mkdir(parents=True, exist_ok=True)

    extent = (
        float(np.nanmin(x_grid)),
        float(np.nanmax(x_grid)),
        float(np.nanmin(y_grid)),
        float(np.nanmax(y_grid)),
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140, constrained_layout=True)
    image = ax.imshow(
        omega_cycle[0],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    overlay_cylinders_with_heat(ax, centers, heat_powers, cylinder_radius)

    def update(frame_idx: int):
        image.set_data(omega_cycle[frame_idx])
        ax.set_title(f"Case {case_id} - Phase: {float(phase_bin_centers[frame_idx]):.3f}")
        return (image,)

    animation = FuncAnimation(
        fig,
        update,
        frames=int(omega_cycle.shape[0]),
        interval=max(int(1000 / max(fps, 1)), 1),
        blit=False,
        repeat=True,
    )
    writer = PillowWriter(fps=fps)
    animation.save(gif_out, writer=writer)
    plt.close(fig)

    print(f"Saved GIF to: {gif_out}")


def main() -> None:
    """Entry point for the inspection helper."""
    args = parse_args()
    h5_path = args.h5_path.expanduser().resolve()

    if args.fps <= 0:
        raise ValueError("--fps must be a positive integer.")
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        print_root_summary(h5_file)

        if "cases" not in h5_file:
            raise KeyError(f"No 'cases' group found in HDF5 file: {h5_path}")

        cases_group = h5_file["cases"]
        if args.case_id not in cases_group:
            available = sorted(cases_group.keys())
            preview = ", ".join(available[:10])
            suffix = "" if len(available) <= 10 else ", ..."
            raise KeyError(
                f"Case id '{args.case_id}' not found under 'cases/'. "
                f"Available case ids ({len(available)} total): {preview}{suffix}"
            )

        case_group = cases_group[args.case_id]
        print_case_summary(args.case_id, case_group)

        channel_order = get_channel_order(case_group, h5_file)
        print(f"Resolved channel_order: {channel_order}")

        if args.gif_out is not None:
            plot_vorticity_gif(
                case_id=args.case_id,
                case_group=case_group,
                channel_order=channel_order,
                gif_out=args.gif_out,
                fps=args.fps,
            )


if __name__ == "__main__":
    main()
