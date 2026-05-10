from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


CHANNEL_INDEX = {"u": 0, "v": 1, "p": 2, "omega": 3, "temperature": 4}
RESAMPLE_BY_NAME = {
    "nearest": "NEAREST",
    "bilinear": "BILINEAR",
    "bicubic": "BICUBIC",
    "lanczos": "LANCZOS",
}


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def _field_scale(values: np.ndarray, field_name: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if field_name in {"u", "v", "omega"}:
        mag = max(abs(lo), abs(hi), 1.0e-9)
        return -mag, mag
    if abs(hi - lo) < 1.0e-12:
        pad = max(abs(hi), 1.0) * 1.0e-3
        return hi - pad, hi + pad
    return lo, hi


def _module_draw_style(heat: float, heat_min: float, heat_max: float) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    denom = max(float(heat_max) - float(heat_min), 1.0e-9)
    t = float(np.clip((heat - heat_min) / denom, 0.0, 1.0))
    fill = (int(45 + 190 * t), int(120 - 50 * t), int(210 - 150 * t), 95)
    outline = (20, 28, 38, 255)
    return fill, outline


def render_field_images(
    field_grid: np.ndarray,
    output_dir: Path,
    modules: List[Mapping[str, Any]],
    *,
    domain_length_x: float,
    domain_length_y: float,
    module_radius: float,
    fields: Iterable[str] = ("temperature", "u", "v", "p", "omega"),
    display_smoothing: bool = True,
    display_scale: int = 3,
    render_interpolation: str = "bicubic",
) -> Dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-web")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib
    from PIL import Image, ImageDraw

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    arr = np.asarray(field_grid, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("field_grid must have shape [H, W, C].")
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    display_scale = max(1, int(display_scale))
    display_width = width * display_scale
    display_height = height * display_scale
    render_interpolation = str(render_interpolation).lower()
    if render_interpolation not in RESAMPLE_BY_NAME:
        render_interpolation = "bicubic"
    effective_interpolation = render_interpolation if display_smoothing else "nearest"
    resampling = getattr(Image.Resampling, RESAMPLE_BY_NAME[effective_interpolation])
    heat_values = [_safe_float(item.get("heat_power"), 0.0) for item in modules]
    heat_min = min(heat_values) if heat_values else 0.0
    heat_max = max(heat_values) if heat_values else 1.0
    metadata: Dict[str, Any] = {
        "frame_count": 1,
        "fields": {},
        "raw_resolution": {"nx": width, "ny": height},
        "display_resolution": {"width": display_width, "height": display_height},
        "display_smoothing": bool(display_smoothing),
        "display_scale": display_scale,
        "render_interpolation": effective_interpolation,
        "note": "Static steady-state field render; raw arrays are stored in the NPZ export.",
    }

    for field_name in fields:
        if field_name not in CHANNEL_INDEX or CHANNEL_INDEX[field_name] >= arr.shape[-1]:
            continue
        values = arr[..., CHANNEL_INDEX[field_name]]
        vmin, vmax = _field_scale(values, field_name)
        cmap_name = "inferno" if field_name == "temperature" else "RdBu_r" if field_name in {"u", "v", "omega"} else "viridis"
        cmap = cm.get_cmap(cmap_name)
        denom = max(vmax - vmin, 1.0e-12)
        normalized = np.clip((values - vmin) / denom, 0.0, 1.0)
        rgba = cmap(normalized, bytes=True)
        raw_image = Image.fromarray(np.flipud(rgba), mode="RGBA")
        image = raw_image.resize((display_width, display_height), resampling)
        draw = ImageDraw.Draw(image, mode="RGBA")
        for item in modules:
            cx = _safe_float(item.get("x")) / max(domain_length_x, 1.0e-9) * display_width
            cy = display_height - _safe_float(item.get("y")) / max(domain_length_y, 1.0e-9) * display_height
            rx = float(module_radius) / max(domain_length_x, 1.0e-9) * display_width
            ry = float(module_radius) / max(domain_length_y, 1.0e-9) * display_height
            fill, outline = _module_draw_style(_safe_float(item.get("heat_power")), heat_min, heat_max)
            box = (cx - rx, cy - ry, cx + rx, cy + ry)
            draw.ellipse(box, fill=fill, outline=outline, width=max(2, display_scale * 2))
            inset = max(1, display_scale)
            draw.ellipse(
                (box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset),
                outline=(255, 255, 255, 210),
                width=max(1, display_scale),
            )
        path = output_dir / f"{field_name}.png"
        image.save(path)
        metadata["fields"][field_name] = {
            "vmin": vmin,
            "vmax": vmax,
            "frames": [f"{field_name}.png"],
        }

    with (output_dir / "render_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def render_internal_summary(internal: Optional[np.ndarray], modules: List[Mapping[str, Any]], output_path: Path) -> Optional[Path]:
    if internal is None:
        return None
    arr = np.asarray(internal, dtype=np.float32)
    if arr.size == 0:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-web")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(len(modules), arr.shape[0])
    values = arr[..., 0] if arr.shape[-1] == 1 else arr
    means = [float(np.nanmean(values[i])) for i in range(count)]
    peaks = [float(np.nanmax(values[i])) for i in range(count)]
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    x = np.arange(count)
    ax.bar(x - 0.18, means, width=0.36, label="mean")
    ax.bar(x + 0.18, peaks, width=0.36, label="peak")
    ax.set_xlabel("module")
    ax.set_ylabel("temperature")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def render_interface_curves(interface: Optional[np.ndarray], modules: List[Mapping[str, Any]], output_path: Path) -> Optional[Path]:
    if interface is None:
        return None
    arr = np.asarray(interface, dtype=np.float32)
    if arr.size == 0 or arr.ndim < 3:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-web")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(len(modules), arr.shape[0], 4)
    if count <= 0:
        return None
    theta = np.linspace(0.0, 2.0 * math.pi, arr.shape[1], endpoint=False)
    fig, axes = plt.subplots(count, 2, figsize=(10.0, max(2.7 * count, 3.0)), constrained_layout=True)
    axes = np.asarray(axes).reshape(count, 2)
    for row in range(count):
        axes[row, 0].plot(theta, arr[row, :, 0], color="#b3365b", lw=1.6)
        axes[row, 1].plot(theta, arr[row, :, 1], color="#2e6f9e", lw=1.6)
        axes[row, 0].set_ylabel(f"M{row}")
        axes[row, 0].set_title("T_surface")
        axes[row, 1].set_title("q_normal")
        for col in range(2):
            axes[row, col].grid(True, alpha=0.25)
            axes[row, col].set_xlabel("theta")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def render_organization_matrices(aux: Optional[Mapping[str, Any]], output_path: Path) -> Optional[Path]:
    if not isinstance(aux, Mapping) or "A_mh" not in aux or "A_eh" not in aux:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-web")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_mh = np.asarray(aux["A_mh"], dtype=np.float32)
    a_eh = np.asarray(aux["A_eh"], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(a_mh, aspect="auto", cmap="viridis")
    axes[0].set_title("module to hyperedge")
    axes[0].set_xlabel("hyperedge")
    axes[0].set_ylabel("module")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(a_eh, aspect="auto", cmap="magma")
    axes[1].set_title("environment to hyperedge")
    axes[1].set_xlabel("hyperedge")
    axes[1].set_ylabel("environment token")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path
