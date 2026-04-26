from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


CHANNEL_INDEX = {"u": 0, "v": 1, "p": 2, "omega": 3}
RESAMPLE_BY_NAME = {
    "nearest": "NEAREST",
    "bilinear": "BILINEAR",
    "bicubic": "BICUBIC",
    "lanczos": "LANCZOS",
}


def _field_scale(values: np.ndarray, field_name: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if field_name == "omega":
        mag = max(abs(vmin), abs(vmax), 1e-9)
        return -mag, mag
    if abs(vmax - vmin) < 1e-12:
        pad = max(abs(vmax), 1.0) * 1e-3
        return vmin - pad, vmax + pad
    return vmin, vmax


def render_frames(
    field_cycle: np.ndarray,
    output_dir: Path,
    cylinders: List[Dict[str, float]],
    domain_length_x: float,
    domain_length_y: float,
    fields: Iterable[str] = ("u", "v", "p", "omega"),
    display_smoothing: bool = True,
    display_scale: int = 3,
    render_interpolation: str = "bicubic",
) -> Dict:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-modulardt")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib
    from PIL import Image, ImageDraw

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    arr = np.asarray(field_cycle)
    output_dir.mkdir(parents=True, exist_ok=True)
    display_scale = max(1, int(display_scale))
    render_interpolation = str(render_interpolation).lower()
    if render_interpolation not in RESAMPLE_BY_NAME:
        render_interpolation = "bicubic"
    height = int(arr.shape[1])
    width = int(arr.shape[2])
    display_width = width * display_scale
    display_height = height * display_scale
    effective_interpolation = render_interpolation if display_smoothing else "nearest"
    resampling = getattr(Image.Resampling, RESAMPLE_BY_NAME[effective_interpolation])
    metadata = {
        "frame_count": int(arr.shape[0]),
        "fields": {},
        "raw_resolution": {"nx": width, "ny": height},
        "display_resolution": {"width": display_width, "height": display_height},
        "display_smoothing": bool(display_smoothing),
        "display_scale": display_scale,
        "render_interpolation": effective_interpolation,
        "kpi_source": "raw_model_grid",
        "note": "Frame PNGs are presentation renders; KPI curves and NPZ exports use the raw model grid.",
    }

    for field_name in fields:
        if field_name not in CHANNEL_INDEX:
            continue
        channel = CHANNEL_INDEX[field_name]
        values = arr[..., channel]
        vmin, vmax = _field_scale(values, field_name)
        field_dir = output_dir / field_name
        field_dir.mkdir(parents=True, exist_ok=True)
        cmap = cm.get_cmap("RdBu_r" if field_name == "omega" else "viridis")
        denom = max(vmax - vmin, 1e-12)

        for frame_idx in range(arr.shape[0]):
            normalized = np.clip((values[frame_idx] - vmin) / denom, 0.0, 1.0)
            rgba = cmap(normalized, bytes=True)
            raw_image = Image.fromarray(np.flipud(rgba), mode="RGBA")
            image = raw_image.resize((display_width, display_height), resampling)
            draw = ImageDraw.Draw(image)
            for cyl in cylinders:
                cx = float(cyl["x"]) / max(domain_length_x, 1e-6) * display_width
                cy = display_height - (float(cyl["y"]) / max(domain_length_y, 1e-6) * display_height)
                rx = 0.5 / max(domain_length_x, 1e-6) * display_width
                ry = 0.5 / max(domain_length_y, 1e-6) * display_height
                box = (
                    min(cx - rx, cx + rx),
                    min(cy - ry, cy + ry),
                    max(cx - rx, cx + rx),
                    max(cy - ry, cy + ry),
                )
                outline_width = max(2, display_scale * 2)
                draw.ellipse(box, outline=(0, 0, 0, 255), width=outline_width)
                if (box[2] - box[0]) > 2 and (box[3] - box[1]) > 2:
                    inset = max(1, display_scale)
                    draw.ellipse(
                        (box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset),
                        outline=(255, 255, 255, 230),
                        width=max(1, display_scale),
                    )
            image.save(field_dir / f"{frame_idx:03d}.png")

        metadata["fields"][field_name] = {
            "vmin": vmin,
            "vmax": vmax,
            "frames": [f"{field_name}/{i:03d}.png" for i in range(arr.shape[0])],
        }

    with (output_dir.parent / "render_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata
