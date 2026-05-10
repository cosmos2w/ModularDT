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


def _scale_from_arrays(arrays: Iterable[np.ndarray], *, symmetric: bool = False, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    finite_parts = []
    for values in arrays:
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_parts.append(finite.reshape(-1))
    if not finite_parts:
        return fallback
    values = np.concatenate(finite_parts)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if symmetric:
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
    scale_overrides: Optional[Mapping[str, tuple[float, float]]] = None,
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
        vmin, vmax = scale_overrides.get(field_name, _field_scale(values, field_name)) if scale_overrides else _field_scale(values, field_name)
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


def render_error_field_images(
    error_grid: np.ndarray,
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
    scale_overrides: Optional[Mapping[str, tuple[float, float]]] = None,
) -> Dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-web")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib
    from PIL import Image, ImageDraw

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    arr = np.asarray(error_grid, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("error_grid must have shape [H, W, C].")
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
        "note": "Relative absolute error render: abs(prediction - truth) / stabilized abs(truth).",
    }

    cmap = cm.get_cmap("magma")
    for field_name in fields:
        if field_name not in CHANNEL_INDEX or CHANNEL_INDEX[field_name] >= arr.shape[-1]:
            continue
        values = arr[..., CHANNEL_INDEX[field_name]]
        if scale_overrides and field_name in scale_overrides:
            _, vmax = scale_overrides[field_name]
            vmax = max(float(vmax), 1.0e-8)
        else:
            finite = values[np.isfinite(values)]
            vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
            vmax = max(vmax, 1.0e-8)
        normalized = np.clip(values / vmax, 0.0, 1.0)
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
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=outline, width=max(2, display_scale * 2))
        path = output_dir / f"{field_name}.png"
        image.save(path)
        metadata["fields"][field_name] = {"vmin": 0.0, "vmax": vmax, "frames": [f"{field_name}.png"]}

    with (output_dir / "render_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _internal_module_image(internal: np.ndarray, mask: np.ndarray, module_index: int) -> Optional[np.ndarray]:
    arr = np.asarray(internal, dtype=np.float32)
    if arr.size == 0 or module_index < 0 or module_index >= arr.shape[0]:
        return None
    local = np.asarray(arr[module_index], dtype=np.float32)
    if local.ndim >= 3 and local.shape[-1] == 1:
        local = local[..., 0]
    mask_bool = np.asarray(mask, dtype=bool)
    if local.shape == mask_bool.shape:
        image = local.astype(np.float32, copy=True)
        image[~mask_bool] = np.nan
        return image
    flat = local.reshape(-1)
    image = np.full(mask_bool.shape, np.nan, dtype=np.float32)
    image[mask_bool] = flat[: int(np.sum(mask_bool))]
    return image


def render_internal_temperature_images(
    internal: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    output_dir: Path,
    modules: List[Mapping[str, Any]],
    *,
    module_indices: Optional[Iterable[int]] = None,
    scale: Optional[tuple[float, float]] = None,
    error: bool = False,
    prefix: str = "module",
    display_scale: int = 4,
) -> Dict[str, Any]:
    arr = np.asarray(internal, dtype=np.float32) if internal is not None else np.asarray([], dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=bool) if mask is not None else np.asarray([], dtype=bool)
    if arr.size == 0 or mask_arr.size == 0:
        return {"available": False, "modules": [], "scale": None}
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    from PIL import Image, ImageDraw

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    indices = list(module_indices) if module_indices is not None else list(range(min(len(modules), arr.shape[0])))
    images = [(idx, _internal_module_image(arr, mask_arr, idx)) for idx in indices]
    valid_images = [(idx, image) for idx, image in images if image is not None]
    if not valid_images:
        return {"available": False, "modules": [], "scale": None}
    vmin, vmax = scale if scale is not None else _scale_from_arrays([image for _, image in valid_images], fallback=(0.0, 1.0))
    if error:
        vmin = 0.0
        vmax = max(float(vmax), 1.0e-8)
    cmap = cm.get_cmap("magma" if error else "inferno")
    denom = max(float(vmax) - float(vmin), 1.0e-12)
    scale_factor = max(1, int(display_scale))
    module_meta = []
    for idx, image in valid_images:
        normalized = np.clip((image - float(vmin)) / denom, 0.0, 1.0)
        rgba = cmap(np.nan_to_num(normalized, nan=0.0), bytes=True)
        rgba[..., 3] = np.where(np.isfinite(image), 255, 0).astype(np.uint8)
        raw = Image.fromarray(np.flipud(rgba), mode="RGBA")
        rendered = raw.resize((raw.width * scale_factor, raw.height * scale_factor), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(rendered, mode="RGBA")
        draw.ellipse((2, 2, rendered.width - 3, rendered.height - 3), outline=(20, 28, 38, 230), width=max(1, scale_factor))
        path = output_dir / f"{prefix}_{idx:02d}.png"
        rendered.save(path)
        module_meta.append(
            {
                "index": int(idx),
                "label": f"M{int(idx) + 1}",
                "heat_power": _safe_float(modules[idx].get("heat_power"), 0.0) if idx < len(modules) else 0.0,
                "file": path.name,
            }
        )
    return {"available": True, "modules": module_meta, "scale": {"vmin": float(vmin), "vmax": float(vmax)}, "error": bool(error)}


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


def render_organization_domain_overlay(
    modules: List[Mapping[str, Any]],
    heat_powers: Iterable[float],
    aux: Optional[Mapping[str, Any]],
    output_path: Path,
    *,
    domain_length_x: float,
    domain_length_y: float,
    module_radius: float,
    env_token_xy: Optional[Any] = None,
) -> Optional[Path]:
    if not isinstance(aux, Mapping) or "A_mh" not in aux or "A_eh" not in aux:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-web")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a_mh = np.asarray(aux.get("A_mh"), dtype=np.float32)
    a_eh = np.asarray(aux.get("A_eh"), dtype=np.float32)
    if a_mh.ndim != 2 or a_eh.ndim != 2 or a_mh.size == 0 or a_eh.size == 0:
        return None
    env_coords = np.asarray(env_token_xy if env_token_xy is not None else aux.get("env_coords", []), dtype=np.float32).reshape(-1, 2)
    if env_coords.shape[0] != a_eh.shape[0]:
        nx = max(int(round(math.sqrt(a_eh.shape[0] * max(float(domain_length_x) / max(float(domain_length_y), 1.0e-8), 1.0)))), 1)
        ny = max(int(math.ceil(a_eh.shape[0] / nx)), 1)
        xs = np.linspace(0.35, float(domain_length_x) - 0.35, nx, dtype=np.float32)
        ys = np.linspace(0.35, float(domain_length_y) - 0.35, ny, dtype=np.float32)
        grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)
        env_coords = grid[: a_eh.shape[0]]
    k = int(max(a_mh.shape[1], a_eh.shape[1]))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(idx % 10) for idx in range(max(k, 1))]
    dominant_env = np.argmax(a_eh, axis=1)
    confidence = np.max(a_eh, axis=1)
    dominant_mod = np.argmax(a_mh, axis=1)
    mod_conf = np.max(a_mh, axis=1)
    heat = list(heat_powers)

    fig, ax = plt.subplots(figsize=(9.4, 4.4), constrained_layout=True)
    ax.add_patch(plt.Rectangle((0.0, 0.0), float(domain_length_x), float(domain_length_y), fill=False, lw=1.4, color="#172026"))
    ax.annotate("inlet", xy=(0.15, domain_length_y * 0.5), xytext=(0.75, domain_length_y * 0.5), arrowprops={"arrowstyle": "<-", "lw": 1.2}, va="center", fontsize=9)
    ax.annotate("outlet", xy=(domain_length_x - 0.15, domain_length_y * 0.5), xytext=(domain_length_x - 1.1, domain_length_y * 0.5), arrowprops={"arrowstyle": "->", "lw": 1.2}, va="center", ha="right", fontsize=9)
    for idx, (x, y) in enumerate(env_coords):
        hidx = int(dominant_env[idx]) if idx < dominant_env.shape[0] else 0
        alpha = 0.18 + 0.62 * float(np.clip(confidence[idx] if idx < confidence.shape[0] else 0.5, 0.0, 1.0))
        ax.scatter(float(x), float(y), s=26, color=colors[hidx], alpha=alpha, edgecolors="none")
    for idx, module in enumerate(modules):
        hidx = int(dominant_mod[idx]) if idx < dominant_mod.shape[0] else 0
        cx = _safe_float(module.get("x"))
        cy = _safe_float(module.get("y"))
        q = _safe_float(module.get("heat_power"), heat[idx] if idx < len(heat) else 0.0)
        ring = 1.6 + 3.4 * float(np.clip(mod_conf[idx] if idx < mod_conf.shape[0] else 0.5, 0.0, 1.0))
        ax.add_patch(plt.Circle((cx, cy), float(module_radius), facecolor=colors[hidx], edgecolor="#111827", lw=ring, alpha=0.55))
        ax.text(cx, cy, f"M{idx + 1}\nq={q:.2g}", ha="center", va="center", fontsize=8, color="#111827", weight="bold")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[idx], label=f"H{idx}", markersize=7) for idx in range(k)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(k, 8), frameon=False, fontsize=8)
    ax.set_xlim(-0.1, float(domain_length_x) + 0.1)
    ax.set_ylim(-0.1, float(domain_length_y) + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Organizer domain overlay")
    ax.grid(True, alpha=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path
