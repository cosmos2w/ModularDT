"""
Plot the configured multi-cylinder computational domain with vivid styling.

python src/plot_domain_shape.py --config-json config_inert.json
python src/plot_domain_shape.py --config-json config_active.json

"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np

from multicyl_common import (
    SimulationConfig,
    config_from_dict,
    default_config_dir,
    default_domain_shape_dir,
    materialize_layout,
    resolve_config_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the multi-cylinder computational domain from a JSON config.")
    parser.add_argument(
        "--config-json",
        type=str,
        required=True,
        help=f"JSON config file name or path. Relative paths are loaded from {default_config_dir()}.",
    )
    return parser.parse_args()


def load_config(config_arg: str) -> tuple[SimulationConfig, Path]:
    config_path = resolve_config_path(config_arg)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    config = materialize_layout(config_from_dict(raw))
    return config, config_path


def make_background(nx: int = 1200, ny: int = 700) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)
    base = 0.55 * xx + 0.45 * (1.0 - yy)
    wave = 0.12 * np.sin(3.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    plume = np.exp(-((xx - 0.18) ** 2 / 0.06 + (yy - 0.58) ** 2 / 0.18))
    return base + wave + 0.2 * plume


def save_domain_plot(cfg: SimulationConfig, config_path: Path) -> Path:
    output_dir = default_domain_shape_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"domain_{config_path.stem}_case_{cfg.save.case_id}_{stamp}.png"
    out_path = output_dir / file_name

    cmap = LinearSegmentedColormap.from_list(
        "domain_glow",
        ["#061826", "#0f3f5c", "#0a9396", "#94d2bd", "#f9c74f"],
    )

    fig, ax = plt.subplots(figsize=(13.5, 7.5), dpi=170, constrained_layout=True)
    fig.patch.set_facecolor("#03131d")
    ax.set_facecolor("#082433")

    bg = make_background()
    ax.imshow(bg, extent=(0.0, cfg.domain.lx, 0.0, cfg.domain.ly), origin="lower", cmap=cmap, alpha=0.92, aspect="auto")

    # Add faint mesh lines so the computational resolution reads visually.
    x_lines = np.linspace(0.0, cfg.domain.lx, min(cfg.domain.nx, 28) + 1)
    y_lines = np.linspace(0.0, cfg.domain.ly, min(cfg.domain.ny, 16) + 1)
    for xv in x_lines:
        ax.plot([xv, xv], [0.0, cfg.domain.ly], color=(1.0, 1.0, 1.0, 0.07), lw=0.8)
    for yv in y_lines:
        ax.plot([0.0, cfg.domain.lx], [yv, yv], color=(1.0, 1.0, 1.0, 0.07), lw=0.8)

    border = Rectangle(
        (0.0, 0.0),
        cfg.domain.lx,
        cfg.domain.ly,
        fill=False,
        lw=2.6,
        edgecolor="#caf0f8",
        joinstyle="round",
    )
    ax.add_patch(border)

    # Stylized flow arrows to hint at the bulk flow direction.
    arrow_y = np.linspace(0.18 * cfg.domain.ly, 0.82 * cfg.domain.ly, 5)
    for idx, y0 in enumerate(arrow_y):
        alpha = 0.35 + 0.08 * idx
        arrow = FancyArrowPatch(
            (0.6, y0),
            (3.2, y0),
            arrowstyle="-|>",
            mutation_scale=16,
            lw=2.2,
            color=(0.85, 0.96, 1.0, alpha),
        )
        ax.add_patch(arrow)

    powers = cfg.layout.heat_powers or [0.0 for _ in range(cfg.layout.num_cylinders)]
    power_max = max(max(abs(float(power)) for power in powers), 1e-8)
    active_mode = cfg.thermal.enabled

    for idx, ((cx, cy), qdot) in enumerate(zip(cfg.layout.centers or [], powers)):
        if active_mode:
            heat_level = float(qdot) / power_max
            glow_color = plt.cm.coolwarm(0.5 + 0.5 * heat_level)
            halo_radius = cfg.domain.cylinder_radius * (1.85 + 0.25 * abs(heat_level))
            halo = Circle((cx, cy), halo_radius, color=glow_color, alpha=0.24, ec="none")
            ax.add_patch(halo)
            face_color = glow_color
        else:
            face_color = "#90e0ef"

        cyl = Circle(
            (cx, cy),
            cfg.domain.cylinder_radius,
            facecolor=face_color,
            edgecolor="#f1faee",
            lw=1.8,
            alpha=0.96,
        )
        ax.add_patch(cyl)

        label = f"C{idx}"
        if active_mode:
            label += f" | q={qdot:.2f}"
        ax.text(
            cx,
            cy + 1.38 * cfg.domain.cylinder_radius,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#f8f9fa",
            weight="bold",
        )

    title = f"Multi-cylinder Domain | case {cfg.save.case_id} | mode {cfg.mode}"
    subtitle = (
        f"{cfg.domain.nx}x{cfg.domain.ny} cells  |  "
        f"Lx={cfg.domain.lx:.1f}, Ly={cfg.domain.ly:.1f}  |  "
        f"Re={cfg.flow.re:.1f}  |  cylinders={cfg.layout.num_cylinders}"
    )
    ax.set_title(title, fontsize=20, color="#f8f9fa", pad=18, weight="bold")
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#d9ed92",
    )

    ax.text(
        0.015,
        0.02,
        f"Config: {config_path.name}",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#e9ecef",
        bbox={"facecolor": (0.02, 0.08, 0.11, 0.45), "edgecolor": (1, 1, 1, 0.12), "boxstyle": "round,pad=0.35"},
    )

    ax.set_xlim(0.0, cfg.domain.lx)
    ax.set_ylim(0.0, cfg.domain.ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=13, color="#f8f9fa")
    ax.set_ylabel("y", fontsize=13, color="#f8f9fa")
    ax.tick_params(colors="#e0fbfc", labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    cfg, config_path = load_config(args.config_json)
    out_path = save_domain_plot(cfg, config_path)
    print(f"Saved domain plot to: {out_path}")


if __name__ == "__main__":
    main()
