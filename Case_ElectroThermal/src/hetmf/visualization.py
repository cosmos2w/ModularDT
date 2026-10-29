"""Saved-field visual checks. Figures are local generated payloads."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .environment import render_environment
from .resources import require_owned_root


def _panel(axis, field, title, *, cmap="viridis", signed=False, contacts=(),
           label_voltages=False):
    field = np.asarray(field)
    bound = np.nanmax(np.abs(field)) if signed else None
    kwargs = {"vmin": -bound, "vmax": bound} if signed and bound else {}
    image = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1),
                        interpolation="nearest", cmap=cmap, **kwargs)
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xticks((0, 0.5, 1))
    axis.set_yticks((0, 0.5, 1))
    for contact in contacts:
        axis.add_patch(Circle((contact["x"], contact["y"]), contact["r"],
                              fill=False, edgecolor="white", linewidth=1.1))
        if label_voltages:
            axis.text(contact["x"], contact["y"], f"{contact['u']:+.2f}",
                      ha="center", va="center", color="white", fontsize=6,
                      bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.6, "edgecolor": "none"})
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.02)


def visualize_case(root: Path, scene: dict, grid: int = 96) -> dict:
    root = require_owned_root(root)
    state_id = scene["state_id"]
    with h5py.File(root / "raw" / f"{state_id}_n{grid}.h5") as handle:
        v = handle["potential"][:]
        theta = handle["temperature_rise"][:]
        qs = handle["q_sheet"][:]
        qc = handle["q_contact"][:]
        jx = handle["jx_face"][:]
        jy = handle["jy_face"][:]
        currents = handle["terminal_current"][:]
    s, k, beta = render_environment(scene["environment"], grid, grid)
    jmag = np.hypot((jx[:, :-1] + jx[:, 1:]) / 2, (jy[:-1] + jy[1:]) / 2)
    contacts = scene["contacts"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    panels = ((s, "Electrical conductance s", "viridis", False),
              (k, "Thermal conductance κ", "viridis", False),
              (beta, "Cooling β", "viridis", False),
              (v, "Potential v", "coolwarm", True),
              (jmag, "Reconstructed |j|", "magma", False),
              (theta, "Temperature rise θ", "magma", False),
              (qs, "Sheet Joule q", "magma", False),
              (qc, "Contact Joule q", "magma", False))
    for index, (axis, (field, title, cmap, signed)) in enumerate(zip(axes.flat, panels)):
        _panel(axis, field, title, cmap=cmap, signed=signed, contacts=contacts,
               label_voltages=index == 0)
    axis = axes.flat[-1]
    axis.bar(np.arange(len(currents)), currents)
    axis.axhline(0, color="black", lw=0.5)
    axis.set_xticks(np.arange(len(currents)), [str(c["id"]) for c in contacts], rotation=60)
    axis.set_title("Terminal current I (dimensionless)", fontsize=9)
    axis.text(0.02, 0.98, f"ΣI={sum(currents):.2e}\nP={sum(c['u']*i for c,i in zip(contacts,currents)):.3g}",
              transform=axis.transAxes, va="top", fontsize=8)
    fig.suptitle(f"{state_id} | pilot physical solve | {grid}² | dimensionless", fontsize=13)
    stem = root / "figures" / f"{state_id}_n{grid}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=120)
    plt.close(fig)
    return {"pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png"))}


def visualize_quartet(root: Path, scenes: list[dict], grid: int = 96,
                      quartet_id: str | None = None) -> dict:
    root = require_owned_root(root)
    if len(scenes) != 4:
        raise ValueError("quartet requires four endpoints")
    arrays = []
    for scene in scenes:
        with h5py.File(root / "raw" / f"{scene['state_id']}_n{grid}.h5") as handle:
            arrays.append((handle["potential"][:], handle["temperature_rise"][:]))
    vmax_v = max(float(np.max(abs(pair[0]))) for pair in arrays)
    vmax_t = max(float(np.max(pair[1])) for pair in arrays)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    for row in range(2):
        scale = vmax_v if row == 0 else vmax_t
        name = "v" if row == 0 else "θ"
        cmap = "coolwarm" if row == 0 else "magma"
        for col, pair in enumerate(arrays):
            field = pair[row]
            image = axes[row, col].imshow(field, origin="lower", extent=(0, 1, 0, 1),
                                          cmap=cmap, vmin=-scale if row == 0 else 0, vmax=scale)
            axes[row, col].set_title(f"{name} {['00','10','01','11'][col]}")
            axes[row, col].set_aspect("equal")
        fig.colorbar(image, ax=axes[row, :4].tolist(), shrink=0.65, pad=0.01)
        interaction = arrays[3][row] - arrays[1][row] - arrays[2][row] + arrays[0][row]
        bound = float(np.max(np.abs(interaction)))
        interaction_image = axes[row, 4].imshow(interaction, origin="lower", extent=(0, 1, 0, 1),
                                                cmap="coolwarm", vmin=-bound or -1, vmax=bound or 1)
        axes[row, 4].set_title(f"interaction {name}; max={bound:.2e}")
        axes[row, 4].set_aspect("equal")
        fig.colorbar(interaction_image, ax=axes[row, 4], shrink=0.65, pad=0.01)
    title = quartet_id or scenes[1].get("quartet_id") or scenes[0]["state_id"]
    fig.suptitle(f"{title} | pilot quartet | {grid}² | dimensionless")
    stem = root / "figures" / f"quartet_{title}_n{grid}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=110)
    plt.close(fig)
    return {"pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png"))}


def visualize_thermal_null(root: Path, plan: dict, grid: int = 96) -> dict:
    """Exact electrical-parent comparison for R02/R08/R09."""
    root = require_owned_root(root)
    lookup = {s["state_id"]: s for s in plan["scenes"]}
    rows = []
    for sid in ("R02", "R08", "R09"):
        scene = lookup[sid]
        _, kappa, beta = render_environment(scene["environment"], grid, grid)
        with h5py.File(root / "raw" / f"{sid}_n{grid}.h5") as handle:
            rows.append((sid, kappa, beta, handle["potential"][:], handle["temperature_rise"][:]))
    theta_min = min(float(row[4].min()) for row in rows)
    theta_max = max(float(row[4].max()) for row in rows)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for index, (sid, kappa, beta, potential, theta) in enumerate(rows):
        for col, (field, title, cmap) in enumerate(((kappa, "κ", "viridis"),
                                                     (beta, "β", "viridis"),
                                                     (theta, "θ", "magma"))):
            image = axes[index, col].imshow(field, origin="lower", extent=(0, 1, 0, 1),
                                            cmap=cmap, vmin=theta_min if col == 2 else None,
                                            vmax=theta_max if col == 2 else None)
            axes[index, col].set_title(f"{sid} {title}; max |Δv|={np.max(abs(potential-rows[0][3])):.1e}" if col == 2 else f"{sid} {title}")
            axes[index, col].set_aspect("equal")
            fig.colorbar(image, ax=axes[index, col], shrink=0.7)
    fig.suptitle(f"R02 electrical parent vs R08/R09 thermal-only changes | {grid}² | dimensionless")
    stem = root / "figures" / f"R02_R08_R09_electrical_null_n{grid}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=115)
    plt.close(fig)
    return {"pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png"))}


def visualize_numerical_board(root: Path, summary: dict, forecasts: dict) -> dict:
    root = require_owned_root(root)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    keys = ("potential_relative_l2", "temperature_rise_relative_l2", "q_total_relative_l2",
            "terminal_current_relative_l2")
    labels = ("v", "θ", "q", "I")
    x = np.arange(len(keys))
    for offset, pair in enumerate(("48_vs_96", "96_vs_192")):
        vals = [summary["refinement_max_by_pair"][pair][key] for key in keys]
        axes[0, 0].bar(x + (offset - 0.5) * 0.36, vals, 0.36, label=pair)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Maximum relative L2 / current error")
    axes[0, 0].set_title("Representative refinement (12 saved inputs)")
    axes[0, 0].legend()
    residual = summary["max_post_storage_residuals"]
    axes[0, 1].bar(range(len(residual)), list(residual.values()))
    axes[0, 1].set_xticks(range(len(residual)), [k.replace("_relative", "") for k in residual], rotation=35, ha="right")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Maximum post-storage conservation residual (320 files)")
    panel = summary["throughput_panel"]
    axes[1, 0].bar(list(panel["by_family_seconds"]),
                   [v["mean"] for v in panel["by_family_seconds"].values()])
    axes[1, 0].tick_params(axis="x", labelrotation=25)
    axes[1, 0].set_ylabel("Mean complete CPU seconds/state")
    axes[1, 0].set_title("256-state native-96 CPU panel")
    names = [name for name in ("compact", "standard", "expanded") if name in forecasts["profiles"]]
    axes[1, 1].bar(names, [forecasts["profiles"][n]["raw_gib"] for n in names], label="raw FP64 source + shard")
    axes[1, 1].bar(names, [forecasts["profiles"][n]["processed_gib"] for n in names],
                   bottom=[forecasts["profiles"][n]["raw_gib"] for n in names], label="processed FP32")
    axes[1, 1].set_ylabel("Projected GiB")
    axes[1, 1].set_title("Measured pilot-byte extrapolation")
    axes[1, 1].legend()
    fig.suptitle("HET-MF pilot numerical and cost board | dimensionless physical fields")
    stem = root / "figures" / "pilot_numerical_cost_board"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=120)
    plt.close(fig)
    return {"pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png"))}


def visualize_physical_responses(root: Path) -> dict:
    """Show actual terminal operators and four-solve contrasts, not learned support."""
    root = require_owned_root(root)
    basis = json.loads((root / "diagnostics" / "terminal_response_basis.json").read_text())
    summary = json.loads((root / "diagnostics" / "pilot_summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, sid in zip(axes.flat[:3], ("R02", "R05", "R10")):
        matrix = np.asarray(basis[sid]["matrix"], dtype=float)
        scale = float(np.max(abs(matrix)))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-scale, vmax=scale)
        axis.set_title(f"{sid}: physical terminal C; I = C u")
        axis.set_xlabel("input contact index")
        axis.set_ylabel("output contact index")
        fig.colorbar(image, ax=axis, shrink=0.8)
    ax = axes.flat[3]
    rows = summary["interaction_summary"]
    labels = list(rows)
    x = np.arange(len(labels))
    floor = 1e-18
    ax.bar(x - 0.18, [max(rows[q]["potential_interaction_fraction"], floor) for q in labels],
           width=0.36, label="potential")
    ax.bar(x + 0.18, [max(rows[q]["temperature_interaction_fraction"], floor) for q in labels],
           width=0.36, label="temperature")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("four-solve interaction / endpoint scale")
    ax.set_title("Controlled pilot quartets; zero shown at floor")
    ax.legend()
    fig.suptitle("Physical terminal responses and counterfactual interactions | dimensionless")
    stem = root / "figures" / "physical_responses"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=120)
    plt.close(fig)
    return {"pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png"))}
