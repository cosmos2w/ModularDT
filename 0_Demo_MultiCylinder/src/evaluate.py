from __future__ import annotations

"""Evaluate a trained checkpoint on a packed inert multi-cylinder HDF5 dataset.

By default this script finds the most recent training run whose directory starts
with "Case{case_id}_" under Saved_Model/ and loads best_model.pt. It then
reconstructs a chosen dataset case at a requested phase and generates a quick-
look comparison plot for [u, v, p, omega].

Checkpoint loading is decoder-agnostic as long as the saved `model_config`
matches one of the decoder types supported by `src/model.py`, including
`deeponet`.

Run it like:
python src/evaluate.py --case-id 0001 --dataset-case-id 0161 --dataset-split train --latest

"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from model import HypergraphNeuralFieldModel, ModelConfig

DEMO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hypergraph-organized neural field model.")
    parser.add_argument("--case-id", type=str, required=True, 
                        help="Training run case_id used in Saved_Model/Case{case_id}_*.")
    parser.add_argument("--dataset-case-id", type=str, default=None, 
                        help="Packed dataset case id to reconstruct. Defaults to first case in the selected split.")
    parser.add_argument("--dataset-split", type=str, default="test", 
                        help="Which split to search when dataset-case-id is omitted.")
    parser.add_argument("--tau", type=float, default=None, 
                        help="Normalized phase value in [0, 1]. If omitted, use the first saved phase bin.")
    parser.add_argument("--latest", action="store_true", 
                        help="Load latest_model.pt instead of best_model.pt.")
    parser.add_argument("--device", type=str, default=None, 
                        help="Torch device override.")
    parser.add_argument("--saved-model-dir", type=str, default="./Saved_Model", 
                        help="Root directory holding training runs.")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Optional directory for evaluation figures and arrays.")
    parser.add_argument("--query-batch-size", type=int, default=32768, 
                        help="Number of spatial queries per decoder chunk.")

    parser.add_argument("--organization-threshold", type=float, default=0.15,
                        help="Minimum soft weight used when drawing organization edges.")
    parser.add_argument("--topk-me-links", type=int, default=3,
                        help="Number of strongest module-environment links to draw per cylinder.")
    parser.add_argument("--organization-view", choices=["all", "physical", "matrices", "sankey"], default="all",
                        help="Which organization diagnostic view to render.")
    parser.add_argument("--organization-topk-cylinders", type=int, default=3,
                        help="Number of top cylinder memberships to list for each hyperedge.")
    parser.add_argument("--organization-topk-env", type=int, default=5,
                        help="Number of top environment tokens to list for each hyperedge.")
    parser.add_argument("--organization-min-gap", type=float, default=0.08,
                        help="Minimum normalized vertical gap for Sankey node layout.")
    parser.add_argument("--organization-table", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the hyperedge summary table in the physical organization view.")

    return parser.parse_args()

def default_saved_model_dir() -> Path:
    return (DEMO_ROOT / "Saved_Model").resolve()


def resolve_demo_config_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DEMO_ROOT / path).resolve()


def sort_case_ids(case_ids: List[str]) -> List[str]:
    def key_fn(case_id: str):
        try:
            return (0, int(case_id))
        except (TypeError, ValueError):
            return (1, str(case_id))

    return sorted(case_ids, key=key_fn)


def find_latest_run(case_id: str, saved_model_dir: Path) -> Path:
    prefix = f"Case{case_id}_"
    candidates = [p for p in saved_model_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No run directory found in {saved_model_dir} for case_id={case_id}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(run_dir: Path, latest: bool) -> Dict:
    ckpt_path = run_dir / ("latest_model.pt" if latest else "best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def build_model_from_checkpoint(checkpoint: Dict, device: torch.device) -> HypergraphNeuralFieldModel:
    model_cfg = ModelConfig.from_dict(checkpoint["model_config"])
    model = HypergraphNeuralFieldModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def choose_dataset_case(h5_path: Path, dataset_case_id: Optional[str], split: str) -> str:
    with h5py.File(h5_path, "r") as h5_file:
        case_ids = []
        for case_id in sort_case_ids(list(h5_file["cases"].keys())):
            grp = h5_file["cases"][case_id]
            case_split = grp.attrs.get("split", "all")
            if split in {"all", case_split}:
                case_ids.append(case_id)
        if not case_ids:
            raise RuntimeError(f"No dataset cases found for split='{split}' in {h5_path}")
        if dataset_case_id is None:
            return case_ids[0]
        if dataset_case_id not in case_ids:
            raise KeyError(f"dataset_case_id={dataset_case_id} not found in split='{split}'. Available: {case_ids[:8]}...")
        return dataset_case_id


def load_dataset_case(h5_path: Path, case_id: str) -> Dict:
    with h5py.File(h5_path, "r") as h5_file:
        grp = h5_file["cases"][case_id]
        if "canonical_cycle" not in grp or "phase_bin_centers" not in grp:
            raise KeyError(
                f"Case {case_id} does not contain 'canonical_cycle' and 'phase_bin_centers'. "
                "Re-run preprocessing with --save-full-canonical-cycles."
            )
        return {
            "case_id": case_id,
            "split": grp.attrs.get("split", "all"),
            "re": float(grp.attrs["re"]),
            "num_cylinders": int(grp.attrs["num_cylinders"]),
            "dominant_frequency": float(grp.attrs["dominant_frequency"]),
            "cylinder_radius": float(grp.attrs.get("cylinder_radius", 0.5)),
            "centers": np.asarray(grp["cylinder_centers"], dtype=np.float32),
            "x_grid": np.asarray(grp["x_grid"], dtype=np.float32),
            "y_grid": np.asarray(grp["y_grid"], dtype=np.float32),
            "phase_bin_centers": np.asarray(grp["phase_bin_centers"], dtype=np.float32),
            "canonical_cycle": np.asarray(grp["canonical_cycle"], dtype=np.float32),
            "mean_field": np.asarray(grp["mean_field"], dtype=np.float32),
            "rms_field": np.asarray(grp["rms_field"], dtype=np.float32),
        }


def build_structure_tensors(case: Dict, max_num_cylinders: int, device: torch.device) -> Dict[str, torch.Tensor]:
    centers = case["centers"]
    padded = np.zeros((1, max_num_cylinders, 2), dtype=np.float32)
    mask = np.zeros((1, max_num_cylinders), dtype=np.float32)
    padded[0, : centers.shape[0]] = centers
    mask[0, : centers.shape[0]] = 1.0
    return {
        "re_values": torch.tensor([[case["re"]]], dtype=torch.float32, device=device),
        "num_cylinders": torch.tensor([[case["num_cylinders"]]], dtype=torch.float32, device=device),
        "centers": torch.from_numpy(padded).to(device=device),
        "cyl_mask": torch.from_numpy(mask).to(device=device),
    }

def _grid_domain_bounds(case: Dict) -> Dict[str, float]:
    x_grid = case["x_grid"]
    y_grid = case["y_grid"]

    x_min = float(x_grid.min())
    x_max = float(x_grid.max())
    y_min = float(y_grid.min())
    y_max = float(y_grid.max())

    dx = float(np.mean(np.diff(x_grid[0]))) if x_grid.shape[1] > 1 else 0.0
    dy = float(np.mean(np.diff(y_grid[:, 0]))) if y_grid.shape[0] > 1 else 0.0

    lx = (x_max - x_min) + dx
    ly = (y_max - y_min) + dy
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "lx": lx, "ly": ly}


def _env_coords_to_physical(env_coords_norm: np.ndarray, case: Dict) -> np.ndarray:
    bounds = _grid_domain_bounds(case)
    env_xy = env_coords_norm.copy()
    env_xy[:, 0] = bounds["x_min"] + env_xy[:, 0] * bounds["lx"]
    env_xy[:, 1] = bounds["y_min"] + env_xy[:, 1] * bounds["ly"]
    return env_xy


def _coords_norm_to_physical(coords_norm: np.ndarray, case: Dict) -> np.ndarray:
    bounds = _grid_domain_bounds(case)
    coords_xy = coords_norm.copy()
    coords_xy[..., 0] = bounds["x_min"] + coords_xy[..., 0] * bounds["lx"]
    coords_xy[..., 1] = bounds["y_min"] + coords_xy[..., 1] * bounds["ly"]
    return coords_xy


def _as_numpy_first(out: Dict, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if key not in out:
        return default
    value = out[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim >= 3:
        return value[0]
    if value.ndim == 2 and key in {"hyper_strength", "hyper_wake_extent"}:
        return value[0]
    return value


def extract_organization_arrays(out: Dict, case: Dict) -> Dict:
    centers = np.asarray(case["centers"], dtype=np.float32)
    num_cyl = centers.shape[0]
    A_me = _as_numpy_first(out, "A_me")
    A_mh = _as_numpy_first(out, "A_mh")
    A_eh = _as_numpy_first(out, "A_eh")
    env_coords_norm = _as_numpy_first(out, "env_coords")
    hyper_source_norm = _as_numpy_first(out, "hyper_source_coords")
    hyper_wake_norm = _as_numpy_first(out, "hyper_wake_coords", hyper_source_norm)
    hyper_wake_axis = _as_numpy_first(out, "hyper_wake_axis")

    if A_me is None or A_mh is None or A_eh is None or env_coords_norm is None or hyper_source_norm is None:
        raise KeyError("Organization outputs are missing one of A_me, A_mh, A_eh, env_coords, hyper_source_coords.")

    A_me = np.asarray(A_me[:num_cyl], dtype=np.float32)
    A_mh = np.asarray(A_mh[:num_cyl], dtype=np.float32)
    A_eh = np.asarray(A_eh, dtype=np.float32)
    env_coords_norm = np.asarray(env_coords_norm, dtype=np.float32)
    hyper_source_norm = np.asarray(hyper_source_norm, dtype=np.float32)
    hyper_wake_norm = np.asarray(hyper_wake_norm, dtype=np.float32)
    num_hyper = A_eh.shape[1]

    if hyper_wake_axis is None:
        hyper_wake_axis = hyper_wake_norm - hyper_source_norm
    hyper_wake_axis = np.asarray(hyper_wake_axis, dtype=np.float32).reshape(num_hyper, 2)

    hyper_wake_extent = _as_numpy_first(out, "hyper_wake_extent")
    if hyper_wake_extent is None:
        hyper_wake_extent = np.full((num_hyper,), np.nan, dtype=np.float32)
    hyper_wake_extent = np.asarray(hyper_wake_extent, dtype=np.float32).reshape(-1)[:num_hyper]

    hyper_strength = _as_numpy_first(out, "hyper_strength")
    if hyper_strength is None:
        module_mass = A_mh.sum(axis=0) / max(float(num_cyl), 1.0)
        env_mass = A_eh.mean(axis=0)
        hyper_strength = 0.5 * (module_mass + env_mass)
    hyper_strength = np.asarray(hyper_strength, dtype=np.float32).reshape(-1)[:num_hyper]

    env_xy = _env_coords_to_physical(env_coords_norm, case)
    hyper_source_xy = _coords_norm_to_physical(hyper_source_norm, case)
    hyper_wake_xy = _coords_norm_to_physical(hyper_wake_norm, case)
    token_group = np.argmax(A_eh, axis=1)
    token_conf = np.max(A_eh, axis=1)
    bounds = _grid_domain_bounds(case)
    colors = plt.get_cmap("tab10")(np.arange(num_hyper) % 10)

    return {
        "centers": centers,
        "cylinder_radius": float(case.get("cylinder_radius", 0.5)),
        "A_me": A_me,
        "A_mh": A_mh,
        "A_eh": A_eh,
        "env_coords_norm": env_coords_norm,
        "env_xy": env_xy,
        "hyper_source_norm": hyper_source_norm,
        "hyper_wake_norm": hyper_wake_norm,
        "hyper_source_xy": hyper_source_xy,
        "hyper_wake_xy": hyper_wake_xy,
        "hyper_wake_axis": hyper_wake_axis,
        "hyper_wake_extent": hyper_wake_extent,
        "hyper_strength": hyper_strength,
        "token_group": token_group,
        "token_conf": token_conf,
        "bounds": bounds,
        "colors": colors,
    }


def periodic_min_image_delta_physical(p0: np.ndarray, p1: np.ndarray, bounds: Dict[str, float]) -> tuple[float, float]:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    lx = float(bounds["lx"])
    ly = float(bounds["ly"])
    if lx > 0.0:
        dx = ((dx + 0.5 * lx) % lx) - 0.5 * lx
    if ly > 0.0:
        dy = ((dy + 0.5 * ly) % ly) - 0.5 * ly
    return dx, dy


def periodic_shifted_endpoint(p0: np.ndarray, p1: np.ndarray, bounds: Dict[str, float]) -> np.ndarray:
    dx, dy = periodic_min_image_delta_physical(p0, p1, bounds)
    p0 = np.asarray(p0, dtype=np.float64)
    return np.asarray([p0[0] + dx, p0[1] + dy], dtype=np.float64)


def _wrap_point_to_bounds(point: np.ndarray, bounds: Dict[str, float]) -> np.ndarray:
    wrapped = np.asarray(point, dtype=np.float64).copy()
    wrapped[0] = bounds["x_min"] + ((wrapped[0] - bounds["x_min"]) % bounds["lx"])
    wrapped[1] = bounds["y_min"] + ((wrapped[1] - bounds["y_min"]) % bounds["ly"])
    return wrapped


def draw_periodic_segment(ax, p0: np.ndarray, p1: np.ndarray, bounds: Dict[str, float], **plot_kwargs) -> None:
    shifted = periodic_shifted_endpoint(p0, p1, bounds)
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    wraps = bool(np.linalg.norm(shifted - p1) > 1e-6)
    kwargs = dict(plot_kwargs)
    if wraps:
        kwargs["alpha"] = min(float(kwargs.get("alpha", 1.0)), 0.55)
    ax.plot([p0[0], shifted[0]], [p0[1], shifted[1]], **kwargs)
    if wraps:
        mid = _wrap_point_to_bounds(0.5 * (p0 + shifted), bounds)
        ax.scatter(
            [mid[0]],
            [mid[1]],
            marker="x",
            s=24,
            c=[kwargs.get("color", "k")],
            linewidths=0.8,
            alpha=0.75,
            zorder=kwargs.get("zorder", 3) + 1,
        )


def topk_cylinder_members(A_mh: np.ndarray, k: int, top_n: int = 3) -> List[Dict]:
    top_idx = np.argsort(-A_mh[:, k])[:max(0, top_n)]
    return [{"id": int(i), "weight": float(A_mh[i, k])} for i in top_idx]


def topk_env_members(A_eh: np.ndarray, k: int, env_xy: np.ndarray, top_n: int = 5) -> List[Dict]:
    top_idx = np.argsort(-A_eh[:, k])[:max(0, top_n)]
    return [
        {"id": int(j), "weight": float(A_eh[j, k]), "x": float(env_xy[j, 0]), "y": float(env_xy[j, 1])}
        for j in top_idx
    ]


def compute_hyperedge_summary(
    org: Dict,
    *,
    case_id: str,
    tau_value: float,
    topk_cylinders: int = 3,
    topk_env: int = 5,
) -> List[Dict]:
    A_mh = org["A_mh"]
    A_eh = org["A_eh"]
    env_xy = org["env_xy"]
    token_group = org["token_group"]
    summaries = []
    for k in range(A_eh.shape[1]):
        top_cyl = topk_cylinder_members(A_mh, k, top_n=topk_cylinders)
        top_env = topk_env_members(A_eh, k, env_xy, top_n=topk_env)
        env_token_count = int(np.sum(token_group == k))
        env_mass_sum = float(np.sum(A_eh[:, k]))
        summaries.append(
            {
                "case_id": str(case_id),
                "tau": float(tau_value),
                "hyperedge_id": int(k),
                "strength": float(org["hyper_strength"][k]),
                "source": {
                    "x": float(org["hyper_source_xy"][k, 0]),
                    "y": float(org["hyper_source_xy"][k, 1]),
                },
                "wake": {
                    "x": float(org["hyper_wake_xy"][k, 0]),
                    "y": float(org["hyper_wake_xy"][k, 1]),
                },
                "wake_axis": {
                    "x": float(org["hyper_wake_axis"][k, 0]),
                    "y": float(org["hyper_wake_axis"][k, 1]),
                },
                "wake_extent": float(org["hyper_wake_extent"][k]),
                "top_cylinders": top_cyl,
                "env_token_count": env_token_count,
                "env_mass_sum": env_mass_sum,
                "env_mass_mean": float(np.mean(A_eh[:, k])),
                "top_env_tokens": top_env,
            }
        )
    return summaries


def _format_members(prefix: str, members: List[Dict], limit: Optional[int] = None) -> str:
    shown = members if limit is None else members[:limit]
    return ", ".join(f"{prefix}{m['id']}:{m['weight']:.2f}" for m in shown)


def write_organization_summary(save_csv: Path, save_json: Path, summaries: List[Dict]) -> None:
    rows = []
    for item in summaries:
        rows.append(
            {
                "case_id": item["case_id"],
                "tau": item["tau"],
                "hyperedge_id": item["hyperedge_id"],
                "strength": item["strength"],
                "source_x": item["source"]["x"],
                "source_y": item["source"]["y"],
                "wake_x": item["wake"]["x"],
                "wake_y": item["wake"]["y"],
                "axis_x": item["wake_axis"]["x"],
                "axis_y": item["wake_axis"]["y"],
                "extent": item["wake_extent"],
                "env_token_count": item["env_token_count"],
                "env_mass_sum": item["env_mass_sum"],
                "env_mass_mean": item["env_mass_mean"],
                "top_cylinders": ",".join(f"C{m['id']}" for m in item["top_cylinders"]),
                "top_cylinder_weights": ",".join(f"{m['weight']:.6g}" for m in item["top_cylinders"]),
                "top_env_tokens": ",".join(f"E{m['id']}" for m in item["top_env_tokens"]),
                "top_env_weights": ",".join(f"{m['weight']:.6g}" for m in item["top_env_tokens"]),
            }
        )
    with save_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with save_json.open("w", encoding="utf-8") as f:
        json.dump({"hyperedges": summaries}, f, indent=2)


def render_organization_physical_summary(
    save_path: Path,
    org: Dict,
    summaries: List[Dict],
    case: Dict,
    *,
    threshold: float = 0.15,
    topk_me_links: int = 3,
    show_table: bool = True,
) -> None:
    bounds = org["bounds"]
    extent = (bounds["x_min"], bounds["x_min"] + bounds["lx"], bounds["y_min"], bounds["y_min"] + bounds["ly"])
    num_hyper = org["A_eh"].shape[1]
    num_cyl = org["centers"].shape[0]
    fig, axes = plt.subplots(
        1,
        2 if show_table else 1,
        figsize=(16, 7) if show_table else (9, 7),
        dpi=150,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]} if show_table else None,
    )
    ax = axes[0] if show_table else axes
    ax.set_title("Physical organizer overlay")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for k in range(num_hyper):
        mask = org["token_group"] == k
        if np.any(mask):
            ax.scatter(
                org["env_xy"][mask, 0],
                org["env_xy"][mask, 1],
                s=16 + 80 * org["token_conf"][mask],
                c=[org["colors"][k]],
                alpha=0.12 + 0.62 * org["token_conf"][mask],
                linewidths=0.0,
                zorder=1,
            )

    arrow_len = 0.09 * min(bounds["lx"], bounds["ly"])
    axis_phys = np.stack(
        [org["hyper_wake_axis"][:, 0] * bounds["lx"], org["hyper_wake_axis"][:, 1] * bounds["ly"]],
        axis=-1,
    )
    axis_norm = np.linalg.norm(axis_phys, axis=1, keepdims=True)
    axis_phys = np.where(axis_norm > 1e-8, axis_phys / np.maximum(axis_norm, 1e-8), np.array([[1.0, 0.0]]))

    for k in range(num_hyper):
        color = org["colors"][k]
        draw_periodic_segment(
            ax,
            org["hyper_source_xy"][k],
            org["hyper_wake_xy"][k],
            bounds,
            linestyle=":",
            color=color,
            alpha=0.75,
            linewidth=1.4,
            zorder=3,
        )
        ax.scatter(org["hyper_source_xy"][k, 0], org["hyper_source_xy"][k, 1], s=100, marker="X", c=[color], edgecolors="k", linewidths=0.7, zorder=5)
        ax.scatter(org["hyper_wake_xy"][k, 0], org["hyper_wake_xy"][k, 1], s=130 + 180 * float(org["hyper_strength"][k]), marker="*", c=[color], edgecolors="k", linewidths=0.7, zorder=5)
        ax.arrow(
            org["hyper_wake_xy"][k, 0],
            org["hyper_wake_xy"][k, 1],
            arrow_len * axis_phys[k, 0],
            arrow_len * axis_phys[k, 1],
            color=color,
            width=0.0,
            head_width=0.10,
            head_length=0.18,
            length_includes_head=True,
            zorder=6,
        )
        ax.text(org["hyper_source_xy"][k, 0], org["hyper_source_xy"][k, 1], f"S{k}", fontsize=8, ha="right", va="top", zorder=7)
        ax.text(org["hyper_wake_xy"][k, 0], org["hyper_wake_xy"][k, 1], f"H{k}", fontsize=9, ha="left", va="bottom", zorder=7)

    for i in range(num_cyl):
        top_idx = np.argsort(-org["A_me"][i])[:max(0, topk_me_links)]
        for j in top_idx:
            w = float(org["A_me"][i, j])
            if w < 0.5 * threshold:
                continue
            k = int(org["token_group"][j])
            draw_periodic_segment(
                ax,
                org["centers"][i],
                org["env_xy"][j],
                bounds,
                color=org["colors"][k],
                alpha=0.08 + 0.35 * w,
                linewidth=0.4 + 1.5 * w,
                zorder=2,
            )

    for i in range(num_cyl):
        for k in range(num_hyper):
            w = float(org["A_mh"][i, k])
            if w < threshold:
                continue
            draw_periodic_segment(
                ax,
                org["centers"][i],
                org["hyper_source_xy"][k],
                bounds,
                linestyle="--",
                color=org["colors"][k],
                alpha=0.18 + 0.62 * w,
                linewidth=0.8 + 3.0 * w,
                zorder=4,
            )

    for i, (cx, cy) in enumerate(org["centers"]):
        ax.add_patch(plt.Circle((cx, cy), org["cylinder_radius"], fill=False, color="k", lw=1.2, zorder=8))
        ax.text(cx, cy, f"C{i}", fontsize=8, ha="center", va="center", zorder=9)

    legend_items = [
        Line2D([0], [0], marker="o", color="k", markerfacecolor="white", lw=0, label="cylinder"),
        Line2D([0], [0], marker="o", color="gray", lw=0, label="env token"),
        Line2D([0], [0], marker="X", color="k", lw=0, label="source center"),
        Line2D([0], [0], marker="*", color="k", lw=0, label="wake center"),
        Line2D([0], [0], color="k", lw=1.4, label="wake axis"),
        Line2D([0], [0], color="k", lw=1.4, linestyle="--", label="cylinder->hyperedge"),
        Line2D([0], [0], color="k", lw=1.4, linestyle=":", label="source->wake"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.85)
    ax.text(0.01, 0.01, "Periodic shortest-image links are used.\nWake arrows show learned wake-axis direction in periodic coordinates.", transform=ax.transAxes, fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

    if show_table:
        table_ax = axes[1]
        table_ax.axis("off")
        table_ax.set_title("Hyperedge summary")
        lines = []
        for item in summaries:
            k = item["hyperedge_id"]
            lines.extend(
                [
                    f"H{k}  strength={item['strength']:.3f}",
                    f"  src=({item['source']['x']:.2f},{item['source']['y']:.2f})  wake=({item['wake']['x']:.2f},{item['wake']['y']:.2f})",
                    f"  axis=({item['wake_axis']['x']:.2f},{item['wake_axis']['y']:.2f})  extent={item['wake_extent']:.3f}",
                    f"  cyl: {_format_members('C', item['top_cylinders'])}",
                    f"  env: n={item['env_token_count']}  mass={item['env_mass_sum']:.2f}  top {_format_members('E', item['top_env_tokens'], limit=3)}",
                    "",
                ]
            )
        table_ax.text(0.0, 0.98, "\n".join(lines), ha="left", va="top", fontsize=8.2, family="monospace", transform=table_ax.transAxes)

    fig.suptitle(f"Case {case['case_id']} | tau organization diagnostics")
    fig.savefig(save_path)
    plt.close(fig)


def render_organization_matrices(save_path: Path, org: Dict, summaries: List[Dict]) -> None:
    A_mh = org["A_mh"]
    A_eh = org["A_eh"]
    num_hyper = A_eh.shape[1]
    cols = max(2, int(np.ceil(np.sqrt(num_hyper))))
    rows = int(np.ceil(num_hyper / cols))
    fig = plt.figure(figsize=(max(13, cols * 3.1), 7.2 + rows * 2.7), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(2 + rows, cols)
    ax_mh = fig.add_subplot(gs[0, : max(1, cols // 2)])
    ax_eh = fig.add_subplot(gs[0, max(1, cols // 2) :])

    im = ax_mh.imshow(A_mh, aspect="auto", vmin=0.0, vmax=max(1.0, float(A_mh.max())), cmap="viridis")
    ax_mh.set_title("Module -> Hyperedge assignment A_mh")
    ax_mh.set_xlabel("hyperedge")
    ax_mh.set_ylabel("cylinder")
    ax_mh.set_xticks(np.arange(num_hyper), labels=[f"H{k}" for k in range(num_hyper)])
    ax_mh.set_yticks(np.arange(A_mh.shape[0]), labels=[f"C{i}" for i in range(A_mh.shape[0])])
    if A_mh.size <= 80:
        for i in range(A_mh.shape[0]):
            for k in range(num_hyper):
                ax_mh.text(k, i, f"{A_mh[i, k]:.2f}", ha="center", va="center", fontsize=7, color="white" if A_mh[i, k] > 0.5 else "black")
    fig.colorbar(im, ax=ax_mh, fraction=0.046, pad=0.03)

    im = ax_eh.imshow(A_eh, aspect="auto", vmin=0.0, vmax=max(1.0, float(A_eh.max())), cmap="viridis")
    ax_eh.set_title("Environment -> Hyperedge assignment A_eh")
    ax_eh.set_xlabel("hyperedge")
    ax_eh.set_ylabel("env token index")
    ax_eh.set_xticks(np.arange(num_hyper), labels=[f"H{k}" for k in range(num_hyper)])
    if A_eh.shape[0] <= 32:
        ax_eh.set_yticks(np.arange(A_eh.shape[0]))
    else:
        tick_idx = np.linspace(0, A_eh.shape[0] - 1, min(9, A_eh.shape[0])).astype(int)
        ax_eh.set_yticks(tick_idx, labels=[str(i) for i in tick_idx])
    fig.colorbar(im, ax=ax_eh, fraction=0.046, pad=0.03)

    bounds = org["bounds"]
    extent = (bounds["x_min"], bounds["x_min"] + bounds["lx"], bounds["y_min"], bounds["y_min"] + bounds["ly"])
    for k in range(num_hyper):
        ax = fig.add_subplot(gs[1 + (k // cols), k % cols])
        weights = A_eh[:, k]
        sc = ax.scatter(org["env_xy"][:, 0], org["env_xy"][:, 1], c=weights, s=16 + 70 * weights, cmap="viridis", vmin=0.0, vmax=max(1.0, float(A_eh.max())), linewidths=0.0)
        for cx, cy in org["centers"]:
            ax.add_patch(plt.Circle((cx, cy), org["cylinder_radius"], fill=False, color="k", lw=0.8))
        ax.scatter(org["hyper_source_xy"][k, 0], org["hyper_source_xy"][k, 1], marker="X", s=70, c=[org["colors"][k]], edgecolors="k")
        ax.scatter(org["hyper_wake_xy"][k, 0], org["hyper_wake_xy"][k, 1], marker="*", s=100, c=[org["colors"][k]], edgecolors="k")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_title(f"H{k}: strength={summaries[k]['strength']:.2f}, n={summaries[k]['env_token_count']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if num_hyper:
        fig.colorbar(sc, ax=fig.axes[-num_hyper:], fraction=0.025, pad=0.01)
    fig.suptitle("Organization matrix diagnostics")
    fig.savefig(save_path)
    plt.close(fig)


def spread_positions(desired_y: np.ndarray, min_gap: float = 0.08, y_min: float = 0.08, y_max: float = 0.92) -> np.ndarray:
    desired_y = np.asarray(desired_y, dtype=np.float64)
    n = desired_y.size
    if n == 0:
        return desired_y
    if n == 1:
        return np.asarray([float(np.clip(desired_y[0], y_min, y_max))])
    order = np.argsort(desired_y)
    sorted_y = np.clip(desired_y[order], y_min, y_max)
    span = y_max - y_min
    gap = min(float(min_gap), span / max(n - 1, 1))
    for i in range(1, n):
        sorted_y[i] = max(sorted_y[i], sorted_y[i - 1] + gap)
    overflow = sorted_y[-1] - y_max
    if overflow > 0:
        sorted_y -= overflow
    for i in range(n - 2, -1, -1):
        sorted_y[i] = min(sorted_y[i], sorted_y[i + 1] - gap)
    underflow = y_min - sorted_y[0]
    if underflow > 0:
        sorted_y += underflow
    sorted_y = np.clip(sorted_y, y_min, y_max)
    out = np.empty_like(sorted_y)
    out[order] = sorted_y
    return out


def _curved_edge(ax, p0: tuple[float, float], p1: tuple[float, float], *, color, linewidth: float, alpha: float, rad: float) -> None:
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        shrinkA=8,
        shrinkB=8,
        zorder=1,
    )
    ax.add_patch(patch)


def render_organization_sankey(save_path: Path, org: Dict, summaries: List[Dict], *, threshold: float = 0.15, min_gap: float = 0.08) -> None:
    centers = org["centers"]
    A_mh = org["A_mh"]
    A_eh = org["A_eh"]
    bounds = org["bounds"]
    num_cyl = centers.shape[0]
    num_hyper = A_eh.shape[1]
    fig, ax = plt.subplots(figsize=(13, 7), dpi=150, constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Sankey-style organizer topology\nC_i = cylinder/module, H_k = interaction hyperedge, EnvGroup_k = dominated environment tokens")

    cyl_desired = np.clip((centers[:, 1] - bounds["y_min"]) / max(bounds["ly"], 1e-6), 0.08, 0.92)
    hyper_desired = np.clip((org["hyper_wake_xy"][:, 1] - bounds["y_min"]) / max(bounds["ly"], 1e-6), 0.08, 0.92)
    env_desired = np.clip(hyper_desired + 0.035 * np.where(np.arange(num_hyper) % 2 == 0, 1.0, -1.0), 0.08, 0.92)
    cyl_y = spread_positions(cyl_desired, min_gap=min_gap * 0.75)
    hyper_y = spread_positions(hyper_desired, min_gap=min_gap)
    env_y = spread_positions(env_desired, min_gap=min_gap)

    module_pos = {i: (0.10, float(cyl_y[i])) for i in range(num_cyl)}
    hyper_pos = {k: (0.50, float(hyper_y[k])) for k in range(num_hyper)}
    env_pos = {k: (0.88, float(env_y[k])) for k in range(num_hyper)}

    for i in range(num_cyl):
        for k in range(num_hyper):
            w = float(A_mh[i, k])
            if w < threshold:
                continue
            _curved_edge(
                ax,
                module_pos[i],
                hyper_pos[k],
                color=org["colors"][k],
                linewidth=0.7 + 4.2 * w,
                alpha=min(0.95, 0.15 + 0.85 * w),
                rad=0.10 if (i + k) % 2 == 0 else -0.10,
            )

    for k in range(num_hyper):
        mass = float(A_eh[:, k].mean())
        n_tokens = summaries[k]["env_token_count"]
        width = 1.0 + 8.0 * max(mass, n_tokens / max(float(A_eh.shape[0]), 1.0))
        _curved_edge(ax, hyper_pos[k], env_pos[k], color=org["colors"][k], linewidth=width, alpha=0.35 + 0.5 * min(1.0, width / 8.0), rad=-0.08)

    bbox = {"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.0}
    for i in range(num_cyl):
        x, y = module_pos[i]
        ax.scatter([x], [y], s=130, c="white", edgecolors="k", zorder=4)
        ax.text(x - 0.035, y, f"C{i}", ha="right", va="center", fontsize=9, bbox=bbox, zorder=5)

    for k in range(num_hyper):
        x, y = hyper_pos[k]
        ax.scatter([x], [y], s=180, marker="*", c=[org["colors"][k]], edgecolors="k", zorder=4)
        ax.text(x, y + 0.035, f"H{k}", ha="center", va="bottom", fontsize=9, bbox=bbox, zorder=5)

    for k in range(num_hyper):
        x, y = env_pos[k]
        ax.scatter([x], [y], s=145, marker="s", c=[org["colors"][k]], edgecolors="k", zorder=4)
        ax.text(
            x + 0.035,
            y,
            f"EnvGroup_{k}\nn={summaries[k]['env_token_count']}\nmass={summaries[k]['env_mass_sum']:.1f}",
            ha="left",
            va="center",
            fontsize=8,
            bbox=bbox,
            zorder=5,
        )

    ax.text(0.03, 0.965, "Line width is proportional to soft assignment weight.", ha="left", va="top", fontsize=9)
    _curved_edge(ax, (0.35, 0.055), (0.44, 0.055), color="0.25", linewidth=0.7 + 4.2 * 0.2, alpha=0.75, rad=0.0)
    _curved_edge(ax, (0.55, 0.055), (0.64, 0.055), color="0.25", linewidth=0.7 + 4.2 * 0.8, alpha=0.75, rad=0.0)
    ax.text(0.395, 0.025, "w=0.2", ha="center", va="center", fontsize=8)
    ax.text(0.595, 0.025, "w=0.8", ha="center", va="center", fontsize=8)
    ax.text(0.10, 0.99, "Modules", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(0.50, 0.99, "Interaction hyperedges", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(0.88, 0.99, "Environment groups", ha="center", va="top", fontsize=10, weight="bold")
    fig.savefig(save_path)
    plt.close(fig)


def render_soft_organization(
    output_dir: Path,
    out: Dict,
    case: Dict,
    *,
    tau_value: float,
    phase_idx: int,
    threshold: float = 0.15,
    topk_me_links: int = 3,
    organization_view: str = "all",
    topk_cylinders: int = 3,
    topk_env: int = 5,
    min_gap: float = 0.08,
    show_table: bool = True,
) -> Dict[str, str]:
    org = extract_organization_arrays(out, case)
    summaries = compute_hyperedge_summary(
        org,
        case_id=case["case_id"],
        tau_value=tau_value,
        topk_cylinders=topk_cylinders,
        topk_env=topk_env,
    )
    base = f"case_{case['case_id']}_tau_{phase_idx:03d}"
    csv_path = output_dir / f"organization_summary_{base}.csv"
    json_path = output_dir / f"organization_summary_{base}.json"
    write_organization_summary(csv_path, json_path, summaries)

    paths = {"summary_csv": str(csv_path), "summary_json": str(json_path)}
    if organization_view in {"all", "physical"}:
        path = output_dir / f"organization_physical_{base}.png"
        render_organization_physical_summary(path, org, summaries, case, threshold=threshold, topk_me_links=topk_me_links, show_table=show_table)
        paths["physical"] = str(path)
    if organization_view in {"all", "matrices"}:
        path = output_dir / f"organization_matrices_{base}.png"
        render_organization_matrices(path, org, summaries)
        paths["matrices"] = str(path)
    if organization_view in {"all", "sankey"}:
        path = output_dir / f"organization_sankey_{base}.png"
        render_organization_sankey(path, org, summaries, threshold=threshold, min_gap=min_gap)
        paths["sankey"] = str(path)
    return paths

def render_quicklook(
    save_path: Path,
    pred_field: np.ndarray,
    gt_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    tau_value: float,
    case: Dict,
) -> None:
    channels = ["u", "v", "p", "omega"]
    cmaps = ["coolwarm", "coolwarm", "magma", "RdBu_r"]
    extent = (float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max()))

    fig, axes = plt.subplots(4, 3, figsize=(13, 14), dpi=150, constrained_layout=True)
    cylinder_radius = float(case.get("cylinder_radius", 0.5))
    for i, (name, cmap) in enumerate(zip(channels, cmaps)):
        pred = pred_field[..., i]
        gt = gt_field[..., i]
        err = pred - gt
        for j, (arr, title, use_cmap) in enumerate(
            [
                (gt, f"GT {name}", cmap),
                (pred, f"Pred {name}", cmap),
                (err, f"Error {name}", "coolwarm"),
            ]
        ):
            ax = axes[i, j]
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=use_cmap, aspect="equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        for cx, cy in case["centers"]:
            for j in range(3):
                circle = plt.Circle((cx, cy), cylinder_radius, fill=False, color="k", lw=1.0)
                axes[i, j].add_patch(circle)

    fig.suptitle(f"Case {case['case_id']} | split={case['split']} | tau={tau_value:.3f}")
    fig.savefig(save_path)
    plt.close(fig)


def render_animation(
    save_path: Path,
    model: torch.nn.Module,
    structure: Dict[str, torch.Tensor],
    case: Dict,
    device: torch.device,
    query_batch_size: int = 32768,
) -> None:
    print("Generating canonical cycle animation...")
    x_grid_t = torch.from_numpy(case["x_grid"]).to(device)
    y_grid_t = torch.from_numpy(case["y_grid"]).to(device)
    phase_bins = case["phase_bin_centers"]
    
    # Pre-compute all frames to find global min/max for stable colors
    frames_pred, frames_gt = [], []
    for frame_idx, tau_val in enumerate(phase_bins):
        tau_t = torch.tensor([tau_val], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model.reconstruct_full_grid(structure, x_grid_t, y_grid_t, tau_t, query_batch_size)
        frames_pred.append(out["pred_field"][0].detach().cpu().numpy())
        frames_gt.append(case["canonical_cycle"][frame_idx])
    
    # Omega is channel 3
    omega_preds = np.stack([f[..., 3] for f in frames_pred])
    omega_gts = np.stack([f[..., 3] for f in frames_gt])
    vmax = float(np.percentile(np.abs(omega_gts), 99.0))
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120, constrained_layout=True)
    extent = (float(case["x_grid"].min()), float(case["x_grid"].max()), float(case["y_grid"].min()), float(case["y_grid"].max()))
    
    im_gt = axes[0].imshow(omega_gts[0], origin="lower", extent=extent, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title("GT Vorticity")
    im_pred = axes[1].imshow(omega_preds[0], origin="lower", extent=extent, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title("Pred Vorticity")

    for ax in axes:
        for cx, cy in case["centers"]:
            ax.add_patch(plt.Circle((cx, cy), float(case.get("cylinder_radius", 0.5)), fill=False, color="k", lw=1.0))

    def update(frame_idx):
        im_gt.set_data(omega_gts[frame_idx])
        im_pred.set_data(omega_preds[frame_idx])
        fig.suptitle(f"Phase Tau: {phase_bins[frame_idx]:.3f}")
        return [im_gt, im_pred]

    anim = FuncAnimation(fig, update, frames=len(phase_bins), blit=False)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)

def main() -> None:
    args = parse_args()
    saved_model_dir = resolve_demo_config_path(args.saved_model_dir)
    run_dir = find_latest_run(args.case_id, saved_model_dir)
    checkpoint = load_checkpoint(run_dir, latest=args.latest)
    device = select_device(args.device)
    model = build_model_from_checkpoint(checkpoint, device=device)

    train_cfg = checkpoint["config"]
    packed_h5_path = resolve_demo_config_path(train_cfg["dataset"]["packed_h5_path"])
    dataset_case_id = choose_dataset_case(packed_h5_path, args.dataset_case_id, args.dataset_split)
    case = load_dataset_case(packed_h5_path, dataset_case_id)

    tau_value = float(case["phase_bin_centers"][0]) if args.tau is None else float(args.tau)
    phase_idx = int(np.argmin(np.abs(case["phase_bin_centers"] - tau_value)))
    tau_value = float(case["phase_bin_centers"][phase_idx])

    structure = build_structure_tensors(case, max_num_cylinders=model.cfg.max_num_cylinders, device=device)
    x_grid_t = torch.from_numpy(case["x_grid"]).to(device=device)
    y_grid_t = torch.from_numpy(case["y_grid"]).to(device=device)
    tau_t = torch.tensor([tau_value], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model.reconstruct_full_grid(structure, x_grid_t, y_grid_t, tau=tau_t, query_batch_size=args.query_batch_size)

    pred_field = out["pred_field"][0].detach().cpu().numpy()
    gt_field = case["canonical_cycle"][phase_idx]

    mse = float(np.mean((pred_field - gt_field) ** 2))
    freq_pred = float(out["freq_pred"].reshape(-1)[0].detach().cpu().item()) if "freq_pred" in out else float("nan")
    print(
        f"Loaded run: {run_dir.name}\n"
        f"Dataset case: {case['case_id']} | split={case['split']} | tau={tau_value:.3f} | phase_idx={phase_idx}\n"
        f"Field MSE: {mse:.6e}\n"
        f"Predicted frequency: {freq_pred:.6e} | GT frequency: {case['dominant_frequency']:.6e}"
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = run_dir / "evaluation" / f"CaseMode_{args.case_id}_Case_Data_{case['case_id']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / f"quicklook_case_{case['case_id']}_tau_{phase_idx:03d}.png"
    render_quicklook(fig_path, pred_field, gt_field, case["x_grid"], case["y_grid"], tau_value, case)

    org_paths = render_soft_organization(
        output_dir,
        out,
        case,
        tau_value=tau_value,
        phase_idx=phase_idx,
        threshold=args.organization_threshold,
        topk_me_links=args.topk_me_links,
        organization_view=args.organization_view,
        topk_cylinders=args.organization_topk_cylinders,
        topk_env=args.organization_topk_env,
        min_gap=args.organization_min_gap,
        show_table=args.organization_table,
    )

    gif_path = output_dir / f"animation_case_{case['case_id']}_tau_{phase_idx:03d}.gif"
    render_animation(gif_path, model, structure, case, device, args.query_batch_size)

    org_arrays = extract_organization_arrays(out, case)
    np.savez_compressed(
        output_dir / f"reconstruction_case_{case['case_id']}_tau_{phase_idx:03d}.npz",
        pred_field=pred_field.astype(np.float32),
        gt_field=gt_field.astype(np.float32),
        x_grid=case["x_grid"].astype(np.float32),
        y_grid=case["y_grid"].astype(np.float32),
        tau=np.asarray([tau_value], dtype=np.float32),
        predicted_frequency=np.asarray([freq_pred], dtype=np.float32),
        gt_frequency=np.asarray([case["dominant_frequency"]], dtype=np.float32),
        hyper_source_coords=org_arrays["hyper_source_norm"].astype(np.float32),
        hyper_wake_coords=org_arrays["hyper_wake_norm"].astype(np.float32),
        hyper_wake_axis=org_arrays["hyper_wake_axis"].astype(np.float32),
        hyper_wake_extent=org_arrays["hyper_wake_extent"].astype(np.float32),
        hyper_strength=org_arrays["hyper_strength"].astype(np.float32),
    )

    with (output_dir / f"evaluation_summary_case_{case['case_id']}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "checkpoint": "latest_model.pt" if args.latest else "best_model.pt",
                "decoder_type": model.cfg.decoder_type,
                "dataset_case_id": case["case_id"],
                "split": case["split"],
                "tau": tau_value,
                "phase_idx": phase_idx,
                "field_mse": mse,
                "predicted_frequency": freq_pred,
                "gt_frequency": case["dominant_frequency"],
                "num_hyperedges": int(out["A_eh"].shape[-1]),
                "quicklook_path": str(fig_path),
                "organization_paths": org_paths,
            },
            f,
            indent=2,
        )

    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
