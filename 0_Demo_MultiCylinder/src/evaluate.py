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
python src/evaluate.py --case-id 0004 --dataset-case-id 0161 

"""

import argparse
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

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


def _compute_hyperedge_centroids(env_xy: np.ndarray, A_eh: np.ndarray) -> np.ndarray:
    """Weighted physical centroids of hyperedges in environment space."""
    num_hyper = A_eh.shape[1]
    centroids = np.zeros((num_hyper, 2), dtype=np.float32)
    for k in range(num_hyper):
        w = A_eh[:, k]
        w_sum = float(np.sum(w)) + 1e-8
        centroids[k] = np.sum(env_xy * w[:, None], axis=0) / w_sum
    return centroids


def render_soft_organization(
    save_path: Path,
    out: Dict,
    case: Dict,
    *,
    threshold: float = 0.15,
    topk_me_links: int = 3,
) -> None:
    """
    Visualize the learned soft organization in two views:
      (1) domain-space soft overlay
      (2) abstract tripartite hypergraph
    """
    centers = np.asarray(case["centers"], dtype=np.float32)
    num_cyl = centers.shape[0]
    cylinder_radius = float(case.get("cylinder_radius", 0.5))

    A_me = out["A_me"][0].detach().cpu().numpy()[:num_cyl]         # [N, M_env]
    A_mh = out["A_mh"][0].detach().cpu().numpy()[:num_cyl]         # [N, K_h]
    A_eh = out["A_eh"][0].detach().cpu().numpy()                   # [M_env, K_h]
    env_coords_norm = out["env_coords"][0].detach().cpu().numpy()  # [M_env, 2]

    env_xy = _env_coords_to_physical(env_coords_norm, case)
    centroids = _compute_hyperedge_centroids(env_xy, A_eh)

    token_group = np.argmax(A_eh, axis=1)
    token_conf = np.max(A_eh, axis=1)

    num_hyper = A_eh.shape[1]
    colors = plt.get_cmap("tab10")(np.arange(num_hyper) % 10)

    bounds = _grid_domain_bounds(case)
    extent = (bounds["x_min"], bounds["x_min"] + bounds["lx"], bounds["y_min"], bounds["y_min"] + bounds["ly"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), dpi=150, constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel 1: physical-domain overlay
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.set_title("Soft organization in physical domain")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Environment tokens colored by dominant hyperedge assignment.
    for k in range(num_hyper):
        mask = token_group == k
        if np.any(mask):
            ax.scatter(
                env_xy[mask, 0],
                env_xy[mask, 1],
                s=20 + 80 * token_conf[mask],
                c=[colors[k]],
                alpha=0.15 + 0.65 * token_conf[mask],
                linewidths=0.0,
            )

    # Hyperedge centroids.
    for k in range(num_hyper):
        ax.scatter(
            centroids[k, 0],
            centroids[k, 1],
            s=180,
            marker="*",
            c=[colors[k]],
            edgecolors="k",
            linewidths=0.7,
            zorder=4,
        )
        ax.text(
            centroids[k, 0],
            centroids[k, 1],
            f"H{k}",
            fontsize=9,
            ha="left",
            va="bottom",
            color="k",
            zorder=5,
        )

    # Strong cylinder -> environment links from A_me.
    for i in range(num_cyl):
        top_idx = np.argsort(-A_me[i])[:max(1, topk_me_links)]
        for j in top_idx:
            w = float(A_me[i, j])
            if w < 0.5 * threshold:
                continue
            k = int(token_group[j])
            ax.plot(
                [centers[i, 0], env_xy[j, 0]],
                [centers[i, 1], env_xy[j, 1]],
                color=colors[k],
                alpha=0.10 + 0.45 * w,
                linewidth=0.6 + 2.0 * w,
                zorder=1,
            )

    # Strong cylinder -> hyperedge links from A_mh.
    for i in range(num_cyl):
        for k in range(num_hyper):
            w = float(A_mh[i, k])
            if w < threshold:
                continue
            ax.plot(
                [centers[i, 0], centroids[k, 0]],
                [centers[i, 1], centroids[k, 1]],
                linestyle="--",
                color=colors[k],
                alpha=0.15 + 0.65 * w,
                linewidth=0.8 + 2.8 * w,
                zorder=2,
            )

    # Cylinders.
    for i, (cx, cy) in enumerate(centers):
        ax.add_patch(plt.Circle((cx, cy), cylinder_radius, fill=False, color="k", lw=1.2, zorder=6))
        ax.text(cx, cy, f"C{i}", fontsize=8, ha="center", va="center", zorder=7)

    # ------------------------------------------------------------------
    # Panel 2: abstract tripartite hypergraph
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.set_title("Abstract tripartite hypergraph view")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Module node vertical layout sorted by physical y for readability.
    cyl_order = np.argsort(centers[:, 1])
    module_y = np.linspace(0.1, 0.9, num_cyl) if num_cyl > 1 else np.array([0.5], dtype=np.float32)
    module_pos = {}
    for rank, i in enumerate(cyl_order):
        module_pos[int(i)] = (0.08, float(module_y[rank]))

    # Hyperedge and environment-group nodes use physical centroid y.
    hyper_y = np.clip((centroids[:, 1] - bounds["y_min"]) / max(bounds["ly"], 1e-6), 0.08, 0.92)
    hyper_pos = {k: (0.50, float(hyper_y[k])) for k in range(num_hyper)}
    env_group_pos = {k: (0.90, float(hyper_y[k])) for k in range(num_hyper)}

    # Draw module->hyperedge edges.
    for i in range(num_cyl):
        x0, y0 = module_pos[i]
        for k in range(num_hyper):
            w = float(A_mh[i, k])
            if w < threshold:
                continue
            x1, y1 = hyper_pos[k]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=colors[k],
                alpha=0.15 + 0.75 * w,
                linewidth=0.8 + 3.2 * w,
            )

    # Draw hyperedge->environment-group edges using total token mass.
    token_assign = np.argmax(A_eh, axis=1)
    for k in range(num_hyper):
        mass = float(np.mean(A_eh[:, k]))
        x0, y0 = hyper_pos[k]
        x1, y1 = env_group_pos[k]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=colors[k],
            alpha=0.2 + 0.75 * mass,
            linewidth=1.2 + 10.0 * mass,
        )

    # Draw nodes.
    for i in range(num_cyl):
        x, y = module_pos[i]
        ax.scatter([x], [y], s=120, c="white", edgecolors="k", zorder=3)
        ax.text(x - 0.03, y, f"C{i}", ha="right", va="center", fontsize=9)

    for k in range(num_hyper):
        x, y = hyper_pos[k]
        ax.scatter([x], [y], s=180, marker="*", c=[colors[k]], edgecolors="k", zorder=4)
        ax.text(x, y + 0.04, f"H{k}", ha="center", va="bottom", fontsize=9)

    for k in range(num_hyper):
        x, y = env_group_pos[k]
        n_tokens = int(np.sum(token_assign == k))
        ax.scatter([x], [y], s=130, marker="s", c=[colors[k]], edgecolors="k", zorder=4)
        ax.text(x + 0.03, y, f"E{k}\n(n={n_tokens})", ha="left", va="center", fontsize=8)

    fig.savefig(save_path)
    plt.close(fig)

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

    org_path = output_dir / f"organization_case_{case['case_id']}_tau_{phase_idx:03d}.png"
    render_soft_organization( org_path, out, case, threshold=args.organization_threshold, topk_me_links=args.topk_me_links)

    gif_path = output_dir / f"animation_case_{case['case_id']}_tau_{phase_idx:03d}.gif"
    render_animation(gif_path, model, structure, case, device, args.query_batch_size)

    np.savez_compressed(
        output_dir / f"reconstruction_case_{case['case_id']}_tau_{phase_idx:03d}.npz",
        pred_field=pred_field.astype(np.float32),
        gt_field=gt_field.astype(np.float32),
        x_grid=case["x_grid"].astype(np.float32),
        y_grid=case["y_grid"].astype(np.float32),
        tau=np.asarray([tau_value], dtype=np.float32),
        predicted_frequency=np.asarray([freq_pred], dtype=np.float32),
        gt_frequency=np.asarray([case["dominant_frequency"]], dtype=np.float32),
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
                "quicklook_path": str(fig_path),
                "organization_path": str(org_path),
            },
            f,
            indent=2,
        )

    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
