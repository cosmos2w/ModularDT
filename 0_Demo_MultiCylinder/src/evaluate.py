from __future__ import annotations

"""Evaluate a trained checkpoint on a packed inert multi-cylinder HDF5 dataset.

By default this script finds the most recent training run whose directory starts
with "Case{case_id}_" under Saved_Model/ and loads best_model.pt. It then
reconstructs a chosen dataset case at a requested phase and generates a quick-
look comparison plot for [u, v, p, omega].
"""

import argparse
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

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (run_dir / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / f"quicklook_case_{case['case_id']}_tau_{phase_idx:03d}.png"
    render_quicklook(fig_path, pred_field, gt_field, case["x_grid"], case["y_grid"], tau_value, case)
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
                "dataset_case_id": case["case_id"],
                "split": case["split"],
                "tau": tau_value,
                "phase_idx": phase_idx,
                "field_mse": mse,
                "predicted_frequency": freq_pred,
                "gt_frequency": case["dominant_frequency"],
                "quicklook_path": str(fig_path),
            },
            f,
            indent=2,
        )

    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
