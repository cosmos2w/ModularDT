from __future__ import annotations

"""Evaluate one Unified Forward Model checkpoint on one ChannelThermal case."""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-unified-forward")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

SANDBOX_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SANDBOX_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diagnostics import compute_hypergraph_diagnostics  # noqa: E402
from train_unified import resolve_device, stats_from_json  # noqa: E402
from unified_model_core import UnifiedHypergraphNeuralField  # noqa: E402
from unified_types import BatchData, UnifiedForwardConfig  # noqa: E402


CHANNEL_ORDER_FALLBACK = ["u", "v", "p", "omega", "temperature"]
CHECKPOINT_NAMES = {
    "best_by_field_mse": "best_by_field_mse_model.pt",
    "field": "best_by_field_mse_model.pt",
    "best_field": "best_by_field_mse_model.pt",
    "best_by_temperature_mse": "best_by_temperature_mse_model.pt",
    "temperature": "best_by_temperature_mse_model.pt",
    "best_temperature": "best_by_temperature_mse_model.pt",
    "best_by_loss": "best_by_loss_model.pt",
    "loss": "best_by_loss_model.pt",
    "best": "best_model.pt",
    "latest": "latest_model.pt",
    "lastest": "latest_model.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="best_by_field_mse",
        help="Checkpoint selector or .pt path. Selectors: best_by_field_mse, best_by_temperature_mse, best_by_loss, best, latest.",
    )
    parser.add_argument("--run-dir", default=None, help="Directory containing checkpoint files.")
    parser.add_argument("--Run_ID", "--run-id", dest="run_id", default=None, help="Run serial such as 0002; searches --runs-root recursively.")
    parser.add_argument("--model-name", default=None, help="Optional substring for selecting a model subdirectory inside an ablation run, e.g. H3_current_like.")
    parser.add_argument("--runs-root", default=str(SANDBOX_ROOT / "results"), help="Root searched when --run-dir is omitted.")
    parser.add_argument("--dataset", default=None, help="Packed ChannelThermal HDF5 path. Defaults to checkpoint config.")
    parser.add_argument("--split", default="test", help="Dataset split: train, val, test, or all.")
    parser.add_argument("--case-id", default=None, help="Exact processed case id, e.g. 0001.")
    parser.add_argument("--case-index", type=int, default=0, help="Index within --split when --case-id is omitted.")
    parser.add_argument("--device", default=None, help="Torch device override, for example cpu, cuda, or cuda:0.")
    parser.add_argument("--output-dir", default=None, help="Evaluation output directory. Defaults beside the checkpoint.")
    parser.add_argument("--query-batch-size", type=int, default=32768, help="Dense grid query chunk size.")
    parser.add_argument(
        "--organization-view",
        choices=["all", "physical", "matrices", "schematic", "none"],
        default="all",
        help="Which organization diagnostics to render.",
    )
    parser.add_argument("--organization-link-threshold", type=float, default=0.25, help="Minimum A_mh link weight drawn.")
    return parser.parse_args()


def resolve_path(path: str | Path, *, base: Path = SANDBOX_ROOT) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    cwd_candidate = (Path.cwd() / value).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / value).resolve()


def checkpoint_file_name(selector: str) -> str:
    cleaned = str(selector).strip().lower()
    if cleaned in CHECKPOINT_NAMES:
        return CHECKPOINT_NAMES[cleaned]
    raise ValueError(f"Unknown checkpoint selector {selector!r}. Use a .pt path or one of {sorted(CHECKPOINT_NAMES)}.")


def normalize_run_id(value: str) -> str:
    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be numeric, e.g. 0002; got {raw!r}.")
    return f"{int(raw):04d}"


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    selector = str(args.checkpoint).strip()
    direct = resolve_path(selector)
    if direct.suffix == ".pt" and direct.exists():
        return direct
    if direct.is_dir():
        return resolve_checkpoint_in_dir(direct, selector="best_by_field_mse")

    if args.run_dir:
        return resolve_checkpoint_in_dir(resolve_path(args.run_dir), selector=selector)

    if args.run_id:
        run_dir = latest_matching_run_dir(resolve_path(args.runs_root), normalize_run_id(args.run_id))
        return resolve_checkpoint_in_dir(run_dir, selector=selector, recursive=True, name_filter=args.model_name)

    if selector.lower() in CHECKPOINT_NAMES:
        return find_latest_checkpoint(resolve_path(args.runs_root), selector=selector)

    maybe_dir = resolve_path(selector)
    if maybe_dir.exists() and maybe_dir.is_dir():
        return resolve_checkpoint_in_dir(maybe_dir, selector="best_by_field_mse")
    raise FileNotFoundError(f"Could not resolve checkpoint: {selector}")


def resolve_checkpoint_in_dir(
    run_dir: Path,
    *,
    selector: str,
    recursive: bool = False,
    name_filter: Optional[str] = None,
) -> Path:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if selector.endswith(".pt"):
        candidate = resolve_path(selector)
        if candidate.exists():
            return candidate
    preferred = run_dir / checkpoint_file_name(selector)
    if preferred.exists():
        return preferred.resolve()
    fallback_order = [
        "best_by_field_mse_model.pt",
        "best_by_temperature_mse_model.pt",
        "best_by_loss_model.pt",
        "best_model.pt",
        "latest_model.pt",
    ]
    for name in fallback_order:
        candidate = run_dir / name
        if candidate.exists():
            print(f"[warning] {preferred.name} not found; using {candidate.name}.")
            return candidate.resolve()
    if recursive:
        names = [checkpoint_file_name(selector), *fallback_order]
        lowered_filter = str(name_filter).lower() if name_filter else None
        for name in names:
            candidates = [path for path in run_dir.rglob(name) if path.is_file()]
            if lowered_filter:
                candidates = [path for path in candidates if lowered_filter in str(path.parent).lower()]
            if candidates:
                candidates.sort(key=lambda path: (path.stat().st_mtime, str(path)))
                chosen = candidates[-1]
                if chosen.name != checkpoint_file_name(selector):
                    print(f"[warning] {checkpoint_file_name(selector)} not found recursively; using {chosen.name}.")
                return chosen.resolve()
    raise FileNotFoundError(f"No checkpoint files found in {run_dir}")


def latest_matching_run_dir(root: Path, run_id: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Runs root not found: {root}")
    pattern = re.compile(rf"(^|/)Run_{re.escape(run_id)}(_|$)")
    matches = [path for path in root.rglob("Run_*") if path.is_dir() and pattern.search(str(path.relative_to(root)))]
    matches += [path for path in root.rglob("*") if path.is_dir() and path.name.startswith(f"Run_{run_id}_")]
    matches = sorted(set(matches), key=lambda path: (path.stat().st_mtime, str(path)))
    if not matches:
        raise FileNotFoundError(f"No run directories under {root} match Run_ID={run_id}.")
    return matches[-1]


def find_latest_checkpoint(root: Path, *, selector: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Runs root not found: {root}")
    name = checkpoint_file_name(selector)
    candidates = [(path.stat().st_mtime, path) for path in root.rglob(name) if path.is_file()]
    if not candidates and name != "best_model.pt":
        candidates = [(path.stat().st_mtime, path) for path in root.rglob("best_model.pt") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No {name} found under {root}.")
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1].resolve()


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_path_name(value: object) -> str:
    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe or "case"


def evaluation_output_dir(base_dir_arg: Optional[str], checkpoint_path: Path, case_id: object) -> Path:
    if base_dir_arg:
        base = resolve_path(base_dir_arg)
    else:
        base = checkpoint_path.parent / "eval_unified"
    return base / f"{safe_path_name(case_id)}_{current_timestamp()}"


def decode_strings(values: Any) -> List[str]:
    arr = np.asarray(values)
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in arr.reshape(-1)]


def read_case_ids(h5: h5py.File, split: str) -> List[str]:
    if "case_ids" in h5 and "splits" in h5:
        case_ids = decode_strings(h5["case_ids"][...])
        splits = [item.lower() for item in decode_strings(h5["splits"][...])]
    else:
        case_ids = sorted(h5["cases"].keys())
        splits = [str(h5["cases"][case_id].attrs.get("split", "all")).lower() for case_id in case_ids]
    split_l = str(split).lower()
    if split_l == "all":
        return case_ids
    return [case_id for case_id, item in zip(case_ids, splits) if item == split_l]


def select_case_id(dataset_path: Path, split: str, case_id: Optional[str], case_index: int) -> str:
    with h5py.File(dataset_path, "r") as h5:
        selected = read_case_ids(h5, split)
        if not selected and split != "all":
            print(f"[warning] No cases found for split={split!r}; falling back to split='all'.")
            selected = read_case_ids(h5, "all")
        if not selected:
            raise RuntimeError(f"No cases found in {dataset_path}.")
        if case_id is not None:
            if str(case_id) not in selected:
                raise KeyError(f"case_id={case_id!r} not found in split={split!r}.")
            return str(case_id)
        return selected[min(max(int(case_index), 0), len(selected) - 1)]


def read_array(group: h5py.Group, names: Iterable[str], *, required: bool = True) -> Optional[np.ndarray]:
    for name in names:
        if name in group:
            return np.asarray(group[name])
    if required:
        raise KeyError(f"None of these datasets exists in case group: {list(names)}")
    return None


def read_dense_channelthermal_sample(dataset_path: Path, split: str, case_id: Optional[str], case_index: int) -> Dict[str, Any]:
    selected_case_id = select_case_id(dataset_path, split, case_id, case_index)
    with h5py.File(dataset_path, "r") as h5:
        group = h5["cases"][selected_case_id]
        channel_order = decode_strings(h5["channel_order"][...]) if "channel_order" in h5 else CHANNEL_ORDER_FALLBACK
        sample = {
            "case_id": str(selected_case_id),
            "split": str(group.attrs.get("split", split)),
            "channel_order": channel_order,
            "x_grid": read_array(group, ["x_grid", "grid_x"]),
            "y_grid": read_array(group, ["y_grid", "grid_y"]),
            "steady_field": read_array(group, ["steady_field", "field", "state"]),
            "module_centers": read_array(group, ["module_centers", "centers"]),
            "module_present": read_array(group, ["module_present", "present"], required=False),
            "heat_powers": read_array(group, ["heat_powers", "heat_power"], required=False),
            "module_mask": read_array(group, ["module_mask"], required=False),
        }
    field = np.asarray(sample["steady_field"], dtype=np.float32)
    if field.ndim == 3 and field.shape[0] <= len(sample["channel_order"]) and field.shape[-1] > len(sample["channel_order"]):
        field = np.moveaxis(field, 0, -1)
    sample["steady_field"] = field.astype(np.float32)
    sample["x_grid"] = np.asarray(sample["x_grid"], dtype=np.float32)
    sample["y_grid"] = np.asarray(sample["y_grid"], dtype=np.float32)
    sample["module_centers"] = np.asarray(sample["module_centers"], dtype=np.float32)
    present = sample["module_present"]
    if present is None:
        present = np.isfinite(sample["module_centers"]).all(axis=-1).astype(np.float32)
    sample["module_present"] = np.asarray(present, dtype=np.float32).reshape(-1)
    heat = sample["heat_powers"]
    if heat is None:
        heat = np.zeros((sample["module_centers"].shape[0],), dtype=np.float32)
    sample["heat_powers"] = np.asarray(heat, dtype=np.float32).reshape(-1)
    if sample["module_mask"] is not None:
        sample["module_mask"] = np.asarray(sample["module_mask"], dtype=bool)
    return sample


def domain_lengths_from_grid(x_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[float, float]:
    dx = float(np.mean(np.diff(x_grid[0]))) if x_grid.ndim == 2 and x_grid.shape[1] > 1 else 0.0
    dy = float(np.mean(np.diff(y_grid[:, 0]))) if y_grid.ndim == 2 and y_grid.shape[0] > 1 else 0.0
    return float(x_grid.max() - x_grid.min() + abs(dx)), float(y_grid.max() - y_grid.min() + abs(dy))


def batch_from_sample(
    sample: Dict[str, Any],
    query_xy: np.ndarray,
    model_cfg: UnifiedForwardConfig,
    target_stats: Dict[str, Any],
    heat_feature_mode: str,
) -> BatchData:
    max_modules = int(model_cfg.max_num_modules)
    centers = np.zeros((max_modules, 2), dtype=np.float32)
    present = np.zeros((max_modules,), dtype=np.float32)
    features = np.zeros((max_modules, 8), dtype=np.float32)

    raw_centers = np.asarray(sample["module_centers"], dtype=np.float32)
    raw_present = np.asarray(sample["module_present"], dtype=np.float32)
    raw_heat = np.asarray(sample["heat_powers"], dtype=np.float32)
    count = min(raw_centers.shape[0], max_modules)
    centers[:count] = raw_centers[:count]
    present[: min(raw_present.shape[0], max_modules)] = raw_present[:max_modules]
    heat = np.zeros((max_modules,), dtype=np.float32)
    heat[: min(raw_heat.shape[0], max_modules)] = raw_heat[:max_modules]

    lx, ly = domain_lengths_from_grid(sample["x_grid"], sample["y_grid"])
    case_heat_scale = max(float(np.max(np.abs(heat))) if heat.size else 0.0, 1.0)
    dataset_heat_scale = max(float(target_stats.get("dataset_heat_scale", 1.0)), 1.0)
    heat_case_relative = heat / case_heat_scale
    heat_dataset_scaled = heat / dataset_heat_scale
    mode = str(heat_feature_mode)
    if mode == "case_relative":
        heat_dataset_scaled = np.zeros_like(heat_dataset_scaled)
    elif mode == "dataset_scaled":
        heat_case_relative = np.zeros_like(heat_case_relative)

    features[:, 0] = centers[:, 0] / max(lx, 1.0e-6)
    features[:, 1] = centers[:, 1] / max(ly, 1.0e-6)
    features[:, 2] = float(model_cfg.module_radius) / max(min(lx, ly), 1.0e-6)
    features[:, 3] = heat_case_relative
    features[:, 4] = heat_dataset_scaled
    features[:, 5] = present

    global_context = np.zeros((8,), dtype=np.float32)
    global_context[0] = 0.0
    global_context[1] = lx
    global_context[2] = ly
    global_context[3] = float(np.sum(present))
    global_context[4] = float(np.sum(np.abs(heat)))

    return BatchData(
        module_centers=torch.from_numpy(centers).unsqueeze(0),
        module_present=torch.from_numpy(present).unsqueeze(0),
        module_features=torch.from_numpy(features).unsqueeze(0),
        global_context=torch.from_numpy(global_context).unsqueeze(0),
        query_xy=torch.from_numpy(query_xy.astype(np.float32)).unsqueeze(0),
        query_time=None,
        target_field=None,
        case_name="channelthermal",
        metadata={"case_id": str(sample["case_id"]), "split": str(sample["split"]), "synthetic": False},
    )


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {str(key)[7:] if str(key).startswith("module.") else str(key): value for key, value in state_dict.items()}


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[UnifiedHypergraphNeuralField, Dict[str, Any], UnifiedForwardConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    payload = checkpoint.get("config_resolved", {})
    model_cfg = UnifiedForwardConfig.from_dict(checkpoint.get("model_config", payload.get("model", {})))
    model = UnifiedHypergraphNeuralField(model_cfg).to(device)
    try:
        model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
    except RuntimeError:
        dummy = make_dummy_batch(model_cfg, device)
        with torch.no_grad():
            model(dummy)
        model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
    model.eval()
    return model, checkpoint, model_cfg


def make_dummy_batch(model_cfg: UnifiedForwardConfig, device: torch.device) -> BatchData:
    q = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
    m = int(model_cfg.max_num_modules)
    return BatchData(
        module_centers=torch.zeros((1, m, 2), dtype=torch.float32, device=device),
        module_present=torch.zeros((1, m), dtype=torch.float32, device=device),
        module_features=torch.zeros((1, m, 8), dtype=torch.float32, device=device),
        global_context=torch.zeros((1, 8), dtype=torch.float32, device=device),
        query_xy=q,
        query_time=None,
        target_field=None,
        case_name="channelthermal",
        metadata={},
    )


def denormalize_field(field: np.ndarray, target_stats: Dict[str, Any]) -> np.ndarray:
    if not bool(target_stats.get("normalize_targets", False)):
        return field
    mean = np.asarray(target_stats["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(target_stats["std"], dtype=np.float32).reshape(1, -1)
    return field * np.maximum(std, 1.0e-6) + mean


def predict_dense_case(
    model: UnifiedHypergraphNeuralField,
    sample: Dict[str, Any],
    model_cfg: UnifiedForwardConfig,
    target_stats: Dict[str, Any],
    heat_feature_mode: str,
    device: torch.device,
    query_batch_size: int,
) -> Dict[str, Any]:
    x_grid = sample["x_grid"]
    y_grid = sample["y_grid"]
    query_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1).astype(np.float32)
    pred_chunks: List[np.ndarray] = []
    first_output: Optional[Dict[str, Any]] = None
    with torch.no_grad():
        for start in range(0, query_xy.shape[0], int(query_batch_size)):
            chunk = query_xy[start : start + int(query_batch_size)]
            batch = batch_from_sample(sample, chunk, model_cfg, target_stats, heat_feature_mode).to(device)
            output = model(batch)
            pred = output["pred_field"].detach().cpu().numpy()[0]
            pred_chunks.append(pred)
            if first_output is None:
                first_output = {key: tensor_to_numpy(value) for key, value in output.items()}
    if first_output is None:
        raise RuntimeError("No prediction chunks were produced.")
    pred_flat = np.concatenate(pred_chunks, axis=0)
    pred_flat = denormalize_field(pred_flat, target_stats)
    pred_grid = pred_flat.reshape(*x_grid.shape, int(model_cfg.field_dim))
    return {"pred_field_grid": pred_grid.astype(np.float32), "organizer_aux": first_output}


def tensor_to_numpy(value: Any) -> Any:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim == 0:
            return float(arr)
        return arr
    return value


def module_and_fluid_masks(sample: Dict[str, Any], module_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    if sample.get("module_mask") is not None:
        module_mask = np.asarray(sample["module_mask"], dtype=bool)
    else:
        x_grid = np.asarray(sample["x_grid"], dtype=np.float32)
        y_grid = np.asarray(sample["y_grid"], dtype=np.float32)
        module_mask = np.zeros(x_grid.shape, dtype=bool)
        centers = np.asarray(sample["module_centers"], dtype=np.float32)
        present = np.asarray(sample["module_present"], dtype=np.float32) > 0.5
        for idx in np.flatnonzero(present):
            cx, cy = centers[idx]
            module_mask |= np.hypot(x_grid - float(cx), y_grid - float(cy)) <= float(module_radius)
    return module_mask, ~module_mask


def channel_cmap(name: str) -> str:
    return {"u": "coolwarm", "v": "coolwarm", "p": "magma", "omega": "RdBu_r", "temperature": "inferno"}.get(name, "viridis")


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    diff = pred - gt
    flat_diff = diff.reshape(-1)
    flat_gt = gt.reshape(-1)
    finite = np.isfinite(flat_diff) & np.isfinite(flat_gt)
    flat_diff = flat_diff[finite]
    flat_gt = flat_gt[finite]
    if flat_diff.size == 0:
        return {"l2_norm": float("nan"), "mse": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "mae": float("nan"), "relative_l2": float("nan"), "normalizer": float("nan"), "num_values": 0.0}
    l2_norm = float(np.linalg.norm(flat_diff, ord=2))
    mse = float(np.mean(flat_diff * flat_diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(flat_diff)))
    gt_norm = float(np.linalg.norm(flat_gt, ord=2))
    span = float(np.max(flat_gt) - np.min(flat_gt))
    rms_scale = float(np.sqrt(np.mean(flat_gt * flat_gt)))
    normalizer = span if span > 1.0e-12 else max(rms_scale, 1.0e-12)
    return {
        "l2_norm": l2_norm,
        "mse": mse,
        "rmse": rmse,
        "nrmse": float(rmse / max(normalizer, 1.0e-12)),
        "mae": mae,
        "relative_l2": float(l2_norm / max(gt_norm, 1.0e-12)),
        "normalizer": float(normalizer),
        "num_values": float(flat_diff.size),
    }


def masked_error_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(prediction)
    gt = np.asarray(target)
    valid = np.asarray(mask, dtype=bool)
    if pred.ndim == valid.ndim + 1:
        return error_metrics(pred[valid, :], gt[valid, :])
    return error_metrics(pred[valid], gt[valid])


def draw_module_outlines(ax: Any, sample: Dict[str, Any], module_radius: float, color: str = "#e6e6e6") -> None:
    centers = np.asarray(sample["module_centers"], dtype=np.float32)
    present = np.asarray(sample["module_present"], dtype=np.float32) > 0.5
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        ax.add_patch(Circle((float(cx), float(cy)), float(module_radius), fill=False, color=color, lw=0.9))


def plot_field_quicklook(
    output_path: Path,
    sample: Dict[str, Any],
    pred_field: np.ndarray,
    channel_order: Sequence[str],
    module_radius: float,
) -> None:
    gt = np.asarray(sample["steady_field"], dtype=np.float32)[..., : pred_field.shape[-1]]
    preferred = [name for name in ["u", "v", "p", "omega", "temperature"] if name in channel_order[: pred_field.shape[-1]]]
    if not preferred:
        preferred = list(channel_order[: min(3, pred_field.shape[-1])])
    _, fluid_mask = module_and_fluid_masks(sample, module_radius)
    x_min, x_max = float(np.min(sample["x_grid"])), float(np.max(sample["x_grid"]))
    y_min, y_max = float(np.min(sample["y_grid"])), float(np.max(sample["y_grid"]))
    extent = (x_min, x_max, y_min, y_max)
    domain_aspect = max((x_max - x_min) / max(y_max - y_min, 1.0e-12), 1.0e-6)
    panel_width = 3.5
    panel_height = max(1.9, panel_width / domain_aspect)
    fig, axes = plt.subplots(len(preferred), 3, figsize=(3.0 * panel_width, panel_height * len(preferred)), constrained_layout=True)
    if len(preferred) == 1:
        axes = axes[None, :]
    for row, name in enumerate(preferred):
        idx = list(channel_order).index(name)
        gt_img = np.where(fluid_mask, gt[..., idx], np.nan)
        pred_img = np.where(fluid_mask, pred_field[..., idx], np.nan)
        err_img = np.abs(pred_img - gt_img)
        metrics = masked_error_metrics(pred_field[..., idx], gt[..., idx], fluid_mask)
        vmin = float(np.nanmin(gt_img))
        vmax = float(np.nanmax(gt_img))
        for col, (image, title, cmap_name) in enumerate(
            [
                (gt_img, f"GT {name}", channel_cmap(name)),
                (pred_img, f"Pred {name}", channel_cmap(name)),
                (err_img, f"Abs error {name}\nrelL2={metrics['relative_l2']:.4e}", "magma"),
            ]
        ):
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#303030")
            im = axes[row, col].imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin if col < 2 else None,
                vmax=vmax if col < 2 else None,
                aspect="equal",
            )
            draw_module_outlines(axes[row, col], sample, module_radius)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def extract_organization_arrays(sample: Dict[str, Any], aux: Dict[str, Any], model_cfg: UnifiedForwardConfig) -> Dict[str, np.ndarray]:
    centers = pad_array(np.asarray(sample["module_centers"], dtype=np.float32), int(model_cfg.max_num_modules), 2)
    present = pad_vector(np.asarray(sample["module_present"], dtype=np.float32), int(model_cfg.max_num_modules)) > 0.5
    heat = pad_vector(np.asarray(sample["heat_powers"], dtype=np.float32), int(model_cfg.max_num_modules))
    env_coords = np.asarray(aux.get("env_coords", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    A_eh = np.asarray(aux.get("A_eh", np.zeros((env_coords.shape[0], int(model_cfg.num_hyperedges)), dtype=np.float32)), dtype=np.float32)
    A_mh = np.asarray(aux.get("A_mh", np.zeros((centers.shape[0], int(model_cfg.num_hyperedges)), dtype=np.float32)), dtype=np.float32)
    strength = np.asarray(aux.get("hyper_strength", np.ones((A_eh.shape[-1],), dtype=np.float32)), dtype=np.float32)
    module_mass = np.asarray(aux.get("hyper_module_mass", np.zeros_like(strength)), dtype=np.float32)
    env_mass = np.asarray(aux.get("hyper_env_mass", np.zeros_like(strength)), dtype=np.float32)
    source = np.asarray(aux.get("hyper_source_coords", np.zeros((strength.shape[0], 2), dtype=np.float32)), dtype=np.float32)
    region = np.asarray(aux.get("hyper_region_coords", np.zeros((strength.shape[0], 2), dtype=np.float32)), dtype=np.float32)
    return {
        "centers": centers,
        "present": present,
        "heat": heat,
        "env_coords": env_coords,
        "A_eh": A_eh,
        "A_mh": A_mh,
        "strength": strength,
        "module_mass": module_mass,
        "env_mass": env_mass,
        "source": source,
        "region": region,
    }


def pad_array(values: np.ndarray, length: int, width: int) -> np.ndarray:
    out = np.zeros((length, width), dtype=np.float32)
    count = min(values.shape[0], length)
    out[:count, : min(values.shape[1], width)] = values[:count, :width]
    return out


def pad_vector(values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros((length,), dtype=np.float32)
    out[: min(values.shape[0], length)] = values[:length]
    return out


def compute_organization_diagnostics(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    A_eh = np.asarray(arrays["A_eh"], dtype=np.float64)
    A_mh = np.asarray(arrays["A_mh"], dtype=np.float64)
    present = np.asarray(arrays["present"], dtype=bool)
    strength = np.asarray(arrays["strength"], dtype=np.float64)
    if A_eh.size == 0:
        return {
            "dominant_env_fraction_max": 0.0,
            "dominant_env_effective_edges": 0.0,
            "soft_env_effective_edges": 0.0,
            "A_eh_spatial_variation": 0.0,
            "hyperedge_column_cosine_mean": 0.0,
            "active_edge_count": 0.0,
            "collapse_warning": False,
        }
    eps = 1.0e-12
    num_h = A_eh.shape[-1]
    dominant = A_eh.argmax(axis=-1)
    counts = np.bincount(dominant, minlength=num_h).astype(np.float64)
    count_frac = counts / max(float(counts.sum()), eps)
    env_mass = A_eh.mean(axis=0)
    env_mass = env_mass / max(float(env_mass.sum()), eps)
    count_entropy = -float(np.sum(count_frac * np.log(np.maximum(count_frac, eps))))
    soft_entropy = -float(np.sum(env_mass * np.log(np.maximum(env_mass, eps))))
    module_part = A_mh * present[:, None].astype(np.float64) if A_mh.size else np.zeros((0, num_h))
    combined = np.concatenate([module_part, A_eh], axis=0)
    cols = combined / np.maximum(np.linalg.norm(combined, axis=0, keepdims=True), eps)
    cosine = np.clip(cols.T @ cols, -1.0, 1.0)
    offdiag = cosine[~np.eye(num_h, dtype=bool)] if num_h > 1 else np.asarray([], dtype=np.float64)
    dominant_max = float(np.max(count_frac)) if count_frac.size else 0.0
    return {
        "dominant_env_fraction_max": dominant_max,
        "dominant_env_effective_edges": float(np.exp(count_entropy)),
        "soft_env_effective_edges": float(np.exp(soft_entropy)),
        "A_eh_spatial_variation": float(np.std(A_eh, axis=0).mean()),
        "hyperedge_column_cosine_mean": float(np.mean(offdiag)) if offdiag.size else 0.0,
        "active_edge_count": float(np.sum(strength >= 0.05)),
        "collapse_warning": bool(dominant_max > 0.90),
    }


def organization_rows(arrays: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    present = arrays["present"]
    strength = arrays["strength"]
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((0,), dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    for hidx in range(strength.shape[0]):
        module_scores = A_mh[:, hidx] if A_mh.size else np.zeros((present.shape[0],), dtype=np.float32)
        valid_scores = [(idx, float(module_scores[idx])) for idx in np.flatnonzero(present)]
        valid_scores.sort(key=lambda item: item[1], reverse=True)
        top = [(idx, score) for idx, score in valid_scores[:4] if score > 1.0e-6]
        rows.append(
            {
                "hyperedge_id": int(hidx),
                "module_mass": float(arrays["module_mass"][hidx]),
                "env_mass": float(arrays["env_mass"][hidx]),
                "hyper_strength": float(strength[hidx]),
                "top_modules": ";".join(f"M{idx}" for idx, _ in top),
                "top_module_weights": ";".join(f"{score:.4f}" for _, score in top),
                "dominant_env_count": int(np.sum(dominant_env == hidx)),
                "source_x": float(arrays["source"][hidx, 0]),
                "source_y": float(arrays["source"][hidx, 1]),
                "region_x": float(arrays["region"][hidx, 0]),
                "region_y": float(arrays["region"][hidx, 1]),
                "active": bool(strength[hidx] >= 0.05),
                "low_strength": bool(strength[hidx] < 0.05),
            }
        )
    return rows


def save_organization_summary(output_dir: Path, arrays: Dict[str, np.ndarray]) -> Tuple[Path, Path, Dict[str, Any]]:
    rows = organization_rows(arrays)
    diagnostics = compute_organization_diagnostics(arrays)
    csv_path = output_dir / "organization_summary.csv"
    json_path = output_dir / "organization_summary.json"
    fieldnames = [
        "hyperedge_id",
        "module_mass",
        "env_mass",
        "hyper_strength",
        "top_modules",
        "top_module_weights",
        "dominant_env_count",
        "source_x",
        "source_y",
        "region_x",
        "region_y",
        "active",
        "low_strength",
        "dominant_env_fraction_max",
        "dominant_env_effective_edges",
        "soft_env_effective_edges",
        "A_eh_spatial_variation",
        "hyperedge_column_cosine_mean",
        "active_edge_count",
        "collapse_warning",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{**row, **diagnostics} for row in rows])
    write_json(
        json_path,
        {
            "hyperedges": rows,
            "diagnostics": diagnostics,
            "visual_encoding": {
                "env_token_color": "dominant hyperedge argmax(A_eh)",
                "env_token_alpha": "max assignment confidence max(A_eh)",
                "module_link_width": "A_mh",
                "hyperedge_strength": "sqrt(module_mass * env_mass)",
            },
        },
    )
    if diagnostics.get("collapse_warning"):
        print(f"[warning] Organizer may be collapsed: dominant_env_fraction_max={diagnostics['dominant_env_fraction_max']:.3f}.")
    return csv_path, json_path, diagnostics


def plot_organization_overview(
    output_path: Path,
    sample: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    module_radius: float,
    channel_order: Sequence[str],
    link_threshold: float,
) -> None:
    centers = arrays["centers"]
    present = arrays["present"]
    heat = arrays["heat"]
    env_coords = arrays["env_coords"]
    A_eh = arrays["A_eh"]
    A_mh = arrays["A_mh"]
    strength = arrays["strength"]
    source = arrays["source"]
    region = arrays["region"]
    temp_idx = list(channel_order).index("temperature") if "temperature" in channel_order else min(sample["steady_field"].shape[-1] - 1, 0)
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    ax.imshow(
        sample["steady_field"][..., temp_idx],
        origin="lower",
        extent=(float(np.min(sample["x_grid"])), float(np.max(sample["x_grid"])), float(np.min(sample["y_grid"])), float(np.max(sample["y_grid"]))),
        cmap="inferno",
        alpha=0.42,
        aspect="auto",
    )
    num_h = max(int(strength.shape[0]), 1)
    cmap = plt.get_cmap("tab10", num_h)
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((env_coords.shape[0],), dtype=np.int64)
    confidence = A_eh.max(axis=-1) if A_eh.size else np.ones((env_coords.shape[0],), dtype=np.float32)
    if env_coords.size:
        ax.scatter(
            env_coords[:, 0],
            env_coords[:, 1],
            c=dominant_env,
            cmap=cmap,
            s=18.0 + 42.0 * confidence,
            edgecolor="white",
            linewidth=0.25,
            alpha=np.clip(0.25 + 0.75 * confidence, 0.25, 1.0),
        )
    heat_abs = np.abs(heat)
    heat_scale = heat_abs / max(float(np.nanmax(heat_abs)) if heat_abs.size else 0.0, 1.0e-6)
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        color = "#fdae61" if heat[module_idx] >= 0 else "#74add1"
        ax.add_patch(Circle((float(cx), float(cy)), float(module_radius), fill=True, color=color, alpha=0.20 + 0.35 * float(heat_scale[module_idx]), lw=0.0))
        ax.add_patch(Circle((float(cx), float(cy)), float(module_radius), fill=False, color=color, lw=1.2 + 1.4 * float(heat_scale[module_idx])))
        ax.text(float(cx), float(cy), f"M{module_idx}", ha="center", va="center", color="white", fontsize=8, weight="bold")
    for hidx in range(num_h):
        alpha = float(np.clip(strength[hidx], 0.12, 1.0))
        color = cmap(hidx)
        ax.plot([source[hidx, 0], region[hidx, 0]], [source[hidx, 1], region[hidx, 1]], color=color, lw=1.0 + 2.0 * alpha, alpha=alpha)
        ax.scatter(source[hidx, 0], source[hidx, 1], marker="x", s=35 + 70 * alpha, color=color, linewidth=1.5)
        ax.scatter(region[hidx, 0], region[hidx, 1], marker="*", s=65 + 125 * alpha, color=color, edgecolor="black", linewidth=0.45)
        ax.text(region[hidx, 0], region[hidx, 1], f"H{hidx}\n{strength[hidx]:.2f}", color="white", fontsize=7, ha="center", va="center")
        for module_idx in np.flatnonzero(present):
            weight = float(A_mh[module_idx, hidx]) if A_mh.size else 0.0
            if weight >= float(link_threshold):
                ax.plot([centers[module_idx, 0], source[hidx, 0]], [centers[module_idx, 1], source[hidx, 1]], color=color, lw=0.5 + 1.8 * weight, alpha=0.18 + 0.55 * weight)
    diag = compute_organization_diagnostics(arrays)
    ax.set_title(f"Hypergraph organization (dom={diag['dominant_env_fraction_max']:.2f}, softEff={diag['soft_env_effective_edges']:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_organization_matrices(output_path: Path, arrays: Dict[str, np.ndarray]) -> None:
    A_mh = arrays["A_mh"]
    A_eh = arrays["A_eh"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    dominant_env = A_eh.argmax(axis=-1) if A_eh.size else np.zeros((0,), dtype=np.int64)
    sort_idx = np.lexsort((np.arange(A_eh.shape[0]), dominant_env)) if A_eh.size else np.arange(0)
    fig = plt.figure(figsize=(12.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.08, 1.45])
    ax_mh = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[0, 1])
    ax_eh = fig.add_subplot(gs[0, 2])
    labels = [f"H{i}\nS={strength[i]:.2f}\nM={module_mass[i]:.2f} E={env_mass[i]:.2f}" for i in range(strength.shape[0])]
    im1 = ax_mh.imshow(A_mh.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(A_mh)) if A_mh.size else 1.0, 1.0e-6))
    ax_mh.set_title("A_mh modules x hyperedges")
    ax_mh.set_xlabel("module")
    ax_mh.set_ylabel("hyperedge")
    ax_mh.set_xticks(np.arange(A_mh.shape[0]))
    ax_mh.set_xticklabels([f"M{i}" for i in range(A_mh.shape[0])], rotation=45, ha="right")
    ax_mh.set_yticks(np.arange(strength.shape[0]))
    ax_mh.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im1, ax=ax_mh, fraction=0.046, pad=0.04)
    strip = dominant_env[sort_idx][:, None] if A_eh.size else np.zeros((0, 1), dtype=np.int64)
    ax_strip.imshow(strip, aspect="auto", cmap=plt.get_cmap("tab10", max(strength.shape[0], 1)))
    ax_strip.set_title("dom", fontsize=8)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    im2 = ax_eh.imshow(A_eh[sort_idx].T if A_eh.size else A_eh.T, aspect="auto", cmap="viridis", vmin=0.0)
    ax_eh.set_title("A_eh env tokens sorted by dominant hyperedge")
    ax_eh.set_xlabel("sorted env token")
    ax_eh.set_ylabel("hyperedge")
    ax_eh.set_yticks(np.arange(strength.shape[0]))
    ax_eh.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im2, ax=ax_eh, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_organization_schematic(output_path: Path, arrays: Dict[str, np.ndarray], link_threshold: float) -> None:
    centers = arrays["centers"]
    present = arrays["present"]
    A_mh = arrays["A_mh"]
    strength = arrays["strength"]
    module_mass = arrays["module_mass"]
    env_mass = arrays["env_mass"]
    source = arrays["source"]
    region = arrays["region"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    num_h = max(strength.shape[0], 1)
    cmap = plt.get_cmap("tab10", num_h)
    for hidx in range(strength.shape[0]):
        color = cmap(hidx) if strength[hidx] >= 0.05 else (0.55, 0.55, 0.55, 1.0)
        alpha = float(np.clip(strength[hidx], 0.12, 0.85))
        cx = 0.5 * (source[hidx, 0] + region[hidx, 0])
        cy = 0.5 * (source[hidx, 1] + region[hidx, 1])
        ax.add_patch(Circle((float(cx), float(cy)), 0.34 + 0.8 * float(env_mass[hidx]), color=color, alpha=0.12 + 0.18 * alpha, lw=0.0))
        ax.text(float(cx), float(cy), f"H{hidx}\nM={module_mass[hidx]:.2f} E={env_mass[hidx]:.2f}\nS={strength[hidx]:.2f}", ha="center", va="center", fontsize=8, color="black")
    for module_idx in np.flatnonzero(present):
        cx, cy = centers[module_idx]
        ax.scatter(cx, cy, s=150, color="#fdae61", edgecolor="black", linewidth=0.7, zorder=3)
        ax.text(cx, cy, f"M{module_idx}", ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=4)
        for hidx in range(strength.shape[0]):
            weight = float(A_mh[module_idx, hidx]) if A_mh.size else 0.0
            if weight < float(link_threshold):
                continue
            color = cmap(hidx) if strength[hidx] >= 0.05 else (0.55, 0.55, 0.55, 1.0)
            ax.plot([cx, source[hidx, 0]], [cy, source[hidx, 1]], color=color, lw=0.4 + 2.2 * weight, alpha=0.20 + 0.55 * weight)
    ax.scatter(source[:, 0], source[:, 1], marker="x", s=55, color="black", linewidth=1.3, label="source")
    ax.scatter(region[:, 0], region[:, 1], marker="*", s=95, color="black", linewidth=0.7, label="region")
    ax.set_title("Conceptual organization schematic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_prediction_outputs(
    output_dir: Path,
    sample: Dict[str, Any],
    pred_field: np.ndarray,
    arrays: Dict[str, np.ndarray],
    model_cfg: UnifiedForwardConfig,
) -> Path:
    module_mask, fluid_mask = module_and_fluid_masks(sample, float(model_cfg.module_radius))
    npz_path = output_dir / "evaluation_outputs.npz"
    np.savez_compressed(
        npz_path,
        pred_field_grid=pred_field.astype(np.float32),
        gt_field_grid=np.asarray(sample["steady_field"], dtype=np.float32)[..., : pred_field.shape[-1]],
        module_mask=module_mask.astype(np.uint8),
        fluid_mask=fluid_mask.astype(np.uint8),
        A_mh=arrays["A_mh"].astype(np.float32),
        A_eh=arrays["A_eh"].astype(np.float32),
        hyper_strength=arrays["strength"].astype(np.float32),
        hyper_module_mass=arrays["module_mass"].astype(np.float32),
        hyper_env_mass=arrays["env_mass"].astype(np.float32),
        hyper_source_coords=arrays["source"].astype(np.float32),
        hyper_region_coords=arrays["region"].astype(np.float32),
        env_coords=arrays["env_coords"].astype(np.float32),
        module_centers=arrays["centers"].astype(np.float32),
        module_present=arrays["present"].astype(np.uint8),
    )
    return npz_path


def summarize_reconstruction(
    checkpoint_path: Path,
    sample: Dict[str, Any],
    pred_field: np.ndarray,
    model_cfg: UnifiedForwardConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    gt = np.asarray(sample["steady_field"], dtype=np.float32)[..., : pred_field.shape[-1]]
    _, fluid_mask = module_and_fluid_masks(sample, float(model_cfg.module_radius))
    channel_order = list(sample["channel_order"])[: pred_field.shape[-1]]
    field_metrics = error_metrics(pred_field, gt)
    field_metrics_fluid = masked_error_metrics(pred_field, gt, fluid_mask)
    channel_metrics = {}
    channel_metrics_fluid = {}
    for idx, name in enumerate(channel_order):
        channel_metrics[str(name)] = error_metrics(pred_field[..., idx], gt[..., idx])
        channel_metrics_fluid[str(name)] = masked_error_metrics(pred_field[..., idx], gt[..., idx], fluid_mask)
    temperature_metrics = channel_metrics.get("temperature")
    temperature_metrics_fluid = channel_metrics_fluid.get("temperature")
    return {
        "checkpoint": str(checkpoint_path),
        "case_id": str(sample["case_id"]),
        "split": str(sample["split"]),
        "metric_note": "l2_norm is aggregate Euclidean norm; relative_l2 divides by target L2; nrmse divides RMSE by target range or RMS when range is near zero.",
        "metric_mask_note": "Fluid metrics exclude solid module interiors using module_mask when available.",
        "field_l2_error": field_metrics["l2_norm"],
        "field_rmse": field_metrics["rmse"],
        "field_nrmse": field_metrics["nrmse"],
        "field_relative_l2": field_metrics["relative_l2"],
        "field_mse_fluid": field_metrics_fluid["mse"],
        "field_rmse_fluid": field_metrics_fluid["rmse"],
        "field_nrmse_fluid": field_metrics_fluid["nrmse"],
        "field_relative_l2_fluid": field_metrics_fluid["relative_l2"],
        "temperature_rmse": temperature_metrics["rmse"] if temperature_metrics else None,
        "temperature_nrmse": temperature_metrics["nrmse"] if temperature_metrics else None,
        "temperature_relative_l2": temperature_metrics["relative_l2"] if temperature_metrics else None,
        "temperature_rmse_fluid": temperature_metrics_fluid["rmse"] if temperature_metrics_fluid else None,
        "temperature_nrmse_fluid": temperature_metrics_fluid["nrmse"] if temperature_metrics_fluid else None,
        "temperature_relative_l2_fluid": temperature_metrics_fluid["relative_l2"] if temperature_metrics_fluid else None,
        "field_metrics": field_metrics,
        "field_metrics_fluid": field_metrics_fluid,
        "field_channel_metrics": channel_metrics,
        "field_channel_metrics_fluid": channel_metrics_fluid,
        "channel_order": channel_order,
        "outputs": {
            "global_field_quicklook": str(output_dir / "global_field_quicklook.png"),
            "npz": str(output_dir / "evaluation_outputs.npz"),
        },
    }


def write_metrics_csv(path: Path, summary: Dict[str, Any], organization_diagnostics: Dict[str, Any]) -> None:
    row = flatten_scalars({**summary, "organization": organization_diagnostics})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def flatten_scalars(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_scalars(value, prefix=f"{name}."))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
    return out


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def copy_figure_alias(source: Path, alias: Path) -> None:
    if source.resolve() != alias.resolve():
        shutil.copyfile(source, alias)


def infer_dataset_path(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> Path:
    if args.dataset:
        return resolve_path(args.dataset)
    payload = checkpoint.get("config_resolved", {})
    data_cfg = payload.get("data", {})
    dataset_path = data_cfg.get("channelthermal_dataset_path", "../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")
    return resolve_path(dataset_path, base=SANDBOX_ROOT)


def target_stats_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    raw = checkpoint.get("target_stats")
    if raw is None:
        raw = checkpoint.get("config_resolved", {}).get("training", {}).get("target_stats")
    if not raw:
        raise KeyError("Checkpoint is missing target_stats.")
    stats = stats_from_json(raw)
    stats["mean"] = stats["mean"].detach().cpu().numpy().astype(np.float32)
    stats["std"] = stats["std"].detach().cpu().numpy().astype(np.float32)
    stats["dataset_heat_scale"] = float(raw.get("dataset_heat_scale", 1.0))
    stats["module_heat_feature_mode"] = str(raw.get("module_heat_feature_mode", checkpoint.get("config_resolved", {}).get("training", {}).get("module_heat_feature_mode", "both")))
    return stats


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_arg(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = resolve_device(args.device or "auto")
    model, checkpoint, model_cfg = load_model(checkpoint_path, device)
    target_stats = target_stats_from_checkpoint(checkpoint)
    dataset_path = infer_dataset_path(args, checkpoint)
    sample = read_dense_channelthermal_sample(dataset_path, args.split, args.case_id, args.case_index)
    output_dir = evaluation_output_dir(args.output_dir, checkpoint_path, sample["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    heat_feature_mode = str(
        checkpoint.get("config_resolved", {})
        .get("training", {})
        .get("module_heat_feature_mode", target_stats.get("module_heat_feature_mode", "both"))
    )
    prediction = predict_dense_case(
        model=model,
        sample=sample,
        model_cfg=model_cfg,
        target_stats=target_stats,
        heat_feature_mode=heat_feature_mode,
        device=device,
        query_batch_size=int(args.query_batch_size),
    )
    pred_field = prediction["pred_field_grid"]
    arrays = extract_organization_arrays(sample, prediction["organizer_aux"], model_cfg)
    org_csv, org_json, org_diagnostics = save_organization_summary(output_dir, arrays)
    plot_field_quicklook(output_dir / "global_field_quicklook.png", sample, pred_field, sample["channel_order"], float(model_cfg.module_radius))
    org_outputs: Dict[str, str] = {}
    if args.organization_view != "none":
        if args.organization_view in {"all", "physical"}:
            overview = output_dir / "organization_overview.png"
            plot_organization_overview(overview, sample, arrays, float(model_cfg.module_radius), sample["channel_order"], float(args.organization_link_threshold))
            legacy = output_dir / "organizer_visualization.png"
            copy_figure_alias(overview, legacy)
            org_outputs["organization_overview"] = str(overview)
            org_outputs["organizer_visualization"] = str(legacy)
        if args.organization_view in {"all", "matrices"}:
            matrices = output_dir / "organization_matrices.png"
            plot_organization_matrices(matrices, arrays)
            org_outputs["organization_matrices"] = str(matrices)
        if args.organization_view in {"all", "schematic"}:
            schematic = output_dir / "organization_schematic.png"
            plot_organization_schematic(schematic, arrays, float(args.organization_link_threshold))
            org_outputs["organization_schematic"] = str(schematic)
    npz_path = save_prediction_outputs(output_dir, sample, pred_field, arrays, model_cfg)
    summary = summarize_reconstruction(checkpoint_path, sample, pred_field, model_cfg, output_dir)
    summary["dataset"] = str(dataset_path)
    summary["device"] = str(device)
    summary["model_config"] = model_cfg.to_dict()
    summary["target_stats"] = {
        "normalize_targets": bool(target_stats.get("normalize_targets", False)),
        "mean": np.asarray(target_stats["mean"]).tolist(),
        "std": np.asarray(target_stats["std"]).tolist(),
        "dataset_heat_scale": float(target_stats.get("dataset_heat_scale", 1.0)),
        "module_heat_feature_mode": heat_feature_mode,
    }
    summary["outputs"].update(
        {
            "organization_summary_csv": str(org_csv),
            "organization_summary_json": str(org_json),
            "eval_metrics_csv": str(output_dir / "eval_metrics.csv"),
            "npz": str(npz_path),
            **org_outputs,
        }
    )
    org_tensor_diag = compute_hypergraph_diagnostics({key: torch.as_tensor(value).unsqueeze(0) if isinstance(value, np.ndarray) else value for key, value in prediction["organizer_aux"].items()})
    summary["organization_diagnostics"] = {**org_tensor_diag, **org_diagnostics}
    summary["headline_metrics"] = {
        "field_nrmse_fluid": summary["field_nrmse_fluid"],
        "field_relative_l2_fluid": summary["field_relative_l2_fluid"],
        "temperature_nrmse_fluid": summary["temperature_nrmse_fluid"],
        "temperature_relative_l2_fluid": summary["temperature_relative_l2_fluid"],
        "active_edge_count": summary["organization_diagnostics"].get("active_edge_count"),
        "dominant_env_fraction_max": summary["organization_diagnostics"].get("dominant_env_fraction_max"),
    }
    write_metrics_csv(output_dir / "eval_metrics.csv", summary, summary["organization_diagnostics"])
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
