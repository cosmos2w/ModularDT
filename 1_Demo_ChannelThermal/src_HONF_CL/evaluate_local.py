"""Evaluate and visualize a Stage-A local surrogate checkpoint.

Inputs are a copied or legacy local surrogate checkpoint plus a packed local
HDF5 dataset. Outputs are internal-temperature and interface quicklook plots
plus JSON metrics. This executable is ChannelThermal-specific and uses the
copied local architecture with strict state-dict loading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from data.datasets import H5Normalizer, LocalModuleDataset
from common.runtime import (
    current_timestamp,
    load_trusted_checkpoint,
    recursive_to_device,
    resolve_demo_path,
    select_device,
    strip_module_prefix,
    write_json,
)
from local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
from evaluation_tools.plots import error_metrics, plot_local_interface, plot_local_internal


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line options for this workflow."""

    parser = argparse.ArgumentParser(description="Evaluate a local module surrogate checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint selector: best, latest/lastest, or a direct .pt path.",
    )
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None, help="Numeric run serial used to find the latest matching saved model, e.g. 0001.")
    parser.add_argument("--saved-root", type=str, default="./Saved_Model_HONF_CL/Local", help="Root directory containing local saved-model runs.")
    parser.add_argument("--dataset", type=str, default=None, help="Override packed local HDF5 path.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument("--case-index", type=int, default=0, help="Index within the selected split.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for quicklook outputs.")
    return parser.parse_args()


def checkpoint_file_name(selector: str) -> str:
    """Perform the checkpoint file name operation used by this module."""

    cleaned = str(selector).strip().lower()
    if cleaned == "best":
        return "best_model.pt"
    if cleaned in {"latest", "lastest"}:
        return "latest_model.pt"
    raise ValueError("--checkpoint must be 'best', 'latest'/'lastest', or a direct checkpoint path.")


def normalize_run_id(value: str) -> str:
    """Normalize run id."""

    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return f"{int(raw):04d}"


def latest_run_dir(saved_root: Path, run_id: str) -> Path:
    """Perform the latest run dir operation used by this module."""

    normalized = normalize_run_id(run_id)
    patterns = (f"Run_{normalized}_*", f"{normalized}_*", f"{normalized}*")
    matches = sorted({path for pattern in patterns for path in saved_root.glob(pattern) if path.is_dir()})
    if not matches:
        raise FileNotFoundError(f"No saved local runs found under {saved_root} with Run_ID={normalized!r}.")
    return matches[-1]


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    """Resolve checkpoint arg."""

    selector = str(args.checkpoint)
    if args.run_id:
        saved_root = resolve_demo_path(args.saved_root)
        run_dir = latest_run_dir(saved_root, args.run_id)
        return (run_dir / checkpoint_file_name(selector)).resolve()
    candidate = resolve_demo_path(selector)
    if candidate.suffix == ".pt" or candidate.exists():
        return candidate
    if selector.strip().lower() in {"best", "latest", "lastest"}:
        raise ValueError("--Run_ID is required when --checkpoint is 'best' or 'latest'.")
    saved_root = resolve_demo_path(args.saved_root)
    return (latest_run_dir(saved_root, selector) / "best_model.pt").resolve()


def tensorize_sample(sample: Dict, device: torch.device) -> Dict:
    """Perform the tensorize sample operation used by this module."""

    out = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(value).unsqueeze(0)
        else:
            out[key] = value
    return recursive_to_device(out, device)


def safe_path_name(value: object) -> str:
    """Perform the safe path name operation used by this module."""

    raw = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return safe or "case"


def evaluation_output_dir(base_dir_arg: str | None, checkpoint_path: Path, case_id: object) -> Path:
    """Perform the evaluation output dir operation used by this module."""

    base_dir = Path(base_dir_arg) if base_dir_arg else checkpoint_path.parent / "eval_local"
    return resolve_demo_path(base_dir) / f"{safe_path_name(case_id)}_{current_timestamp()}"


def main() -> int:
    """Run this command-line workflow and return its process status."""

    args = parse_args()
    checkpoint_path = resolve_checkpoint_arg(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    train_cfg = checkpoint.get("train_config", {})
    dataset_cfg = train_cfg.get("dataset", {})
    dataset_path = args.dataset or dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5")
    normalize_inputs = bool(dataset_cfg.get("normalize_inputs", False))
    normalize_targets = bool(dataset_cfg.get("normalize_targets", False))
    checkpoint_stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint.get("local_normalization_stats", {}).items()
    }
    checkpoint_normalizer = H5Normalizer(checkpoint_stats) if checkpoint_stats else None
    dataset = LocalModuleDataset(
        dataset_path,
        split=args.split,
        normalize_inputs=normalize_inputs,
        normalize_targets=normalize_targets,
        include_grid=True,
        normalizer=checkpoint_normalizer,
    )
    raw_dataset = LocalModuleDataset(dataset_path, split=args.split, normalize_inputs=False, normalize_targets=False, include_grid=True)
    if len(dataset) == 0:
        dataset = LocalModuleDataset(
            dataset_path,
            split="all",
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            include_grid=True,
            normalizer=checkpoint_normalizer,
        )
        raw_dataset = LocalModuleDataset(dataset_path, split="all", normalize_inputs=False, normalize_targets=False, include_grid=True)
    if len(dataset) == 0:
        raise RuntimeError("No local module cases are available for evaluation.")

    device = select_device(args.device)
    model_config = LocalModuleConfig.from_dict(checkpoint.get("model_config", {}))
    model = LocalModuleSurrogate(model_config).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()

    index = min(max(int(args.case_index), 0), len(dataset) - 1)
    sample = dataset[index]
    raw_sample = raw_dataset[index]
    batch = tensorize_sample(sample, device)
    with torch.no_grad():
        outputs = model(batch["module_params"], batch["port_tokens"], batch["internal_query_points"])

    pred_internal = outputs["internal_temperature"][0].detach().cpu().numpy()
    pred_interface = outputs["interface_pred"][0].detach().cpu().numpy()
    target_internal = sample["internal_temperature_targets"]
    target_interface = sample["interface_targets"]
    if normalize_targets:
        pred_internal = dataset.normalizer.denormalize_internal_temperature(pred_internal)
        pred_interface = dataset.normalizer.denormalize_interface_targets(pred_interface)
        target_internal = raw_sample["internal_temperature_targets"]
        target_interface = raw_sample["interface_targets"]
    internal_metrics = error_metrics(pred_internal.reshape(-1), target_internal.reshape(-1))
    interface_metrics = error_metrics(pred_interface, target_interface)
    t_surface_metrics = error_metrics(pred_interface[:, 0], target_interface[:, 0])
    q_normal_metrics = error_metrics(pred_interface[:, 1], target_interface[:, 1])
    roughness = raw_sample.get("local_target_roughness", np.zeros((4,), dtype=np.float32))
    n_active_modes = int(np.asarray(raw_sample.get("n_active_modes", [-1])).reshape(-1)[0])

    output_dir = evaluation_output_dir(args.output_dir, checkpoint_path, raw_sample["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_local_internal(
        output_dir / "internal_temperature_comparison.png",
        {**raw_sample, "internal_temperature_targets": target_internal},
        pred_internal,
        internal_metrics,
    )
    plot_local_interface(
        output_dir / "interface_curve_comparison.png",
        {**raw_sample, "interface_targets": target_interface},
        pred_interface,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "metric_note": "l2_error is the aggregate Euclidean norm over all values; rmse is usually better for visual comparison.",
        "internal_l2_error": internal_metrics["l2_norm"],
        "interface_l2_error": interface_metrics["l2_norm"],
        "T_surface_rmse": t_surface_metrics["rmse"],
        "T_surface_relative_l2": t_surface_metrics["relative_l2"],
        "q_normal_rmse": q_normal_metrics["rmse"],
        "q_normal_relative_l2": q_normal_metrics["relative_l2"],
        "internal_rmse": internal_metrics["rmse"],
        "interface_rmse": interface_metrics["rmse"],
        "internal_mae": internal_metrics["mae"],
        "interface_mae": interface_metrics["mae"],
        "internal_relative_l2": internal_metrics["relative_l2"],
        "interface_relative_l2": interface_metrics["relative_l2"],
        "internal_num_values": int(internal_metrics["num_values"]),
        "interface_num_values": int(interface_metrics["num_values"]),
        "internal_metrics": internal_metrics,
        "interface_metrics": interface_metrics,
        "T_surface_metrics": t_surface_metrics,
        "q_normal_metrics": q_normal_metrics,
        "solver_type": str(raw_sample.get("solver_type", "unknown")),
        "n_active_modes": n_active_modes,
        "roughness_metrics": {
            "roughness_T_surface": float(roughness[0]) if roughness.size > 0 else None,
            "roughness_q_normal": float(roughness[1]) if roughness.size > 1 else None,
            "highfreq_ratio_T_surface": float(roughness[2]) if roughness.size > 2 else None,
            "highfreq_ratio_q_normal": float(roughness[3]) if roughness.size > 3 else None,
        },
        "interface_targets_smoothed": bool(getattr(raw_dataset, "interface_targets_smoothed", False)),
        "outputs": {
            "internal_temperature_comparison": str(output_dir / "internal_temperature_comparison.png"),
            "interface_curve_comparison": str(output_dir / "interface_curve_comparison.png"),
        },
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
