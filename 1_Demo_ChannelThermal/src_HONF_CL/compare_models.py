"""Compare trained ChannelThermal HONF checkpoints on one aligned case set.

Inputs are Run_ID selectors or explicit HONF-CL checkpoint paths. Outputs are
case/model metric tables, summary tables, logs, and presentation-ready figures
under Saved_Model_HONF_CL/CompareModels/Run_<timestamp> by default.

Example usage:
python src_HONF_CL/compare_models.py \
  --Run_ID 0004 \
  --Run_ID 0005 \
  --Run_ID 0006 \
  --label NoPretrained \
  --label UniformH \
  --label ValidatedCore \
  --checkpoint-selector best \
  --case-ratio 1.0 \
  --device cuda:2

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-honf_cl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from data.datasets import CHANNEL_ORDER, GlobalChannelThermalDataset, H5Normalizer
from evaluation_tools.plots import module_and_fluid_masks
from common.runtime import current_timestamp, load_trusted_checkpoint, resolve_demo_path, select_device, write_json
from evaluate import (
    checkpoint_file_name,
    denormalize_predictions,
    extract_organization_arrays,
    hypergraph_diagnostics,
    latest_run_dir,
    load_model,
    predict_case,
)


EPS = 1.0e-12
PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line options for this workflow."""

    parser = argparse.ArgumentParser(description="Compare multiple ChannelThermal HONF-CL checkpoints.")
    parser.add_argument("--Run_ID", dest="run_ids", action="append", default=[], help="Run_ID to compare; repeat for multiple runs.")
    parser.add_argument("--checkpoint-path", action="append", default=[], help="Explicit .pt checkpoint path; repeat for multiple checkpoints.")
    parser.add_argument(
        "--checkpoint-selector",
        default="best",
        choices=["best", "latest", "best_predicted", "best_by_field_mse", "best_by_temperature_mse"],
        help="Checkpoint file to pick when resolving Run_IDs.",
    )
    parser.add_argument("--label", action="append", default=[], help="Optional model label; labels are assigned in model order.")
    parser.add_argument("--saved-root", default="./Saved_Model_HONF_CL/Global")
    parser.add_argument("--dataset", default=None, help="Optional packed HDF5 dataset path override.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-ratio", type=float, default=1.0, help="Fraction of selected split cases to evaluate, in (0, 1].")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--query-batch-size", type=int, default=32768)
    parser.add_argument("--local-port-condition-mode", choices=["teacher", "predicted", "mixed"], default="predicted")
    parser.add_argument("--mixed-teacher-ratio", type=float, default=0.5)
    parser.add_argument("--return-routing-maps", action="store_true")
    parser.add_argument("--save-debug-npz", action="store_true")
    parser.add_argument("--checkpoint-load-retries", type=int, default=5, help="Retries for reading checkpoints that may be actively written.")
    parser.add_argument("--checkpoint-load-retry-delay", type=float, default=2.0, help="Seconds to wait between checkpoint read retries.")
    parser.add_argument("--allow-checkpoint-fallback", action="store_true", help="Permit substitution of another best checkpoint if the requested file is unreadable.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def ensure_dirs(root: Path) -> Dict[str, Path]:
    """Ensure dirs."""

    paths = {
        "root": root,
        "logs": root / "logs",
        "tables": root / "tables",
        "fig_recon": root / "figures" / "reconstruction",
        "fig_hyper": root / "figures" / "hypergraph",
        "fig_summary": root / "figures" / "summary",
        "debug_npz": root / "debug_npz",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred_fields: Optional[Sequence[str]] = None) -> None:
    """Write csv."""

    fields: List[str] = []
    for field in preferred_fields or []:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_label(value: object) -> str:
    """Perform the safe label operation used by this module."""

    raw = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_") or "model"


def resolve_model_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Resolve model specs."""

    saved_root = resolve_demo_path(args.saved_root)
    specs: List[Dict[str, Any]] = []
    for run_id in args.run_ids:
        run_dir = latest_run_dir(saved_root, run_id)
        checkpoint_path = (run_dir / checkpoint_file_name(args.checkpoint_selector)).resolve()
        if not checkpoint_path.exists() and args.checkpoint_selector == "best_predicted" and args.allow_checkpoint_fallback:
            fallback = (run_dir / "best_model.pt").resolve()
            print(f"[warning] {checkpoint_path.name} not found; falling back to {fallback.name}.")
            checkpoint_path = fallback
        specs.append({"source": f"Run_ID:{run_id}", "run_id": str(run_id), "run_dir": run_dir, "checkpoint_path": checkpoint_path})
    for raw_path in args.checkpoint_path:
        checkpoint_path = resolve_demo_path(raw_path)
        specs.append({"source": f"path:{raw_path}", "run_id": "", "run_dir": checkpoint_path.parent, "checkpoint_path": checkpoint_path})
    if not specs:
        raise ValueError("Provide at least one --Run_ID or --checkpoint-path.")
    if len(args.label) > len(specs):
        raise ValueError(f"Received {len(args.label)} labels for {len(specs)} models.")
    for idx, spec in enumerate(specs):
        checkpoint_path = Path(spec["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        label = args.label[idx] if idx < len(args.label) else f"{Path(spec['run_dir']).name}:{checkpoint_path.stem}"
        spec["model_index"] = idx
        spec["label"] = label
    return specs


def load_checkpoint_with_retries(checkpoint_path: Path, *, retries: int, retry_delay: float) -> Dict[str, Any]:
    """Load checkpoint with retries."""

    attempts = max(int(retries), 1)
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - checkpoint read errors vary by PyTorch/storage backend.
            last_error = exc
            if attempt < attempts:
                tqdm.write(
                    f"[warning] failed to read {checkpoint_path.name} "
                    f"(attempt {attempt}/{attempts}): {type(exc).__name__}: {str(exc)[:160]}; retrying..."
                )
                time.sleep(max(float(retry_delay), 0.0))
    raise RuntimeError(f"Could not read checkpoint after {attempts} attempts: {checkpoint_path}") from last_error


def fallback_checkpoint_candidates(spec: Dict[str, Any]) -> List[Path]:
    """Perform the fallback checkpoint candidates operation used by this module."""

    run_dir = Path(spec["run_dir"])
    requested = Path(spec["checkpoint_path"]).resolve()
    names = [
        requested.name,
        "best_model.pt",
        "best_predicted_model.pt",
        "best_by_field_mse_model.pt",
        "best_by_temperature_mse_model.pt",
    ]
    candidates: List[Path] = []
    for name in names:
        path = (run_dir / name).resolve()
        if path.exists() and path not in candidates:
            candidates.append(path)
    return candidates


def load_checkpoint_for_spec(
    spec: Dict[str, Any],
    *,
    retries: int,
    retry_delay: float,
    allow_fallback: bool = False,
) -> tuple[Dict[str, Any], Path]:
    """Load the requested checkpoint and optionally permit declared fallbacks."""

    requested = Path(spec["checkpoint_path"]).resolve()
    try:
        checkpoint = load_checkpoint_with_retries(requested, retries=retries, retry_delay=retry_delay)
        spec["checkpoint_path"] = requested
        spec["checkpoint_fallback_from"] = ""
        return checkpoint, requested
    except Exception as requested_error:  # noqa: BLE001
        if not allow_fallback or not str(spec.get("source", "")).startswith("Run_ID:"):
            raise
        tqdm.write(
            f"[warning] requested checkpoint is unreadable for {spec.get('label', spec.get('source'))}: "
            f"{requested} ({type(requested_error).__name__}: {str(requested_error)[:180]}). Trying stable fallbacks."
        )
        for candidate in fallback_checkpoint_candidates(spec):
            if candidate == requested:
                continue
            try:
                checkpoint = load_checkpoint_with_retries(candidate, retries=retries, retry_delay=retry_delay)
            except Exception as fallback_error:  # noqa: BLE001
                tqdm.write(f"[warning] fallback checkpoint unreadable: {candidate.name} ({type(fallback_error).__name__}: {str(fallback_error)[:160]})")
                continue
            spec["checkpoint_path"] = candidate
            spec["checkpoint_fallback_from"] = str(requested)
            tqdm.write(f"[warning] using fallback checkpoint for {spec.get('label', spec.get('source'))}: {candidate}")
            return checkpoint, candidate
        raise RuntimeError(f"No readable checkpoint found for {spec.get('label', spec.get('source'))} in {spec.get('run_dir')}.") from requested_error


def load_model_with_retries(checkpoint_path: Path, *, device: torch.device, retries: int, retry_delay: float) -> tuple[Any, Dict[str, Any]]:
    """Load model with retries."""

    attempts = max(int(retries), 1)
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return load_model(checkpoint_path, device)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                tqdm.write(
                    f"[warning] failed to load model from {checkpoint_path.name} "
                    f"(attempt {attempt}/{attempts}): {type(exc).__name__}: {str(exc)[:160]}; retrying..."
                )
                time.sleep(max(float(retry_delay), 0.0))
    raise RuntimeError(f"Could not load model after {attempts} attempts: {checkpoint_path}") from last_error


def load_checkpoint_and_model_for_spec(
    spec: Dict[str, Any],
    *,
    device: torch.device,
    retries: int,
    retry_delay: float,
    allow_fallback: bool = False,
) -> tuple[Dict[str, Any], Path, Any]:
    """Load checkpoint and model for spec."""

    requested = Path(spec["checkpoint_path"]).resolve()
    candidates = [requested]
    if allow_fallback and str(spec.get("source", "")).startswith("Run_ID:"):
        candidates = fallback_checkpoint_candidates(spec)
    first_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            checkpoint = load_checkpoint_with_retries(candidate, retries=retries, retry_delay=retry_delay)
            model, _ = load_model_with_retries(candidate, device=device, retries=retries, retry_delay=retry_delay)
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc
            tqdm.write(f"[warning] checkpoint/model load failed for {candidate.name}: {type(exc).__name__}: {str(exc)[:180]}")
            continue
        spec["checkpoint_path"] = candidate
        spec["checkpoint_fallback_from"] = "" if candidate == requested else str(requested)
        if candidate != requested:
            tqdm.write(f"[warning] using fallback checkpoint for {spec.get('label', spec.get('source'))}: {candidate}")
        return checkpoint, candidate, model
    raise RuntimeError(f"No readable checkpoint/model pair found for {spec.get('label', spec.get('source'))} in {spec.get('run_dir')}.") from first_error


def selected_case_indices(dataset: GlobalChannelThermalDataset, ratio: float, seed: int) -> np.ndarray:
    """Perform the selected case indices operation used by this module."""

    total = len(dataset)
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    ratio = float(ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"--case-ratio must be in (0, 1], got {ratio}.")
    if ratio >= 1.0:
        return np.arange(total, dtype=np.int64)
    count = max(1, int(math.ceil(total * ratio)))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def finite_float(value: Any) -> float:
    """Perform the finite float operation used by this module."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def normalized_relative_l2(prediction: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Perform the normalized relative l2 operation used by this module."""

    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if pred.ndim == valid.ndim + 1:
            pred = pred[valid, :]
            gt = gt[valid, :]
        else:
            pred = pred[valid]
            gt = gt[valid]
    diff = pred.reshape(-1) - gt.reshape(-1)
    target_flat = gt.reshape(-1)
    if diff.size == 0:
        return float("nan")
    return float(np.linalg.norm(diff, ord=2) / max(float(np.linalg.norm(target_flat, ord=2)), EPS))


def row_normalized_entropy(values: np.ndarray, axis: int = -1) -> float:
    """Perform the row normalized entropy operation used by this module."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    arr = np.clip(arr, EPS, None)
    entropy = -np.sum(arr * np.log(arr), axis=axis)
    denom = math.log(max(arr.shape[axis], 2))
    return float(np.mean(entropy / denom))


def normalize_prediction_targets(
    predictions: Dict[str, Any],
    raw_sample: Dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    *,
    checkpoint_targets_normalized: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
    """Normalize prediction targets."""

    target_field = raw_sample["steady_field"][..., : predictions["pred_field_grid"].shape[-1]]
    target_internal = raw_sample["module_internal_temperature_points"][..., None]
    target_interface = raw_sample["interface_target"]
    targets = {
        "field": dataset.normalizer.normalize_fields(target_field),
        "internal": dataset.normalizer.normalize_internal_temperature(target_internal),
        "interface": dataset.normalizer.normalize_interface_targets(target_interface),
    }
    if checkpoint_targets_normalized:
        pred = {
            "field": np.asarray(predictions["pred_field_grid"], dtype=np.float32),
            "internal": np.asarray(predictions["pred_internal_temperature"], dtype=np.float32),
            "interface": np.asarray(predictions["pred_interface"], dtype=np.float32),
        }
        return pred, targets, "dataset_normalized"
    pred = {
        "field": dataset.normalizer.normalize_fields(np.asarray(predictions["pred_field_grid"], dtype=np.float32)),
        "internal": dataset.normalizer.normalize_internal_temperature(np.asarray(predictions["pred_internal_temperature"], dtype=np.float32)),
        "interface": dataset.normalizer.normalize_interface_targets(np.asarray(predictions["pred_interface"], dtype=np.float32)),
    }
    return pred, targets, "dataset_normalized_from_physical"


def per_module_metric_rows(
    *,
    base_row: Dict[str, Any],
    pred: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    module_present: np.ndarray,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Perform the per module metric rows operation used by this module."""

    rows: List[Dict[str, Any]] = []
    internal_values: List[float] = []
    t_values: List[float] = []
    q_values: List[float] = []
    for module_idx in np.flatnonzero(np.asarray(module_present, dtype=np.float32) > 0.5):
        row = dict(base_row)
        row["module_idx"] = int(module_idx)
        internal = normalized_relative_l2(pred["internal"][module_idx, :, 0], target["internal"][module_idx, :, 0])
        t_surface = normalized_relative_l2(pred["interface"][module_idx, :, 0], target["interface"][module_idx, :, 0])
        q_normal = normalized_relative_l2(pred["interface"][module_idx, :, 1], target["interface"][module_idx, :, 1])
        row.update(
            {
                "internal_temperature_norm_l2": internal,
                "t_surface_norm_l2": t_surface,
                "q_normal_norm_l2": q_normal,
            }
        )
        rows.append(row)
        internal_values.append(internal)
        t_values.append(t_surface)
        q_values.append(q_normal)
    return rows, {
        "internal_temperature_mean_norm_l2": float(np.nanmean(internal_values)) if internal_values else float("nan"),
        "t_surface_mean_norm_l2": float(np.nanmean(t_values)) if t_values else float("nan"),
        "q_normal_mean_norm_l2": float(np.nanmean(q_values)) if q_values else float("nan"),
        "active_module_count": float(len(internal_values)),
    }


def reconstruction_metrics(
    *,
    base_row: Dict[str, Any],
    predictions: Dict[str, Any],
    raw_sample: Dict[str, Any],
    dataset: GlobalChannelThermalDataset,
    checkpoint_targets_normalized: bool,
    channel_order: Sequence[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Perform the reconstruction metrics operation used by this module."""

    pred_norm, target_norm, target_space = normalize_prediction_targets(
        predictions,
        raw_sample,
        dataset,
        checkpoint_targets_normalized=checkpoint_targets_normalized,
    )
    _, fluid_mask = module_and_fluid_masks(raw_sample, pred_norm["field"])
    row = dict(base_row)
    row["target_space"] = target_space
    row["global_field_fluid_norm_l2"] = normalized_relative_l2(pred_norm["field"], target_norm["field"], fluid_mask)
    row["global_field_all_norm_l2"] = normalized_relative_l2(pred_norm["field"], target_norm["field"])
    module_present = np.asarray(raw_sample["structure"]["module_present"], dtype=np.float32) > 0.5
    if np.any(module_present):
        row["internal_module_cell_norm_l2"] = normalized_relative_l2(
            pred_norm["internal"][module_present, :, 0],
            target_norm["internal"][module_present, :, 0],
        )
    else:
        row["internal_module_cell_norm_l2"] = float("nan")
    for idx, name in enumerate(channel_order[: pred_norm["field"].shape[-1]]):
        row[f"field_{name}_fluid_norm_l2"] = normalized_relative_l2(pred_norm["field"][..., idx], target_norm["field"][..., idx], fluid_mask)
        row[f"field_{name}_all_norm_l2"] = normalized_relative_l2(pred_norm["field"][..., idx], target_norm["field"][..., idx])
    module_rows, local_summary = per_module_metric_rows(
        base_row=base_row,
        pred=pred_norm,
        target=target_norm,
        module_present=raw_sample["structure"]["module_present"],
    )
    row.update(local_summary)
    return row, module_rows


def matrix_row_normalize(values: np.ndarray) -> np.ndarray:
    """Perform the matrix row normalize operation used by this module."""

    arr = np.asarray(values, dtype=np.float64)
    denom = arr.sum(axis=-1, keepdims=True)
    return np.divide(arr, np.maximum(denom, EPS))


def hypergraph_metrics(base_row: Dict[str, Any], raw_sample: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Perform the hypergraph metrics operation used by this module."""

    row = dict(base_row)
    aux = predictions.get("organizer_aux", {})
    structure_targets = raw_sample.get("structure_targets", {})
    arrays = extract_organization_arrays(raw_sample, aux)
    A_mh = matrix_row_normalize(np.asarray(arrays.get("A_mh", np.zeros((0, 0))), dtype=np.float64))
    A_eh = matrix_row_normalize(np.asarray(arrays.get("A_eh", np.zeros((0, 0))), dtype=np.float64))
    present = np.asarray(raw_sample["structure"]["module_present"], dtype=np.float32) > 0.5
    has_solved_targets = float(np.asarray(structure_targets.get("has_solved_structure_targets", [0.0])).reshape(-1)[0])
    row["has_solved_structure_targets"] = has_solved_targets
    if has_solved_targets > 0.5 and A_mh.size and A_eh.size:
        env_module_pred = A_eh @ A_mh.T
        if "env_module_influence_target" in structure_targets:
            target = np.asarray(structure_targets["env_module_influence_target"], dtype=np.float64)
            comparable = env_module_pred.shape[0] == target.shape[0] and env_module_pred.shape[1] >= target.shape[1]
            row["env_module_influence_shape_mismatch"] = float(not comparable)
            row["env_module_influence_pred_rows"] = float(env_module_pred.shape[0])
            row["env_module_influence_target_rows"] = float(target.shape[0])
            if comparable:
                row["env_module_influence_norm_l2"] = normalized_relative_l2(env_module_pred[:, : target.shape[1]], target)
            else:
                row["env_module_influence_norm_l2"] = float("nan")
        module_affinity_pred = matrix_row_normalize(A_mh @ A_mh.T)
        if "module_affinity_target" in structure_targets:
            target = np.asarray(structure_targets["module_affinity_target"], dtype=np.float64)
            mask = present[:, None] & present[None, :]
            row["module_affinity_norm_l2"] = normalized_relative_l2(module_affinity_pred[: target.shape[0], : target.shape[1]], target, mask)
    if has_solved_targets > 0.5 and "active_edge_count_target" in structure_targets:
        target_active = np.asarray(structure_targets["active_edge_count_target"], dtype=np.float64).reshape(-1)
        strength = np.asarray(aux.get("hyper_strength", np.zeros((0,))), dtype=np.float64).reshape(-1)
        pred_active = np.asarray([float(np.sum(strength > 0.05))], dtype=np.float64)
        row["active_edge_count_target_norm_l2"] = normalized_relative_l2(pred_active, target_active)
    diag = hypergraph_diagnostics(predictions)
    for section_name, section in diag.items():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            row[f"{section_name}_{key}"] = finite_float(value)
    if A_mh.size:
        row["intrinsic_A_mh_entropy_norm"] = row_normalized_entropy(A_mh, axis=-1)
    if A_eh.size:
        row["intrinsic_A_eh_entropy_norm"] = row_normalized_entropy(A_eh, axis=-1)
    strength = np.asarray(aux.get("hyper_strength", np.zeros((0,))), dtype=np.float64).reshape(-1)
    if strength.size:
        row["intrinsic_soft_active_edge_count"] = float(np.sum(np.clip(strength, 0.0, 1.0)))
    return row


def summarize_rows(rows: Sequence[Dict[str, Any]], group_key: str, metric_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Aggregate metrics per group, preserving declared model order when available."""

    if group_key == "model_label":
        labels = model_labels_in_order(rows)
    else:
        labels = sorted({str(row[group_key]) for row in rows})
    out: List[Dict[str, Any]] = []
    for label in labels:
        subset = [row for row in rows if str(row[group_key]) == label]
        summary: Dict[str, Any] = {group_key: label, "num_cases": len(subset)}
        first = subset[0] if subset else {}
        for key in ("model_index", "checkpoint", "run_dir"):
            if key in first:
                summary[key] = first[key]
        for metric in metric_names:
            values = np.asarray([finite_float(row.get(metric)) for row in subset], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_std"] = float(np.std(values))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        out.append(summary)
    return out


def model_labels_in_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Return unique model labels ordered by their resolved ``model_index``."""

    first_index: Dict[str, int] = {}
    for position, row in enumerate(rows):
        label = str(row.get("model_label", ""))
        first_index.setdefault(label, int(row.get("model_index", position)))
    return [label for label, _ in sorted(first_index.items(), key=lambda item: item[1])]


def numeric_metric_names(rows: Sequence[Dict[str, Any]], exclude: Iterable[str]) -> List[str]:
    """Perform the numeric metric names operation used by this module."""

    excluded = set(exclude)
    names: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key in excluded or key in names:
                continue
            if math.isfinite(finite_float(value)):
                names.append(key)
    return names


def setup_plot_style() -> None:
    """Perform the setup plot style operation used by this module."""

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def collect_metric_data(rows: Sequence[Dict[str, Any]], metric: str, labels: Sequence[str]) -> List[np.ndarray]:
    """Perform the collect metric data operation used by this module."""

    data = []
    for label in labels:
        values = np.asarray([finite_float(row.get(metric)) for row in rows if str(row.get("model_label")) == label], dtype=np.float64)
        data.append(values[np.isfinite(values)])
    return data


def plot_violin(path: Path, rows: Sequence[Dict[str, Any]], metrics: Sequence[str], title: str, ylabel: str = "Normalized L2") -> None:
    """Plot violin."""

    labels = model_labels_in_order(rows)
    if not labels or not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(6.4, 4.8 * len(metrics)), 5.0), constrained_layout=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        data = collect_metric_data(rows, metric, labels)
        valid_positions = [idx + 1 for idx, values in enumerate(data) if values.size]
        valid_data = [values for values in data if values.size]
        if valid_data:
            parts = ax.violinplot(valid_data, positions=valid_positions, showmeans=True, showmedians=True)
            for idx, body in enumerate(parts["bodies"]):
                body.set_facecolor(PALETTE[idx % len(PALETTE)])
                body.set_edgecolor("#2f2f2f")
                body.set_alpha(0.72)
            for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
                if key in parts:
                    parts[key].set_color("#2f2f2f")
                    parts[key].set_linewidth(1.0)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path)
    plt.close(fig)


def plot_grouped_bar(path: Path, summary_rows: Sequence[Dict[str, Any]], metrics: Sequence[str], title: str, *, suffix: str = "_mean") -> None:
    """Plot grouped bar."""

    labels = [str(row["model_label"]) if "model_label" in row else str(row.get("label", "")) for row in summary_rows]
    if not labels or not metrics:
        return
    x = np.arange(len(labels), dtype=np.float64)
    width = min(0.82 / max(len(metrics), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8.2, 1.5 * len(labels) + 1.65 * len(metrics)), 5.2), constrained_layout=False)
    for idx, metric in enumerate(metrics):
        values = [finite_float(row.get(f"{metric}{suffix}")) for row in summary_rows]
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=metric.replace("_norm_l2", "").replace("_", " "), color=PALETTE[idx % len(PALETTE)], edgecolor="#303030", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean normalized L2" if suffix == "_mean" else suffix.strip("_"))
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(3, len(metrics)))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_bar_panels(path: Path, summary_rows: Sequence[Dict[str, Any]], metrics: Sequence[str], title: str, *, suffix: str = "_mean", ylabel: str = "Mean normalized L2") -> None:
    """Plot metric bar panels."""

    labels = [str(row["model_label"]) if "model_label" in row else str(row.get("label", "")) for row in summary_rows]
    metrics = [metric for metric in metrics if any(math.isfinite(finite_float(row.get(f"{metric}{suffix}"))) for row in summary_rows)]
    if not labels or not metrics:
        return
    ncols = min(3, len(metrics))
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(6.0, 4.6 * ncols), max(4.0, 3.8 * nrows)), constrained_layout=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    x = np.arange(len(labels), dtype=np.float64)
    for idx, metric in enumerate(metrics):
        ax = axes_arr[idx // ncols, idx % ncols]
        values = np.asarray([finite_float(row.get(f"{metric}{suffix}")) for row in summary_rows], dtype=np.float64)
        colors = [PALETTE[item % len(PALETTE)] for item in range(len(labels))]
        ax.bar(x, values, width=0.62, color=colors, edgecolor="#303030", linewidth=0.55)
        finite = values[np.isfinite(values)]
        if finite.size:
            ymin = min(0.0, float(np.min(finite)) * 0.95)
            ymax = float(np.max(finite))
            pad = max((ymax - ymin) * 0.12, abs(ymax) * 0.08, 1.0e-6)
            ax.set_ylim(ymin, ymax + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(metric.replace("_norm_l2", "").replace("_", " "))
        ax.set_ylabel(ylabel)
    for idx in range(len(metrics), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path)
    plt.close(fig)


def save_figures(paths: Dict[str, Path], per_case_rows: Sequence[Dict[str, Any]], summary_rows: Sequence[Dict[str, Any]], hyper_rows: Sequence[Dict[str, Any]], hyper_summary_rows: Sequence[Dict[str, Any]], channel_order: Sequence[str], return_routing_maps: bool) -> None:
    """Save figures."""

    setup_plot_style()
    plot_violin(
        paths["fig_recon"] / "global_field_errors_violin.png",
        per_case_rows,
        ["global_field_fluid_norm_l2"],
        "Global Field Reconstruction Error (Fluid Domain)",
    )
    channel_metrics = [f"field_{name}_fluid_norm_l2" for name in channel_order if any(f"field_{name}_fluid_norm_l2" in row for row in per_case_rows)]
    plot_violin(paths["fig_recon"] / "field_channel_errors_violin.png", per_case_rows, channel_metrics[:5], "Per-Field Fluid Errors")
    plot_metric_bar_panels(paths["fig_recon"] / "field_channel_errors_bar.png", summary_rows, channel_metrics[:5], "Mean Per-Field Fluid Errors")
    plot_violin(
        paths["fig_recon"] / "internal_module_cell_average_error_violin.png",
        per_case_rows,
        ["internal_module_cell_norm_l2"],
        "Internal-Module Cell Average Error",
    )
    local_metrics = ["internal_temperature_mean_norm_l2", "t_surface_mean_norm_l2", "q_normal_mean_norm_l2"]
    plot_violin(paths["fig_recon"] / "local_interface_errors_violin.png", per_case_rows, local_metrics, "Local Internal And Interface Errors")
    plot_grouped_bar(paths["fig_recon"] / "local_interface_errors_bar.png", summary_rows, local_metrics, "Mean Local Internal And Interface Errors")
    plot_grouped_bar(
        paths["fig_summary"] / "primary_reconstruction_summary.png",
        summary_rows,
        ["global_field_fluid_norm_l2", "internal_temperature_mean_norm_l2", "t_surface_mean_norm_l2", "q_normal_mean_norm_l2"],
        "Primary Reconstruction Summary",
    )
    target_metrics = ["env_module_influence_norm_l2", "module_affinity_norm_l2", "active_edge_count_target_norm_l2"]
    plot_grouped_bar(paths["fig_hyper"] / "hypergraph_target_agreement_bar.png", hyper_summary_rows, target_metrics, "Hypergraph Target Agreement")
    intrinsic_metrics = [
        "static_organization_active_edge_count",
        "intrinsic_A_mh_entropy_norm",
        "intrinsic_A_eh_entropy_norm",
        "static_organization_module_mass_max",
        "static_organization_env_mass_max",
    ]
    plot_violin(paths["fig_hyper"] / "hypergraph_intrinsic_violin.png", hyper_rows, intrinsic_metrics, "Intrinsic Hypergraph Metrics", ylabel="Metric value")
    plot_metric_bar_panels(paths["fig_hyper"] / "hypergraph_intrinsic_bar.png", hyper_summary_rows, intrinsic_metrics, "Mean Intrinsic Hypergraph Metrics", ylabel="Mean metric value")
    if return_routing_maps:
        routing_metrics = ["routing_query_attention_entropy", "routing_query_attention_effective_edges", "routing_query_attention_max"]
        plot_violin(paths["fig_hyper"] / "routing_metrics_violin.png", hyper_rows, routing_metrics, "Routing Metrics", ylabel="Metric value")
        plot_grouped_bar(paths["fig_hyper"] / "routing_metrics_bar.png", hyper_summary_rows, routing_metrics, "Mean Routing Metrics")


def save_debug_npz(path: Path, predictions: Dict[str, Any], raw_sample: Dict[str, Any]) -> None:
    """Save debug npz."""

    payload = {
        "pred_field_grid": np.asarray(predictions["pred_field_grid"], dtype=np.float32),
        "gt_field_grid": np.asarray(raw_sample["steady_field"], dtype=np.float32),
        "pred_internal_temperature": np.asarray(predictions["pred_internal_temperature"], dtype=np.float32),
        "gt_internal_temperature": np.asarray(raw_sample["module_internal_temperature_points"], dtype=np.float32),
        "pred_interface": np.asarray(predictions["pred_interface"], dtype=np.float32),
        "gt_interface": np.asarray(raw_sample["interface_target"], dtype=np.float32),
    }
    np.savez_compressed(path, **payload)


def main() -> int:
    """Run this command-line workflow and return its process status."""

    args = parse_args()
    model_specs = resolve_model_specs(args)
    output_root = resolve_demo_path(args.output_dir) if args.output_dir else resolve_demo_path(args.saved_root) / "CompareModels" / f"Run_{current_timestamp()}"
    paths = ensure_dirs(output_root)
    device = select_device(args.device)
    first_checkpoint, first_checkpoint_path = load_checkpoint_for_spec(
        model_specs[0],
        retries=int(args.checkpoint_load_retries),
        retry_delay=float(args.checkpoint_load_retry_delay),
        allow_fallback=bool(args.allow_checkpoint_fallback),
    )
    first_dataset_cfg = dict(first_checkpoint.get("train_config", {}).get("dataset", {}))
    model_specs[0]["checkpoint_path"] = first_checkpoint_path
    dataset_path = args.dataset or first_dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")
    raw_dataset = GlobalChannelThermalDataset(dataset_path, split=args.split, points_per_case=1, normalize_inputs=False, normalize_targets=False, random_point_sampling=False, include_grid=True, include_structure_targets=True)
    actual_split = args.split
    if len(raw_dataset) == 0:
        print(f"[warning] split={args.split!r} has no cases; falling back to split='all'.")
        actual_split = "all"
        raw_dataset = GlobalChannelThermalDataset(dataset_path, split="all", points_per_case=1, normalize_inputs=False, normalize_targets=False, random_point_sampling=False, include_grid=True, include_structure_targets=True)
    case_indices = selected_case_indices(raw_dataset, args.case_ratio, args.seed)
    selected_case_rows = [
        {"case_order": int(order), "dataset_index": int(idx), "case_id": str(raw_dataset.selected_case_ids[int(idx)]), "split": actual_split}
        for order, idx in enumerate(case_indices)
    ]
    write_csv(paths["logs"] / "selected_cases.csv", selected_case_rows, ["case_order", "dataset_index", "case_id", "split"])

    per_case_rows: List[Dict[str, Any]] = []
    per_module_rows: List[Dict[str, Any]] = []
    hyper_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    channel_order = list(CHANNEL_ORDER)

    model_bar = tqdm(model_specs, desc="models", unit="model")
    for spec in model_bar:
        checkpoint_path = Path(spec["checkpoint_path"])
        model_bar.set_postfix_str(str(spec["label"]))
        checkpoint, checkpoint_path, model = load_checkpoint_and_model_for_spec(
            spec,
            device=device,
            retries=int(args.checkpoint_load_retries),
            retry_delay=float(args.checkpoint_load_retry_delay),
            allow_fallback=bool(args.allow_checkpoint_fallback),
        )
        dataset_cfg = dict(checkpoint.get("train_config", {}).get("dataset", {}))
        model_dataset_path = args.dataset or dataset_cfg.get("packed_h5_path", dataset_path)
        if str(resolve_demo_path(model_dataset_path)) != str(resolve_demo_path(dataset_path)):
            print(f"[warning] model {spec['label']} dataset path differs; using comparison dataset {dataset_path}.")
        checkpoint_targets_normalized = bool(dataset_cfg.get("normalize_targets", False))
        checkpoint_stats = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in checkpoint.get("global_normalization_stats", {}).items()
        }
        checkpoint_normalizer = H5Normalizer(checkpoint_stats) if checkpoint_stats else None
        input_dataset = GlobalChannelThermalDataset(
            dataset_path,
            split=actual_split,
            points_per_case=1,
            normalize_inputs=bool(dataset_cfg.get("normalize_inputs", False)),
            normalize_targets=checkpoint_targets_normalized,
            random_point_sampling=False,
            include_grid=True,
            normalizer=checkpoint_normalizer,
        )
        if list(input_dataset.selected_case_ids) != list(raw_dataset.selected_case_ids):
            raise ValueError(
                f"Model {spec['label']} does not expose the same ordered case IDs as the raw comparison view."
            )
        configured_fields = list(model.config.channelthermal.field_names)
        dataset_fields = list(input_dataset.channel_order or CHANNEL_ORDER)
        if configured_fields != dataset_fields:
            raise ValueError(
                f"Model {spec['label']} field order {configured_fields} does not match dataset order {dataset_fields}."
            )
        channel_order = input_dataset.channel_order or channel_order
        manifest_rows.append(
            {
                "model_index": int(spec["model_index"]),
                "model_label": str(spec["label"]),
                "source": str(spec["source"]),
                "run_id": str(spec.get("run_id", "")),
                "run_dir": str(spec["run_dir"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_fallback_from": str(spec.get("checkpoint_fallback_from", "")),
                "checkpoint_selector": str(args.checkpoint_selector),
                "normalize_inputs": bool(dataset_cfg.get("normalize_inputs", False)),
                "normalize_targets": checkpoint_targets_normalized,
                "dataset_path": str(resolve_demo_path(dataset_path)),
            }
        )
        tqdm.write(f"[model] {spec['label']} -> {checkpoint_path}")
        case_iter = tqdm(
            enumerate(case_indices),
            total=len(case_indices),
            desc=f"cases:{spec['label']}",
            unit="case",
            leave=False,
        )
        for case_order, dataset_idx in case_iter:
            raw_sample = raw_dataset[int(dataset_idx)]
            sample = input_dataset[int(dataset_idx)]
            case_id = str(raw_sample["case_id"])
            case_iter.set_postfix_str(case_id)
            with torch.no_grad():
                predictions = predict_case(
                    model,
                    sample,
                    device,
                    query_batch_size=int(args.query_batch_size),
                    local_port_condition_mode=str(args.local_port_condition_mode),
                    mixed_teacher_ratio=float(args.mixed_teacher_ratio),
                    return_routing_maps=bool(args.return_routing_maps),
                )
            base_row = {
                "model_index": int(spec["model_index"]),
                "model_label": str(spec["label"]),
                "checkpoint": str(checkpoint_path),
                "run_dir": str(spec["run_dir"]),
                "case_order": int(case_order),
                "dataset_index": int(dataset_idx),
                "case_id": case_id,
                "split": actual_split,
            }
            metric_row, module_rows = reconstruction_metrics(
                base_row=base_row,
                predictions=predictions,
                raw_sample=raw_sample,
                dataset=input_dataset,
                checkpoint_targets_normalized=checkpoint_targets_normalized,
                channel_order=channel_order,
            )
            per_case_rows.append(metric_row)
            per_module_rows.extend(module_rows)
            hyper_rows.append(hypergraph_metrics(base_row, raw_sample, predictions))
            if args.save_debug_npz:
                npz_name = f"{safe_label(spec['label'])}__{safe_label(case_id)}.npz"
                physical_predictions = denormalize_predictions(dict(predictions), input_dataset, checkpoint_targets_normalized)
                save_debug_npz(paths["debug_npz"] / npz_name, physical_predictions, raw_sample)

    write_csv(paths["logs"] / "model_manifest.csv", manifest_rows)
    write_json(
        paths["logs"] / "run_config.json",
        {
            "output_root": str(output_root),
            "dataset_path": str(resolve_demo_path(dataset_path)),
            "split": actual_split,
            "case_ratio": float(args.case_ratio),
            "seed": int(args.seed),
            "checkpoint_selector": str(args.checkpoint_selector),
            "local_port_condition_mode": str(args.local_port_condition_mode),
            "mixed_teacher_ratio": float(args.mixed_teacher_ratio),
            "return_routing_maps": bool(args.return_routing_maps),
            "save_debug_npz": bool(args.save_debug_npz),
            "num_models": len(model_specs),
            "num_cases": len(case_indices),
        },
    )

    common_summary_exclude = {
        "model_index",
        "model_label",
        "checkpoint",
        "run_dir",
        "case_order",
        "dataset_index",
        "case_id",
        "split",
        "target_space",
    }
    reconstruction_metrics_to_summarize = numeric_metric_names(
        per_case_rows,
        exclude={*common_summary_exclude, "active_module_count"},
    )
    hyper_metrics_to_summarize = numeric_metric_names(
        hyper_rows,
        exclude={
            *common_summary_exclude,
            "has_solved_structure_targets",
            "env_module_influence_shape_mismatch",
            "env_module_influence_pred_rows",
            "env_module_influence_target_rows",
        },
    )
    model_summary_rows = summarize_rows(per_case_rows, "model_label", reconstruction_metrics_to_summarize)
    hyper_summary_rows = summarize_rows(hyper_rows, "model_label", hyper_metrics_to_summarize)

    write_csv(paths["tables"] / "per_case_metrics.csv", per_case_rows)
    write_csv(paths["tables"] / "per_module_metrics.csv", per_module_rows)
    write_csv(paths["tables"] / "model_summary_metrics.csv", model_summary_rows)
    write_csv(paths["tables"] / "hypergraph_case_metrics.csv", hyper_rows)
    write_csv(paths["tables"] / "hypergraph_summary_metrics.csv", hyper_summary_rows)
    save_figures(paths, per_case_rows, model_summary_rows, hyper_rows, hyper_summary_rows, channel_order, bool(args.return_routing_maps))
    print(f"[done] wrote comparison outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
